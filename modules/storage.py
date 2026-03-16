import json
import math
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
import config
from modules.ingestion import ensure_container


# clean up numpy/pandas types so json.dumps doesnt break
# converts numpy ints to python ints, numpy floats to python floats
# replaces NaN and Inf with None so the json is valid
def _clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


# upload processed output to azure ngsim-output container
def upload_output(blob_service: BlobServiceClient, output_data: str, blob_name: str):
    container_client = ensure_container(blob_service, config.AZURE_CONTAINER_OUTPUT)
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(output_data, overwrite=True)
    print(f"  Output uploaded: {blob_name}")


# upload the scenarios json file to azure
def upload_scenarios_json(blob_service: BlobServiceClient, scenarios: list, blob_name: str):
    container_client = ensure_container(blob_service, config.AZURE_CONTAINER_OUTPUT)
    blob_client = container_client.get_blob_client(blob_name)
    # clean numpy/pandas types before converting to json
    clean_scenarios = _clean_for_json(scenarios)
    json_data = json.dumps(clean_scenarios, indent=2)
    blob_client.upload_blob(json_data.encode("utf-8"), overwrite=True)
    print(f"  Scenarios JSON uploaded: {blob_name} ({len(scenarios)} scenarios)")


# list all files in a container
def list_blobs(blob_service: BlobServiceClient, container_name: str) -> list:
    container_client = blob_service.get_container_client(container_name)
    return [blob.name for blob in container_client.list_blobs()]


# save scenarios to a local json file
def save_output_local(scenarios: list, file_path: str):
    clean_scenarios = _clean_for_json(scenarios)
    with open(file_path, "w") as f:
        json.dump(clean_scenarios, f, indent=2)
    print(f"  Output saved locally: {file_path} ({len(scenarios)} scenarios)")


# save a summary csv that can be opened in excel
def save_output_csv(scenarios: list, file_path: str):
    rows = []
    for s in scenarios:
        rows.append({
            "scenario_id": s["scenario_id"],
            "scenario_type": s["scenario_type"],
            "ego_vehicle_id": s["ego_vehicle_id"],
            "start_frame": s["start_frame"],
            "end_frame": s["end_frame"],
            "start_time_ms": s["start_time_ms"],
            "end_time_ms": s["end_time_ms"],
            "ego_lane": s.get("ego_lane", ""),
            "num_surrounding_vehicles": len(s.get("surrounding_vehicles", [])),
            "surrounding_vehicle_ids": ",".join(str(v) for v in s.get("surrounding_vehicles", []))
        })
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)
    print(f"  Summary CSV saved: {file_path}")
