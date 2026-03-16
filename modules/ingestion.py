import os
import pandas as pd
from azure.storage.blob import BlobServiceClient, ContainerClient
import config


# connect to azure blob storage using the connection string from .env
def connect_blob_storage() -> BlobServiceClient:
    if not config.AZURE_STORAGE_CONNECTION_STRING:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set in environment.")
    return BlobServiceClient.from_connection_string(config.AZURE_STORAGE_CONNECTION_STRING)


# create a blob container if it doesnt exist yet
def ensure_container(blob_service: BlobServiceClient, container_name: str) -> ContainerClient:
    container_client = blob_service.get_container_client(container_name)
    try:
        container_client.get_container_properties()
        print(f"  Container '{container_name}' already exists.")
    except Exception:
        container_client.create_container()
        print(f"  Container '{container_name}' created.")
    return container_client


# upload the raw csv file to azure ngsim-raw container
def upload_raw_data(blob_service: BlobServiceClient, local_file_path: str) -> str:
    container_client = ensure_container(blob_service, config.AZURE_CONTAINER_RAW)
    blob_name = os.path.basename(local_file_path)
    blob_client = container_client.get_blob_client(blob_name)

    file_size = os.path.getsize(local_file_path)
    print(f"  Uploading '{blob_name}' ({file_size / 1024:.1f} KB) to container '{config.AZURE_CONTAINER_RAW}'...")

    with open(local_file_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)

    print(f"  Upload complete: {blob_name}")
    return blob_name


# download the csv back from azure and load it into a dataframe
def download_raw_data(blob_service: BlobServiceClient, blob_name: str) -> pd.DataFrame:
    container_client = blob_service.get_container_client(config.AZURE_CONTAINER_RAW)
    blob_client = container_client.get_blob_client(blob_name)

    print(f"  Downloading '{blob_name}' from Azure Blob Storage...")
    stream = blob_client.download_blob()
    data = stream.readall().decode("utf-8")

    from io import StringIO
    df = load_ngsim_data_from_string(data)
    print(f"  Downloaded {len(df)} records.")
    return df


# load ngsim data from a local csv file
# reads the csv, fixes column names, filters to us-101, removes duplicates
def load_ngsim_local(file_path: str) -> pd.DataFrame:
    print(f"  Loading local file: {file_path}")

    if file_path.endswith(".csv"):
        try:
            df = pd.read_csv(file_path)
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, sep=r"\s+", header=None, names=config.NGSIM_COLUMNS)
        except Exception:
            df = pd.read_csv(file_path, sep=r"\s+", header=None, names=config.NGSIM_COLUMNS)
    else:
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=config.NGSIM_COLUMNS)

    # fix column names from the data.gov format to our standard
    df = _normalize_columns(df)

    # filter to only us-101 data if the dataset has multiple freeways
    if "Location" in df.columns:
        before = len(df)
        df = df[df["Location"].str.contains("us-101", case=False, na=False)].copy()
        print(f"  Filtered to US-101 records: {len(df)} (from {before})")
        df = df.drop(columns=["Location"], errors="ignore")

    # remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        print(f"  Removed {before - len(df)} duplicate rows.")

    if list(df.columns) != config.NGSIM_COLUMNS and len(df.columns) == len(config.NGSIM_COLUMNS):
        df.columns = config.NGSIM_COLUMNS

    print(f"  Loaded {len(df)} records, {df['Vehicle_ID'].nunique()} unique vehicles.")
    return df


# fix column names - data.gov uses different names than standard ngsim
# like Space_Headway instead of Space_Hdwy, v_vel instead of v_Vel
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "v_length": "v_Length",
        "v_width": "v_Width",
        "v_class": "v_Class",
        "v_vel": "v_Vel",
        "v_acc": "v_Acc",
        "Space_Headway": "Space_Hdwy",
        "Time_Headway": "Time_Hdwy",
    }
    df = df.rename(columns=column_map)

    # drop extra columns from the data.gov format that we dont need
    extra_cols = ["O_Zone", "D_Zone", "Int_ID", "Section_ID", "Direction", "Movement"]
    df = df.drop(columns=[c for c in extra_cols if c in df.columns], errors="ignore")

    return df


# same as load_ngsim_local but reads from a string instead of a file
# used when downloading from azure
def load_ngsim_data_from_string(data_str: str) -> pd.DataFrame:
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(data_str))
        if len(df.columns) == 1:
            df = pd.read_csv(StringIO(data_str), sep=r"\s+", header=None, names=config.NGSIM_COLUMNS)
    except Exception:
        df = pd.read_csv(StringIO(data_str), sep=r"\s+", header=None, names=config.NGSIM_COLUMNS)

    df = _normalize_columns(df)

    if "Location" in df.columns:
        df = df[df["Location"].str.contains("us-101", case=False, na=False)].copy()
        df = df.drop(columns=["Location"], errors="ignore")

    df = df.drop_duplicates()
    return df
