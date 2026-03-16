import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

from modules.ingestion import (
    load_ngsim_local,
    connect_blob_storage,
    upload_raw_data,
    download_raw_data,
)
from modules.preprocessing import preprocess
from modules.scenarios.car_following import detect_car_following
from modules.scenarios.stop_and_go import detect_stop_and_go
from modules.scenarios.lane_change import detect_lane_change
from modules.output import format_scenarios, print_summary, print_example_outputs
from modules.storage import save_output_local, save_output_csv, upload_scenarios_json
from modules.visualization import visualize_all


# handle command line arguments like --data, --azure, --generate-sample
def parse_args():
    parser = argparse.ArgumentParser(
        description="NGSIM Scenario Extraction - Phase 1 Monolithic System"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=config.NGSIM_DATA_PATH,
        help="Path to NGSIM data file (CSV or TXT)",
    )
    parser.add_argument(
        "--azure", action="store_true", help="Enable Azure Blob Storage integration"
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate sample NGSIM data for testing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Directory for output files",
    )
    return parser.parse_args()


def generate_sample_data(output_path: str):
    from generate_sample import generate_sample_ngsim

    generate_sample_ngsim(output_path)


# main pipeline - runs all 5 steps
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("NGSIM SCENARIO EXTRACTION SYSTEM")
    print("Phase 1 - Modular Monolithic Application")
    print("Cloud Provider: Microsoft Azure")
    print("=" * 60)

    start_time = time.time()

    # generate sample data if requested
    if args.generate_sample:
        sample_path = os.path.join("data", "sample_ngsim.csv")
        os.makedirs("data", exist_ok=True)
        generate_sample_data(sample_path)
        args.data = sample_path

    # step 1: load ngsim data (locally or from azure)
    print("\n[STEP 1] DATA INGESTION")
    print("-" * 40)

    blob_service = None
    if args.azure:
        try:
            blob_service = connect_blob_storage()
            print("  Connected to Azure Blob Storage.")
            blob_name = upload_raw_data(blob_service, args.data)
            print("  Downloading data from Azure for processing...")
            df = download_raw_data(blob_service, blob_name)
        except Exception as e:
            print(f"  Azure connection failed: {e}")
            print("  Falling back to local file loading...")
            df = load_ngsim_local(args.data)
    else:
        print("  Running in local mode (use --azure for cloud integration).")
        df = load_ngsim_local(args.data)

    # step 2: clean the data - filter lanes, remove bad records
    print("\n[STEP 2] DATA PREPROCESSING")
    print("-" * 40)
    df = preprocess(df)

    # step 3: run all 3 scenario detectors using sliding window
    print("\n[STEP 3] SCENARIO DETECTION")
    print("-" * 40)

    print("  Detecting car-following scenarios...")
    cf_scenarios = detect_car_following(df)
    print(f"  Found {len(cf_scenarios)} car-following scenarios.")

    print("  Detecting stop-and-go scenarios...")
    sg_scenarios = detect_stop_and_go(df)
    print(f"  Found {len(sg_scenarios)} stop-and-go scenarios.")

    print("  Detecting lane change scenarios...")
    lc_scenarios = detect_lane_change(df)
    print(f"  Found {len(lc_scenarios)} lane change scenarios.")

    # step 4: format results and save as json + csv, upload to azure
    print("\n[STEP 4] OUTPUT GENERATION")
    print("-" * 40)
    all_scenarios = format_scenarios(cf_scenarios, sg_scenarios, lc_scenarios)

    # save locally
    json_path = os.path.join(args.output_dir, "scenarios_output.json")
    csv_path = os.path.join(args.output_dir, "scenarios_summary.csv")
    save_output_local(all_scenarios, json_path)
    save_output_csv(all_scenarios, csv_path)

    # upload to azure if --azure flag was used
    if blob_service:
        try:
            upload_scenarios_json(blob_service, all_scenarios, "scenarios_output.json")
        except Exception as e:
            print(f"  Azure upload failed: {e}")

    # step 5: generate charts to verify scenarios are correct
    print("\n[STEP 5] VISUALIZATION")
    print("-" * 40)
    visualize_all(df, all_scenarios, args.output_dir)

    print_summary(all_scenarios)
    print_example_outputs(all_scenarios)

    elapsed = time.time() - start_time
    print(f"\nProcessing completed in {elapsed:.2f} seconds.")
    print(f"Results saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
