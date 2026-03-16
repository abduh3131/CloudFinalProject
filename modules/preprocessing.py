import pandas as pd
import numpy as np
import config


# main preprocessing function
# filters to mainline lanes, removes bad data, computes extra features
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("  Preprocessing NGSIM data...")
    original_count = len(df)

    # make sure columns have the right data types
    df = enforce_types(df)

    # remove rows that are missing important fields
    df = df.dropna(subset=["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y", "v_Vel", "Lane_ID"])

    # only keep mainline lanes 1-5 (removes ramps and auxiliary lanes)
    df = df[df["Lane_ID"].isin(config.LC_LANE_MAINLINE)].copy()

    # remove bad sensor data - no negative speeds, no crazy acceleration
    df = df[df["v_Vel"] >= 0].copy()
    df = df[df["v_Acc"].abs() < 40].copy()  # over 40 ft/s^2 is a sensor error

    # sort by vehicle and frame so everything is in order
    df = df.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)

    # add extra useful columns
    df = compute_derived_features(df)

    removed = original_count - len(df)
    print(f"  Preprocessing complete: {len(df)} records retained ({removed} removed).")
    print(f"  Unique vehicles: {df['Vehicle_ID'].nunique()}, Frame range: {df['Frame_ID'].min()}-{df['Frame_ID'].max()}")
    return df


# make sure vehicle_id and frame_id are ints, speeds and positions are floats
def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    int_cols = ["Vehicle_ID", "Frame_ID", "Total_Frames", "v_Class", "Lane_ID", "Preceding", "Following"]
    float_cols = ["Global_Time", "Local_X", "Local_Y", "Global_X", "Global_Y",
                  "v_Length", "v_Width", "v_Vel", "v_Acc", "Space_Hdwy", "Time_Hdwy"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# add extra columns - speed in mph, relative time, how many frames each vehicle has
def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # relative time in seconds from the start of the dataset
    min_time = df["Global_Time"].min()
    df["Relative_Time_s"] = (df["Global_Time"] - min_time) / 1000.0

    # convert speed from ft/s to mph for readability
    df["Speed_mph"] = df["v_Vel"] * 0.681818

    # count how many frames each vehicle appears in
    vehicle_frame_counts = df.groupby("Vehicle_ID")["Frame_ID"].transform("count")
    df["Vehicle_Frame_Count"] = vehicle_frame_counts

    return df
