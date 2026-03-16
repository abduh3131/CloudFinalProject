import pandas as pd
import numpy as np
import config


# main function - scan all vehicles and find stop-and-go scenarios
# looks for pattern: hard braking -> near stop -> acceleration
# also checks that nearby cars are slow too (real congestion not just one car)
def detect_stop_and_go(df: pd.DataFrame) -> list:
    scenarios = []
    window = config.FRAMES_PER_WINDOW  # 50 records = 5 seconds at 10hz
    vehicle_ids = df["Vehicle_ID"].unique()

    for ego_id in vehicle_ids:
        ego_data = df[df["Vehicle_ID"] == ego_id].sort_values("Frame_ID").reset_index(drop=True)

        if len(ego_data) < window:
            continue

        # slide a window of 50 records across this vehicles data
        i = 0
        while i <= len(ego_data) - window:
            ego_window = ego_data.iloc[i:i + window]

            # vehicle has to stay in the same lane
            if ego_window["Lane_ID"].nunique() != 1:
                i += window // 5
                continue

            ego_lane = ego_window["Lane_ID"].iloc[0]
            start_frame = ego_window["Frame_ID"].iloc[0]
            end_frame = ego_window["Frame_ID"].iloc[-1]

            # get the acceleration and speed values for this window
            accel_values = ego_window["v_Acc"].values
            vel_values = ego_window["v_Vel"].values

            # check 1: window must have all three phases - braking, near stop, speeding up
            has_decel = np.any(accel_values < config.SG_DECEL_THRESHOLD_FT_S2)      # hard braking
            has_accel = np.any(accel_values > config.SG_ACCEL_THRESHOLD_FT_S2)      # speeding up after
            has_low_speed = np.any(vel_values < config.SG_LOW_SPEED_THRESHOLD_FT_S) # almost stopped

            if not (has_decel and has_accel and has_low_speed):
                i += window // 5
                continue

            # check 2: braking has to happen BEFORE the acceleration (right order)
            decel_indices = np.where(accel_values < config.SG_DECEL_THRESHOLD_FT_S2)[0]
            accel_indices = np.where(accel_values > config.SG_ACCEL_THRESHOLD_FT_S2)[0]

            if decel_indices[0] >= accel_indices[-1]:
                i += window // 5
                continue

            # check 3: nearby cars must also be slow/braking (proves its real traffic congestion)
            nearby_congested = _check_nearby_congestion(
                df, ego_id, ego_lane, ego_window, start_frame, end_frame
            )

            if not nearby_congested:
                i += window // 5
                continue

            # all checks passed - this is a stop-and-go scenario
            surrounding = _get_surrounding_vehicles(df, ego_id, ego_lane, ego_window, start_frame, end_frame)

            min_speed = float(np.min(vel_values))
            max_decel = float(np.min(accel_values))
            max_accel = float(np.max(accel_values))

            # save the scenario with all the details
            scenario = {
                "scenario_type": "stop_and_go",
                "ego_vehicle_id": int(ego_id),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time_ms": int(ego_window["Global_Time"].iloc[0]),
                "end_time_ms": int(ego_window["Global_Time"].iloc[-1]),
                "ego_lane": int(ego_lane),
                "min_speed_ft_s": round(min_speed, 2),
                "max_deceleration_ft_s2": round(max_decel, 2),
                "max_acceleration_ft_s2": round(max_accel, 2),
                "surrounding_vehicles": surrounding,
                "ego_trajectory": _extract_trajectory(ego_window),
            }
            scenarios.append(scenario)

            i += window
            continue

    return scenarios


# check if nearby cars are also going slow or braking
# this proves its real traffic congestion not just one car randomly braking
def _check_nearby_congestion(df, ego_id, ego_lane, ego_window, start_frame, end_frame):
    ego_y_mean = ego_window["Local_Y"].mean()
    nearby = df[
        (df["Frame_ID"] >= start_frame) &
        (df["Frame_ID"] <= end_frame) &
        (df["Vehicle_ID"] != ego_id) &
        (df["Lane_ID"].between(ego_lane - 1, ego_lane + 1)) &
        (df["Local_Y"].between(ego_y_mean - config.LC_SURROUNDING_RADIUS_FT,
                                ego_y_mean + config.LC_SURROUNDING_RADIUS_FT))
    ]

    if nearby.empty:
        return False

    # check up to 10 nearby vehicles
    for vid in nearby["Vehicle_ID"].unique()[:10]:
        v_data = nearby[nearby["Vehicle_ID"] == vid]
        if len(v_data) >= 5:
            if (v_data["v_Vel"].min() < config.SG_LOW_SPEED_THRESHOLD_FT_S or
                v_data["v_Acc"].min() < config.SG_DECEL_THRESHOLD_FT_S2):
                return True
    return False


# find vehicles near the ego vehicle during this window
def _get_surrounding_vehicles(df, ego_id, ego_lane, ego_window, start_frame, end_frame):
    ego_y_mean = ego_window["Local_Y"].mean()
    surrounding_mask = (
        (df["Frame_ID"] >= start_frame) &
        (df["Frame_ID"] <= end_frame) &
        (df["Vehicle_ID"] != ego_id) &
        (df["Local_Y"].between(ego_y_mean - config.LC_SURROUNDING_RADIUS_FT,
                                ego_y_mean + config.LC_SURROUNDING_RADIUS_FT)) &
        (df["Lane_ID"].between(ego_lane - 1, ego_lane + 1))
    )
    return sorted(df[surrounding_mask]["Vehicle_ID"].unique().tolist())


# grab a few data points from the vehicles trajectory for the output
def _extract_trajectory(vehicle_data):
    data = vehicle_data.reset_index(drop=True)
    sampled = data.iloc[::10] if len(data) > 10 else data
    trajectory = []
    for _, row in sampled.iterrows():
        trajectory.append({
            "frame": int(row["Frame_ID"]),
            "x": round(float(row["Local_X"]), 2),
            "y": round(float(row["Local_Y"]), 2),
            "vel": round(float(row["v_Vel"]), 2),
            "acc": round(float(row["v_Acc"]), 2),
            "lane": int(row["Lane_ID"])
        })
    return trajectory
