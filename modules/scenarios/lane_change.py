import pandas as pd
import numpy as np
import config


# main function - scan all vehicles and find lane change scenarios
# checks: lane id changes, lanes are adjacent (diff of 1), both lanes are mainline
def detect_lane_change(df: pd.DataFrame) -> list:
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

            start_frame = ego_window["Frame_ID"].iloc[0]
            end_frame = ego_window["Frame_ID"].iloc[-1]

            # check 1: does the lane id change during this window?
            lanes_in_window = ego_window["Lane_ID"].values
            unique_lanes = np.unique(lanes_in_window)

            if len(unique_lanes) < 2:
                i += window // 5
                continue

            # find the exact point where the lane changes
            lane_changes = np.where(np.diff(lanes_in_window) != 0)[0]
            if len(lane_changes) == 0:
                i += window // 5
                continue

            change_idx = lane_changes[0]
            source_lane = int(lanes_in_window[change_idx])
            dest_lane = int(lanes_in_window[change_idx + 1])

            # check 2: lanes must be adjacent (difference of 1, like lane 2 to lane 3)
            if abs(dest_lane - source_lane) != 1:
                i += window // 5
                continue

            # check 3: both lanes have to be mainline lanes 1-5 (no ramps)
            if source_lane not in config.LC_LANE_MAINLINE or dest_lane not in config.LC_LANE_MAINLINE:
                i += window // 5
                continue

            # all checks passed - this is a lane change
            change_frame = int(ego_window["Frame_ID"].iloc[change_idx])

            # find nearby vehicles in both the source and destination lanes
            surrounding = _get_surrounding_both_lanes(
                df, ego_id, source_lane, dest_lane, ego_window, start_frame, end_frame
            )

            # save the scenario with all the details
            scenario = {
                "scenario_type": "lane_change",
                "ego_vehicle_id": int(ego_id),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time_ms": int(ego_window["Global_Time"].iloc[0]),
                "end_time_ms": int(ego_window["Global_Time"].iloc[-1]),
                "source_lane": int(source_lane),
                "destination_lane": int(dest_lane),
                "ego_lane": int(source_lane),
                "change_direction": "left" if dest_lane < source_lane else "right",
                "change_frame": change_frame,
                "ego_avg_speed_ft_s": round(float(ego_window["v_Vel"].mean()), 2),
                "surrounding_vehicles": surrounding,
                "ego_trajectory": _extract_trajectory(ego_window),
            }
            scenarios.append(scenario)

            i += window
            continue

    return scenarios


# find nearby vehicles in both the source and destination lanes
def _get_surrounding_both_lanes(df, ego_id, source_lane, dest_lane, ego_window, start_frame, end_frame):
    ego_y_mean = ego_window["Local_Y"].mean()
    lanes_to_check = set()
    for lane in [source_lane, dest_lane]:
        lanes_to_check.add(lane)
        lanes_to_check.add(lane - 1)
        lanes_to_check.add(lane + 1)
    lanes_to_check = lanes_to_check.intersection(set(config.LC_LANE_MAINLINE))

    surrounding_mask = (
        (df["Frame_ID"] >= start_frame) &
        (df["Frame_ID"] <= end_frame) &
        (df["Vehicle_ID"] != ego_id) &
        (df["Local_Y"].between(ego_y_mean - config.LC_SURROUNDING_RADIUS_FT,
                                ego_y_mean + config.LC_SURROUNDING_RADIUS_FT)) &
        (df["Lane_ID"].isin(lanes_to_check))
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
