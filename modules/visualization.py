import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # dont show windows, just save files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import config


# generate all 6 visualizations
def visualize_all(df: pd.DataFrame, scenarios: list, output_dir: str):
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    print("  Generating scenario visualizations...")

    # bar chart showing total count of each type
    plot_summary(scenarios, viz_dir)

    # one example chart for each scenario type to prove they work
    cf = [s for s in scenarios if s["scenario_type"] == "car_following"]
    sg = [s for s in scenarios if s["scenario_type"] == "stop_and_go"]
    lc = [s for s in scenarios if s["scenario_type"] == "lane_change"]

    if cf:
        plot_car_following(df, cf[0], viz_dir)
    if sg:
        plot_stop_and_go(df, sg[0], viz_dir)
    if lc:
        plot_lane_change(df, lc[0], viz_dir)

    # which lanes had the most scenarios
    plot_lane_distribution(scenarios, viz_dir)

    # speed comparison across scenario types
    plot_speed_distributions(scenarios, viz_dir)

    count = len([f for f in os.listdir(viz_dir) if f.endswith(".png")])
    print(f"  {count} visualizations saved to: {viz_dir}/")


# bar chart - how many of each scenario type were detected
def plot_summary(scenarios, viz_dir):
    cf_count = sum(1 for s in scenarios if s["scenario_type"] == "car_following")
    sg_count = sum(1 for s in scenarios if s["scenario_type"] == "stop_and_go")
    lc_count = sum(1 for s in scenarios if s["scenario_type"] == "lane_change")

    fig, ax = plt.subplots(figsize=(8, 5))
    types = ["Car-Following", "Stop-and-Go", "Lane Change"]
    counts = [cf_count, sg_count, lc_count]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    bars = ax.bar(types, counts, color=colors, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=14)

    ax.set_ylabel("Number of Scenarios", fontsize=12)
    ax.set_title("Detected Driving Scenarios - NGSIM US-101", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    total = sum(counts)
    ax.text(0.98, 0.95, f"Total: {total}", transform=ax.transAxes,
            ha="right", va="top", fontsize=12, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "scenario_summary.png"), dpi=150)
    plt.close()


# plot one car-following example - shows both vehicles speeds should be similar
def plot_car_following(df, scenario, viz_dir):
    ego_id = scenario["ego_vehicle_id"]
    lead_id = scenario["lead_vehicle_id"]
    start = scenario["start_frame"]
    end = scenario["end_frame"]

    ego_data = df[(df["Vehicle_ID"] == ego_id) &
                  (df["Frame_ID"] >= start) & (df["Frame_ID"] <= end)].sort_values("Frame_ID")
    lead_data = df[(df["Vehicle_ID"] == lead_id) &
                   (df["Frame_ID"] >= start) & (df["Frame_ID"] <= end)].sort_values("Frame_ID")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(f"Car-Following Verification - {scenario['scenario_id']}", fontsize=14, fontweight="bold")

    # top: speed comparison
    ax1.plot(range(len(ego_data)), ego_data["v_Vel"].values, "b-", linewidth=2, label=f"Ego (Vehicle {ego_id})")
    ax1.plot(range(len(lead_data)), lead_data["v_Vel"].values, "r--", linewidth=2, label=f"Lead (Vehicle {lead_id})")
    ax1.set_ylabel("Speed (ft/s)", fontsize=11)
    ax1.set_title("Speed Comparison (similar speeds = correct car-following)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # bottom: gap between vehicles
    if "Space_Hdwy" in ego_data.columns:
        hdwy = ego_data["Space_Hdwy"].values
        ax2.plot(range(len(ego_data)), hdwy, "g-", linewidth=2, label="Space Headway")
        ax2.axhline(y=np.mean(hdwy), color="gray", linestyle=":", label=f"Avg: {np.mean(hdwy):.1f} ft")
        ax2.set_ylabel("Headway (ft)", fontsize=11)
        ax2.set_title("Space Headway (consistent gap = correct car-following)", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    ax2.set_xlabel("Record Index (within 5-second window)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "car_following_example.png"), dpi=150)
    plt.close()


# plot one stop-and-go example - shows speed dropping and acceleration pattern
def plot_stop_and_go(df, scenario, viz_dir):
    ego_id = scenario["ego_vehicle_id"]
    start = scenario["start_frame"]
    end = scenario["end_frame"]

    ego_data = df[(df["Vehicle_ID"] == ego_id) &
                  (df["Frame_ID"] >= start) & (df["Frame_ID"] <= end)].sort_values("Frame_ID")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Stop-and-Go Verification - {scenario['scenario_id']}", fontsize=14, fontweight="bold")

    x = range(len(ego_data))

    # top: speed over time - should drop below threshold
    ax1.plot(x, ego_data["v_Vel"].values, "b-", linewidth=2, label=f"Vehicle {ego_id}")
    ax1.axhline(y=config.SG_LOW_SPEED_THRESHOLD_FT_S, color="red", linestyle=":",
                label=f"Low speed threshold ({config.SG_LOW_SPEED_THRESHOLD_FT_S} ft/s)")
    ax1.fill_between(x, 0, config.SG_LOW_SPEED_THRESHOLD_FT_S, alpha=0.1, color="red")
    ax1.set_ylabel("Speed (ft/s)", fontsize=11)
    ax1.set_title("Speed (drops to near-stop = correct stop-and-go)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # bottom: acceleration - red bars = braking, green bars = speeding up
    acc = ego_data["v_Acc"].values
    colors = ["red" if a < config.SG_DECEL_THRESHOLD_FT_S2 else
              "green" if a > config.SG_ACCEL_THRESHOLD_FT_S2 else "gray" for a in acc]
    ax2.bar(x, acc, color=colors, width=1.0, alpha=0.7)
    ax2.axhline(y=config.SG_DECEL_THRESHOLD_FT_S2, color="red", linestyle=":",
                label=f"Decel threshold ({config.SG_DECEL_THRESHOLD_FT_S2} ft/s\u00B2)")
    ax2.axhline(y=config.SG_ACCEL_THRESHOLD_FT_S2, color="green", linestyle=":",
                label=f"Accel threshold ({config.SG_ACCEL_THRESHOLD_FT_S2} ft/s\u00B2)")
    ax2.set_ylabel("Acceleration (ft/s\u00B2)", fontsize=11)
    ax2.set_xlabel("Record Index (within 5-second window)", fontsize=11)
    ax2.set_title("Acceleration (red=braking, green=accelerating)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "stop_and_go_example.png"), dpi=150)
    plt.close()


# plot one lane change example - shows lane id changing and lateral position shifting
def plot_lane_change(df, scenario, viz_dir):
    ego_id = scenario["ego_vehicle_id"]
    start = scenario["start_frame"]
    end = scenario["end_frame"]

    ego_data = df[(df["Vehicle_ID"] == ego_id) &
                  (df["Frame_ID"] >= start) & (df["Frame_ID"] <= end)].sort_values("Frame_ID")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Lane Change Verification - {scenario['scenario_id']}", fontsize=14, fontweight="bold")

    x = range(len(ego_data))

    # top: lane id over time - should change from source to destination
    ax1.plot(x, ego_data["Lane_ID"].values, "b-o", linewidth=2, markersize=3)
    ax1.set_ylabel("Lane ID", fontsize=11)
    ax1.set_yticks(range(1, 6))
    ax1.set_title(f"Lane ID (change from {scenario['source_lane']} to {scenario['destination_lane']} = correct lane change)", fontsize=11)
    ax1.grid(True, alpha=0.3)

    src = scenario["source_lane"]
    dst = scenario["destination_lane"]
    ax1.axhline(y=src, color="blue", linestyle=":", alpha=0.5, label=f"Source: Lane {src}")
    ax1.axhline(y=dst, color="red", linestyle=":", alpha=0.5, label=f"Dest: Lane {dst}")
    ax1.legend(fontsize=9)

    # middle: lateral position - should shift sideways
    ax2.plot(x, ego_data["Local_X"].values, "g-", linewidth=2)
    ax2.set_ylabel("Lateral Position (ft)", fontsize=11)
    ax2.set_title("Lateral Position (shift confirms physical lane change)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # bottom: speed during the lane change
    ax3.plot(x, ego_data["v_Vel"].values, "orange", linewidth=2)
    ax3.set_ylabel("Speed (ft/s)", fontsize=11)
    ax3.set_xlabel("Record Index (within 5-second window)", fontsize=11)
    ax3.set_title("Speed During Lane Change", fontsize=11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "lane_change_example.png"), dpi=150)
    plt.close()


# bar chart - how many scenarios in each lane
def plot_lane_distribution(scenarios, viz_dir):
    lane_counts = {}
    for s in scenarios:
        lane = s.get("ego_lane", 0)
        stype = s["scenario_type"]
        key = (lane, stype)
        lane_counts[key] = lane_counts.get(key, 0) + 1

    fig, ax = plt.subplots(figsize=(10, 5))
    lanes = [1, 2, 3, 4, 5]
    types = ["car_following", "stop_and_go", "lane_change"]
    labels = ["Car-Following", "Stop-and-Go", "Lane Change"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    bar_width = 0.25
    for i, (stype, label, color) in enumerate(zip(types, labels, colors)):
        counts = [lane_counts.get((l, stype), 0) for l in lanes]
        positions = [l + (i - 1) * bar_width for l in lanes]
        ax.bar(positions, counts, bar_width, label=label, color=color, edgecolor="white")

    ax.set_xlabel("Lane ID", fontsize=12)
    ax.set_ylabel("Number of Scenarios", fontsize=12)
    ax.set_title("Scenario Distribution Across Lanes", fontsize=14, fontweight="bold")
    ax.set_xticks(lanes)
    ax.set_xticklabels([f"Lane {l}" for l in lanes])
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "lane_distribution.png"), dpi=150)
    plt.close()


# box plot - comparing speeds across scenario types
def plot_speed_distributions(scenarios, viz_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    data = {"Car-Following": [], "Stop-and-Go": [], "Lane Change": []}
    for s in scenarios:
        speed = s.get("ego_avg_speed_ft_s", None)
        if speed is None and "ego_trajectory" in s and s["ego_trajectory"]:
            speed = np.mean([t["vel"] for t in s["ego_trajectory"]])
        if speed is not None:
            if s["scenario_type"] == "car_following":
                data["Car-Following"].append(speed)
            elif s["scenario_type"] == "stop_and_go":
                data["Stop-and-Go"].append(speed)
            elif s["scenario_type"] == "lane_change":
                data["Lane Change"].append(speed)

    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    bp = ax.boxplot([data[k] for k in data.keys()], labels=data.keys(), patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Average Speed (ft/s)", fontsize=12)
    ax.set_title("Speed Distribution by Scenario Type", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "speed_distributions.png"), dpi=150)
    plt.close()
