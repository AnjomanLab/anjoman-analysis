#!/usr/bin/env python3
"""
Anjoman Swarm Firmware - 1x1 Meter Square Maneuver (UMBmark) Benchmark Analyzer
Evaluates Step 3 Decoupled Kalman Filter, loop closure return-to-origin error,
and renders high-resolution 2D individual robot footprint trajectories.
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ==============================================================================
# 1. HARDWARE SPECIFICATIONS PER ROBOT
# ==============================================================================
ROBOT_DIMS = {
    1: {"track_width": 0.1237, "length": 0.165, "width": 0.145, "name": "Robot 1"},
    2: {"track_width": 0.1248, "length": 0.160, "width": 0.135, "name": "Robot 2"},
    3: {"track_width": 0.1163, "length": 0.160, "width": 0.135, "name": "Robot 3"},
    4: {"track_width": 0.1209, "length": 0.160, "width": 0.135, "name": "Robot 4"}
}

WHEEL_DIAMETER_M = 0.0537
WHEEL_WIDTH_M    = 0.0280
OUTPUT_DIR       = "analysis_results_square"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. 2D ROBOT FOOTPRINT RENDERING HELPER
# ==============================================================================
def draw_robot_footprint(ax, x, y, theta_deg, dims, body_color="#1f77b4", alpha=0.6, label=None):
    """
    Renders 2D oriented chassis and wheels at position (x,y) with heading theta_deg.
    Convention: +Y is forward (theta=0), turning CCW (+theta) rotates towards -X.
    """
    th = math.radians(theta_deg)
    # Unit vectors: forward (uf) and lateral-left (ul)
    uf = np.array([-math.sin(th), math.cos(th)])
    ul = np.array([-math.cos(th), -math.sin(th)])
    ur = -ul

    L = dims["length"]
    W = dims["width"]
    tw = dims["track_width"]

    # Chassis corners
    center = np.array([x, y])
    fl = center + (L / 2.0) * uf + (W / 2.0) * ul
    fr = center + (L / 2.0) * uf + (W / 2.0) * ur
    rr = center - (L / 2.0) * uf + (W / 2.0) * ur
    rl = center - (L / 2.0) * uf + (W / 2.0) * ul

    chassis = Polygon([fl, fr, rr, rl], closed=True, facecolor=body_color, edgecolor="black",
                      alpha=alpha, linewidth=1.2, zorder=5, label=label)
    ax.add_patch(chassis)

    # Render Left and Right Wheels
    w_len = WHEEL_DIAMETER_M
    w_wid = WHEEL_WIDTH_M

    for is_left in [True, False]:
        side_vec = ul if is_left else ur
        w_center = center + (tw / 2.0) * side_vec
        w_fl = w_center + (w_len / 2.0) * uf + (w_wid / 2.0) * side_vec
        w_fr = w_center + (w_len / 2.0) * uf - (w_wid / 2.0) * side_vec
        w_rr = w_center - (w_len / 2.0) * uf - (w_wid / 2.0) * side_vec
        w_rl = w_center - (w_len / 2.0) * uf + (w_wid / 2.0) * side_vec
        wheel = Polygon([w_fl, w_fr, w_rr, w_rl], closed=True, facecolor="black",
                        edgecolor="gray", alpha=0.85, zorder=6)
        ax.add_patch(wheel)

    # Heading Arrow
    arrow_len = 0.12
    ax.arrow(x, y, uf[0] * arrow_len, uf[1] * arrow_len, head_width=0.035,
             head_length=0.035, fc="darkred", ec="darkred", zorder=7)

# ==============================================================================
# 3. SQUARE MANEUVER DATA PROCESSING ENGINE
# ==============================================================================
def process_square_data(filepath):
    try:
        base = os.path.basename(filepath)
        rid = int(base.split("square_1m_r")[1].split(".csv")[0])
    except Exception:
        return None

    dims = ROBOT_DIMS.get(rid, ROBOT_DIMS[2])
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    t_raw = df["elapsed_ms"].values / 1000.0
    t = t_raw - t_raw[0]

    posX = df["pos_x_m"].values
    posY = df["pos_y_m"].values
    yaw_deg = df["yaw_deg"].values
    bias_dps = df["bias_dps"].values if "bias_dps" in df.columns else np.zeros(len(df))
    slip = df["slip"].values if "slip" in df.columns else np.zeros(len(df))
    state = df["state"].values

    # 1. Loop Closure & Return-To-Origin Metrics
    final_x = posX[-1]
    final_y = posY[-1]
    final_yaw = yaw_deg[-1]

    pos_error_m = math.sqrt(final_x**2 + final_y**2)
    pos_error_mm = pos_error_m * 1000.0
    yaw_error_deg = final_yaw - 360.0

    # Total actual path length
    dx = np.diff(posX)
    dy = np.diff(posY)
    path_len_m = np.sum(np.sqrt(dx**2 + dy**2))

    # 2. Leg Lengths Extraction (States 1, 2, 3, 4)
    leg_lengths = []
    for s in [1, 2, 3, 4]:
        idx = np.where(state == s)[0]
        if len(idx) > 1:
            seg_dx = np.diff(posX[idx])
            seg_dy = np.diff(posY[idx])
            seg_len = np.sum(np.sqrt(seg_dx**2 + seg_dy**2))
            leg_lengths.append(seg_len)
        else:
            leg_lengths.append(1.0)

    # 3. Corner Turn Angles (States 11, 12, 13, 14)
    turn_angles = []
    for s in [11, 12, 13, 14]:
        idx = np.where(state == s)[0]
        if len(idx) > 1:
            turn_d = yaw_deg[idx[-1]] - yaw_deg[idx[0]]
            turn_angles.append(turn_d)
        else:
            turn_angles.append(90.0)

    # 4. Kalman Filter Metrics
    active_mask = (state > 0)
    slip_percentage = (np.sum(slip[active_mask]) / max(1, np.sum(active_mask))) * 100.0
    steady_bias_dps = np.mean(bias_dps[-300:]) if len(bias_dps) > 300 else bias_dps[-1]

    # Power metrics
    vbus = df["vbus_v"].values
    curr_ma = df["current_ma"].values
    v_drop = vbus[0] - np.min(vbus[active_mask])
    mean_curr = np.mean(curr_ma[active_mask])

    return {
        "id": rid,
        "dims": dims,
        "df": df,
        "t": t,
        "posX": posX,
        "posY": posY,
        "yaw_deg": yaw_deg,
        "bias_dps": bias_dps,
        "slip": slip,
        "state": state,
        "final_x": final_x,
        "final_y": final_y,
        "final_yaw": final_yaw,
        "pos_error_mm": pos_error_mm,
        "yaw_error_deg": yaw_error_deg,
        "path_len_m": path_len_m,
        "leg_lengths": leg_lengths,
        "turn_angles": turn_angles,
        "slip_percentage": slip_percentage,
        "steady_bias_dps": steady_bias_dps,
        "v_drop": v_drop,
        "mean_curr": mean_curr
    }

# ==============================================================================
# 4. RUN PROCESSING
# ==============================================================================
files = sorted(glob.glob("square_1m_r*.csv"))
if not files:
    print("[ERROR] No files matching 'square_1m_r*.csv' found in current directory!")
    exit(1)

results = [process_square_data(f) for f in files if process_square_data(f) is not None]
results.sort(key=lambda x: x["id"])

# Print Summary Report
lines = []
def p(msg=""):
    print(msg)
    lines.append(msg)

p("=" * 115)
p("   ANJOMAN 1x1m SQUARE BENCHMARK (UMBMARK) - KALMAN FILTER VALIDATION REPORT")
p("=" * 115)
p(f"{'Robot':<7}|{'Final X (m)':<12}|{'Final Y (m)':<12}|{'Return Error':<14}|{'Final Yaw':<12}|{'Yaw Error':<12}|{'Path Len (m)':<13}|{'Slip Gating'}")
p("-" * 115)

for r in results:
    p(f"R{r['id']:<6}| {r['final_x']:>9.4f} m | {r['final_y']:>9.4f} m | {r['pos_error_mm']:>9.1f} mm | {r['final_yaw']:>8.2f}°   | {r['yaw_error_deg']:>8.2f}°   | {r['path_len_m']:>10.4f} m | {r['slip_percentage']:>7.2f} %")

p("-" * 115)
p("\n" + "=" * 115)
p("   DETAILED SEGMENT KINEMATICS (Leg Targets: 1.000m | Turn Targets: 90.0 deg)")
p("=" * 115)
p(f"{'Robot':<7}|{'Leg 1 (m)':<11}|{'Leg 2 (m)':<11}|{'Leg 3 (m)':<11}|{'Leg 4 (m)':<11}|{'Turn 1 (°)':<12}|{'Turn 2 (°)':<12}|{'Turn 3 (°)':<12}|{'Turn 4 (°)'}")
p("-" * 115)

for r in results:
    legs = r["leg_lengths"]
    turns = r["turn_angles"]
    p(f"R{r['id']:<6}| {legs[0]:<9.3f} | {legs[1]:<9.3f} | {legs[2]:<9.3f} | {legs[3]:<9.3f} | {turns[0]:>7.1f}°    | {turns[1]:>7.1f}°    | {turns[2]:>7.1f}°    | {turns[3]:>7.1f}°")

p("=" * 115)

with open(os.path.join(OUTPUT_DIR, "square_report.txt"), "w") as f_out:
    f_out.write("\n".join(lines) + "\n")

# Save Summary CSV
summary_df = pd.DataFrame([{
    "robot_id": r["id"],
    "final_x_m": round(r["final_x"], 4),
    "final_y_m": round(r["final_y"], 4),
    "pos_error_mm": round(r["pos_error_mm"], 1),
    "final_yaw_deg": round(r["final_yaw"], 2),
    "yaw_error_deg": round(r["yaw_error_deg"], 2),
    "path_length_m": round(r["path_len_m"], 4),
    "slip_percentage": round(r["slip_percentage"], 2),
    "gyro_bias_dps": round(r["steady_bias_dps"], 4),
    "leg1_m": round(r["leg_lengths"][0], 3),
    "leg2_m": round(r["leg_lengths"][1], 3),
    "leg3_m": round(r["leg_lengths"][2], 3),
    "leg4_m": round(r["leg_lengths"][3], 3),
    "v_drop_v": round(r["v_drop"], 3),
    "mean_curr_ma": round(r["mean_curr"], 1)
} for r in results])
summary_df.to_csv(os.path.join(OUTPUT_DIR, "square_summary.csv"), index=False)

# ==============================================================================
# 5. INDIVIDUAL HIGH-RESOLUTION 2D ROBOT FOOTPRINT PLOTS (1 PER ROBOT)
# ==============================================================================
# Nominal Square Coordinates: (0,0) -> (0,1) -> (-1,1) -> (-1,0) -> (0,0)
nominal_square = [
    [0.0, 0.0],
    [0.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 0.0],
    [0.0, 0.0]
]
nom_x = [p[0] for p in nominal_square]
nom_y = [p[1] for p in nominal_square]

for r in results:
    rid = r["id"]
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    # Plot Nominal Track
    ax.plot(nom_x, nom_y, color="black", linestyle="--", linewidth=1.5, label="Nominal 1x1m Path", zorder=2)

    # Plot Filtered 2D Odometry Path
    ax.plot(r["posX"], r["posY"], color="#1f77b4", linewidth=2.8, label="Kalman Filter 2D Path", zorder=3)

    # Highlight Corners & Sample Points for Robot Footprint Rendering
    # State transitions: 1->11 (C1), 2->12 (C2), 3->13 (C3), 4->14 (C4)
    state = r["state"]
    sample_indices = [0] # Start
    for s in [11, 12, 13, 14]:
        idx = np.where(state == s)[0]
        if len(idx) > 0:
            sample_indices.append(idx[0])
    sample_indices.append(len(state) - 1) # Endpoint

    for idx_step in sample_indices:
        x_pt = r["posX"][idx_step]
        y_pt = r["posY"][idx_step]
        yaw_pt = r["yaw_deg"][idx_step]
        draw_robot_footprint(ax, x_pt, y_pt, yaw_pt, r["dims"], body_color="#2ca02c", alpha=0.35)

    # Highlight Origin and Endpoint
    ax.scatter([0], [0], color="black", s=150, zorder=8, label="Start Origin (0,0)")
    ax.scatter([r["final_x"]], [r["final_y"]], color="red", s=180, marker="X", zorder=9,
               label=f"Final Return Point (Error: {r['pos_error_mm']:.1f} mm)")

    ax.set_title(f"Robot {rid} - 1x1 Meter Square UMBmark Benchmark\nReturn Error: ΔPos = {r['pos_error_mm']:.1f} mm | ΔYaw = {r['yaw_error_deg']:+.2f}°", fontsize=14, pad=15)
    ax.set_xlabel("Lateral Position X (meters) [Negative = Left]", fontsize=12)
    ax.set_ylabel("Longitudinal Position Y (meters) [Positive = Forward]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper left", fontsize=11)
    ax.axis("equal")
    ax.set_xlim(-1.4, 0.4)
    ax.set_ylim(-0.3, 1.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"01_2d_square_robot_{rid}.png"))
    plt.close()

# ==============================================================================
# 6. COMPOSITE TRAJECTORY & KALMAN FILTER TELEMETRY FIGURES
# ==============================================================================
colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}

# --- FIGURE 2: ALL ROBOTS 2D OVERVIEW ---
plt.figure(figsize=(12, 12), dpi=300)
plt.plot(nom_x, nom_y, color="black", linestyle="--", linewidth=1.5, label="Nominal 1x1m Path")
for r in results:
    rid = r["id"]
    plt.plot(r["posX"], r["posY"], color=colors[rid], linewidth=2.5,
             label=f"R{rid} (ΔPos={r['pos_error_mm']:.1f}mm, ΔYaw={r['yaw_error_deg']:+.1f}°)")
    plt.scatter([r["final_x"]], [r["final_y"]], color=colors[rid], s=120, zorder=5)

plt.scatter([0], [0], color="black", s=140, zorder=6, label="Origin (0,0)")
plt.title("Comparative Multi-Robot 1x1m Square Trajectories (Decoupled Kalman Filter)", fontsize=14, pad=12)
plt.xlabel("X Position (m)", fontsize=12)
plt.ylabel("Y Position (m)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(loc="upper left", fontsize=10)
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_all_robots_trajectories.png"))
plt.close()

# --- FIGURE 3: KALMAN FILTER TELEMETRY & SLIP GATING ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
for r in results:
    rid = r["id"]
    t = r["t"]
    ax1.plot(t, r["yaw_deg"], color=colors[rid], linewidth=2.0, label=f"R{rid} Filtered Yaw (Final={r['final_yaw']:.1f}°)")
    ax2.plot(t, r["bias_dps"], color=colors[rid], linewidth=1.8, label=f"R{rid} Bias Drift (End={r['steady_bias_dps']:.3f}°/s)")

# Mark 90, 180, 270, 360 reference levels
for deg in [90, 180, 270, 360]:
    ax1.axhline(deg, color="gray", linestyle=":", linewidth=1.0)

ax1.set_title("Kalman Filter Estimated Heading Angle: 4x90° Corners to 360° Loop Closure", fontsize=13)
ax1.set_ylabel("Heading Angle (deg)", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.legend(loc="upper left", fontsize=10)

ax2.axhline(0, color="black", linestyle="--", linewidth=1.0)
ax2.set_title("Kalman Filter Real-Time Gyro Z Bias Tracking b_gz(t)", fontsize=13)
ax2.set_xlabel("Elapsed Time (seconds)", fontsize=12)
ax2.set_ylabel("Estimated Bias (deg/s)", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.7)
ax2.legend(loc="lower right", fontsize=10)

plt.suptitle("Step 3 Decoupled Kalman Filter Estimation Dynamics", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_kalman_filter_telemetry.png"), dpi=300)
plt.close()

p(f"\n[SUCCESS] Generated individual footprint plots and reports in '{OUTPUT_DIR}/'.")
