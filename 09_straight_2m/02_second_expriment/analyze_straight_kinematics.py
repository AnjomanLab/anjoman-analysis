#!/usr/bin/env python3
"""
Anjoman Swarm Firmware - 2-Meter Straight-Line Kinematics & Velocity Benchmark Analyzer
Step 1 & Step 2.1 Combined Metrology Engine
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. GROUND TRUTH COORDINATES & PHYSICAL CONSTANTS
# ==============================================================================
# Ground truth coordinates measured manually on the floor (in meters)
# Convention: +y = forward motion, +x = lateral deviation to the right
GROUND_TRUTH = {
    1: {"xf": -0.55, "yf": 2.05, "track_width": 0.1350},
    2: {"xf": -0.77, "yf": 1.95, "track_width": 0.1250},
    3: {"xf": -0.55, "yf": 2.07, "track_width": 0.1250},
    4: {"xf":  1.42, "yf": 1.45, "track_width": 0.1250}
}

NOMINAL_WHEEL_RADIUS_M = 0.0250  # 25 mm
NOMINAL_TARGET_DIST_M  = 2.0000  # 2.000 m
CPR                    = 4096.0

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. METROLOGY & KINEMATICS EXTRACTION ENGINE
# ==============================================================================
def analyze_robot(robot_id, filepath, gt):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Filter cruising phase (state == 1)
    cruise_df = df[df["state"] == 1].copy()
    if cruise_df.empty:
        cruise_df = df.copy()

    # Time in seconds
    t_start = cruise_df["elapsed_ms"].iloc[0] / 1000.0
    t = (cruise_df["elapsed_ms"] / 1000.0) - t_start

    # 1. Closed-Loop Velocity Tracking Metrics (Step 1)
    target_rpm = cruise_df["target_rpm_l"].iloc[0]
    rpm_l = cruise_df["meas_rpm_l"].values
    rpm_r = cruise_df["meas_rpm_r"].values

    # Steady state: skip initial 1.0s acceleration transient
    steady_mask = t > 1.0
    if not np.any(steady_mask):
        steady_mask = np.ones(len(t), dtype=bool)

    mean_rpm_l = np.mean(rpm_l[steady_mask])
    mean_rpm_r = np.mean(rpm_r[steady_mask])
    std_rpm_l  = np.std(rpm_l[steady_mask])
    std_rpm_r  = np.std(rpm_r[steady_mask])

    rmse_rpm_l = np.sqrt(np.mean((rpm_l[steady_mask] - target_rpm)**2))
    rmse_rpm_r = np.sqrt(np.mean((rpm_r[steady_mask] - target_rpm)**2))

    # Power metrics
    vbus_init  = df["vbus_v"].iloc[0]
    vbus_final = df["vbus_v"].iloc[-1]
    vbus_drop  = vbus_init - vbus_final
    mean_curr  = cruise_df["current_ma"].mean()
    peak_curr  = cruise_df["current_ma"].max()

    # 2. Kinematics & Arc Reconstruction (Step 2.1)
    final_steps_l = df["steps_l"].iloc[-1]
    final_steps_r = df["steps_r"].iloc[-1]
    avg_steps     = (final_steps_l + final_steps_r) / 2.0

    dist_nom_l = (final_steps_l / CPR) * (2.0 * math.pi * NOMINAL_WHEEL_RADIUS_M)
    dist_nom_r = (final_steps_r / CPR) * (2.0 * math.pi * NOMINAL_WHEEL_RADIUS_M)
    dist_nom_avg = (dist_nom_l + dist_nom_r) / 2.0

    xf = gt["xf"]
    yf = gt["yf"]
    W  = gt["track_width"]

    # Arc Kinematics: Circle passing through (0,0) with tangent on y-axis
    # (x - xc)^2 + y^2 = xc^2  =>  xc = (x^2 + y^2) / (2x)
    xc = (xf**2 + yf**2) / (2.0 * xf)
    Rc = abs(xc)

    # Subtended arc angle (in radians and degrees)
    theta_arc_rad = 2.0 * math.atan2(abs(xf), yf)
    theta_arc_deg = math.degrees(theta_arc_rad)
    if xf < 0:
        # Curving to the left
        theta_arc_deg_signed = -theta_arc_deg
    else:
        # Curving to the right
        theta_arc_deg_signed = +theta_arc_deg

    # Actual ground arc length
    S_actual = Rc * theta_arc_rad

    # Effective Wheel Radius (r_eff)
    r_eff = NOMINAL_WHEEL_RADIUS_M * (S_actual / dist_nom_avg)
    scale_factor = S_actual / dist_nom_avg

    # Wheel Diameter Ratio (Ed = r_R / r_L)
    # Along an arc: S_R / S_L = (Rc + W/2) / (Rc - W/2)
    # If turning left (xf < 0), Right wheel travels on outside: S_R > S_L => Ed > 1
    # If turning right (xf > 0), Left wheel travels on outside: S_L > S_R => Ed < 1
    if xf < 0:
        Ed_ground = (Rc + W / 2.0) / (Rc - W / 2.0)
    else:
        Ed_ground = (Rc - W / 2.0) / (Rc + W / 2.0)

    # Encoder ticks ratio
    Ed_encoder = (final_steps_r / final_steps_l) if final_steps_l != 0 else 1.0

    # Heading from IMU Gyro
    final_imu_yaw = df["yaw_deg"].iloc[-1]
    gyro_z_mean = df["gyro_z_dps"].iloc[:200].mean()

    return {
        "robot_id": robot_id,
        "df": df,
        "cruise_df": cruise_df,
        "t": t,
        "target_rpm": target_rpm,
        "mean_rpm_l": mean_rpm_l,
        "mean_rpm_r": mean_rpm_r,
        "rmse_rpm_l": rmse_rpm_l,
        "rmse_rpm_r": rmse_rpm_r,
        "std_rpm_l": std_rpm_l,
        "std_rpm_r": std_rpm_r,
        "vbus_init": vbus_init,
        "vbus_final": vbus_final,
        "vbus_drop": vbus_drop,
        "mean_curr": mean_curr,
        "peak_curr": peak_curr,
        "steps_l": final_steps_l,
        "steps_r": final_steps_r,
        "dist_nom_l": dist_nom_l,
        "dist_nom_r": dist_nom_r,
        "dist_nom_avg": dist_nom_avg,
        "xf": xf,
        "yf": yf,
        "Rc": Rc,
        "theta_arc_deg": theta_arc_deg_signed,
        "S_actual": S_actual,
        "r_eff_mm": r_eff * 1000.0,
        "scale_factor": scale_factor,
        "Ed_ground": Ed_ground,
        "Ed_encoder": Ed_encoder,
        "imu_yaw_deg": final_imu_yaw
    }

# ==============================================================================
# 3. RUN BATCH PROCESSING
# ==============================================================================
files = sorted(glob.glob("straight_2m_r*.csv"))
if not files:
    print("[ERROR] No 'straight_2m_r*.csv' files found in the current directory!")
    exit(1)

results = []
for f in files:
    # Extract robot ID from filename
    try:
        rid = int(f.split("straight_2m_r")[1].split(".csv")[0])
    except Exception:
        continue
    if rid in GROUND_TRUTH:
        res = analyze_robot(rid, f, GROUND_TRUTH[rid])
        results.append(res)

if not results:
    print("[ERROR] Could not match any files to defined ground truth robot IDs!")
    exit(1)

# ==============================================================================
# 4. PRINT REPORT & SAVE SUMMARY FILES
# ==============================================================================
report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(msg)

log("=" * 95)
log("   ANJOMAN 2-METER BENCHMARK - COMPREHENSIVE CONTROL & KINEMATICS REPORT")
log("=" * 95)
log(f"{'Robot':<7}|{'r_eff (mm)':<12}|{'Scale':<8}|{'Ed (Ground)':<13}|{'Ed (Enc)':<10}|{'Arc Angle':<12}|{'IMU Yaw':<10}|{'Rc (m)':<8}|{'S_act (m)'}")
log("-" * 95)

summary_rows = []
for r in results:
    log(f"R{r['robot_id']:<6}| {r['r_eff_mm']:<10.3f} | {r['scale_factor']:<6.4f} | {r['Ed_ground']:<11.5f} | {r['Ed_encoder']:<8.5f} | {r['theta_arc_deg']:>7.2f}°    | {r['imu_yaw_deg']:>6.2f}°   | {r['Rc']:<6.3f} | {r['S_actual']:<6.3f}")
    summary_rows.append({
        "robot_id": r["robot_id"],
        "r_eff_mm": round(r["r_eff_mm"], 4),
        "scale_factor": round(r["scale_factor"], 5),
        "Ed_ground": round(r["Ed_ground"], 5),
        "Ed_encoder": round(r["Ed_encoder"], 5),
        "radius_curvature_m": round(r["Rc"], 4),
        "arc_angle_deg": round(r["theta_arc_deg"], 2),
        "imu_yaw_deg": round(r["imu_yaw_deg"], 2),
        "actual_arc_m": round(r["S_actual"], 4),
        "rmse_rpm_l": round(r["rmse_rpm_l"], 3),
        "rmse_rpm_r": round(r["rmse_rpm_r"], 3),
        "mean_rpm_l": round(r["mean_rpm_l"], 2),
        "mean_rpm_r": round(r["mean_rpm_r"], 2),
        "vbus_drop_v": round(r["vbus_drop"], 3),
        "mean_current_ma": round(r["mean_curr"], 1)
    })

log("-" * 95)
log("\n" + "=" * 95)
log("   STEP 1: CLOSED-LOOP WHEEL VELOCITY TRACKING AUDIT (Target: 57.30 RPM)")
log("=" * 95)
log(f"{'Robot':<7}|{'Mean RPM (L)':<14}|{'Mean RPM (R)':<14}|{'RMSE L (RPM)':<14}|{'RMSE R (RPM)':<14}|{'Vbus Drop':<12}|{'Mean Current'}")
log("-" * 95)
for r in results:
    log(f"R{r['robot_id']:<6}| {r['mean_rpm_l']:<12.2f} | {r['mean_rpm_r']:<12.2f} | {r['rmse_rpm_l']:<12.3f} | {r['rmse_rpm_r']:<12.3f} | {r['vbus_drop']:<10.3f}V | {r['mean_curr']:<6.1f} mA")
log("=" * 95)

# Save text report and CSV
with open(os.path.join(OUTPUT_DIR, "kinematics_report.txt"), "w") as f_out:
    f_out.write("\n".join(report_lines) + "\n")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "kinematics_summary.csv"), index=False)
log(f"\n[INFO] Text report saved to: {OUTPUT_DIR}/kinematics_report.txt")
log(f"[INFO] Summary CSV saved to: {OUTPUT_DIR}/kinematics_summary.csv")

# ==============================================================================
# 5. VISUALIZATION ENGINE (4 MULTI-PANEL FIGURES)
# ==============================================================================
colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}

# --- FIGURE 1: 2D GROUND TRAJECTORY RECONSTRUCTION ---
plt.figure(figsize=(9, 9))
plt.axvline(0, color="gray", linestyle="--", linewidth=1, label="Nominal Track (X=0)")
plt.axhline(2.0, color="darkred", linestyle=":", linewidth=1.5, label="Nominal Target (Y=2.0m)")

for r in results:
    rid = r["robot_id"]
    xf, yf = r["xf"], r["yf"]
    xc = (xf**2 + yf**2) / (2.0 * xf)
    Rc = abs(xc)
    theta_total = 2.0 * math.atan2(abs(xf), yf)

    # Parametric circle arc from (0,0) to (xf, yf)
    angles = np.linspace(0, theta_total, 200)
    if xf < 0:
        # Center is at (-Rc, 0), starts at (0,0) moving in +y
        arc_x = xc + Rc * np.cos(angles)
        arc_y = Rc * np.sin(angles)
    else:
        # Center is at (+Rc, 0), starts at (0,0) moving in +y
        arc_x = xc - Rc * np.cos(angles)
        arc_y = Rc * np.sin(angles)

    plt.plot(arc_x, arc_y, color=colors[rid], linewidth=2.5, label=f"R{rid} (Rc={Rc:.2f}m, Ed={r['Ed_ground']:.4f})")
    plt.scatter([xf], [yf], color=colors[rid], s=90, zorder=5)
    plt.annotate(f"R{rid}\n({xf:+.2f}, {yf:+.2f})", (xf, yf), textcoords="offset points", xytext=(10, -5), fontweight="bold")

plt.scatter([0], [0], color="black", s=100, zorder=6, label="Start (0,0)")
plt.title("Reconstructed 2D Ground Trajectories (2-Meter Straight Benchmark)", fontsize=13, pad=12)
plt.xlabel("Lateral Deviation X (meters) [Negative = Left | Positive = Right]", fontsize=11)
plt.ylabel("Forward Travel Y (meters)", fontsize=11)
plt.xlim(-1.2, 1.8)
plt.ylim(-0.1, 2.3)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_2d_ground_trajectories.png"), dpi=300)
plt.close()

# --- FIGURE 2: VELOCITY TRACKING PERFORMANCE (RPM vs TIME) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, r in enumerate(results):
    ax = axes[idx]
    rid = r["robot_id"]
    cdf = r["cruise_df"]
    t = (cdf["elapsed_ms"] - cdf["elapsed_ms"].iloc[0]) / 1000.0

    ax.axhline(r["target_rpm"], color="black", linestyle="--", linewidth=1.5, label="Target (57.30 RPM)")
    ax.plot(t, cdf["meas_rpm_l"], color="#1f77b4", linewidth=1.5, label=f"Left (Mean={r['mean_rpm_l']:.1f})")
    ax.plot(t, cdf["meas_rpm_r"], color="#d62728", linewidth=1.5, label=f"Right (Mean={r['mean_rpm_r']:.1f})")

    ax.set_title(f"Robot {rid} - Speed Tracking (RMSE: L={r['rmse_rpm_l']:.2f}, R={r['rmse_rpm_r']:.2f})", fontsize=11)
    ax.set_ylabel("Wheel Speed (RPM)")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower right")

axes[2].set_xlabel("Time (s)")
axes[3].set_xlabel("Time (s)")
plt.suptitle("Wheel Velocity Closed-Loop Step Response (100 Hz)", fontsize=14, y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_velocity_tracking_comparison.png"), dpi=300)
plt.close()

# --- FIGURE 3: GYRO HEADING DRIFT VS TIME ---
plt.figure(figsize=(10, 6))
for r in results:
    rid = r["robot_id"]
    df = r["df"]
    t = (df["elapsed_ms"] - df["elapsed_ms"].iloc[0]) / 1000.0
    plt.plot(t, df["yaw_deg"], color=colors[rid], linewidth=2, label=f"R{rid} IMU (Final={r['imu_yaw_deg']:.1f}°, Arc={r['theta_arc_deg']:.1f}°)")

plt.title("Integrated Heading Angle (IMU Gyro Z) During 2m Run", fontsize=13)
plt.xlabel("Elapsed Time (s)", fontsize=11)
plt.ylabel("Integrated Yaw (degrees)", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_heading_drift_comparison.png"), dpi=300)
plt.close()

# --- FIGURE 4: BATTERY VOLTAGE & CURRENT DYNAMICS ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for r in results:
    rid = r["robot_id"]
    df = r["df"]
    t = (df["elapsed_ms"] - df["elapsed_ms"].iloc[0]) / 1000.0
    ax1.plot(t, df["vbus_v"], color=colors[rid], linewidth=1.8, label=f"R{rid} (Drop={r['vbus_drop']:.2f}V)")
    ax2.plot(t, df["current_ma"], color=colors[rid], linewidth=1.5, label=f"R{rid} (Mean={r['mean_curr']:.0f}mA)")

ax1.set_title("Dynamic Bus Voltage (INA226)", fontsize=12)
ax1.set_ylabel("Voltage (V)")
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.legend(loc="upper right")

ax2.set_title("Total Drivetrain Current Draw (INA226)", fontsize=12)
ax2.set_xlabel("Elapsed Time (s)")
ax2.set_ylabel("Current (mA)")
ax2.grid(True, linestyle=":", alpha=0.7)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_power_and_voltage_dynamics.png"), dpi=300)
plt.close()

log(f"\n[SUCCESS] Generated 4 analysis plots in '{OUTPUT_DIR}/'.")
