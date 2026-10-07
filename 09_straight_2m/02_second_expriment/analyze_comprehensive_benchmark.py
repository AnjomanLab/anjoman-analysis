#!/usr/bin/env python3
"""
Anjoman Swarm Firmware - Autonomous Control & Kinematics Analyzer
Purely data-driven metrology engine without visual observation bias.
Works for single runs or comparative benchmarks.
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. HARDWARE CONSTANTS PER ROBOT
# ==============================================================================
TRACK_WIDTHS = {
    1: 0.1350,  # Robot 1: 135 mm
    2: 0.1250,  # Robot 2: 125 mm
    3: 0.1250,  # Robot 3: 125 mm
    4: 0.1250   # Robot 4: 125 mm
}

CPR = 4096.0
OUTPUT_DIR = "analysis_results_comprehensive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. DATA-DRIVEN ANALYSIS ENGINE
# ==============================================================================
def process_robot_data(filepath):
    # Extract Robot ID from filename (e.g. straight_2m_r1.csv)
    try:
        base = os.path.basename(filepath)
        rid = int(base.split("straight_2m_r")[1].split(".csv")[0])
    except Exception:
        return None

    W = TRACK_WIDTHS.get(rid, 0.1250)

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Time in seconds
    t_raw = df["elapsed_ms"].values / 1000.0
    t0 = t_raw[0]
    t = t_raw - t0
    dt = np.diff(t, prepend=0.010)

    # Separate Cruise Phase (state == 1)
    cruise_mask = (df["state"] == 1).values
    if not np.any(cruise_mask):
        cruise_mask = np.ones(len(df), dtype=bool)

    t_cruise = t[cruise_mask]
    t_cruise_rel = t_cruise - t_cruise[0]

    # --- A. CONTROL LOOP METRICS (STEP 1) ---
    # Robust extraction of non-zero target RPM during cruise phase
    target_arr = df["target_rpm_l"].values[cruise_mask] if "target_rpm_l" in df.columns else np.array([])
    valid_targets = target_arr[target_arr > 1.0]
    target_rpm = float(np.median(valid_targets)) if len(valid_targets) > 0 else 57.30
    rpm_l = df["meas_rpm_l"].values[cruise_mask]
    rpm_r = df["meas_rpm_r"].values[cruise_mask]
    duty_l = df["pwm_duty_l"].values[cruise_mask]
    duty_r = df["pwm_duty_r"].values[cruise_mask]

    # Rise time (time to reach 90% of target RPM)
    def calc_rise_time(rpm_arr, t_arr, target):
        idx = np.where(rpm_arr >= 0.90 * target)[0]
        return t_arr[idx[0]] if len(idx) > 0 else np.nan

    rise_time_l = calc_rise_time(rpm_l, t_cruise_rel, target_rpm)
    rise_time_r = calc_rise_time(rpm_r, t_cruise_rel, target_rpm)

    # Steady-state evaluation (skip first 1.2 seconds of acceleration)
    ss_mask = t_cruise_rel > 1.2
    if not np.any(ss_mask):
        ss_mask = np.ones(len(t_cruise_rel), dtype=bool)

    ss_rpm_l = rpm_l[ss_mask]
    ss_rpm_r = rpm_r[ss_mask]

    mean_rpm_l = np.mean(ss_rpm_l)
    mean_rpm_r = np.mean(ss_rpm_r)
    std_rpm_l  = np.std(ss_rpm_l)
    std_rpm_r  = np.std(ss_rpm_r)

    rmse_rpm_l = np.sqrt(np.mean((ss_rpm_l - target_rpm)**2))
    rmse_rpm_r = np.sqrt(np.mean((ss_rpm_r - target_rpm)**2))
    mae_rpm_l  = np.mean(np.abs(ss_rpm_l - target_rpm))
    mae_rpm_r  = np.mean(np.abs(ss_rpm_r - target_rpm))

    if target_rpm > 1.0:
        overshoot_l = max(0.0, float((np.max(rpm_l) - target_rpm) / target_rpm * 100.0))
        overshoot_r = max(0.0, float((np.max(rpm_r) - target_rpm) / target_rpm * 100.0))
    else:
        overshoot_l = 0.0
        overshoot_r = 0.0

    # --- B. TICK SYNCHRONIZATION & ASYMMETRY ---
    steps_l = df["steps_l"].values
    steps_r = df["steps_r"].values
    tick_diff = steps_l - steps_r

    max_tick_diff = np.max(np.abs(tick_diff[cruise_mask]))
    final_tick_diff = tick_diff[-1]
    ed_encoder = (steps_r[-1] / steps_l[-1]) if steps_l[-1] != 0 else 1.0

    # Effort difference
    duty_diff = np.mean(duty_l[ss_mask] - duty_r[ss_mask])

    # --- C. PURE DEAD-RECKONING (WHEEL ODOMETRY) ---
    # Convert step increments to displacements
    d_steps_l = np.diff(steps_l, prepend=0)
    d_steps_r = np.diff(steps_r, prepend=0)

    # Distance per tick derived from reported dist_m if available, else standard CPR
    total_dist_l = df["dist_m_l"].iloc[-1]
    total_dist_r = df["dist_m_r"].iloc[-1]
    r_wheel_calc = (total_dist_l / steps_l[-1]) * (CPR / (2.0 * math.pi)) if steps_l[-1] > 0 else 0.02685

    ds_l = (d_steps_l / CPR) * (2.0 * math.pi * r_wheel_calc)
    ds_r = (d_steps_r / CPR) * (2.0 * math.pi * r_wheel_calc)
    ds_avg = (ds_l + ds_r) / 2.0

    # Standard Diff-Drive Equations (Y = forward, X = lateral right)
    # Heading theta: 0 = +Y forward, positive = clockwise (right), negative = CCW (left)
    d_theta_wheel = (ds_l - ds_r) / W
    theta_wheel = np.cumsum(d_theta_wheel)

    x_wheel = np.zeros(len(df))
    y_wheel = np.zeros(len(df))
    for k in range(1, len(df)):
        th_mid = theta_wheel[k-1] + d_theta_wheel[k] / 2.0
        x_wheel[k] = x_wheel[k-1] + ds_avg[k] * np.sin(th_mid)
        y_wheel[k] = y_wheel[k-1] + ds_avg[k] * np.cos(th_mid)

    # --- D. GYRO-AIDED DEAD-RECKONING ---
    # IMU yaw: Positive is CCW (left), Negative is CW (right).
    # To match X = right, theta_gyro_cw = -yaw_deg (in radians)
    raw_yaw_deg = df["yaw_deg"].values
    theta_gyro = -np.radians(raw_yaw_deg)

    x_gyro = np.zeros(len(df))
    y_gyro = np.zeros(len(df))
    for k in range(1, len(df)):
        th_mid = (theta_gyro[k-1] + theta_gyro[k]) / 2.0
        x_gyro[k] = x_gyro[k-1] + ds_avg[k] * np.sin(th_mid)
        y_gyro[k] = y_gyro[k-1] + ds_avg[k] * np.cos(th_mid)

    # Path Tortuosity (Straightness Ratio: Euclidean displacement / Path Length)
    euclid_dist = math.sqrt(x_gyro[-1]**2 + y_gyro[-1]**2)
    path_length = np.sum(ds_avg)
    straightness_ratio = (euclid_dist / path_length) if path_length > 0 else 1.0

    # --- E. ELECTRICAL & POWER DYNAMICS ---
    vbus = df["vbus_v"].values
    current_ma = df["current_ma"].values
    v_init = vbus[0]
    v_min = np.min(vbus[cruise_mask])
    v_drop = v_init - v_min
    mean_curr = np.mean(current_ma[cruise_mask])
    peak_curr = np.max(current_ma[cruise_mask])

    # Total energy consumed (Joules) = sum(V * I * dt)
    power_watts = vbus * (current_ma / 1000.0)
    energy_joules = np.sum(power_watts * dt)

    return {
        "id": rid,
        "filepath": filepath,
        "df": df,
        "t": t,
        "t_cruise": t_cruise_rel,
        "cruise_mask": cruise_mask,
        "ss_mask": ss_mask,
        "target_rpm": target_rpm,
        "rpm_l": rpm_l,
        "rpm_r": rpm_r,
        "duty_l": duty_l,
        "duty_r": duty_r,
        "mean_rpm_l": mean_rpm_l,
        "mean_rpm_r": mean_rpm_r,
        "std_rpm_l": std_rpm_l,
        "std_rpm_r": std_rpm_r,
        "rmse_rpm_l": rmse_rpm_l,
        "rmse_rpm_r": rmse_rpm_r,
        "mae_rpm_l": mae_rpm_l,
        "mae_rpm_r": mae_rpm_r,
        "overshoot_l": overshoot_l,
        "overshoot_r": overshoot_r,
        "rise_time_l": rise_time_l,
        "rise_time_r": rise_time_r,
        "steps_l": steps_l[-1],
        "steps_r": steps_r[-1],
        "tick_diff": tick_diff,
        "max_tick_diff": max_tick_diff,
        "final_tick_diff": final_tick_diff,
        "ed_encoder": ed_encoder,
        "duty_diff": duty_diff,
        "x_wheel": x_wheel,
        "y_wheel": y_wheel,
        "x_gyro": x_gyro,
        "y_gyro": y_gyro,
        "final_x_gyro": x_gyro[-1],
        "final_y_gyro": y_gyro[-1],
        "final_yaw_deg": raw_yaw_deg[-1],
        "final_wheel_yaw_deg": np.degrees(theta_wheel[-1]),
        "path_length": path_length,
        "straightness_ratio": straightness_ratio,
        "v_init": v_init,
        "v_drop": v_drop,
        "mean_curr": mean_curr,
        "peak_curr": peak_curr,
        "energy_joules": energy_joules,
        "r_wheel_mm": r_wheel_calc * 1000.0
    }

# ==============================================================================
# 3. RUN BATCH PROCESSING
# ==============================================================================
files = sorted(glob.glob("straight_2m_r*.csv"))
if not files:
    print("[ERROR] No files matching 'straight_2m_r*.csv' found in working directory!")
    exit(1)

results = []
for f in files:
    res = process_robot_data(f)
    if res is not None:
        results.append(res)

results.sort(key=lambda x: x["id"])

# ==============================================================================
# 4. PRINT REPORT & SAVE TO TEXT AND CSV
# ==============================================================================
lines = []
def p(msg=""):
    print(msg)
    lines.append(msg)

p("=" * 115)
p("   ANJOMAN COMPREHENSIVE KINEMATICS & CONTROL PERFORMANCE REPORT (PURE DATA-DRIVEN)")
p("=" * 115)
p(f"{'Robot':<7}|{'Final X (m)':<12}|{'Final Y (m)':<12}|{'Path Len (m)':<13}|{'Straightness':<14}|{'Gyro Yaw':<12}|{'Wheel Yaw':<12}|{'Ed (Ticks)'}")
p("-" * 115)

for r in results:
    p(f"R{r['id']:<6}| {r['final_x_gyro']:>9.4f} m | {r['final_y_gyro']:>9.4f} m | {r['path_length']:>10.4f} m | {r['straightness_ratio']:>10.5f}   | {r['final_yaw_deg']:>8.2f}°   | {r['final_wheel_yaw_deg']:>8.2f}°   | {r['ed_encoder']:>9.5f}")

p("-" * 115)
p("\n" + "=" * 115)
p("   STEP 1: WHEEL VELOCITY CONTROL & TICK SYNCHRONIZATION AUDIT (Target: 57.30 RPM)")
p("=" * 115)
p(f"{'Robot':<7}|{'Mean L':<9}|{'Mean R':<9}|{'RMSE L':<9}|{'RMSE R':<9}|{'Jitter σ_L':<11}|{'Jitter σ_R':<11}|{'Max ΔTicks':<12}|{'End ΔTicks':<11}|{'ΔPWM (L-R)'}")
p("-" * 115)

for r in results:
    p(f"R{r['id']:<6}| {r['mean_rpm_l']:<7.2f} | {r['mean_rpm_r']:<7.2f} | {r['rmse_rpm_l']:<7.3f} | {r['rmse_rpm_r']:<7.3f} | {r['std_rpm_l']:<9.3f} | {r['std_rpm_r']:<9.3f} | {r['max_tick_diff']:<10} | {r['final_tick_diff']:<9} | {r['duty_diff']:>+7.4f}")

p("-" * 115)
p("\n" + "=" * 115)
p("   ELECTRICAL & POWER METROLOGY (INA226 DYNAMIC LOG)")
p("=" * 115)
p(f"{'Robot':<7}|{'V_init (V)':<12}|{'V_drop (V)':<12}|{'Mean Current':<15}|{'Peak Current':<15}|{'Total Energy (Joules)'}")
p("-" * 115)

for r in results:
    p(f"R{r['id']:<6}| {r['v_init']:<10.2f} | {r['v_drop']:<10.3f} | {r['mean_curr']:<11.1f} mA | {r['peak_curr']:<11.1f} mA | {r['energy_joules']:<10.2f} J")
p("=" * 115)

# Save Text and CSV Reports
with open(os.path.join(OUTPUT_DIR, "kinematics_report.txt"), "w") as f_out:
    f_out.write("\n".join(lines) + "\n")

summary_rows = []
for r in results:
    summary_rows.append({
        "robot_id": r["id"],
        "final_x_gyro_m": round(r["final_x_gyro"], 4),
        "final_y_gyro_m": round(r["final_y_gyro"], 4),
        "path_length_m": round(r["path_length"], 4),
        "straightness_ratio": round(r["straightness_ratio"], 5),
        "gyro_yaw_deg": round(r["final_yaw_deg"], 2),
        "wheel_yaw_deg": round(r["final_wheel_yaw_deg"], 2),
        "ed_ticks": round(r["ed_encoder"], 5),
        "mean_rpm_l": round(r["mean_rpm_l"], 2),
        "mean_rpm_r": round(r["mean_rpm_r"], 2),
        "rmse_rpm_l": round(r["rmse_rpm_l"], 3),
        "rmse_rpm_r": round(r["rmse_rpm_r"], 3),
        "std_rpm_l": round(r["std_rpm_l"], 3),
        "std_rpm_r": round(r["std_rpm_r"], 3),
        "max_tick_diff": r["max_tick_diff"],
        "final_tick_diff": r["final_tick_diff"],
        "duty_diff": round(r["duty_diff"], 4),
        "v_drop_v": round(r["v_drop"], 3),
        "mean_curr_ma": round(r["mean_curr"], 1),
        "energy_joules": round(r["energy_joules"], 2)
    })

pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "kinematics_summary.csv"), index=False)
p(f"\n[INFO] Complete textual report saved to: {OUTPUT_DIR}/kinematics_report.txt")
p(f"[INFO] Summary CSV saved to: {OUTPUT_DIR}/kinematics_summary.csv")

# ==============================================================================
# 5. ULTRA-WIDE HIGH-RESOLUTION VISUALIZATIONS (18x10 INCHES @ 300 DPI)
# ==============================================================================
colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}

# --- FIGURE 1: 2D TRAJECTORY RECONSTRUCTION (WHEEL vs GYRO) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), sharey=True)

# Left: Wheel Odometry
ax1.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Nominal Track (X=0)")
ax1.axhline(2.0, color="darkred", linestyle=":", linewidth=1.5, label="Target (Y=2.0m)")
for r in results:
    rid = r["id"]
    ax1.plot(r["x_wheel"], r["y_wheel"], color=colors[rid], linewidth=2.5, label=f"R{rid} Wheel Odo (End: X={r['x_wheel'][-1]:+.2f}m)")
    ax1.scatter(r["x_wheel"][-1], r["y_wheel"][-1], color=colors[rid], s=120, zorder=5)

ax1.set_title("Pure Wheel Differential Odometry", fontsize=14, pad=10)
ax1.set_xlabel("Lateral Deviation X (m) [Left < 0 | Right > 0]", fontsize=12)
ax1.set_ylabel("Forward Travel Y (m)", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.legend(loc="upper left", fontsize=10)
ax1.set_xlim(-1.5, 1.5)

# Right: Gyro-Enhanced Odometry
ax2.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Nominal Track (X=0)")
ax2.axhline(2.0, color="darkred", linestyle=":", linewidth=1.5, label="Target (Y=2.0m)")
for r in results:
    rid = r["id"]
    ax2.plot(r["x_gyro"], r["y_gyro"], color=colors[rid], linewidth=2.5, label=f"R{rid} Gyro-Aided (End: X={r['final_x_gyro']:+.2f}m, Yaw={r['final_yaw_deg']:+.1f}°)")
    ax2.scatter(r["final_x_gyro"], r["final_y_gyro"], color=colors[rid], s=120, zorder=5)

ax2.set_title("Gyro-Aided Dead Reckoning (IMU Z-Axis Fusion)", fontsize=14, pad=10)
ax2.set_xlabel("Lateral Deviation X (m) [Left < 0 | Right > 0]", fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.7)
ax2.legend(loc="upper left", fontsize=10)
ax2.set_xlim(-1.5, 1.5)

plt.suptitle("Comparative 2D Ground Trajectory Reconstruction", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_2d_trajectory_reconstruction.png"), dpi=300)
plt.close()

# --- FIGURE 2: HIGH-RESOLUTION VELOCITY TRACKING & JITTER ---
fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, r in enumerate(results):
    ax = axes[idx]
    rid = r["id"]
    t = r["t"]

    ax.axhline(r["target_rpm"], color="black", linestyle="--", linewidth=1.5, label="Target (57.30 RPM)")
    ax.plot(t, r["df"]["meas_rpm_l"], color="#1f77b4", linewidth=1.2, alpha=0.9, label=f"Left (Mean={r['mean_rpm_l']:.1f}, σ={r['std_rpm_l']:.2f})")
    ax.plot(t, r["df"]["meas_rpm_r"], color="#d62728", linewidth=1.2, alpha=0.9, label=f"Right (Mean={r['mean_rpm_r']:.1f}, σ={r['std_rpm_r']:.2f})")

    ax.set_title(f"Robot {rid} - 100 Hz Wheel Velocity (RMSE: L={r['rmse_rpm_l']:.2f}, R={r['rmse_rpm_r']:.2f})", fontsize=12)
    ax.set_ylabel("Speed (RPM)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower right", fontsize=10)

axes[2].set_xlabel("Time (s)", fontsize=12)
axes[3].set_xlabel("Time (s)", fontsize=12)
plt.suptitle("Wheel Velocity Closed-Loop Dynamics & Jitter Analysis", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_velocity_tracking_high_res.png"), dpi=300)
plt.close()

# --- FIGURE 3: TICK SYNCHRONIZATION ERROR (ΔTICKS) & HEADING DRIFT ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

for r in results:
    rid = r["id"]
    t = r["t"]
    ax1.plot(t, r["tick_diff"], color=colors[rid], linewidth=2.0, label=f"R{rid} (Max |Δ|={r['max_tick_diff']}, Final={r['final_tick_diff']})")
    ax2.plot(t, r["df"]["yaw_deg"], color=colors[rid], linewidth=2.0, label=f"R{rid} (Final Yaw={r['final_yaw_deg']:.2f}°)")

ax1.axhline(0, color="black", linestyle="--", linewidth=1.0)
ax1.set_title("Cross-Coupled Synchronization Error: (Steps_Left - Steps_Right)", fontsize=13)
ax1.set_ylabel("Tick Difference (Ticks)", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.legend(loc="upper left", fontsize=10)

ax2.axhline(0, color="black", linestyle="--", linewidth=1.0)
ax2.set_title("Integrated Heading Angle (IMU Gyro Z)", fontsize=13)
ax2.set_xlabel("Elapsed Time (s)", fontsize=12)
ax2.set_ylabel("Yaw Angle (degrees)", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.7)
ax2.legend(loc="upper left", fontsize=10)

plt.suptitle("Micro-Dynamics: Synchronization Tightness & Directional Stability", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_tick_sync_and_heading.png"), dpi=300)
plt.close()

# --- FIGURE 4: ACTUATOR EFFORT (PWM DUTY) & ELECTRICAL POWER ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

for r in results:
    rid = r["id"]
    t = r["t"]
    # Mean PWM effort of both motors
    avg_duty = (r["df"]["pwm_duty_l"] + r["df"]["pwm_duty_r"]) / 2.0
    ax1.plot(t, avg_duty, color=colors[rid], linewidth=1.8, label=f"R{rid} Mean Duty")
    ax2.plot(t, r["df"]["vbus_v"], color=colors[rid], linewidth=1.8, label=f"R{rid} Vbus (Drop={r['v_drop']:.2f}V)")

ax1.set_title("Actuator Control Effort (Average PWM Duty Cycle)", fontsize=13)
ax1.set_ylabel("Duty Cycle [-1.0, 1.0]", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.legend(loc="lower right", fontsize=10)

ax2.set_title("Battery Rail Voltage Dynamics (INA226 Vbus)", fontsize=13)
ax2.set_xlabel("Elapsed Time (s)", fontsize=12)
ax2.set_ylabel("Voltage (V)", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.7)
ax2.legend(loc="lower right", fontsize=10)

plt.suptitle("Power Plant & Actuator Effort Profiles", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_actuator_effort_and_voltage.png"), dpi=300)
plt.close()

p(f"\n[SUCCESS] Generated 4 ultra-wide plots in '{OUTPUT_DIR}/'.")
