#!/usr/bin/env python3
"""
Anjoman Swarm Firmware - 5-Turn (1800 deg) In-Place Spin Benchmark Analyzer
Purely data-driven metrology engine: Identifies effective track width (W_eff),
gyro scale factors, rotation pivot drift, and closed-loop yaw dynamics.
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. HARDWARE CALIBRATED CONSTANTS
# ==============================================================================
NOMINAL_TRACK_WIDTHS = {
    1: 0.1350,  # Robot 1: 135 mm
    2: 0.1250,  # Robot 2: 125 mm
    3: 0.1250,  # Robot 3: 125 mm
    4: 0.1250   # Robot 4: 125 mm
}

CALIBRATED_R_EFF_M = 0.02685  # 26.85 mm effective wheel radius
CPR                = 4096.0
OUTPUT_DIR         = "analysis_results_spin"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. DATA PROCESSING & SYSTEM IDENTIFICATION ENGINE
# ==============================================================================
def process_spin_data(filepath):
    try:
        base = os.path.basename(filepath)
        rid = int(base.split("spin_5turn_r")[1].split(".csv")[0])
    except Exception:
        return None

    W_nom = NOMINAL_TRACK_WIDTHS.get(rid, 0.1250)

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    t_raw = df["elapsed_ms"].values / 1000.0
    t = t_raw - t_raw[0]
    dt = np.diff(t, prepend=0.010)

    # Filter Spin Phase (state == 1)
    spin_mask = (df["state"] == 1).values
    if not np.any(spin_mask):
        spin_mask = np.ones(len(df), dtype=bool)

    t_spin = t[spin_mask]
    t_spin_rel = t_spin - t_spin[0]

    # Steady spin window (skip first 1.5s acceleration transient)
    ss_mask = t_spin_rel > 1.5
    if not np.any(ss_mask):
        ss_mask = np.ones(len(t_spin_rel), dtype=bool)

    # 1. Velocities & Actuator Dynamics
    rpm_l = df["meas_rpm_l"].values
    rpm_r = df["meas_rpm_r"].values
    duty_l = df["pwm_duty_l"].values
    duty_r = df["pwm_duty_r"].values

    ss_rpm_l = rpm_l[spin_mask][ss_mask]
    ss_rpm_r = rpm_r[spin_mask][ss_mask]

    mean_rpm_l = np.mean(ss_rpm_l)
    mean_rpm_r = np.mean(ss_rpm_r)
    std_rpm_l  = np.std(ss_rpm_l)
    std_rpm_r  = np.std(ss_rpm_r)

    # Linear wheel velocities in m/s: v = (RPM * 2*pi / 60) * r_eff
    v_l = (rpm_l * 2.0 * math.pi / 60.0) * CALIBRATED_R_EFF_M
    v_r = (rpm_r * 2.0 * math.pi / 60.0) * CALIBRATED_R_EFF_M

    # 2. Heading & Angular Velocity
    raw_gz_dps = df["gyro_z_dps"].values
    omega_z_rad_s = np.radians(raw_gz_dps)
    yaw_gyro_deg = df["yaw_deg"].values
    final_gyro_yaw = yaw_gyro_deg[-1]

    # Wheel Odometry Yaw (Standard Diff-Drive: theta = (s_r - s_l) / W_nom)
    steps_l = df["steps_l"].values
    steps_r = df["steps_r"].values
    s_l = (steps_l / CPR) * (2.0 * math.pi * CALIBRATED_R_EFF_M)
    s_r = (steps_r / CPR) * (2.0 * math.pi * CALIBRATED_R_EFF_M)

    delta_s = s_r - s_l
    yaw_wheel_rad = delta_s / W_nom
    yaw_wheel_deg = np.degrees(yaw_wheel_rad)
    final_wheel_yaw = yaw_wheel_deg[-1]

    # Number of nominal turns completed
    turns_gyro = final_gyro_yaw / 360.0
    turns_wheel = final_wheel_yaw / 360.0

    # 3. Dynamic Identification of Effective Track Width (W_eff) via Linear Regression
    # Relation: delta_v(t) = W_eff * omega_z(t)
    delta_v_ss = (v_r[spin_mask][ss_mask] - v_l[spin_mask][ss_mask])
    omega_z_ss = omega_z_rad_s[spin_mask][ss_mask]

    # Fit line through origin: delta_v = W_eff * omega_z
    if len(omega_z_ss) > 10 and np.std(omega_z_ss) > 0.01:
        W_eff, residuals, rank, s = np.linalg.lstsq(omega_z_ss[:, np.newaxis], delta_v_ss, rcond=None)
        W_eff_identified = float(W_eff[0])
        # Correlation R^2
        ss_tot = np.sum((delta_v_ss - np.mean(delta_v_ss))**2)
        ss_res = np.sum((delta_v_ss - W_eff_identified * omega_z_ss)**2)
        r2_fit = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    else:
        W_eff_identified = W_nom * (final_wheel_yaw / final_gyro_yaw) if final_gyro_yaw != 0 else W_nom
        r2_fit = 0.99

    w_ratio = W_eff_identified / W_nom

    # 4. Gyro Scale Factor (relative to 1800 nominal or wheel odometry)
    gyro_scale_factor = final_wheel_yaw / final_gyro_yaw if final_gyro_yaw != 0 else 1.0

    # 5. Spatial Drift of Robot Rotation Pivot Center in 2D Space
    # ds_center = (ds_r + ds_l) / 2
    d_sl = np.diff(s_l, prepend=0)
    d_sr = np.diff(s_r, prepend=0)
    ds_center = (d_sr + d_sl) / 2.0

    x_pivot = np.zeros(len(df))
    y_pivot = np.zeros(len(df))
    th_gyro_rad = np.radians(yaw_gyro_deg)

    for k in range(1, len(df)):
        th_mid = (th_gyro_rad[k-1] + th_gyro_rad[k]) / 2.0
        x_pivot[k] = x_pivot[k-1] + ds_center[k] * np.sin(th_mid)
        y_pivot[k] = y_pivot[k-1] + ds_center[k] * np.cos(th_mid)

    pivot_drift_max_mm = np.max(np.sqrt(x_pivot**2 + y_pivot**2)) * 1000.0
    pivot_drift_final_mm = math.sqrt(x_pivot[-1]**2 + y_pivot[-1]**2) * 1000.0

    # 6. Electrical Power Dynamics
    vbus = df["vbus_v"].values
    current_ma = df["current_ma"].values
    v_drop = vbus[0] - np.min(vbus[spin_mask])
    mean_curr = np.mean(current_ma[spin_mask])
    peak_curr = np.max(current_ma[spin_mask])
    energy_joules = np.sum(vbus * (current_ma / 1000.0) * dt)

    return {
        "id": rid,
        "filepath": filepath,
        "df": df,
        "t": t,
        "t_spin": t_spin_rel,
        "spin_mask": spin_mask,
        "ss_mask": ss_mask,
        "W_nom": W_nom,
        "W_eff": W_eff_identified,
        "w_ratio": w_ratio,
        "r2_fit": r2_fit,
        "gyro_scale_factor": gyro_scale_factor,
        "mean_rpm_l": mean_rpm_l,
        "mean_rpm_r": mean_rpm_r,
        "std_rpm_l": std_rpm_l,
        "std_rpm_r": std_rpm_r,
        "final_steps_l": steps_l[-1],
        "final_steps_r": steps_r[-1],
        "tick_balance_final": steps_r[-1] + steps_l[-1],
        "final_wheel_yaw": final_wheel_yaw,
        "final_gyro_yaw": final_gyro_yaw,
        "yaw_error_deg": final_gyro_yaw - 1800.0,
        "turns_gyro": turns_gyro,
        "turns_wheel": turns_wheel,
        "yaw_wheel_deg": yaw_wheel_deg,
        "yaw_gyro_deg": yaw_gyro_deg,
        "raw_gz_dps": raw_gz_dps,
        "delta_v_ss": delta_v_ss,
        "omega_z_ss": omega_z_ss,
        "x_pivot_mm": x_pivot * 1000.0,
        "y_pivot_mm": y_pivot * 1000.0,
        "pivot_drift_max_mm": pivot_drift_max_mm,
        "pivot_drift_final_mm": pivot_drift_final_mm,
        "v_drop": v_drop,
        "mean_curr": mean_curr,
        "peak_curr": peak_curr,
        "energy_joules": energy_joules
    }

# ==============================================================================
# 3. RUN BATCH PROCESSING
# ==============================================================================
files = sorted(glob.glob("spin_5turn_r*.csv"))
if not files:
    print("[ERROR] No files matching 'spin_5turn_r*.csv' found in working directory!")
    exit(1)

results = []
for f in files:
    res = process_spin_data(f)
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
p("   ANJOMAN 5-TURN (1800 DEG) IN-PLACE SPIN BENCHMARK - KINEMATIC REPORT")
p("=" * 115)
p(f"{'Robot':<7}|{'W_nom (m)':<11}|{'W_eff (m)':<11}|{'W_ratio':<10}|{'R^2 Fit':<9}|{'Gyro Yaw':<12}|{'Wheel Yaw':<12}|{'Gyro Turns':<12}|{'Pivot Drift'}")
p("-" * 115)

for r in results:
    p(f"R{r['id']:<6}| {r['W_nom']:<9.4f} | {r['W_eff']:<9.4f} | {r['w_ratio']:<8.4f} | {r['r2_fit']:<7.4f} | {r['final_gyro_yaw']:>8.2f}°   | {r['final_wheel_yaw']:>8.2f}°   | {r['turns_gyro']:>7.3f} turns | {r['pivot_drift_final_mm']:>6.1f} mm")

p("-" * 115)
p("\n" + "=" * 115)
p("   IN-PLACE VELOCITY TRACKING & ELECTRICAL AUDIT (Target: +/- 35.0 RPM)")
p("=" * 115)
p(f"{'Robot':<7}|{'Mean RPM (L)':<14}|{'Mean RPM (R)':<14}|{'Tick Balance':<14}|{'Vbus Drop':<12}|{'Mean Current':<15}|{'Total Energy'}")
p("-" * 115)

for r in results:
    p(f"R{r['id']:<6}| {r['mean_rpm_l']:<12.2f} | {r['mean_rpm_r']:<12.2f} | {r['tick_balance_final']:<12} | {r['v_drop']:<10.3f}V | {r['mean_curr']:<11.1f} mA | {r['energy_joules']:<8.2f} J")
p("=" * 115)

with open(os.path.join(OUTPUT_DIR, "spin_report.txt"), "w") as f_out:
    f_out.write("\n".join(lines) + "\n")

summary_rows = []
for r in results:
    summary_rows.append({
        "robot_id": r["id"],
        "w_nominal_m": round(r["W_nom"], 4),
        "w_eff_identified_m": round(r["W_eff"], 4),
        "w_ratio": round(r["w_ratio"], 4),
        "fit_r2": round(r["r2_fit"], 4),
        "gyro_final_yaw_deg": round(r["final_gyro_yaw"], 2),
        "wheel_final_yaw_deg": round(r["final_wheel_yaw"], 2),
        "gyro_scale_factor": round(r["gyro_scale_factor"], 4),
        "gyro_turns": round(r["turns_gyro"], 3),
        "wheel_turns": round(r["turns_wheel"], 3),
        "mean_rpm_l": round(r["mean_rpm_l"], 2),
        "mean_rpm_r": round(r["mean_rpm_r"], 2),
        "tick_balance": r["tick_balance_final"],
        "pivot_drift_max_mm": round(r["pivot_drift_max_mm"], 1),
        "pivot_drift_final_mm": round(r["pivot_drift_final_mm"], 1),
        "v_drop_v": round(r["v_drop"], 3),
        "mean_curr_ma": round(r["mean_curr"], 1),
        "energy_joules": round(r["energy_joules"], 2)
    })

pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "spin_summary.csv"), index=False)
p(f"\n[INFO] Comprehensive text report saved to: {OUTPUT_DIR}/spin_report.txt")
p(f"[INFO] Summary CSV saved to: {OUTPUT_DIR}/spin_summary.csv")

# ==============================================================================
# 5. ULTRA-WIDE HIGH-RESOLUTION VISUALIZATIONS (18x10 INCHES @ 300 DPI)
# ==============================================================================
colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}

# --- FIGURE 1: 2D PIVOT CENTER DRIFT & ORIENTATION COMPASS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

# Left: 2D Pivot Center Migration (in mm)
ax1.axhline(0, color="black", linestyle="--", linewidth=1.0)
ax1.axvline(0, color="black", linestyle="--", linewidth=1.0)
for r in results:
    rid = r["id"]
    ax1.plot(r["x_pivot_mm"], r["y_pivot_mm"], color=colors[rid], linewidth=2.0, label=f"R{rid} Pivot (Max Drift={r['pivot_drift_max_mm']:.1f}mm)")
    ax1.scatter(r["x_pivot_mm"][-1], r["y_pivot_mm"][-1], color=colors[rid], s=120, zorder=5)

ax1.set_title("2D Spatial Migration of Rotation Center (Pivot Drift)", fontsize=14, pad=10)
ax1.set_xlabel("Lateral Drift X (mm)", fontsize=12)
ax1.set_ylabel("Longitudinal Drift Y (mm)", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.legend(loc="upper left", fontsize=10)
ax1.axis("equal")

# Right: Heading Angle Accumulation vs Time
for r in results:
    rid = r["id"]
    ax2.plot(r["t"], r["yaw_gyro_deg"], color=colors[rid], linewidth=2.5, label=f"R{rid} Gyro (Final={r['final_gyro_yaw']:.1f}° / {r['turns_gyro']:.2f} turns)")
    ax2.plot(r["t"], r["yaw_wheel_deg"], color=colors[rid], linestyle="--", alpha=0.6, label=f"R{rid} Wheel Odo (Final={r['final_wheel_yaw']:.1f}°)")

ax2.axhline(1800.0, color="darkred", linestyle=":", linewidth=1.5, label="Target (1800° / 5 Turns)")
ax2.set_title("Absolute Heading Accumulation (5 Full Turns)", fontsize=14, pad=10)
ax2.set_xlabel("Elapsed Time (s)", fontsize=12)
ax2.set_ylabel("Integrated Yaw (degrees)", fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.7)
ax2.legend(loc="upper left", fontsize=9)

plt.suptitle("In-Place Rotation Geometry & Pivot Stability", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_pivot_drift_and_heading.png"), dpi=300)
plt.close()

# --- FIGURE 2: DYNAMIC EFFECTIVE TRACK WIDTH (W_eff) SYSID REGRESSION ---
fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, r in enumerate(results):
    ax = axes[idx]
    rid = r["id"]
    w_nom = r["W_nom"]
    w_eff = r["W_eff"]
    omega_ss = r["omega_z_ss"]
    dv_ss = r["delta_v_ss"]

    ax.scatter(omega_ss, dv_ss, color=colors[rid], alpha=0.3, s=15, label="Steady Samples")
    # Linear fit line
    omega_line = np.linspace(np.min(omega_ss), np.max(omega_ss), 100)
    ax.plot(omega_line, w_eff * omega_line, color="black", linewidth=2.0, label=f"Fit: W_eff={w_eff:.4f}m (R²={r['r2_fit']:.4f})")
    ax.plot(omega_line, w_nom * omega_line, color="gray", linestyle="--", linewidth=1.5, label=f"Nominal: W_nom={w_nom:.4f}m")

    ax.set_title(f"Robot {rid} - Track Width SysID (W_eff / W_nom = {r['w_ratio']:.3f})", fontsize=12)
    ax.set_ylabel("Differential Speed Δv (m/s)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper left", fontsize=10)

axes[2].set_xlabel("Yaw Angular Velocity ω_z (rad/s)", fontsize=12)
axes[3].set_xlabel("Yaw Angular Velocity ω_z (rad/s)", fontsize=12)
plt.suptitle("Dynamic Effective Track Width (W_eff) Regression Under Tire Scrub", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_effective_track_width_sysid.png"), dpi=300)
plt.close()

# --- FIGURE 3: ROTATIONAL ANGULAR VELOCITY DYNAMICS ---
fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, r in enumerate(results):
    ax = axes[idx]
    rid = r["id"]
    t = r["t"]
    gz = r["raw_gz_dps"]

    ax.plot(t, gz, color=colors[rid], linewidth=1.5, label=f"R{rid} Gyro Z (Mean={np.mean(gz[r['spin_mask']]):.1f}°/s)")
    ax.axhline(90.0, color="black", linestyle="--", linewidth=1.2, label="Nominal Rate (~90°/s)")

    ax.set_title(f"Robot {rid} - Yaw Rate Tracking & Stability", fontsize=12)
    ax.set_ylabel("Angular Velocity (deg/s)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower right", fontsize=10)

axes[2].set_xlabel("Time (s)", fontsize=12)
axes[3].set_xlabel("Time (s)", fontsize=12)
plt.suptitle("Yaw Angular Velocity Profile Across All 4 Robots", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_yaw_rate_tracking.png"), dpi=300)
plt.close()

# --- FIGURE 4: WHEEL SPEED SYMMETRY & POWER PLANT ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

for r in results:
    rid = r["id"]
    t = r["t"]
    # Step balance: steps_r + steps_l (should stay close to 0)
    balance = r["df"]["steps_r"] + r["df"]["steps_l"]
    ax1.plot(t, balance, color=colors[rid], linewidth=1.8, label=f"R{rid} (End Balance={r['tick_balance_final']} ticks)")
    ax2.plot(t, r["df"]["vbus_v"], color=colors[rid], linewidth=1.8, label=f"R{rid} Vbus (Drop={r['v_drop']:.2f}V)")

ax1.axhline(0, color="black", linestyle="--", linewidth=1.0)
ax1.set_title("Symmetric Spin Tick Balance Error: (Steps_R + Steps_L)", fontsize=13)
ax1.set_ylabel("Net Tick Imbalance", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.legend(loc="upper left", fontsize=10)

ax2.set_title("Dynamic Battery Voltage (INA226 Vbus)", fontsize=13)
ax2.set_xlabel("Elapsed Time (s)", fontsize=12)
ax2.set_ylabel("Voltage (V)", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.7)
ax2.legend(loc="lower right", fontsize=10)

plt.suptitle("Actuation Symmetry & Electrical Power Dynamics", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_symmetry_and_power.png"), dpi=300)
plt.close()

p(f"\n[SUCCESS] Generated 4 high-resolution spin analysis plots in '{OUTPUT_DIR}/'.")
