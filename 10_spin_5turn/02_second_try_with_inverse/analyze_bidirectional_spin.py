#!/usr/bin/env python3
"""
Anjoman Swarm Firmware - 10-Turn Bidirectional (CCW + CW) Spin Benchmark Analyzer
Decoupled UMBmark Metrology Engine with Individual High-Resolution Petal Drift Plots
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NOMINAL_TRACK_WIDTHS = {1: 0.1350, 2: 0.1250, 3: 0.1250, 4: 0.1250}
CALIBRATED_R_EFF_M   = 0.02685
CPR                  = 4096.0
OUTPUT_DIR           = "analysis_results_bi_spin"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_bi_spin(filepath):
    try:
        base = os.path.basename(filepath)
        rid = int(base.split("spin_bi_10turn_r")[1].split(".csv")[0])
    except Exception:
        return None

    W_nom = NOMINAL_TRACK_WIDTHS.get(rid, 0.1250)
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    t_raw = df["elapsed_ms"].values / 1000.0
    t = t_raw - t_raw[0]

    # Mask phases: 1 = CCW, 3 = CW
    mask_ccw = (df["phase"] == 1).values
    mask_cw  = (df["phase"] == 3).values

    steps_l = df["steps_l"].values
    steps_r = df["steps_r"].values
    yaw_gyro = df["integrated_yaw_deg"].values if "integrated_yaw_deg" in df.columns else df["yaw_deg"].values

    # Continuous displacement (meters)
    s_l = (steps_l / CPR) * (2.0 * math.pi * CALIBRATED_R_EFF_M)
    s_r = (steps_r / CPR) * (2.0 * math.pi * CALIBRATED_R_EFF_M)

    # 1. CCW Phase Metrology (Nominal: +3600 deg)
    ccw_indices = np.where(mask_ccw)[0]
    if len(ccw_indices) > 0:
        idx_c0, idx_c1 = ccw_indices[0], ccw_indices[-1]
        delta_sl_ccw = s_l[idx_c1] - s_l[idx_c0]
        delta_sr_ccw = s_r[idx_c1] - s_r[idx_c0]
        theta_wheel_ccw = (delta_sr_ccw - delta_sl_ccw) / W_nom * (180.0 / math.pi)
        theta_gyro_ccw = yaw_gyro[idx_c1] - yaw_gyro[idx_c0]
    else:
        theta_wheel_ccw = theta_gyro_ccw = 3600.0

    # 2. CW Phase Metrology (Nominal: -3600 deg)
    cw_indices = np.where(mask_cw)[0]
    if len(cw_indices) > 0:
        idx_w0, idx_w1 = cw_indices[0], cw_indices[-1]
        delta_sl_cw = s_l[idx_w1] - s_l[idx_w0]
        delta_sr_cw = s_r[idx_w1] - s_r[idx_w0]
        theta_wheel_cw = (delta_sr_cw - delta_sl_cw) / W_nom * (180.0 / math.pi)
        theta_gyro_cw = yaw_gyro[idx_w1] - yaw_gyro[idx_w0]
    else:
        theta_wheel_cw = theta_gyro_cw = -3600.0

    # 3. UMBmark Decoupling Formula
    # Effective track width decoupled from wheel diameter ratio:
    # W_eff = W_nom * (3600 / (|theta_ccw| + |theta_cw|) / 2)
    avg_rotation_mag = (abs(theta_gyro_ccw) + abs(theta_gyro_cw)) / 2.0
    W_eff_decoupled = W_nom * (3600.0 / avg_rotation_mag) if avg_rotation_mag > 10.0 else W_nom
    w_ratio = W_eff_decoupled / W_nom

    # Residual diameter asymmetry ratio: Ed = |theta_ccw| / |theta_cw|
    Ed_spin = abs(theta_gyro_ccw) / abs(theta_gyro_cw) if abs(theta_gyro_cw) > 10.0 else 1.0

    # Net loop closure heading error after full 10 CCW + 10 CW (Nominal return = 0 deg)
    net_return_yaw_deg = yaw_gyro[-1]

    # 4. 2D Pivot Center Drift Tracking (Astroid/Hypocycloid Flower Petals)
    ds_l = np.diff(s_l, prepend=s_l[0])
    ds_r = np.diff(s_r, prepend=s_r[0])
    ds_mid = (ds_l + ds_r) / 2.0

    th_rad = np.radians(yaw_gyro)
    x_pivot = np.zeros(len(df))
    y_pivot = np.zeros(len(df))
    for k in range(1, len(df)):
        th_mid = (th_rad[k-1] + th_rad[k]) / 2.0
        x_pivot[k] = x_pivot[k-1] + ds_mid[k] * np.sin(th_mid)
        y_pivot[k] = y_pivot[k-1] + ds_mid[k] * np.cos(th_mid)

    x_pivot_mm = x_pivot * 1000.0
    y_pivot_mm = y_pivot * 1000.0
    max_drift_mm = np.max(np.sqrt(x_pivot_mm**2 + y_pivot_mm**2))
    end_drift_mm = math.sqrt(x_pivot_mm[-1]**2 + y_pivot_mm[-1]**2)

    return {
        "id": rid,
        "df": df,
        "t": t,
        "W_nom": W_nom,
        "W_eff_decoupled": W_eff_decoupled,
        "w_ratio": w_ratio,
        "Ed_spin": Ed_spin,
        "theta_wheel_ccw": theta_wheel_ccw,
        "theta_gyro_ccw": theta_gyro_ccw,
        "theta_wheel_cw": theta_wheel_cw,
        "theta_gyro_cw": theta_gyro_cw,
        "net_return_yaw_deg": net_return_yaw_deg,
        "x_pivot_mm": x_pivot_mm,
        "y_pivot_mm": y_pivot_mm,
        "max_drift_mm": max_drift_mm,
        "end_drift_mm": end_drift_mm,
        "mask_ccw": mask_ccw,
        "mask_cw": mask_cw,
        "yaw_gyro": yaw_gyro
    }

files = sorted(glob.glob("spin_bi_10turn_r*.csv"))
if not files:
    print("[ERROR] No files matching 'spin_bi_10turn_r*.csv' found!")
    exit(1)

results = [process_bi_spin(f) for f in files if process_bi_spin(f) is not None]
results.sort(key=lambda x: x["id"])

# Print Summary Table
lines = []
def p(msg=""):
    print(msg)
    lines.append(msg)

p("=" * 125)
p("   ANJOMAN 10-TURN BIDIRECTIONAL (CCW + CW) BENCHMARK - DECOUPLED UMBmark REPORT")
p("=" * 125)
p(f"{'Robot':<7}|{'W_nom (m)':<11}|{'W_eff (m)':<11}|{'W_ratio':<10}|{'Ed (Spin)':<11}|{'CCW Gyro':<12}|{'CW Gyro':<12}|{'Return Error':<14}|{'Max Drift'}")
p("-" * 125)

for r in results:
    p(f"R{r['id']:<6}| {r['W_nom']:<9.4f} | {r['W_eff_decoupled']:<9.4f} | {r['w_ratio']:<8.4f} | {r['Ed_spin']:<9.5f} | {r['theta_gyro_ccw']:>8.1f}°   | {r['theta_gyro_cw']:>8.1f}°   | {r['net_return_yaw_deg']:>9.2f}°     | {r['max_drift_mm']:>6.1f} mm")

p("=" * 125)

with open(os.path.join(OUTPUT_DIR, "bidirectional_spin_report.txt"), "w") as f_out:
    f_out.write("\n".join(lines) + "\n")

# Save Summary CSV
summary_df = pd.DataFrame([{
    "robot_id": r["id"],
    "w_nominal_m": round(r["W_nom"], 4),
    "w_eff_decoupled_m": round(r["W_eff_decoupled"], 4),
    "w_ratio": round(r["w_ratio"], 4),
    "ed_spin": round(r["Ed_spin"], 5),
    "ccw_gyro_deg": round(r["theta_gyro_ccw"], 2),
    "cw_gyro_deg": round(r["theta_gyro_cw"], 2),
    "return_yaw_error_deg": round(r["net_return_yaw_deg"], 2),
    "max_pivot_drift_mm": round(r["max_drift_mm"], 2),
    "end_pivot_drift_mm": round(r["end_drift_mm"], 2)
} for r in results])
summary_df.to_csv(os.path.join(OUTPUT_DIR, "bidirectional_spin_summary.csv"), index=False)

# --- INDIVIDUAL HIGH-RESOLUTION PETAL DRIFT PLOTS (1 PER ROBOT) ---
for r in results:
    rid = r["id"]
    plt.figure(figsize=(10, 10), dpi=300)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.0)

    # Plot CCW Phase in Blue, CW Phase in Magenta
    x_mm = r["x_pivot_mm"]
    y_mm = r["y_pivot_mm"]
    m_ccw = r["mask_ccw"]
    m_cw  = r["mask_cw"]

    plt.plot(x_mm[m_ccw], y_mm[m_ccw], color="#1f77b4", linewidth=2.0, label="10-Turn CCW Phase")
    plt.plot(x_mm[m_cw], y_mm[m_cw], color="#d62728", linewidth=2.0, linestyle="-", label="10-Turn CW Phase")

    plt.scatter([0], [0], color="black", s=100, zorder=6, label="Start Origin (0,0)")
    plt.scatter([x_mm[-1]], [y_mm[-1]], color="green", s=120, zorder=7, label=f"Final Endpoint ({r['end_drift_mm']:.1f} mm)")
    plt.title(f"Robot {rid} - 2D Rotation Pivot Drift (Flower Petal Hypocycloid)\nDecoupled W_eff = {r['W_eff_decoupled']:.4f} m (Ratio: {r['w_ratio']:.4f})", fontsize=13, pad=12)
    plt.xlabel("Lateral Center Drift X (mm)", fontsize=12)
    plt.ylabel("Longitudinal Center Drift Y (mm)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="upper left", fontsize=10)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"petal_drift_r{rid}.png"))
    plt.close()

# --- OVERALL COMPARATIVE HEADING & VOLTAGE FIGURE ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}

for r in results:
    rid = r["id"]
    ax1.plot(r["t"], r["yaw_gyro"], color=colors[rid], linewidth=2.0, label=f"R{rid} (Net Return={r['net_return_yaw_deg']:.1f}°)")
    ax2.plot(r["t"], r["df"]["vbus_v"], color=colors[rid], linewidth=1.8, label=f"R{rid} Vbus")

ax1.axhline(0, color="black", linestyle="--", linewidth=1.0)
ax1.axhline(3600, color="gray", linestyle=":", linewidth=1.2, label="+3600° (10 CCW)")
ax1.set_title("Full 20-Turn Bidirectional Heading Tracking Profile", fontsize=14)
ax1.set_ylabel("Integrated Yaw (degrees)", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.legend(loc="upper right", fontsize=10)

ax2.set_title("INA226 Dynamic Rail Voltage", fontsize=14)
ax2.set_xlabel("Elapsed Time (seconds)", fontsize=12)
ax2.set_ylabel("Voltage (V)", fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.7)
ax2.legend(loc="lower right", fontsize=10)

plt.suptitle("Bidirectional 10-Turn Multi-Robot Benchmark Overview", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "00_bidirectional_overview.png"), dpi=300)
plt.close()

p(f"\n[SUCCESS] Generated 4 individual high-res petal plots and summary in '{OUTPUT_DIR}/'.")
