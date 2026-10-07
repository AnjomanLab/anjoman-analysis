#!/usr/bin/env python3
"""
Anjoman Swarm Metrology & 2D Maneuver Kinematic Reconstruction Engine
Analyzes 4-robot maneuver CSV logs, detects actuator faults, evaluates motor-UWB EMI,
and reconstructs the 2D spatial trajectory and rigid formation geometry.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ==============================================================================
# 1. LOAD CSV DATASETS
# ==============================================================================
FILES = {
    1: "robot_1_maneuver.csv",
    2: "robot_2_maneuver.csv",
    3: "robot_3_maneuver.csv",
    4: "robot_4_maneuver.csv"
}

data = {}
print("=" * 85)
print("ANJOMAN SWARM 4-ROBOT FORMATION MANEUVER AUDIT")
print("=" * 85)

for r_id, fname in FILES.items():
    if not os.path.exists(fname):
        # Try alternate name if user named it test.csv
        alt = f"robot_{r_id}_test.csv"
        if os.path.exists(alt):
            fname = alt
        else:
            print(f"[MISSING] File for Robot {r_id} not found ({fname})")
            continue
    df = pd.read_csv(fname)
    df["t_sec"] = (df["t_ms"] - df["t_ms"].iloc[0]) / 1000.0
    data[r_id] = df
    print(f"Robot {r_id}: Loaded {len(df)} records from {fname} (Duration: {df['t_sec'].iloc[-1]:.1f}s)")

if not data:
    sys.exit("[FATAL] No CSV files found to analyze!")

# ==============================================================================
# 2. AUTOMATIC ROOT-CAUSE DIAGNOSIS (ESPECIALLY ROBOT 4)
# ==============================================================================
print("\n" + "=" * 85)
print("ACTUATOR & MOTION DIAGNOSIS (WHY DID ROBOT 4 NOT MOVE?)")
print("=" * 85)

for r_id in sorted(data.keys()):
    df = data[r_id]
    max_v_cmd = df["v_cmd"].abs().max()
    max_rpm_l = df["rpm_l"].abs().max()
    max_rpm_r = df["rpm_r"].abs().max()
    avg_vbat  = df["vbat"].mean()
    max_curr  = df["current_a"].max()
    dist_disp = np.hypot(df["x"].iloc[-1] - df["x"].iloc[0], df["y"].iloc[-1] - df["y"].iloc[0])

    print(f"\n--- Diagnostic Profile: Robot {r_id} ---")
    print(f"  Command Peak Velocity   : {max_v_cmd:.3f} m/s")
    print(f"  Left Wheel Peak RPM     : {max_rpm_l:.1f} RPM")
    print(f"  Right Wheel Peak RPM    : {max_rpm_r:.1f} RPM")
    print(f"  Net Physical Displacement: {dist_disp:.3f} m")
    print(f"  Battery Voltage (Mean)  : {avg_vbat:.2f} V")
    print(f"  Motor Current Peak      : {max_curr:.2f} A")

    # Automated Diagnosis Logic
    if dist_disp < 0.05:
        if max_v_cmd < 0.01:
            print("  >>> VERDICT: [RADIO SYNC FAILURE] Robot never received the Start Trigger from Robot 1!")
            print("      Action: Check ESP-NOW antenna / WiFi channel configuration.")
        elif max_rpm_l < 1.0 and max_rpm_r < 1.0:
            if avg_vbat < 5.5:
                print("  >>> VERDICT: [POWER CUT-OFF] Battery voltage collapsed below 5.5V (BMS Cutoff)!")
            else:
                print("  >>> VERDICT: [ACTUATOR LOCK / STICTION] Velocity was commanded, but motor did not rotate!")
                print("      Action: Check motor wiring on pins 15/16 or increase deadband in RobotConfig.h.")
    else:
        print(f"  >>> VERDICT: [NORMAL MOTION] Robot actively tracked trajectory (Displacement: {dist_disp:.2f} m)")

# ==============================================================================
# 3. ELECTRICAL & UWB EMI INTERFERENCE ANALYSIS
# ==============================================================================
print("\n" + "=" * 85)
print("ELECTRICAL POWER & MOTOR-UWB INTERFERENCE (EMI METRICS)")
print("=" * 85)
header = f"{'Robot':<6} | {'V_Bat Min (V)':<13} | {'I_Motor Max (A)':<15} | {'Std Noise (Mean)':<16} | {'Noise-Current Corr'}"
print(header)
print("-" * len(header))

for r_id in sorted(data.keys()):
    df = data[r_id]
    v_min = df["vbat"].min()
    i_max = df["current_a"].max()
    noise_mean = df["uwb_std_noise"].mean()
    
    # Cross-correlation between motor current and UWB noise floor
    if df["current_a"].std() > 1e-4 and df["uwb_std_noise"].std() > 1e-4:
        corr_ni = np.corrcoef(df["current_a"], df["uwb_std_noise"])[0, 1]
        corr_str = f"{corr_ni:+.3f}"
    else:
        corr_str = "0.000 (Stable)"

    print(f"R{r_id:<5} | {v_min:<13.2f} | {i_max:<15.2f} | {noise_mean:<16.1f} | {corr_str}")

# ==============================================================================
# 4. HIGH-RESOLUTION 2D SPATIAL SIMULATION & VISUALIZATION (14x10 in @ 300 DPI)
# ==============================================================================
print("\n[PLOTTING] Generating 2D Swarm Trajectory and Kinematic Audit Plots...")

# Figure 1: 2D Spatial Plane Trajectory Reconstruction
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
colors = {1: "red", 2: "blue", 3: "green", 4: "purple"}

# Draw Initial and Final Reference Squares
sq_init = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
sq_final = np.array([[1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5]]) # Rotated 90 + 3m

poly_init = Polygon(sq_init, fill=False, edgecolor="black", linestyle="--", linewidth=1.5, label="Initial Square (2.0 m)")
poly_final = Polygon(sq_final, fill=False, edgecolor="darkgreen", linestyle=":", linewidth=2.0, label="Final Target Square (3.0 m)")
ax.add_patch(poly_init)
ax.add_patch(poly_final)

# Plot Robot Trajectories
for r_id in sorted(data.keys()):
    df = data[r_id]
    ax.plot(df["x"], df["y"], color=colors[r_id], linewidth=2.2, label=f"Robot {r_id} Path")
    ax.scatter(df["x"].iloc[0], df["y"].iloc[0], color=colors[r_id], s=100, marker="o", edgecolors="black", zorder=5)
    ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color=colors[r_id], s=120, marker="s", edgecolors="black", zorder=5)

    # Plot Heading orientation arrows at intervals
    step = max(len(df) // 8, 1)
    for k in range(0, len(df), step):
        xk, yk, thk = df["x"].iloc[k], df["y"].iloc[k], df["heading"].iloc[k]
        ax.arrow(xk, yk, 0.10 * np.cos(thk), 0.10 * np.sin(thk),
                 head_width=0.04, head_length=0.04, fc=colors[r_id], ec=colors[r_id], alpha=0.7)

ax.set_title("Swarm 2D Trajectory Reconstruction (Rotation 90° + Expansion to 3m)", fontsize=14, fontweight="bold")
ax.set_xlabel("Formation X (meters)", fontsize=12)
ax.set_ylabel("Formation Y (meters)", fontsize=12)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig("swarm_2d_trajectory_reconstruction.png")
plt.close()

# Figure 2: Velocity & Actuation Response Over Time
fig, axes = plt.subplots(4, 1, figsize=(12, 10), dpi=300, sharex=True)
for r_id in sorted(data.keys()):
    df = data[r_id]
    axes[0].plot(df["t_sec"], df["v_cmd"], color=colors[r_id], label=f"R{r_id}")
    axes[1].plot(df["t_sec"], df["omega_cmd"], color=colors[r_id], label=f"R{r_id}")
    axes[2].plot(df["t_sec"], df["rpm_l"], color=colors[r_id], label=f"R{r_id} Left")
    axes[3].plot(df["t_sec"], df["rpm_r"], color=colors[r_id], label=f"R{r_id} Right")

axes[0].set_ylabel("v_cmd (m/s)")
axes[0].set_title("Chassis Linear Velocity Commands")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="upper right")

axes[1].set_ylabel("omega_cmd (rad/s)")
axes[1].set_title("Chassis Angular Velocity Commands")
axes[1].grid(True, alpha=0.3)

axes[2].set_ylabel("Left RPM")
axes[2].set_title("Measured Left Wheel Speed (Closed-Loop Feedback)")
axes[2].grid(True, alpha=0.3)

axes[3].set_ylabel("Right RPM")
axes[3].set_xlabel("Elapsed Time (seconds)")
axes[3].set_title("Measured Right Wheel Speed (Closed-Loop Feedback)")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("actuation_velocity_profiles.png")
plt.close()

# Figure 3: Motor Current vs UWB Noise Floor (EMI Assessment)
fig, ax1 = plt.subplots(figsize=(12, 5), dpi=300)
ax2 = ax1.twinx()

for r_id in sorted(data.keys()):
    df = data[r_id]
    ax1.plot(df["t_sec"], df["current_a"], color=colors[r_id], linestyle="-", alpha=0.8, label=f"R{r_id} Motor Current")
    ax2.plot(df["t_sec"], df["uwb_std_noise"], color=colors[r_id], linestyle=":", alpha=0.6, label=f"R{r_id} UWB Noise Floor")

ax1.set_xlabel("Elapsed Time (seconds)", fontsize=12)
ax1.set_ylabel("Total Motor Current (Amperes)", fontsize=12, color="black")
ax2.set_ylabel("UWB Receiver Noise Floor (std_noise)", fontsize=12, color="darkblue")
plt.title("Impact of Motor Current and Inverter Switching on UWB Noise Floor (EMI Audit)", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("motor_current_vs_uwb_noise.png")
plt.close()

print("\n" + "=" * 85)
print("SAVED VISUALIZATIONS:")
print("  1. swarm_2d_trajectory_reconstruction.png  (2D Spatial Path & Rigid Square)")
print("  2. actuation_velocity_profiles.png         (Closed-loop RPM & Commanded Velocities)")
print("  3. motor_current_vs_uwb_noise.png          (Electromagnetic Interference Check)")
print("=" * 85 + "\n")
