#!/usr/bin/env python3
"""
Anjoman Swarm: Straight-Line Ground Trajectory & Veering Drift Analyzer
Reconstructs 2D (X-Y) Planar Trajectory, Quantifies Lateral Deviation,
Calculates Curvature Radius, and Solves Wheel Ratio Discrepancy (E_d = r_R / r_L).
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "plots_ground"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

# Physical Constants for Robot 1
WHEELBASE_NOMINAL_M = 0.1350  # Track width: 13.5 cm
WHEEL_DIAM_NOMINAL_M = 0.0500 # Wheel diameter: 5.0 cm
WHEEL_RADIUS_NOMINAL_M = WHEEL_DIAM_NOMINAL_M / 2.0
CPR = 4096.0

def parse_ground_csv(filepath):
    """Robust parser stripping PlatformIO timestamps and parsing 15 columns."""
    records = []
    pattern = re.compile(
        r'(?:.*>\s*)?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([-\d]+)\s*,\s*([-\d]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)'
    )

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if "TimeUs" in line or not line or "=" in line or "[" in line:
                continue
            match = pattern.search(line)
            if match:
                try:
                    records.append({
                        'time_us': int(match.group(1)),
                        'cycle': int(match.group(2)),
                        'phase': int(match.group(3)),
                        'dist_target_m': float(match.group(4)),
                        'dist_odom_m': float(match.group(5)),
                        'pwm_l': float(match.group(6)),
                        'pwm_r': float(match.group(7)),
                        'steps_l': int(match.group(8)),
                        'steps_r': int(match.group(9)),
                        'rpm_l': float(match.group(10)),
                        'rpm_r': float(match.group(11)),
                        'gyro_z_dps': float(match.group(12)),
                        'heading_deg': float(match.group(13)),
                        'accel_x': float(match.group(14)),
                        'accel_y': float(match.group(15))
                    })
                except ValueError:
                    continue

    df = pd.DataFrame(records)
    return df

def reconstruct_2d_kinematics(df):
    """
    Isolates the first forward drive leg (Phase 1) and reconstructs:
    1. Encoder-only 2D Odometry (x_enc, y_enc)
    2. Gyro-aided 2D Odometry (x_gyro, y_gyro)
    """
    # Isolate first forward movement (Phase 1 or Cycle 1)
    forward_df = df[(df['phase'] == 1) & (df['cycle'] == 1)].copy()
    if len(forward_df) < 20:
        # Fallback to entire dataset if phase 1 is short
        forward_df = df.copy()

    forward_df['t_sec'] = (forward_df['time_us'] - forward_df['time_us'].iloc[0]) * 1e-6
    
    # Delta steps per interval
    d_ticks_l = forward_df['steps_l'].diff().fillna(0)
    d_ticks_r = forward_df['steps_r'].diff().fillna(0)

    # Linear displacements per wheel (m)
    ds_l = (d_ticks_l / CPR) * (2.0 * np.pi * WHEEL_RADIUS_NOMINAL_M)
    ds_r = (d_ticks_r / CPR) * (2.0 * np.pi * WHEEL_RADIUS_NOMINAL_M)
    ds_avg = (ds_l + ds_r) / 2.0

    # 1. Encoder Heading Integration
    d_theta_enc = (ds_r - ds_l) / WHEELBASE_NOMINAL_M
    theta_enc = np.cumsum(d_theta_enc)

    # 2. Gyro Heading (from raw gyro Z or integrated heading_deg)
    theta_gyro = np.radians(forward_df['heading_deg'] - forward_df['heading_deg'].iloc[0])

    # 2D Planar Positions (Runge-Kutta / Midpoint Integration)
    # Forward along X-axis, Lateral drift along Y-axis
    x_enc = np.cumsum(ds_avg * np.cos(theta_enc))
    y_enc = np.cumsum(ds_avg * np.sin(theta_enc))

    x_gyro = np.cumsum(ds_avg * np.cos(theta_gyro))
    y_gyro = np.cumsum(ds_avg * np.sin(theta_gyro))

    forward_df['x_enc_m'] = x_enc
    forward_df['y_enc_m'] = y_enc
    forward_df['x_gyro_m'] = x_gyro
    forward_df['y_gyro_m'] = y_gyro

    return forward_df

def main():
    filename = "ground_r1_2m.csv"
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    if not os.path.exists(filename):
        print(f"[FATAL] File not found: {filename}")
        sys.exit(1)

    print("=" * 85)
    print(f"       ANJOMAN SWARM - STRAIGHT-LINE TRAJECTORY & VEERING ANALYSIS: {filename}")
    print("=" * 85)

    df_raw = parse_ground_csv(filename)
    if len(df_raw) < 10:
        print("[ERROR] Insufficient data rows in file!")
        sys.exit(1)

    df = reconstruct_2d_kinematics(df_raw)

    # -------------------------------------------------------------
    # Metrological Calculations
    # -------------------------------------------------------------
    total_time_s = df['t_sec'].iloc[-1]
    total_dist_x = df['x_gyro_m'].iloc[-1]
    final_lateral_dev_y = df['y_gyro_m'].iloc[-1]
    max_lateral_dev_y = df['y_gyro_m'].abs().max()
    final_heading_deg = np.degrees(df['y_gyro_m'].iloc[-1] / max(total_dist_x, 1e-3)) # Approx
    final_true_heading_deg = df['heading_deg'].iloc[-1] - df['heading_deg'].iloc[0]

    total_ticks_l = df['steps_l'].iloc[-1] - df['steps_l'].iloc[0]
    total_ticks_r = df['steps_r'].iloc[-1] - df['steps_r'].iloc[0]

    # Wheel Diameter Discrepancy Ratio: E_d = r_R / r_L
    # If robot veers left, right wheel traveled farther than left
    if total_ticks_l > 0:
        tick_ratio = total_ticks_r / total_ticks_l
    else:
        tick_ratio = 1.0

    # Curvature Radius R (m)
    # theta = ArcLength / R  => R = ArcLength / theta
    final_heading_rad = np.radians(abs(final_true_heading_deg))
    if final_heading_rad > 1e-4:
        radius_of_curvature_m = total_dist_x / final_heading_rad
    else:
        radius_of_curvature_m = np.inf

    # Analytical Wheel Ratio Correction (Borenstein UMBark Formula)
    # E_d = (2 * R + W) / (2 * R - W)
    if np.isfinite(radius_of_curvature_m) and radius_of_curvature_m > WHEELBASE_NOMINAL_M:
        if final_true_heading_deg > 0: # Veering Left
            e_d_factor = (2.0 * radius_of_curvature_m + WHEELBASE_NOMINAL_M) / (2.0 * radius_of_curvature_m - WHEELBASE_NOMINAL_M)
        else: # Veering Right
            e_d_factor = (2.0 * radius_of_curvature_m - WHEELBASE_NOMINAL_M) / (2.0 * radius_of_curvature_m + WHEELBASE_NOMINAL_M)
    else:
        e_d_factor = 1.0

    # -------------------------------------------------------------
    # Console Report
    # -------------------------------------------------------------
    print("\n" + "#" * 85)
    print("                     KINEMATIC VEERING & DRIFT REPORT")
    print("#" * 85)
    print(f"Forward Run Duration         : {total_time_s:.2f} Seconds")
    print(f"Longitudinal Distance (X)    : {total_dist_x:.4f} m  (Target: 2.000 m)")
    print(f"Lateral Deviation (Y)        : {final_lateral_dev_y * 100.0:+.2f} cm  (Max Dev: {max_lateral_dev_y * 100.0:.2f} cm)")
    print(f"Net Heading Angular Veering  : {final_true_heading_deg:+.2f} Degrees ({'Veering LEFT' if final_true_heading_deg > 0 else 'Veering RIGHT'})")
    print(f"Curvature Radius (R)         : {radius_of_curvature_m:.3f} meters")
    print("-" * 85)
    print(f"Left Wheel Total Ticks       : {total_ticks_l:,} ticks")
    print(f"Right Wheel Total Ticks      : {total_ticks_r:,} ticks (Delta: {total_ticks_r - total_ticks_l:+d} ticks)")
    print(f"Wheel Rotation Speed Ratio   : {tick_ratio:.4f} (Ideal: 1.0000)")
    print(f"Identified Wheel Ratio (E_d) : {e_d_factor:.5f} (r_R / r_L correction multiplier)")
    print("#" * 85)

    # Root Cause Diagnostic
    print("\n[ROOT-CAUSE DIAGNOSIS]:")
    if abs(total_ticks_r - total_ticks_l) > 200:
        print(" -> CAUSE: CONTROLLER / WHEEL SPEED MISMATCH.")
        print(f"    Right wheel turned {abs(total_ticks_r - total_ticks_l)} more pulses than Left wheel.")
        print("    The PI velocity controller on Left wheel was lagging or Right motor received higher duty.")
    else:
        print(" -> CAUSE: MECHANICAL TIRE CONTACT / DIAMETER ASYMMETRY.")
        print("    Encoder pulses are nearly equal, but robot curved due to physical tire deformation.")

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: 2D Floor Planar Trajectory (X vs Y)
    ax1 = plt.subplot(2, 2, (1, 3))
    ax1.plot([0, 2.0], [0, 0], 'k--', lw=2, label='Ideal Straight Path (Ground Truth)')
    ax1.plot(df['x_enc_m'], df['y_enc_m'], 'b-', lw=2, label='Encoder Odometry Path')
    ax1.plot(df['x_gyro_m'], df['y_gyro_m'], 'r-', lw=2.5, label='Gyro-Aided Floor Path (True Motion)')
    
    # Mark start and end points
    ax1.scatter([0], [0], color='green', s=100, zorder=5, label='Start (0, 0)')
    ax1.scatter([total_dist_x], [final_lateral_dev_y], color='red', s=120, zorder=5, 
                label=f'Stop Point (X={total_dist_x:.2f}m, Y={final_lateral_dev_y*100:+.1f}cm)')
    
    ax1.set_title("2D Floor Planar Trajectory (X: Forward, Y: Lateral Veering)", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Longitudinal Distance X (Meters)")
    ax1.set_ylabel("Lateral Deviation Y (Meters)")
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Heading Angle (Degrees) vs Distance
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df['x_gyro_m'], df['heading_deg'] - df['heading_deg'].iloc[0], 'r-', lw=2)
    ax2.axhline(0, color='black', linestyle=':', lw=1.2)
    ax2.set_title("Robot Heading Angle (Yaw) vs Traveled Distance", fontweight='bold')
    ax2.set_xlabel("Forward Distance X (m)")
    ax2.set_ylabel("Heading Angle (Degrees)")
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Wheel Speeds (RPM) vs Time
    ax3 = plt.subplot(2, 2, 4)
    ax3.plot(df['t_sec'], df['rpm_l'], 'b-', lw=1.5, label='Left Wheel RPM')
    ax3.plot(df['t_sec'], df['rpm_r'], 'r--', lw=1.5, label='Right Wheel RPM')
    ax3.set_title("Left vs Right Wheel Velocity Profile", fontweight='bold')
    ax3.set_xlabel("Time (Seconds)")
    ax3.set_ylabel("Speed (RPM)")
    ax3.legend(loc='lower right')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_filepath = os.path.join(OUTPUT_DIR, "straight_line_2d_trajectory.png")
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"\n[SAVED] 2D Trajectory and Veering Plot: {plot_filepath}")

if __name__ == "__main__":
    main()
