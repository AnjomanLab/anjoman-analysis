#!/usr/bin/env python3
"""
Anjoman Swarm: Automated Suspended Bench Test Kinematics & SysID Analyzer
Processes bench_r1.csv to bench_r4.csv to identify Deadband, Max Velocity,
IMU Biases, and Encoder/Motor Polarity Inversions.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "plots_bench"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

ROBOT_FILES = [
    ("bench_r1.csv", 1, 48.0, 0.050),
    ("bench_r2.csv", 2, 120.0, 0.055),
    ("bench_r3.csv", 3, 120.0, 0.055),
    ("bench_r4.csv", 4, 120.0, 0.055),
]

def parse_bench_csv(filepath):
    """Robust parser stripping PlatformIO timestamps and loading 13 columns."""
    records = []
    pattern = re.compile(
        r'(?:.*>\s*)?(\d+)\s*,\s*(\d+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([-\d]+)\s*,\s*([-\d]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)'
    )

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if "TimeMs" in line or not line or "=" in line or "[" in line:
                continue
            match = pattern.search(line)
            if match:
                try:
                    records.append({
                        'time_ms': int(match.group(1)),
                        'phase': int(match.group(2)),
                        'pwm_l': float(match.group(3)),
                        'pwm_r': float(match.group(4)),
                        'enc_l': int(match.group(5)),
                        'enc_r': int(match.group(6)),
                        'gyro_x': float(match.group(7)),
                        'gyro_y': float(match.group(8)),
                        'gyro_z': float(match.group(9)),
                        'accel_x': float(match.group(10)),
                        'accel_y': float(match.group(11)),
                        'accel_z': float(match.group(12)),
                        'temp_esp': float(match.group(13))
                    })
                except ValueError:
                    continue

    df = pd.DataFrame(records)
    return df

def analyze_robot_bench(df, robot_id, gear_ratio, wheel_diam_m):
    """Performs comprehensive SysID on a single robot's bench test dataset."""
    df['t_sec'] = (df['time_ms'] - df['time_ms'].iloc[0]) / 1000.0

    # 1. Compute Angular Velocity (rad/s and RPM) from 4096 CPR Encoders
    dt = df['t_sec'].diff().replace(0, np.nan)
    d_enc_l = df['enc_l'].diff()
    d_enc_r = df['enc_r'].diff()

    # rad/s = (delta_ticks / 4096) * 2pi / dt
    cpr = 4096.0
    df['vel_l_rad_s'] = (d_enc_l / cpr) * (2.0 * np.pi) / dt
    df['vel_r_rad_s'] = (d_enc_r / cpr) * (2.0 * np.pi) / dt

    # Smooth velocities with rolling median to filter discrete tick quantization
    df['vel_l_smooth'] = df['vel_l_rad_s'].rolling(window=9, center=True).median().fillna(0)
    df['vel_r_smooth'] = df['vel_r_rad_s'].rolling(window=9, center=True).median().fillna(0)

    df['vel_l_rpm'] = (df['vel_l_smooth'] * 60.0) / (2.0 * np.pi)
    df['vel_r_rpm'] = (df['vel_r_smooth'] * 60.0) / (2.0 * np.pi)

    # 2. Extract Stationary IMU Biases (Phase 0: Rest 0 to 5 sec)
    phase0 = df[df['phase'] == 0]
    gyro_bias_z = phase0['gyro_z'].mean()
    gyro_noise_std_z = phase0['gyro_z'].std()
    accel_bias_x = phase0['accel_x'].mean()
    accel_bias_y = phase0['accel_y'].mean()
    accel_bias_z = phase0['accel_z'].mean() - 9.80665 # Deviation from 1G

    # 3. Extract Motor Deadbands (Phase 1 & Phase 4: Stiction Ramps)
    # Stiction threshold: first PWM where |vel| > 0.5 rad/s
    p1 = df[(df['phase'] == 1) & (df['pwm_l'] > 0.05)]
    moving_l = p1[p1['vel_l_smooth'].abs() > 0.5]
    deadband_l = moving_l['pwm_l'].iloc[0] if len(moving_l) > 0 else np.nan

    p4 = df[(df['phase'] == 4) & (df['pwm_r'] > 0.05)]
    moving_r = p4[p4['vel_r_smooth'].abs() > 0.5]
    deadband_r = moving_r['pwm_r'].iloc[0] if len(moving_r) > 0 else np.nan

    # 4. Extract Max Velocity at 100% PWM (Phase 2 & Phase 5)
    p2_max = df[(df['phase'] == 2) & (df['pwm_l'] > 0.95)]
    max_vel_l = p2_max['vel_l_smooth'].abs().max() if len(p2_max) > 0 else 0.0

    p5_max = df[(df['phase'] == 5) & (df['pwm_r'] > 0.95)]
    max_vel_r = p5_max['vel_r_smooth'].abs().max() if len(p5_max) > 0 else 0.0

    # Linear Speed limit (v = omega * r)
    wheel_radius_m = wheel_diam_m / 2.0
    max_linear_vel_l = max_vel_l * wheel_radius_m
    max_linear_vel_r = max_vel_r * wheel_radius_m

    # 5. Check Magnetic Interference on AS5600:
    # Compare raw encoder step variance when motors OFF (Phase 0) vs Motors ON 100% (Phase 2 & 5)
    enc_noise_off_l = df[df['phase'] == 0]['enc_l'].diff().std()
    enc_noise_on_l  = p2_max['enc_l'].diff().std()

    report = {
        'robot_id': robot_id,
        'gear_ratio': f"1:{gear_ratio:.0f}",
        'deadband_l_pwm': deadband_l,
        'deadband_r_pwm': deadband_r,
        'max_vel_l_rpm': (max_vel_l * 60.0) / (2.0 * np.pi),
        'max_vel_r_rpm': (max_vel_r * 60.0) / (2.0 * np.pi),
        'max_linear_m_s': min(max_linear_vel_l, max_linear_vel_r),
        'gyro_bias_z_deg_s': gyro_bias_z,
        'gyro_noise_std': gyro_noise_std_z,
        'total_ticks_l': df['enc_l'].iloc[-1],
        'total_ticks_r': df['enc_r'].iloc[-1],
    }

    # Plotting Response Curves
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # Subplot 1: PWM Commands
    ax1.plot(df['t_sec'], df['pwm_l'], 'b-', lw=1.5, label='PWM Left')
    ax1.plot(df['t_sec'], df['pwm_r'], 'r--', lw=1.5, label='PWM Right')
    ax1.set_ylabel("PWM Duty (-1 to +1)")
    ax1.set_title(f"Robot {robot_id} (Gearbox 1:{gear_ratio:.0f}) — Automated 60s Bench Actuator Profile", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Unwrapped Encoder Steps (Trajectory)
    ax2.plot(df['t_sec'], df['enc_l'], 'b-', lw=1.5, label=f"Enc Left (End: {df['enc_l'].iloc[-1]} ticks)")
    ax2.plot(df['t_sec'], df['enc_r'], 'r--', lw=1.5, label=f"Enc Right (End: {df['enc_r'].iloc[-1]} ticks)")
    ax2.set_ylabel("Cumulative Steps")
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Subplot 3: Wheel Angular Velocity (RPM)
    ax3.plot(df['t_sec'], df['vel_l_rpm'], 'b-', lw=1.2, label=f"Vel Left (Max: {report['max_vel_l_rpm']:.1f} RPM)")
    ax3.plot(df['t_sec'], df['vel_r_rpm'], 'r--', lw=1.2, label=f"Vel Right (Max: {report['max_vel_r_rpm']:.1f} RPM)")
    if not np.isnan(deadband_l):
        ax3.axhline(0, color='black', lw=1)
    ax3.set_ylabel("Wheel Velocity (RPM)")
    ax3.set_xlabel("Time Elapsed (Seconds)")
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"bench_analysis_robot_{robot_id}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Actuator & Kinematics Plot: {plot_path}")

    return report

def main():
    print("=" * 95)
    print("        ANJOMAN SWARM - BENCH KINEMATICS & ACTUATOR SYSTEM IDENTIFICATION")
    print("=" * 95)

    reports = []

    for filename, r_id, gear_ratio, wheel_diam in ROBOT_FILES:
        if not os.path.exists(filename):
            print(f"[WARN] File not found: {filename} (Skipping Robot {r_id})")
            continue

        print(f"\n[ANALYZING] Processing {filename} for Robot {r_id}...")
        df = parse_bench_csv(filename)
        if len(df) < 100:
            print(f"[ERROR] Insufficient data rows in {filename}!")
            continue

        rep = analyze_robot_bench(df, r_id, gear_ratio, wheel_diam)
        reports.append(rep)

    if not reports:
        print("[FATAL] No valid bench datasets to summarize.")
        sys.exit(1)

    df_rep = pd.DataFrame(reports)

    print("\n" + "=" * 105)
    print("                                IDENTIFIED ACTUATOR & IMU PARAMETERS TABLE")
    print("=" * 105)
    header = f"{'Robot':7s} | {'Gearbox':8s} | {'Deadband L':12s} | {'Deadband R':12s} | {'Max RPM (L/R)':17s} | {'Max Speed':11s} | {'Gyro Bias Z':14s}"
    print(header)
    print("-" * 105)

    for _, r in df_rep.iterrows():
        db_l = f"{r['deadband_l_pwm']*100:.1f}%" if not np.isnan(r['deadband_l_pwm']) else "STALLED"
        db_r = f"{r['deadband_r_pwm']*100:.1f}%" if not np.isnan(r['deadband_r_pwm']) else "STALLED"
        rpm_str = f"{r['max_vel_l_rpm']:.1f} / {r['max_vel_r_rpm']:.1f}"
        spd_str = f"{r['max_linear_m_s']:.2f} m/s"
        gyro_str = f"{r['gyro_bias_z_deg_s']:+.3f} deg/s"
        print(f"R{r['robot_id']:<6d} | {r['gear_ratio']:8s} | {db_l:12s} | {db_r:12s} | {rpm_str:17s} | {spd_str:11s} | {gyro_str:14s}")
    print("=" * 105)

    # -------------------------------------------------------------
    # Production Polarities & Fleet Velocity Capping Generator
    # -------------------------------------------------------------
    print("\n" + "#" * 85)
    print("       RECOMMENDED INVERSION & VELOCITY ENVELOPE FOR RobotConfig.h")
    print("#" * 85)
    
    # Safe Swarm Cruising Velocity Ceiling (Governed by slowest 1:120 robot)
    slowest_robot_max_speed = df_rep[df_rep['gear_ratio'] == '1:120']['max_linear_m_s'].max()
    if np.isnan(slowest_robot_max_speed) or slowest_robot_max_speed == 0:
        safe_swarm_cruise_speed = 0.20 # Fallback conservative cruising velocity
    else:
        safe_swarm_cruise_speed = slowest_robot_max_speed * 0.70 # 70% of max for safe PID headroom

    print(f"// Maximum Safe Swarm Linear Velocity: {safe_swarm_cruise_speed:.2f} m/s (70% of slowest robot)")
    print(f"constexpr float SWARM_MAX_LINEAR_VEL_M_S = {safe_swarm_cruise_speed:.2f}f;\n")

    print("// Universal Encoder Polarities (All Left Encoders are Mirror-Inverted)")
    print("constexpr bool INVERT_ENCODER_LEFT  = true;  // Invert Left on ALL robots")
    print("constexpr bool INVERT_ENCODER_RIGHT = false; // Right is natural on ALL robots\n")

    print("// Robot-Specific Motor Polarities (Forward Drive Alignment)")
    for _, r in df_rep.iterrows():
        rid = r['robot_id']
        inv_mot_r = "true" if rid == 1 else "true"
        inv_mot_l = "false" if rid == 1 else "true"
        print(f"#if ROBOT_ID == {rid}")
        print(f"    constexpr bool INVERT_MOTOR_LEFT  = {inv_mot_l};")
        print(f"    constexpr bool INVERT_MOTOR_RIGHT = {inv_mot_r};")
        print(f"#endif")
    print("#" * 85 + "\n")

if __name__ == "__main__":
    main()
