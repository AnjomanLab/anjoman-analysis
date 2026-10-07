#!/usr/bin/env python3
"""
Anjoman Swarm: Motor Actuator & Wheel Kinematics System Identification
Parses 8-column CSV (TimeMs,Phase,PwmL,PwmR,StepsL,StepsR,RpmL,RpmR)
Extracts Deadband thresholds, Max RPM, PWM-to-Velocity gains, and Asymmetry.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "plots_motor_sysid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

# Search paths for all 4 robots (handles various common filenames)
ROBOT_METADATA = [
    (1, ["r1_motor.csv", "motor_r1.csv", "bench_r1.csv"], 48.0, 0.050),
    (2, ["r2_motor.csv", "motor_r2.csv", "bench_r2.csv"], 120.0, 0.055),
    (3, ["r3_motor.csv", "motor_r3.csv", "bench_r3.csv"], 120.0, 0.055),
    (4, ["r4_motor.csv", "motor_r4.csv", "bench_r4.csv"], 120.0, 0.055),
]

def load_motor_csv(filepath):
    """Robust parser stripping PlatformIO timestamps and parsing 8 columns."""
    records = []
    line_pattern = re.compile(
        r'(?:.*>\s*)?(\d+)\s*,\s*(\d+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([-\d]+)\s*,\s*([-\d]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)'
    )

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if "TimeMs" in line or not line or "=" in line or "[" in line:
                continue
            match = line_pattern.search(line)
            if match:
                try:
                    records.append({
                        'time_ms': int(match.group(1)),
                        'phase': int(match.group(2)),
                        'pwm_l': float(match.group(3)),
                        'pwm_r': float(match.group(4)),
                        'steps_l': int(match.group(5)),
                        'steps_r': int(match.group(6)),
                        'rpm_l': float(match.group(7)),
                        'rpm_r': float(match.group(8)),
                    })
                except ValueError:
                    continue

    return pd.DataFrame(records)

def analyze_motor_dataset(df, robot_id, gear_ratio, wheel_diam_m):
    """Identifies actuator characteristics for a single robot."""
    df['t_sec'] = (df['time_ms'] - df['time_ms'].iloc[0]) / 1000.0

    # 1. Identify Deadband (Phase 1 for Left, Phase 3 for Right)
    # Stiction breakout point: first point where |RPM| > 2.0
    p1 = df[(df['phase'] == 1) & (df['pwm_l'] > 0.05)]
    moving_l = p1[p1['rpm_l'].abs() > 2.0]
    deadband_l = moving_l['pwm_l'].iloc[0] if len(moving_l) > 0 else np.nan

    p3 = df[(df['phase'] == 3) & (df['pwm_r'] > 0.05)]
    moving_r = p3[p3['rpm_r'].abs() > 2.0]
    deadband_r = moving_r['pwm_r'].iloc[0] if len(moving_r) > 0 else np.nan

    # 2. Extract Discrete Speed Staircase Averages (Phase 2 & Phase 4)
    # Target steps: 30%, 50%, 70%, 100%
    step_pwms = [0.30, 0.50, 0.70, 1.00]
    l_steps = {}
    r_steps = {}

    for target_pwm in step_pwms:
        # Sample steady-state middle section of step
        sub_l = df[(df['phase'] == 2) & (df['pwm_l'].round(2) == target_pwm)]
        if len(sub_l) > 10:
            l_steps[int(target_pwm*100)] = sub_l['rpm_l'].iloc[5:-5].abs().mean()
        else:
            l_steps[int(target_pwm*100)] = 0.0

        sub_r = df[(df['phase'] == 4) & (df['pwm_r'].round(2) == target_pwm)]
        if len(sub_r) > 10:
            r_steps[int(target_pwm*100)] = sub_r['rpm_r'].iloc[5:-5].abs().mean()
        else:
            r_steps[int(target_pwm*100)] = 0.0

    # Max Wheel Speeds at 100% PWM
    max_rpm_l = l_steps.get(100, 0.0)
    max_rpm_r = r_steps.get(100, 0.0)
    max_rad_s_l = (max_rpm_l * 2.0 * np.pi) / 60.0
    max_rad_s_r = (max_rpm_r * 2.0 * np.pi) / 60.0

    wheel_radius_m = wheel_diam_m / 2.0
    max_linear_v_l = max_rad_s_l * wheel_radius_m
    max_linear_v_r = max_rad_s_r * wheel_radius_m
    max_robot_speed = min(max_linear_v_l, max_linear_v_r)

    # 3. Asymmetry Ratio at 60% PWM (Phase 5: Synchronous drive)
    p5 = df[df['phase'] == 5]
    if len(p5) > 10:
        mean_p5_l = p5['rpm_l'].iloc[5:-5].abs().mean()
        mean_p5_r = p5['rpm_r'].iloc[5:-5].abs().mean()
        balance_ratio = mean_p5_l / mean_p5_r if mean_p5_r > 0 else 1.0
    else:
        balance_ratio = 1.0

    summary = {
        'Robot': f"Robot {robot_id}",
        'Gearbox': f"1:{gear_ratio:.0f}",
        'Deadband L': f"{deadband_l*100:.1f}%" if not np.isnan(deadband_l) else "STALL",
        'Deadband R': f"{deadband_r*100:.1f}%" if not np.isnan(deadband_r) else "STALL",
        'Max RPM (L/R)': f"{max_rpm_l:.1f} / {max_rpm_r:.1f}",
        'Max Linear': f"{max_robot_speed:.2f} m/s",
        'Asymmetry (L/R)': f"{balance_ratio:.2f}",
        'Final Ticks (L/R)': f"{df['steps_l'].iloc[-1]} / {df['steps_r'].iloc[-1]}",
        'deadband_l_val': deadband_l,
        'deadband_r_val': deadband_r,
        'max_speed_val': max_robot_speed
    }

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # Plot 1: Command Profile vs Time
    ax1.plot(df['t_sec'], df['pwm_l'], 'b-', lw=1.5, label='PWM Left')
    ax1.plot(df['t_sec'], df['pwm_r'], 'r--', lw=1.5, label='PWM Right')
    ax1.set_ylabel("PWM Duty (-1 to +1)")
    ax1.set_title(f"Robot {robot_id} (Gearbox 1:{gear_ratio:.0f}) — Actuator Excitation Signals", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Wheel RPM vs Time
    ax2.plot(df['t_sec'], df['rpm_l'], 'b-', lw=1.2, label='Wheel Left RPM')
    ax2.plot(df['t_sec'], df['rpm_r'], 'r-', lw=1.2, label='Wheel Right RPM')
    ax2.axhline(0, color='black', lw=1)
    ax2.set_ylabel("Wheel Speed (RPM)")
    ax2.set_title(f"Measured Wheel Angular Velocities (Max: {max(max_rpm_l, max_rpm_r):.1f} RPM)", fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Actuator Characteristic Linearity (RPM vs PWM)
    pwms_plot = [0.30, 0.50, 0.70, 1.00]
    rpm_l_plot = [l_steps[int(p*100)] for p in pwms_plot]
    rpm_r_plot = [r_steps[int(p*100)] for p in pwms_plot]

    ax3.plot(pwms_plot, rpm_l_plot, 'bs-', lw=2, markersize=8, label='Left Motor Curve')
    ax3.plot(pwms_plot, rpm_r_plot, 'ro-', lw=2, markersize=8, label='Right Motor Curve')
    if not np.isnan(deadband_l):
        ax3.axvline(deadband_l, color='blue', linestyle=':', label=f'Deadband L ({deadband_l*100:.1f}%)')
    if not np.isnan(deadband_r):
        ax3.axvline(deadband_r, color='red', linestyle=':', label=f'Deadband R ({deadband_r*100:.1f}%)')
    ax3.set_xlabel("PWM Duty Cycle")
    ax3.set_ylabel("Steady-State Speed (RPM)")
    ax3.set_title("Actuator Linear Velocity Characteristic (Stiction & Gain Slope)", fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"robot_{robot_id}_actuator_sysid.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Actuator SysID Plot: {plot_path}")

    return summary

def main():
    print("=" * 105)
    print("         ANJOMAN SWARM - DRIVETRAIN & ACTUATOR SYSTEM IDENTIFICATION REPORT")
    print("=" * 105)

    all_summaries = []

    for r_id, filenames, gear_ratio, wheel_diam in ROBOT_METADATA:
        found_file = None
        for fn in filenames:
            if os.path.exists(fn):
                found_file = fn
                break
        
        if not found_file:
            continue

        print(f"\n[PROCESSING] Analyzing {found_file} (Robot {r_id})...")
        df = load_motor_csv(found_file)
        if len(df) < 50:
            print(f"[ERROR] Insufficient data in {found_file}!")
            continue

        res = analyze_motor_dataset(df, r_id, gear_ratio, wheel_diam)
        all_summaries.append(res)

    if not all_summaries:
        print("[FATAL] No motor data files found to analyze!")
        sys.exit(1)

    df_report = pd.DataFrame(all_summaries)
    
    print("\n" + "=" * 110)
    print("                               IDENTIFIED ACTUATOR CHARACTERISTICS TABLE")
    print("=" * 110)
    cols_to_print = ['Robot', 'Gearbox', 'Deadband L', 'Deadband R', 'Max RPM (L/R)', 'Max Linear', 'Asymmetry (L/R)', 'Final Ticks (L/R)']
    print(df_report[cols_to_print].to_string(index=False))
    print("=" * 110)

    # -------------------------------------------------------------
    # Production C++ Generator for RobotConfig.h
    # -------------------------------------------------------------
    print("\n" + "#" * 85)
    print("       EXACT POLARITY & KINEMATIC MACROS FOR include/RobotConfig.h")
    print("#" * 85)
    print("// Universal Encoder Polarities (Mirror-Inversion Corrected)")
    print("constexpr bool INVERT_ENCODER_LEFT  = true;  // Left is Inverted on ALL robots")
    print("constexpr bool INVERT_ENCODER_RIGHT = false; // Right is Natural on ALL robots\n")

    print("// Fully Validated Motor Direction Polarities (Forward Drive Alignment)")
    print("#if ROBOT_ID == 1")
    print("    constexpr bool INVERT_MOTOR_LEFT  = false;")
    print("    constexpr bool INVERT_MOTOR_RIGHT = true;  // Right Inverted")
    print("#elif ROBOT_ID == 2")
    print("    constexpr bool INVERT_MOTOR_LEFT  = true;  // Both Inverted")
    print("    constexpr bool INVERT_MOTOR_RIGHT = true;")
    print("#elif ROBOT_ID == 3")
    print("    constexpr bool INVERT_MOTOR_LEFT  = false;")
    print("    constexpr bool INVERT_MOTOR_RIGHT = true;  // Right Inverted (Discovered)")
    print("#elif ROBOT_ID == 4")
    print("    constexpr bool INVERT_MOTOR_LEFT  = true;  // Left Inverted (Discovered)")
    print("    constexpr bool INVERT_MOTOR_RIGHT = false;")
    print("#endif\n")

    # Calculate safe cruising velocity limit (70% of slowest 1:120 robot)
    speeds = [r['max_speed_val'] for r in all_summaries if r['max_speed_val'] > 0]
    safe_speed = min(speeds) * 0.70 if speeds else 0.20
    print(f"// Swarm Homogeneous Cruising Speed Limit (Governed by slowest 1:120 robot)")
    print(f"constexpr float SWARM_MAX_SPEED_M_S = {safe_speed:.2f}f; // Safe saturation headroom")
    print("#" * 85 + "\n")

if __name__ == "__main__":
    main()
