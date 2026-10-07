#!/usr/bin/env python3
"""
Anjoman Swarm: Deep Actuator & Kinematics System Identification Engine
Processes 12-column microsecond telemetry (TimeUs,Phase,PwmCmdL,PwmCmdR,ModeL,ModeR,RawAngL,RawAngR,DeltaL,DeltaR,StepsL,StepsR)
Extracts Deadbands, Transfer Function Gains (K), Time Constants (tau), Asymmetry Trim, and Coast-Down Damping.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

OUTPUT_DIR = "plots_sysid_deep"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

ROBOT_METADATA = [
    (1, ["sysid_r1.csv", "r1_motor.csv", "bench_r1.csv"], 48.0, 0.050, 0.135),
    (2, ["sysid_r2.csv", "r2_motor.csv", "bench_r2.csv"], 120.0, 0.055, 0.125),
    (3, ["sysid_r3.csv", "r3_motor.csv", "bench_r3.csv"], 120.0, 0.055, 0.125),
    (4, ["sysid_r4.csv", "r4_motor.csv", "bench_r4.csv"], 120.0, 0.055, 0.125),
]

def parse_sysid_csv(filepath):
    """Robust parser stripping PlatformIO timestamps and parsing 12 columns."""
    records = []
    pattern = re.compile(
        r'(?:.*>\s*)?(\d+)\s*,\s*(\d+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([-\d]+)\s*,\s*([-\d]+)\s*,\s*([-\d]+)\s*,\s*([-\d]+)'
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
                        'phase': int(match.group(2)),
                        'pwm_l': float(match.group(3)),
                        'pwm_r': float(match.group(4)),
                        'mode_l': int(match.group(5)),
                        'mode_r': int(match.group(6)),
                        'raw_ang_l': int(match.group(7)),
                        'raw_ang_r': int(match.group(8)),
                        'delta_l': int(match.group(9)),
                        'delta_r': int(match.group(10)),
                        'steps_l': int(match.group(11)),
                        'steps_r': int(match.group(12))
                    })
                except ValueError:
                    continue

    df = pd.DataFrame(records)
    return df

def first_order_step_response(t, k_gain, tau):
    """First-order transfer function step response: y(t) = K * (1 - exp(-t / tau))"""
    return k_gain * (1.0 - np.exp(-t / np.maximum(tau, 1e-4)))

def analyze_actuator_deep(df, robot_id, gear_ratio, wheel_diam_m, track_width_m):
    """Comprehensive physical and dynamic parameter identification."""
    wheel_radius_m = wheel_diam_m / 2.0
    cpr = 4096.0

    # 1. Microsecond Time Base and Angular Velocity (rad/s and RPM)
    df['t_sec'] = (df['time_us'] - df['time_us'].iloc[0]) * 1e-6
    dt = df['t_sec'].diff().replace(0, np.nan)
    
    # Instantaneous raw velocities (rad/s)
    df['vel_l_rad_s'] = (df['delta_l'] / cpr) * (2.0 * np.pi) / dt
    df['vel_r_rad_s'] = (df['delta_r'] / cpr) * (2.0 * np.pi) / dt

    # Smooth velocities with 9-point median filter to reject discrete quantization
    df['vel_l_smooth'] = df['vel_l_rad_s'].rolling(window=9, center=True).median().fillna(0)
    df['vel_r_smooth'] = df['vel_r_rad_s'].rolling(window=9, center=True).median().fillna(0)
    df['rpm_l'] = (df['vel_l_smooth'] * 60.0) / (2.0 * np.pi)
    df['rpm_r'] = (df['vel_r_smooth'] * 60.0) / (2.0 * np.pi)

    # -------------------------------------------------------------
    # 2. Phase 0: Standstill Sensor Noise & Jitter
    # -------------------------------------------------------------
    p0 = df[df['phase'] == 0]
    jitter_ticks_l = p0['delta_l'].std()
    jitter_ticks_r = p0['delta_r'].std()

    # -------------------------------------------------------------
    # 3. Phase 1 & 2: Left Motor Static Parameters (Fwd & Rev)
    # -------------------------------------------------------------
    # Forward Staircase (Phase 1)
    p1 = df[df['phase'] == 1]
    levels_fwd = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80, 1.00]
    l_rpm_fwd = []
    
    for lvl in levels_fwd:
        sub = p1[(p1['pwm_l'] >= lvl - 0.02) & (p1['pwm_l'] <= lvl + 0.02)]
        val = sub['rpm_l'].iloc[len(sub)//3 : -5].mean() if len(sub) > 15 else 0.0
        l_rpm_fwd.append(max(0.0, val))

    # Deadband Left Forward: First level with RPM > 1.5
    db_l_fwd = 1.0
    for lvl, rpm in zip(levels_fwd, l_rpm_fwd):
        if rpm > 1.5:
            db_l_fwd = lvl
            break

    # Reverse Staircase (Phase 2)
    p2 = df[df['phase'] == 2]
    levels_rev = [-0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.60, -0.80, -1.00]
    l_rpm_rev = []
    
    for lvl in levels_rev:
        sub = p2[(p2['pwm_l'] >= lvl - 0.02) & (p2['pwm_l'] <= lvl + 0.02)]
        val = sub['rpm_l'].iloc[len(sub)//3 : -5].mean() if len(sub) > 15 else 0.0
        l_rpm_rev.append(min(0.0, val))

    db_l_rev = -1.0
    for lvl, rpm in zip(levels_rev, l_rpm_rev):
        if abs(rpm) > 1.5:
            db_l_rev = abs(lvl)
            break

    # Gain calculation (RPM / PWM) in linear region (PWM 0.4 to 1.0)
    slope_l_fwd, _ = np.polyfit(levels_fwd[5:], l_rpm_fwd[5:], 1)
    slope_l_rev, _ = np.polyfit(np.abs(levels_rev[5:]), np.abs(l_rpm_rev[5:]), 1)

    # -------------------------------------------------------------
    # 4. Phase 4 & 5: Right Motor Static Parameters (Fwd & Rev)
    # -------------------------------------------------------------
    p4 = df[df['phase'] == 4]
    r_rpm_fwd = []
    for lvl in levels_fwd:
        sub = p4[(p4['pwm_r'] >= lvl - 0.02) & (p4['pwm_r'] <= lvl + 0.02)]
        val = sub['rpm_r'].iloc[len(sub)//3 : -5].mean() if len(sub) > 15 else 0.0
        r_rpm_fwd.append(max(0.0, val))

    db_r_fwd = 1.0
    for lvl, rpm in zip(levels_fwd, r_rpm_fwd):
        if rpm > 1.5:
            db_r_fwd = lvl
            break

    p5 = df[df['phase'] == 5]
    r_rpm_rev = []
    for lvl in levels_rev:
        sub = p5[(p5['pwm_r'] >= lvl - 0.02) & (p5['pwm_r'] <= lvl + 0.02)]
        val = sub['rpm_r'].iloc[len(sub)//3 : -5].mean() if len(sub) > 15 else 0.0
        r_rpm_rev.append(min(0.0, val))

    db_r_rev = -1.0
    for lvl, rpm in zip(levels_rev, r_rpm_rev):
        if abs(rpm) > 1.5:
            db_r_rev = abs(lvl)
            break

    slope_r_fwd, _ = np.polyfit(levels_fwd[5:], r_rpm_fwd[5:], 1)
    slope_r_rev, _ = np.polyfit(np.abs(levels_rev[5:]), np.abs(r_rpm_rev[5:]), 1)

    # -------------------------------------------------------------
    # 5. Phase 3 & 6: Dynamic Step Response & Time Constant (tau)
    # -------------------------------------------------------------
    # Fit step 0 -> 0.70 PWM for Left Motor
    tau_l_ms = np.nan
    p3_step = df[(df['phase'] == 3) & (df['pwm_l'] > 0.65)]
    if len(p3_step) > 20:
        t_step = p3_step['t_sec'].values - p3_step['t_sec'].iloc[0]
        v_step = p3_step['vel_l_smooth'].values
        try:
            popt, _ = curve_fit(first_order_step_response, t_step[:40], v_step[:40], p0=[v_step.max(), 0.1], maxfev=2000)
            tau_l_ms = popt[1] * 1000.0
        except Exception:
            tau_l_ms = np.nan

    # Fit step 0 -> 0.70 PWM for Right Motor
    tau_r_ms = np.nan
    p6_step = df[(df['phase'] == 6) & (df['pwm_r'] > 0.65)]
    if len(p6_step) > 20:
        t_step = p6_step['t_sec'].values - p6_step['t_sec'].iloc[0]
        v_step = p6_step['vel_r_smooth'].values
        try:
            popt, _ = curve_fit(first_order_step_response, t_step[:40], v_step[:40], p0=[v_step.max(), 0.1], maxfev=2000)
            tau_r_ms = popt[1] * 1000.0
        except Exception:
            tau_r_ms = np.nan

    # -------------------------------------------------------------
    # 6. Phase 7: Differential Asymmetry & Open-Loop Yaw Veering
    # -------------------------------------------------------------
    p7 = df[df['phase'] == 7]
    sync_levels = [0.30, 0.50, 0.70, 0.90]
    asymmetry_ratios = []
    open_loop_yaw_rates_deg = []

    for sl in sync_levels:
        sub = p7[(p7['pwm_l'] >= sl - 0.05) & (p7['pwm_l'] <= sl + 0.05)]
        if len(sub) > 20:
            vl = sub['vel_l_smooth'].iloc[15:-5].mean()
            vr = sub['vel_r_smooth'].iloc[15:-5].mean()
            ratio = vl / vr if vr != 0 else 1.0
            asymmetry_ratios.append(ratio)
            # Yaw rate = (r * vr - r * vl) / track_width
            yaw_rate_rad_s = (wheel_radius_m * vr - wheel_radius_m * vl) / track_width_m
            open_loop_yaw_rates_deg.append(np.degrees(yaw_rate_rad_s))

    mean_asymmetry = np.mean(asymmetry_ratios) if asymmetry_ratios else 1.0
    mean_yaw_veering = np.mean(open_loop_yaw_rates_deg) if open_loop_yaw_rates_deg else 0.0

    # -------------------------------------------------------------
    # 7. Max Absolute Velocities
    # -------------------------------------------------------------
    max_rpm_l = max(l_rpm_fwd[-1], abs(l_rpm_rev[-1]))
    max_rpm_r = max(r_rpm_fwd[-1], abs(r_rpm_rev[-1]))
    max_linear_v = min(max_rpm_l, max_rpm_r) * (2.0 * np.pi / 60.0) * wheel_radius_m

    metrics = {
        'robot_id': robot_id,
        'gearbox': f"1:{gear_ratio:.0f}",
        'db_l_fwd': db_l_fwd,
        'db_r_fwd': db_r_fwd,
        'db_l_rev': db_l_rev,
        'db_r_rev': db_r_rev,
        'gain_l_rpm': slope_l_fwd,
        'gain_r_rpm': slope_r_fwd,
        'tau_l_ms': tau_l_ms,
        'tau_r_ms': tau_r_ms,
        'max_rpm_l': max_rpm_l,
        'max_rpm_r': max_rpm_r,
        'max_linear_m_s': max_linear_v,
        'asymmetry_ratio': mean_asymmetry,
        'open_loop_yaw_deg_s': mean_yaw_veering,
        'standstill_jitter_ticks': max(jitter_ticks_l, jitter_ticks_r)
    }

    # -------------------------------------------------------------
    # Visualizations per Robot
    # -------------------------------------------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot A: Full Actuator Velocity Trajectory
    ax1.plot(df['t_sec'], df['rpm_l'], 'b-', lw=1.2, label=f'Wheel Left (Max: {max_rpm_l:.1f} RPM)')
    ax1.plot(df['t_sec'], df['rpm_r'], 'r--', lw=1.2, label=f'Wheel Right (Max: {max_rpm_r:.1f} RPM)')
    ax1.axhline(0, color='black', lw=1)
    ax1.set_title(f"Robot {robot_id} (1:{gear_ratio:.0f}) — 160s Actuator Velocity Response", fontweight='bold')
    ax1.set_ylabel("Angular Velocity (RPM)")
    ax1.set_xlabel("Time (Seconds)")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot B: Static Transfer Function Curves (Fwd & Rev)
    all_pwms_l = levels_rev[::-1] + [0.0] + levels_fwd
    all_rpms_l = l_rpm_rev[::-1] + [0.0] + l_rpm_fwd
    all_pwms_r = levels_rev[::-1] + [0.0] + levels_fwd
    all_rpms_r = r_rpm_rev[::-1] + [0.0] + r_rpm_fwd

    ax2.plot(all_pwms_l, all_rpms_l, 'bs-', lw=2, label='Left Actuator Curve')
    ax2.plot(all_pwms_r, all_rpms_r, 'ro-', lw=2, label='Right Actuator Curve')
    ax2.axvline(0, color='black', lw=1)
    ax2.axhline(0, color='black', lw=1)
    ax2.set_title("Static Characteristic Curves & Deadband Zones", fontweight='bold')
    ax2.set_xlabel("PWM Command Duty (-1 to +1)")
    ax2.set_ylabel("Steady-State Velocity (RPM)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plot C: Transient Step Response Zoom
    if len(p3_step) > 20:
        ax3.plot(t_step[:50] * 1000.0, v_step[:50], 'b.-', label=f'Left Measured (tau ~ {tau_l_ms:.1f} ms)')
    if len(p6_step) > 20:
        t_step_r = (p6_step['t_sec'].values - p6_step['t_sec'].iloc[0]) * 1000.0
        ax3.plot(t_step_r[:50], p6_step['vel_r_smooth'].values[:50], 'r.--', label=f'Right Measured (tau ~ {tau_r_ms:.1f} ms)')
    ax3.set_title("Transient Dynamic Step Response (0 to 0.70 PWM)", fontweight='bold')
    ax3.set_xlabel("Time from Step Onset (ms)")
    ax3.set_ylabel("Angular Velocity (rad/s)")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Plot D: Differential Speed Asymmetry Across Synchronous PWM Steps
    if asymmetry_ratios:
        ax4.bar([f"{int(s*100)}%" for s in sync_levels], asymmetry_ratios, color='#2ca02c', width=0.5)
        ax4.axhline(1.0, color='red', linestyle='--', lw=1.5, label='Ideal Symmetry (1.0)')
        ax4.set_title(f"Differential Balance Ratio (VL / VR) | Open-Loop Veering: {mean_yaw_veering:+.1f} deg/s", fontweight='bold')
        ax4.set_ylabel("Speed Ratio (VL / VR)")
        ax4.set_xlabel("PWM Step Level")
        ax4.set_ylim([0.7, 1.3])
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"robot_{robot_id}_deep_sysid.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[SAVED] Deep SysID Plot: {fig_path}")

    return metrics

def main():
    print("=" * 115)
    print("           ANJOMAN SWARM - DEEP ACTUATOR & KINEMATICS IDENTIFICATION REPORT")
    print("=" * 115)

    reports = []

    for r_id, filenames, gear_ratio, wheel_diam, track_width in ROBOT_METADATA:
        found_file = None
        for fn in filenames:
            if os.path.exists(fn):
                found_file = fn
                break

        if not found_file:
            print(f"[SKIP] No data file found for Robot {r_id}.")
            continue

        print(f"\n[ANALYZING] Processing {found_file} for Robot {r_id}...")
        df = parse_sysid_csv(found_file)
        if len(df) < 200:
            print(f"[ERROR] Insufficient samples in {found_file}!")
            continue

        rep = analyze_actuator_deep(df, r_id, gear_ratio, wheel_diam, track_width)
        reports.append(rep)

    if not reports:
        print("[FATAL] No datasets were processed successfully.")
        sys.exit(1)

    df_rep = pd.DataFrame(reports)

    # -------------------------------------------------------------
    # Table 1: Static Characteristics & Friction Identification
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("                      TABLE 1: STATIC ACTUATOR CHARACTERISTICS & FRICTION IDENTIFICATION")
    print("=" * 115)
    header1 = f"{'Robot':7s} | {'Gearbox':8s} | {'Deadband L (Fwd/Rev)':23s} | {'Deadband R (Fwd/Rev)':23s} | {'Gain K_L (RPM/PWM)':20s} | {'Gain K_R (RPM/PWM)':20s}"
    print(header1)
    print("-" * 115)
    for _, r in df_rep.iterrows():
        db_l = f"{r['db_l_fwd']*100:.1f}% / {r['db_l_rev']*100:.1f}%"
        db_r = f"{r['db_r_fwd']*100:.1f}% / {r['db_r_rev']*100:.1f}%"
        g_l = f"{r['gain_l_rpm']:.1f} RPM"
        g_r = f"{r['gain_r_rpm']:.1f} RPM"
        print(f"R{r['robot_id']:<6d} | {r['gearbox']:8s} | {db_l:23s} | {db_r:23s} | {g_l:20s} | {g_r:20s}")
    print("=" * 115)

    # -------------------------------------------------------------
    # Table 2: Dynamic Transient & Time Constants (First-Order Model)
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("                      TABLE 2: DYNAMIC TRANSIENT & TIME CONSTANTS (tau_m)")
    print("=" * 115)
    header2 = f"{'Robot':7s} | {'Time Constant tau_L':22s} | {'Time Constant tau_R':22s} | {'Max No-Load Speed (RPM)':25s} | {'Max Linear Speed':18s}"
    print(header2)
    print("-" * 115)
    for _, r in df_rep.iterrows():
        tl = f"{r['tau_l_ms']:.1f} ms" if np.isfinite(r['tau_l_ms']) else "N/A"
        tr = f"{r['tau_r_ms']:.1f} ms" if np.isfinite(r['tau_r_ms']) else "N/A"
        rpm_str = f"{r['max_rpm_l']:.1f} / {r['max_rpm_r']:.1f}"
        lin_spd = f"{r['max_linear_m_s']:.2f} m/s ({r['max_linear_m_s']*100:.0f} cm/s)"
        print(f"R{r['robot_id']:<6d} | {tl:22s} | {tr:22s} | {rpm_str:25s} | {lin_spd:18s}")
    print("=" * 115)

    # -------------------------------------------------------------
    # Table 3: Differential Symmetry & Open-Loop Steering Drift
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("                      TABLE 3: DIFFERENTIAL BALANCE & OPEN-LOOP VEERING DRIFT")
    print("=" * 115)
    header3 = f"{'Robot':7s} | {'Asymmetry Ratio (VL/VR)':25s} | {'Open-Loop Yaw Drift (deg/s)':30s} | {'Standstill Jitter':20s}"
    print(header3)
    print("-" * 115)
    for _, r in df_rep.iterrows():
        asym = f"{r['asymmetry_ratio']:.4f} (Ideal: 1.000)"
        yaw = f"{r['open_loop_yaw_deg_s']:+.2f} deg/s"
        jit = f"{r['standstill_jitter_ticks']:.2f} ticks"
        print(f"R{r['robot_id']:<6d} | {asym:25s} | {yaw:30s} | {jit:20s}")
    print("=" * 115)

    # -------------------------------------------------------------
    # Fleet-Wide Swarm Envelope Summary
    # -------------------------------------------------------------
    speeds = [r['max_linear_m_s'] for r in reports if r['max_linear_m_s'] > 0]
    slowest_max = min(speeds) if speeds else 0.25
    safe_swarm_cruise = slowest_max * 0.70 # 70% headroom for closed-loop steering

    print("\n" + "#" * 90)
    print("                        SWARM-WIDE OPERATIONAL ENVELOPE")
    print("#" * 90)
    print(f"1. Slowest Robot In Fleet          : {slowest_max:.2f} m/s (Governed by 1:120 Gearbox)")
    print(f"2. Safe Swarm Cruising Speed Ceiling: {safe_swarm_cruise:.2f} m/s ({safe_swarm_cruise*100:.1f} cm/s)")
    print(f"3. Cruising Velocity Headroom       : 30% Reserved for PID Steering Corrections")
    print("#" * 90 + "\n")

if __name__ == "__main__":
    main()
