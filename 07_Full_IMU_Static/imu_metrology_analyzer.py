#!/usr/bin/env python3
"""
Anjoman Swarm: Deep IMU (Bosch BMI160) Metrology & Allan Deviation Analyzer
Processes imu_r1_static.csv to imu_r4_static.csv (30,000 samples @ 100 Hz per robot).
Extracts 6-DoF Biases, Allan Variance, Thermal Sensitivity, and EKF Noise Covariances.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

OUTPUT_DIR = "plots_imu_sysid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

TARGET_FILES = [
    ("imu_r1_static.csv", 1),
    ("imu_r2_static.csv", 2),
    ("imu_r3_static.csv", 3),
    ("imu_r4_static.csv", 4),
]

GRAVITY_STANDARD = 9.80665 # Standard Earth gravity in m/s^2

def parse_imu_csv(filepath):
    """Robust parser stripping PlatformIO timestamps and extracting 10 columns."""
    records = []
    line_pattern = re.compile(
        r'(?:.*>\s*)?(\d+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*(\d+)'
    )

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if "TimeUs" in line or not line or "=" in line or "[" in line:
                continue
            match = line_pattern.search(line)
            if match:
                try:
                    records.append({
                        'time_us': int(match.group(1)),
                        'gx_dps': float(match.group(2)),
                        'gy_dps': float(match.group(3)),
                        'gz_dps': float(match.group(4)),
                        'ax_mps2': float(match.group(5)),
                        'ay_mps2': float(match.group(6)),
                        'az_mps2': float(match.group(7)),
                        'temp_imu': float(match.group(8)),
                        'temp_esp': float(match.group(9)),
                        'status': int(match.group(10))
                    })
                except ValueError:
                    continue

    df = pd.DataFrame(records)
    return df

def compute_allan_deviation(data, dt, max_tau=60.0):
    """
    Computes Overlapping Allan Deviation (OADEV) for inertial noise profiling.
    data: 1D array of gyro rates in deg/s or rad/s
    dt: sampling interval in seconds (0.01s for 100 Hz)
    """
    n = len(data)
    # Logarithmically spaced integration cluster factors
    max_m = int(max_tau / dt)
    m_values = np.unique(np.logspace(0, np.log10(max_m / 2), 40).astype(int))
    m_values = m_values[(m_values > 0) & (m_values < n // 2)]

    taus = m_values * dt
    adevs = []

    for m in m_values:
        # Cluster averages
        clusters = np.add.reduceat(data, np.arange(0, n - (n % m), m)) / m
        # Allan variance of cluster differences
        diff = np.diff(clusters)
        avar = 0.5 * np.mean(diff ** 2)
        adevs.append(np.sqrt(avar))

    return np.array(taus), np.array(adevs)

def analyze_single_robot(df, robot_id):
    """Analyzes a 30,000-sample stationary dataset for a single robot."""
    df['t_sec'] = (df['time_us'] - df['time_us'].iloc[0]) * 1e-6
    dt = np.median(df['t_sec'].diff().dropna())
    sampling_freq = 1.0 / dt

    # Convert Gyro to rad/s for EKF
    df['gz_rad_s'] = np.radians(df['gz_dps'])

    # Total Accelerometer Norm
    df['accel_norm'] = np.sqrt(df['ax_mps2']**2 + df['ay_mps2']**2 + df['az_mps2']**2)

    # Static Tilt Angles (Roll and Pitch in Degrees)
    roll_deg = np.degrees(np.arctan2(df['ay_mps2'].mean(), df['az_mps2'].mean()))
    pitch_deg = np.degrees(np.arctan2(-df['ax_mps2'].mean(), np.sqrt(df['ay_mps2'].mean()**2 + df['az_mps2'].mean()**2)))

    # Gyro Biases (Mean @ Rest)
    bg_x = df['gx_dps'].mean()
    bg_y = df['gy_dps'].mean()
    bg_z = df['gz_dps'].mean()
    bg_z_rad = df['gz_rad_s'].mean()

    # Gyro Noise (White Noise Standard Deviation)
    sigma_gx = df['gx_dps'].std()
    sigma_gy = df['gy_dps'].std()
    sigma_gz = df['gz_dps'].std()
    sigma_gz_rad = df['gz_rad_s'].std()

    # Accelerometer Biases (Ax and Ay should be 0, Az should be 9.81)
    ba_x = df['ax_mps2'].mean()
    ba_y = df['ay_mps2'].mean()
    ba_z = df['az_mps2'].mean() - GRAVITY_STANDARD
    sigma_ax = df['ax_mps2'].std()
    sigma_ay = df['ay_mps2'].std()
    sigma_az = df['az_mps2'].std()

    # Thermal Sensitivity: Gyro Z vs BMI160 Silicon Temperature
    slope_temp, intercept, r_temp, _, _ = stats.linregress(df['temp_imu'], df['gz_dps'])
    thermal_drift_per_c = slope_temp # (deg/s) / degC

    # Allan Deviation for Gyro Z
    taus, adevs = compute_allan_deviation(df['gz_dps'].values, dt)
    # Angle Random Walk (ARW) at tau = 1 second
    idx_1s = np.argmin(np.abs(taus - 1.0))
    arw_estimate = adevs[idx_1s] # deg/s / sqrt(Hz)

    metrics = {
        'robot_id': robot_id,
        'samples': len(df),
        'duration_s': df['t_sec'].iloc[-1],
        'freq_hz': sampling_freq,
        'gyro_bias_z_dps': bg_z,
        'gyro_bias_z_rad': bg_z_rad,
        'gyro_noise_std_rad': sigma_gz_rad,
        'accel_bias_x': ba_x,
        'accel_bias_y': ba_y,
        'accel_bias_z': ba_z,
        'accel_noise_std': df['accel_norm'].std(),
        'accel_norm_mean': df['accel_norm'].mean(),
        'roll_deg': roll_deg,
        'pitch_deg': pitch_deg,
        'thermal_drift_slope': thermal_drift_per_c,
        'thermal_r2': r_temp**2,
        'arw_1s': arw_estimate,
        'taus': taus,
        'adevs': adevs,
        'temp_imu_mean': df['temp_imu'].mean(),
        'temp_esp_mean': df['temp_esp'].mean(),
    }

    return metrics, df

def main():
    print("=" * 115)
    print("         ANJOMAN SWARM - 6-DOF IMU (BMI160) METROLOGY & ALLAN VARIANCE ANALYZER")
    print("=" * 115)

    all_metrics = []
    dataframes = {}

    for filename, r_id in TARGET_FILES:
        if not os.path.exists(filename):
            print(f"[SKIP] File not found: {filename:20s} (Robot {r_id})")
            continue

        print(f"[PARSING] Loading {filename:20s} for Robot {r_id}...")
        df = parse_imu_csv(filename)
        if len(df) < 500:
            print(f"[ERROR] Insufficient data points in {filename}!")
            continue

        met, clean_df = analyze_single_robot(df, r_id)
        all_metrics.append(met)
        dataframes[r_id] = clean_df

    if not all_metrics:
        print("[FATAL] No valid IMU stationary files to analyze!")
        sys.exit(1)

    # -------------------------------------------------------------
    # Table 1: Stationary Gyroscope Biases & EKF Noise Variances
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("                    TABLE 1: GYROSCOPE BIASES, NOISE (1-SIGMA), AND EKF VARIANCES")
    print("=" * 115)
    header1 = f"{'Robot':7s} | {'Samples':8s} | {'Gyro Z Bias (deg/s)':20s} | {'Gyro Z Bias (rad/s)':20s} | {'Noise Std (rad/s)':18s} | {'Discrete Var Q_gyro':20s}"
    print(header1)
    print("-" * 115)
    for m in all_metrics:
        rid = f"R{m['robot_id']}"
        samp = f"{m['samples']:,}"
        b_dps = f"{m['gyro_bias_z_dps']:+.4f} deg/s"
        b_rad = f"{m['gyro_bias_z_rad']:+.6f} rad/s"
        sig_rad = f"{m['gyro_noise_std_rad']:.6f} rad/s"
        q_var = f"{m['gyro_noise_std_rad']**2:.8f} (rad/s)^2"
        print(f"{rid:<7s} | {samp:>8s} | {b_dps:20s} | {b_rad:20s} | {sig_rad:18s} | {q_var:20s}")
    print("=" * 115)

    # -------------------------------------------------------------
    # Table 2: Accelerometer Biases, Gravity Norm & Body Tilt
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("                    TABLE 2: ACCELEROMETER BIAS VECTORS & REST TILT ANGLES")
    print("=" * 115)
    header2 = f"{'Robot':7s} | {'Bias X (m/s^2)':16s} | {'Bias Y (m/s^2)':16s} | {'Bias Z (Dev from G)':20s} | {'Total ||a|| (m/s^2)':20s} | {'Body Roll / Pitch':20s}"
    print(header2)
    print("-" * 115)
    for m in all_metrics:
        rid = f"R{m['robot_id']}"
        bx = f"{m['accel_bias_x']:+.4f}"
        by = f"{m['accel_bias_y']:+.4f}"
        bz = f"{m['accel_bias_z']:+.4f}"
        norm_a = f"{m['accel_norm_mean']:.4f} m/s^2"
        tilt = f"{m['roll_deg']:+.2f}° / {m['pitch_deg']:+.2f}°"
        print(f"{rid:<7s} | {bx:16s} | {by:16s} | {bz:20s} | {norm_a:20s} | {tilt:20s}")
    print("=" * 115)

    # -------------------------------------------------------------
    # Table 3: Thermal Drift Sensitivity & Allan Random Walk
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("                    TABLE 3: THERMAL BIAS DRIFT SENSITIVITY & ALLAN RANDOM WALK")
    print("=" * 115)
    header3 = f"{'Robot':7s} | {'BMI160 Temp (Mean)':20s} | {'ESP32 Temp (Mean)':19s} | {'Thermal Drift Slope':22s} | {'R^2 Temp':10s} | {'ARW (Noise Density)':22s}"
    print(header3)
    print("-" * 115)
    for m in all_metrics:
        rid = f"R{m['robot_id']}"
        t_imu = f"{m['temp_imu_mean']:.2f} °C"
        t_esp = f"{m['temp_esp_mean']:.2f} °C"
        t_slope = f"{m['thermal_drift_slope']*1000:+.3f} (mdps)/°C"
        r2_t = f"{m['thermal_r2']:.3f}"
        arw = f"{m['arw_1s']:.4f} deg/s/√Hz"
        print(f"{rid:<7s} | {t_imu:20s} | {t_esp:19s} | {t_slope:22s} | {r2_t:10s} | {arw:22s}")
    print("=" * 115)

    # -------------------------------------------------------------
    # Visualizations (Saved to plots_imu_sysid/)
    # -------------------------------------------------------------
    print("\n[PLOTTING] Generating High-Resolution Diagnostic Figures...")

    # Figure 1: Gyro Z Trajectory across all 4 robots
    plt.figure(figsize=(14, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, m in enumerate(all_metrics):
        rid = m['robot_id']
        sub_df = dataframes[rid]
        plt.plot(sub_df['t_sec'] / 60.0, sub_df['gz_dps'], lw=0.5, alpha=0.7, color=colors[idx],
                 label=f"Robot {rid} (Bias: {m['gyro_bias_z_dps']:+.3f} deg/s | σ: {m['gyro_noise_std_rad']*180/np.pi:.3f} deg/s)")
        plt.axhline(m['gyro_bias_z_dps'], color=colors[idx], linestyle='--', lw=1.2)
    plt.title("Stationary Gyroscope Z Bias Stability & Trajectory over 5 Minutes", fontweight='bold')
    plt.xlabel("Time Elapsed (Minutes)")
    plt.ylabel("Angular Velocity (deg/s)")
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    p1 = os.path.join(OUTPUT_DIR, "fig1_gyro_z_trajectories_5min.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[SAVED] {p1}")

    # Figure 2: Overlapping Allan Deviation (OADEV) Log-Log Curves
    plt.figure(figsize=(11, 6))
    for idx, m in enumerate(all_metrics):
        rid = m['robot_id']
        plt.loglog(m['taus'], m['adevs'], 'o-', lw=1.5, markersize=3, color=colors[idx],
                   label=f"Robot {rid} Gyro Z (ARW: {m['arw_1s']:.3f} deg/s/√Hz)")
    plt.title("Overlapping Allan Deviation σ(τ) — Inertial Noise & Bias Instability", fontweight='bold')
    plt.xlabel("Cluster Time τ (Seconds)")
    plt.ylabel("Allan Deviation σ(τ) [deg/s]")
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    p2 = os.path.join(OUTPUT_DIR, "fig2_allan_deviation_gyro_z.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[SAVED] {p2}")

    # Figure 3: Thermal Correlation (Gyro Z vs BMI160 Silicon Temperature)
    fig, axes = plt.subplots(1, len(all_metrics), figsize=(4 * len(all_metrics), 4), sharey=True)
    if len(all_metrics) == 1: axes = [axes]
    for idx, (ax, m) in enumerate(zip(axes, all_metrics)):
        rid = m['robot_id']
        sub_df = dataframes[rid]
        sns.scatterplot(data=sub_df.iloc[::10], x='temp_imu', y='gz_dps', ax=ax, color=colors[idx], alpha=0.2, s=6)
        sns.regplot(data=sub_df.iloc[::20], x='temp_imu', y='gz_dps', ax=ax, scatter=False, color='black',
                    line_kws={'lw': 1.5, 'linestyle': '--'})
        ax.set_title(f"Robot {rid} (Slope: {m['thermal_drift_slope']*1000:+.1f} mdps/°C)", fontweight='bold')
        ax.set_xlabel("BMI160 Temp (°C)")
        ax.set_ylabel("Gyro Z (deg/s)" if idx == 0 else "")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    p3 = os.path.join(OUTPUT_DIR, "fig3_gyro_vs_silicon_temperature.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"[SAVED] {p3}")

    # Figure 4: Total Accelerometer Gravity Norm Distribution
    plt.figure(figsize=(10, 5))
    norm_data = [dataframes[m['robot_id']]['accel_norm'] for m in all_metrics]
    labels = [f"Robot {m['robot_id']}" for m in all_metrics]
    #sns.boxplot(data=norm_data, palette=colors[:len(all_metrics)])
    sns.boxplot(data=norm_data)
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.axhline(GRAVITY_STANDARD, color='red', linestyle='--', lw=1.5, label=f'Standard Gravity ({GRAVITY_STANDARD:.3f} m/s²)')
    plt.title("Accelerometer Static Gravity Norm ||a|| across Robots", fontweight='bold')
    plt.ylabel("Measured Norm (m/s²)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    p4 = os.path.join(OUTPUT_DIR, "fig4_accel_gravity_norm_boxplot.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"[SAVED] {p4}")

    print("\n" + "#" * 90)
    print("        EKF INERTIAL PROCESS NOISE MATRIX VALUES (100 Hz Discrete Time)")
    print("#" * 90)
    for m in all_metrics:
        print(f"Robot {m['robot_id']}:")
        print(f"  constexpr float GYRO_BIAS_Z_RAD_S  = {m['gyro_bias_z_rad']:+.6f}f;")
        print(f"  constexpr float VAR_GYRO_Z_RAD_S2  = {m['gyro_noise_std_rad']**2:.8f}f; // Q_gyro diagonal")
        print(f"  constexpr float VAR_ACCEL_MPS2     = {m['accel_noise_std']**2:.8f}f; // Q_accel diagonal")
    print("#" * 90 + "\n")

if __name__ == "__main__":
    main()
