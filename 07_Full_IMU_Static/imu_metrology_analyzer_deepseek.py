#!/usr/bin/env python3
"""
Anjoman Swarm: Deep IMU (Bosch BMI160) Metrology & Allan Deviation Analyzer (Enhanced)
Processes imu_r1_static.csv to imu_r4_static.csv (30000 samples @ 100 Hz per robot).
Extracts full 6-DoF biases, Allan variance, thermal sensitivity, and EKF noise covariances.
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit, OptimizeWarning

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

GRAVITY_STANDARD = 9.80665  # m/s²

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
    return pd.DataFrame(records)

def compute_overlapping_allan_deviation(data, dt, max_tau=60.0):
    """
    Computes Overlapping Allan Deviation (OADEV) using sliding window averages.
    data: 1D array, dt: sampling interval in seconds.
    Returns taus and adevs.
    """
    n = len(data)
    max_m = int(max_tau / dt)
    # Generate log-spaced cluster sizes
    m_values = np.unique(np.logspace(0, np.log10(min(max_m, n//2)), 50).astype(int))
    m_values = m_values[(m_values > 0) & (m_values < n // 2)]

    taus = m_values * dt
    adevs = []

    # Use cumulative sum for efficient moving average
    cumsum = np.cumsum(data)
    for m in m_values:
        # Compute block averages using sliding window
        block_avg = (cumsum[m:] - cumsum[:-m]) / m
        # Differences between adjacent block averages
        diff = block_avg[1:] - block_avg[:-1]
        avar = 0.5 * np.mean(diff ** 2)
        adevs.append(np.sqrt(avar))

    return np.array(taus), np.array(adevs)

def fit_allan_parameters(taus, adevs):
    """
    Fit ARW and bias instability from Allan deviation curve.
    Model: sqrt(A^2/tau + B^2) where A = ARW, B = bias instability.
    """
    mask = taus > 0
    t = taus[mask]
    a = adevs[mask]
    if len(t) < 3:
        return np.nan, np.nan

    def model(tau, A, B):
        return np.sqrt(A**2 / tau + B**2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            p0 = [a[0] * np.sqrt(t[0]), a.min()]
            popt, _ = curve_fit(model, t, a, p0=p0, maxfev=10000)
            return popt[0], popt[1]
        except (RuntimeError, ValueError):
            # Fallback: use value at 1s for ARW and minimum for bias instability
            idx_1s = np.argmin(np.abs(t - 1.0))
            return a[idx_1s], a.min()

def analyze_single_robot(df, robot_id):
    """Full 6-DoF analysis for a stationary dataset."""
    df['t_sec'] = (df['time_us'] - df['time_us'].iloc[0]) * 1e-6
    dt = np.median(df['t_sec'].diff().dropna())
    fs = 1.0 / dt

    # Convert gyro to rad/s
    for axis in ['x', 'y', 'z']:
        df[f'g{axis}_rad_s'] = np.radians(df[f'g{axis}_dps'])

    # Estimate initial roll/pitch from accelerometer (small angles)
    roll_rad = np.arctan2(df['ay_mps2'].mean(), df['az_mps2'].mean())
    pitch_rad = np.arctan2(-df['ax_mps2'].mean(), np.sqrt(df['ay_mps2'].mean()**2 + df['az_mps2'].mean()**2))
    roll_deg = np.degrees(roll_rad)
    pitch_deg = np.degrees(pitch_rad)

    # Rotation matrix to correct gravity projection
    R = np.array([
        [np.cos(pitch_rad), 0, -np.sin(pitch_rad)],
        [np.sin(roll_rad)*np.sin(pitch_rad), np.cos(roll_rad), np.sin(roll_rad)*np.cos(pitch_rad)],
        [np.cos(roll_rad)*np.sin(pitch_rad), -np.sin(roll_rad), np.cos(roll_rad)*np.cos(pitch_rad)]
    ])
    gravity_body = R.T @ np.array([0, 0, GRAVITY_STANDARD])  # gravity in body frame

    # Corrected accelerometer biases (subtract gravity projection)
    ba_x = df['ax_mps2'].mean() - gravity_body[0]
    ba_y = df['ay_mps2'].mean() - gravity_body[1]
    ba_z = df['az_mps2'].mean() - gravity_body[2]

    # Accelerometer scale factor (based on norm)
    accel_norm = np.sqrt(df['ax_mps2']**2 + df['ay_mps2']**2 + df['az_mps2']**2)
    scale_factor_accel = accel_norm.mean() / GRAVITY_STANDARD

    # Gyro biases (all axes)
    bg_x = df['gx_dps'].mean()
    bg_y = df['gy_dps'].mean()
    bg_z = df['gz_dps'].mean()

    # Gyro noise (std dev)
    sigma_gx = df['gx_dps'].std()
    sigma_gy = df['gy_dps'].std()
    sigma_gz = df['gz_dps'].std()

    # Thermal sensitivity for each axis (linear regression vs temp_imu)
    thermal = {}
    for ch in ['gx', 'gy', 'gz', 'ax', 'ay', 'az']:
        slope, intercept, r, _, _ = stats.linregress(df['temp_imu'], df[f'{ch}_dps' if ch.startswith('g') else f'{ch}_mps2'])
        thermal[ch] = {'slope': slope, 'r2': r**2}

    # Allan deviations for each gyro axis
    allan = {}
    for axis in ['x', 'y', 'z']:
        taus, adevs = compute_overlapping_allan_deviation(df[f'g{axis}_dps'].values, dt)
        arw, bias_inst = fit_allan_parameters(taus, adevs)
        allan[axis] = {'taus': taus, 'adevs': adevs, 'arw': arw, 'bias_instability': bias_inst}

    metrics = {
        'robot_id': robot_id,
        'samples': len(df),
        'duration_s': df['t_sec'].iloc[-1],
        'freq_hz': fs,
        # Gyro biases (dps and rad/s)
        'gyro_bias': {'x_dps': bg_x, 'y_dps': bg_y, 'z_dps': bg_z,
                      'x_rad': np.radians(bg_x), 'y_rad': np.radians(bg_y), 'z_rad': np.radians(bg_z)},
        'gyro_noise_std_rad': {'x': np.radians(sigma_gx), 'y': np.radians(sigma_gy), 'z': np.radians(sigma_gz)},
        # Accelerometer biases (corrected)
        'accel_bias': {'x': ba_x, 'y': ba_y, 'z': ba_z},
        'accel_noise_std': {'x': df['ax_mps2'].std(), 'y': df['ay_mps2'].std(), 'z': df['az_mps2'].std()},
        'accel_scale_factor': scale_factor_accel,
        'roll_deg': roll_deg,
        'pitch_deg': pitch_deg,
        'thermal': thermal,
        'allan': allan,
        'temp_imu_mean': df['temp_imu'].mean(),
        'temp_esp_mean': df['temp_esp'].mean(),
    }
    return metrics, df

def print_tables(all_metrics):
    """Print all tables in console."""
    # Table 1: Gyro biases and noises
    print("\n" + "="*120)
    print("TABLE 1: GYROSCOPE BIASES & NOISE (ALL AXES)")
    print("="*120)
    header = f"{'Robot':6s} | {'Bias X (dps)':12s} | {'Bias Y (dps)':12s} | {'Bias Z (dps)':12s} | {'σ X (rad/s)':12s} | {'σ Y (rad/s)':12s} | {'σ Z (rad/s)':12s}"
    print(header)
    print("-"*120)
    for m in all_metrics:
        r = f"R{m['robot_id']}"
        b = m['gyro_bias']
        s = m['gyro_noise_std_rad']
        print(f"{r:6s} | {b['x_dps']:+10.4f} | {b['y_dps']:+10.4f} | {b['z_dps']:+10.4f} | {s['x']:12.6f} | {s['y']:12.6f} | {s['z']:12.6f}")
    print("="*120)

    # Table 2: Accelerometer biases
    print("\n" + "="*120)
    print("TABLE 2: ACCELEROMETER BIASES (CORRECTED FOR TILT) & SCALE FACTOR")
    print("="*120)
    header = f"{'Robot':6s} | {'Bias X (m/s²)':14s} | {'Bias Y (m/s²)':14s} | {'Bias Z (m/s²)':14s} | {'Scale Factor':14s} | {'Roll/Pitch (deg)':18s}"
    print(header)
    print("-"*120)
    for m in all_metrics:
        r = f"R{m['robot_id']}"
        a = m['accel_bias']
        sf = m['accel_scale_factor']
        tilt = f"{m['roll_deg']:+.2f} / {m['pitch_deg']:+.2f}"
        print(f"{r:6s} | {a['x']:+12.4f} | {a['y']:+12.4f} | {a['z']:+12.4f} | {sf:14.6f} | {tilt:18s}")
    print("="*120)

    # Table 3: Thermal sensitivity
    print("\n" + "="*120)
    print("TABLE 3: THERMAL DRIFT SENSITIVITY (SLOPE per °C) & R²")
    print("="*120)
    header = f"{'Robot':6s} | {'Gx (mdps/°C)':14s} | {'Gy (mdps/°C)':14s} | {'Gz (mdps/°C)':14s} | {'Ax (µg/°C)':12s} | {'Ay (µg/°C)':12s} | {'Az (µg/°C)':12s}"
    print(header)
    print("-"*120)
    for m in all_metrics:
        r = f"R{m['robot_id']}"
        th = m['thermal']
        gx = th['gx']['slope'] * 1000  # mdps
        gy = th['gy']['slope'] * 1000
        gz = th['gz']['slope'] * 1000
        ax = th['ax']['slope'] * 1e6   # µg
        ay = th['ay']['slope'] * 1e6
        az = th['az']['slope'] * 1e6
        print(f"{r:6s} | {gx:+12.2f} | {gy:+12.2f} | {gz:+12.2f} | {ax:+10.2f} | {ay:+10.2f} | {az:+10.2f}")
    print("="*120)

    # Table 4: Allan variance parameters
    print("\n" + "="*120)
    print("TABLE 4: ALLAN VARIANCE PARAMETERS FOR GYRO AXES (ARW & BIAS INSTABILITY)")
    print("="*120)
    header = f"{'Robot':6s} | {'ARW X (deg/√h)':16s} | {'BI X (deg/h)':12s} | {'ARW Y (deg/√h)':16s} | {'BI Y (deg/h)':12s} | {'ARW Z (deg/√h)':16s} | {'BI Z (deg/h)':12s}"
    print(header)
    print("-"*120)
    for m in all_metrics:
        r = f"R{m['robot_id']}"
        al = m['allan']
        arw_x = al['x']['arw'] * 60  # convert deg/s/√Hz to deg/√h
        bi_x = al['x']['bias_instability'] * 3600  # deg/s to deg/h
        arw_y = al['y']['arw'] * 60
        bi_y = al['y']['bias_instability'] * 3600
        arw_z = al['z']['arw'] * 60
        bi_z = al['z']['bias_instability'] * 3600
        print(f"{r:6s} | {arw_x:16.4f} | {bi_x:12.4f} | {arw_y:16.4f} | {bi_y:12.4f} | {arw_z:16.4f} | {bi_z:12.4f}")
    print("="*120)

def plot_results(all_metrics, dataframes):
    """Generate comprehensive diagnostic plots."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    n_robots = len(all_metrics)

    # Figure 1: Gyro Z trajectories
    plt.figure(figsize=(14, 6))
    for idx, m in enumerate(all_metrics):
        sub_df = dataframes[m['robot_id']]
        plt.plot(sub_df['t_sec']/60, sub_df['gz_dps'], lw=0.5, alpha=0.7, color=colors[idx],
                 label=f"R{m['robot_id']} (bias={m['gyro_bias']['z_dps']:+.3f} dps, σ={np.degrees(m['gyro_noise_std_rad']['z']):.3f} dps)")
        plt.axhline(m['gyro_bias']['z_dps'], color=colors[idx], linestyle='--', lw=1.2)
    plt.title("Stationary Gyroscope Z Bias Stability (5 min)")
    plt.xlabel("Time (min)")
    plt.ylabel("Angular rate (deg/s)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_gyro_z_trajectories.png", dpi=300)
    plt.close()

    # Figure 2: Allan deviation for all gyro axes (one subplot per robot)
    fig, axes = plt.subplots(1, n_robots, figsize=(5*n_robots, 5), sharey=True)
    if n_robots == 1:
        axes = [axes]
    for idx, (ax, m) in enumerate(zip(axes, all_metrics)):
        for axis, col in zip(['x','y','z'], ['#1f77b4','#2ca02c','#d62728']):
            taus = m['allan'][axis]['taus']
            adevs = m['allan'][axis]['adevs']
            ax.loglog(taus, adevs, 'o-', color=col, markersize=3, label=f'{axis.upper()} (ARW={m["allan"][axis]["arw"]:.3f})')
        ax.set_title(f"Robot {m['robot_id']} Allan Deviation")
        ax.set_xlabel("τ (s)")
        ax.set_ylabel("σ(τ) [deg/s]" if idx==0 else "")
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_allan_all_axes.png", dpi=300)
    plt.close()

    # Figure 3: Thermal correlation for Gyro Z
    fig, axes = plt.subplots(1, n_robots, figsize=(5*n_robots, 4), sharey=True)
    if n_robots == 1:
        axes = [axes]
    for idx, (ax, m) in enumerate(zip(axes, all_metrics)):
        sub_df = dataframes[m['robot_id']]
        sns.scatterplot(data=sub_df.iloc[::10], x='temp_imu', y='gz_dps', ax=ax, color=colors[idx], alpha=0.2, s=6)
        sns.regplot(data=sub_df.iloc[::20], x='temp_imu', y='gz_dps', ax=ax, scatter=False, color='black',
                    line_kws={'lw':1.5, 'linestyle':'--'})
        slope_mdps = m['thermal']['gz']['slope'] * 1000
        ax.set_title(f"R{m['robot_id']} (slope={slope_mdps:+.1f} mdps/°C)")
        ax.set_xlabel("BMI160 Temp (°C)")
        ax.set_ylabel("Gyro Z (deg/s)" if idx==0 else "")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_thermal_gyro_z.png", dpi=300)
    plt.close()

    # Figure 4: Accelerometer norm boxplot
    plt.figure(figsize=(10,5))
    norm_data = [np.sqrt(dataframes[m['robot_id']]['ax_mps2']**2 +
                         dataframes[m['robot_id']]['ay_mps2']**2 +
                         dataframes[m['robot_id']]['az_mps2']**2) for m in all_metrics]
    labels = [f"R{m['robot_id']}" for m in all_metrics]
    # استفاده از plt.boxplot ساده به جای sns.boxplot
    plt.boxplot(norm_data, labels=labels, showmeans=True)
    plt.axhline(GRAVITY_STANDARD, color='red', linestyle='--', label=f'Standard g = {GRAVITY_STANDARD:.3f} m/s²')
    plt.title("Accelerometer Gravity Norm Distribution")
    plt.ylabel("Norm (m/s²)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_accel_norm_boxplot.png", dpi=300)
    plt.close()

    print(f"\n[SAVED] Plots in {OUTPUT_DIR}/")

def main():
    print("="*120)
    print("         ANJOMAN SWARM - ENHANCED 6-DOF IMU METROLOGY ANALYZER")
    print("="*120)

    all_metrics = []
    dataframes = {}

    for filename, r_id in TARGET_FILES:
        if not os.path.exists(filename):
            print(f"[SKIP] File not found: {filename}")
            continue
        print(f"[PARSING] {filename} ...")
        df = parse_imu_csv(filename)
        if len(df) < 500:
            print(f"[ERROR] Insufficient data in {filename}")
            continue
        met, clean_df = analyze_single_robot(df, r_id)
        all_metrics.append(met)
        dataframes[r_id] = clean_df

    if not all_metrics:
        print("[FATAL] No valid data.")
        sys.exit(1)

    print_tables(all_metrics)
    plot_results(all_metrics, dataframes)

    # Output EKF parameters
    print("\n" + "#"*90)
    print("        EKF PROCESS NOISE (Q) AND MEASUREMENT NOISE (R) SUGGESTIONS")
    print("#"*90)
    for m in all_metrics:
        rid = m['robot_id']
        print(f"Robot {rid}:")
        # Gyro process noise (continuous: rad²/s²/Hz → discrete: *dt)
        q_gyro = np.diag([m['gyro_noise_std_rad']['x']**2,
                          m['gyro_noise_std_rad']['y']**2,
                          m['gyro_noise_std_rad']['z']**2])
        q_accel = np.diag([m['accel_noise_std']['x']**2,
                           m['accel_noise_std']['y']**2,
                           m['accel_noise_std']['z']**2])
        print(f"  Q_gyro (continuous, rad²/s²): {q_gyro}")
        print(f"  Q_accel (continuous, m²/s⁴): {q_accel}")
        print(f"  Note: For discrete EKF, multiply by dt = {1/m['freq_hz']:.4f} s")
    print("#"*90)

if __name__ == "__main__":
    main()
