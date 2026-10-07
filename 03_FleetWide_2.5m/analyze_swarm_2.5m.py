#!/usr/bin/env python3
"""
Anjoman Swarm: 5-Pair High-Resolution UWB Metrology & Drift Analysis
Excludes R2-R3 to prevent matrix corruption.
Visualizes Raw vs Calibrated data, Thermal Drift, and Global Antenna Delays.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE_DIR, "2.5_experiment")
PLOTS_DIR = os.path.join(EXP_DIR, "plots_5pairs")
os.makedirs(PLOTS_DIR, exist_ok=True)

GROUND_TRUTH_M = 2.500
SPEED_OF_LIGHT = 299792458.0
TIME_UNIT_SEC  = 0.000000000015650040064103

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

# ------------------------------------------------------------------------------
# 2. DATA PARSING
# ------------------------------------------------------------------------------
VALID_PAIRS = {
    'R1-R2': 'r1_r2_2.5.txt',
    'R1-R3': 'r1_r3_2.5.txt',
    'R1-R4': 'r1_r4_2.5.txt',
    'R2-R4': 'r2_r4_2.5.txt',
    'R3-R4': 'r3_r4_2.5.txt'
}

def parse_log(filepath):
    if not os.path.exists(filepath):
        return None
    records = []
    # Matches the 13 columns output
    pattern = re.compile(
        r'(?:.*>\s*)?(\d+)\s*,\s*(\d+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)'
    )
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "timestamp" in line: continue
            m = pattern.search(line.strip())
            if m:
                # Basic sanity filter: discard obvious hardware reset glitches (e.g. > 100m)
                raw_d = float(m.group(3))
                cal_d = float(m.group(4))
                if 10.0 < raw_d < 80.0:
                    records.append({
                        'timestamp_ms': int(m.group(1)),
                        'range_raw_m': raw_d,
                        'range_cal_m': cal_d,
                        'rssi_dbm': float(m.group(5)),
                        'temp_esp_1': float(m.group(12)),
                        'temp_esp_2': float(m.group(13))
                    })
    
    df = pd.DataFrame(records)
    if not df.empty:
        df['rel_time_s'] = (df['timestamp_ms'] - df['timestamp_ms'].iloc[0]) / 1000.0
        df['temp_diff'] = df['temp_esp_1'] - df['temp_esp_2']
    return df

datasets = {}
print("[INFO] Loading 5 healthy pairwise datasets...")
for pair, fname in VALID_PAIRS.items():
    fpath = os.path.join(EXP_DIR, fname)
    df = parse_log(fpath)
    if df is not None and not df.empty:
        datasets[pair] = df
        print(f"  -> {pair}: {len(df)} samples")

# ------------------------------------------------------------------------------
# 3. STATISTICAL SUMMARY
# ------------------------------------------------------------------------------
summary_rows = []
raw_offsets_for_matrix = []

for pair, df_p in datasets.items():
    # Calculate drift on CALIBRATED data to see residual firmware error
    slope_cal, _, r_val_cal, _, _ = stats.linregress(df_p['rel_time_s'], df_p['range_cal_m'])
    
    # Calculate drift on RAW data
    slope_raw, _, _, _, _ = stats.linregress(df_p['rel_time_s'], df_p['range_raw_m'])
    
    mean_raw = df_p['range_raw_m'].mean()
    raw_offset = mean_raw - GROUND_TRUTH_M
    raw_offsets_for_matrix.append(raw_offset)

    summary_rows.append({
        'Pair': pair,
        'Raw Mean (m)': np.round(mean_raw, 3),
        'Calibrated Mean (m)': np.round(df_p['range_cal_m'].mean(), 3),
        'Calibrated Noise (cm)': np.round(df_p['range_cal_m'].std() * 100.0, 2),
        'Raw Drift (mm/s)': np.round(slope_raw * 1000.0, 3),
        'Calibrated Drift (mm/s)': np.round(slope_cal * 1000.0, 3),
        'RSSI (dBm)': np.round(df_p['rssi_dbm'].mean(), 1)
    })

df_summary = pd.DataFrame(summary_rows)
print("\n" + "="*90)
print(df_summary.to_string(index=False))
print("="*90)

# ------------------------------------------------------------------------------
# 4. GLOBAL ANTENNA DELAY MATRIX (5 Equations, 4 Unknowns)
# ------------------------------------------------------------------------------
# Pairs order: R1-R2, R1-R3, R1-R4, R2-R4, R3-R4
A_matrix = np.array([
    [1, 1, 0, 0],  # 1-2
    [1, 0, 1, 0],  # 1-3
    [1, 0, 0, 1],  # 1-4
    [0, 1, 0, 1],  # 2-4
    [0, 0, 1, 1]   # 3-4
], dtype=float)

D_vector = np.array(raw_offsets_for_matrix)

# Solve OLS: T_ant = (A^T A)^-1 A^T D
T_ant_meters, residuals, rank, s = np.linalg.lstsq(A_matrix, D_vector, rcond=None)
T_ant_ticks = (T_ant_meters / (2.0 * SPEED_OF_LIGHT * TIME_UNIT_SEC)).astype(int) + 16436

print("\n[GLOBAL OLS] Recovered Individual Antenna Delays:")
for i in range(4):
    print(f"  Robot {i+1}: {T_ant_meters[i]:.4f} m | DW1000 Register Ticks: {T_ant_ticks[i]}")

# ------------------------------------------------------------------------------
# 5. HIGH-RES VISUALIZATIONS
# ------------------------------------------------------------------------------
print("\n[INFO] Generating Plots...")
colors = sns.color_palette("Set1", n_colors=5)

# FIGURE 1: CALIBRATED DISTANCE (Performance of current Firmware)
fig1, axes1 = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
for idx, (pair, df_p) in enumerate(datasets.items()):
    ax = axes1[idx]
    ax.plot(df_p['rel_time_s'], df_p['range_cal_m'], color=colors[idx], alpha=0.8, lw=1)
    
    mean_cal = df_p['range_cal_m'].mean()
    # Dynamic zoom: +/- 15 cm from the mean to expose micro-drift and noise
    ax.set_ylim(mean_cal - 0.15, mean_cal + 0.15)
    
    # Ground Truth Line (if visible in this window)
    ax.axhline(GROUND_TRUTH_M, color='k', linestyle='--', lw=2, label="Ground Truth (2.5m)")
    
    # Trendline
    z = np.polyfit(df_p['rel_time_s'], df_p['range_cal_m'], 1)
    p = np.poly1d(z)
    ax.plot(df_p['rel_time_s'], p(df_p['rel_time_s']), 'r-', lw=2, label=f"Residual Drift: {z[0]*1000:.2f} mm/s")
    
    ax.set_title(f"{pair} - Firmware Calibrated Output", fontweight='bold')
    ax.set_ylabel("Distance (m)")
    ax.legend(loc='upper right', fontsize=8)

axes1[-1].set_xlabel("Elapsed Time (Seconds)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig1_calibrated_performance.png"))
plt.close()

# FIGURE 2: RAW DISTANCE DRIFT (Hardware Level Behavior)
fig2, axes2 = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
for idx, (pair, df_p) in enumerate(datasets.items()):
    ax = axes2[idx]
    # Mean centering the raw data to purely visualize the drift slope
    centered_raw = df_p['range_raw_m'] - df_p['range_raw_m'].mean()
    ax.plot(df_p['rel_time_s'], centered_raw * 100.0, color='gray', alpha=0.7, lw=1) # Plot in cm
    
    z = np.polyfit(df_p['rel_time_s'], centered_raw * 100.0, 1)
    p = np.poly1d(z)
    ax.plot(df_p['rel_time_s'], p(df_p['rel_time_s']), 'b-', lw=2, label=f"Hardware Drift: {z[0]*10:.2f} mm/s")
    
    ax.set_title(f"{pair} - Raw Hardware Drift Profile (Mean Centered)", fontweight='bold')
    ax.set_ylabel("Deviation (cm)")
    ax.legend(loc='upper right', fontsize=8)

axes2[-1].set_xlabel("Elapsed Time (Seconds)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig2_raw_hardware_drift.png"))
plt.close()

# FIGURE 3: THERMAL CORRELATION (Drift vs Temp Difference)
fig3, ax3 = plt.subplots(figsize=(10, 6))
for idx, (pair, df_p) in enumerate(datasets.items()):
    # Remove mean to compare pure thermal correlation
    err_centered = df_p['range_cal_m'] - df_p['range_cal_m'].mean()
    ax3.scatter(df_p['temp_diff'], err_centered * 1000.0, s=4, alpha=0.3, color=colors[idx], label=pair)

ax3.set_title("Impact of Thermal Asymmetry on Ranging Error", fontweight='bold')
ax3.set_xlabel("Temperature Difference (T_Initiator - T_Responder) [°C]")
ax3.set_ylabel("Ranging Deviation from Mean (mm)")
ax3.legend(frameon=True, markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig3_thermal_correlation.png"))
plt.close()

# FIGURE 4: NOISE & DRIFT BAR CHARTS (Fixing Seaborn warnings)
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=df_summary, x='Pair', y='Calibrated Noise (cm)', hue='Pair', ax=ax4a, palette='viridis', legend=False)
ax4a.set_title("Calibrated Measurement Noise (1-Sigma)", fontweight='bold')
ax4a.set_ylabel("Standard Deviation (cm)")

sns.barplot(data=df_summary, x='Pair', y='Calibrated Drift (mm/s)', hue='Pair', ax=ax4b, palette='coolwarm', legend=False)
ax4b.set_title("Residual Temporal Drift in Firmware", fontweight='bold')
ax4b.set_ylabel("Drift Slope (mm/second)")
ax4b.axhline(0, color='k', lw=1)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig4_noise_and_drift_summary.png"))
plt.close()

print("[COMPLETE] High-res plots saved to 'plots_5pairs' directory.")
