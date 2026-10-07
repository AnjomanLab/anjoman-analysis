#!/usr/bin/env python3
"""
Anjoman Swarm: Direct 6-Edge Pairwise Calibration Engine for 3.000m Benchmark
Extracts exact empirical offsets and linear de-drifting parameters for all pairs.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

GROUND_TRUTH_M = 3.000
OUTPUT_DIR = "plots_pairwise_3m"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

TARGET_FILES = [
    ("r1_r2_3m.txt", "R1-R2"),
    ("r1_r3_3m.txt", "R1-R3"),
    ("r1_r4_3m.txt", "R1-R4"),
    ("r2_r3_3m.txt", "R2-R3"),
    ("r2_r4_3m.txt", "R2-R4"),
    ("r3_r4_3m.txt", "R3-R4"),
]

def analyze_and_calibrate_3m():
    results = []
    calibrated_frames = []

    print("=" * 95)
    print("           ANJOMAN SWARM - 6-EDGE PAIRWISE METROLOGY & CALIBRATION (3.000 m)")
    print("=" * 95)
    print(f"{'Pair':8s} | {'Samples':7s} | {'Raw Mean':10s} | {'Identified Offset':18s} | {'Drift (mm/s)':13s} | {'Noise Std':10s}")
    print("-" * 95)

    calibration_constants = {}

    for filename, pair_label in TARGET_FILES:
        if not os.path.exists(filename):
            print(f"[WARN] File not found: {filename} (Skipping)")
            continue

        df = pd.read_csv(filename)
        df = df[df['status'] == 1].copy()
        
        # Relative time in seconds
        t_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0
        df['t_sec'] = t_sec

        raw_dist = df['range_raw_m'].values
        n_samples = len(df)
        raw_mean = np.mean(raw_dist)

        # 1. Linear Drift Identification: d_raw(t) = d0 + beta * t
        slope_drift, intercept_d0 = np.polyfit(t_sec, raw_dist, 1)
        drift_mm_s = slope_drift * 1000.0

        # 2. Exact Static Offset to match 3.000m Ground Truth at t=0
        identified_offset = intercept_d0 - GROUND_TRUTH_M

        # 3. Apply Calibration: d_cal(t) = d_raw(t) - identified_offset - (slope_drift * t)
        calibrated_dist = raw_dist - identified_offset - (slope_drift * t_sec)
        df['calibrated_dist'] = calibrated_dist
        df['error_cm'] = (calibrated_dist - GROUND_TRUTH_M) * 100.0
        df['pair_label'] = pair_label

        cal_std_cm = np.std(calibrated_dist) * 100.0
        cal_mean = np.mean(calibrated_dist)
        cal_bias_cm = (cal_mean - GROUND_TRUTH_M) * 100.0

        calibration_constants[pair_label] = {
            'offset_m': identified_offset,
            'drift_slope': slope_drift
        }

        print(f"{pair_label:8s} | {n_samples:7d} | {raw_mean:9.4f}m | {identified_offset:16.4f} m | {drift_mm_s:+11.4f} mm/s | {cal_std_cm:8.2f} cm")
        
        calibrated_frames.append(df)
        results.append({
            'Pair': pair_label,
            'Raw Mean (m)': raw_mean,
            'Calib Mean (m)': cal_mean,
            'Bias (cm)': cal_bias_cm,
            'Std (cm)': cal_std_cm,
            'Drift (mm/s)': drift_mm_s,
            'Offset (m)': identified_offset
        })

    print("=" * 95)
    
    df_all = pd.concat(calibrated_frames, ignore_index=True)

    # -------------------------------------------------------------
    # Plotting Calibrated Trajectories
    # -------------------------------------------------------------
    pairs = sorted(df_all['pair_label'].unique())
    fig, axes = plt.subplots(len(pairs), 1, figsize=(13, 3.0 * len(pairs)), sharex=False)
    if len(pairs) == 1: axes = [axes]

    for idx, (ax, p_name) in enumerate(zip(axes, pairs)):
        sub = df_all[df_all['pair_label'] == p_name]
        t_min = sub['t_sec'] / 60.0

        ax.plot(t_min, sub['calibrated_dist'], lw=1.2, color='#1f77b4', label=f'{p_name} Calibrated')
        ax.axhline(GROUND_TRUTH_M, color='red', linestyle='--', lw=1.8, label=f'Ground Truth ({GROUND_TRUTH_M:.3f} m)')
        
        ax.set_title(f"Pair {p_name} — Flat Calibrated Range (Mean: {sub['calibrated_dist'].mean():.4f} m | Bias: {(sub['calibrated_dist'].mean()-3.0)*100:+.2f} cm | Std: {sub['calibrated_dist'].std()*100:.1f} cm)", fontweight='bold')
        ax.set_ylabel("Distance (m)")
        ax.set_xlabel("Time (Minutes)")
        ax.set_ylim([GROUND_TRUTH_M - 0.3, GROUND_TRUTH_M + 0.3])
        ax.legend(loc='upper right', frameon=True, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "pairwise_calibrated_3m_trajectories.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[SAVED] Comparison Trajectories: {plot_path}")

    # -------------------------------------------------------------
    # Generate C++ Code for RobotConfig.h
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print("      EXACT PAIRWISE CALIBRATION MATRIX FOR include/RobotConfig.h")
    print("#" * 80)
    print("// Pairwise Offset Lookup Matrix (in Meters) for N=4 Leaderless Swarm")
    print("constexpr float UWB_PAIR_OFFSETS[4][4] = {")
    print("    //    R1           R2           R3           R4")
    
    r12 = calibration_constants.get('R1-R2', {}).get('offset_m', 22.272)
    r13 = calibration_constants.get('R1-R3', {}).get('offset_m', 23.991)
    r14 = calibration_constants.get('R1-R4', {}).get('offset_m', 22.116)
    r23 = calibration_constants.get('R2-R3', {}).get('offset_m', 32.391)
    r24 = calibration_constants.get('R2-R4', {}).get('offset_m', 40.694)
    r34 = calibration_constants.get('R3-R4', {}).get('offset_m', 38.157)

    print(f"    {{   0.0000f,   {r12:.4f}f,   {r13:.4f}f,   {r14:.4f}f }}, // R1")
    print(f"    {{  {r12:.4f}f,    0.0000f,   {r23:.4f}f,   {r24:.4f}f }}, // R2")
    print(f"    {{  {r13:.4f}f,   {r23:.4f}f,    0.0000f,   {r34:.4f}f }}, // R3")
    print(f"    {{  {r14:.4f}f,   {r24:.4f}f,   {r34:.4f}f,    0.0000f }}  // R4")
    print("};\n")
    print("// Helper to get calibrated distance between Robot i and Robot j")
    print("inline float getCalibratedDistance(uint8_t myId, uint8_t peerId, float rawDist) {")
    print("    if (myId < 1 || myId > 4 || peerId < 1 || peerId > 4 || myId == peerId) return rawDist;")
    print("    return rawDist - UWB_PAIR_OFFSETS[myId - 1][peerId - 1];")
    print("}")
    print("#" * 80 + "\n")

if __name__ == "__main__":
    analyze_and_calibrate_3m()
