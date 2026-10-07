#!/usr/bin/env python3
"""
Anjoman Swarm: Physics-Informed Multi-Node UWB Calibration Engine
Solves individual node antenna delays (T_ant_1..4) and CFO clock-drift sensitivity (alpha).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares

# Configuration
GROUND_TRUTH_M = 3.000
SPEED_OF_LIGHT = 299792458.0
TIME_UNIT_SEC = 0.000000000015650040064103  # 15.65 ps per DW1000 tick

OUTPUT_DIR = "plots_physics_calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

# Target Pair Files
FILE_PAIRS = [
    ("r1_r2_3m.txt", 1, 2),
    ("r1_r3_3m.txt", 1, 3),
    ("r1_r4_3m.txt", 1, 4),
    ("r2_r3_3m.txt", 2, 3),
    ("r2_r4_3m.txt", 2, 4),
    ("r3_r4_3m.txt", 3, 4),
]

def load_dataset():
    """Loads and combines all available pairwise UWB test logs."""
    combined_data = []

    for filename, node_i, node_j in FILE_PAIRS:
        if not os.path.exists(filename):
            print(f"[WARN] File not found: {filename} (Skipping)")
            continue

        try:
            df = pd.read_csv(filename)
            df = df[df['status'] == 1].copy() # Filter successful packets only
            # Compute hardware time parameters if not present
            if 't_round' not in df.columns:
                df['t_round'] = (df['tRx1'] - df['tTx1']) % (1 << 40)
            if 't_reply' not in df.columns:
                df['t_reply'] = (df['tTx2'] - df['tRx2']) % (1 << 40)

            df['node_i'] = node_i
            df['node_j'] = node_j
            df['pair_label'] = f"R{node_i}-R{node_j}"
            df['ground_truth_m'] = GROUND_TRUTH_M

            # Compute raw hardware distance
            tof_ticks_raw = (df['t_round'] - df['t_reply']) / 2.0
            df['raw_dist_calc'] = tof_ticks_raw * TIME_UNIT_SEC * SPEED_OF_LIGHT

            combined_data.append(df)
            print(f"[LOADED] {filename:15s} | Samples: {len(df):6d} | Node {node_i} <-> Node {node_j}")
        except Exception as e:
            print(f"[ERROR] Failed to parse {filename}: {e}")

    if not combined_data:
        print("[FATAL] No valid data files found!")
        sys.exit(1)

    return pd.concat(combined_data, ignore_index=True)

def physics_calibration_residuals(params, df):
    """
    Residual function for OLS optimization.
    params = [T_ant_1, T_ant_2, T_ant_3, T_ant_4, alpha_cfo]
    """
    t_ant = {1: params[0], 2: params[1], 3: params[2], 4: params[3]}
    alpha_cfo = params[4]

    # Model: ToF_cfo = [t_round - t_reply * (1 - alpha * CFO * 1e-6)] / 2
    cfo_correction = 1.0 - (alpha_cfo * df['cfo_ppm'] * 1e-6)
    t_reply_comp = df['t_reply'] * cfo_correction
    tof_comp_ticks = (df['t_round'] - t_reply_comp) / 2.0
    dist_cfo = tof_comp_ticks * TIME_UNIT_SEC * SPEED_OF_LIGHT

    # Subtract node-specific antenna delays
    t_ant_i = df['node_i'].map(t_ant)
    t_ant_j = df['node_j'].map(t_ant)
    dist_calibrated = dist_cfo - (t_ant_i + t_ant_j)

    # Residual relative to ground truth (3.000 m)
    return dist_calibrated - df['ground_truth_m']

def apply_calibration(params, df):
    """Applies the identified physical calibration parameters to the dataset."""
    t_ant = {1: params[0], 2: params[1], 3: params[2], 4: params[3]}
    alpha_cfo = params[4]

    cfo_correction = 1.0 - (alpha_cfo * df['cfo_ppm'] * 1e-6)
    t_reply_comp = df['t_reply'] * cfo_correction
    tof_comp_ticks = (df['t_round'] - t_reply_comp) / 2.0
    dist_cfo = tof_comp_ticks * TIME_UNIT_SEC * SPEED_OF_LIGHT

    t_ant_i = df['node_i'].map(t_ant)
    t_ant_j = df['node_j'].map(t_ant)
    df['dist_calibrated_physics'] = dist_cfo - (t_ant_i + t_ant_j)
    df['calib_error_m'] = df['dist_calibrated_physics'] - df['ground_truth_m']
    return df

def main():
    print("=" * 80)
    print("      ANJOMAN SWARM - PHYSICS-INFORMED UWB SYSTEM IDENTIFICATION")
    print("=" * 80)

    df_all = load_dataset()

    # Initial Guesses: [T_ant_1, T_ant_2, T_ant_3, T_ant_4, alpha_cfo]
    initial_params = [5.0, 18.0, 19.0, 18.0, 1.0]

    # Bounds: Antenna delays must be positive, alpha must be within [-10, 10]
    bounds = (
        [0.0, 0.0, 0.0, 0.0, -10.0],
        [30.0, 30.0, 30.0, 30.0, 10.0]
    )

    print("\n[OPTIMIZATION] Running Multi-Node Constrained Least Squares Optimization...")
    opt_result = least_squares(
        physics_calibration_residuals,
        initial_params,
        bounds=bounds,
        args=(df_all,),
        loss='soft_l1', # Robust to outliers
        f_scale=0.05
    )

    p_opt = opt_result.x
    T_ant = {1: p_opt[0], 2: p_opt[1], 3: p_opt[2], 4: p_opt[3]}
    alpha_opt = p_opt[4]

    print("\n" + "#" * 80)
    print("                   IDENTIFIED PHYSICAL PARAMETERS")
    print("#" * 80)
    for node_id in range(1, 5):
        delay_m = T_ant[node_id]
        delay_ticks = int(round((delay_m / SPEED_OF_LIGHT) / TIME_UNIT_SEC))
        print(f"Robot {node_id} Hardware Antenna Delay (T_ant_{node_id}): {delay_m:8.4f} m  ({delay_ticks} DW1000 ticks)")
    print(f"Universal CFO Clock-Drift Coupling (alpha) : {alpha_opt:8.6f}")
    print("#" * 80)

    # Apply calibration across all data
    df_calibrated = apply_calibration(p_opt, df_all)

    # -------------------------------------------------------------
    # Performance Evaluation Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("                BEFORE vs AFTER CALIBRATION ACCURACY REPORT (3.000 m)")
    print("=" * 90)
    print(f"{'Pair':10s} | {'Raw Mean':10s} | {'Calib Mean':12s} | {'Bias (cm)':10s} | {'Noise Std':10s} | {'Drift (mm/s)':12s}")
    print("-" * 90)

    for pair_name, group in df_calibrated.groupby('pair_label'):
        raw_m = group['raw_dist_calc'].mean()
        cal_m = group['dist_calibrated_physics'].mean()
        bias_cm = (cal_m - GROUND_TRUTH_M) * 100.0
        std_cm = group['dist_calibrated_physics'].std() * 100.0

        # Calculate remaining linear drift slope
        t_sec = (group['timestamp'] - group['timestamp'].iloc[0]) / 1000.0
        slope, _ = np.polyfit(t_sec, group['dist_calibrated_physics'], 1)
        drift_mm_s = slope * 1000.0

        print(f"{pair_name:10s} | {raw_m:9.4f}m | {cal_m:10.4f}m  | {bias_cm:+8.2f}cm | {std_cm:8.2f}cm | {drift_mm_s:+10.4f} mm/s")
    print("=" * 90)

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    print("\n[PLOTTING] Generating Calibration Comparison Figures...")
    pairs = sorted(df_calibrated['pair_label'].unique())
    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 3.2 * len(pairs)), sharex=False)
    if len(pairs) == 1: axes = [axes]

    for idx, (ax, pair_name) in enumerate(zip(axes, pairs)):
        sub = df_calibrated[df_calibrated['pair_label'] == pair_name]
        t_min = (sub['timestamp'] - sub['timestamp'].iloc[0]) / 60000.0

        ax.plot(t_min, sub['dist_calibrated_physics'], lw=1.2, color='#1f77b4', label='Physics-Calibrated Distance')
        ax.axhline(GROUND_TRUTH_M, color='red', linestyle='--', lw=1.8, label=f'Ground Truth ({GROUND_TRUTH_M:.3f} m)')
        
        ax.set_title(f"Pair {pair_name} — Physics-Informed Calibrated Distance (Mean: {sub['dist_calibrated_physics'].mean():.3f} m | Std: {sub['dist_calibrated_physics'].std()*100:.1f} cm)", fontweight='bold')
        ax.set_ylabel("Distance (m)")
        ax.set_xlabel("Time Elapsed (Minutes)")
        ax.set_ylim([GROUND_TRUTH_M - 0.5, GROUND_TRUTH_M + 0.5])
        ax.legend(loc='upper right', frameon=True, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "physics_calibrated_trajectories.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[SAVED] {plot_path}")

    # Residual Error Boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_calibrated, x='pair_label', y='calib_error_m', palette='Blues')
    plt.axhline(0, color='red', linestyle='--', lw=1.5)
    plt.title("Residual Measurement Error Distribution across all 6 Pairs (cm)", fontweight='bold')
    plt.ylabel("Error relative to 3.000 m (m)")
    plt.xlabel("Robot Pair")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    box_path = os.path.join(OUTPUT_DIR, "residual_error_boxplots.png")
    plt.savefig(box_path, dpi=300)
    print(f"[SAVED] {box_path}")

    # -------------------------------------------------------------
    # Production C++ Firmware Configuration Generator
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print("        COPY-PASTE CONFIGURATION FOR include/RobotConfig.h")
    print("#" * 80)
    print("namespace Config {")
    print(f"    // Identified Hardware Antenna Delays (in Meters)")
    print(f"    constexpr float ANTENNA_DELAY_R1_M = {T_ant[1]:.4f}f;")
    print(f"    constexpr float ANTENNA_DELAY_R2_M = {T_ant[2]:.4f}f;")
    print(f"    constexpr float ANTENNA_DELAY_R3_M = {T_ant[3]:.4f}f;")
    print(f"    constexpr float ANTENNA_DELAY_R4_M = {T_ant[4]:.4f}f;")
    print(f"    constexpr float CFO_COMPENSATION_ALPHA = {alpha_opt:.6f}f;\n")
    print("    #if ROBOT_ID == 1")
    print("        constexpr float LOCAL_ANTENNA_DELAY_M = ANTENNA_DELAY_R1_M;")
    print("    #elif ROBOT_ID == 2")
    print("        constexpr float LOCAL_ANTENNA_DELAY_M = ANTENNA_DELAY_R2_M;")
    print("    #elif ROBOT_ID == 3")
    print("        constexpr float LOCAL_ANTENNA_DELAY_M = ANTENNA_DELAY_R3_M;")
    print("    #elif ROBOT_ID == 4")
    print("        constexpr float LOCAL_ANTENNA_DELAY_M = ANTENNA_DELAY_R4_M;")
    print("    #endif")
    print("} // namespace Config")
    print("#" * 80 + "\n")

if __name__ == "__main__":
    main()
