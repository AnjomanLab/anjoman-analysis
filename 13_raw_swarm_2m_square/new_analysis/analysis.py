import glob
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

SPEED_OF_LIGHT = 299792458.0
TIME_UNIT_SEC = 1.0 / (499.2e6 * 128.0)  # ~15.65e-12 s
DEFAULT_ANTENNA_DELAY = 16436  # DW1000 default ticks

# Ground truth distances for the 2m square layout (R2, R3, R4)
# Geometry: R2=(0, 2), R3=(0, 0), R4=(2, 0)
TRUE_DISTANCES = {
    (2, 3): 2.000,
    (3, 2): 2.000,
    (3, 4): 2.000,
    (4, 3): 2.000,
    (2, 4): np.sqrt(2.0**2 + 2.0**2),  # ~2.8284 m
    (4, 2): np.sqrt(2.0**2 + 2.0**2),
}

def unwrap_40bit(t_end, t_start):
    return (t_end - t_start) & 0xFFFFFFFFFF

def load_data():
    files = glob.glob("dataset_raw_static_square_r*.csv")
    if not files:
        files = glob.glob("*10min.csv")
    
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df[full_df['status'] == 1].copy()
    
    # Filter out Robot 1 (focus purely on active fleet R2, R3, R4)
    full_df = full_df[(full_df['init_id'] > 1) & (full_df['resp_id'] > 1)].copy()
    return full_df

def compute_raw_metrics(df):
    results = {}
    
    # Recompute pure ToF without firmware bias assumptions
    t_round = unwrap_40bit(df['tRx1'], df['tTx1'])
    t_reply = unwrap_40bit(df['tTx2'], df['tRx2'])
    
    # Valid physical exchanges only
    valid_mask = t_round > t_reply
    df = df[valid_mask].copy()
    t_round = t_round[valid_mask]
    t_reply = t_reply[valid_mask]
    
    tof_ticks = (t_round - t_reply) / 2.0
    # In old firmware, antenna delay of 16436 was configured in hardware registers
    # Raw hardware distance:
    df['reconstructed_dist_m'] = tof_ticks * TIME_UNIT_SEC * SPEED_OF_LIGHT

    print("=" * 80)
    print("DIRECTED LINK STATISTICS (Ground Truth vs Reconstructed Raw)")
    print("=" * 80)
    print(f"{'Link':<8} | {'Count':<6} | {'True (m)':<9} | {'Mean (m)':<9} | {'Bias (m)':<9} | {'Std (cm)':<8} | {'CFO (ppm)':<9}")
    print("-" * 80)
    
    grouped = df.groupby(['init_id', 'resp_id'])
    link_stats = {}
    
    for (init, resp), group in grouped:
        true_d = TRUE_DISTANCES.get((init, resp), np.nan)
        mean_d = group['reconstructed_dist_m'].mean()
        bias = mean_d - true_d
        std_cm = group['reconstructed_dist_m'].std() * 100.0
        cfo_mean = group['cfo_ppm'].mean()
        
        link_stats[(init, resp)] = {
            'true_d': true_d,
            'mean_d': mean_d,
            'bias': bias,
            'std_m': group['reconstructed_dist_m'].std()
        }
        
        print(f"R{init}->R{resp:<4} | {len(group):<6} | {true_d:<9.4f} | {mean_d:<9.4f} | {bias:<+9.4f} | {std_cm:<8.2f} | {cfo_mean:<+9.2f}")

    return df, link_stats

def solve_antenna_delays(link_stats):
    """
    Measurement equation:
    d_measured_ij = d_true_ij + (c / 2) * (delta_delay_i + delta_delay_j) * TIME_UNIT_SEC
    Where delta_delay is the correction to the default 16436 ticks.
    """
    pairs = list(link_stats.keys())
    
    # Node mapping: R2 -> 0, R3 -> 1, R4 -> 2
    node_map = {2: 0, 3: 1, 4: 2}
    
    def residuals(delta_delays):
        res = []
        for (i, j), stat in link_stats.items():
            idx_i = node_map[i]
            idx_j = node_map[j]
            # Time delay error in distance
            time_err_m = (delta_delays[idx_i] + delta_delays[idx_j]) * TIME_UNIT_SEC * SPEED_OF_LIGHT / 2.0
            model_d = stat['true_d'] + time_err_m
            res.append(stat['mean_d'] - model_d)
        return np.array(res)

    # Initial guess: 0 correction to 16436
    x0 = np.array([0.0, 0.0, 0.0])
    sol = least_squares(residuals, x0, method='lm')
    
    delta_ticks = sol.x
    calibrated_delays = DEFAULT_ANTENNA_DELAY + delta_ticks
    
    print("\n" + "=" * 80)
    print("EXTRACTED PER-ROBOT OPTIMAL ANTENNA DELAYS (Table 2.7 Resolution)")
    print("=" * 80)
    for robot_id in [2, 3, 4]:
        idx = node_map[robot_id]
        print(f"Robot {robot_id}:")
        print(f"  Base Configured Delay : {DEFAULT_ANTENNA_DELAY} ticks")
        print(f"  Correction Required   : {delta_ticks[idx]:+.1f} ticks ({delta_ticks[idx] * TIME_UNIT_SEC * SPEED_OF_LIGHT / 2.0 * 100.0:+.2f} cm)")
        print(f"  CALIBRATED VALUE      : {int(round(calibrated_delays[idx]))} ticks (0x{int(round(calibrated_delays[idx])):04X})")

    # Evaluate residuals after applying individual delays
    print("-" * 80)
    print("RESIDUAL LINK BIAS AFTER INDIVIDUAL ANTENNA DELAYS:")
    for (i, j), stat in link_stats.items():
        idx_i = node_map[i]
        idx_j = node_map[j]
        pred_bias = (delta_ticks[idx_i] + delta_ticks[idx_j]) * TIME_UNIT_SEC * SPEED_OF_LIGHT / 2.0
        remaining_bias_cm = (stat['bias'] - pred_bias) * 100.0
        print(f"Link R{i}->R{j}: Remaining Error = {remaining_bias_cm:+.2f} cm")

def main():
    print("Reading and analyzing 2m square static dataset...")
    df = load_data()
    df, stats = compute_raw_metrics(df)
    solve_antenna_delays(stats)

if __name__ == '__main__':
    main()
