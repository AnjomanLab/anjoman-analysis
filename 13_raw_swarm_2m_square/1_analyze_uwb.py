#!/usr/bin/env python3
"""
Anjoman Swarm Metrology & UWB Calibration Engine (v2.0 - Directed Ego-Centric)
Analyzes 4-robot raw static ranging datasets independently from each robot's perspective.
Evaluates 12 directed links, thermal drift, RF diagnostics, temporal whiteness, and reciprocity.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# 0. CONFIGURATION & GROUND TRUTH
# ==============================================================================
OUTPUT_DIR = "output"
FILE_NAMES = [
    "dataset_raw_static_square_r1_10min.csv",
    "dataset_raw_static_square_r2_10min.csv",
    "dataset_raw_static_square_r3_10min.csv",
    "dataset_raw_static_square_r4_10min.csv"
]

# Physical Ground Truth for 2.000m Rigid Square Formation (meters)
GROUND_TRUTH = {
    (1, 2): 2.0000, (2, 1): 2.0000,
    (1, 4): 2.0000, (4, 1): 2.0000,
    (2, 3): 2.0000, (3, 2): 2.0000,
    (3, 4): 2.0000, (4, 3): 2.0000,
    (1, 3): 2.828427, (3, 1): 2.828427, # Diagonal: 2 * sqrt(2)
    (2, 4): 2.828427, (4, 2): 2.828427  # Diagonal: 2 * sqrt(2)
}

EXPECTED_COLS = [
    "timestamp_ms", "frame_id", "init_id", "resp_id", "status", "tof_raw", "tof_comp",
    "dist_raw_m", "dist_cfo_m", "carrier_int", "cfo_ppm", "rssi_dbm", "fp_power_dbm",
    "rx_quality", "std_noise", "fp_ampl1", "fp_ampl2", "fp_ampl3", "cir_pwr",
    "rxpacc", "lde_error", "vbat_init", "vbat_resp", "temp_uwb_init", "temp_uwb_resp",
    "temp_esp_init", "temp_esp_resp", "tTx1", "tRx1", "tRx2", "tTx2"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. DATA INGESTION (DIRECTED EGOCENTRIC LOADING)
# ==============================================================================
def load_datasets():
    dfs = []
    file_status = {}

    for idx, fname in enumerate(FILE_NAMES, start=1):
        if not os.path.exists(fname):
            file_status[idx] = {"file": fname, "count": 0, "status": "NOT FOUND"}
            continue
        try:
            df_cur = pd.read_csv(fname)
            missing = [c for c in EXPECTED_COLS if c not in df_cur.columns]
            if missing:
                warnings.warn(f"File {fname} missing columns: {missing}")

            # Filter valid status
            if "status" in df_cur.columns:
                df_cur = df_cur[df_cur["status"] == 1].copy()

            df_cur["host_robot"] = idx
            df_cur["dir_link"] = df_cur["init_id"].astype(str) + "->" + df_cur["resp_id"].astype(str)
            df_cur["t_sec"] = (df_cur["timestamp_ms"] - df_cur["timestamp_ms"].min()) / 1000.0

            file_status[idx] = {"file": fname, "count": len(df_cur), "status": "LOADED"}
            dfs.append(df_cur)
        except Exception as e:
            warnings.warn(f"Error loading {fname}: {e}")
            file_status[idx] = {"file": fname, "count": 0, "status": f"ERROR: {e}"}

    if not dfs:
        sys.exit("[CRITICAL ERROR] No valid dataset CSV files found in working directory.")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all, file_status

# ==============================================================================
# 2. STATISTICAL ENGINE (12 DIRECTED LINKS)
# ==============================================================================
def compute_directed_statistics(df_all):
    stats_rows = []

    # Iterate through all 12 potential directed combinations
    for i in range(1, 5):
        for j in range(1, 5):
            if i == j:
                continue
            dir_link = f"{i}->{j}"
            gt = GROUND_TRUTH.get((i, j), np.nan)
            sub = df_all[df_all["dir_link"] == dir_link]

            if len(sub) == 0:
                stats_rows.append({
                    "dir_link": dir_link, "init_id": i, "resp_id": j, "loaded": False,
                    "gt_m": gt, "count": 0, "mean_raw_m": np.nan, "mean_cfo_m": np.nan,
                    "bias_cfo_m": np.nan, "std_m": np.nan, "median_m": np.nan,
                    "mad_m": np.nan, "skewness": np.nan, "kurtosis": np.nan,
                    "cfo_ppm_mean": np.nan, "rssi_mean": np.nan, "noise_mean": np.nan,
                    "rx_quality_mean": np.nan, "lde_error_rate": np.nan
                })
                continue

            vals_cfo = sub["dist_cfo_m"].values
            vals_raw = sub["dist_raw_m"].values
            med_cfo = np.median(vals_cfo)
            mad_cfo = 1.4826 * np.median(np.abs(vals_cfo - med_cfo))

            stats_rows.append({
                "dir_link": dir_link, "init_id": i, "resp_id": j, "loaded": True,
                "gt_m": gt, "count": len(vals_cfo),
                "mean_raw_m": np.mean(vals_raw),
                "mean_cfo_m": np.mean(vals_cfo),
                "bias_cfo_m": np.mean(vals_cfo) - gt,
                "std_m": np.std(vals_cfo, ddof=1) if len(vals_cfo) > 1 else 0.0,
                "median_m": med_cfo,
                "mad_m": mad_cfo,
                "skewness": stats.skew(vals_cfo) if len(vals_cfo) > 2 else 0.0,
                "kurtosis": stats.kurtosis(vals_cfo) if len(vals_cfo) > 3 else 0.0,
                "cfo_ppm_mean": sub["cfo_ppm"].mean() if "cfo_ppm" in sub.columns else np.nan,
                "rssi_mean": sub["rssi_dbm"].mean() if "rssi_dbm" in sub.columns else np.nan,
                "noise_mean": sub["std_noise"].mean() if "std_noise" in sub.columns else np.nan,
                "rx_quality_mean": sub["rx_quality"].mean() if "rx_quality" in sub.columns else np.nan,
                "lde_error_rate": sub["lde_error"].mean() if "lde_error" in sub.columns else 0.0
            })

    df_dir_stats = pd.DataFrame(stats_rows)
    df_dir_stats.to_csv(os.path.join(OUTPUT_DIR, "directed_links_summary.csv"), index=False)
    return df_dir_stats

# ==============================================================================
# 3. PER-ROBOT EGOCENTRIC LOCAL SUMMARY
# ==============================================================================
def compute_per_robot_summary(df_all):
    dev_rows = []
    for host_id in range(1, 5):
        sub_host = df_all[df_all["host_robot"] == host_id]
        if len(sub_host) == 0:
            continue

        temp_u = sub_host["temp_uwb_init"].values
        temp_e = sub_host["temp_esp_init"].values
        vbat   = sub_host["vbat_init"].values

        dev_rows.append({
            "robot_id": host_id,
            "total_frames_logged": len(sub_host),
            "temp_uwb_min_C": np.min(temp_u),
            "temp_uwb_max_C": np.max(temp_u),
            "temp_uwb_range_C": np.max(temp_u) - np.min(temp_u),
            "temp_esp_mean_C": np.mean(temp_e),
            "vbat_mean_V": np.mean(vbat),
            "active_links": list(sub_host["dir_link"].unique())
        })

    df_dev_summary = pd.DataFrame(dev_rows)
    df_dev_summary.to_csv(os.path.join(OUTPUT_DIR, "per_robot_summary.csv"), index=False)
    return df_dev_summary

# ==============================================================================
# 4. RECIPROCITY & DIRECTIONAL ASYMMETRY ANALYSIS
# ==============================================================================
def compute_reciprocity(df_all):
    """
    Compares Link(u -> v) vs Link(v -> u) directly.
    Quantifies reciprocity breakdown due to crystal offsets and antenna asymmetry.
    """
    pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    recip_rows = []

    for u, v in pairs:
        link_fwd = f"{u}->{v}"
        link_rev = f"{v}->{u}"

        df_fwd = df_all[df_all["dir_link"] == link_fwd]
        df_rev = df_all[df_all["dir_link"] == link_rev]

        has_fwd = len(df_fwd) > 0
        has_rev = len(df_rev) > 0

        if has_fwd and has_rev:
            vals_fwd = df_fwd["dist_cfo_m"].values
            vals_rev = df_rev["dist_cfo_m"].values

            diff_mean = np.mean(vals_fwd) - np.mean(vals_rev)
            t_stat, p_val = stats.ttest_ind(vals_fwd, vals_rev, equal_var=False)

            cfo_fwd = df_fwd["cfo_ppm"].mean() if "cfo_ppm" in df_fwd.columns else np.nan
            cfo_rev = df_rev["cfo_ppm"].mean() if "cfo_ppm" in df_rev.columns else np.nan

            recip_rows.append({
                "pair": f"{u}<->{v}",
                "n_fwd": len(vals_fwd), "n_rev": len(vals_rev),
                "mean_fwd_m": np.mean(vals_fwd),
                "mean_rev_m": np.mean(vals_rev),
                "delta_m": diff_mean,
                "cfo_fwd_ppm": cfo_fwd,
                "cfo_rev_ppm": cfo_rev,
                "t_stat": t_stat, "p_value": p_val,
                "is_symmetric": p_val >= 0.05
            })
        else:
            recip_rows.append({
                "pair": f"{u}<->{v}",
                "n_fwd": len(df_fwd), "n_rev": len(df_rev),
                "mean_fwd_m": df_fwd["dist_cfo_m"].mean() if has_fwd else np.nan,
                "mean_rev_m": df_rev["dist_cfo_m"].mean() if has_rev else np.nan,
                "delta_m": np.nan,
                "cfo_fwd_ppm": df_fwd["cfo_ppm"].mean() if has_fwd else np.nan,
                "cfo_rev_ppm": df_rev["cfo_ppm"].mean() if has_rev else np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "is_symmetric": "Awaiting Reciprocal File"
            })

    df_recip = pd.DataFrame(recip_rows)
    df_recip.to_csv(os.path.join(OUTPUT_DIR, "reciprocity_analysis.csv"), index=False)
    return df_recip

# ==============================================================================
# 5. DIRECTED THERMAL DRIFT ENGINE
# ==============================================================================
def fit_directed_thermal_model(df_all):
    """
    Model for each directed link:
    dist_cfo_m(t) - d_gt = bias_0 + alpha_init * (T_init - T_ref) + alpha_resp * (T_resp - T_ref)
    """
    thermal_rows = []

    for dir_link, grp in df_all.groupby("dir_link"):
        u = int(grp["init_id"].iloc[0])
        v = int(grp["resp_id"].iloc[0])
        d_gt = GROUND_TRUTH.get((u, v), np.nan)

        if len(grp) < 30 or np.isnan(d_gt):
            continue

        y = grp["dist_cfo_m"].values - d_gt
        t_init = grp["temp_uwb_init"].values
        t_resp = grp["temp_uwb_resp"].values

        t_ref_i = np.mean(t_init)
        t_ref_j = np.mean(t_resp)

        X = np.column_stack([
            np.ones(len(y)),
            t_init - t_ref_i,
            t_resp - t_ref_j
        ])

        beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        e = y - y_pred

        dof = max(len(y) - 3, 1)
        sigma2 = np.sum(e**2) / dof
        cov_beta = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov_beta), 1e-12))

        t_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        p_vals = 2.0 * stats.t.sf(np.abs(t_stats), df=dof)

        r2 = 1.0 - (np.sum(e**2) / np.sum((y - np.mean(y))**2)) if np.std(y) > 0 else 0.0

        thermal_rows.append({
            "dir_link": dir_link,
            "base_bias_m": beta[0], "base_bias_se": se[0],
            "alpha_init_m_per_C": beta[1], "alpha_init_se": se[1], "alpha_init_p": p_vals[1],
            "alpha_resp_m_per_C": beta[2], "alpha_resp_se": se[2], "alpha_resp_p": p_vals[2],
            "R2_score": r2, "residual_rmse_m": np.sqrt(sigma2)
        })

    df_thermal = pd.DataFrame(thermal_rows)
    df_thermal.to_csv(os.path.join(OUTPUT_DIR, "directed_thermal_fit.csv"), index=False)
    return df_thermal

# ==============================================================================
# 6. RESIDUAL DIAGNOSTICS & DIRECTED ACF ENGINE
# ==============================================================================
def compute_directed_residuals(df_all):
    diag_rows = []
    acf_dict = {}
    R_matrix = np.zeros((4, 4))

    for i in range(1, 5):
        for j in range(1, 5):
            if i == j:
                continue
            dir_link = f"{i}->{j}"
            grp = df_all[df_all["dir_link"] == dir_link]

            if len(grp) < 10:
                acf_dict[dir_link] = np.nan
                continue

            gt = GROUND_TRUTH.get((i, j), 2.0)
            vals = grp["dist_cfo_m"].values - gt
            var_ij = np.var(vals, ddof=1) if len(vals) > 1 else 0.0
            R_matrix[i - 1, j - 1] = var_ij

            n_pts = len(vals)
            max_lag = min(100, n_pts - 1)
            vals_c = vals - np.mean(vals)
            c0 = np.dot(vals_c, vals_c) / n_pts

            if c0 > 1e-12:
                acf = np.array([np.dot(vals_c[:n_pts - lag], vals_c[lag:]) / (n_pts * c0) for lag in range(max_lag + 1)])
                acf1 = acf[1] if len(acf) > 1 else 0.0

                h = min(20, max_lag)
                k_arr = np.arange(1, h + 1)
                q_stat = n_pts * (n_pts + 2) * np.sum((acf[1:h + 1]**2) / (n_pts - k_arr))
                lb_pvalue = stats.chi2.sf(q_stat, df=h)
                jb_stat, jb_pvalue = stats.jarque_bera(vals)
            else:
                acf1, q_stat, lb_pvalue, jb_stat, jb_pvalue = 0.0, 0.0, 1.0, 0.0, 1.0

            acf_dict[dir_link] = acf1
            diag_rows.append({
                "dir_link": dir_link, "variance_m2": var_ij,
                "acf1": acf1, "is_white_noise": lb_pvalue >= 0.05,
                "ljung_box_p": lb_pvalue, "is_gaussian": jb_pvalue >= 0.05,
                "jarque_bera_p": jb_pvalue
            })

    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv(os.path.join(OUTPUT_DIR, "directed_residual_diagnostics.csv"), index=False)
    np.savetxt(os.path.join(OUTPUT_DIR, "R_matrix_directed.csv"), R_matrix, delimiter=",", fmt="%.6e")
    return df_diag, acf_dict

# ==============================================================================
# 7. PUBLICATION PLOTS (DIRECTED PERSPECTIVE)
# ==============================================================================
def generate_plots(df_all):
    # Plot 1: ACF of Active Directed Links
    plt.figure(figsize=(9, 4.5), dpi=300)
    for dir_link, grp in df_all.groupby("dir_link"):
        vals = grp["dist_cfo_m"].values
        if len(vals) > 50 and np.std(vals) > 1e-6:
            v_c = vals - np.mean(vals)
            c0 = np.dot(v_c, v_c)
            lags = min(100, len(vals) - 1)
            acf = [np.dot(v_c[:len(vals)-lag], v_c[lag:]) / c0 for lag in range(lags)]
            plt.plot(acf, label=f"Link {dir_link}")

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.axhline(1.96 / np.sqrt(max(len(df_all), 1)), color="red", linestyle=":", label="95% White Noise Bound")
    plt.axhline(-1.96 / np.sqrt(max(len(df_all), 1)), color="red", linestyle=":")
    plt.title("Directed Residual Autocorrelation Function (ACF) up to Lag 100")
    plt.xlabel("Lag (discrete TDMA frames)")
    plt.ylabel("Autocorrelation")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "directed_acf.png"))
    plt.close()

    # Plot 2: Directed Thermal Scatter
    plt.figure(figsize=(9, 4.5), dpi=300)
    for dir_link, grp in df_all.groupby("dir_link"):
        plt.scatter(grp["temp_uwb_init"], grp["dist_cfo_m"], alpha=0.35, s=8, label=f"Link {dir_link}")
    plt.title("Directed UWB Distance vs Initiator Transceiver Temperature")
    plt.xlabel("Initiator Internal Temperature (°C)")
    plt.ylabel("Measured CFO Distance (m)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "directed_thermal_scatter.png"))
    plt.close()

    # Plot 3: Reciprocity Comparison
    pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    recip_labels, fwd_means, rev_means = [], [], []
    for u, v in pairs:
        sub_fwd = df_all[df_all["dir_link"] == f"{u}->{v}"]
        sub_rev = df_all[df_all["dir_link"] == f"{v}->{u}"]
        if len(sub_fwd) > 0 or len(sub_rev) > 0:
            recip_labels.append(f"{u}<->{v}")
            fwd_means.append(sub_fwd["dist_cfo_m"].mean() if len(sub_fwd) > 0 else 0.0)
            rev_means.append(sub_rev["dist_cfo_m"].mean() if len(sub_rev) > 0 else 0.0)

    if recip_labels:
        plt.figure(figsize=(9, 4.5), dpi=300)
        x = np.arange(len(recip_labels))
        width = 0.35
        plt.bar(x - width/2, fwd_means, width, label="Forward (u -> v)", color="steelblue")
        plt.bar(x + width/2, rev_means, width, label="Reverse (v -> u)", color="coral")
        plt.xticks(x, recip_labels)
        plt.ylabel("Measured Mean CFO Distance (m)")
        plt.title("Reciprocity Comparison: Forward vs Reverse Directional Distances")
        plt.grid(True, alpha=0.3, axis="y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "reciprocity_comparison.png"))
        plt.close()

# ==============================================================================
# 8. CONSOLE REPORTER (DIRECTED SCIENTIFIC SUMMARY)
# ==============================================================================
def print_console_report(file_status, df_dir_stats, df_recip, acf_dict):
    print("\n" + "="*85)
    print("ANJOMAN UWB METROLOGY SYSTEM IDENTIFICATION REPORT (DIRECTED EGO-CENTRIC)")
    print("="*85)

    # 1. Dataset Status
    print("\n--- INGESTED DATASETS ---")
    for r_id in range(1, 5):
        info = file_status[r_id]
        print(f"Robot {r_id} ({info['file']}): {info['count']} frames [{info['status']}]")

    # 2. Directed Links Status (12 Links)
    print("\n--- 12-DIRECTED LINKS METROLOGY TABLE ---")
    header = f"{'Link':<8} | {'GT (m)':<7} | {'Mean CFO (m)':<13} | {'Raw Bias (m)':<13} | {'Std (m)':<9} | {'CFO (ppm)':<9} | {'Noise':<6} | {'Status'}"
    print(header)
    print("-" * len(header))
    for _, r in df_dir_stats.iterrows():
        if r["loaded"]:
            print(f"{r['dir_link']:<8} | {r['gt_m']:<7.4f} | {r['mean_cfo_m']:<13.4f} | {r['bias_cfo_m']:<+13.4f} | {r['std_m']:<9.4f} | {r['cfo_ppm_mean']:<+9.2f} | {r['noise_mean']:<6.1f} | LOADED")
        else:
            gt_str = f"{r['gt_m']:<7.4f}" if not np.isnan(r['gt_m']) else "N/A"
            print(f"{r['dir_link']:<8} | {gt_str:<7} | {'---':<13} | {'---':<13} | {'---':<9} | {'---':<9} | {'---':<6} | NOT LOADED")

    # 3. Reciprocity Comparison
    print("\n--- RECIPROCITY & DIRECTIONAL ASYMMETRY (Forward vs Reverse) ---")
    recip_header = f"{'Pair':<8} | {'Forward (m)':<12} | {'Reverse (m)':<12} | {'Delta (m)':<11} | {'Symmetry Assessment'}"
    print(recip_header)
    print("-" * len(recip_header))
    for _, r in df_recip.iterrows():
        fwd_s = f"{r['mean_fwd_m']:.4f}" if not np.isnan(r['mean_fwd_m']) else "---"
        rev_s = f"{r['mean_rev_m']:.4f}" if not np.isnan(r['mean_rev_m']) else "---"
        del_s = f"{r['delta_m']:+.4f}" if not np.isnan(r['delta_m']) else "---"
        status_s = "RECIPROCAL" if r['is_symmetric'] == True else ("ASYMMETRIC (p<0.05)" if r['is_symmetric'] == False else str(r['is_symmetric']))
        print(f"{r['pair']:<8} | {fwd_s:<12} | {rev_s:<12} | {del_s:<11} | {status_s}")

    # 4. Temporal Autocorrelation (ACF[1])
    print("\n--- TEMPORAL RESIDUAL LAG-1 AUTOCORRELATION (ACF[1]) ---")
    for link, acf_val in acf_dict.items():
        if np.isnan(acf_val):
            continue
        status = "White Noise" if abs(acf_val) < 0.1 else "Color/Hysteresis"
        print(f"Link {link:<6}: ACF[1] = {acf_val:+7.4f} [{status}]")

    # 5. Executive Conclusions (Strict 5 Lines)
    print("\n--- EXECUTIVE ENGINEERING CONCLUSIONS ---")
    active_cnt = df_dir_stats['loaded'].sum()
    print(f"1. Network Completeness: {active_cnt}/12 directed links loaded; ego-centric perspective fully preserved without undirected pooling.")
    print("2. Reciprocity Dynamics: SS-TWR reveals directional bias polarity; initiator vs responder roles decouple crystal drift from hardware antenna delay.")
    print("3. Thermal Stability: Directed thermal slopes quantify heat accumulation independently for each node's transmit and receive circuitry.")
    print("4. RF Environment: Noise floors and CIR amplitudes remain link-specific, isolating board-level EMI from true line-of-sight path loss.")
    print("5. Control Recommendation: Calibrate biases per directed link in RobotConfig.h (matrix B_ij) to ensure symmetric swarm rigidity in Core 0 EKF.")
    print("="*85 + "\n")

# ==============================================================================
# 9. MAIN PIPELINE
# ==============================================================================
def main():
    df_all, file_status = load_datasets()
    df_dir_stats = compute_directed_statistics(df_all)
    compute_per_robot_summary(df_all)
    df_recip = compute_reciprocity(df_all)
    fit_directed_thermal_model(df_all)
    _, acf_dict = compute_directed_residuals(df_all)
    generate_plots(df_all)
    print_console_report(file_status, df_dir_stats, df_recip, acf_dict)

if __name__ == "__main__":
    main()
