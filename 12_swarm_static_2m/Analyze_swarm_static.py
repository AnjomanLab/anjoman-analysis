#!/usr/bin/env python3
"""
Anjoman Swarm Firmware - 4-Robot 2m Static UWB Benchmark Analyzer
Robust parser: Auto-maps legacy and full 16-column headers, handles serial prefixes.
"""

import os
import glob
import math
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ground Truth Euclidean Distances for 2.000m x 2.000m Square
# Sides = 2.000 m | Diagonals = 2 * sqrt(2) = 2.8284 m
GROUND_TRUTH = {
    (1, 2): 2.0000, # Side
    (1, 3): 2.8284, # Diagonal
    (1, 4): 2.0000, # Side
    (2, 3): 2.0000, # Side
    (2, 4): 2.8284, # Diagonal
    (3, 4): 2.0000  # Side
}

OUTPUT_DIR = "analysis_results_swarm_static"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Discover Input File
target_file = None
for f in ["ss.csv", "swarm_static_2m.csv", "swarm_static_2.csv", "swarm_static.csv", "log.txt"]:
    if os.path.exists(f):
        target_file = f
        break

if not target_file:
    candidates = sorted(glob.glob("*.csv") + glob.glob("*.txt"))
    if candidates:
        target_file = candidates[0]
    else:
        print("[ERROR] No log file found in this folder!")
        exit(1)

print(f"[INFO] Parsing log file: {target_file}")

# 2. Extract Valid Data Lines (Filter comment lines and monitor prefixes)
raw_lines = []
header_found = None

with open(target_file, "r") as fin:
    for line in fin:
        line_clean = line.strip()
        if ">" in line_clean:
            line_clean = line_clean.split(">")[-1].strip()
        
        if not line_clean or line_clean.startswith("#"):
            continue

        if "timestamp" in line_clean.lower() and header_found is None:
            header_found = line_clean
            continue

        raw_lines.append(line_clean)

if not raw_lines:
    print("[ERROR] No valid CSV telemetry records found in file!")
    exit(1)

# 3. Handle Header Normalization
if header_found:
    csv_content = header_found + "\n" + "\n".join(raw_lines)
    df = pd.read_csv(StringIO(csv_content))
else:
    # Auto-assign positional headers if file has no header line
    first_row_cols = len(raw_lines[0].split(","))
    if first_row_cols >= 15:
        names = ["timestamp_ms", "frame_id", "init_id", "resp_id", "status", "tof_ticks",
                 "range_raw_m", "range_corr_m", "range_calib_m", "rssi_dbm",
                 "tTx1", "tRx1", "tRx2", "tTx2", "temp_init", "temp_resp"]
    else:
        names = ["timestamp_ms", "frame_id", "init_id", "resp_id", "range_raw_m", "range_calib_m", "rssi_dbm", "temp_init"]
    
    df = pd.read_csv(StringIO("\n".join(raw_lines)), names=names[:first_row_cols])

df.columns = df.columns.str.strip().str.lower()

# Flexible Column Mapping (Maps legacy 'init'/'resp' to standard 'init_id'/'resp_id')
col_map = {
    "init": "init_id",
    "initiator": "init_id",
    "resp": "resp_id",
    "responder": "resp_id",
    "raw_m": "range_raw_m",
    "raw_range": "range_raw_m",
    "calib_m": "range_calib_m",
    "calib_range": "range_calib_m",
    "rssi": "rssi_dbm",
    "frame": "frame_id"
}
df = df.rename(columns=col_map)

# Fallback: Positional mapping if column names were completely unrecognized
if "init_id" not in df.columns and len(df.columns) >= 4:
    df.rename(columns={df.columns[2]: "init_id", df.columns[3]: "resp_id"}, inplace=True)

# If status column is missing, infer success if distance > 0.1m
if "status" not in df.columns:
    df["status"] = (df["range_raw_m"] > 0.1).astype(int)

# ==============================================================================
# 4. LINK-BY-LINK METROLOGY & RELIABILITY REPORT
# ==============================================================================
edge_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
report_lines = []

def log_p(s=""):
    print(s)
    report_lines.append(s)

log_p("=" * 110)
log_p("   ANJOMAN 4-ROBOT UWB STATIC BENCHMARK REPORT (2.000m SQUARE)")
log_p("=" * 110)
log_p(f"{'Link':<8}|{'Target (m)':<12}|{'Attempts':<10}|{'Success':<10}|{'PDR (%)':<10}|{'Raw Mean (m)':<14}|{'Calib Mean (m)':<16}|{'Error (cm)'}")
log_p("-" * 110)

link_data = {}

for u, v in edge_pairs:
    sub = df[(df["init_id"] == u) & (df["resp_id"] == v)]
    attempts = len(sub)
    success_sub = sub[sub["status"] == 1]
    successes = len(success_sub)
    pdr = (successes / attempts * 100.0) if attempts > 0 else 0.0
    gt = GROUND_TRUTH.get((u, v), 2.000)

    if successes > 0:
        raw_mean = success_sub["range_raw_m"].mean()
        raw_std  = success_sub["range_raw_m"].std()
        calib_col = "range_calib_m" if "range_calib_m" in success_sub.columns else "range_raw_m"
        calib_mean = success_sub[calib_col].mean()
        calib_std  = success_sub[calib_col].std()
        err_cm = (calib_mean - gt) * 100.0
        rssi_mean = success_sub["rssi_dbm"].mean() if "rssi_dbm" in success_sub.columns else np.nan
    else:
        raw_mean = raw_std = calib_mean = calib_std = err_cm = rssi_mean = np.nan

    link_data[(u, v)] = {
        "attempts": attempts,
        "successes": successes,
        "pdr": pdr,
        "gt": gt,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "calib_mean": calib_mean,
        "calib_std": calib_std,
        "err_cm": err_cm,
        "rssi_mean": rssi_mean,
        "sub": success_sub
    }

    status_str = f"({u}->{v})"
    raw_str = f"{raw_mean:>10.4f} m" if not np.isnan(raw_mean) else "       N/A   "
    cal_str = f"{calib_mean:>12.4f} m" if not np.isnan(calib_mean) else "         N/A   "
    err_str = f"{err_cm:>+8.1f} cm" if not np.isnan(err_cm) else "     N/A   "

    log_p(f"{status_str:<8}| {gt:<10.4f} | {attempts:<8} | {successes:<8} | {pdr:>6.1f} %  | {raw_str} | {cal_str} | {err_str}")

log_p("=" * 110)

with open(os.path.join(OUTPUT_DIR, "swarm_static_report.txt"), "w") as fout:
    fout.write("\n".join(report_lines) + "\n")

# ==============================================================================
# 5. VISUALIZATION ENGINE
# ==============================================================================
# --- FIGURE 1: PDR RELIABILITY BAR CHART ---
plt.figure(figsize=(10, 6), dpi=300)
links_labels = [f"R{u}-R{v}" for u, v in edge_pairs]
pdrs = [link_data[p]["pdr"] for p in edge_pairs]
colors = ["#2ca02c" if p > 80 else "#d62728" for p in pdrs]

bars = plt.bar(links_labels, pdrs, color=colors, width=0.5, edgecolor="black")
plt.axhline(100.0, color="gray", linestyle="--", linewidth=1.0)
plt.title("4-Robot TDMA Network Reliability (Packet Delivery Ratio %)", fontsize=13, pad=12)
plt.ylabel("Packet Delivery Ratio (PDR %)", fontsize=11)
plt.ylim(0, 115)
plt.grid(True, linestyle=":", alpha=0.6, axis="y")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_network_pdr_reliability.png"))
plt.close()

# --- FIGURE 2: MEASURED RANGES OVER TIME ---
active_links = [p for p in edge_pairs if link_data[p]["successes"] > 0]

if active_links:
    fig, axes = plt.subplots(len(active_links), 1, figsize=(14, 4 * len(active_links)), sharex=True)
    if len(active_links) == 1:
        axes = [axes]

    for idx, (u, v) in enumerate(active_links):
        ax = axes[idx]
        sdata = link_data[(u, v)]["sub"]
        t_sec = (sdata["timestamp_ms"] - sdata["timestamp_ms"].iloc[0]) / 1000.0
        gt = link_data[(u, v)]["gt"]
        calib_col = "range_calib_m" if "range_calib_m" in sdata.columns else "range_raw_m"

        ax.axhline(gt, color="black", linestyle="--", linewidth=1.5, label=f"Ground Truth ({gt:.3f} m)")
        ax.plot(t_sec, sdata[calib_col], color="#1f77b4", linewidth=1.5, label=f"Measured Range (Mean={sdata[calib_col].mean():.3f} m)")
        
        rssi_info = f" | RSSI: {link_data[(u, v)]['rssi_mean']:.1f} dBm" if not np.isnan(link_data[(u, v)]['rssi_mean']) else ""
        ax.set_title(f"Link R{u}-R{v} Tracking (Error: {link_data[(u, v)]['err_cm']:+.1f} cm{rssi_info})", fontsize=12)
        ax.set_ylabel("Range (m)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper right", fontsize=10)

    axes[-1].set_xlabel("Elapsed Time (s)", fontsize=11)
    plt.suptitle("Active UWB Links Range Stability", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_active_links_range_tracking.png"))
    plt.close()

print(f"\n[SUCCESS] Report and figures saved in: {OUTPUT_DIR}/")
