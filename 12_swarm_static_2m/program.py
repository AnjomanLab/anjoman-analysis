#!/usr/bin/env python3
"""
Anjoman Swarm Firmware - 3000-Frame Comprehensive UWB Metrology Analyzer
Usage: python3 analyze_swarm_static_comprehensive.py <filename.csv>

Analyzes:
1. Proof of Clock-Offset Compensation (Uncompensated vs Compensated ToF)
2. Long-term Thermal Transient Dynamics & Drift Rate past 3 minutes
3. Extraction of Empirical Measurement Noise Covariance Matrix (R) for Kalman Filter
4. Link-by-Link Packet Delivery Ratio (PDR) and Outlier Rate
"""

import os
import sys
import math
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# 1. GROUND TRUTH EUCLIDEAN GEOMETRY (2.000m x 2.000m SQUARE)
# ==============================================================================
GROUND_TRUTH = {
    (1, 2): 2.0000,   # Side
    (1, 3): 2.8284,   # Diagonal (2 * sqrt(2))
    (1, 4): 2.0000,   # Side
    (2, 3): 2.0000,   # Side
    (2, 4): 2.8284,   # Diagonal
    (3, 4): 2.0000    # Side
}

EDGE_PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
OUTPUT_DIR = "analysis_results_swarm_static"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. CLI ARGUMENT PARSING
# ==============================================================================
if len(sys.argv) < 2:
    print("Usage: python3 analyze_swarm_static_comprehensive.py <filename.csv>")
    print("[INFO] No file specified, searching for default files...")
    candidates = ["swarm_static_2m.csv", "swarm_static_2.csv", "ss.csv",
                  "03_swarm_static_2m.csv"]
    target_file = next((f for f in candidates if os.path.exists(f)), None)
    if not target_file:
        print("[ERROR] No CSV file found!")
        sys.exit(1)
else:
    target_file = sys.argv[1]

if not os.path.exists(target_file):
    print(f"[ERROR] File '{target_file}' does not exist!")
    sys.exit(1)

print(f"[INFO] Processing dataset: {target_file}")


# ==============================================================================
# 3. ROBUST CSV LOADER  --  FIXED
# ==============================================================================
def _clean_line(line: str) -> str:
    """Strip monitor prompt prefixes, comments and whitespace."""
    l = line.strip()
    # Remove any "...$ command" or "uart> " prefixes (take content after last '>')
    if ">" in l:
        l = l.split(">")[-1].strip()
    # Also strip shell-prompt prefixes like "user@host:~$ "
    if "$" in l:
        # Only strip if it looks like a prompt (contains $ followed by python/comma data)
        parts = l.split("$")
        # keep the part after the last '$' only if the preceding text has no commas
        if parts[0].count(",") == 0:
            l = parts[-1].strip()
    return l


def load_uwb_csv(path: str) -> pd.DataFrame:
    """
    Load UWB swarm CSV with robust handling of:
      * UTF-8 BOM  (utf-8-sig)
      * monitor prompt prefixes and comment lines
      * a header that may or may not be present
    """
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        raw_lines = f.readlines()

    header_line = None
    header_idx = -1
    for i, raw in enumerate(raw_lines):
        l = _clean_line(raw)
        if not l or l.startswith("#"):
            continue
        # The header is the first line that mentions 'timestamp' AND has commas
        if "timestamp" in l.lower() and "," in l:
            header_line = l
            header_idx = i
            break

    data_lines = []
    start = header_idx + 1 if header_idx >= 0 else 0
    for raw in raw_lines[start:]:
        l = _clean_line(raw)
        if not l or l.startswith("#"):
            continue
        if l.lower().startswith("timestamp"):
            continue  # skip duplicate headers
        data_lines.append(l)

    if not data_lines:
        print("[ERROR] No valid data records found in file!")
        sys.exit(1)

    if header_line is not None:
        raw_csv = header_line + "\n" + "\n".join(data_lines)
        df = pd.read_csv(StringIO(raw_csv), engine="python")
    else:
        df = pd.read_csv(StringIO("\n".join(data_lines)), header=None, engine="python")

    return df


df = load_uwb_csv(target_file)

# ---- Normalise column names -------------------------------------------------
df.columns = [str(c).strip().lower() for c in df.columns]

# Aliases -> canonical names
col_rename = {
    "timestamp":   "timestamp_ms",
    "time":        "timestamp_ms",
    "frame":       "frame_id",
    "init":        "init_id",
    "initiator":   "init_id",
    "initiator_id":"init_id",
    "resp":        "resp_id",
    "responder":   "resp_id",
    "responder_id":"resp_id",
    "raw_m":       "dist_uncomp_m",
    "corr_m":      "dist_clock_m",
    "calib_m":     "dist_calib_m",
    "cfo":         "cfo_ppm",
    "rssi":        "rssi_dbm",
}
df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns},
          inplace=True)

# If the file had no header at all, fall back to positional naming.
if "init_id" not in df.columns or "resp_id" not in df.columns:
    n = len(df.columns)
    fallback = ["timestamp_ms", "frame_id", "init_id", "resp_id", "status",
                "carrier_int", "cfo_ppm", "tof_uncomp", "tof_comp",
                "dist_uncomp_m", "dist_clock_m", "dist_rssi_m",
                "dist_calib_m", "rssi_dbm"]
    fallback += [f"col{i}" for i in range(len(fallback), n)]
    df.columns = fallback[:n]

# ---- CRITICAL FIX: force numeric types on identity/status columns -----------
for col in ("init_id", "resp_id", "frame_id", "status"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows without a valid initiator / responder (e.g. the 0,0 filler rows)
df = df.dropna(subset=["init_id", "resp_id"])
df["init_id"] = df["init_id"].astype(int)
df["resp_id"] = df["resp_id"].astype(int)
if "status" in df.columns:
    df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(0).astype(int)

# ---- Timestamps -------------------------------------------------------------
c_time = "timestamp_ms" if "timestamp_ms" in df.columns else df.columns[0]
df[c_time] = pd.to_numeric(df[c_time], errors="coerce")
df = df.dropna(subset=[c_time]).reset_index(drop=True)

# Distances are also coerced to numeric
for col in ("dist_uncomp_m", "dist_clock_m", "dist_calib_m",
            "dist_rssi_m", "cfo_ppm"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

t0 = df[c_time].iloc[0]
df["time_sec"] = (df[c_time] - t0) / 1000.0
time_sec = df["time_sec"]

# ---- Debug sanity print -----------------------------------------------------
print(f"[DEBUG] Parsed columns : {list(df.columns)}")
print(f"[DEBUG] init_id uniques: {sorted(df['init_id'].unique().tolist())}")
print(f"[DEBUG] resp_id uniques: {sorted(df['resp_id'].unique().tolist())}")

# ==============================================================================
# 4. STATISTICAL & METROLOGY ANALYSIS ENGINE
# ==============================================================================
report = []
def log(s=""):
    print(s)
    report.append(s)

log("=" * 115)
log("   ANJOMAN 3000-FRAME UWB METROLOGY REPORT & KALMAN FILTER CALIBRATION")
log(f"   Input File: {target_file} | Total Records: {len(df)} | "
    f"Duration: {time_sec.max():.1f} s ({time_sec.max()/60.0:.1f} min)")
log("=" * 115)
header = (f"{'Link':<8}|{'Target (m)':<11}|{'PDR (%)':<9}|{'Uncomp (m)':<12}|"
          f"{'ClockComp (m)':<14}|{'Residual Bias':<14}|{'Noise σ (cm)':<13}|"
          f"{'CFO (ppm)'}")
log(header)
log("-" * 115)

stats_summary = {}
R_matrix = np.zeros((4, 4))
sigma_matrix = np.zeros((4, 4))

for u, v in EDGE_PAIRS:
    sub = df[(df["init_id"] == u) & (df["resp_id"] == v)].copy()
    attempts = len(sub)

    # Determine "successful" rows
    if "status" in sub.columns:
        valid_sub = sub[sub["status"] == 1].copy()
    elif "dist_clock_m" in sub.columns:
        valid_sub = sub[sub["dist_clock_m"] > 0.1].copy()
    else:
        valid_sub = sub.copy()

    successes = len(valid_sub)
    pdr = (successes / attempts * 100.0) if attempts > 0 else 0.0
    gt = GROUND_TRUTH.get((u, v), 2.000)

    if successes > 5:
        c_uncomp = "dist_uncomp_m" if "dist_uncomp_m" in valid_sub.columns else None
        c_clock = ("dist_clock_m" if "dist_clock_m" in valid_sub.columns
                   else valid_sub.columns[6])
        c_cfo = "cfo_ppm" if "cfo_ppm" in valid_sub.columns else None

        mean_uncomp = (valid_sub[c_uncomp].mean()
                       if c_uncomp and c_uncomp in valid_sub.columns else np.nan)
        mean_clock = valid_sub[c_clock].mean()
        std_clock = valid_sub[c_clock].std()

        # Steady-state window: after 300 s, else last half
        ss_sub = valid_sub[valid_sub["time_sec"] >= 300.0]
        if len(ss_sub) < 10:
            ss_sub = valid_sub.iloc[int(len(valid_sub) * 0.5):]

        ss_mean = ss_sub[c_clock].mean()
        ss_std = ss_sub[c_clock].std()

        # Robust sigma via MAD
        med = np.median(ss_sub[c_clock])
        mad = np.median(np.abs(ss_sub[c_clock] - med))
        robust_sigma = mad * 1.4826

        residual_bias_cm = (ss_mean - gt) * 100.0
        cfo_mean = (valid_sub[c_cfo].mean()
                    if c_cfo and c_cfo in valid_sub.columns else np.nan)

        R_matrix[u-1, v-1] = R_matrix[v-1, u-1] = ss_std ** 2
        sigma_matrix[u-1, v-1] = sigma_matrix[v-1, u-1] = ss_std

        stats_summary[(u, v)] = {
            "attempts": attempts,
            "successes": successes,
            "pdr": pdr,
            "gt": gt,
            "mean_uncomp": mean_uncomp,
            "mean_clock": mean_clock,
            "ss_mean": ss_mean,
            "ss_std_cm": ss_std * 100.0,
            "robust_sigma_cm": robust_sigma * 100.0,
            "bias_cm": residual_bias_cm,
            "cfo_mean": cfo_mean,
            "valid_sub": valid_sub,
            "ss_sub": ss_sub,
        }

        uncomp_str = f"{mean_uncomp:>9.3f} m" if not np.isnan(mean_uncomp) else "   N/A   "
        cfo_str = f"{cfo_mean:>+8.2f}" if not np.isnan(cfo_mean) else "   N/A  "
        log(f"R{u}-R{v:<4}| {gt:<9.4f} | {pdr:>6.1f} %  | {uncomp_str} | "
            f"{ss_mean:>10.4f} m   | {residual_bias_cm:>+9.2f} cm   | "
            f"{ss_std*100.0:>8.2f} cm    | {cfo_str}")
    else:
        stats_summary[(u, v)] = {"pdr": 0.0, "successes": 0, "attempts": attempts}
        log(f"R{u}-R{v:<4}| {gt:<9.4f} | {pdr:>6.1f} %  |    DOWN   |      DOWN     |"
            f"      DOWN      |     DOWN     |    N/A  ")

log("-" * 115)
log("\n" + "=" * 115)
log("   EKF MEASUREMENT NOISE COVARIANCE MATRIX R (m^2) [Link-Specific]")
log("=" * 115)
for i in range(4):
    row_str = "  ".join([f"{R_matrix[i, j]:8.5f}" if R_matrix[i, j] > 0
                         else "  0.00000 " for j in range(4)])
    log(f"   R[{i+1}, :] = [ {row_str} ]")

log("=" * 115)
log("\n" + "=" * 115)
log("   RECOMMENDED MEASUREMENT STD DEVIATION MATRIX σ (cm)")
log("=" * 115)
for i in range(4):
    row_str = "  ".join([f"{sigma_matrix[i, j]*100.0:6.2f} cm"
                         if sigma_matrix[i, j] > 0 else "   0.00 cm "
                         for j in range(4)])
    log(f"   σ[{i+1}, :] = [ {row_str} ]")
log("=" * 115)

with open(os.path.join(OUTPUT_DIR, "uwb_comprehensive_report.txt"), "w") as f_out:
    f_out.write("\n".join(report) + "\n")

pd.DataFrame(R_matrix, index=[1, 2, 3, 4], columns=[1, 2, 3, 4]).to_csv(
    os.path.join(OUTPUT_DIR, "uwb_noise_covariance_R.csv"))

# ==============================================================================
# 5. PLOTTING ENGINE
# ==============================================================================
colors = {(1, 2): "#1f77b4", (1, 3): "#ff7f0e", (1, 4): "#2ca02c",
          (2, 3): "#d62728", (2, 4): "#9467bd", (3, 4): "#8c564b"}

active_pairs = [p for p in EDGE_PAIRS if stats_summary[p]["successes"] > 10]

# --- FIGURE 1: CLOCK-COMPENSATION PROOF --------------------------------------
if active_pairs:
    fig, axes = plt.subplots(len(active_pairs), 1,
                             figsize=(16, 3.5 * len(active_pairs)), sharex=True)
    if len(active_pairs) == 1:
        axes = [axes]

    for idx, pair in enumerate(active_pairs):
        ax = axes[idx]
        sdata = stats_summary[pair]["valid_sub"]
        t_m = sdata["time_sec"] / 60.0
        gt = stats_summary[pair]["gt"]

        ax.axhline(gt, color="black", linestyle="--", linewidth=1.5,
                   label=f"Ground Truth ({gt:.3f} m)")
        if "dist_uncomp_m" in sdata.columns:
            ax.plot(t_m, sdata["dist_uncomp_m"], color="gray", alpha=0.5,
                    linestyle=":",
                    label=f"Uncompensated SS-TWR "
                          f"(Mean={sdata['dist_uncomp_m'].mean():.2f}m)")

        c_clock = ("dist_clock_m" if "dist_clock_m" in sdata.columns
                   else sdata.columns[6])
        ax.plot(t_m, sdata[c_clock], color=colors.get(pair, "blue"),
                linewidth=1.8,
                label=f"Clock-Compensated "
                      f"(Steady Mean={stats_summary[pair]['ss_mean']:.3f}m, "
                      f"Bias={stats_summary[pair]['bias_cm']:+.1f}cm)")

        ax.set_title(f"Link R{pair[0]}-R{pair[1]} | "
                     f"Carrier Integrator Correction Effect", fontsize=12)
        ax.set_ylabel("Range (m)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper right", fontsize=10)

    axes[-1].set_xlabel("Elapsed Time (minutes)", fontsize=12)
    plt.suptitle("Proof of Clock-Offset Compensation Across 10-Minute Run",
                 fontsize=15, y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_clock_compensation_proof.png"), dpi=300)
    plt.close()

# --- FIGURE 2: THERMAL & CFO TRANSIENTS --------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

for pair in active_pairs:
    sdata = stats_summary[pair]["valid_sub"]
    t_m = sdata["time_sec"] / 60.0

    if "temp_esp_init" in sdata.columns:
        ax1.plot(t_m, sdata["temp_esp_init"],
                 label=f"R{pair[0]} ESP Temp", linewidth=1.5)
    if "cfo_ppm" in sdata.columns:
        ax2.plot(t_m, sdata["cfo_ppm"], color=colors.get(pair, "blue"),
                 label=f"R{pair[0]}-R{pair[1]} CFO (ppm)", linewidth=1.5)

ax1.set_title("Long-Term Temperature Rise Curve "
              "(Thermal Transient Beyond 3 Min)", fontsize=13)
ax1.set_ylabel("Temperature (°C)", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.7)
if ax1.get_legend_handles_labels()[0]:
    ax1.legend(loc="lower right", fontsize=10)

ax2.set_title("Carrier Frequency Offset (CFO in PPM) Dynamics Over Time",
              fontsize=13)
ax2.set_xlabel("Elapsed Time (minutes)", fontsize=12)
ax2.set_ylabel("Clock Offset (PPM)", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.7)
if ax2.get_legend_handles_labels()[0]:
    ax2.legend(loc="upper right", fontsize=10)

plt.suptitle("Thermal Equilibrium & Crystal Clock Frequency Evolution",
             fontsize=15, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_thermal_and_cfo_transients.png"), dpi=300)
plt.close()

# --- FIGURE 3: STEADY-STATE NOISE HISTOGRAMS ---------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
axes = axes.flatten()

for idx, pair in enumerate(EDGE_PAIRS):
    ax = axes[idx]
    if pair in active_pairs and len(stats_summary[pair]["ss_sub"]) > 10:
        sdata = stats_summary[pair]["ss_sub"]
        c_clock = ("dist_clock_m" if "dist_clock_m" in sdata.columns
                   else sdata.columns[6])
        errors_cm = (sdata[c_clock] - stats_summary[pair]["gt"]) * 100.0

        ax.hist(errors_cm, bins=30, color=colors.get(pair, "blue"),
                edgecolor="black", alpha=0.75, density=True)
        ax.axvline(0, color="darkred", linestyle="--", linewidth=1.5,
                   label="Ground Truth Zero")
        ax.axvline(stats_summary[pair]["bias_cm"], color="black",
                   linestyle=":", linewidth=1.8,
                   label=f"Bias: {stats_summary[pair]['bias_cm']:+.1f}cm")

        ax.set_title(f"Link R{pair[0]}-R{pair[1]} Error "
                     f"(σ = {stats_summary[pair]['ss_std_cm']:.2f} cm)", fontsize=11)
        ax.set_xlabel("Range Error (cm)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)
    else:
        ax.text(0.5, 0.5, "NO DATA / LINK DOWN", ha="center", va="center",
                fontsize=12, color="red")
        ax.set_title(f"Link R{pair[0]}-R{pair[1]} [OFFLINE]", fontsize=11)

plt.suptitle("Steady-State Gaussian Noise Profiles & Measurement Variance (R)",
             fontsize=15, y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_steady_state_noise_distribution.png"),
            dpi=300)
plt.close()

# --- FIGURE 4: NETWORK PDR HEALTH --------------------------------------------
plt.figure(figsize=(10, 6), dpi=300)
labels = [f"R{u}-R{v}" for u, v in EDGE_PAIRS]
pdrs = [stats_summary[p]["pdr"] for p in EDGE_PAIRS]
bar_colors = ["#2ca02c" if p > 85 else ("#ff7f0e" if p > 50 else "#d62728")
              for p in pdrs]

bars = plt.bar(labels, pdrs, color=bar_colors, width=0.5, edgecolor="black")
plt.axhline(100.0, color="gray", linestyle="--", linewidth=1.0)
plt.title("Swarm Full-Mesh TDMA Reliability (Packet Delivery Ratio %)",
          fontsize=13, pad=12)
plt.ylabel("PDR (%)", fontsize=11)
plt.ylim(0, 115)
plt.grid(True, linestyle=":", alpha=0.6, axis="y")
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, y + 2.0, f"{y:.1f}%",
             ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_network_pdr_health.png"), dpi=300)
plt.close()

log(f"\n[SUCCESS] Generated report and 4 multi-panel plots in: {OUTPUT_DIR}/")
log(f"[INFO] Measurement Covariance Matrix R saved to: "
    f"{OUTPUT_DIR}/uwb_noise_covariance_R.csv")
