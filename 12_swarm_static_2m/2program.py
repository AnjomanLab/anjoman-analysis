#!/usr/bin/env python3
"""
Anjoman Swarm Firmware — UWB Metrology, Calibration & Noise Analyzer

Usage:
    python3 analyze_swarm_static_comprehensive.py <filename.csv>

Pipeline (correct UWB estimator design order):
    1. Clock-offset correction               (firmware)
    2. Antenna / timestamp calibration       (THIS SCRIPT — Section 2)
    3. Thermal characterisation              (THIS SCRIPT — Section 5)
    4. Residual bias compensation            (THIS SCRIPT — Section 3)
    5. Residual noise + autocorrelation      (THIS SCRIPT — Sections 3,4)
    6. Measurement covariance R (6x6)        (THIS SCRIPT — Section 6)
    7. Simple range filter                   (downstream)
    8. EKF / IEKF                            (downstream — NOT here)

NOTE: This script does NOT run a Kalman filter.  It characterises the
      UWB sensor so that a later EKF can be fed a valid R and bias model.
"""

import os
import sys
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# 0. GEOMETRY  (2 m x 2 m square)
# ============================================================================
GROUND_TRUTH = {
    (1, 2): 2.0000,   # side
    (1, 3): 2.8284,   # diagonal
    (1, 4): 2.0000,   # side
    (2, 3): 2.0000,   # side
    (2, 4): 2.8284,   # diagonal
    (3, 4): 2.0000,   # side
}
EDGE_PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
LINK_INDEX = {p: i for i, p in enumerate(EDGE_PAIRS)}
NODES      = [1, 2, 3, 4]
CORE_PAIRS = [(2, 3), (2, 4), (3, 4)]

STEADY_STATE_START_S = 300.0
OUTPUT_DIR = "analysis_results_swarm_static"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 1. CLI
# ============================================================================
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


# ============================================================================
# 2. ROBUST CSV LOADER
# ============================================================================
def _clean_line(line: str) -> str:
    l = line.strip()
    if ">" in l:
        l = l.split(">")[-1].strip()
    if "$" in l:
        parts = l.split("$")
        if parts[0].count(",") == 0:
            l = parts[-1].strip()
    return l


def load_uwb_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        raw = f.readlines()

    header_line, header_idx = None, -1
    for i, line in enumerate(raw):
        l = _clean_line(line)
        if not l or l.startswith("#"):
            continue
        if "timestamp" in l.lower() and "," in l:
            header_line, header_idx = l, i
            break

    data = []
    start = header_idx + 1 if header_idx >= 0 else 0
    for line in raw[start:]:
        l = _clean_line(line)
        if not l or l.startswith("#") or l.lower().startswith("timestamp"):
            continue
        data.append(l)

    if not data:
        print("[ERROR] No valid data records found in file!")
        sys.exit(1)

    if header_line is not None:
        csv_text = header_line + "\n" + "\n".join(data)
        df = pd.read_csv(StringIO(csv_text), engine="python")
    else:
        df = pd.read_csv(StringIO("\n".join(data)), header=None, engine="python")
    return df


df = load_uwb_csv(target_file)
df.columns = [str(c).strip().lower() for c in df.columns]

# ---- Column normalisation ---------------------------------------------------
rename = {
    "timestamp": "timestamp_ms", "time": "timestamp_ms",
    "frame": "frame_id",
    "init": "init_id", "initiator": "init_id", "initiator_id": "init_id",
    "resp": "resp_id", "responder": "resp_id", "responder_id": "resp_id",
    "raw_m": "dist_uncomp_m", "corr_m": "dist_clock_m", "calib_m": "dist_calib_m",
    "cfo": "cfo_ppm", "rssi": "rssi_dbm",
}
df.rename(columns={k: v for k, v in rename.items() if k in df.columns},
          inplace=True)

if "init_id" not in df.columns or "resp_id" not in df.columns:
    n = len(df.columns)
    fallback = ["timestamp_ms", "frame_id", "init_id", "resp_id", "status",
                "carrier_int", "cfo_ppm", "tof_uncomp", "tof_comp",
                "dist_uncomp_m", "dist_clock_m", "dist_rssi_m",
                "dist_calib_m", "rssi_dbm", "tTx1", "tRx1", "tRx2", "tTx2",
                "temp_uwb_init", "temp_uwb_resp", "temp_esp_init", "temp_esp_resp"]
    fallback += [f"col{i}" for i in range(len(fallback), n)]
    df.columns = fallback[:n]

# ---- Force numeric on identity columns (CRITICAL FIX) -----------------------
for col in ("init_id", "resp_id", "frame_id", "status"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["init_id", "resp_id"]).reset_index(drop=True)
df["init_id"] = df["init_id"].astype(int)
df["resp_id"] = df["resp_id"].astype(int)
if "status" in df.columns:
    df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(0).astype(int)

c_time = "timestamp_ms" if "timestamp_ms" in df.columns else df.columns[0]
df[c_time] = pd.to_numeric(df[c_time], errors="coerce")
df = df.dropna(subset=[c_time]).reset_index(drop=True)

numeric_cols = ("dist_uncomp_m", "dist_clock_m", "dist_calib_m", "dist_rssi_m",
                "cfo_ppm", "carrier_int",
                "temp_uwb_init", "temp_uwb_resp", "temp_esp_init", "temp_esp_resp")
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["time_sec"] = (df[c_time] - df[c_time].iloc[0]) / 1000.0

print(f"[DEBUG] columns        : {list(df.columns)}")
print(f"[DEBUG] init_id uniques: {sorted(df['init_id'].unique())}")
print(f"[DEBUG] resp_id uniques: {sorted(df['resp_id'].unique())}")


# ============================================================================
# 3. REPORT BUFFER
# ============================================================================
report = []
def log(s=""):
    print(s)
    report.append(s)


# ============================================================================
# 4. PER-LINK ANALYSIS (raw + steady-state)
# ============================================================================
link = {p: {"gt": GROUND_TRUTH[p]} for p in EDGE_PAIRS}

for pair in EDGE_PAIRS:
    u, v = pair
    sub = df[(df["init_id"] == u) & (df["resp_id"] == v)].copy()
    attempts = len(sub)

    if "status" in sub.columns:
        valid = sub[sub["status"] == 1].copy()
    elif "dist_clock_m" in sub.columns:
        valid = sub[sub["dist_clock_m"] > 0.1].copy()
    else:
        valid = sub.copy()

    successes = len(valid)
    pdr = (successes / attempts * 100.0) if attempts else 0.0

    link[pair].update({"attempts": attempts, "successes": successes,
                       "pdr": pdr, "valid": valid})

    if successes < 5:
        continue

    c_clock  = ("dist_clock_m" if "dist_clock_m" in valid.columns
                else ("dist_calib_m" if "dist_calib_m" in valid.columns
                      else valid.columns[6]))
    c_uncomp = "dist_uncomp_m" if "dist_uncomp_m" in valid.columns else None
    c_cfo    = "cfo_ppm" if "cfo_ppm" in valid.columns else None

    ss = valid[valid["time_sec"] >= STEADY_STATE_START_S]
    if len(ss) < 10:
        ss = valid.iloc[int(len(valid) * 0.5):]

    ss_mean = ss[c_clock].mean()
    ss_std  = ss[c_clock].std()

    mad = np.median(np.abs(ss[c_clock] - np.median(ss[c_clock])))
    robust_sigma = mad * 1.4826

    link[pair].update({
        "c_clock": c_clock, "c_uncomp": c_uncomp, "c_cfo": c_cfo,
        "mean_uncomp": (valid[c_uncomp].mean() if c_uncomp else np.nan),
        "mean_clock_all": valid[c_clock].mean(),
        "ss": ss,
        "ss_mean": ss_mean,
        "ss_std": ss_std,
        "ss_std_cm": ss_std * 100.0,
        "robust_sigma_cm": robust_sigma * 100.0,
        "raw_bias_m": ss_mean - GROUND_TRUTH[pair],
        "raw_bias_cm": (ss_mean - GROUND_TRUTH[pair]) * 100.0,
        "cfo_mean": (valid[c_cfo].mean() if c_cfo else np.nan),
    })


# ============================================================================
# 5. DEVICE-SPECIFIC ADDITIVE BIAS MODEL  z_ij = d_ij + b_i + b_j
# ============================================================================
def fit_additive_bias(excess):
    """LSQ fit of b_i + b_j = excess_ij with sum(b) = 0 constraint."""
    keys = list(excess.keys())
    if len(keys) < 3:
        return None, None
    A = np.zeros((len(keys), 4))
    y = np.zeros(len(keys))
    for k, (u, v) in enumerate(keys):
        A[k, u - 1] = 1.0
        A[k, v - 1] = 1.0
        y[k] = excess[(u, v)]
    A_aug = np.vstack([A, np.ones((1, 4))])
    y_aug = np.concatenate([y, [0.0]])
    b, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
    fit_res = {pair: y[k] - (b[pair[0]-1] + b[pair[1]-1])
               for k, pair in enumerate(keys)}
    return b, fit_res


excess_all  = {p: link[p]["raw_bias_m"] for p in EDGE_PAIRS
               if link[p]["successes"] >= 5}
excess_core = {p: excess_all[p] for p in CORE_PAIRS if p in excess_all}

b_all, res_all = fit_additive_bias(excess_all)

# Core-swarm exact 3x3 solution (R1 excluded)
b_core = None
if len(excess_core) == 3:
    y23, y24, y34 = (excess_core[(2,3)], excess_core[(2,4)],
                     excess_core[(3,4)])
    b2 = (y23 + y24 - y34) / 2.0
    b3 = (y23 + y34 - y24) / 2.0
    b4 = (y24 + y34 - y23) / 2.0
    b_core = np.array([np.nan, b2, b3, b4])

# b1 estimates from each R1 link (assuming core b2/b3/b4)
b1_est = {}
if b_core is not None:
    for (u, v) in [(1,2), (1,3), (1,4)]:
        if (u, v) in excess_all:
            b1_est[(u, v)] = excess_all[(u, v)] - b_core[v - 1]

# Robust reference b1 = median of estimates (immune to a single outlier)
b1_ref = float(np.median(list(b1_est.values()))) if b1_est else 0.0
b1_spread = (max(b1_est.values()) - min(b1_est.values())) if b1_est else 0.0
r1_anomaly = b1_spread > 1.0   # > 1 m inconsistency = model failure for R1


# ============================================================================
# 6. RESIDUAL NOISE AFTER BIAS REMOVAL
# ============================================================================
for pair in EDGE_PAIRS:
    ld = link[pair]
    if ld["successes"] < 5:
        continue
    u, v = pair

    if b_core is not None and u >= 2:
        bi, bj = b_core[u-1], b_core[v-1]
    elif b_core is not None and u == 1:
        bi, bj = b1_ref, b_core[v-1]
    elif b_all is not None:
        bi, bj = b_all[u-1], b_all[v-1]
    else:
        bi = bj = 0.0

    ss = ld["ss"]
    resid = ss[ld["c_clock"]] - (ld["gt"] + bi + bj)

    ld.update({
        "bias_i": bi, "bias_j": bj,
        "residual": resid,
        "res_mean_cm": resid.mean() * 100.0,
        "res_std_cm":  resid.std()  * 100.0,
        "res_mad_cm":  np.median(np.abs(resid - np.median(resid)))
                       * 1.4826 * 100.0,
    })


# ============================================================================
# 7. AUTOCORRELATION (whiteness test)
# ============================================================================
def acf(x, max_lag=20):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 3:
        return np.array([1.0])
    denom = np.sum(x * x)
    if denom == 0:
        return np.zeros(min(max_lag, n - 1) + 1)
    lags = np.arange(0, min(max_lag, n - 1) + 1)
    return np.array([np.sum(x[:n-k] * x[k:]) / denom for k in lags])


for pair in EDGE_PAIRS:
    ld = link[pair]
    if ld["successes"] >= 50 and "residual" in ld:
        ld["acf"] = acf(ld["residual"].values, max_lag=20)


# ============================================================================
# 8. THERMAL CORRELATION
# ============================================================================
for pair in EDGE_PAIRS:
    ld = link[pair]
    if ld["successes"] < 50:
        continue
    valid = ld["valid"]
    if "temp_uwb_init" not in valid.columns:
        continue
    T = valid["temp_uwb_init"].values

    if ld["c_cfo"]:
        F = valid[ld["c_cfo"]].values
        ok = np.isfinite(T) & np.isfinite(F)
        if ok.sum() > 20 and T[ok].std() > 0.5:
            slope, _ = np.polyfit(T[ok], F[ok], 1)
            ld["cfo_vs_T_slope"] = slope
            ld["cfo_vs_T_corr"]  = np.corrcoef(T[ok], F[ok])[0, 1]

    R = valid[ld["c_clock"]].values
    ok = np.isfinite(T) & np.isfinite(R)
    if ok.sum() > 20 and T[ok].std() > 0.5:
        slope, _ = np.polyfit(T[ok], R[ok], 1)
        ld["range_vs_T_slope_cm_per_C"] = slope * 100.0
        ld["range_vs_T_corr"]           = np.corrcoef(T[ok], R[ok])[0, 1]


# ============================================================================
# 9. 6x6 MEASUREMENT COVARIANCE MATRIX  (link-indexed)
# ============================================================================
R6 = np.full((6, 6), np.nan)

for pair in EDGE_PAIRS:
    ld = link[pair]
    if "residual" in ld and len(ld["residual"]) > 5:
        R6[LINK_INDEX[pair], LINK_INDEX[pair]] = ld["residual"].std() ** 2

# Cross-covariance from frame-aligned residuals
aligned = {}
for pair in EDGE_PAIRS:
    ld = link[pair]
    if "residual" not in ld:
        continue
    ss = ld["ss"]
    if "frame_id" not in ss.columns:
        continue
    tmp = pd.DataFrame({"frame_id": ss["frame_id"].values,
                        "r": ld["residual"].values})
    aligned[pair] = tmp.groupby("frame_id")["r"].mean()

for i, p1 in enumerate(EDGE_PAIRS):
    for j, p2 in enumerate(EDGE_PAIRS):
        if i >= j:
            continue
        if p1 in aligned and p2 in aligned:
            j_ = pd.concat([aligned[p1].rename("r1"), aligned[p2].rename("r2")],
                           axis=1, join="inner").dropna()
            if len(j_) > 20:
                cov = np.cov(j_["r1"], j_["r2"])[0, 1]
                R6[i, j] = R6[j, i] = cov

# Fill only genuinely unknown off-diagonals with 0 (uncorrelated assumption)
off = ~np.eye(6, dtype=bool)
R6[off & np.isnan(R6)] = 0.0


# ============================================================================
# 10. REPORT
# ============================================================================
log("=" * 118)
log("   ANJOMAN SWARM — UWB METROLOGY, CALIBRATION & NOISE COVARIANCE REPORT")
log(f"   Input File: {target_file}")
log(f"   Total Records: {len(df)} | Duration: "
    f"{df['time_sec'].max():.1f} s ({df['time_sec'].max()/60:.2f} min)")
log("=" * 118)
log("")
log("   This pipeline does NOT implement a Kalman filter.  It characterises")
log("   the UWB sensor so that a later EKF can be fed a valid R and bias.")
log("")

# ---- S1: raw link stats -----------------------------------------------------
log("=" * 118)
log("   SECTION 1 — RAW LINK STATISTICS (after clock-offset correction)")
log("=" * 118)
log(f"{'Link':<8}|{'GT (m)':<9}|{'PDR(%)':<8}|{'Uncomp(m)':<11}|"
    f"{'ClockComp (m)':<15}|{'Raw bias (cm)':<15}|{'σ_raw (cm)':<12}|"
    f"{'CFO(ppm)':<10}")
log("-" * 118)
for pair in EDGE_PAIRS:
    u, v = pair
    ld = link[pair]
    if ld["successes"] < 5:
        log(f"R{u}-R{v:<4}| {ld['gt']:<9.4f}| {ld['pdr']:>6.1f} |    DOWN     |"
            f"      DOWN      |      DOWN       |     DOWN      |    N/A")
        continue
    unc = (f"{ld['mean_uncomp']:>7.3f} m"
           if np.isfinite(ld['mean_uncomp']) else "   N/A   ")
    cfo = (f"{ld['cfo_mean']:>+7.2f}"
           if np.isfinite(ld['cfo_mean']) else "  N/A  ")
    log(f"R{u}-R{v:<4}| {ld['gt']:<9.4f}| {ld['pdr']:>6.1f} | {unc} |"
        f" {ld['ss_mean']:>10.4f} m    | {ld['raw_bias_cm']:>+10.2f} cm  |"
        f" {ld['ss_std_cm']:>8.2f}   | {cfo}")
log("-" * 118)
log("   ⚠  Raw bias in the tens of metres is SYSTEMATIC, not noise.  No")
log("      filter can fix a 30–45 m offset.  It MUST be calibrated out first.")
log("")

# ---- S2: device bias model --------------------------------------------------
log("=" * 118)
log("   SECTION 2 — DEVICE-SPECIFIC ADDITIVE BIAS MODEL   z_ij = d_ij + b_i + b_j")
log("=" * 118)
log("   (b_i is in metres.  Identifiability requires sum(b) = 0.)")
log("")
if b_all is not None:
    log(f"   Full-model LSQ fit (all 6 links, sum=0):")
    for i in range(4):
        log(f"      b_{i+1} = {b_all[i]:+9.4f}  m")
    log("   Fit residuals (measured − predicted):")
    for pair in EDGE_PAIRS:
        if pair in (res_all or {}):
            log(f"      R{pair[0]}-R{pair[1]} : {res_all[pair]*100:+9.2f} cm")
log("")
if b_core is not None:
    log(f"   Core-swarm exact solution (R2/R3/R4 only — R1 excluded):")
    log(f"      b_2 = {b_core[1]:+9.4f}  m")
    log(f"      b_3 = {b_core[2]:+9.4f}  m")
    log(f"      b_4 = {b_core[3]:+9.4f}  m")
    log("")
    log(f"   b_1 estimates from R1's three links (should agree if model holds):")
    for pair, val in b1_est.items():
        log(f"      from R{pair[0]}-R{pair[1]} : b_1 ≈ {val:+9.4f} m")
    log(f"   → spread = {b1_spread:.4f} m")
    if r1_anomaly:
        log("   ⚠  R1 IS INCONSISTENT WITH THE MODEL.  R1 must be calibrated")
        log("      separately — likely a TX/RX delay asymmetry or a different")
        log("      timestamp reference inside R1.")
    else:
        log("   ✓  R1 is consistent with the additive-bias model.")
log("=" * 118)

# ---- S3: residuals after bias removal ---------------------------------------
log("")
log("=" * 118)
log("   SECTION 3 — RESIDUAL NOISE AFTER BIAS REMOVAL  (the REAL σ)")
log("=" * 118)
log(f"{'Link':<8}|{'GT (m)':<9}|{'b_i(m)':<10}|{'b_j(m)':<10}|"
    f"{'Mean resid(cm)':<16}|{'σ_resid (cm)':<14}|{'MAD-σ (cm)':<12}")
log("-" * 118)
for pair in EDGE_PAIRS:
    ld = link[pair]
    if "residual" not in ld:
        continue
    u, v = pair
    log(f"R{u}-R{v:<4}| {ld['gt']:<9.4f}| {ld['bias_i']:>+8.4f} | "
        f"{ld['bias_j']:>+8.4f} | {ld['res_mean_cm']:>+12.2f}   | "
        f"{ld['res_std_cm']:>10.2f}     | {ld['res_mad_cm']:>8.2f}")
log("-" * 118)
log("   Large residual mean = model failure for that link (typically R1).")
log("   Small residual mean + small σ = link ready for EKF.")
log("")

# ---- S4: whiteness ---------------------------------------------------------
log("=" * 118)
log("   SECTION 4 — WHITENESS TEST (residual autocorrelation)")
log("=" * 118)
log("   Lag-1 ACF ≈ 0   → white noise   → can be modelled as R")
log("   Lag-1 ACF > 0.3 → time-correlated → augment state (bias state)")
log("")
log(f"{'Link':<8}|{'ACF[1]':<10}|{'ACF[5]':<10}|{'ACF[10]':<10}|{'Verdict':<30}")
log("-" * 118)
for pair in EDGE_PAIRS:
    ld = link[pair]
    if "acf" not in ld:
        continue
    a = ld["acf"]
    l1  = a[1]  if len(a) > 1  else np.nan
    l5  = a[5]  if len(a) > 5  else np.nan
    l10 = a[10] if len(a) > 10 else np.nan
    if not np.isfinite(l1):
        verdict = "insufficient data"
    elif abs(l1) < 0.15:
        verdict = "WHITE  (OK for R)"
    elif abs(l1) < 0.35:
        verdict = "mild correlation"
    else:
        verdict = "CORRELATED (augment state)"
    log(f"R{pair[0]}-R{pair[1]:<4}| {l1:>+8.3f} | {l5:>+8.3f} | "
        f"{l10:>+8.3f} | {verdict:<30}")
log("-" * 118)

# ---- S5: thermal -----------------------------------------------------------
log("")
log("=" * 118)
log("   SECTION 5 — THERMAL CORRELATION   (dCFO/dT  and  dRange/dT)")
log("=" * 118)
log(f"{'Link':<8}|{'dCFO/dT (ppm/°C)':<22}|{'corr':<8}|"
    f"{'dRange/dT (cm/°C)':<22}|{'corr':<8}")
log("-" * 118)
for pair in EDGE_PAIRS:
    ld = link[pair]
    if ld["successes"] < 50:
        continue
    sc, rc = ld.get("cfo_vs_T_slope"), ld.get("cfo_vs_T_corr")
    sr, rr = ld.get("range_vs_T_slope_cm_per_C"), ld.get("range_vs_T_corr")
    sc_s = f"{sc:>+10.4f}" if sc is not None else "    N/A   "
    rc_s = f"{rc:>+7.3f}"  if rc is not None else "  N/A  "
    sr_s = f"{sr:>+10.3f}" if sr is not None else "    N/A   "
    rr_s = f"{rr:>+7.3f}"  if rr is not None else "  N/A  "
    log(f"R{pair[0]}-R{pair[1]:<4}| {sc_s:<22}| {rc_s:<8}|"
        f" {sr_s:<22}| {rr_s:<8}")
log("-" * 118)
log("   |corr| near 1.0 → drift is thermal, not random.  Prefer a bias state")
log("                     with temperature input over inflating R.")
log("")

# ---- S6: 6x6 matrix --------------------------------------------------------
log("=" * 118)
log("   SECTION 6 — 6x6 MEASUREMENT COVARIANCE MATRIX  R  (LINK-INDEXED)")
log("=" * 118)
log("   Order:  " + "   ".join(f"R{u}-R{v}" for (u, v) in EDGE_PAIRS))
log("   Units: m².  Diagonal = residual variance.  Off-diag = cross-cov.")
log("   NaN = link had no data (must be excluded or handled in EKF).")
log("")
for i in range(6):
    row = "  ".join([f"{R6[i, j]:10.6f}" if np.isfinite(R6[i, j])
                     else "       N/A " for j in range(6)])
    log(f"   R[{i+1}, :] = [ {row} ]")
log("")
log(f"   Marginal σ per link (cm):")
for k, pair in enumerate(EDGE_PAIRS):
    v = R6[k, k]
    if np.isfinite(v):
        log(f"      R{pair[0]}-R{pair[1]} : {np.sqrt(max(v,0))*100:8.3f} cm "
            f"(var = {v:.6f} m²)")
    else:
        log(f"      R{pair[0]}-R{pair[1]} :     N/A")
log("=" * 118)

# ---- S7: expected impact ---------------------------------------------------
log("")
log("=" * 118)
log("   SECTION 7 — WHAT FILTERING CAN AND CANNOT FIX")
log("=" * 118)
log("   ✔ Filtering (moving average / EKF) can REDUCE VARIANCE — e.g. the")
log("     R2-R4 link already shows ~2 cm σ and cannot be improved much by")
log("     any estimator.")
log("   ✘ Filtering CANNOT fix a 30–45 m systematic bias.  The EKF sees")
log("     z = 42 m ± 2 cm and has no way to know the true range is 2.8 m.")
log("")
log("   Recommended action order:")
log("     1.  Install firmware-level TX/RX antenna-delay correction.")
log("     2.  Re-run this script; verify raw_bias drops below ~10 cm.")
log("     3.  If R1 remains anomalous, isolate its ranging chain.")
log("     4.  Only then feed the 6x6 R (this file) into the EKF.")
log("     5.  If ACF[1] > 0.3, add one bias state per link to the EKF.")
log("=" * 118)

# ---- save ------------------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "uwb_comprehensive_report.txt"), "w") as f:
    f.write("\n".join(report) + "\n")

pd.DataFrame(R6,
             index=[f"R{u}-R{v}" for (u, v) in EDGE_PAIRS],
             columns=[f"R{u}-R{v}" for (u, v) in EDGE_PAIRS]
             ).to_csv(os.path.join(OUTPUT_DIR, "uwb_noise_covariance_R_6x6.csv"))
np.save(os.path.join(OUTPUT_DIR, "uwb_noise_covariance_R_6x6.npy"), R6)

if b_all is not None:
    bias_df = pd.DataFrame({
        "node":      [1, 2, 3, 4],
        "b_full_m":  b_all,
        "b_core_m":  (b_core if b_core is not None else [np.nan]*4),
    })
    bias_df.to_csv(os.path.join(OUTPUT_DIR, "uwb_device_biases.csv"), index=False)


# ============================================================================
# 11. PLOTTING
# ============================================================================
colors = {(1,2): "#1f77b4", (1,3): "#ff7f0e", (1,4): "#2ca02c",
          (2,3): "#d62728", (2,4): "#9467bd", (3,4): "#8c564b"}

active = [p for p in EDGE_PAIRS if link[p]["successes"] > 10]

# ---- FIG 1: clock-compensation proof ---------------------------------------
if active:
    fig, axes = plt.subplots(len(active), 1,
                             figsize=(16, 3.2 * len(active)), sharex=True)
    if len(active) == 1:
        axes = [axes]
    for ax, pair in zip(axes, active):
        ld = link[pair]
        sdata = ld["valid"]
        t_m = sdata["time_sec"] / 60.0
        ax.axhline(ld["gt"], color="black", linestyle="--", linewidth=1.5,
                   label=f"GT ({ld['gt']:.3f} m)")
        if ld["c_uncomp"]:
            ax.plot(t_m, sdata[ld["c_uncomp"]], color="gray", alpha=0.5,
                    linestyle=":", label="uncompensated")
        ax.plot(t_m, sdata[ld["c_clock"]], color=colors[pair], linewidth=1.5,
                label=f"clock-comp  (mean={ld['ss_mean']:.3f} m, "
                      f"bias={ld['raw_bias_cm']:+.1f} cm)")
        ax.set_title(f"Link R{pair[0]}-R{pair[1]}", fontsize=11)
        ax.set_ylabel("Range (m)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Time (minutes)")
    plt.suptitle("Clock-offset compensation (pre-calibration)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_clock_compensation_proof.png"),
                dpi=300)
    plt.close()

# ---- FIG 2: thermal + CFO --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
for pair in active:
    ld = link[pair]
    sdata = ld["valid"]
    t_m = sdata["time_sec"] / 60.0
    if "temp_uwb_init" in sdata.columns:
        ax1.plot(t_m, sdata["temp_uwb_init"],
                 label=f"R{pair[0]} (init) T", linewidth=1.2)
    if ld["c_cfo"]:
        ax2.plot(t_m, sdata[ld["c_cfo"]], color=colors[pair], linewidth=1.2,
                 label=f"R{pair[0]}-R{pair[1]} CFO")
ax1.set_title("Thermal transient (initiator UWB die temperature)")
ax1.set_ylabel("Temperature (°C)")
ax1.grid(True, linestyle=":", alpha=0.6)
if ax1.get_legend_handles_labels()[0]:
    ax1.legend(loc="lower right", fontsize=9)
ax2.set_title("CFO evolution with time")
ax2.set_xlabel("Time (minutes)")
ax2.set_ylabel("CFO (ppm)")
ax2.grid(True, linestyle=":", alpha=0.6)
if ax2.get_legend_handles_labels()[0]:
    ax2.legend(loc="upper right", fontsize=9)
plt.suptitle("Thermal state & crystal clock drift", fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_thermal_and_cfo.png"), dpi=300)
plt.close()

# ---- FIG 3: device bias model ----------------------------------------------
if b_all is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    labels = [f"R{u}-R{v}" for (u, v) in EDGE_PAIRS]
    measured  = [excess_all.get(p, np.nan) * 100 for p in EDGE_PAIRS]
    predicted = [(b_all[u-1] + b_all[v-1]) * 100 for (u, v) in EDGE_PAIRS]
    x, w = np.arange(6), 0.38
    ax1.bar(x - w/2, measured, w, label="measured excess",
            color="#1f77b4", edgecolor="black")
    ax1.bar(x + w/2, predicted, w, label="model prediction",
            color="#ff7f0e", edgecolor="black")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("z_mean − GT  (cm)")
    ax1.set_title("Additive-bias model   z_ij = d_ij + b_i + b_j")
    ax1.legend(); ax1.grid(True, linestyle=":", alpha=0.6, axis="y")

    nodes = [1, 2, 3, 4]
    colors_b = ["#d62728" if i == 0 else "#2ca02c" for i in range(4)]
    ax2.bar([f"R{n}" for n in nodes], np.array(b_all) * 100,
            color=colors_b, edgecolor="black", width=0.55)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("b_i  (cm)")
    ax2.set_title(f"Per-device bias  (spread of R1 estimates = "
                  f"{b1_spread:.2f} m)")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    for i, v in enumerate(b_all):
        ax2.text(i, v*100 + (30 if v >= 0 else -80),
                 f"{v*100:+.1f} cm", ha="center",
                 fontsize=10, fontweight="bold")

    plt.suptitle("Device-specific additive bias estimation", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_device_bias_model.png"), dpi=300)
    plt.close()

# ---- FIG 4: residuals after bias removal -----------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=False)
axes = axes.flatten()
for idx, pair in enumerate(EDGE_PAIRS):
    ax = axes[idx]
    ld = link[pair]
    if "residual" in ld and len(ld["residual"]) > 10:
        r_cm = ld["residual"].values * 100.0
        ax.hist(r_cm, bins=40, color=colors[pair], edgecolor="black",
                alpha=0.75, density=True)
        ax.axvline(0, color="darkred", linestyle="--", linewidth=1.5,
                   label="zero")
        ax.axvline(r_cm.mean(), color="black", linestyle=":",
                   linewidth=1.5, label=f"mean={r_cm.mean():+.2f}cm")
        ax.set_title(f"R{pair[0]}-R{pair[1]}   σ = {ld['res_std_cm']:.2f} cm",
                     fontsize=11)
        ax.set_xlabel("Residual (cm)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "NO DATA", ha="center", va="center",
                fontsize=14, color="red")
        ax.set_title(f"R{pair[0]}-R{pair[1]}")
plt.suptitle("Residual noise after removing device bias (the real σ)",
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_residuals_after_bias_removal.png"),
            dpi=300)
plt.close()

# ---- FIG 5: autocorrelation ------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
axes = axes.flatten()
for idx, pair in enumerate(EDGE_PAIRS):
    ax = axes[idx]
    ld = link[pair]
    if "acf" in ld and len(ld["acf"]) > 2:
        ac = ld["acf"]
        lags = np.arange(len(ac))
        ax.stem(lags, ac, basefmt=" ")
        ax.axhline(0, color="black", linewidth=0.8)
        n = len(ld["residual"])
        ci = 1.96 / np.sqrt(max(n, 2))
        ax.axhline(+ci, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(-ci, color="red", linestyle="--", linewidth=1.0)
        ax.set_title(f"R{pair[0]}-R{pair[1]}   ACF[1]={ac[1]:+.3f}",
                     fontsize=11)
        ax.set_xlabel("lag")
        ax.grid(True, linestyle=":", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "NO DATA", ha="center", va="center",
                fontsize=14, color="red")
        ax.set_title(f"R{pair[0]}-R{pair[1]}")
plt.suptitle("Residual autocorrelation — whiteness test "
             "(red dashed = 95 % CI)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_autocorrelation.png"), dpi=300)
plt.close()

# ---- FIG 6: PDR ------------------------------------------------------------
plt.figure(figsize=(10, 6), dpi=300)
labels = [f"R{u}-R{v}" for (u, v) in EDGE_PAIRS]
pdrs = [link[p]["pdr"] for p in EDGE_PAIRS]
cols = ["#2ca02c" if p > 85 else ("#ff7f0e" if p > 50 else "#d62728")
        for p in pdrs]
bars = plt.bar(labels, pdrs, color=cols, edgecolor="black", width=0.55)
plt.axhline(100, color="gray", linestyle="--", linewidth=1.0)
for b in bars:
    y = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, y + 2, f"{y:.1f}%",
             ha="center", fontweight="bold")
plt.title("Packet Delivery Ratio per link")
plt.ylabel("PDR (%)")
plt.ylim(0, 115)
plt.grid(True, linestyle=":", alpha=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_pdr.png"), dpi=300)
plt.close()


log("")
log(f"[SUCCESS] Report + plots written to: {OUTPUT_DIR}/")
log(f"[INFO] 6x6 R  →  uwb_noise_covariance_R_6x6.csv / .npy")
if b_all is not None:
    log(f"[INFO] biases → uwb_device_biases.csv")
