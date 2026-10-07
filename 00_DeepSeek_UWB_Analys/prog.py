#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
 UWB CORE SWARM ANALYZER  —  R2 / R3 / R4 only  (v2, bug-fixed)
========================================================================
 Analyzes links 2-3, 2-4, 3-4.
 Model:
     z_ij = d_ij + b_i + b_j
            + α_i·(T_i - T_ref,i) - α_j·(T_j - T_ref,j)
            + noise
========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import KFold

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.autolayout": True,
})
np.set_printoptions(precision=4, suppress=True)

# ======================================================================
# 0. CONFIG
# ======================================================================
CSV_PATH = "swarm_log.csv"                # ← change this
OUT_DIR  = Path("core_swarm_analysis")
OUT_DIR.mkdir(exist_ok=True)

# Ground truth for the current layout
GT_DISTANCES = {
    "2-3": 2.0000,
    "2-4": 2.8284,
    "3-4": 2.0000,
}
CORE_LINKS   = ["2-3", "2-4", "3-4"]
CORE_DEVICES = [2, 3, 4]

REQUIRE_STATUS_OK  = True
ROBUST_AGGREGATION = True
AGG_NAME           = "median" if ROBUST_AGGREGATION else "mean"

# ======================================================================
# 1. LOAD & FILTER
# ======================================================================
print("=" * 76)
print(" UWB CORE SWARM ANALYZER  (R2 / R3 / R4 only)")
print("=" * 76)

df = pd.read_csv(CSV_PATH)
print(f"\n[1] Loaded {len(df)} rows, {df['frame_id'].nunique()} frames")

if REQUIRE_STATUS_OK and "status" in df.columns:
    before = len(df)
    df = df[df["status"] == 1].copy()
    print(f"    status==1 filter: {before} → {len(df)} rows")

# Keep only core links
df = df[((df["init_id"].isin(CORE_DEVICES)) &
         (df["resp_id"].isin(CORE_DEVICES)))].copy()
print(f"    core-only filter: {len(df)} rows")

# CRITICAL: reset index so iterrows() gives 0-based sequential indices
df = df.reset_index(drop=True)

# Normalize link direction (i < j)
flip = df["init_id"] > df["resp_id"]
df["i_id"] = np.where(flip, df["resp_id"], df["init_id"]).astype(int)
df["j_id"] = np.where(flip, df["init_id"], df["resp_id"]).astype(int)
df["link"] = df["i_id"].astype(str) + "-" + df["j_id"].astype(str)
df["t"]    = (df["timestamp_ms"] - df["timestamp_ms"].min()) / 1000.0

# Temperatures aligned to (i, j)
df["Ti"] = np.where(flip, df["temp_uwb_resp"], df["temp_uwb_init"])
df["Tj"] = np.where(flip, df["temp_uwb_init"], df["temp_uwb_resp"])

print(f"    final rows: {len(df)}")
print(f"    links     : {sorted(df['link'].unique())}")
print(f"    duration  : {df['t'].max():.1f} s")

# ======================================================================
# 2. AVAILABLE STAGES
# ======================================================================
STAGE_MAP = {
    "uncomp"      : "dist_uncomp_m",
    "clock"       : "dist_clock_m",
    "after_bias"  : "dist_after_bias_m",
    "after_therm" : "dist_after_thermal_m",
    "ekf_pred"    : "dist_ekf_pred_m",
}
present = {k: v for k, v in STAGE_MAP.items() if v in df.columns}
print(f"\n[2] Stage columns present: {list(present.keys())}")

# ======================================================================
# 3. PER-LINK STATISTICS
# ======================================================================
print("\n[3] Per-link statistics (mean ± std in meters)")
print("-" * 76)
print(f"{'link':6s} {'n':>5s} | " +
      " | ".join(f"{k:>14s}" for k in present.keys()))
print("-" * 76)

aggfunc = np.median if ROBUST_AGGREGATION else np.mean
link_stats = {}
for link in CORE_LINKS:
    g = df[df["link"] == link]
    if len(g) == 0:
        continue
    row = {"link": link, "n": len(g)}
    cells = []
    for key, col in present.items():
        m = aggfunc(g[col].values)
        s = g[col].std()
        row[f"{key}_mean"] = m
        row[f"{key}_std"]  = s
        cells.append(f"{m:7.3f}±{s:5.3f}")
    link_stats[link] = row
    print(f"{link:6s} {len(g):5d} | " + " | ".join(cells))

print(f"\n    Aggregation: {AGG_NAME}")

# ======================================================================
# 4. RMSE vs GROUND TRUTH
# ======================================================================
print("\n[4] Per-link RMSE vs ground truth (cm)")
print("-" * 76)
print(f"{'link':6s} {'GT':>8s} | " +
      " | ".join(f"{k:>10s}" for k in present.keys()))
print("-" * 76)
rmse_table = {}
for link in CORE_LINKS:
    g = df[df["link"] == link]
    if len(g) == 0:
        continue
    gt = GT_DISTANCES[link]
    row = {"link": link, "gt": gt}
    cells = []
    for key, col in present.items():
        err = g[col].values - gt
        rmse = float(np.sqrt(np.mean(err ** 2)) * 100)
        row[f"rmse_{key}"] = rmse
        cells.append(f"{rmse:9.2f}cm")
    rmse_table[link] = row
    print(f"{link:6s} {gt:8.4f} | " + " | ".join(cells))

pd.DataFrame(rmse_table).T.to_csv(OUT_DIR / "rmse_table.csv")

# ======================================================================
# 5. BIAS CALIBRATION
# ======================================================================
print("\n" + "=" * 76)
print(" [5] BIAS CALIBRATION  (additive model, 3 unknowns, 3 links)")
print("=" * 76)

calib_stage = "clock" if "clock" in present else "uncomp"
calib_col   = STAGE_MAP[calib_stage]

A_bias = np.array([
    [1, 1, 0],   # 2-3 → b2 + b3
    [1, 0, 1],   # 2-4 → b2 + b4
    [0, 1, 1],   # 3-4 → b3 + b4
], dtype=float)

y_bias = np.array([
    link_stats["2-3"][f"{calib_stage}_mean"] - GT_DISTANCES["2-3"],
    link_stats["2-4"][f"{calib_stage}_mean"] - GT_DISTANCES["2-4"],
    link_stats["3-4"][f"{calib_stage}_mean"] - GT_DISTANCES["3-4"],
])

b_vec, residuals, rank, sv = np.linalg.lstsq(A_bias, y_bias, rcond=None)
b2, b3, b4 = b_vec

print(f"    Calibration stage: {calib_stage}")
print(f"    Rank of A_bias: {rank} (expect 3)")
print(f"\n    Residuals (z - d) per link [m]:")
for link, val in zip(CORE_LINKS, y_bias):
    print(f"      {link}: {val:+10.4f}")

print(f"\n    Fitted device biases [m]:")
print(f"      b_2 = {b2:+10.4f}")
print(f"      b_3 = {b3:+10.4f}")
print(f"      b_4 = {b4:+10.4f}")

pred_bias = A_bias @ b_vec
err_bias  = y_bias - pred_bias
print(f"\n    Model fit quality:")
print(f"      {'link':6s} {'(z-d)':>11s} {'(b_i+b_j)':>12s} {'err':>11s}")
for link, yv, pv, ev in zip(CORE_LINKS, y_bias, pred_bias, err_bias):
    print(f"      {link:6s} {yv:+11.4f} {pv:+12.4f} {ev:+11.4f}")

rmse_bias = np.sqrt(np.mean(err_bias ** 2)) * 100
print(f"\n      RMSE of residuals: {rmse_bias:.2f} cm")

# Constrained alternative
b_constrained = b_vec - b_vec.mean()
print(f"\n    [Alternative: sum(b)=0 constraint]")
for i, d in enumerate(CORE_DEVICES):
    print(f"      b_{d} = {b_constrained[i]:+10.4f}")

b_final = b_vec
bias_map = {2: b2, 3: b3, 4: b4}

# ======================================================================
# 6. THERMAL MODEL FIT
# ======================================================================
print("\n" + "=" * 76)
print(" [6] THERMAL MODEL FIT  (antisymmetric, 3 unknowns)")
print("=" * 76)
print(" Model:  Δr_ij = α_i·(T_i - T_ref,i) - α_j·(T_j - T_ref,j)")

# Per-device reference temperature
T_ref = {}
for d in CORE_DEVICES:
    Ts = pd.concat([
        df.loc[df["i_id"] == d, "Ti"],
        df.loc[df["j_id"] == d, "Tj"],
    ])
    T_ref[d] = float(Ts.mean())

print(f"\n    T_ref per device [°C]:")
for d in CORE_DEVICES:
    Ts = pd.concat([
        df.loc[df["i_id"] == d, "Ti"],
        df.loc[df["j_id"] == d, "Tj"],
    ])
    print(f"      dev {d}: {T_ref[d]:7.3f}  "
          f"(range {Ts.min():.1f} .. {Ts.max():.1f})")

df["dTi"] = df["Ti"] - df["i_id"].map(T_ref)
df["dTj"] = df["Tj"] - df["j_id"].map(T_ref)

# Apply bias removal
df["z_bias"] = (df[calib_col]
                - df["i_id"].map(bias_map)
                - df["j_id"].map(bias_map))

# Mean-centre per link
df["resid"] = (df["z_bias"]
               - df.groupby("link")["z_bias"].transform(AGG_NAME))

# Build design matrix
idx_of = {d: k for k, d in enumerate(CORE_DEVICES)}
N = len(df)
A_therm = np.zeros((N, 3))
for r, (_, row) in enumerate(df.iterrows()):
    ii = int(row["i_id"])
    jj = int(row["j_id"])
    A_therm[r, idx_of[ii]] =  row["dTi"]
    A_therm[r, idx_of[jj]] = -row["dTj"]

y_therm = df["resid"].values
alpha, *_ = np.linalg.lstsq(A_therm, y_therm, rcond=None)

# Coefficient uncertainty
res_a = y_therm - A_therm @ alpha
dof   = max(1, N - 3)
sig2  = float(res_a @ res_a) / dof
cov_a = sig2 * np.linalg.inv(A_therm.T @ A_therm)
std_a = np.sqrt(np.diag(cov_a))

print(f"\n    Fitted α (antisymmetric):")
for d, a, s in zip(CORE_DEVICES, alpha, std_a):
    print(f"      α_{d} = {a*100:+9.4f} ± {s*100:.4f} cm/°C")

# Cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for tr, te in kf.split(A_therm):
    c, *_ = np.linalg.lstsq(A_therm[tr], y_therm[tr], rcond=None)
    cv_scores.append(np.std(y_therm[te] - A_therm[te] @ c) * 100)
cv_scores = np.array(cv_scores)

print(f"\n    Model quality:")
print(f"      RMSE train: {np.std(res_a)*100:.3f} cm")
print(f"      CV 5-fold : {cv_scores.mean():.3f} ± {cv_scores.std():.3f} cm")

# Helper: thermal correction for a specific link
def compute_residual(g):
    """Return post-thermal residual (in meters), median-centred."""
    ii = int(g["i_id"].iloc[0]); jj = int(g["j_id"].iloc[0])
    thermal = (alpha[idx_of[ii]] * g["dTi"].values
               - alpha[idx_of[jj]] * g["dTj"].values)
    resid = (g[calib_col].values
             - bias_map[ii] - bias_map[jj]
             - thermal)
    return resid - np.median(resid)

# ======================================================================
# 7. VARIANCE MATRIX
# ======================================================================
print("\n" + "=" * 76)
print(" [7] MEASUREMENT VARIANCE MATRIX  (post-thermal)")
print("=" * 76)

R_mat = np.zeros((4, 4))
R_report = {}
for link in CORE_LINKS:
    g = df[df["link"] == link]
    if len(g) == 0:
        continue
    ii, jj = map(int, link.split("-"))
    resid = compute_residual(g)
    var = float(np.var(resid))
    R_mat[ii - 1, jj - 1] = var
    R_mat[jj - 1, ii - 1] = var
    R_report[link] = (var, np.sqrt(var) * 100)

print(f"\n    {'link':6s} {'variance (m²)':>16s} {'σ (cm)':>10s}")
for link, (v, s) in R_report.items():
    print(f"    {link:6s} {v:16.6f} {s:10.3f}")

# ======================================================================
# 8. PLOTS
# ======================================================================
print("\n[8] Generating plots ...")

# --- Fig 1: correction chain ---
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, link in zip(axes, CORE_LINKS):
    g = df[df["link"] == link].sort_values("t")
    ax.plot(g["t"], g[STAGE_MAP["uncomp"]],      lw=0.6, alpha=0.4,
            color="gray",     label="uncomp")
    ax.plot(g["t"], g[STAGE_MAP["clock"]],       lw=0.7, alpha=0.6,
            color="silver",   label="clock-comp")
    ax.plot(g["t"], g[STAGE_MAP["after_bias"]],  lw=0.7, alpha=0.8,
            color="salmon",   label="after bias")
    ax.plot(g["t"], g[STAGE_MAP["after_therm"]], lw=0.9, alpha=0.9,
            color="seagreen", label="after thermal")
    ax.axhline(GT_DISTANCES[link], color="k", ls="--", lw=0.8,
               label=f"GT={GT_DISTANCES[link]:.3f}")
    ax.set_ylabel("range [m]")
    ax.set_title(f"Link {link}")
    ax.legend(fontsize=8, loc="best")
axes[-1].set_xlabel("t [s]")
plt.suptitle("Correction chain — core links", fontsize=12)
plt.savefig(OUT_DIR / "correction_chain.png", dpi=130)
plt.close()

# --- Fig 2: bias calibration ---
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(CORE_LINKS))
w = 0.35
ax.bar(x - w/2, y_bias * 100, w, label="measured (z - d)",
       color="steelblue", edgecolor="k")
ax.bar(x + w/2, pred_bias * 100, w, label="predicted (b_i + b_j)",
       color="salmon", edgecolor="k")
ax.set_xticks(x)
ax.set_xticklabels(CORE_LINKS)
ax.set_ylabel("residual [cm]")
ax.set_title(f"Bias calibration — RMSE = {rmse_bias:.2f} cm")
ax.legend()
plt.savefig(OUT_DIR / "bias_calibration.png", dpi=130)
plt.close()

# --- Fig 3: thermal residuals per link ---
fig, axes = plt.subplots(3, 2, figsize=(14, 8))
for r, link in enumerate(CORE_LINKS):
    g = df[df["link"] == link].sort_values("t")
    resid = compute_residual(g)

    ax = axes[r, 0]
    ax.plot(g["t"], resid * 100, lw=0.5, color="steelblue")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_ylabel("residual [cm]")
    ax.set_title(f"{link}  σ = {np.std(resid)*100:.2f} cm")
    if r == 2:
        ax.set_xlabel("t [s]")

    ax = axes[r, 1]
    plot_acf(resid, lags=50, ax=ax, title=f"{link} ACF")
plt.tight_layout()
plt.savefig(OUT_DIR / "thermal_residuals.png", dpi=130)
plt.close()

# --- Fig 4: α fitting visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for link in CORE_LINKS:
    g = df[df["link"] == link]
    ii, jj = map(int, link.split("-"))
    thermal = (alpha[idx_of[ii]] * g["dTi"].values
               - alpha[idx_of[jj]] * g["dTj"].values)
    y = (g[calib_col].values
         - bias_map[ii] - bias_map[jj]
         - thermal)
    x = g["dTi"].values - g["dTj"].values
    ax.scatter(x, y * 100, s=1, alpha=0.15, label=link)
ax.set_xlabel("ΔT_i − ΔT_j [°C]")
ax.set_ylabel("(z − d − b_i − b_j) [cm]")
ax.set_title("Thermal residual vs temperature difference")
ax.legend(markerscale=8, fontsize=8)

ax = axes[1]
colors = ["tab:blue", "tab:green", "tab:orange"]
ax.bar([f"α_{d}" for d in CORE_DEVICES],
       alpha * 100, yerr=std_a * 100, capsize=5,
       color=colors, edgecolor="k")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("α_i  [cm/°C]")
ax.set_title("Fitted thermal coefficients")
plt.savefig(OUT_DIR / "alpha_fit.png", dpi=130)
plt.close()

# --- Fig 5: temperature trajectories ---
fig, ax = plt.subplots(figsize=(12, 4))
for d, color in zip(CORE_DEVICES, ["tab:blue", "tab:green", "tab:orange"]):
    Ts_i = df[df["i_id"] == d][["t", "Ti"]].rename(columns={"Ti": "T"})
    Ts_j = df[df["j_id"] == d][["t", "Tj"]].rename(columns={"Tj": "T"})
    T_all = pd.concat([Ts_i, Ts_j]).sort_values("t")
    ax.plot(T_all["t"], T_all["T"], lw=0.6, color=color, label=f"R{d}")
ax.set_xlabel("t [s]"); ax.set_ylabel("T [°C]")
ax.set_title("UWB chip temperature")
ax.legend()
plt.savefig(OUT_DIR / "temperature.png", dpi=130)
plt.close()

# --- Fig 6: residual histograms ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, link in zip(axes, CORE_LINKS):
    g = df[df["link"] == link]
    resid = compute_residual(g) * 100
    ax.hist(resid, bins=60, color="steelblue", edgecolor="k",
            alpha=0.75, density=True)
    mu, sd = resid.mean(), resid.std()
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), "r-", lw=1.5,
            label=f"σ = {sd:.2f} cm")
    ax.set_title(f"{link}")
    ax.set_xlabel("residual [cm]")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "residual_histograms.png", dpi=130)
plt.close()

print(f"    Plots saved to: {OUT_DIR.resolve()}")

# ======================================================================
# 9. FIRMWARE COEFFICIENTS
# ======================================================================
print("\n" + "=" * 76)
print(" [9] FIRMWARE COEFFICIENTS  (ready to paste)")
print("=" * 76)

print(f"\n// ------------------------------------------------------------------")
print(f"// UWB calibration for R2, R3, R4  (R1 excluded)")
print(f"// Based on {len(df)} samples, {df['frame_id'].nunique()} frames")
print(f"// ------------------------------------------------------------------")

print(f"\nconstexpr float UWB_DEVICE_BIAS[5] = {{")
print(f"     0.0000f,")
print(f"     0.0000f,   // R1  — excluded")
print(f"    {b2:+11.6f}f,   // R2")
print(f"    {b3:+11.6f}f,   // R3")
print(f"    {b4:+11.6f}f    // R4")
print(f"}};")

print(f"\nconstexpr float UWB_ALPHA[5] = {{")
print(f"     0.000000f,")
print(f"     0.000000f,   // R1  — excluded")
print(f"    {alpha[0]:+11.6f}f,   // R2")
print(f"    {alpha[1]:+11.6f}f,   // R3")
print(f"    {alpha[2]:+11.6f}f    // R4")
print(f"}};")

print(f"\nconstexpr float UWB_T_REF[5] = {{")
print(f"     0.0f,")
print(f"     0.0f,        // R1  — excluded")
print(f"    {T_ref[2]:7.4f}f,   // R2")
print(f"    {T_ref[3]:7.4f}f,   // R3")
print(f"    {T_ref[4]:7.4f}f    // R4")
print(f"}};")

print(f"\nconstexpr float UWB_VARIANCE_POST_THERMAL[4][4] = {{")
for ii in range(4):
    cells = ", ".join(f"{R_mat[ii, jj]:.5f}f" for jj in range(4))
    print(f"    {{ {cells} }},")
print(f"}};")

# ======================================================================
# 10. VERDICT
# ======================================================================
print("\n" + "=" * 76)
print(" [10] VERDICT")
print("=" * 76)

if rmse_bias < 5:
    print(f"    Bias fit       : ✓ excellent (RMSE = {rmse_bias:.2f} cm)")
elif rmse_bias < 15:
    print(f"    Bias fit       : OK (RMSE = {rmse_bias:.2f} cm)")
else:
    print(f"    Bias fit       : ⚠ poor (RMSE = {rmse_bias:.2f} cm)")

if cv_scores.mean() < 10:
    print(f"    Thermal model  : ✓ good (CV = {cv_scores.mean():.2f} cm)")
elif cv_scores.mean() < 20:
    print(f"    Thermal model  : OK (CV = {cv_scores.mean():.2f} cm)")
else:
    print(f"    Thermal model  : ⚠ weak (CV = {cv_scores.mean():.2f} cm)")

print(f"\n    Physical sanity check:")
for d, a in zip(CORE_DEVICES, alpha):
    a_cm = a * 100
    if abs(a_cm) < 15:
        tag = "reasonable"
    elif abs(a_cm) < 30:
        tag = "high but plausible"
    else:
        tag = "⚠ suspicious"
    print(f"      α_{d} = {a_cm:+7.3f} cm/°C  [{tag}]")

if rmse_bias < 15 and cv_scores.mean() < 15:
    print(f"\n    ✓ Core swarm (R2/R3/R4) is READY to deploy in firmware.")
    print(f"      Copy the four blocks above into RobotConfig.h.")
else:
    print(f"\n    ⚠ Check the calibration results before deploying.")

print(f"\n    All outputs: {OUT_DIR.resolve()}")
print("=" * 76)
