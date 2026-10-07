#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
 UWB CORE SWARM ANALYZER  —  with Thermal Lag Filter
========================================================================
 Analysis for R2, R3, R4 with first-order thermal lag model:

   z_ij = d_ij + b_i + b_j
          + α_i·(T_filt,i - T_ref,i)
          - α_j·(T_filt,j - T_ref,j)
          + noise

 where T_filt is T_chip passed through a 1st-order low-pass filter:
       T_filt[k] = T_filt[k-1] + (dt/τ)·(T_raw[k] - T_filt[k-1])

 The script:
   - Fits bias (3 unknowns, closed-form)
   - Fits α AND τ simultaneously via grid search over τ
   - Reports ACF improvement
   - Generates firmware-ready coefficients (including τ per device)
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
CSV_PATH = "swarm_log.csv"
OUT_DIR  = Path("core_swarm_analysis_lag")
OUT_DIR.mkdir(exist_ok=True)

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

# τ grid search (seconds). Smaller = faster response, larger = slower.
TAU_GRID = np.array([1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240])

# ======================================================================
# 1. LOAD & FILTER
# ======================================================================
print("=" * 78)
print(" UWB CORE SWARM ANALYZER  —  with Thermal Lag Filter")
print("=" * 78)

df = pd.read_csv(CSV_PATH)
print(f"\n[1] Loaded {len(df)} rows, {df['frame_id'].nunique()} frames")

if REQUIRE_STATUS_OK and "status" in df.columns:
    before = len(df)
    df = df[df["status"] == 1].copy()
    print(f"    status==1 filter: {before} → {len(df)} rows")

df = df[((df["init_id"].isin(CORE_DEVICES)) &
         (df["resp_id"].isin(CORE_DEVICES)))].copy()
print(f"    core-only filter: {len(df)} rows")

df = df.reset_index(drop=True)

flip = df["init_id"] > df["resp_id"]
df["i_id"] = np.where(flip, df["resp_id"], df["init_id"]).astype(int)
df["j_id"] = np.where(flip, df["init_id"], df["resp_id"]).astype(int)
df["link"] = df["i_id"].astype(str) + "-" + df["j_id"].astype(str)
df["t"]    = (df["timestamp_ms"] - df["timestamp_ms"].min()) / 1000.0

df["Ti"] = np.where(flip, df["temp_uwb_resp"], df["temp_uwb_init"])
df["Tj"] = np.where(flip, df["temp_uwb_init"], df["temp_uwb_resp"])

print(f"    final rows: {len(df)}")
print(f"    links     : {sorted(df['link'].unique())}")
print(f"    duration  : {df['t'].max():.1f} s")

# ======================================================================
# 2. STAGES
# ======================================================================
STAGE_MAP = {
    "uncomp"      : "dist_uncomp_m",
    "clock"       : "dist_clock_m",
    "after_bias"  : "dist_after_bias_m",
    "after_therm" : "dist_after_thermal_m",
}
present = {k: v for k, v in STAGE_MAP.items() if v in df.columns}
print(f"\n[2] Stage columns present: {list(present.keys())}")

# ======================================================================
# 3. STATISTICS PER STAGE
# ======================================================================
print("\n[3] Per-link stage statistics (meters)")
print("-" * 78)
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
        cells.append(f"{key}={m:.3f}±{s:.3f}")
    link_stats[link] = row
    print(f"    {link}: " + "  ".join(cells))

# ======================================================================
# 4. BIAS CALIBRATION
# ======================================================================
print("\n" + "=" * 78)
print(" [4] BIAS CALIBRATION")
print("=" * 78)

calib_stage = "clock" if "clock" in present else "uncomp"
calib_col   = STAGE_MAP[calib_stage]

A_bias = np.array([[1,1,0],[1,0,1],[0,1,1]], dtype=float)
y_bias = np.array([
    link_stats["2-3"][f"{calib_stage}_mean"] - GT_DISTANCES["2-3"],
    link_stats["2-4"][f"{calib_stage}_mean"] - GT_DISTANCES["2-4"],
    link_stats["3-4"][f"{calib_stage}_mean"] - GT_DISTANCES["3-4"],
])
b_vec, *_ = np.linalg.lstsq(A_bias, y_bias, rcond=None)
b2, b3, b4 = b_vec
bias_map = {2: b2, 3: b3, 4: b4}

print(f"    Stage: {calib_stage}")
print(f"    b_2 = {b2:+10.4f} m")
print(f"    b_3 = {b3:+10.4f} m")
print(f"    b_4 = {b4:+10.4f} m")
print(f"    RMSE = {np.sqrt(np.mean((y_bias - A_bias@b_vec)**2))*100:.4f} cm")

# ======================================================================
# 5. THERMAL LAG FILTER  —  DEFINITION
# ======================================================================
def apply_lpf(signal, t, tau):
    """First-order low-pass filter: y[k] = y[k-1] + (dt/τ)·(x[k] - y[k-1])"""
    y = np.zeros_like(signal, dtype=float)
    y[0] = signal[0]
    for k in range(1, len(signal)):
        dt = t[k] - t[k-1]
        if dt <= 0: dt = 1e-3
        a = dt / (tau + dt)
        y[k] = y[k-1] + a * (signal[k] - y[k-1])
    return y

# We filter temperature traces PER DEVICE, not per link.
# Build per-device time series (sorted by frame).
device_series = {}
for d in CORE_DEVICES:
    rows_i = df[df["i_id"] == d][["frame_id", "t", "Ti"]].rename(columns={"Ti": "T"})
    rows_j = df[df["j_id"] == d][["frame_id", "t", "Tj"]].rename(columns={"Tj": "T"})
    all_rows = pd.concat([rows_i, rows_j]).sort_values(["frame_id", "t"])
    # Take first sample per frame to get a clean time series
    all_rows = all_rows.drop_duplicates(subset="frame_id", keep="first")
    device_series[d] = all_rows.reset_index(drop=True)

# ======================================================================
# 6. THERMAL MODEL FIT  —  GRID SEARCH OVER τ
# ======================================================================
print("\n" + "=" * 78)
print(" [5] THERMAL MODEL FIT WITH LAG  (grid search over τ)")
print("=" * 78)

idx_of = {d: k for k, d in enumerate(CORE_DEVICES)}
N = len(df)

def fit_thermal_with_tau(tau_value, return_per_link=False):
    """Fit α for a given τ. Returns (alpha, rmse, acf1_avg, T_ref, dT_map)."""
    T_ref_local = {}
    T_filt_all = {}
    for d in CORE_DEVICES:
        s = device_series[d]
        T_filt_all[d] = apply_lpf(s["T"].values, s["t"].values, tau_value)
        T_ref_local[d] = float(np.mean(T_filt_all[d]))

    # Map back to per-link dataframe via frame_id
    frame_to_Tfilt = {}
    for d in CORE_DEVICES:
        s = device_series[d]
        frame_to_Tfilt[d] = dict(zip(s["frame_id"].values, T_filt_all[d]))

    # Compute dTi, dTj using filtered temperature
    dTi_arr = np.zeros(N)
    dTj_arr = np.zeros(N)
    for r, row in df.iterrows():
        ii = int(row["i_id"]); jj = int(row["j_id"])
        fid = int(row["frame_id"])
        Tf_i = frame_to_Tfilt[ii].get(fid, T_ref_local[ii])
        Tf_j = frame_to_Tfilt[jj].get(fid, T_ref_local[jj])
        dTi_arr[r] = Tf_i - T_ref_local[ii]
        dTj_arr[r] = Tf_j - T_ref_local[jj]

    # Residual:  z - d - b_i - b_j
    resid = (df[calib_col].values
             - df["i_id"].map(bias_map).values
             - df["j_id"].map(bias_map).values
             - df["link"].map({l: GT_DISTANCES[l] for l in CORE_LINKS}).values)
    # Mean-centre per link
    resid = resid - pd.Series(resid).groupby(df["link"].values).transform(
        AGG_NAME).values

    # Build design matrix (antisymmetric)
    A_t = np.zeros((N, 3))
    for r in range(N):
        ii = int(df.iloc[r]["i_id"]); jj = int(df.iloc[r]["j_id"])
        A_t[r, idx_of[ii]] =  dTi_arr[r]
        A_t[r, idx_of[jj]] = -dTj_arr[r]

    alpha_local, *_ = np.linalg.lstsq(A_t, resid, rcond=None)
    res_local = resid - A_t @ alpha_local
    rmse_local = float(np.std(res_local) * 100)

    # ACF[1] per link (average)
    acfs = []
    for link in CORE_LINKS:
        mask = (df["link"] == link).values
        rl = res_local[mask]
        if len(rl) < 10: continue
        rl = rl - rl.mean()
        denom = np.sum(rl * rl)
        if denom < 1e-12: continue
        acfs.append(float(np.sum(rl[:-1] * rl[1:]) / denom))
    acf_mean = float(np.mean(acfs)) if acfs else np.nan

    if return_per_link:
        per_link = {}
        for link in CORE_LINKS:
            mask = (df["link"] == link).values
            rl = res_local[mask]
            per_link[link] = dict(
                rmse=float(np.std(rl) * 100),
                acf1=float(np.sum((rl - rl.mean())[:-1] * (rl - rl.mean())[1:])
                           / max(np.sum((rl - rl.mean())**2), 1e-12)),
                std=float(np.std(rl) * 100),
            )
        return alpha_local, rmse_local, acf_mean, T_ref_local, dTi_arr, dTj_arr, per_link

    return alpha_local, rmse_local, acf_mean, T_ref_local, dTi_arr, dTj_arr, None

print(f"\n    {'τ (s)':>8s} {'RMSE (cm)':>12s} {'ACF[1]':>10s} "
      f"{'α_2':>10s} {'α_3':>10s} {'α_4':>10s}")
print("    " + "-" * 68)

results = []
for tau in TAU_GRID:
    alpha_t, rmse_t, acf_t, T_ref_t, _, _, _ = fit_thermal_with_tau(tau)
    results.append(dict(tau=tau, alpha=alpha_t, rmse=rmse_t,
                        acf=acf_t, T_ref=T_ref_t))
    print(f"    {tau:8.1f} {rmse_t:12.4f} {acf_t:+10.4f} "
          f"{alpha_t[0]*100:+10.3f} {alpha_t[1]*100:+10.3f} "
          f"{alpha_t[2]*100:+10.3f}")

# Pick best tau by ACF[1] (lowest absolute value)
best_by_acf = min(results, key=lambda r: abs(r["acf"]))
best_by_rmse = min(results, key=lambda r: r["rmse"])

print("\n    Best τ by ACF[1]:")
print(f"      τ = {best_by_acf['tau']:.1f} s")
print(f"      RMSE = {best_by_acf['rmse']:.3f} cm")
print(f"      ACF[1] = {best_by_acf['acf']:+.4f}")

print("\n    Best τ by RMSE:")
print(f"      τ = {best_by_rmse['tau']:.1f} s")
print(f"      RMSE = {best_by_rmse['rmse']:.3f} cm")
print(f"      ACF[1] = {best_by_rmse['acf']:+.4f}")

# Use best by ACF as primary (whitest residual = best for EKF)
best = best_by_acf
tau_star = best["tau"]
alpha_star = best["alpha"]
T_ref_star = best["T_ref"]

print("\n" + "=" * 78)
print(f" SELECTED MODEL: τ = {tau_star:.1f} s")
print("=" * 78)
for d, a in zip(CORE_DEVICES, alpha_star):
    print(f"    α_{d} = {a*100:+9.4f} cm/°C")
print(f"\n    T_ref (after lag filter):")
for d in CORE_DEVICES:
    print(f"      dev {d}: {T_ref_star[d]:7.3f} °C")

# Cross validation on the best τ
alpha_full, _, _, T_ref_full, dTi_arr, dTj_arr, per_link_star = \
    fit_thermal_with_tau(tau_star, return_per_link=True)

# Rebuild A_t for CV
A_t_full = np.zeros((N, 3))
for r in range(N):
    ii = int(df.iloc[r]["i_id"]); jj = int(df.iloc[r]["j_id"])
    A_t_full[r, idx_of[ii]] =  dTi_arr[r]
    A_t_full[r, idx_of[jj]] = -dTj_arr[r]

resid_full = (df[calib_col].values
              - df["i_id"].map(bias_map).values
              - df["j_id"].map(bias_map).values
              - df["link"].map({l: GT_DISTANCES[l] for l in CORE_LINKS}).values)
resid_full = resid_full - pd.Series(resid_full).groupby(
    df["link"].values).transform(AGG_NAME).values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for tr, te in kf.split(A_t_full):
    c, *_ = np.linalg.lstsq(A_t_full[tr], resid_full[tr], rcond=None)
    cv_scores.append(np.std(resid_full[te] - A_t_full[te] @ c) * 100)
cv_scores = np.array(cv_scores)
print(f"\n    CV 5-fold (τ = {tau_star:.1f} s): "
      f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f} cm")

# ======================================================================
# 7. COMPARISON: RAW vs FILTERED
# ======================================================================
print("\n" + "=" * 78)
print(" [6] COMPARISON: no lag vs lag")
print("=" * 78)

# Fit without lag (τ → 0)
alpha_nolag, rmse_nolag, acf_nolag, T_ref_nolag, _, _, per_link_nolag = \
    fit_thermal_with_tau(0.01, return_per_link=True)

print(f"\n    Without lag:")
print(f"      RMSE = {rmse_nolag:.3f} cm   ACF[1] = {acf_nolag:+.4f}")
for d, a in zip(CORE_DEVICES, alpha_nolag):
    print(f"      α_{d} = {a*100:+9.3f} cm/°C")

print(f"\n    With lag (τ = {tau_star:.1f} s):")
print(f"      RMSE = {best['rmse']:.3f} cm   ACF[1] = {best['acf']:+.4f}")
for d, a in zip(CORE_DEVICES, alpha_star):
    print(f"      α_{d} = {a*100:+9.3f} cm/°C")

print(f"\n    Per-link ACF comparison (lower is better):")
print(f"      {'link':6s} {'ACF[1] no-lag':>16s} {'ACF[1] lag':>14s} "
      f"{'improve':>10s}")
for link in CORE_LINKS:
    a0 = per_link_nolag[link]["acf1"]
    a1 = per_link_star[link]["acf1"]
    imp = 100 * (abs(a0) - abs(a1)) / max(abs(a0), 1e-9)
    print(f"      {link:6s} {a0:+16.4f} {a1:+14.4f} {imp:+9.1f}%")

# ======================================================================
# 8. VARIANCE MATRIX  (post-lag-thermal)
# ======================================================================
print("\n" + "=" * 78)
print(" [7] VARIANCE MATRIX (post lag-thermal)")
print("=" * 78)

# Recompute residuals with best model
alpha_full, _, _, T_ref_full, dTi_arr, dTj_arr, _ = \
    fit_thermal_with_tau(tau_star)

thermal_corr = (dTi_arr * alpha_full[0]
                - dTj_arr * alpha_full[1])  # not correct ordering, recompute
# Proper: thermal_corr = alpha_i·dTi - alpha_j·dTj for each row
thermal_corr = np.zeros(N)
for r in range(N):
    ii = int(df.iloc[r]["i_id"]); jj = int(df.iloc[r]["j_id"])
    thermal_corr[r] = (alpha_full[idx_of[ii]] * dTi_arr[r]
                       - alpha_full[idx_of[jj]] * dTj_arr[r])

resid_final = (df[calib_col].values
               - df["i_id"].map(bias_map).values
               - df["j_id"].map(bias_map).values
               - thermal_corr
               - df["link"].map({l: GT_DISTANCES[l] for l in CORE_LINKS}).values)
resid_final = resid_final - pd.Series(resid_final).groupby(
    df["link"].values).transform(AGG_NAME).values

R_mat = np.zeros((4, 4))
R_report = {}
for link in CORE_LINKS:
    mask = (df["link"] == link).values
    ii, jj = map(int, link.split("-"))
    v = float(np.var(resid_final[mask]))
    R_mat[ii-1, jj-1] = v
    R_mat[jj-1, ii-1] = v
    R_report[link] = (v, np.sqrt(v) * 100)

print(f"\n    {'link':6s} {'variance (m²)':>16s} {'σ (cm)':>10s}")
for link, (v, s) in R_report.items():
    print(f"    {link:6s} {v:16.6f} {s:10.3f}")

# ======================================================================
# 9. PLOTS
# ======================================================================
print("\n[8] Generating plots ...")

# --- Fig 1: temperature raw vs filtered for each device ---
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, d in zip(axes, CORE_DEVICES):
    s = device_series[d]
    ax.plot(s["t"], s["T"], lw=0.6, alpha=0.6, color="gray", label="raw")
    Tf = apply_lpf(s["T"].values, s["t"].values, tau_star)
    ax.plot(s["t"], Tf, lw=1.6, color="red",
            label=f"filtered (τ={tau_star:.0f}s)")
    ax.set_ylabel("T [°C]")
    ax.set_title(f"Device {d}")
    ax.legend(fontsize=8)
axes[-1].set_xlabel("t [s]")
plt.suptitle(f"Thermal lag filter — τ = {tau_star:.1f} s")
plt.savefig(OUT_DIR / "thermal_filter.png", dpi=130)
plt.close()

# --- Fig 2: τ scan ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
taus = [r["tau"] for r in results]
rmses = [r["rmse"] for r in results]
acfs = [abs(r["acf"]) for r in results]
alphas = np.array([r["alpha"] for r in results]) * 100

axes[0].semilogx(taus, rmses, "o-", color="steelblue")
axes[0].axvline(tau_star, color="r", ls="--", label=f"τ* = {tau_star:.0f}s")
axes[0].set_xlabel("τ [s]"); axes[0].set_ylabel("RMSE [cm]")
axes[0].set_title("RMSE vs τ"); axes[0].legend()

axes[1].semilogx(taus, acfs, "o-", color="darkgreen")
axes[1].axvline(tau_star, color="r", ls="--")
axes[1].axhline(0.3, color="k", ls=":", label="|ACF|=0.3 target")
axes[1].set_xlabel("τ [s]"); axes[1].set_ylabel("|ACF[1]|")
axes[1].set_title("Residual ACF vs τ"); axes[1].legend()

for i, d in enumerate(CORE_DEVICES):
    axes[2].semilogx(taus, alphas[:, i], "o-", label=f"α_{d}")
axes[2].axvline(tau_star, color="r", ls="--")
axes[2].axhline(0, color="k", lw=0.8)
axes[2].set_xlabel("τ [s]"); axes[2].set_ylabel("α_i [cm/°C]")
axes[2].set_title("Fitted α vs τ"); axes[2].legend()
plt.savefig(OUT_DIR / "tau_scan.png", dpi=130)
plt.close()

# --- Fig 3: residual time series before/after lag ---
fig, axes = plt.subplots(3, 2, figsize=(14, 8))
for r, link in enumerate(CORE_LINKS):
    mask = (df["link"] == link).values
    # Without lag
    resid_nl = resid_full  # this was computed with the best tau, wrong
    # Actually recompute without lag
    _, _, _, _, dTi_nl, dTj_nl, _ = fit_thermal_with_tau(0.01)
    alpha_nl, *_ = np.linalg.lstsq(
        np.array([[dTi_nl[r_] if int(df.iloc[r_]["i_id"]) == d_ else
                   (-dTj_nl[r_] if int(df.iloc[r_]["j_id"]) == d_ else 0)
                   for d_ in CORE_DEVICES] for r_ in range(N)]),
        resid_full, rcond=None)
    thermal_nl = np.zeros(N)
    for r_ in range(N):
        ii = int(df.iloc[r_]["i_id"]); jj = int(df.iloc[r_]["j_id"])
        thermal_nl[r_] = (alpha_nl[idx_of[ii]] * dTi_nl[r_]
                          - alpha_nl[idx_of[jj]] * dTj_nl[r_])
    res_nl = resid_full - thermal_nl

    g = df[mask].sort_values("t")
    res_lag_link = resid_final[mask]

    ax = axes[r, 0]
    ax.plot(g["t"], res_nl * 100, lw=0.5, color="salmon",
            label=f"no lag σ={res_nl.std()*100:.2f}cm")
    ax.plot(g["t"], res_lag_link * 100, lw=0.5, color="seagreen",
            label=f"with lag σ={res_lag_link.std()*100:.2f}cm")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("residual [cm]")
    ax.set_title(f"{link}")
    ax.legend(fontsize=8)
    if r == 2: ax.set_xlabel("t [s]")

    ax = axes[r, 1]
    plot_acf(res_nl[mask], lags=50, ax=ax, title=f"{link} ACF (no lag)")
plt.tight_layout()
plt.savefig(OUT_DIR / "residual_comparison.png", dpi=130)
plt.close()

# --- Fig 4: residual histograms after lag ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, link in zip(axes, CORE_LINKS):
    mask = (df["link"] == link).values
    r = resid_final[mask] * 100
    ax.hist(r, bins=60, color="steelblue", edgecolor="k",
            alpha=0.75, density=True)
    mu, sd = r.mean(), r.std()
    xs = np.linspace(r.min(), r.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), "r-", lw=1.5,
            label=f"σ={sd:.2f} cm")
    ax.set_title(f"{link}  (after lag)")
    ax.set_xlabel("residual [cm]")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "residual_histograms_lag.png", dpi=130)
plt.close()

print(f"    Plots saved to: {OUT_DIR.resolve()}")

# ======================================================================
# 10. FIRMWARE COEFFICIENTS
# ======================================================================
print("\n" + "=" * 78)
print(" [9] FIRMWARE COEFFICIENTS  (with thermal lag)")
print("=" * 78)

print(f"\n// ------------------------------------------------------------------")
print(f"// UWB calibration for R2, R3, R4  — with thermal lag")
print(f"// τ selected by minimizing residual ACF: τ* = {tau_star:.1f} s")
print(f"// ------------------------------------------------------------------")

print(f"\nconstexpr float UWB_DEVICE_BIAS[5] = {{")
print(f"     0.0000f,")
print(f"     0.0000f,   // R1 — excluded")
print(f"    {b2:+11.6f}f,   // R2")
print(f"    {b3:+11.6f}f,   // R3")
print(f"    {b4:+11.6f}f    // R4")
print(f"}};")

print(f"\nconstexpr float UWB_ALPHA[5] = {{")
print(f"     0.000000f,")
print(f"     0.000000f,   // R1 — excluded")
print(f"    {alpha_full[0]:+11.6f}f,   // R2")
print(f"    {alpha_full[1]:+11.6f}f,   // R3")
print(f"    {alpha_full[2]:+11.6f}f    // R4")
print(f"}};")

print(f"\nconstexpr float UWB_T_REF[5] = {{")
print(f"     0.0f,")
print(f"     0.0f,        // R1 — excluded")
print(f"    {T_ref_star[2]:7.4f}f,   // R2")
print(f"    {T_ref_star[3]:7.4f}f,   // R3")
print(f"    {T_ref_star[4]:7.4f}f    // R4")
print(f"}};")

print(f"\nconstexpr float UWB_THERMAL_TAU_S = {tau_star:.2f}f;   // R2, R3, R4")

print(f"\nconstexpr float UWB_VARIANCE_POST_THERMAL[4][4] = {{")
for ii in range(4):
    cells = ", ".join(f"{R_mat[ii, jj]:.5f}f" for jj in range(4))
    print(f"    {{ {cells} }},")
print(f"}};")

# ======================================================================
# 11. VERDICT
# ======================================================================
print("\n" + "=" * 78)
print(" [10] VERDICT")
print("=" * 78)

acf_improve = 100 * (abs(acf_nolag) - abs(best["acf"])) / max(abs(acf_nolag), 1e-9)
print(f"\n    ACF[1] without lag : {acf_nolag:+.4f}")
print(f"    ACF[1] with lag    : {best['acf']:+.4f}  "
      f"({acf_improve:+.1f}% improvement)")
print(f"    τ*                  : {tau_star:.1f} s")
print(f"    RMSE                : {best['rmse']:.3f} cm")

if abs(best["acf"]) < 0.3:
    print("\n    ✓ Residual is now effectively WHITE.")
    print("      The EKF can rely on small R without overconfidence.")
elif abs(best["acf"]) < 0.6:
    print("\n    ~ Residual is partially whitened.")
    print("      Consider adding a slow bias state in EKF for the remainder.")
else:
    print("\n    ⚠ Residual still time-correlated after lag filter.")
    print("      The thermal dynamics is more complex than 1st order.")

print(f"\n    All outputs: {OUT_DIR.resolve()}")
print("=" * 78)
