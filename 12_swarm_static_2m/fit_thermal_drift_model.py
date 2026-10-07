#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
 UWB Thermal Drift Modeling – v2 (robust)
========================================================================
 Improvements over v1:
   1. R1 excluded from joint fit (known to be inconsistent).
   2. Optional per-link linear time detrending to remove residual
      clock drift before thermal fitting.
   3. Optional quadratic temperature terms.
   4. Per-link diagnostic PNG + summary PNG.
   5. Collinearity check between T and time before fitting.
========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

# ----------------------------------------------------------------------
# 1.  Configuration
# ----------------------------------------------------------------------
CSV_PATH       = "03_swarm_static_2m.csv"
DIST_COL       = "dist_calib_m"       # or "dist_clock_m"

# --- Which devices participate in the joint thermal fit ---
ALL_DEVICES    = [1, 2, 3, 4]
FIT_DEVICES    = [2, 3, 4]            # R1 excluded from joint LSQ

# --- Model options ---
USE_QUADRATIC  = True                 # include (ΔT)^2 terms
USE_TIME_TERM  = False                # add a per-link linear time term
DETREND_TIME   = True                 # remove per-link linear time trend first
RIDGE_LAMBDA   = 1e-6

# --- Splitting ---
TEST_FRACTION  = 0.2
RANDOM_SEED    = 42

# --- Output ---
OUT_DIR        = Path("thermal_drift_results_v2")
OUT_DIR.mkdir(exist_ok=True)

TEMP_INIT      = "temp_uwb_init"
TEMP_RESP      = "temp_uwb_resp"

np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# ----------------------------------------------------------------------
# 2.  Load and normalise
# ----------------------------------------------------------------------
print("=" * 72)
print(" Loading data ...")
df = pd.read_csv(CSV_PATH)
print(f"  raw rows            : {len(df)}")

if "status" in df.columns:
    df = df[df["status"] == 1].copy()
    print(f"  rows with status=1  : {len(df)}")

df = df.dropna(subset=[TEMP_INIT, TEMP_RESP, DIST_COL]).reset_index(drop=True)

# Always order init < resp so each link appears once
flip = df["init_id"] > df["resp_id"]
df["i_id"] = np.where(flip, df["resp_id"], df["init_id"]).astype(int)
df["j_id"] = np.where(flip, df["init_id"], df["resp_id"]).astype(int)
df["Ti"]   = np.where(flip, df[TEMP_RESP], df[TEMP_INIT])
df["Tj"]   = np.where(flip, df[TEMP_INIT], df[TEMP_RESP])
df["link"] = df["i_id"].astype(str) + "-" + df["j_id"].astype(str)

# Time in seconds from first sample
df["t"] = (df["timestamp_ms"] - df["timestamp_ms"].min()) / 1000.0

print(f"  links               : {sorted(df['link'].unique())}")
print(f"  frames              : {df['frame_id'].nunique() if 'frame_id' in df else 'n/a'}")
print(f"  duration            : {df['t'].max():.1f} s")

# ----------------------------------------------------------------------
# 3.  Collinearity diagnostics
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print(" COLLINEARITY CHECK  (temperature vs time)")
print("=" * 72)
for d in ALL_DEVICES:
    T_series = pd.concat([
        df.loc[df["i_id"] == d, ["t", "Ti"]].rename(columns={"Ti": "T"}),
        df.loc[df["j_id"] == d, ["t", "Tj"]].rename(columns={"Tj": "T"}),
    ])
    if len(T_series) > 1:
        c = T_series["t"].corr(T_series["T"])
        flag = "⚠  HIGH" if abs(c) > 0.9 else ""
        print(f"   device {d}:  corr(t, T) = {c:+.4f}   {flag}")

# ----------------------------------------------------------------------
# 4.  Per-device reference temperature
# ----------------------------------------------------------------------
temp_samples = {d: [] for d in ALL_DEVICES}
for d in ALL_DEVICES:
    temp_samples[d].extend(df.loc[df["i_id"] == d, "Ti"].tolist())
    temp_samples[d].extend(df.loc[df["j_id"] == d, "Tj"].tolist())
T_ref = {d: float(np.mean(v)) for d, v in temp_samples.items()}

print("\n Per-device reference temperature (°C):")
for d in ALL_DEVICES:
    print(f"   device {d}: {T_ref[d]:7.3f}  "
          f"(range {min(temp_samples[d]):.1f} .. {max(temp_samples[d]):.1f})")

# ----------------------------------------------------------------------
# 5.  Per-link mean + optional linear detrending
# ----------------------------------------------------------------------
df["dr"] = df[DIST_COL] - df.groupby("link")[DIST_COL].transform("mean")

if DETREND_TIME:
    print("\n Detrending per-link linear time trend ...")
    slopes = {}
    for link, g in df.groupby("link"):
        # robust linear fit of dr vs t
        A1 = np.column_stack([g["t"].values, np.ones(len(g))])
        coef, *_ = np.linalg.lstsq(A1, g["dr"].values, rcond=None)
        slopes[link] = coef[0]
        df.loc[g.index, "dr"] -= coef[0] * g["t"].values
        print(f"   {link}: slope = {coef[0]*100:+.4f} cm/s")

# Mean-centered temperature deviations
df["dTi"] = df["Ti"] - df["i_id"].map(T_ref)
df["dTj"] = df["Tj"] - df["j_id"].map(T_ref)

# ----------------------------------------------------------------------
# 6.  Train / test split by frame
# ----------------------------------------------------------------------
if "frame_id" in df.columns:
    frames = df["frame_id"].unique().copy()
    rng    = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(frames)
    n_test = max(1, int(len(frames) * TEST_FRACTION))
    test_frames = set(frames[:n_test])
    train_mask  = ~df["frame_id"].isin(test_frames)
    test_mask   =  df["frame_id"].isin(test_frames)
else:
    n = len(df)
    perm = np.random.default_rng(RANDOM_SEED).permutation(n)
    n_test = int(n * TEST_FRACTION)
    test_idx = perm[:n_test]
    train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
    test_mask  = ~train_mask

# ----------------------------------------------------------------------
# 7.  Design matrix builder (handles joint fit on FIT_DEVICES only)
# ----------------------------------------------------------------------
def build_design(sub, fit_devices, quadratic=False, time_term=False):
    """Return (A, names, b) for the rows in `sub` that only involve
    devices in `fit_devices`."""
    idx_of = {d: i for i, d in enumerate(fit_devices)}
    rows, keep = [], []
    for k_, (_, row) in enumerate(sub.iterrows()):
        i, j = int(row["i_id"]), int(row["j_id"])
        if i not in idx_of or j not in idx_of:
            continue
        ncol = len(fit_devices)
        a = np.zeros(ncol + (ncol if quadratic else 0) + (1 if time_term else 0))
        a[idx_of[i]] = row["dTi"]
        a[idx_of[j]] = row["dTj"]
        if quadratic:
            a[ncol + idx_of[i]] = row["dTi"] ** 2
            a[ncol + idx_of[j]] = row["dTj"] ** 2
        if time_term:
            a[-1] = row["t"]
        rows.append(a)
        keep.append(k_)
    names = [f"k{d}" for d in fit_devices]
    if quadratic:
        names += [f"q{d}" for d in fit_devices]
    if time_term:
        names += ["c_time"]
    return np.array(rows), names, sub.iloc[keep]["dr"].values, sub.iloc[keep]

# ----------------------------------------------------------------------
# 8.  Solve LSQ with ridge
# ----------------------------------------------------------------------
A_tr, names, b_tr, sub_tr = build_design(
    df[train_mask], FIT_DEVICES, USE_QUADRATIC, USE_TIME_TERM)
A_te, _,     b_te, sub_te = build_design(
    df[test_mask],  FIT_DEVICES, USE_QUADRATIC, USE_TIME_TERM)

print(f"\n LSQ system: train {A_tr.shape},  test {A_te.shape}")

n_par = A_tr.shape[1]
AtA = A_tr.T @ A_tr + RIDGE_LAMBDA * np.eye(n_par)
Atb = A_tr.T @ b_tr
try:
    coef = np.linalg.solve(AtA, Atb)
except np.linalg.LinAlgError:
    coef, *_ = np.linalg.lstsq(AtA, Atb, rcond=None)

# Parameter uncertainty
res_tr = A_tr @ coef - b_tr
dof    = max(1, A_tr.shape[0] - n_par)
sigma2 = float(res_tr @ res_tr) / dof
try:
    cov_coef = sigma2 * np.linalg.inv(A_tr.T @ A_tr)
    std_coef = np.sqrt(np.diag(cov_coef))
except np.linalg.LinAlgError:
    std_coef = np.full(n_par, np.nan)

print("\n" + "=" * 72)
print(" ESTIMATED PARAMETERS  (joint fit devices = "
      f"{FIT_DEVICES})")
print("=" * 72)
for name, c, s in zip(names, coef, std_coef):
    unit = "m/°C" if name.startswith("k") else ("m/°C²" if name.startswith("q")
                                                 else "m/s")
    print(f"   {name:8s} = {c:+.6f} {unit:6s}  ± {s:.6f}")

# ----------------------------------------------------------------------
# 9.  Evaluation
# ----------------------------------------------------------------------
def rmse(x): return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

# --- joint evaluation on FIT_DEVICES links only ---
print("\n" + "=" * 72)
print(" FIT QUALITY  (links among fit devices only)")
print("=" * 72)
print(f"   Train RMSE : {rmse(A_tr @ coef - b_tr)*100:8.3f} cm")
print(f"   Test  RMSE : {rmse(A_te @ coef - b_te)*100:8.3f} cm")
print(f"   Train σ    : {np.std(A_tr @ coef - b_tr)*100:8.3f} cm")

# --- per-link evaluation for ALL links ---
idx_of_full = {d: i for i, d in enumerate(FIT_DEVICES)}

def predict_for_row(row):
    """Return correction (m) for a given row using fitted coefficients."""
    i, j = int(row["i_id"]), int(row["j_id"])
    corr = 0.0
    if i in idx_of_full:
        corr += coef[idx_of_full[i]] * row["dTi"]
        if USE_QUADRATIC:
            corr += coef[len(FIT_DEVICES) + idx_of_full[i]] * row["dTi"] ** 2
    if j in idx_of_full:
        corr += coef[idx_of_full[j]] * row["dTj"]
        if USE_QUADRATIC:
            corr += coef[len(FIT_DEVICES) + idx_of_full[j]] * row["dTj"] ** 2
    if USE_TIME_TERM:
        corr += coef[-1] * row["t"]
    return corr

print("\n" + "=" * 72)
print(" PER-LINK RMSE  (mean-centred residuals)")
print("=" * 72)
print(f"   {'link':8s}  {'before':>10s}  {'after':>10s}  {'improve':>10s}  {'%':>6s}  note")
print("   " + "-" * 62)

per_link = {}
for link, grp in df.groupby("link"):
    i, j = map(int, link.split("-"))
    corr = np.array([predict_for_row(r) for _, r in grp.iterrows()])
    r_b  = rmse(grp["dr"].values)
    r_a  = rmse(grp["dr"].values - corr)
    pct  = 100.0 * (r_b - r_a) / r_b if r_b > 0 else 0.0
    note = "" if (i in FIT_DEVICES and j in FIT_DEVICES) else "  (extrapolated)"
    per_link[link] = (r_b, r_a, pct, grp.copy(), corr)
    print(f"   {link:8s}  {r_b*100:8.3f} cm  {r_a*100:8.3f} cm  "
          f"{(r_b-r_a)*100:8.3f} cm  {pct:5.1f}%{note}")

resid_all = np.concatenate(
    [g["dr"].values - c for (_, _, _, g, c) in per_link.values()])
print("   " + "-" * 62)
print(f"   overall before : "
      f"{rmse(np.concatenate([g['dr'].values for (_,_,_,g,_) in per_link.values()]))*100:8.3f} cm")
print(f"   overall after  : {rmse(resid_all)*100:8.3f} cm")

# ----------------------------------------------------------------------
# 10.  Per-link diagnostic figures
# ----------------------------------------------------------------------
print("\n Generating per-link figures ...")

def save_link_figure(link, grp, corr, r_b, r_a):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.plot(grp["t"].values, grp["dr"].values * 100,
            color="salmon", lw=0.8, alpha=0.8, label="before")
    ax.plot(grp["t"].values, (grp["dr"].values - corr) * 100,
            color="seagreen", lw=0.8, alpha=0.9, label="after")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_ylabel("mean-centred range [cm]")
    ax.set_title(f"Link {link}  |  RMSE before = {r_b*100:.2f} cm  "
                 f"→  after = {r_a*100:.2f} cm")
    ax.legend(loc="best"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(grp["t"].values, grp["Ti"].values, label=f"T dev {int(grp['i_id'].iloc[0])}")
    ax.plot(grp["t"].values, grp["Tj"].values, label=f"T dev {int(grp['j_id'].iloc[0])}")
    ax.set_ylabel("UWB chip T [°C]"); ax.legend(loc="best"); ax.grid(alpha=0.3)

    ax = axes[2]
    resid = (grp["dr"].values - corr) * 100
    ax.hist(resid, bins=50, color="steelblue", edgecolor="k",
            alpha=0.75, density=True)
    mu, sd = float(np.mean(resid)), float(np.std(resid))
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(xs, np.exp(-0.5*((xs-mu)/sd)**2)/(sd*np.sqrt(2*np.pi)),
            "r-", lw=1.6, label=f"Gaussian μ={mu:.2f}, σ={sd:.2f} cm")
    ax.axvline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel("residual [cm]"); ax.set_ylabel("density")
    ax.legend(loc="best"); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"link_{link}.png", dpi=120)
    plt.close(fig)

for link, (r_b, r_a, pct, grp, corr) in per_link.items():
    save_link_figure(link, grp, corr, r_b, r_a)

# ----------------------------------------------------------------------
# 11.  Summary figure (4 panels)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# (a) histogram of all residuals
ax = axes[0, 0]
ax.hist(resid_all * 100, bins=70, color="steelblue",
        edgecolor="k", alpha=0.75, density=True)
mu, sd = float(np.mean(resid_all*100)), float(np.std(resid_all*100))
xs = np.linspace(resid_all.min()*100, resid_all.max()*100, 200)
ax.plot(xs, np.exp(-0.5*((xs-mu)/sd)**2)/(sd*np.sqrt(2*np.pi)),
        "r-", lw=2, label=f"Gaussian σ={sd:.2f} cm")
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("residual [cm]"); ax.set_ylabel("density")
ax.set_title("All residuals after compensation")
ax.legend(); ax.grid(alpha=0.3)

# (b) per-link before/after
ax = axes[0, 1]
links = list(per_link.keys()); x = np.arange(len(links)); w = 0.38
r_b = [per_link[l][0]*100 for l in links]
r_a = [per_link[l][1]*100 for l in links]
ax.bar(x - w/2, r_b, w, label="before", color="salmon",   edgecolor="k")
ax.bar(x + w/2, r_a, w, label="after",  color="seagreen", edgecolor="k")
ax.set_xticks(x); ax.set_xticklabels(links, rotation=0)
ax.set_ylabel("RMSE [cm]"); ax.set_title("Per-link RMSE before/after")
ax.legend(); ax.grid(alpha=0.3, axis="y")

# (c) coefficients with error bars
ax = axes[1, 0]
labels = names
values = coef * 100
errs   = std_coef * 100
colors = ["cornflowerblue" if n.startswith("k")
          else ("tomato" if n.startswith("q") else "gray") for n in labels]
ax.bar(labels, values, yerr=errs, capsize=5,
       color=colors, edgecolor="k")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("coefficient  [cm / °C⁽ⁿ⁾]")
ax.set_title("Fitted thermal coefficients (± 1σ)")
ax.tick_params(axis="x", rotation=30)
ax.grid(alpha=0.3, axis="y")

# (d) temperature trajectories per device
ax = axes[1, 1]
for d in ALL_DEVICES:
    sub_i = df[df["i_id"] == d][["t", "Ti"]].rename(columns={"Ti":"T"})
    sub_j = df[df["j_id"] == d][["t", "Tj"]].rename(columns={"Tj":"T"})
    sub = pd.concat([sub_i, sub_j]).sort_values("t")
    ax.plot(sub["t"].values, sub["T"].values, lw=0.8, label=f"dev {d}")
ax.set_xlabel("time [s]"); ax.set_ylabel("UWB chip T [°C]")
ax.set_title("Temperature trajectories")
ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "summary_all_links.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------
# 12.  Firmware coefficients
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print(" FIRMWARE COEFFICIENTS")
print("=" * 72)
print(" T_ref [°C] = {")
for d in ALL_DEVICES:
    print(f"     {d}: {T_ref[d]:.4f},")
print(" }")

print("\n // thermal model: correction = sum_n k_n * dT^n  (m)")
print(" k1 [m/°C] = {")
for d in FIT_DEVICES:
    print(f"     {d}: {coef[FIT_DEVICES.index(d)]:+.8f},")
print(" }")

if USE_QUADRATIC:
    print(" k2 [m/°C²] = {")
    for d in FIT_DEVICES:
        print(f"     {d}: {coef[len(FIT_DEVICES) + FIT_DEVICES.index(d)]:+.8e},")
    print(" }")

if USE_TIME_TERM:
    print(f" c_time [m/s] = {coef[-1]:+.8e}")

if 1 not in FIT_DEVICES:
    print("\n ⚠  Device 1 (R1) not included in the joint fit.")
    print("    Apply an additional device-specific offset for R1.")

# ----------------------------------------------------------------------
# 13.  Reusable compensation function
# ----------------------------------------------------------------------
K1 = {d: coef[FIT_DEVICES.index(d)]            for d in FIT_DEVICES}
K2 = {d: coef[len(FIT_DEVICES)+FIT_DEVICES.index(d)] for d in FIT_DEVICES} \
        if USE_QUADRATIC else {d: 0.0 for d in FIT_DEVICES}
C_TIME = coef[-1] if USE_TIME_TERM else 0.0

def compensate(dist_raw, init_id, resp_id, T_init, T_resp, t=0.0,
               b=None, T_ref=None, k1=None, k2=None, c_time=None):
    """Apply full thermal compensation to a raw UWB range."""
    if b       is None: b       = {d: 0.0 for d in ALL_DEVICES}
    if T_ref   is None: T_ref   = {d: 25.0 for d in ALL_DEVICES}
    if k1      is None: k1      = K1
    if k2      is None: k2      = K2
    if c_time  is None: c_time  = C_TIME

    corr = b.get(init_id, 0.0) + b.get(resp_id, 0.0)
    if init_id in k1:
        dT = np.asarray(T_init) - T_ref[init_id]
        corr = corr + k1[init_id]*dT + k2[init_id]*dT**2
    if resp_id in k1:
        dT = np.asarray(T_resp) - T_ref[resp_id]
        corr = corr + k1[resp_id]*dT + k2[resp_id]*dT**2
    corr = corr + c_time * t
    return np.asarray(dist_raw) - corr

# quick self-test
row = df.iloc[0]
d_corr = compensate(row[DIST_COL], int(row["i_id"]), int(row["j_id"]),
                    row["Ti"], row["Tj"], row["t"], T_ref=T_ref)
print(f"\n Example: link {row['link']}  raw={row[DIST_COL]:.4f} m  "
      f"→ corrected={float(d_corr):.4f} m")

print(f"\n Outputs in: {OUT_DIR.resolve()}")
print("  • summary_all_links.png")
print("  • link_<i>-<j>.png   (one per link)")
print("=" * 72)
