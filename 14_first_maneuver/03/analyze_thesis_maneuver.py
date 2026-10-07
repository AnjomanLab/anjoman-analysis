#!/usr/bin/env python3
"""
Anjoman Swarm Metrology & Formation Control Thesis-Grade Evaluation Engine
Compares:
  1. Raw UWB Distance
  2. Deterministic Calibrated UWB Distance (Layer 1 Preprocessor)
  3. Fused / Filtered State Distance (Kinematic Fusion & Cooperative Tracking)
Generates high-resolution publication figures (300 DPI) in thesis_figures/
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ==============================================================================
# 0. CONFIGURATION & DIRECTORIES
# ==============================================================================
FIGURES_DIR = "thesis_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

FILES = {
    1: "robot_1_maneuver.csv",
    2: "robot_2_maneuver.csv",
    3: "robot_3_maneuver.csv",
    4: "robot_4_maneuver.csv"
}

# ==============================================================================
# 1. DATA INGESTION & SYNCHRONIZATION
# ==============================================================================
data = {}
print("=" * 90)
print("ANJOMAN SWARM THESIS METROLOGY & KINEMATICS AUDIT PIPELINE")
print("=" * 90)

for r_id, fname in FILES.items():
    if not os.path.exists(fname):
        alt = f"robot_{r_id}_test.csv"
        if os.path.exists(alt): fname = alt
        else:
            print(f"[MISSING] {fname}")
            continue
    df = pd.read_csv(fname)
    df["t_sec"] = (df["t_ms"] - df["t_ms"].iloc[0]) / 1000.0
    data[r_id] = df
    print(f"Loaded Robot {r_id}: {len(df)} samples | Duration: {df['t_sec'].iloc[-1]:.2f} s")

if len(data) < 2:
    sys.exit("[ERROR] Insufficient datasets found for multi-agent evaluation.")

# ==============================================================================
# 2. THEORETICAL GROUND TRUTH TRAJECTORY GENERATOR
# ==============================================================================
def get_ground_truth(t_sec):
    """Computes exact analytical reference positions and inter-robot distances."""
    T1 = 20.0
    T2 = 10.0
    L0 = 2.000
    Lf = 3.000
    
    if t_sec < T1:
        tau = t_sec / T1
        s = 10.0*(tau**3) - 15.0*(tau**4) + 6.0*(tau**5)
        phi = (np.pi / 2.0) * s
        L = L0
    elif t_sec < (T1 + T2):
        tau = (t_sec - T1) / T2
        s = 10.0*(tau**3) - 15.0*(tau**4) + 6.0*(tau**5)
        phi = np.pi / 2.0
        L = L0 + (Lf - L0) * s
    else:
        phi = np.pi / 2.0
        L = Lf

    # Corner positions relative to center (0,0)
    unit_offsets = {
        1: np.array([-0.5, -0.5]),
        2: np.array([+0.5, -0.5]),
        3: np.array([+0.5, +0.5]),
        4: np.array([-0.5, +0.5])
    }
    
    R_mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    poses = {r: R_mat @ (L * unit_offsets[r]) for r in range(1, 5)}
    
    # Ground truth edge distances
    d_gt = {}
    for i in range(1, 5):
        for j in range(1, 5):
            if i != j:
                d_gt[(i, j)] = np.linalg.norm(poses[i] - poses[j])
                
    return poses, d_gt, L, phi

# ==============================================================================
# 3. THREE-TIER UWB METROLOGY COMPARISON (Raw vs Calibrated vs Estimated)
# ==============================================================================
print("\n" + "=" * 90)
print("UWB THREE-TIER ERROR EVALUATION (Raw vs Deterministic Calibrated vs Fused State)")
print("=" * 90)

comparison_rows = []

for r_id in sorted(data.keys()):
    df = data[r_id]
    peer_id = int(df["uwb_peer_id"].mode()[0])
    
    # Compute ground truth distance series for this link
    gt_series = np.array([get_ground_truth(t)[1].get((r_id, peer_id), np.nan) for t in df["t_sec"]])
    
    # Tier 1: Raw Distance
    err_raw = df["uwb_raw"].values - gt_series
    # Tier 2: Deterministic Calibrated (Preprocessor Layer 1)
    err_clean = df["uwb_clean"].values - gt_series
    # Tier 3: Odometry-Fused Kinematic Distance
    if peer_id in data:
        # Distance between estimated positions of the two robots
        p_self = np.column_stack([df["x"].values, df["y"].values])
        # Interpolate peer position to self timestamps
        df_peer = data[peer_id]
        px_peer = np.interp(df["t_sec"], df_peer["t_sec"], df_peer["x"])
        py_peer = np.interp(df["t_sec"], df_peer["t_sec"], df_peer["y"])
        p_peer = np.column_stack([px_peer, py_peer])
        d_fused = np.linalg.norm(p_self - p_peer, axis=1)
        err_fused = d_fused - gt_series
    else:
        err_fused = np.full_like(err_clean, np.nan)

    # Compute Statistical Metrics
    comparison_rows.append({
        "Link": f"{r_id}->{peer_id}",
        "Raw_Mean_Bias_m": np.nanmean(err_raw),
        "Raw_RMSE_m": np.sqrt(np.nanmean(err_raw**2)),
        "Calib_Mean_Bias_m": np.nanmean(err_clean),
        "Calib_RMSE_m": np.sqrt(np.nanmean(err_clean**2)),
        "Calib_Std_m": np.nanstd(err_clean),
        "Fused_Mean_Bias_m": np.nanmean(err_fused),
        "Fused_RMSE_m": np.sqrt(np.nanmean(err_fused**2)),
        "Fused_Std_m": np.nanstd(err_fused),
    })

df_comp = pd.DataFrame(comparison_rows)
print(df_comp.to_string(index=False, justify="center"))
df_comp.to_csv(os.path.join(FIGURES_DIR, "uwb_three_tier_comparison.csv"), index=False)

# ==============================================================================
# 4. PUBLICATION-QUALITY FIGURES (300 DPI - LARGE SCALE)
# ==============================================================================
print("\n[GENERATING FIGURES] Exporting publication-ready plots to thesis_figures/ ...")

# ------------------------------------------------------------------------------
# FIGURE 1: 3-TIER UWB METROLOGY COMPARISON (Raw vs Calib vs Fused vs GT)
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), dpi=300, sharex=True)

# Plot representative link (e.g. Robot 2 -> Robot 1 or available link)
rep_id = 2 if 2 in data else 1
df_rep = data[rep_id]
peer_rep = int(df_rep["uwb_peer_id"].mode()[0])
gt_rep = np.array([get_ground_truth(t)[1].get((rep_id, peer_rep), np.nan) for t in df_rep["t_sec"]])

# Top Subplot: Absolute Distances
ax1.plot(df_rep["t_sec"], df_rep["uwb_raw"], color="darkgray", linestyle="--", linewidth=1.5, label="Tier 1: Raw UWB (Uncompensated)")
ax1.set_ylabel("Raw Distance Scale (m)", fontsize=12, color="gray")
ax1.set_title(f"Metrology Paradigm Comparison on Link {rep_id} $\\rightarrow$ {peer_rep}", fontsize=14, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.4)

ax1_sub = ax1.twinx()
ax1_sub.plot(df_rep["t_sec"], gt_rep, color="black", linestyle="-", linewidth=2.5, label="True Geometry $d^*(t)$")
ax1_sub.plot(df_rep["t_sec"], df_rep["uwb_clean"], color="blue", linewidth=1.8, alpha=0.85, label="Tier 2: Calibrated UWB (Deterministic Layer 1)")

if peer_rep in data:
    px_p = np.interp(df_rep["t_sec"], data[peer_rep]["t_sec"], data[peer_rep]["x"])
    py_p = np.interp(df_rep["t_sec"], data[peer_rep]["t_sec"], data[peer_rep]["y"])
    d_fuse = np.linalg.norm(np.column_stack([df_rep["x"], df_rep["y"]]) - np.column_stack([px_p, py_p]), axis=1)
    ax1_sub.plot(df_rep["t_sec"], d_fuse, color="forestgreen", linewidth=2.0, label="Tier 3: Fused Kinematic-UWB State")

ax1_sub.set_ylabel("True & Calibrated Scale (m)", fontsize=12, color="blue")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_sub.get_legend_handles_labels()
ax1_sub.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=10)

# Bottom Subplot: Error Residuals from Ground Truth (cm)
err_c_cm = (df_rep["uwb_clean"] - gt_rep) * 100.0
ax2.plot(df_rep["t_sec"], err_c_cm, color="blue", linewidth=1.5, alpha=0.7, label="Calibrated Residual ($d_{\\text{calib}} - d^*$)")

if peer_rep in data:
    err_f_cm = (d_fuse - gt_rep) * 100.0
    ax2.plot(df_rep["t_sec"], err_f_cm, color="forestgreen", linewidth=2.0, label="Fused State Residual ($d_{\\text{fused}} - d^*$)")

ax2.axhline(0, color="black", linestyle="-", linewidth=1.0)
ax2.axhline(+5.0, color="red", linestyle=":", label="$\\pm 5\\text{ cm}$ Control Tolerance Bound")
ax2.axhline(-5.0, color="red", linestyle=":")
ax2.set_xlabel("Maneuver Elapsed Time (seconds)", fontsize=12)
ax2.set_ylabel("Ranging Error $\\Delta d$ (cm)", fontsize=12)
ax2.set_ylim(-35, 35)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig1_uwb_three_tier_evaluation.png"))
plt.close()

# ------------------------------------------------------------------------------
# FIGURE 2: 2D SPATIAL KINEMATIC RECONSTRUCTION & GEOMETRY RIGIDITY
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 11), dpi=300)
colors = {1: "#d62728", 2: "#1f77b4", 3: "#2ca02c", 4: "#9467bd"}

# Reference Shapes
sq_init = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
sq_final = np.array([[1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5]])

ax.add_patch(Polygon(sq_init, fill=False, edgecolor="black", linestyle="--", linewidth=1.5, label="Initial Geometry ($L=2.0\\text{ m}$, $\\phi=0^\\circ$)"))
ax.add_patch(Polygon(sq_final, fill=False, edgecolor="darkgreen", linestyle=":", linewidth=2.0, label="Final Geometry ($L=3.0\\text{ m}$, $\\phi=90^\\circ$)"))

# Intermediate Rigidity Polygons at t = 10s and t = 20s
for t_snap, alpha_val in [(10.0, 0.25), (20.0, 0.4)]:
    snap_pts = []
    for r in range(1, 5):
        if r in data:
            x_s = np.interp(t_snap, data[r]["t_sec"], data[r]["x"])
            y_s = np.interp(t_snap, data[r]["t_sec"], data[r]["y"])
            snap_pts.append([x_s, y_s])
    if len(snap_pts) == 4:
        poly_snap = Polygon(np.array(snap_pts), fill=False, edgecolor="gray", linestyle="-.", linewidth=1.0, alpha=alpha_val)
        ax.add_patch(poly_snap)

# Actual Trajectory Traces
for r_id in sorted(data.keys()):
    df = data[r_id]
    ax.plot(df["x"], df["y"], color=colors[r_id], linewidth=2.5, label=f"Robot {r_id} Actual Path")
    ax.scatter(df["x"].iloc[0], df["y"].iloc[0], color=colors[r_id], s=120, marker="o", edgecolors="black", zorder=5)
    ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color=colors[r_id], s=140, marker="s", edgecolors="black", zorder=5)

ax.set_title("Swarm Kinematic Plane Reconstruction: 2D Spatial Rigidity Audit", fontsize=14, fontweight="bold")
ax.set_xlabel("Formation Coordinated Frame X (meters)", fontsize=12)
ax.set_ylabel("Formation Coordinated Frame Y (meters)", fontsize=12)
ax.set_xlim(-2.3, 2.3)
ax.set_ylim(-2.3, 2.3)
ax.set_aspect("equal")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig2_spatial_formation_reconstruction.png"))
plt.close()

# ------------------------------------------------------------------------------
# FIGURE 3: MOTOR CURRENT & INVERTER EMI COUPLING AUDIT
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 9), dpi=300, sharex=True)

for r_id in sorted(data.keys()):
    df = data[r_id]
    axes[0].plot(df["t_sec"], df["current_a"], color=colors[r_id], label=f"R{r_id}")
    axes[1].plot(df["t_sec"], df["uwb_std_noise"], color=colors[r_id], label=f"R{r_id}")
    axes[2].plot(df["t_sec"], df["vbat"], color=colors[r_id], label=f"R{r_id}")

axes[0].set_ylabel("Motor Current (A)", fontsize=11)
axes[0].set_title("Total Drivetrain Electrical Current Consumption (INA226)", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="upper right")

axes[1].set_ylabel("UWB Noise Floor (std_noise)", fontsize=11)
axes[1].set_title("DW1000 Receiver Preamble Noise Floor (EMI Diagnostic Metric)", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3)

axes[2].set_ylabel("Battery Bus Voltage (V)", fontsize=11)
axes[2].set_xlabel("Elapsed Maneuver Time (seconds)", fontsize=12)
axes[2].set_title("2S Li-ion Battery Rail Voltage Sag Under Load", fontsize=12, fontweight="bold")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig3_motor_current_and_uwb_emi.png"))
plt.close()

# ------------------------------------------------------------------------------
# FIGURE 4: KINEMATIC TRACKING ERROR & HEADING DYNAMICS
# ------------------------------------------------------------------------------
fig, (ax_err, ax_th) = plt.subplots(2, 1, figsize=(14, 7), dpi=300, sharex=True)

for r_id in sorted(data.keys()):
    df = data[r_id]
    # Reference position error
    t_arr = df["t_sec"].values
    ref_x = np.array([get_ground_truth(t)[0][r_id][0] for t in t_arr])
    ref_y = np.array([get_ground_truth(t)[0][r_id][1] for t in t_arr])
    pos_err_cm = np.hypot(df["x"].values - ref_x, df["y"].values - ref_y) * 100.0
    
    ax_err.plot(t_arr, pos_err_cm, color=colors[r_id], linewidth=1.8, label=f"Robot {r_id} Pose Error")
    ax_th.plot(t_arr, np.degrees(df["heading"]), color=colors[r_id], linewidth=1.8, label=f"Robot {r_id} Heading")

ax_err.set_ylabel("Tracking Error $\\|\\mathbf{p} - \\mathbf{p}^*\\|$ (cm)", fontsize=11)
ax_err.set_title("Virtual Look-Ahead Point Feedback Linearization Tracking Error", fontsize=13, fontweight="bold")
ax_err.grid(True, alpha=0.3)
ax_err.legend(loc="upper right")

ax_th.set_ylabel("Heading Angle $\\theta$ (degrees)", fontsize=11)
ax_th.set_xlabel("Elapsed Time (seconds)", fontsize=12)
ax_th.set_title("Robot Body Heading Angle Dynamics (HeadingKalmanFilter)", fontsize=13, fontweight="bold")
ax_th.grid(True, alpha=0.3)
ax_th.legend(loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig4_tracking_error_and_heading.png"))
plt.close()

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE! Output artifacts generated in thesis_figures/:")
print("  1. thesis_figures/fig1_uwb_three_tier_evaluation.png")
print("  2. thesis_figures/fig2_spatial_formation_reconstruction.png")
print("  3. thesis_figures/fig3_motor_current_and_uwb_emi.png")
print("  4. thesis_figures/fig4_tracking_error_and_heading.png")
print("  5. thesis_figures/uwb_three_tier_comparison.csv")
print("=" * 90 + "\n")
