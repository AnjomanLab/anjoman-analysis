#!/usr/bin/env python3
"""
IMU-focused analysis for Anjoman propulsion SysID data.

Usage:
    python3 analyze_imu.py

Input:
    All sysid_r*.csv files in the same directory as this script.

Output:
    results_imu/
        imu_summary.csv
        imu_phase_summary.csv
        imu_rest_bias.csv
        imu_correlations.csv
        imu_<robot>_overview.png
        imu_<robot>_gyro.png
        imu_<robot>_accel.png
        imu_<robot>_temperature.png
        imu_<robot>_gyro_spectrum.png
        imu_<robot>_accel_spectrum.png
        imu_<robot>_rest_bias.png
        imu_<robot>_gyro_vs_wheel_difference.png
        imu_<robot>_accel_vs_rpm.png

Main goals:
    - IMU data quality / validity
    - gyro bias and noise during rest
    - accelerometer bias / gravity consistency
    - gyro and accelerometer behavior by SysID phase
    - gyro response to motor acceleration/braking
    - relation between gyro and differential wheel RPM
    - vibration signatures in accelerometer / gyro
    - temperature drift
    - IMU saturation / abnormal values
    - cross-robot comparison

Important:
    The script does NOT assume wheel radius or wheelbase.
    Therefore, wheel-RPM differential is used as a normalized proxy
    for rotational motion rather than converting it to rad/s.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal, stats


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_GLOB = "sysid_r*.csv"
OUTPUT_DIR = SCRIPT_DIR / "results_imu"

FS_DEFAULT = 100.0

# Expected columns from Anjoman SysID logger.
REQUIRED_COLUMNS = [
    "timestamp_us",
    "phase",
    "run_id",
    "pwm_l",
    "pwm_r",
    "rpm_l",
    "rpm_r",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "imu_temp",
    "encoder_valid_l",
    "encoder_valid_r",
    "imu_valid",
]

PHASE_NAMES = {
    0: "REST",
    1: "LEFT_FWD_SWEEP",
    2: "LEFT_REV_SWEEP",
    3: "RIGHT_FWD_SWEEP",
    4: "RIGHT_REV_SWEEP",
    5: "LEFT_DYNAMIC",
    6: "RIGHT_DYNAMIC",
    7: "BOTH_SYNC",
    8: "COAST_DOWN",
    9: "TERMINATION",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def finite(x):
    return np.asarray(x)[np.isfinite(x)]


def safe_mean(x):
    x = finite(x)
    return float(np.mean(x)) if len(x) else np.nan


def safe_std(x):
    x = finite(x)
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def safe_rms(x):
    x = finite(x)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan


def safe_median(x):
    x = finite(x)
    return float(np.median(x)) if len(x) else np.nan


def safe_percentile(x, p):
    x = finite(x)
    return float(np.percentile(x, p)) if len(x) else np.nan


def safe_min(x):
    x = finite(x)
    return float(np.min(x)) if len(x) else np.nan


def safe_max(x):
    x = finite(x)
    return float(np.max(x)) if len(x) else np.nan


def robust_mad(x):
    x = finite(x)
    if not len(x):
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if np.sum(mask) < 3:
        return np.nan

    x = x[mask]
    y = y[mask]

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def phase_name(phase):
    try:
        return PHASE_NAMES.get(int(phase), f"PHASE_{int(phase)}")
    except Exception:
        return "UNKNOWN"


def detect_sampling_rate(df):
    ts = pd.to_numeric(df["timestamp_us"], errors="coerce").to_numpy()

    dt = np.diff(ts) / 1e6
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        return FS_DEFAULT

    median_dt = np.median(dt)

    if median_dt <= 0:
        return FS_DEFAULT

    return float(1.0 / median_dt)


def normalize_robot_name(path):
    return path.stem


def load_csv(path):
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"{path.name}: missing columns: {', '.join(missing)}"
        )

    numeric_cols = [
        c for c in REQUIRED_COLUMNS
        if c not in ["phase", "run_id"]
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time_s"] = (
        df["timestamp_us"] - df["timestamp_us"].iloc[0]
    ) / 1e6

    df["phase_name"] = df["phase"].map(phase_name)

    # Basic IMU-derived quantities.
    df["accel_norm"] = np.sqrt(
        df["accel_x"] ** 2
        + df["accel_y"] ** 2
        + df["accel_z"] ** 2
    )

    df["gyro_norm"] = np.sqrt(
        df["gyro_x"] ** 2
        + df["gyro_y"] ** 2
        + df["gyro_z"] ** 2
    )

    # Differential wheel-speed proxy.
    df["rpm_diff"] = df["rpm_r"] - df["rpm_l"]

    # Common-mode wheel speed.
    df["rpm_mean"] = (df["rpm_r"] + df["rpm_l"]) / 2.0

    # Approximate wheel-speed acceleration.
    dt = df["time_s"].diff()
    dt = dt.replace(0, np.nan)

    df["rpm_diff_rate"] = df["rpm_diff"].diff() / dt
    df["rpm_mean_rate"] = df["rpm_mean"].diff() / dt

    return df


# ---------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------

def analyze_quality(df, robot):
    fs = detect_sampling_rate(df)

    ts = df["timestamp_us"].to_numpy(dtype=float)
    dt_us = np.diff(ts)

    duplicate_count = int(np.sum(dt_us == 0))
    reverse_count = int(np.sum(dt_us < 0))

    positive_dt = dt_us[dt_us > 0]

    if len(positive_dt):
        median_dt_us = np.median(positive_dt)
        gap_threshold_us = 2.5 * median_dt_us
        large_gaps = positive_dt > gap_threshold_us
    else:
        median_dt_us = np.nan
        large_gaps = np.array([])

    imu_valid = df["imu_valid"].astype(bool)

    quality = {
        "robot": robot,
        "samples": len(df),
        "duration_s": safe_max(df["time_s"]),
        "sampling_rate_hz": fs,
        "timestamp_duplicates": duplicate_count,
        "timestamp_reversals": reverse_count,
        "timestamp_large_gaps": int(np.sum(large_gaps)),
        "imu_valid_pct": 100.0 * imu_valid.mean(),
        "accel_finite_pct": 100.0 * np.isfinite(
            df[["accel_x", "accel_y", "accel_z"]]
        ).all(axis=1).mean(),
        "gyro_finite_pct": 100.0 * np.isfinite(
            df[["gyro_x", "gyro_y", "gyro_z"]]
        ).all(axis=1).mean(),
        "temperature_finite_pct": 100.0 * np.isfinite(
            df["imu_temp"]
        ).mean(),
    }

    return quality


# ---------------------------------------------------------------------
# Rest-state analysis
# ---------------------------------------------------------------------

def get_rest_data(df):
    rest = df[df["phase"] == 0].copy()

    if len(rest) == 0:
        return rest

    # Phase 0 consists of brake followed by coast.
    # Use the first 2.0 seconds as a conservative static interval.
    t0 = rest["time_s"].min()

    static = rest[
        (rest["time_s"] >= t0 + 0.5)
        & (rest["time_s"] <= t0 + 2.0)
    ].copy()

    return static


def analyze_rest_bias(df, robot):
    rest = get_rest_data(df)

    if len(rest) < 10:
        return {
            "robot": robot,
            "rest_samples": len(rest),
        }

    result = {
        "robot": robot,
        "rest_samples": len(rest),
        "rest_duration_s": (
            rest["time_s"].max() - rest["time_s"].min()
        ),
    }

    for axis in ["x", "y", "z"]:
        a = rest[f"accel_{axis}"]
        g = rest[f"gyro_{axis}"]

        result[f"accel_{axis}_mean"] = safe_mean(a)
        result[f"accel_{axis}_std"] = safe_std(a)
        result[f"accel_{axis}_mad"] = robust_mad(a)

        result[f"gyro_{axis}_mean"] = safe_mean(g)
        result[f"gyro_{axis}_std"] = safe_std(g)
        result[f"gyro_{axis}_mad"] = robust_mad(g)

    result["accel_norm_mean"] = safe_mean(rest["accel_norm"])
    result["accel_norm_std"] = safe_std(rest["accel_norm"])
    result["accel_norm_median"] = safe_median(rest["accel_norm"])

    result["gyro_norm_mean"] = safe_mean(rest["gyro_norm"])
    result["gyro_norm_std"] = safe_std(rest["gyro_norm"])

    result["imu_temp_mean"] = safe_mean(rest["imu_temp"])
    result["imu_temp_std"] = safe_std(rest["imu_temp"])

    return result


# ---------------------------------------------------------------------
# Phase analysis
# ---------------------------------------------------------------------

def analyze_phases(df, robot):
    rows = []

    for phase, group in df.groupby("phase"):
        if len(group) < 5:
            continue

        row = {
            "robot": robot,
            "phase": int(phase),
            "phase_name": phase_name(phase),
            "samples": len(group),
            "duration_s": (
                group["time_s"].max()
                - group["time_s"].min()
            ),
            "imu_valid_pct": 100.0 * group["imu_valid"].mean(),
        }

        for axis in ["x", "y", "z"]:
            row[f"gyro_{axis}_mean"] = safe_mean(
                group[f"gyro_{axis}"]
            )
            row[f"gyro_{axis}_std"] = safe_std(
                group[f"gyro_{axis}"]
            )
            row[f"gyro_{axis}_rms"] = safe_rms(
                group[f"gyro_{axis}"]
            )

            row[f"accel_{axis}_mean"] = safe_mean(
                group[f"accel_{axis}"]
            )
            row[f"accel_{axis}_std"] = safe_std(
                group[f"accel_{axis}"]
            )
            row[f"accel_{axis}_rms"] = safe_rms(
                group[f"accel_{axis}"]
            )

        row["gyro_norm_mean"] = safe_mean(group["gyro_norm"])
        row["gyro_norm_std"] = safe_std(group["gyro_norm"])
        row["gyro_norm_max"] = safe_max(group["gyro_norm"])

        row["accel_norm_mean"] = safe_mean(group["accel_norm"])
        row["accel_norm_std"] = safe_std(group["accel_norm"])
        row["accel_norm_max"] = safe_max(group["accel_norm"])

        row["rpm_diff_mean"] = safe_mean(group["rpm_diff"])
        row["rpm_diff_std"] = safe_std(group["rpm_diff"])

        row["rpm_mean_mean"] = safe_mean(group["rpm_mean"])

        row["imu_temp_mean"] = safe_mean(group["imu_temp"])
        row["imu_temp_std"] = safe_std(group["imu_temp"])

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# IMU ↔ wheel relationship
# ---------------------------------------------------------------------

def analyze_correlations(df, robot):
    signals = {
        "rpm_diff": df["rpm_diff"],
        "abs_rpm_diff": np.abs(df["rpm_diff"]),
        "rpm_mean": df["rpm_mean"],
        "rpm_diff_rate": df["rpm_diff_rate"],
        "rpm_mean_rate": df["rpm_mean_rate"],
        "pwm_l": df["pwm_l"],
        "pwm_r": df["pwm_r"],
        "imu_temp": df["imu_temp"],
    }

    rows = []

    for gyro_axis in ["x", "y", "z"]:
        for signal_name, x in signals.items():
            rows.append({
                "robot": robot,
                "imu_signal": f"gyro_{gyro_axis}",
                "other_signal": signal_name,
                "correlation": correlation(
                    df[f"gyro_{gyro_axis}"],
                    x,
                ),
            })

    for accel_axis in ["x", "y", "z"]:
        for signal_name, x in signals.items():
            rows.append({
                "robot": robot,
                "imu_signal": f"accel_{accel_axis}",
                "other_signal": signal_name,
                "correlation": correlation(
                    df[f"accel_{accel_axis}"],
                    x,
                ),
            })

    # Norms.
    for signal_name, x in signals.items():
        rows.append({
            "robot": robot,
            "imu_signal": "gyro_norm",
            "other_signal": signal_name,
            "correlation": correlation(
                df["gyro_norm"],
                x,
            ),
        })

        rows.append({
            "robot": robot,
            "imu_signal": "accel_norm",
            "other_signal": signal_name,
            "correlation": correlation(
                df["accel_norm"],
                x,
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Temperature drift
# ---------------------------------------------------------------------

def analyze_temperature(df, robot):
    rows = []

    temp = df["imu_temp"].to_numpy(float)

    for imu_signal in [
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "gyro_norm",
        "accel_x",
        "accel_y",
        "accel_z",
        "accel_norm",
    ]:
        y = df[imu_signal].to_numpy(float)

        mask = np.isfinite(temp) & np.isfinite(y)

        if np.sum(mask) < 10:
            continue

        slope, intercept, r, p, stderr = stats.linregress(
            temp[mask],
            y[mask],
        )

        rows.append({
            "robot": robot,
            "imu_signal": imu_signal,
            "temperature_slope": slope,
            "intercept": intercept,
            "r": r,
            "r_squared": r ** 2,
            "p_value": p,
            "stderr": stderr,
            "temperature_min": np.min(temp[mask]),
            "temperature_max": np.max(temp[mask]),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Saturation / abnormal-value analysis
# ---------------------------------------------------------------------

def analyze_ranges(df, robot):
    rows = []

    for col in [
        "accel_x",
        "accel_y",
        "accel_z",
        "accel_norm",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "gyro_norm",
    ]:
        x = finite(df[col])

        if len(x) == 0:
            continue

        rows.append({
            "robot": robot,
            "signal": col,
            "min": np.min(x),
            "p01": np.percentile(x, 1),
            "p05": np.percentile(x, 5),
            "median": np.median(x),
            "p95": np.percentile(x, 95),
            "p99": np.percentile(x, 99),
            "max": np.max(x),
            "mean": np.mean(x),
            "std": np.std(x),
            "rms": np.sqrt(np.mean(x ** 2)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# PSD / vibration analysis
# ---------------------------------------------------------------------

def compute_psd(x, fs):
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(x)

    if np.sum(mask) < 32:
        return None, None

    x = x[mask]

    # Remove DC component.
    x = x - np.mean(x)

    nperseg = min(2048, len(x))

    if nperseg < 32:
        return None, None

    f, pxx = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
    )

    return f, pxx


def spectral_peaks(x, fs, n=10):
    f, pxx = compute_psd(x, fs)

    if f is None:
        return []

    peaks, _ = signal.find_peaks(pxx)

    if len(peaks) == 0:
        return []

    order = peaks[np.argsort(pxx[peaks])[::-1]]

    result = []

    for idx in order[:n]:
        result.append({
            "frequency_hz": float(f[idx]),
            "power": float(pxx[idx]),
        })

    return result


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_overview(df, robot, outdir):
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    t = df["time_s"]

    axes[0].plot(t, df["gyro_x"], label="gyro_x")
    axes[0].plot(t, df["gyro_y"], label="gyro_y")
    axes[0].plot(t, df["gyro_z"], label="gyro_z")
    axes[0].set_ylabel("Gyro")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, df["accel_x"], label="accel_x")
    axes[1].plot(t, df["accel_y"], label="accel_y")
    axes[1].plot(t, df["accel_z"], label="accel_z")
    axes[1].set_ylabel("Accel")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, df["gyro_norm"], label="gyro norm")
    axes[2].plot(t, df["rpm_diff"], label="RPM difference")
    axes[2].set_ylabel("Motion")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(t, df["accel_norm"], label="accel norm")
    axes[3].plot(t, df["imu_temp"], label="IMU temperature")
    axes[3].set_ylabel("Norm / Temp")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend()
    axes[3].grid(True)

    # Mark phase transitions.
    phases = (
        df[["time_s", "phase", "phase_name"]]
        .drop_duplicates("phase")
        .sort_values("time_s")
    )

    for _, row in phases.iterrows():
        for ax in axes:
            ax.axvline(row["time_s"], linestyle="--", alpha=0.25)

    fig.suptitle(f"{robot} — IMU Overview")
    fig.tight_layout()
    fig.savefig(outdir / f"imu_{robot}_overview.png", dpi=160)
    plt.close(fig)


def plot_gyro(df, robot, outdir):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, axis in zip(axes, ["x", "y", "z"]):
        ax.plot(
            df["time_s"],
            df[f"gyro_{axis}"],
            linewidth=0.8,
        )
        ax.set_ylabel(f"gyro {axis}")
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")

    fig.suptitle(f"{robot} — Gyroscope")
    fig.tight_layout()
    fig.savefig(outdir / f"imu_{robot}_gyro.png", dpi=160)
    plt.close(fig)


def plot_accel(df, robot, outdir):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for ax, axis in zip(
        axes[:3],
        ["x", "y", "z"],
    ):
        ax.plot(
            df["time_s"],
            df[f"accel_{axis}"],
            linewidth=0.8,
        )
        ax.set_ylabel(f"accel {axis}")
        ax.grid(True)

    axes[3].plot(
        df["time_s"],
        df["accel_norm"],
        linewidth=0.8,
    )
    axes[3].set_ylabel("norm")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True)

    fig.suptitle(f"{robot} — Accelerometer")
    fig.tight_layout()
    fig.savefig(outdir / f"imu_{robot}_accel.png", dpi=160)
    plt.close(fig)


def plot_temperature(df, robot, outdir):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        df["time_s"],
        df["imu_temp"],
        linewidth=1.0,
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("IMU temperature")
    ax.grid(True)

    fig.suptitle(f"{robot} — IMU Temperature")
    fig.tight_layout()
    fig.savefig(
        outdir / f"imu_{robot}_temperature.png",
        dpi=160,
    )
    plt.close(fig)


def plot_rest_bias(df, robot, outdir):
    rest = get_rest_data(df)

    if len(rest) < 10:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(
        rest["time_s"],
        rest["gyro_x"],
        label="gyro_x",
    )
    axes[0].plot(
        rest["time_s"],
        rest["gyro_y"],
        label="gyro_y",
    )
    axes[0].plot(
        rest["time_s"],
        rest["gyro_z"],
        label="gyro_z",
    )
    axes[0].set_ylabel("Gyro")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        rest["time_s"],
        rest["accel_x"],
        label="accel_x",
    )
    axes[1].plot(
        rest["time_s"],
        rest["accel_y"],
        label="accel_y",
    )
    axes[1].plot(
        rest["time_s"],
        rest["accel_z"],
        label="accel_z",
    )
    axes[1].set_ylabel("Accel")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f"{robot} — Rest-State IMU Bias / Noise")
    fig.tight_layout()
    fig.savefig(
        outdir / f"imu_{robot}_rest_bias.png",
        dpi=160,
    )
    plt.close(fig)


def plot_gyro_vs_wheel_difference(df, robot, outdir):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(
        df["time_s"],
        df["gyro_z"],
        label="gyro_z",
        linewidth=0.8,
    )

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Gyro Z")

    ax2 = ax1.twinx()

    ax2.plot(
        df["time_s"],
        df["rpm_diff"],
        label="RPM_R - RPM_L",
        linewidth=0.8,
        alpha=0.7,
    )

    ax2.set_ylabel("Wheel RPM difference")

    ax1.grid(True)

    fig.suptitle(
        f"{robot} — Gyro Z vs Differential Wheel Speed"
    )

    fig.tight_layout()
    fig.savefig(
        outdir / f"imu_{robot}_gyro_vs_wheel_difference.png",
        dpi=160,
    )
    plt.close(fig)


def plot_accel_vs_rpm(df, robot, outdir):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(
        df["time_s"],
        df["accel_norm"],
        label="Acceleration norm",
        linewidth=0.8,
    )

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Acceleration norm")

    ax2 = ax1.twinx()

    ax2.plot(
        df["time_s"],
        np.abs(df["rpm_mean_rate"]),
        label="|RPM acceleration|",
        linewidth=0.8,
        alpha=0.7,
    )

    ax2.set_ylabel("|RPM acceleration|")

    ax1.grid(True)

    fig.suptitle(
        f"{robot} — Acceleration Vibration vs Wheel-Speed Dynamics"
    )

    fig.tight_layout()
    fig.savefig(
        outdir / f"imu_{robot}_accel_vs_rpm.png",
        dpi=160,
    )
    plt.close(fig)


def plot_spectrum(df, robot, outdir):
    fs = detect_sampling_rate(df)

    signals = [
        ("gyro_x", "Gyro X"),
        ("gyro_y", "Gyro Y"),
        ("gyro_z", "Gyro Z"),
        ("accel_x", "Accel X"),
        ("accel_y", "Accel Y"),
        ("accel_z", "Accel Z"),
    ]

    fig, axes = plt.subplots(
        len(signals),
        1,
        figsize=(12, 18),
    )

    for ax, (column, title) in zip(axes, signals):
        f, pxx = compute_psd(df[column], fs)

        if f is not None:
            ax.semilogy(f, pxx)
            ax.set_xlim(0, min(fs / 2, 50))

        ax.set_ylabel(title)
        ax.grid(True)

    axes[-1].set_xlabel("Frequency [Hz]")

    fig.suptitle(f"{robot} — IMU Power Spectral Density")
    fig.tight_layout()

    fig.savefig(
        outdir / f"imu_{robot}_gyro_spectrum.png",
        dpi=160,
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Spectral report
# ---------------------------------------------------------------------

def analyze_spectrum(df, robot):
    fs = detect_sampling_rate(df)

    rows = []

    for column in [
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "accel_x",
        "accel_y",
        "accel_z",
    ]:
        peaks = spectral_peaks(
            df[column],
            fs,
            n=10,
        )

        for rank, peak in enumerate(peaks, start=1):
            rows.append({
                "robot": robot,
                "signal": column,
                "rank": rank,
                "frequency_hz": peak["frequency_hz"],
                "power": peak["power"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Dynamic gyro metrics
# ---------------------------------------------------------------------

def analyze_dynamic_response(df, robot):
    """
    Quantify gyro behavior around phase transitions.

    This intentionally does not force a physical model because the
    actual mounting orientation of the BMI160 and robot wheel geometry
    are not encoded here.

    It instead measures:
        - gyro baseline before command
        - gyro peak after command
        - change in gyro
        - response RMS
        - relation to wheel-speed differential
    """

    rows = []

    phase_changes = (
        df["phase"]
        .ne(df["phase"].shift())
    )

    starts = df.index[phase_changes].tolist()

    for idx in starts:
        row = df.loc[idx]

        phase = int(row["phase"])

        if phase not in [5, 6, 7, 8]:
            continue

        t = row["time_s"]

        before = df[
            (df["time_s"] >= t - 0.5)
            & (df["time_s"] < t)
        ]

        after = df[
            (df["time_s"] >= t)
            & (df["time_s"] <= t + 0.8)
        ]

        if len(before) < 5 or len(after) < 5:
            continue

        result = {
            "robot": robot,
            "transition_time_s": t,
            "phase": phase,
            "phase_name": phase_name(phase),
            "pwm_l": row["pwm_l"],
            "pwm_r": row["pwm_r"],
        }

        for axis in ["x", "y", "z"]:
            g_before = safe_mean(before[f"gyro_{axis}"])
            g_after = after[f"gyro_{axis}"]

            result[f"gyro_{axis}_baseline"] = g_before
            result[f"gyro_{axis}_peak_abs"] = safe_max(
                np.abs(g_after - g_before)
            )
            result[f"gyro_{axis}_rms_after"] = safe_rms(
                g_after - g_before
            )

        result["rpm_diff_before"] = safe_mean(
            before["rpm_diff"]
        )
        result["rpm_diff_after"] = safe_mean(
            after["rpm_diff"]
        )

        result["rpm_diff_change"] = (
            result["rpm_diff_after"]
            - result["rpm_diff_before"]
        )

        rows.append(result)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Cross-robot comparison
# ---------------------------------------------------------------------

def make_cross_robot_summary(
    quality_rows,
    rest_rows,
    phase_rows,
    temperature_rows,
):
    quality = pd.DataFrame(quality_rows)
    rest = pd.DataFrame(rest_rows)

    if quality.empty:
        return pd.DataFrame()

    rows = []

    for robot in quality["robot"]:
        q = quality[
            quality["robot"] == robot
        ].iloc[0]

        r = rest[
            rest["robot"] == robot
        ]

        row = {
            "robot": robot,
            "samples": q["samples"],
            "duration_s": q["duration_s"],
            "sampling_rate_hz": q["sampling_rate_hz"],
            "imu_valid_pct": q["imu_valid_pct"],
        }

        if not r.empty:
            r = r.iloc[0]

            row["rest_gyro_x_bias"] = r.get(
                "gyro_x_mean", np.nan
            )
            row["rest_gyro_y_bias"] = r.get(
                "gyro_y_mean", np.nan
            )
            row["rest_gyro_z_bias"] = r.get(
                "gyro_z_mean", np.nan
            )

            row["rest_gyro_x_noise"] = r.get(
                "gyro_x_std", np.nan
            )
            row["rest_gyro_y_noise"] = r.get(
                "gyro_y_std", np.nan
            )
            row["rest_gyro_z_noise"] = r.get(
                "gyro_z_std", np.nan
            )

            row["rest_accel_norm"] = r.get(
                "accel_norm_mean", np.nan
            )

            row["rest_accel_norm_std"] = r.get(
                "accel_norm_std", np.nan
            )

            row["rest_temperature"] = r.get(
                "imu_temp_mean", np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        SCRIPT_DIR.glob(INPUT_GLOB)
    )

    if not files:
        raise SystemExit(
            f"No files matching {INPUT_GLOB} found in "
            f"{SCRIPT_DIR}"
        )

    print(f"Found {len(files)} SysID file(s).")
    print(f"Output: {OUTPUT_DIR}")
    print()

    quality_rows = []
    rest_rows = []
    phase_frames = []
    correlation_frames = []
    temperature_frames = []
    range_frames = []
    spectrum_frames = []
    dynamic_frames = []

    for path in files:
        robot = normalize_robot_name(path)

        print(f"Analyzing {path.name} ...")

        try:
            df = load_csv(path)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        quality = analyze_quality(
            df,
            robot,
        )

        quality_rows.append(quality)

        rest = analyze_rest_bias(
            df,
            robot,
        )

        rest_rows.append(rest)

        phase_frames.append(
            analyze_phases(
                df,
                robot,
            )
        )

        correlation_frames.append(
            analyze_correlations(
                df,
                robot,
            )
        )

        temperature_frames.append(
            analyze_temperature(
                df,
                robot,
            )
        )

        range_frames.append(
            analyze_ranges(
                df,
                robot,
            )
        )

        spectrum_frames.append(
            analyze_spectrum(
                df,
                robot,
            )
        )

        dynamic_frames.append(
            analyze_dynamic_response(
                df,
                robot,
            )
        )

        # Plots.
        plot_overview(
            df,
            robot,
            OUTPUT_DIR,
        )

        plot_gyro(
            df,
            robot,
            OUTPUT_DIR,
        )

        plot_accel(
            df,
            robot,
            OUTPUT_DIR,
        )

        plot_temperature(
            df,
            robot,
            OUTPUT_DIR,
        )

        plot_rest_bias(
            df,
            robot,
            OUTPUT_DIR,
        )

        plot_gyro_vs_wheel_difference(
            df,
            robot,
            OUTPUT_DIR,
        )

        plot_accel_vs_rpm(
            df,
            robot,
            OUTPUT_DIR,
        )

        plot_spectrum(
            df,
            robot,
            OUTPUT_DIR,
        )

        print(
            f"  samples={len(df)}, "
            f"fs={detect_sampling_rate(df):.3f} Hz, "
            f"IMU valid={100.0 * df['imu_valid'].mean():.2f}%"
        )

    # -----------------------------------------------------------------
    # Save tables
    # -----------------------------------------------------------------

    pd.DataFrame(quality_rows).to_csv(
        OUTPUT_DIR / "imu_summary.csv",
        index=False,
    )

    pd.DataFrame(rest_rows).to_csv(
        OUTPUT_DIR / "imu_rest_bias.csv",
        index=False,
    )

    if phase_frames:
        pd.concat(
            phase_frames,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR / "imu_phase_summary.csv",
            index=False,
        )

    if correlation_frames:
        pd.concat(
            correlation_frames,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR / "imu_correlations.csv",
            index=False,
        )

    if temperature_frames:
        pd.concat(
            temperature_frames,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR / "imu_temperature_analysis.csv",
            index=False,
        )

    if range_frames:
        pd.concat(
            range_frames,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR / "imu_signal_ranges.csv",
            index=False,
        )

    if spectrum_frames:
        pd.concat(
            spectrum_frames,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR / "imu_spectral_peaks.csv",
            index=False,
        )

    if dynamic_frames:
        pd.concat(
            dynamic_frames,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR / "imu_dynamic_response.csv",
            index=False,
        )

    # Cross-robot summary.
    phase_df = (
        pd.concat(
            phase_frames,
            ignore_index=True,
        )
        if phase_frames
        else pd.DataFrame()
    )

    temp_df = (
        pd.concat(
            temperature_frames,
            ignore_index=True,
        )
        if temperature_frames
        else pd.DataFrame()
    )

    cross = make_cross_robot_summary(
        quality_rows,
        rest_rows,
        phase_df,
        temp_df,
    )

    cross.to_csv(
        OUTPUT_DIR / "imu_cross_robot_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print("IMU ANALYSIS SUMMARY")
    print("=" * 72)

    quality_df = pd.DataFrame(quality_rows)

    if not quality_df.empty:
        print()
        print("DATA QUALITY")
        print(
            quality_df[
                [
                    "robot",
                    "samples",
                    "sampling_rate_hz",
                    "imu_valid_pct",
                    "timestamp_duplicates",
                    "timestamp_reversals",
                    "timestamp_large_gaps",
                ]
            ].to_string(index=False)
        )

    rest_df = pd.DataFrame(rest_rows)

    if not rest_df.empty:
        print()
        print("REST-STATE GYRO BIAS / NOISE")
        cols = [
            "robot",
            "gyro_x_mean",
            "gyro_y_mean",
            "gyro_z_mean",
            "gyro_x_std",
            "gyro_y_std",
            "gyro_z_std",
            "accel_norm_mean",
            "accel_norm_std",
            "imu_temp_mean",
        ]

        available = [
            c for c in cols
            if c in rest_df.columns
        ]

        print(
            rest_df[available].to_string(
                index=False
            )
        )

    print()
    print(f"Results written to: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
