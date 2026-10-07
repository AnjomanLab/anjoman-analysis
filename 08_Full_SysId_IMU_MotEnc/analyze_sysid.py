#!/usr/bin/env python3

"""
ANJOMAN Ground SysID Analyzer
==============================

Usage:
    python3 analyze_sysid.py

The script automatically discovers:
    sysid_r*.csv

in the same directory as this Python file.

Outputs are written to:
    results/

Dependencies:
    numpy
    pandas
    scipy
    matplotlib
"""

from __future__ import annotations

import math
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib

# Safe for headless systems / SSH
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, least_squares


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results"

CSV_PATTERN = "sysid_r*.csv"

EXPECTED_TEST_DURATION_S = 150.0
EXPECTED_SAMPLE_PERIOD_S = 0.010
EXPECTED_SAMPLE_RATE_HZ = 100.0

# Firmware static excitation levels
FINE_LEVELS = np.array([
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.90
])

STATIC_STEP_DURATION_S = 1.2

# Fraction of each static step discarded as transient.
# The final part is used as steady-state.
STATIC_TRANSIENT_FRACTION = 0.50

# Dynamic phase segment definitions.
# (label, duration_s, command)
DYNAMIC_LEFT_SEGMENTS = [
    ("L_0.50_FWD", 3.0, +0.50),
    ("L_BRAKE_1", 2.0, 0.00),
    ("L_0.80_FWD", 3.0, +0.80),
    ("L_BRAKE_2", 2.0, 0.00),
    ("L_0.60_REV", 3.0, -0.60),
    ("L_BRAKE_3", 1.0, 0.00),
]

DYNAMIC_RIGHT_SEGMENTS = [
    ("R_0.50_FWD", 3.0, +0.50),
    ("R_BRAKE_1", 2.0, 0.00),
    ("R_0.80_FWD", 3.0, +0.80),
    ("R_BRAKE_2", 2.0, 0.00),
    ("R_0.60_REV", 3.0, -0.60),
    ("R_BRAKE_3", 1.0, 0.00),
]

# Phase 7
ASYMMETRY_LEVELS = [0.45, 0.60, 0.75, 0.90]
ASYMMETRY_STEP_DURATION_S = 2.5

# Coast-down
COAST_DRIVE_DURATION_S = 4.0
COAST_DURATION_S = 4.0

# Static model minimum points
MIN_STATIC_POINTS_FOR_FIT = 4

# Dynamic fitting
MIN_DYNAMIC_SAMPLES = 10

# Robust clipping
ROBUST_Z_LIMIT = 5.0

# Plot resolution
FIG_DPI = 150


# =============================================================================
# EXPECTED CSV COLUMNS
# =============================================================================

EXPECTED_COLUMNS = [
    "timestamp_us",
    "test_id",
    "phase",
    "run_id",
    "cmd_v",
    "cmd_omega",
    "pwm_l",
    "pwm_r",
    "mode_l",
    "mode_r",
    "raw_angle_l",
    "raw_angle_r",
    "delta_angle_l",
    "delta_angle_r",
    "steps_l",
    "steps_r",
    "rpm_l",
    "rpm_r",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "imu_temperature",
    "battery_voltage",
    "driver_voltage",
    "driver_current",
    "driver_power",
    "encoder_valid_l",
    "encoder_valid_r",
    "imu_valid",
    "ina_valid",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StaticResult:
    robot: int
    side: str
    direction: str
    deadband_fit: float
    gain_rpm_per_pwm: float
    intercept_rpm: float
    r2: float
    rmse_rpm: float
    n_levels: int
    first_active_pwm: float
    max_rpm: float


@dataclass
class DynamicResult:
    robot: int
    side: str
    transition: str
    command: float
    tau_s: float
    rise_time_s: float
    settling_time_s: float
    overshoot_pct: float
    steady_rpm: float
    initial_rpm: float
    final_rpm: float
    fit_r2: float
    fit_rmse_rpm: float
    n_samples: int


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def safe_int(value) -> int:
    try:
        return int(value)
    except Exception:
        return -1


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot <= 1e-15:
        return np.nan

    return float(1.0 - ss_res / ss_tot)


def median_or_nan(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def mean_or_nan(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def std_or_nan(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return float(np.std(x, ddof=1))


def percentile_or_nan(x, q) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else np.nan


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def extract_robot_id(path: Path) -> Optional[int]:
    match = re.fullmatch(r"sysid_r(\d+)\.csv", path.name, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clear_generated_outputs() -> None:
    """
    Clear only files generated by this analyzer.
    The input CSV files are never touched.
    """

    if not OUTPUT_DIR.exists():
        return

    for item in OUTPUT_DIR.iterdir():
        if item.is_file():
            try:
                item.unlink()
            except Exception as exc:
                print(f"Warning: could not remove {item}: {exc}")


# =============================================================================
# DATA LOADING
# =============================================================================

def discover_csv_files() -> list[Path]:
    files = []

    for path in sorted(SCRIPT_DIR.glob(CSV_PATTERN)):
        if path.is_file():
            robot_id = extract_robot_id(path)
            if robot_id is not None:
                files.append(path)

    return files


def load_robot_csv(path: Path) -> pd.DataFrame:
    print(f"Loading: {path.name}")

    df = pd.read_csv(path)

    # Strip accidental whitespace from headers
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"{path.name}: missing columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    # Keep only known expected fields.
    # This also protects against future extra firmware fields.
    df = df[EXPECTED_COLUMNS].copy()

    # Numeric conversion
    for col in EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows without timestamps
    before = len(df)
    df = df.dropna(subset=["timestamp_us"]).copy()

    if len(df) < before:
        print(
            f"  Removed {before - len(df)} rows with invalid timestamp."
        )

    # Timestamp relative to first logged sample
    df["time_s"] = (
        df["timestamp_us"] - df["timestamp_us"].iloc[0]
    ) / 1e6

    # Actual sample interval
    df["dt_s"] = df["timestamp_us"].diff() / 1e6

    # Useful absolute RPM signals
    df["abs_rpm_l"] = df["rpm_l"].abs()
    df["abs_rpm_r"] = df["rpm_r"].abs()

    # Numeric validity flags
    df["encoder_valid_l"] = df["encoder_valid_l"].fillna(0)
    df["encoder_valid_r"] = df["encoder_valid_r"].fillna(0)
    df["imu_valid"] = df["imu_valid"].fillna(0)
    df["ina_valid"] = df["ina_valid"].fillna(0)

    return df.reset_index(drop=True)


# =============================================================================
# DATA QUALITY
# =============================================================================

def analyze_data_quality(df: pd.DataFrame, robot: int) -> dict:
    ts = df["timestamp_us"].to_numpy(dtype=float)
    dt = np.diff(ts) / 1e6

    positive_dt = dt[dt > 0]
    duplicate_count = int(np.sum(dt == 0))
    negative_count = int(np.sum(dt < 0))

    if len(positive_dt):
        median_dt = float(np.median(positive_dt))
        mean_dt = float(np.mean(positive_dt))
        min_dt = float(np.min(positive_dt))
        max_dt = float(np.max(positive_dt))
        actual_rate = 1.0 / median_dt
    else:
        median_dt = mean_dt = min_dt = max_dt = actual_rate = np.nan

    # A "gap" is a missing sample relative to the nominal 10 ms interval.
    #
    # We use > 1.5 nominal periods to avoid treating normal scheduler jitter
    # as a lost record.
    gap_threshold = EXPECTED_SAMPLE_PERIOD_S * 1.5

    gap_mask = positive_dt > gap_threshold
    gap_count = int(np.sum(gap_mask))

    estimated_missing_samples = 0

    for gap in positive_dt[gap_mask]:
        estimated_missing_samples += max(
            0,
            int(round(gap / EXPECTED_SAMPLE_PERIOD_S)) - 1
        )

    duration_s = (
        float((ts[-1] - ts[0]) / 1e6)
        if len(ts) >= 2
        else np.nan
    )

    phase_counts = df["phase"].value_counts().to_dict()

    phase_presence = {
        phase: int(phase_counts.get(phase, 0))
        for phase in range(10)
    }

    quality = {
        "robot": robot,
        "samples": len(df),
        "duration_s": duration_s,
        "sample_rate_hz": actual_rate,
        "median_dt_ms": median_dt * 1000.0,
        "mean_dt_ms": mean_dt * 1000.0,
        "min_dt_ms": min_dt * 1000.0,
        "max_dt_ms": max_dt * 1000.0,
        "timestamp_duplicates": duplicate_count,
        "timestamp_negative_steps": negative_count,
        "timestamp_gaps": gap_count,
        "estimated_missing_samples": estimated_missing_samples,
        "encoder_valid_l_pct": 100.0 * df["encoder_valid_l"].mean(),
        "encoder_valid_r_pct": 100.0 * df["encoder_valid_r"].mean(),
        "imu_valid_pct": 100.0 * df["imu_valid"].mean(),
        "ina_valid_pct": 100.0 * df["ina_valid"].mean(),
    }

    for phase in range(10):
        quality[f"phase_{phase}_samples"] = phase_presence[phase]

    return quality


def validate_phases(df: pd.DataFrame) -> list[str]:
    warnings_list = []

    actual_phases = sorted(
        set(int(x) for x in df["phase"].dropna().unique())
    )

    expected_phases = list(range(10))

    missing = [p for p in expected_phases if p not in actual_phases]
    unexpected = [p for p in actual_phases if p not in expected_phases]

    if missing:
        warnings_list.append(
            f"Missing phase(s): {missing}"
        )

    if unexpected:
        warnings_list.append(
            f"Unexpected phase(s): {unexpected}"
        )

    # Check phase ordering.
    compressed = []
    last = None

    for phase in df["phase"].astype(int):
        if phase != last:
            compressed.append(phase)
            last = phase

    if compressed != sorted(set(compressed)):
        warnings_list.append(
            f"Phase order is not monotonic: {compressed}"
        )

    return warnings_list


# =============================================================================
# PHASE SEGMENTATION
# =============================================================================

def phase_dataframe(df: pd.DataFrame, phase: int) -> pd.DataFrame:
    return df[df["phase"] == phase].copy()


def split_phase_by_fixed_steps(
    df_phase: pd.DataFrame,
    n_steps: int,
    step_duration_s: float,
) -> list[pd.DataFrame]:
    """
    Split a phase into fixed-duration command segments.

    We intentionally derive the local time from the first sample in the phase,
    rather than relying on absolute timestamp, because the firmware timestamp
    does not expose startTestTimeMs.
    """

    if df_phase.empty:
        return [df_phase.copy() for _ in range(n_steps)]

    local_t = (
        df_phase["timestamp_us"]
        - df_phase["timestamp_us"].iloc[0]
    ) / 1e6

    result = []

    for i in range(n_steps):
        start = i * step_duration_s
        end = (i + 1) * step_duration_s

        mask = (local_t >= start) & (local_t < end)

        result.append(df_phase.loc[mask].copy())

    return result


def steady_state_window(
    segment: pd.DataFrame,
    transient_fraction: float = STATIC_TRANSIENT_FRACTION,
) -> pd.DataFrame:
    if segment.empty:
        return segment.copy()

    t0 = segment["timestamp_us"].iloc[0]
    t1 = segment["timestamp_us"].iloc[-1]

    duration = (t1 - t0) / 1e6

    start_after = duration * transient_fraction

    local_t = (segment["timestamp_us"] - t0) / 1e6

    return segment.loc[local_t >= start_after].copy()


# =============================================================================
# STATIC IDENTIFICATION
# =============================================================================

def collect_static_levels(
    df: pd.DataFrame,
    robot: int,
) -> pd.DataFrame:
    records = []

    phase_motor_map = {
        1: ("L", "FWD", "pwm_l", "rpm_l"),
        2: ("L", "REV", "pwm_l", "rpm_l"),
        3: ("R", "FWD", "pwm_r", "rpm_r"),
        4: ("R", "REV", "pwm_r", "rpm_r"),
    }

    for phase, (side, direction, pwm_col, rpm_col) in phase_motor_map.items():

        p = phase_dataframe(df, phase)

        segments = split_phase_by_fixed_steps(
            p,
            n_steps=len(FINE_LEVELS),
            step_duration_s=STATIC_STEP_DURATION_S,
        )

        for i, (expected_level, segment) in enumerate(
            zip(FINE_LEVELS, segments)
        ):

            steady = steady_state_window(segment)

            if steady.empty:
                continue

            pwm_values = steady[pwm_col].to_numpy(dtype=float)
            rpm_values = steady[rpm_col].to_numpy(dtype=float)

            valid = np.isfinite(pwm_values) & np.isfinite(rpm_values)

            pwm_values = pwm_values[valid]
            rpm_values = rpm_values[valid]

            if len(rpm_values) == 0:
                continue

            records.append({
                "robot": robot,
                "phase": phase,
                "side": side,
                "direction": direction,
                "level_index": i,
                "command_pwm_expected": expected_level,
                "command_pwm_mean": mean_or_nan(pwm_values),
                "command_pwm_std": std_or_nan(pwm_values),
                "rpm_mean": mean_or_nan(rpm_values),
                "rpm_std": std_or_nan(rpm_values),
                "rpm_median": median_or_nan(rpm_values),
                "rpm_min": float(np.min(rpm_values)),
                "rpm_max": float(np.max(rpm_values)),
                "n_samples": len(rpm_values),
            })

    return pd.DataFrame(records)


def fit_deadzone_linear(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Fit:

        y = gain * max(x - deadband, 0)

    where:
        x = |PWM|
        y = |RPM|

    This is the primary static model.

    Returns:
        deadband
        gain
        R2
        RMSE
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)

    x = x[valid]
    y = y[valid]

    if len(x) < MIN_STATIC_POINTS_FOR_FIT:
        return np.nan, np.nan, np.nan, np.nan

    def residual(params):
        deadband, gain = params
        pred = gain * np.maximum(x - deadband, 0.0)
        return pred - y

    # Initial estimate
    initial_deadband = min(0.10, max(0.0, float(np.min(x)) * 0.5))

    positive_x = x > initial_deadband

    if np.any(positive_x):
        initial_gain = max(
            1.0,
            float(
                np.sum(x[positive_x] * y[positive_x])
                / max(np.sum(x[positive_x] ** 2), 1e-12)
            ),
        )
    else:
        initial_gain = 100.0

    try:
        result = least_squares(
            residual,
            x0=[initial_deadband, initial_gain],
            bounds=(
                [0.0, 0.0],
                [min(0.5, float(np.max(x))), np.inf],
            ),
        )

        deadband, gain = result.x

    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    prediction = gain * np.maximum(x - deadband, 0.0)

    return (
        float(deadband),
        float(gain),
        r_squared(y, prediction),
        rmse(y, prediction),
    )


def analyze_static(
    static_levels: pd.DataFrame,
    robot: int,
) -> list[StaticResult]:

    results = []

    for side in ["L", "R"]:
        for direction in ["FWD", "REV"]:

            d = static_levels[
                (static_levels["robot"] == robot)
                & (static_levels["side"] == side)
                & (static_levels["direction"] == direction)
            ].copy()

            if d.empty:
                continue

            x = d["command_pwm_mean"].abs().to_numpy(dtype=float)
            y = d["rpm_mean"].abs().to_numpy(dtype=float)

            deadband, gain, r2, fit_rmse = fit_deadzone_linear(x, y)

            # Observed first clearly moving level.
            # A threshold of 5 RPM is intentionally conservative.
            moving = d.loc[d["rpm_mean"].abs() > 5.0]

            if moving.empty:
                first_active_pwm = np.nan
            else:
                first_active_pwm = float(
                    moving["command_pwm_mean"].abs().min()
                )

            results.append(
                StaticResult(
                    robot=robot,
                    side=side,
                    direction=direction,
                    deadband_fit=deadband,
                    gain_rpm_per_pwm=gain,
                    intercept_rpm=0.0,
                    r2=r2,
                    rmse_rpm=fit_rmse,
                    n_levels=len(d),
                    first_active_pwm=first_active_pwm,
                    max_rpm=float(y.max()) if len(y) else np.nan,
                )
            )

    return results


# =============================================================================
# DYNAMIC RESPONSE
# =============================================================================

def segment_dynamic_phase(
    df_phase: pd.DataFrame,
    side: str,
) -> list[tuple[str, float, pd.DataFrame]]:
    """
    Split phase 5 or 6 into its exact firmware-defined segments.
    """

    if df_phase.empty:
        return []

    if side == "L":
        definitions = DYNAMIC_LEFT_SEGMENTS
    else:
        definitions = DYNAMIC_RIGHT_SEGMENTS

    phase_start = df_phase["timestamp_us"].iloc[0]

    local_t = (
        df_phase["timestamp_us"] - phase_start
    ) / 1e6

    result = []

    cursor = 0.0

    for label, duration, command in definitions:
        mask = (local_t >= cursor) & (local_t < cursor + duration)

        segment = df_phase.loc[mask].copy()

        result.append((label, command, segment))

        cursor += duration

    return result


def first_order_response(t, y0, yf, tau):
    return y0 + (yf - y0) * (1.0 - np.exp(-t / tau))


def estimate_tau_63(
    t: np.ndarray,
    y: np.ndarray,
    y0: float,
    yf: float,
) -> float:

    delta = yf - y0

    if abs(delta) < 1e-6:
        return np.nan

    target = y0 + 0.6321205588 * delta

    if delta > 0:
        indices = np.where(y >= target)[0]
    else:
        indices = np.where(y <= target)[0]

    if len(indices) == 0:
        return np.nan

    i = int(indices[0])

    if i == 0:
        return float(t[0])

    # Linear interpolation
    t1, t2 = t[i - 1], t[i]
    y1, y2 = y[i - 1], y[i]

    if abs(y2 - y1) < 1e-12:
        return float(t2)

    return float(
        t1 + (target - y1) * (t2 - t1) / (y2 - y1)
    )


def calculate_rise_time(
    t: np.ndarray,
    y: np.ndarray,
    y0: float,
    yf: float,
) -> float:

    delta = yf - y0

    if abs(delta) < 1e-6:
        return np.nan

    y10 = y0 + 0.10 * delta
    y90 = y0 + 0.90 * delta

    if delta > 0:
        i10 = np.where(y >= y10)[0]
        i90 = np.where(y >= y90)[0]
    else:
        i10 = np.where(y <= y10)[0]
        i90 = np.where(y <= y90)[0]

    if len(i10) == 0 or len(i90) == 0:
        return np.nan

    return float(t[i90[0]] - t[i10[0]])


def calculate_settling_time(
    t: np.ndarray,
    y: np.ndarray,
    y_final: float,
    tolerance: float = 0.02,
) -> float:

    amplitude = abs(y_final)

    # If final value is close to zero, use signal range instead.
    if amplitude < 1e-6:
        amplitude = max(
            1.0,
            float(np.max(np.abs(y - y[0])))
        )

    band = tolerance * amplitude

    inside = np.abs(y - y_final) <= band

    if not np.any(inside):
        return np.nan

    # Settling time = first time after which all samples remain inside.
    for i in range(len(t)):
        if np.all(inside[i:]):
            return float(t[i])

    return np.nan


def calculate_overshoot(
    y: np.ndarray,
    y0: float,
    yf: float,
) -> float:

    delta = yf - y0

    if abs(delta) < 1e-6:
        return np.nan

    if delta > 0:
        peak = float(np.max(y))
        overshoot = max(0.0, peak - yf)
    else:
        valley = float(np.min(y))
        overshoot = max(0.0, yf - valley)

    return float(100.0 * overshoot / abs(delta))


def fit_dynamic_transition(
    segment: pd.DataFrame,
    rpm_col: str,
    command: float,
    label: str,
    robot: int,
    side: str,
    baseline: Optional[pd.DataFrame] = None,
) -> Optional[DynamicResult]:

    if segment.empty:
        return None

    if len(segment) < MIN_DYNAMIC_SAMPLES:
        return None

    t = (
        segment["timestamp_us"].to_numpy(dtype=float)
        - segment["timestamp_us"].iloc[0]
    ) / 1e6

    y = segment[rpm_col].to_numpy(dtype=float)

    valid = np.isfinite(t) & np.isfinite(y)

    t = t[valid]
    y = y[valid]

    if len(y) < MIN_DYNAMIC_SAMPLES:
        return None

    # Baseline from previous brake segment when available.
    if baseline is not None and not baseline.empty:
        base_values = baseline[rpm_col].to_numpy(dtype=float)
        base_values = base_values[np.isfinite(base_values)]

        if len(base_values):
            y0 = float(np.median(base_values[-max(5, len(base_values) // 2):]))
        else:
            y0 = float(y[0])
    else:
        y0 = float(y[0])

    # Final steady value: use final 20% of the segment.
    n_tail = max(5, int(0.20 * len(y)))
    yf = float(np.median(y[-n_tail:]))

    # Fit first-order model.
    tau_fit = np.nan
    fit_r2 = np.nan
    fit_error = np.nan

    try:
        if abs(yf - y0) > 1.0:

            def model(t_, yf_, tau_):
                return y0 + (yf_ - y0) * (
                    1.0 - np.exp(-np.maximum(t_, 0.0) / tau_)
                )

            initial_tau = max(
                0.05,
                estimate_tau_63(t, y, y0, yf)
                if np.isfinite(estimate_tau_63(t, y, y0, yf))
                else 0.5
            )

            popt, _ = curve_fit(
                model,
                t,
                y,
                p0=[yf, initial_tau],
                bounds=(
                    [
                        min(y.min(), yf) - abs(yf - y0),
                        0.001,
                    ],
                    [
                        max(y.max(), yf) + abs(yf - y0),
                        20.0,
                    ],
                ),
                maxfev=10000,
            )

            yf_fit, tau_fit = popt

            y_pred = model(t, yf_fit, tau_fit)

            fit_r2 = r_squared(y, y_pred)
            fit_error = rmse(y, y_pred)

            # Prefer fitted final value if physically meaningful.
            if np.isfinite(yf_fit):
                yf_for_metrics = float(yf_fit)
            else:
                yf_for_metrics = yf

        else:
            yf_for_metrics = yf

    except Exception:
        tau_fit = estimate_tau_63(t, y, y0, yf)
        yf_for_metrics = yf

    if not np.isfinite(tau_fit):
        tau_fit = estimate_tau_63(t, y, y0, yf)

    rise = calculate_rise_time(
        t,
        y,
        y0,
        yf_for_metrics,
    )

    settling = calculate_settling_time(
        t,
        y,
        yf_for_metrics,
    )

    overshoot = calculate_overshoot(
        y,
        y0,
        yf_for_metrics,
    )

    return DynamicResult(
        robot=robot,
        side=side,
        transition=label,
        command=command,
        tau_s=tau_fit,
        rise_time_s=rise,
        settling_time_s=settling,
        overshoot_pct=overshoot,
        steady_rpm=yf_for_metrics,
        initial_rpm=y0,
        final_rpm=float(y[-1]),
        fit_r2=fit_r2,
        fit_rmse_rpm=fit_error,
        n_samples=len(y),
    )


def analyze_dynamic(
    df: pd.DataFrame,
    robot: int,
) -> list[DynamicResult]:

    results = []

    for phase, side, rpm_col in [
        (5, "L", "rpm_l"),
        (6, "R", "rpm_r"),
    ]:

        p = phase_dataframe(df, phase)

        segments = segment_dynamic_phase(
            p,
            side,
        )

        previous_segment = None

        for label, command, segment in segments:

            if abs(command) > 1e-6:

                result = fit_dynamic_transition(
                    segment=segment,
                    rpm_col=rpm_col,
                    command=command,
                    label=label,
                    robot=robot,
                    side=side,
                    baseline=previous_segment,
                )

                if result is not None:
                    results.append(result)

            previous_segment = segment

    return results


# =============================================================================
# ASYMMETRY
# =============================================================================

def analyze_asymmetry(
    df: pd.DataFrame,
    robot: int,
) -> pd.DataFrame:

    p = phase_dataframe(df, 7)

    if p.empty:
        return pd.DataFrame()

    phase_start = p["timestamp_us"].iloc[0]

    local_t = (
        p["timestamp_us"] - phase_start
    ) / 1e6

    records = []

    for i, level in enumerate(ASYMMETRY_LEVELS):

        forward_start = i * 5.0
        # Sequence is:
        # +level, -level, +next, -next...
        # Easier to explicitly calculate.
        forward_start = i * 5.0
        reverse_start = forward_start + 2.5

        for direction, start in [
            ("FWD", forward_start),
            ("REV", reverse_start),
        ]:

            mask = (
                (local_t >= start)
                & (local_t < start + ASYMMETRY_STEP_DURATION_S)
            )

            seg = p.loc[mask].copy()

            if seg.empty:
                continue

            # Ignore first half of each step.
            local_seg_t = (
                seg["timestamp_us"] - seg["timestamp_us"].iloc[0]
            ) / 1e6

            seg = seg.loc[
                local_seg_t >= ASYMMETRY_STEP_DURATION_S * 0.5
            ]

            if seg.empty:
                continue

            left = seg["rpm_l"].to_numpy(dtype=float)
            right = seg["rpm_r"].to_numpy(dtype=float)

            valid = np.isfinite(left) & np.isfinite(right)

            left = left[valid]
            right = right[valid]

            if len(left) == 0:
                continue

            # For simultaneous commands, compare signed RPM.
            mean_l = float(np.mean(left))
            mean_r = float(np.mean(right))

            mean_abs_l = float(np.mean(np.abs(left)))
            mean_abs_r = float(np.mean(np.abs(right)))

            mean_abs = (
                (abs(mean_l) + abs(mean_r)) / 2.0
            )

            if mean_abs > 1e-6:
                mismatch_pct = (
                    100.0
                    * abs(abs(mean_l) - abs(mean_r))
                    / mean_abs
                )
            else:
                mismatch_pct = np.nan

            records.append({
                "robot": robot,
                "level_pwm": level,
                "direction": direction,
                "rpm_l_mean": mean_l,
                "rpm_r_mean": mean_r,
                "rpm_l_std": std_or_nan(left),
                "rpm_r_std": std_or_nan(right),
                "rpm_mismatch_pct": mismatch_pct,
                "rpm_difference": mean_l - mean_r,
                "n_samples": len(left),
            })

    return pd.DataFrame(records)


# =============================================================================
# COAST-DOWN
# =============================================================================

def fit_coastdown_exponential(
    t: np.ndarray,
    rpm: np.ndarray,
) -> tuple[float, float, float]:
    """
    Effective exponential decay:

        |RPM| = A * exp(-t / tau)

    This is an empirical decay time constant.

    It must NOT be interpreted as absolute Coulomb/viscous friction
    coefficients without independent inertia J and a validated mechanical
    model.
    """

    valid = (
        np.isfinite(t)
        & np.isfinite(rpm)
        & (rpm > 0)
    )

    t = t[valid]
    rpm = rpm[valid]

    if len(t) < 10:
        return np.nan, np.nan, np.nan

    # Shift time
    t = t - t[0]

    try:

        def model(t_, A, tau):
            return A * np.exp(-t_ / tau)

        p0 = [
            max(float(rpm[0]), 1.0),
            max(
                0.05,
                float(
                    (t[-1] - t[0])
                    / max(
                        1.0,
                        np.log(max(rpm[0], 1.0) / max(rpm[-1], 1.0))
                    )
                ),
            ),
        ]

        popt, _ = curve_fit(
            model,
            t,
            rpm,
            p0=p0,
            bounds=(
                [0.0, 0.001],
                [np.inf, 1000.0],
            ),
            maxfev=10000,
        )

        A, tau = popt

        pred = model(t, A, tau)

        return (
            float(A),
            float(tau),
            r_squared(rpm, pred),
        )

    except Exception:
        return np.nan, np.nan, np.nan


def analyze_coastdown(
    df: pd.DataFrame,
    robot: int,
) -> pd.DataFrame:

    p = phase_dataframe(df, 8)

    if p.empty:
        return pd.DataFrame()

    phase_start = p["timestamp_us"].iloc[0]

    local_t = (
        p["timestamp_us"] - phase_start
    ) / 1e6

    # Firmware:
    # 0-4: drive +0.75
    # 4-8: coast
    # 8-12: drive -0.75
    # 12-16: coast
    coast_intervals = [
        ("FWD_COAST", 4.0, 8.0),
        ("REV_COAST", 12.0, 16.0),
    ]

    records = []

    for label, start, end in coast_intervals:

        mask = (local_t >= start) & (local_t < end)

        seg = p.loc[mask].copy()

        if seg.empty:
            continue

        for side, rpm_col in [
            ("L", "rpm_l"),
            ("R", "rpm_r"),
        ]:

            t = (
                seg["timestamp_us"].to_numpy(dtype=float)
                - seg["timestamp_us"].iloc[0]
            ) / 1e6

            rpm = np.abs(
                seg[rpm_col].to_numpy(dtype=float)
            )

            valid = np.isfinite(t) & np.isfinite(rpm)

            t = t[valid]
            rpm = rpm[valid]

            A, tau, fit_r2 = fit_coastdown_exponential(
                t,
                rpm,
            )

            records.append({
                "robot": robot,
                "side": side,
                "segment": label,
                "initial_abs_rpm": float(rpm[0]) if len(rpm) else np.nan,
                "final_abs_rpm": float(rpm[-1]) if len(rpm) else np.nan,
                "decay_tau_s": tau,
                "decay_A_rpm": A,
                "fit_r2": fit_r2,
                "duration_s": float(t[-1] - t[0]) if len(t) else np.nan,
                "n_samples": len(rpm),
            })

    return pd.DataFrame(records)


# =============================================================================
# ELECTRICAL ANALYSIS
# =============================================================================

def analyze_electrical(
    df: pd.DataFrame,
    robot: int,
) -> pd.DataFrame:

    records = []

    for phase in range(10):

        p = phase_dataframe(df, phase)

        if p.empty:
            continue

        for column in [
            "battery_voltage",
            "driver_voltage",
            "driver_current",
            "driver_power",
        ]:

            values = p[column].to_numpy(dtype=float)

            valid = values[np.isfinite(values)]

            if len(valid) == 0:
                continue

            records.append({
                "robot": robot,
                "phase": phase,
                "signal": column,
                "mean": float(np.mean(valid)),
                "std": std_or_nan(valid),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "p05": percentile_or_nan(valid, 5),
                "p95": percentile_or_nan(valid, 95),
                "n_samples": len(valid),
            })

    return pd.DataFrame(records)


# =============================================================================
# OVERVIEW
# =============================================================================

def create_overview_plot(
    df: pd.DataFrame,
    robot: int,
) -> None:

    fig, ax = plt.subplots(
        figsize=(12, 5),
        dpi=FIG_DPI,
    )

    t = df["time_s"]

    ax.plot(
        t,
        df["rpm_l"],
        label="Left RPM",
        linewidth=0.8,
    )

    ax.plot(
        t,
        df["rpm_r"],
        label="Right RPM",
        linewidth=0.8,
    )

    # Phase boundaries
    phase_change = df["phase"].ne(df["phase"].shift())

    boundaries = df.loc[
        phase_change,
        ["time_s", "phase"]
    ]

    for _, row in boundaries.iterrows():
        ax.axvline(
            row["time_s"],
            linestyle="--",
            linewidth=0.5,
            alpha=0.4,
        )

    ax.set_title(
        f"Robot {robot} — Ground SysID Overview"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("RPM")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"r{robot}_overview.png",
        dpi=FIG_DPI,
    )

    plt.close(fig)


# =============================================================================
# STATIC PLOT
# =============================================================================

def create_static_plot(
    static_levels: pd.DataFrame,
    robot: int,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 6),
        dpi=FIG_DPI,
    )

    d_robot = static_levels[
        static_levels["robot"] == robot
    ]

    for side in ["L", "R"]:
        for direction in ["FWD", "REV"]:

            d = d_robot[
                (d_robot["side"] == side)
                & (d_robot["direction"] == direction)
            ].copy()

            if d.empty:
                continue

            label = f"{side} {direction}"

            ax.errorbar(
                d["command_pwm_mean"].abs(),
                d["rpm_mean"].abs(),
                yerr=d["rpm_std"],
                marker="o",
                linestyle="None",
                capsize=3,
                label=label,
            )

    ax.set_title(
        f"Robot {robot} — Static PWM–RPM Characteristic"
    )

    ax.set_xlabel("|PWM command|")
    ax.set_ylabel("|RPM|")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"r{robot}_static.png",
        dpi=FIG_DPI,
    )

    plt.close(fig)


# =============================================================================
# DYNAMIC PLOT
# =============================================================================

def create_dynamic_plot(
    df: pd.DataFrame,
    robot: int,
) -> None:

    fig, ax = plt.subplots(
        figsize=(11, 6),
        dpi=FIG_DPI,
    )

    for phase, side, rpm_col in [
        (5, "L", "rpm_l"),
        (6, "R", "rpm_r"),
    ]:

        p = phase_dataframe(df, phase)

        if p.empty:
            continue

        t = (
            p["timestamp_us"] - p["timestamp_us"].iloc[0]
        ) / 1e6

        ax.plot(
            t,
            p[rpm_col],
            label=f"{side} dynamic",
            linewidth=1.0,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
    )

    ax.set_title(
        f"Robot {robot} — Dynamic Step Responses"
    )

    ax.set_xlabel("Time within dynamic phase [s]")
    ax.set_ylabel("RPM")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"r{robot}_dynamic.png",
        dpi=FIG_DPI,
    )

    plt.close(fig)


# =============================================================================
# COASTDOWN PLOT
# =============================================================================

def create_coastdown_plot(
    df: pd.DataFrame,
    robot: int,
) -> None:

    fig, ax = plt.subplots(
        figsize=(10, 6),
        dpi=FIG_DPI,
    )

    p = phase_dataframe(df, 8)

    if not p.empty:

        phase_start = p["timestamp_us"].iloc[0]

        local_t = (
            p["timestamp_us"] - phase_start
        ) / 1e6

        for side, rpm_col in [
            ("L", "rpm_l"),
            ("R", "rpm_r"),
        ]:

            for label, start, end in [
                ("FWD coast", 4.0, 8.0),
                ("REV coast", 12.0, 16.0),
            ]:

                mask = (
                    (local_t >= start)
                    & (local_t < end)
                )

                seg = p.loc[mask]

                if seg.empty:
                    continue

                t = (
                    seg["timestamp_us"]
                    - seg["timestamp_us"].iloc[0]
                ) / 1e6

                ax.plot(
                    t,
                    np.abs(seg[rpm_col]),
                    label=f"{side} {label}",
                    linewidth=1.0,
                )

    ax.set_title(
        f"Robot {robot} — Coast-Down"
    )

    ax.set_xlabel("Time from coast start [s]")
    ax.set_ylabel("|RPM|")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"r{robot}_coastdown.png",
        dpi=FIG_DPI,
    )

    plt.close(fig)


# =============================================================================
# ELECTRICAL PLOT
# =============================================================================

def create_electrical_plot(
    df: pd.DataFrame,
    robot: int,
) -> None:

    fig, ax1 = plt.subplots(
        figsize=(11, 6),
        dpi=FIG_DPI,
    )

    t = df["time_s"]

    ax1.plot(
        t,
        df["driver_current"],
        label="Driver current [mA]",
        linewidth=0.8,
    )

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Driver current [mA]")

    ax2 = ax1.twinx()

    ax2.plot(
        t,
        df["driver_power"],
        label="Driver power [mW]",
        linewidth=0.8,
        linestyle="--",
    )

    ax2.set_ylabel("Driver power [mW]")

    ax1.set_title(
        f"Robot {robot} — Electrical Telemetry"
    )

    ax1.grid(True, alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"r{robot}_electrical.png",
        dpi=FIG_DPI,
    )

    plt.close(fig)


# =============================================================================
# CROSS-ROBOT COMPARISON
# =============================================================================

def create_comparison_static_plot(
    static_results: pd.DataFrame,
) -> None:

    if static_results.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 6),
        dpi=FIG_DPI,
    )

    for robot in sorted(static_results["robot"].unique()):

        d = static_results[
            (static_results["robot"] == robot)
            & (static_results["direction"] == "FWD")
        ]

        if d.empty:
            continue

        ax.plot(
            d["side"] + " " + d["direction"],
            d["gain_rpm_per_pwm"],
            marker="o",
            linestyle="None",
            label=f"Robot {robot}",
        )

    # The x-axis is categorical and somewhat artificial here.
    # The important quantity is the gain comparison.
    ax.set_title(
        "Cross-Robot Static Gain Comparison"
    )

    ax.set_xlabel("Motor / Direction")
    ax.set_ylabel("Gain [RPM / PWM]")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "comparison_static.png",
        dpi=FIG_DPI,
    )

    plt.close(fig)


def create_comparison_dynamic_plot(
    dynamic_results: pd.DataFrame,
) -> None:

    if dynamic_results.empty:
        return

    fig, ax = plt.subplots(
        figsize=(11, 6),
        dpi=FIG_DPI,
    )

    data = dynamic_results.copy()

    for robot in sorted(data["robot"].unique()):

        d = data[data["robot"] == robot]

        ax.plot(
            range(len(d)),
            d["tau_s"],
            marker="o",
            linestyle="None",
            label=f"Robot {robot}",
        )

    ax.set_title(
        "Cross-Robot Dynamic Time Constant"
    )

    ax.set_xlabel("Dynamic transition index")
    ax.set_ylabel("τ [s]")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "comparison_dynamic.png",
        dpi=FIG_DPI,
    )

    plt.close(fig)


def create_comparison_asymmetry_plot(
    asymmetry_results: pd.DataFrame,
) -> None:

    if asymmetry_results.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 6),
        dpi=FIG_DPI,
    )

    for robot in sorted(
        asymmetry_results["robot"].unique()
    ):

        d = asymmetry_results[
            asymmetry_results["robot"] == robot
        ]

        fwd = d[d["direction"] == "FWD"]

        if fwd.empty:
            continue

        ax.plot(
            fwd["level_pwm"],
            fwd["rpm_mismatch_pct"],
            marker="o",
            label=f"Robot {robot}",
        )

    ax.set_title(
        "Cross-Robot Left/Right Speed Asymmetry"
    )

    ax.set_xlabel("|PWM|")
    ax.set_ylabel("RPM mismatch [%]")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "comparison_asymmetry.png",
        dpi=FIG_DPI,
    )

    plt.close(fig)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def format_value(value, digits=3):
    if value is None:
        return "N/A"

    try:
        if not np.isfinite(float(value)):
            return "N/A"

        return f"{float(value):.{digits}f}"

    except Exception:
        return str(value)


def generate_report(
    quality_results: list[dict],
    static_results: list[StaticResult],
    dynamic_results: list[DynamicResult],
    asymmetry_results: pd.DataFrame,
    coastdown_results: pd.DataFrame,
    electrical_results: pd.DataFrame,
    phase_warnings: dict[int, list[str]],
) -> None:

    lines = []

    lines.append("=" * 78)
    lines.append("ANJOMAN GROUND SYSID ANALYSIS")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        "Analyzer automatically discovered sysid_r*.csv "
        "in the script directory."
    )
    lines.append("")

    # -------------------------------------------------------------------------
    # DATA QUALITY
    # -------------------------------------------------------------------------

    lines.append("=" * 78)
    lines.append("1. DATA QUALITY")
    lines.append("=" * 78)

    for q in quality_results:

        robot = q["robot"]

        lines.append("")
        lines.append(f"Robot {robot}")
        lines.append("-" * 40)

        lines.append(
            f"Samples:                 {q['samples']}"
        )

        lines.append(
            f"Duration:                {format_value(q['duration_s'], 3)} s"
        )

        lines.append(
            f"Actual sample rate:      {format_value(q['sample_rate_hz'], 3)} Hz"
        )

        lines.append(
            f"Median sample period:    {format_value(q['median_dt_ms'], 3)} ms"
        )

        lines.append(
            f"Max sample period:       {format_value(q['max_dt_ms'], 3)} ms"
        )

        lines.append(
            f"Timestamp duplicates:    {q['timestamp_duplicates']}"
        )

        lines.append(
            f"Timestamp reversals:     {q['timestamp_negative_steps']}"
        )

        lines.append(
            f"Detected timestamp gaps: {q['timestamp_gaps']}"
        )

        lines.append(
            f"Estimated missing rows:  {q['estimated_missing_samples']}"
        )

        lines.append(
            f"Encoder L valid:         {format_value(q['encoder_valid_l_pct'], 2)} %"
        )

        lines.append(
            f"Encoder R valid:         {format_value(q['encoder_valid_r_pct'], 2)} %"
        )

        lines.append(
            f"IMU valid:               {format_value(q['imu_valid_pct'], 2)} %"
        )

        lines.append(
            f"INA226 valid:            {format_value(q['ina_valid_pct'], 2)} %"
        )

        if phase_warnings.get(robot):
            lines.append("")
            lines.append("Phase warnings:")

            for warning in phase_warnings[robot]:
                lines.append(f"  - {warning}")

    # -------------------------------------------------------------------------
    # STATIC
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append("=" * 78)
    lines.append("2. STATIC MOTOR IDENTIFICATION")
    lines.append("=" * 78)

    if not static_results:
        lines.append("No static results available.")

    else:

        lines.append("")
        lines.append(
            "Primary model: |RPM| = gain * max(|PWM| - deadband, 0)"
        )

        lines.append("")

        header = (
            f"{'Robot':>5} "
            f"{'Side':>5} "
            f"{'Dir':>5} "
            f"{'Deadband':>10} "
            f"{'Gain':>12} "
            f"{'R2':>8} "
            f"{'RMSE':>10} "
            f"{'MaxRPM':>10}"
        )

        lines.append(header)
        lines.append("-" * len(header))

        for r in static_results:

            lines.append(
                f"{r.robot:>5} "
                f"{r.side:>5} "
                f"{r.direction:>5} "
                f"{format_value(r.deadband_fit, 4):>10} "
                f"{format_value(r.gain_rpm_per_pwm, 2):>12} "
                f"{format_value(r.r2, 4):>8} "
                f"{format_value(r.rmse_rpm, 2):>10} "
                f"{format_value(r.max_rpm, 1):>10}"
            )

    # -------------------------------------------------------------------------
    # DYNAMIC
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append("=" * 78)
    lines.append("3. DYNAMIC IDENTIFICATION")
    lines.append("=" * 78)

    if not dynamic_results:
        lines.append("No dynamic results available.")

    else:

        lines.append("")
        lines.append(
            "First-order approximation is used where identifiable."
        )
        lines.append("")

        header = (
            f"{'Robot':>5} "
            f"{'Side':>5} "
            f"{'Transition':>15} "
            f"{'Cmd':>7} "
            f"{'Tau[s]':>9} "
            f"{'Rise[s]':>9} "
            f"{'Settle[s]':>10} "
            f"{'OS[%]':>8} "
            f"{'R2':>8}"
        )

        lines.append(header)
        lines.append("-" * len(header))

        for r in dynamic_results:

            lines.append(
                f"{r.robot:>5} "
                f"{r.side:>5} "
                f"{r.transition:>15} "
                f"{r.command:>7.2f} "
                f"{format_value(r.tau_s, 4):>9} "
                f"{format_value(r.rise_time_s, 4):>9} "
                f"{format_value(r.settling_time_s, 4):>10} "
                f"{format_value(r.overshoot_pct, 2):>8} "
                f"{format_value(r.fit_r2, 4):>8}"
            )

    # -------------------------------------------------------------------------
    # ASYMMETRY
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append("=" * 78)
    lines.append("4. LEFT/RIGHT ASYMMETRY")
    lines.append("=" * 78)

    if asymmetry_results.empty:
        lines.append("No asymmetry results available.")

    else:

        for robot in sorted(
            asymmetry_results["robot"].unique()
        ):

            d = asymmetry_results[
                asymmetry_results["robot"] == robot
            ]

            mismatch = d["rpm_mismatch_pct"].dropna()

            lines.append("")
            lines.append(f"Robot {robot}")

            if len(mismatch):
                lines.append(
                    f"  Mean mismatch:   {mismatch.mean():.3f} %"
                )

                lines.append(
                    f"  Median mismatch: {mismatch.median():.3f} %"
                )

                lines.append(
                    f"  Max mismatch:    {mismatch.max():.3f} %"
                )

                lines.append(
                    f"  P95 mismatch:    {mismatch.quantile(0.95):.3f} %"
                )

    # -------------------------------------------------------------------------
    # COASTDOWN
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append("=" * 78)
    lines.append("5. COAST-DOWN")
    lines.append("=" * 78)

    if coastdown_results.empty:
        lines.append("No coast-down results available.")

    else:

        lines.append("")
        lines.append(
            "Decay tau is an effective empirical parameter."
        )

        lines.append(
            "Absolute friction coefficients cannot be uniquely "
            "identified without independent mechanical inertia J."
        )

        for robot in sorted(
            coastdown_results["robot"].unique()
        ):

            d = coastdown_results[
                coastdown_results["robot"] == robot
            ]

            lines.append("")
            lines.append(f"Robot {robot}")

            for _, row in d.iterrows():

                lines.append(
                    f"  {row['side']} {row['segment']}: "
                    f"tau={format_value(row['decay_tau_s'], 4)} s, "
                    f"R2={format_value(row['fit_r2'], 4)}, "
                    f"RPM {format_value(row['initial_abs_rpm'], 1)}"
                    f" -> "
                    f"{format_value(row['final_abs_rpm'], 1)}"
                )

    # -------------------------------------------------------------------------
    # ELECTRICAL
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append("=" * 78)
    lines.append("6. ELECTRICAL TELEMETRY")
    lines.append("=" * 78)

    lines.append("")
    lines.append(
        "INA226 values are shared driver/bus measurements, "
        "not individual motor measurements."
    )

    if electrical_results.empty:
        lines.append("No electrical results available.")

    else:

        for robot in sorted(
            electrical_results["robot"].unique()
        ):

            lines.append("")
            lines.append(f"Robot {robot}")

            d = electrical_results[
                electrical_results["robot"] == robot
            ]

            for signal in [
                "battery_voltage",
                "driver_voltage",
                "driver_current",
                "driver_power",
            ]:

                s = d[d["signal"] == signal]

                if s.empty:
                    continue

                lines.append(
                    f"  {signal}: "
                    f"overall range "
                    f"{format_value(s['min'].min(), 2)}"
                    f" .. "
                    f"{format_value(s['max'].max(), 2)}"
                )

    # -------------------------------------------------------------------------
    # FINAL INTERPRETATION
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append("=" * 78)
    lines.append("7. INTERPRETATION NOTES")
    lines.append("=" * 78)

    lines.append("")
    lines.append(
        "1. Static identification uses steady-state windows from the "
        "14-level PWM sweeps."
    )

    lines.append(
        "2. cmd_v is NOT treated as measured robot velocity."
    )

    lines.append(
        "3. RPM is treated as the primary propulsion output."
    )

    lines.append(
        "4. Dynamic tau values describe the observed motor/drive response "
        "under this experiment."
    )

    lines.append(
        "5. Coast-down results do not by themselves provide unique "
        "Coulomb/viscous friction coefficients."
    )

    lines.append(
        "6. Driver current/power are bus-level quantities because the "
        "instrumentation is shared."
    )

    lines.append(
        "7. The analyzer does not claim parameter identifiability where "
        "the recorded data do not support it."
    )

    lines.append("")

    report_path = OUTPUT_DIR / "report.txt"

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# RESULTS TABLES
# =============================================================================

def save_results_tables(
    quality_results: list[dict],
    static_results: list[StaticResult],
    dynamic_results: list[DynamicResult],
    asymmetry_results: pd.DataFrame,
    coastdown_results: pd.DataFrame,
    electrical_results: pd.DataFrame,
) -> None:

    # -------------------------------------------------------------------------
    # Main summary results.csv
    # -------------------------------------------------------------------------

    rows = []

    for q in quality_results:

        for metric, value in q.items():

            if metric == "robot":
                continue

            rows.append({
                "category": "quality",
                "robot": q["robot"],
                "side": "",
                "direction": "",
                "metric": metric,
                "value": value,
            })

    for r in static_results:

        base = {
            "category": "static",
            "robot": r.robot,
            "side": r.side,
            "direction": r.direction,
        }

        for metric, value in {
            "deadband_fit": r.deadband_fit,
            "gain_rpm_per_pwm": r.gain_rpm_per_pwm,
            "intercept_rpm": r.intercept_rpm,
            "r2": r.r2,
            "rmse_rpm": r.rmse_rpm,
            "n_levels": r.n_levels,
            "first_active_pwm": r.first_active_pwm,
            "max_rpm": r.max_rpm,
        }.items():

            rows.append({
                **base,
                "metric": metric,
                "value": value,
            })

    for r in dynamic_results:

        base = {
            "category": "dynamic",
            "robot": r.robot,
            "side": r.side,
            "direction": "",
        }

        for metric, value in {
            "command": r.command,
            "tau_s": r.tau_s,
            "rise_time_s": r.rise_time_s,
            "settling_time_s": r.settling_time_s,
            "overshoot_pct": r.overshoot_pct,
            "steady_rpm": r.steady_rpm,
            "initial_rpm": r.initial_rpm,
            "final_rpm": r.final_rpm,
            "fit_r2": r.fit_r2,
            "fit_rmse_rpm": r.fit_rmse_rpm,
            "n_samples": r.n_samples,
        }.items():

            rows.append({
                **base,
                "metric": f"{r.transition}:{metric}",
                "value": value,
            })

    summary_df = pd.DataFrame(rows)

    summary_df.to_csv(
        OUTPUT_DIR / "results.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Static level table
    # -------------------------------------------------------------------------

    if static_results is not None:
        pass

    # static_levels is saved by main()
    # dynamic_steps is saved by main()


# =============================================================================
# CROSS-ROBOT SUMMARY
# =============================================================================

def create_cross_robot_summary(
    static_results: list[StaticResult],
    dynamic_results: list[DynamicResult],
    asymmetry_results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    # Static
    for r in static_results:

        rows.append({
            "robot": r.robot,
            "category": "static",
            "side": r.side,
            "direction": r.direction,
            "parameter": "deadband",
            "value": r.deadband_fit,
        })

        rows.append({
            "robot": r.robot,
            "category": "static",
            "side": r.side,
            "direction": r.direction,
            "parameter": "gain_rpm_per_pwm",
            "value": r.gain_rpm_per_pwm,
        })

        rows.append({
            "robot": r.robot,
            "category": "static",
            "side": r.side,
            "direction": r.direction,
            "parameter": "r2",
            "value": r.r2,
        })

    # Dynamic
    for r in dynamic_results:

        rows.append({
            "robot": r.robot,
            "category": "dynamic",
            "side": r.side,
            "direction": "",
            "parameter": f"{r.transition}:tau_s",
            "value": r.tau_s,
        })

    # Asymmetry
    if not asymmetry_results.empty:

        for robot in sorted(
            asymmetry_results["robot"].unique()
        ):

            d = asymmetry_results[
                asymmetry_results["robot"] == robot
            ]

            mismatch = d["rpm_mismatch_pct"].dropna()

            if len(mismatch):

                rows.append({
                    "robot": robot,
                    "category": "asymmetry",
                    "side": "LR",
                    "direction": "",
                    "parameter": "mean_mismatch_pct",
                    "value": mismatch.mean(),
                })

                rows.append({
                    "robot": robot,
                    "category": "asymmetry",
                    "side": "LR",
                    "direction": "",
                    "parameter": "p95_mismatch_pct",
                    "value": mismatch.quantile(0.95),
                })

    return pd.DataFrame(rows)


# =============================================================================
# TERMINAL SUMMARY
# =============================================================================

def print_terminal_summary(
    quality_results: list[dict],
    static_results: list[StaticResult],
    dynamic_results: list[DynamicResult],
    asymmetry_results: pd.DataFrame,
) -> None:

    print("")
    print("=" * 78)
    print("ANJOMAN GROUND SYSID ANALYSIS")
    print("=" * 78)

    print("")
    print(
        f"Input directory : {SCRIPT_DIR}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print("")

    print("DATA QUALITY")
    print("-" * 78)

    for q in quality_results:

        print(
            f"R{q['robot']}: "
            f"{q['samples']} samples, "
            f"{q['duration_s']:.2f}s, "
            f"{q['sample_rate_hz']:.2f}Hz, "
            f"gaps={q['timestamp_gaps']}, "
            f"missing≈{q['estimated_missing_samples']}"
        )

    print("")

    print("STATIC IDENTIFICATION")
    print("-" * 78)

    for r in static_results:

        print(
            f"R{r.robot} {r.side} {r.direction}: "
            f"deadband={format_value(r.deadband_fit, 4)}, "
            f"gain={format_value(r.gain_rpm_per_pwm, 2)} RPM/PWM, "
            f"R²={format_value(r.r2, 4)}"
        )

    print("")

    print("DYNAMIC IDENTIFICATION")
    print("-" * 78)

    for r in dynamic_results:

        print(
            f"R{r.robot} {r.side} {r.transition}: "
            f"tau={format_value(r.tau_s, 4)}s, "
            f"rise={format_value(r.rise_time_s, 4)}s, "
            f"settle={format_value(r.settling_time_s, 4)}s, "
            f"OS={format_value(r.overshoot_pct, 2)}%"
        )

    print("")

    print("ASYMMETRY")
    print("-" * 78)

    if asymmetry_results.empty:

        print("No asymmetry data.")

    else:

        for robot in sorted(
            asymmetry_results["robot"].unique()
        ):

            d = asymmetry_results[
                asymmetry_results["robot"] == robot
            ]

            mismatch = d["rpm_mismatch_pct"].dropna()

            if len(mismatch):

                print(
                    f"R{robot}: "
                    f"mean={mismatch.mean():.3f}%, "
                    f"median={mismatch.median():.3f}%, "
                    f"P95={mismatch.quantile(0.95):.3f}%, "
                    f"max={mismatch.max():.3f}%"
                )

    print("")
    print("=" * 78)
    print("Analysis complete.")
    print(f"See: {OUTPUT_DIR}")
    print("=" * 78)
    print("")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    ensure_output_dir()
    clear_generated_outputs()

    csv_files = discover_csv_files()

    if not csv_files:
        print("")
        print("ERROR: No sysid_r*.csv files found.")
        print("")
        print(f"Expected files in:")
        print(f"  {SCRIPT_DIR}")
        print("")
        print("Example:")
        print("  sysid_r1.csv")
        print("  sysid_r2.csv")
        print("  sysid_r3.csv")
        print("  sysid_r4.csv")
        print("")
        return

    print("")
    print(
        f"Found {len(csv_files)} SysID file(s):"
    )

    for path in csv_files:
        print(f"  - {path.name}")

    print("")

    quality_results = []
    phase_warnings = {}

    static_level_tables = []
    static_results = []

    dynamic_results = []

    asymmetry_tables = []

    coastdown_tables = []
    electrical_tables = []

    loaded_data = {}

    # -------------------------------------------------------------------------
    # Load + analyze every robot
    # -------------------------------------------------------------------------

    for path in csv_files:

        robot = extract_robot_id(path)

        if robot is None:
            continue

        try:
            df = load_robot_csv(path)

        except Exception as exc:

            print(
                f"ERROR loading {path.name}: {exc}"
            )

            continue

        loaded_data[robot] = df

        # Quality
        q = analyze_data_quality(
            df,
            robot,
        )

        quality_results.append(q)

        # Phase validation
        phase_warnings[robot] = validate_phases(df)

        # Static
        static_levels = collect_static_levels(
            df,
            robot,
        )

        if not static_levels.empty:
            static_level_tables.append(static_levels)

            static_results.extend(
                analyze_static(
                    static_levels,
                    robot,
                )
            )

        # Dynamic
        dynamic_results.extend(
            analyze_dynamic(
                df,
                robot,
            )
        )

        # Asymmetry
        asymmetry = analyze_asymmetry(
            df,
            robot,
        )

        if not asymmetry.empty:
            asymmetry_tables.append(asymmetry)

        # Coastdown
        coast = analyze_coastdown(
            df,
            robot,
        )

        if not coast.empty:
            coastdown_tables.append(coast)

        # Electrical
        electrical = analyze_electrical(
            df,
            robot,
        )

        if not electrical.empty:
            electrical_tables.append(electrical)

        # Plots
        create_overview_plot(
            df,
            robot,
        )

        create_static_plot(
            static_levels,
            robot,
        )

        create_dynamic_plot(
            df,
            robot,
        )

        create_coastdown_plot(
            df,
            robot,
        )

        create_electrical_plot(
            df,
            robot,
        )

    # -------------------------------------------------------------------------
    # Combine tables
    # -------------------------------------------------------------------------

    if static_level_tables:
        all_static_levels = pd.concat(
            static_level_tables,
            ignore_index=True,
        )
    else:
        all_static_levels = pd.DataFrame()

    if asymmetry_tables:
        all_asymmetry = pd.concat(
            asymmetry_tables,
            ignore_index=True,
        )
    else:
        all_asymmetry = pd.DataFrame()

    if coastdown_tables:
        all_coastdown = pd.concat(
            coastdown_tables,
            ignore_index=True,
        )
    else:
        all_coastdown = pd.DataFrame()

    if electrical_tables:
        all_electrical = pd.concat(
            electrical_tables,
            ignore_index=True,
        )
    else:
        all_electrical = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Save detailed tables
    # -------------------------------------------------------------------------

    if not all_static_levels.empty:

        all_static_levels.to_csv(
            OUTPUT_DIR / "static_levels.csv",
            index=False,
        )

    else:

        pd.DataFrame().to_csv(
            OUTPUT_DIR / "static_levels.csv",
            index=False,
        )

    dynamic_rows = []

    for r in dynamic_results:

        dynamic_rows.append({
            "robot": r.robot,
            "side": r.side,
            "transition": r.transition,
            "command": r.command,
            "tau_s": r.tau_s,
            "rise_time_s": r.rise_time_s,
            "settling_time_s": r.settling_time_s,
            "overshoot_pct": r.overshoot_pct,
            "steady_rpm": r.steady_rpm,
            "initial_rpm": r.initial_rpm,
            "final_rpm": r.final_rpm,
            "fit_r2": r.fit_r2,
            "fit_rmse_rpm": r.fit_rmse_rpm,
            "n_samples": r.n_samples,
        })

    pd.DataFrame(dynamic_rows).to_csv(
        OUTPUT_DIR / "dynamic_steps.csv",
        index=False,
    )

    if not all_asymmetry.empty:

        all_asymmetry.to_csv(
            OUTPUT_DIR / "asymmetry.csv",
            index=False,
        )

    else:

        pd.DataFrame().to_csv(
            OUTPUT_DIR / "asymmetry.csv",
            index=False,
        )

    if not all_coastdown.empty:

        all_coastdown.to_csv(
            OUTPUT_DIR / "coastdown.csv",
            index=False,
        )

    else:

        pd.DataFrame().to_csv(
            OUTPUT_DIR / "coastdown.csv",
            index=False,
        )

    if not all_electrical.empty:

        all_electrical.to_csv(
            OUTPUT_DIR / "electrical.csv",
            index=False,
        )

    else:

        pd.DataFrame().to_csv(
            OUTPUT_DIR / "electrical.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # Main summary
    # -------------------------------------------------------------------------

    save_results_tables(
        quality_results=quality_results,
        static_results=static_results,
        dynamic_results=dynamic_results,
        asymmetry_results=all_asymmetry,
        coastdown_results=all_coastdown,
        electrical_results=all_electrical,
    )

    # -------------------------------------------------------------------------
    # Cross-robot data
    # -------------------------------------------------------------------------

    cross_robot = create_cross_robot_summary(
        static_results,
        dynamic_results,
        all_asymmetry,
    )

    cross_robot.to_csv(
        OUTPUT_DIR / "cross_robot_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Cross-robot plots
    # -------------------------------------------------------------------------

    static_result_df = pd.DataFrame([
        {
            "robot": r.robot,
            "side": r.side,
            "direction": r.direction,
            "deadband_fit": r.deadband_fit,
            "gain_rpm_per_pwm": r.gain_rpm_per_pwm,
            "r2": r.r2,
        }
        for r in static_results
    ])

    dynamic_result_df = pd.DataFrame([
        {
            "robot": r.robot,
            "side": r.side,
            "transition": r.transition,
            "tau_s": r.tau_s,
        }
        for r in dynamic_results
    ])

    create_comparison_static_plot(
        static_result_df,
    )

    create_comparison_dynamic_plot(
        dynamic_result_df,
    )

    create_comparison_asymmetry_plot(
        all_asymmetry,
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    generate_report(
        quality_results=quality_results,
        static_results=static_results,
        dynamic_results=dynamic_results,
        asymmetry_results=all_asymmetry,
        coastdown_results=all_coastdown,
        electrical_results=all_electrical,
        phase_warnings=phase_warnings,
    )

    # -------------------------------------------------------------------------
    # Terminal
    # -------------------------------------------------------------------------

    print_terminal_summary(
        quality_results=quality_results,
        static_results=static_results,
        dynamic_results=dynamic_results,
        asymmetry_results=all_asymmetry,
    )


if __name__ == "__main__":
    main()
