#!/usr/bin/env python3

import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

GROUND_TRUTH_M = 3.0

TIMESTAMP_BITS = 40
TIMESTAMP_MODULO = 1 << TIMESTAMP_BITS

OUTPUT_DIR = "plots_static_3m"

# Robust outlier threshold.
# Points farther than this many MADs from the median are marked
# as statistical outliers.
MAD_THRESHOLD = 6.0

# Window size for local drift analysis.
WINDOW_SECONDS = 60.0


# ============================================================
# Utility functions
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def robust_mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    median = np.median(x)
    return np.median(np.abs(x - median))


def robust_sigma(x):
    """
    Convert MAD to approximately equivalent Gaussian sigma.
    """
    mad = robust_mad(x)

    if not np.isfinite(mad):
        return np.nan

    return 1.4826 * mad


def linear_regression(x, y):
    """
    y = slope*x + intercept

    Returns:
        slope
        intercept
        r2
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, np.nan, np.nan

    slope, intercept = np.polyfit(x, y, 1)

    y_hat = slope * x + intercept

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return slope, intercept, r2


def correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return np.corrcoef(x, y)[0, 1]


def percentile_string(x):
    p = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])

    return (
        f"P01={p[0]:.3f}, "
        f"P05={p[1]:.3f}, "
        f"P25={p[2]:.3f}, "
        f"P50={p[3]:.3f}, "
        f"P75={p[4]:.3f}, "
        f"P95={p[5]:.3f}, "
        f"P99={p[6]:.3f}"
    )


# ============================================================
# 40-bit DW1000 timestamp handling
# ============================================================

def timestamp_delta(a, b):
    """
    Compute b-a for 40-bit hardware timestamps,
    correctly handling rollover.
    """

    return (int(b) - int(a)) % TIMESTAMP_MODULO


def timestamp_delta_array(a, b):
    """
    Vectorized version.
    """

    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)

    return (b - a) % TIMESTAMP_MODULO


# ============================================================
# Load data
# ============================================================

def load_data(filename):

    df = pd.read_csv(filename)

    required = [
        "timestamp",
        "sequence",
        "range_raw_m",
        "range_calibrated_m",
        "rssi_dbm",
        "fpp_dbm",
        "cfo_ppm",
        "tTx1",
        "tRx1",
        "tRx2",
        "tTx2",
        "temp_uwb_1",
        "temp_uwb_2",
        "temp_esp_1",
        "temp_esp_2",
        "response_received",
        "status",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing columns:\n" +
            "\n".join(missing)
        )

    return df


# ============================================================
# Data cleaning
# ============================================================

def clean_data(df):

    numeric_columns = [
        "timestamp",
        "sequence",
        "range_raw_m",
        "range_calibrated_m",
        "rssi_dbm",
        "fpp_dbm",
        "cfo_ppm",
        "tTx1",
        "tRx1",
        "tRx2",
        "tTx2",
        "temp_uwb_1",
        "temp_uwb_2",
        "temp_esp_1",
        "temp_esp_2",
        "response_received",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove rows without valid raw range
    df = df[
        np.isfinite(df["range_raw_m"]) &
        np.isfinite(df["timestamp"])
    ].copy()

    return df


# ============================================================
# Timestamp-derived parameters
# ============================================================

def calculate_timestamp_parameters(df):

    # These are hardware timestamp intervals.
    #
    # T_round = tRx2 - tTx1
    # T_reply = tTx2 - tRx1

    df["t_round_ticks"] = timestamp_delta_array(
        df["tTx1"].values,
        df["tRx2"].values
    )

    df["t_reply_ticks"] = timestamp_delta_array(
        df["tRx1"].values,
        df["tTx2"].values
    )

    # Also useful:
    df["tx_to_rx1_ticks"] = timestamp_delta_array(
        df["tTx1"].values,
        df["tRx1"].values
    )

    df["rx2_to_tx2_ticks"] = timestamp_delta_array(
        df["tRx2"].values,
        df["tTx2"].values
    )

    return df


# ============================================================
# Response / validity analysis
# ============================================================

def analyze_reliability(df):

    total = len(df)

    response_received = (
        df["response_received"]
        .fillna(0)
        .astype(int)
    )

    success = int((response_received != 0).sum())
    failures = total - success

    success_rate = (
        100.0 * success / total
        if total > 0 else np.nan
    )

    print("\n")
    print("=" * 80)
    print("RELIABILITY")
    print("-" * 80)

    print(f"Total records       : {total}")
    print(f"Responses received  : {success}")
    print(f"Missing responses   : {failures}")
    print(f"Success rate        : {success_rate:.4f} %")

    print("\nSTATUS COUNTS")
    print("-" * 80)

    print(df["status"].value_counts(dropna=False).to_string())

    return {
        "total": total,
        "success": success,
        "failures": failures,
        "success_rate": success_rate,
    }


# ============================================================
# Range statistics
# ============================================================

def analyze_range(df):

    raw = df["range_raw_m"].values

    raw = raw[np.isfinite(raw)]

    error = raw - GROUND_TRUTH_M

    mean = np.mean(raw)
    median = np.median(raw)

    std = np.std(raw, ddof=1)

    mad = robust_mad(raw)
    robust_sigma_value = robust_sigma(raw)

    bias = np.mean(error)

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    minimum = np.min(raw)
    maximum = np.max(raw)

    p = np.percentile(error, [1, 5, 25, 50, 75, 95, 99])

    # Global drift
    t = df["timestamp"].values.astype(float)

    t = t - t[0]

    slope, intercept, r2 = linear_regression(
        t,
        raw
    )

    drift_mm_s = slope * 1000.0
    drift_mm_min = drift_mm_s * 60.0

    # Outlier detection using MAD
    median_raw = np.median(raw)

    if mad > 0:
        outlier_mask = (
            np.abs(raw - median_raw)
            > MAD_THRESHOLD * mad
        )
    else:
        outlier_mask = np.zeros(len(raw), dtype=bool)

    outlier_count = np.sum(outlier_mask)

    print("\n")
    print("=" * 80)
    print("RAW RANGE METROLOGY")
    print("-" * 80)

    print(f"Ground truth        : {GROUND_TRUTH_M:.3f} m")
    print(f"Samples             : {len(raw)}")

    print(f"Mean                : {mean:.6f} m")
    print(f"Median              : {median:.6f} m")

    print(f"Std                 : {std:.6f} m")
    print(f"Std                 : {std * 100:.3f} cm")

    print(f"MAD                 : {mad:.6f} m")
    print(f"Robust sigma        : {robust_sigma_value:.6f} m")

    print(f"Bias                : {bias:+.6f} m")
    print(f"MAE                 : {mae:.6f} m")
    print(f"RMSE                : {rmse:.6f} m")

    print(f"Min / Max           : {minimum:.6f} / {maximum:.6f} m")

    print("\nERROR PERCENTILES")
    print("-" * 80)

    print(f"P01                 : {p[0] * 1000:+.3f} mm")
    print(f"P05                 : {p[1] * 1000:+.3f} mm")
    print(f"P25                 : {p[2] * 1000:+.3f} mm")
    print(f"P50                 : {p[3] * 1000:+.3f} mm")
    print(f"P75                 : {p[4] * 1000:+.3f} mm")
    print(f"P95                 : {p[5] * 1000:+.3f} mm")
    print(f"P99                 : {p[6] * 1000:+.3f} mm")

    print("\nDRIFT")
    print("-" * 80)

    print(f"Drift rate          : {drift_mm_s:+.6f} mm/s")
    print(f"Drift rate          : {drift_mm_min:+.6f} mm/min")
    print(f"Drift R^2           : {r2:.6f}")

    print("\nOUTLIERS")
    print("-" * 80)

    print(f"MAD threshold       : {MAD_THRESHOLD:.1f}")
    print(f"Outliers            : {outlier_count}")
    print(
        f"Outlier rate        : "
        f"{100.0 * outlier_count / len(raw):.4f} %"
    )

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "mad": mad,
        "robust_sigma": robust_sigma_value,
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "drift_mm_s": drift_mm_s,
        "drift_mm_min": drift_mm_min,
        "drift_r2": r2,
        "outlier_count": outlier_count,
    }


# ============================================================
# Windowed drift
# ============================================================

def analyze_windowed_drift(df):

    t = (
        df["timestamp"].values.astype(float)
        - float(df["timestamp"].iloc[0])
    )

    range_values = df["range_raw"].values \
        if "range_raw" in df else df["range_raw_m"].values

    duration = np.max(t)

    starts = np.arange(
        0,
        duration,
        WINDOW_SECONDS
    )

    rows = []

    for start in starts:

        end = start + WINDOW_SECONDS

        mask = (
            (t >= start) &
            (t < end)
        )

        if np.sum(mask) < 10:
            continue

        local_t = t[mask]
        local_r = range_values[mask]

        slope, intercept, r2 = linear_regression(
            local_t,
            local_r
        )

        rows.append({
            "start_s": start,
            "end_s": end,
            "samples": np.sum(mask),
            "mean_m": np.mean(local_r),
            "std_m": np.std(local_r, ddof=1),
            "drift_mm_s": slope * 1000.0,
            "drift_r2": r2,
        })

    result = pd.DataFrame(rows)

    if len(result) > 0:

        result.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "windowed_drift.csv"
            ),
            index=False
        )

    return result


# ============================================================
# Correlation analysis
# ============================================================

def analyze_correlations(df):

    range_error = (
        df["range_raw_m"] - GROUND_TRUTH_M
    )

    variables = [
        "rssi_dbm",
        "fpp_dbm",
        "cfo_ppm",
        "t_round_ticks",
        "t_reply_ticks",
        "tx_to_rx1_ticks",
        "rx2_to_tx2_ticks",
        "temp_uwb_1",
        "temp_uwb_2",
        "temp_esp_1",
        "temp_esp_2",
    ]

    print("\n")
    print("=" * 80)
    print("CORRELATION WITH RAW RANGE ERROR")
    print("-" * 80)

    rows = []

    for variable in variables:

        if variable not in df.columns:
            continue

        c = correlation(
            range_error,
            df[variable]
        )

        rows.append({
            "variable": variable,
            "correlation": c,
        })

        print(
            f"{variable:25s}: {c:+.6f}"
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "correlations.csv"
        ),
        index=False
    )

    return result


# ============================================================
# Temperature analysis
# ============================================================

def analyze_temperature(df):

    print("\n")
    print("=" * 80)
    print("TEMPERATURE")
    print("-" * 80)

    columns = [
        "temp_uwb_1",
        "temp_uwb_2",
        "temp_esp_1",
        "temp_esp_2",
    ]

    for col in columns:

        x = df[col].dropna().values

        if len(x) == 0:
            continue

        print(f"\n{col}")

        print(f"  Mean       : {np.mean(x):.3f} C")
        print(f"  Std        : {np.std(x, ddof=1):.3f} C")
        print(f"  Min / Max  : {np.min(x):.3f} / {np.max(x):.3f} C")

        corr = correlation(
            df["range_raw_m"] - GROUND_TRUTH_M,
            df[col]
        )

        slope, intercept, r2 = linear_regression(
            df[col],
            df["range_raw_m"]
        )

        print(f"  Correlation: {corr:+.6f}")

        if np.isfinite(slope):
            print(
                f"  Range/temp : "
                f"{slope * 1000:+.3f} mm/C"
            )

            print(
                f"  Regression R^2: {r2:.6f}"
            )


# ============================================================
# RF analysis
# ============================================================

def analyze_rf(df):

    print("\n")
    print("=" * 80)
    print("RF / CLOCK PARAMETERS")
    print("-" * 80)

    columns = [
        "rssi_dbm",
        "fpp_dbm",
        "cfo_ppm",
        "t_round_ticks",
        "t_reply_ticks",
    ]

    for col in columns:

        if col not in df.columns:
            continue

        x = df[col].dropna().values

        if len(x) == 0:
            continue

        print(f"\n{col}")

        print(f"  Mean       : {np.mean(x):.6f}")
        print(f"  Std        : {np.std(x, ddof=1):.6f}")
        print(f"  Min / Max  : {np.min(x):.6f} / {np.max(x):.6f}")


# ============================================================
# Sequence analysis
# ============================================================

def analyze_sequence(df):

    seq = df["sequence"].values.astype(np.int64)

    if len(seq) < 2:
        return

    dseq = np.diff(seq)

    gaps = dseq[dseq != 1]

    print("\n")
    print("=" * 80)
    print("SEQUENCE CONTINUITY")
    print("-" * 80)

    print(f"Samples             : {len(seq)}")
    print(f"Sequence gaps       : {len(gaps)}")

    if len(gaps) > 0:

        print(
            f"Min sequence jump  : {np.min(gaps)}"
        )

        print(
            f"Max sequence jump  : {np.max(gaps)}"
        )


# ============================================================
# Timestamp analysis
# ============================================================

def analyze_timestamps(df):

    print("\n")
    print("=" * 80)
    print("DW1000 TIMESTAMP-DERIVED PARAMETERS")
    print("-" * 80)

    columns = [
        "t_round_ticks",
        "t_reply_ticks",
        "tx_to_rx1_ticks",
        "rx2_to_tx2_ticks",
    ]

    for col in columns:

        x = df[col].dropna().values

        if len(x) == 0:
            continue

        print(f"\n{col}")

        print(
            f"  Mean       : {np.mean(x):.3f} ticks"
        )

        print(
            f"  Std        : {np.std(x, ddof=1):.3f} ticks"
        )

        print(
            f"  Min / Max  : "
            f"{np.min(x):.3f} / {np.max(x):.3f}"
        )


# ============================================================
# Plot helpers
# ============================================================

def savefig(filename):

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=180
    )

    plt.close()


# ============================================================
# Plot 1: raw range trajectory
# ============================================================

def plot_range(df):

    t = (
        df["timestamp"].values
        - df["timestamp"].iloc[0]
    ) / 1000.0

    plt.figure(figsize=(16, 6))

    plt.plot(
        t,
        df["range_raw_m"].values,
        linewidth=0.7
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Raw Range (m)")
    plt.title("UWB Raw Range — Static 3 m Experiment")

    savefig("01_raw_range_time.png")


# ============================================================
# Plot 2: range error
# ============================================================

def plot_error(df):

    t = (
        df["timestamp"].values
        - df["timestamp"].iloc[0]
    ) / 1000.0

    error_mm = (
        df["range_raw_m"].values
        - GROUND_TRUTH_M
    ) * 1000.0

    plt.figure(figsize=(16, 6))

    plt.plot(
        t,
        error_mm,
        linewidth=0.7
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Range Error (mm)")
    plt.title("UWB Range Error — Static 3 m Experiment")

    savefig("02_range_error_time.png")


# ============================================================
# Plot 3: temperature
# ============================================================

def plot_temperature(df):

    t = (
        df["timestamp"].values
        - df["timestamp"].iloc[0]
    ) / 1000.0

    columns = [
        "temp_uwb_1",
        "temp_uwb_2",
        "temp_esp_1",
        "temp_esp_2",
    ]

    plt.figure(figsize=(16, 6))

    for col in columns:

        if col in df.columns:

            plt.plot(
                t,
                df[col],
                linewidth=0.8,
                label=col
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (C)")
    plt.title("UWB / ESP32 Temperature")

    plt.legend()

    savefig("03_temperature_time.png")


# ============================================================
# Plot 4: RSSI
# ============================================================

def plot_rssi(df):

    t = (
        df["timestamp"].values
        - df["timestamp"].iloc[0]
    ) / 1000.0

    plt.figure(figsize=(16, 6))

    plt.plot(
        t,
        df["rssi_dbm"],
        linewidth=0.7
    )

    plt.xlabel("Time (s)")
    plt.ylabel("RSSI (dBm)")
    plt.title("RSSI Over Time")

    savefig("04_rssi_time.png")


# ============================================================
# Plot 5: FPP
# ============================================================

def plot_fpp(df):

    t = (
        df["timestamp"].values
        - df["timestamp"].iloc[0]
    ) / 1000.0

    plt.figure(figsize=(16, 6))

    plt.plot(
        t,
        df["fpp_dbm"],
        linewidth=0.7
    )

    plt.xlabel("Time (s)")
    plt.ylabel("FPP (dBm)")
    plt.title("First Path Power Over Time")

    savefig("05_fpp_time.png")


# ============================================================
# Plot 6: CFO
# ============================================================

def plot_cfo(df):

    t = (
        df["timestamp"].values
        - df["timestamp"].iloc[0]
    ) / 1000.0

    plt.figure(figsize=(16, 6))

    plt.plot(
        t,
        df["cfo_ppm"],
        linewidth=0.7
    )

    plt.xlabel("Time (s)")
    plt.ylabel("CFO (ppm)")
    plt.title("Carrier Frequency Offset Over Time")

    savefig("06_cfo_time.png")


# ============================================================
# Plot 7: timestamp parameters
# ============================================================

def plot_timestamp_parameters(df):

    t = (
        df["timestamp"].values
        - df["timestamp"].iloc[0]
    ) / 1000.0

    plt.figure(figsize=(16, 6))

    plt.plot(
        t,
        df["t_round_ticks"],
        linewidth=0.7,
        label="T_round"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Ticks")
    plt.title("DW1000 T_round")

    plt.legend()

    savefig("07_t_round.png")

    plt.figure(figsize=(16, 6))

    plt.plot(
        t,
        df["t_reply_ticks"],
        linewidth=0.7,
        label="T_reply"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Ticks")
    plt.title("DW1000 T_reply")

    plt.legend()

    savefig("08_t_reply.png")


# ============================================================
# Plot 8: range vs temperature
# ============================================================

def plot_temperature_relationship(df):

    error_mm = (
        df["range_raw_m"] - GROUND_TRUTH_M
    ) * 1000.0

    for col in [
        "temp_uwb_1",
        "temp_uwb_2",
        "temp_esp_1",
        "temp_esp_2",
    ]:

        plt.figure(figsize=(8, 6))

        plt.scatter(
            df[col],
            error_mm,
            s=3
        )

        plt.xlabel(col)
        plt.ylabel("Range Error (mm)")
        plt.title(
            f"Range Error vs {col}"
        )

        savefig(
            f"09_error_vs_{col}.png"
        )


# ============================================================
# Plot 9: range vs RF parameters
# ============================================================

def plot_relationship(df, variable, filename):

    error_mm = (
        df["range_raw_m"] - GROUND_TRUTH_M
    ) * 1000.0

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df[variable],
        error_mm,
        s=3
    )

    plt.xlabel(variable)
    plt.ylabel("Range Error (mm)")
    plt.title(
        f"Range Error vs {variable}"
    )

    savefig(filename)


# ============================================================
# Plot 10: distribution
# ============================================================

def plot_distribution(df):

    error_mm = (
        df["range_raw_m"] - GROUND_TRUTH_M
    ) * 1000.0

    plt.figure(figsize=(10, 6))

    plt.hist(
        error_mm,
        bins=100
    )

    plt.xlabel("Range Error (mm)")
    plt.ylabel("Samples")
    plt.title("Distribution of UWB Range Error")

    savefig("10_error_distribution.png")


# ============================================================
# Plot 11: windowed mean/std
# ============================================================

def plot_windowed_drift(windowed):

    if windowed is None or len(windowed) == 0:
        return

    t = (
        windowed["start_s"].values
        + WINDOW_SECONDS / 2.0
    )

    plt.figure(figsize=(16, 6))

    plt.plot(
        t,
        windowed["mean_m"].values,
        marker="."
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Window Mean Range (m)")
    plt.title(
        "60-second Windowed Mean Range"
    )

    savefig("11_windowed_mean.png")

    plt.figure(figsize=(16, 6))

    plt.plot(
        t,
        windowed["std_m"].values * 1000.0,
        marker="."
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Window Std (mm)")
    plt.title(
        "60-second Windowed Noise"
    )

    savefig("12_windowed_noise.png")

    plt.figure(figsize=(16, 6))

    plt.plot(
        t,
        windowed["drift_mm_s"].values,
        marker="."
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Local Drift (mm/s)")
    plt.title(
        "60-second Local Drift"
    )

    savefig("13_windowed_drift.png")


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python3 "
            "uwb_static_metrology_analysis.py "
            "<file.txt>"
        )

        sys.exit(1)

    filename = sys.argv[1]

    ensure_output_dir()

    print("\n")
    print("=" * 80)
    print("UWB STATIC METROLOGY ANALYSIS")
    print("=" * 80)

    print(f"Input file          : {filename}")
    print(f"Ground truth        : {GROUND_TRUTH_M:.3f} m")
    print(f"Output directory    : {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data(filename)

    df = clean_data(df)

    print(f"Valid rows          : {len(df)}")

    if len(df) < 10:

        print(
            "ERROR: insufficient valid samples."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Timestamp processing
    # --------------------------------------------------------

    df = calculate_timestamp_parameters(df)

    # --------------------------------------------------------
    # Analyses
    # --------------------------------------------------------

    reliability = analyze_reliability(df)

    range_stats = analyze_range(df)

    analyze_sequence(df)

    analyze_timestamps(df)

    analyze_temperature(df)

    analyze_rf(df)

    correlations = analyze_correlations(df)

    windowed = analyze_windowed_drift(df)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plot_range(df)

    plot_error(df)

    plot_temperature(df)

    plot_rssi(df)

    plot_fpp(df)

    plot_cfo(df)

    plot_timestamp_parameters(df)

    plot_temperature_relationship(df)

    plot_relationship(
        df,
        "rssi_dbm",
        "14_error_vs_rssi.png"
    )

    plot_relationship(
        df,
        "fpp_dbm",
        "15_error_vs_fpp.png"
    )

    plot_relationship(
        df,
        "cfo_ppm",
        "16_error_vs_cfo.png"
    )

    plot_relationship(
        df,
        "t_round_ticks",
        "17_error_vs_t_round.png"
    )

    plot_relationship(
        df,
        "t_reply_ticks",
        "18_error_vs_t_reply.png"
    )

    plot_distribution(df)

    plot_windowed_drift(windowed)

    # --------------------------------------------------------
    # Save processed data
    # --------------------------------------------------------

    output_csv = os.path.join(
        OUTPUT_DIR,
        "processed_data.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"Samples             : "
        f"{len(df)}"
    )

    print(
        f"Mean raw range      : "
        f"{range_stats['mean']:.6f} m"
    )

    print(
        f"Median raw range    : "
        f"{range_stats['median']:.6f} m"
    )

    print(
        f"Noise Std           : "
        f"{range_stats['std'] * 100:.3f} cm"
    )

    print(
        f"Robust sigma        : "
        f"{range_stats['robust_sigma'] * 100:.3f} cm"
    )

    print(
        f"Bias                : "
        f"{range_stats['bias'] * 1000:+.3f} mm"
    )

    print(
        f"RMSE                : "
        f"{range_stats['rmse'] * 1000:.3f} mm"
    )

    print(
        f"Drift               : "
        f"{range_stats['drift_mm_s']:+.6f} mm/s"
    )

    print(
        f"Drift               : "
        f"{range_stats['drift_mm_min']:+.6f} mm/min"
    )

    print(
        f"Drift R^2           : "
        f"{range_stats['drift_r2']:.6f}"
    )

    print(
        f"Response success    : "
        f"{reliability['success_rate']:.4f} %"
    )

    print(
        f"Outlier rate        : "
        f"{100.0 * range_stats['outlier_count'] / len(df):.4f} %"
    )

    print("\n")
    print(
        f"[DONE] Results saved to: "
        f"{OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()
