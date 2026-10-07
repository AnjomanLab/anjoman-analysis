#!/usr/bin/env python3

import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


GROUND_TRUTH_M = 2.5
TIMESTAMP_BITS = 40
TIMESTAMP_MOD = 1 << TIMESTAMP_BITS

FILES = [
    "r1_r2_2.5.txt",
    "r1_r3_2.5.txt",
    "r1_r4_2.5.txt",
    # "r2_r3_2.5_with_gap.txt",  # intentionally excluded
    "r2_r4_2.5.txt",
    "r3_r4_2.5.txt",
]


def load_file(path):
    rows = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                rows.append({
                    "timestamp": float(row["timestamp"]),
                    "sequence": int(row["sequence"]),

                    "range_raw": float(row["range_raw_m"]),
                    "range_cal": float(row["range_calibrated_m"]),

                    "rssi": float(row["rssi_dbm"]),

                    "tTx1": int(row["tTx1"]),
                    "tRx1": int(row["tRx1"]),
                    "tRx2": int(row["tRx2"]),
                    "tTx2": int(row["tTx2"]),

                    "temp_uwb_1": float(row["temp_uwb_1"]),
                    "temp_uwb_2": float(row["temp_uwb_2"]),
                    "temp_esp_1": float(row["temp_esp_1"]),
                    "temp_esp_2": float(row["temp_esp_2"]),
                })
            except (ValueError, KeyError):
                continue

    return rows


def unwrap_40bit(values):
    """
    Unwrap DW1000 40-bit timestamps.

    Assumes consecutive samples are sufficiently close
    that the actual increment is much smaller than 2^39.
    """
    if not values:
        return np.array([])

    out = [values[0]]
    offset = 0
    previous = values[0]

    for current in values[1:]:
        delta = current - previous

        if delta < -(TIMESTAMP_MOD // 2):
            offset += TIMESTAMP_MOD

        elif delta > (TIMESTAMP_MOD // 2):
            offset -= TIMESTAMP_MOD

        out.append(current + offset)
        previous = current

    return np.asarray(out, dtype=np.float64)


def linear_drift(time_s, values):
    if len(values) < 2:
        return np.nan, np.nan

    x = np.asarray(time_s)
    y = np.asarray(values)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, np.nan

    slope, intercept = np.polyfit(x, y, 1)

    predicted = slope * x + intercept

    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return slope, r2


def correlation(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return np.corrcoef(x, y)[0, 1]


def percentile_string(values):
    p = np.percentile(values, [1, 5, 50, 95, 99])

    return (
        f"P01={p[0]:+.4f} m, "
        f"P05={p[1]:+.4f} m, "
        f"P50={p[2]:+.4f} m, "
        f"P95={p[3]:+.4f} m, "
        f"P99={p[4]:+.4f} m"
    )


def analyze(path):
    rows = load_file(path)

    if not rows:
        print(f"[WARNING] No valid data: {path}")
        return None

    # ------------------------------------------------------------
    # Basic arrays
    # ------------------------------------------------------------

    timestamp = np.array([r["timestamp"] for r in rows])

    raw = np.array([r["range_raw"] for r in rows])
    cal = np.array([r["range_cal"] for r in rows])

    rssi = np.array([r["rssi"] for r in rows])

    uwb1 = np.array([r["temp_uwb_1"] for r in rows])
    uwb2 = np.array([r["temp_uwb_2"] for r in rows])

    esp1 = np.array([r["temp_esp_1"] for r in rows])
    esp2 = np.array([r["temp_esp_2"] for r in rows])

    # ------------------------------------------------------------
    # Time
    # ------------------------------------------------------------

    time_s = (timestamp - timestamp[0]) / 1000.0

    duration_s = time_s[-1] if len(time_s) else 0

    dt = np.diff(time_s)

    valid_dt = dt[dt > 0]

    sampling_hz = (
        1.0 / np.median(valid_dt)
        if len(valid_dt)
        else np.nan
    )

    # ------------------------------------------------------------
    # Timestamp unwrap
    # ------------------------------------------------------------

    tTx1 = unwrap_40bit([r["tTx1"] for r in rows])
    tRx1 = unwrap_40bit([r["tRx1"] for r in rows])
    tRx2 = unwrap_40bit([r["tRx2"] for r in rows])
    tTx2 = unwrap_40bit([r["tTx2"] for r in rows])

    # Initiator-side round trip
    t_round = tRx1 - tTx1

    # Responder-side reply delay
    t_reply = tTx2 - tRx2

    # ------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------

    raw_error = raw - GROUND_TRUTH_M
    cal_error = cal - GROUND_TRUTH_M

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    def stats(x):
        return {
            "mean": np.mean(x),
            "median": np.median(x),
            "std": np.std(x, ddof=1),
            "min": np.min(x),
            "max": np.max(x),
            "mae": np.mean(np.abs(x)),
            "rmse": np.sqrt(np.mean(x ** 2)),
        }

    raw_stats = stats(raw_error)
    cal_stats = stats(cal_error)

    # ------------------------------------------------------------
    # Drift
    # ------------------------------------------------------------

    raw_drift_mps, raw_r2 = linear_drift(time_s, raw_error)
    cal_drift_mps, cal_r2 = linear_drift(time_s, cal_error)

    # ------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------

    corr = {
        "cal_error_vs_rssi": correlation(cal_error, rssi),

        "cal_error_vs_uwb1_temp":
            correlation(cal_error, uwb1),

        "cal_error_vs_uwb2_temp":
            correlation(cal_error, uwb2),

        "cal_error_vs_esp1_temp":
            correlation(cal_error, esp1),

        "cal_error_vs_esp2_temp":
            correlation(cal_error, esp2),

        "cal_error_vs_t_round":
            correlation(cal_error, t_round),

        "cal_error_vs_t_reply":
            correlation(cal_error, t_reply),
    }

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print(f"FILE: {path}")
    print("=" * 78)

    print(f"Samples             : {len(rows):,}")
    print(f"Duration            : {duration_s / 60:.2f} min")
    print(f"Sampling rate       : {sampling_hz:.3f} Hz")

    print()
    print("RAW RANGE")
    print(f"Mean                : {np.mean(raw):.6f} m")
    print(f"Median              : {np.median(raw):.6f} m")
    print(f"Noise Std           : {raw_stats['std'] * 100:.3f} cm")
    print(f"Bias                : {raw_stats['mean']:+.6f} m")
    print(f"MAE                 : {raw_stats['mae']:.6f} m")
    print(f"RMSE                : {raw_stats['rmse']:.6f} m")
    print(f"Min / Max           : {np.min(raw):.6f} / {np.max(raw):.6f} m")
    print(f"Drift               : {raw_drift_mps * 1000:+.6f} mm/s")
    print(f"Drift               : {raw_drift_mps * 60 * 1000:+.3f} mm/min")
    print(f"Drift R^2            : {raw_r2:.6f}")

    print()
    print("CALIBRATED RANGE")
    print(f"Mean                : {np.mean(cal):.6f} m")
    print(f"Median              : {np.median(cal):.6f} m")
    print(f"Noise Std           : {cal_stats['std'] * 100:.3f} cm")
    print(f"Bias                : {cal_stats['mean']:+.6f} m")
    print(f"MAE                 : {cal_stats['mae']:.6f} m")
    print(f"RMSE                : {cal_stats['rmse']:.6f} m")
    print(f"Min / Max           : {np.min(cal):.6f} / {np.max(cal):.6f} m")
    print(f"Drift               : {cal_drift_mps * 1000:+.6f} mm/s")
    print(f"Drift               : {cal_drift_mps * 60 * 1000:+.3f} mm/min")
    print(f"Drift R^2            : {cal_r2:.6f}")

    print()
    print("PERCENTILES — CALIBRATED ERROR")
    print(percentile_string(cal_error))

    print()
    print("RF")
    print(f"RSSI Mean           : {np.mean(rssi):.3f} dBm")
    print(f"RSSI Std            : {np.std(rssi, ddof=1):.3f} dBm")
    print(f"RSSI Min / Max      : {np.min(rssi):.3f} / {np.max(rssi):.3f} dBm")

    print()
    print("TEMPERATURE")
    print(f"UWB 1               : {np.mean(uwb1):.3f} C")
    print(f"UWB 2               : {np.mean(uwb2):.3f} C")
    print(f"ESP32 1             : {np.mean(esp1):.3f} C")
    print(f"ESP32 2             : {np.mean(esp2):.3f} C")

    print()
    print("TIMESTAMP DERIVED")
    print(f"T_round mean        : {np.mean(t_round):.3f} ticks")
    print(f"T_round std         : {np.std(t_round, ddof=1):.3f} ticks")
    print(f"T_reply mean        : {np.mean(t_reply):.3f} ticks")
    print(f"T_reply std         : {np.std(t_reply, ddof=1):.3f} ticks")

    print()
    print("CORRELATION WITH CALIBRATED RANGE ERROR")

    for name, value in corr.items():
        print(f"{name:30s}: {value:+.6f}")

    return {
        "file": path,
        "samples": len(rows),
        "duration_s": duration_s,
        "sampling_hz": sampling_hz,

        "raw_mean": np.mean(raw),
        "raw_std": raw_stats["std"],
        "raw_bias": raw_stats["mean"],
        "raw_rmse": raw_stats["rmse"],
        "raw_drift_mps": raw_drift_mps,
        "raw_r2": raw_r2,

        "cal_mean": np.mean(cal),
        "cal_std": cal_stats["std"],
        "cal_bias": cal_stats["mean"],
        "cal_rmse": cal_stats["rmse"],
        "cal_drift_mps": cal_drift_mps,
        "cal_r2": cal_r2,

        "rssi_mean": np.mean(rssi),
        "rssi_std": np.std(rssi, ddof=1),

        "uwb1_temp_mean": np.mean(uwb1),
        "uwb2_temp_mean": np.mean(uwb2),

        "corr_rssi": corr["cal_error_vs_rssi"],
        "corr_uwb1": corr["cal_error_vs_uwb1_temp"],
        "corr_uwb2": corr["cal_error_vs_uwb2_temp"],
    }


def plot_pair(path):
    rows = load_file(path)

    if not rows:
        return

    timestamp = np.array([r["timestamp"] for r in rows])
    time_min = (timestamp - timestamp[0]) / 60000.0

    raw = np.array([r["range_raw"] for r in rows])
    cal = np.array([r["range_cal"] for r in rows])

    rssi = np.array([r["rssi"] for r in rows])

    uwb1 = np.array([r["temp_uwb_1"] for r in rows])
    uwb2 = np.array([r["temp_uwb_2"] for r in rows])

    error = cal - GROUND_TRUTH_M

    stem = Path(path).stem

    outdir = Path("plots")
    outdir.mkdir(exist_ok=True)

    # ------------------------------------------------------------
    # Range trajectory
    # ------------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(time_min, raw, label="Raw")
    plt.plot(time_min, cal, label="Calibrated")
    plt.axhline(GROUND_TRUTH_M, linestyle="--", label="Ground Truth")

    plt.xlabel("Time (min)")
    plt.ylabel("Range (m)")
    plt.title(f"UWB Range — {stem}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(outdir / f"{stem}_range.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Calibrated error
    # ------------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(time_min, error * 1000)

    plt.axhline(0, linestyle="--")

    plt.xlabel("Time (min)")
    plt.ylabel("Range Error (mm)")
    plt.title(f"Calibrated Range Error — {stem}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(outdir / f"{stem}_error.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # RSSI
    # ------------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(time_min, rssi)

    plt.xlabel("Time (min)")
    plt.ylabel("RSSI (dBm)")
    plt.title(f"RSSI — {stem}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(outdir / f"{stem}_rssi.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(time_min, uwb1, label="UWB 1")
    plt.plot(time_min, uwb2, label="UWB 2")

    plt.xlabel("Time (min)")
    plt.ylabel("Temperature (C)")
    plt.title(f"UWB Temperature — {stem}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(outdir / f"{stem}_temperature.png", dpi=150)
    plt.close()


def main():
    results = []

    for filename in FILES:
        path = Path(filename)

        if not path.exists():
            print(f"[WARNING] Missing file: {filename}")
            continue

        result = analyze(path)

        if result:
            results.append(result)

        plot_pair(path)

    # ------------------------------------------------------------
    # Cross-pair summary
    # ------------------------------------------------------------

    if not results:
        return

    print()
    print()
    print("=" * 120)
    print("CROSS-PAIR SUMMARY — 2.5 m")
    print("=" * 120)

    header = (
        f"{'Pair':18s}"
        f"{'N':>8s}"
        f"{'Mean(m)':>11s}"
        f"{'Bias(cm)':>11s}"
        f"{'Std(cm)':>10s}"
        f"{'RMSE(cm)':>11s}"
        f"{'Drift(mm/s)':>14s}"
        f"{'R2':>9s}"
        f"{'RSSI':>10s}"
    )

    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{Path(r['file']).stem:18s}"
            f"{r['samples']:8d}"
            f"{r['cal_mean']:11.4f}"
            f"{r['cal_bias'] * 100:11.3f}"
            f"{r['cal_std'] * 100:10.3f}"
            f"{r['cal_rmse'] * 100:11.3f}"
            f"{r['cal_drift_mps'] * 1000:14.6f}"
            f"{r['cal_r2']:9.4f}"
            f"{r['rssi_mean']:10.2f}"
        )

    print()
    print("[DONE] Plots saved under ./plots/")


if __name__ == "__main__":
    main()
