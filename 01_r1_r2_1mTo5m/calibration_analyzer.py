#!/usr/bin/env python3
"""
UWB Systematic Offset and Clock Drift Identification Tool
Project: Anjoman Swarm Firmware
"""

import sys
import os
import re
import statistics

# Ground Truth distance in meters (100 cm = 1.000 m)
GROUND_TRUTH_M = 1.000

def parse_log_file(filepath):
    """
    Robust regex parser extracting distance (meters), RSSI, and exchange counter.
    Handles multiple log formats seamlessly.
    """
    distances = []
    rssis = []
    exchanges = []
    
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None

    # Regex patterns matching various log prints
    pattern_standard = re.compile(r'Distance:\s*([\d\.\-]+)\s*m.*RSSI:\s*([\d\.\-]+)\s*dBm.*(?:Ex|Pings|Locks|Exchanges):\s*(\d+)', re.IGNORECASE)
    pattern_raw = re.compile(r'Dist:\s*([\d\.\-]+)\s*m.*RSSI:\s*([\d\.\-]+)\s*dBm.*(?:Ex|Pings|Locks|Exchanges):\s*(\d+)', re.IGNORECASE)

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            match = pattern_standard.search(line) or pattern_raw.search(line)
            if match:
                try:
                    dist = float(match.group(1))
                    rssi = float(match.group(2))
                    ex = int(match.group(3))
                    
                    # Filter out uninitialized / overflow outliers
                    if 0.0 < dist < 100.0:
                        distances.append(dist)
                        rssis.append(rssi)
                        exchanges.append(ex)
                except ValueError:
                    continue

    return {
        'distance': distances,
        'rssi': rssis,
        'exchange': exchanges
    }

def linear_regression_ols(x_vals, y_vals):
    """
    Calculates Ordinary Least Squares (OLS) slope (drift rate) and intercept.
    """
    n = len(x_vals)
    if n < 2:
        return 0.0, y_vals[0] if n == 1 else 0.0
    
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    
    numerator = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    denominator = sum((x_vals[i] - mean_x) ** 2 for i in range(n))
    
    if denominator == 0:
        return 0.0, mean_y
        
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept

def analyze_dataset(name, data, ground_truth=GROUND_TRUTH_M):
    dist = data['distance']
    rssi = data['rssi']
    n = len(dist)
    
    if n == 0:
        print(f"[WARN] No valid data points found in dataset: {name}")
        return None

    # Time / Sample index estimation (assuming ~2 Hz logging rate)
    time_indices = [i * 0.5 for i in range(n)] # In seconds

    mean_dist = statistics.mean(dist)
    median_dist = statistics.median(dist)
    stdev_dist = statistics.stdev(dist) if n > 1 else 0.0
    min_dist = min(dist)
    max_dist = max(dist)
    p2p_jitter = max_dist - min_dist
    
    mean_rssi = statistics.mean(rssi) if rssi else 0.0

    # Calculate Systematic Offset
    systematic_offset = mean_dist - ground_truth
    
    # Calculate Linear Drift Rate (m/s and mm/s)
    drift_slope, drift_intercept = linear_regression_ols(time_indices, dist)
    drift_rate_mm_per_sec = drift_slope * 1000.0

    print("=" * 70)
    print(f"               STATISTICAL ANALYSIS REPORT: {name}")
    print("=" * 70)
    print(f"Total Samples Parsed      : {n}")
    print(f"Ground Truth Distance     : {ground_truth:.4f} m ({ground_truth * 100.0:.1f} cm)")
    print(f"Mean Measured Distance    : {mean_dist:.4f} m ({mean_dist * 100.0:.2f} cm)")
    print(f"Median Measured Distance  : {median_dist:.4f} m ({median_dist * 100.0:.2f} cm)")
    print(f"Standard Deviation (Noise): {stdev_dist * 100.0:.2f} cm (Variance: {stdev_dist**2:.6f} m^2)")
    print(f"Min / Max Measured        : {min_dist:.4f} m / {max_dist:.4f} m")
    print(f"Peak-to-Peak Jitter       : {p2p_jitter * 100.0:.2f} cm")
    print(f"Mean RSSI                 : {mean_rssi:.1f} dBm")
    print("-" * 70)
    print(f"IDENTIFIED FIXED OFFSET   : {systematic_offset:.4f} m ({systematic_offset * 100.0:.2f} cm)")
    print(f"TEMPORAL DRIFT RATE       : {drift_rate_mm_per_sec:+.4f} mm/s (Slope: {drift_slope:+.6e} m/s)")
    print("=" * 70)

    return {
        'mean': mean_dist,
        'median': median_dist,
        'offset': systematic_offset,
        'drift_slope': drift_slope,
        'stdev': stdev_dist
    }

def main():
    file_r1 = "r1.txt"
    file_r2 = "r2.txt"

    if len(sys.argv) >= 3:
        file_r1 = sys.argv[1]
        file_r2 = sys.argv[2]

    print("[UWB-ANALYZER] Processing input log files...")
    data_r1 = parse_log_file(file_r1)
    data_r2 = parse_log_file(file_r2)

    res_r1 = None
    res_r2 = None

    if data_r1 and len(data_r1['distance']) > 0:
        res_r1 = analyze_dataset("ROBOT 1 (Initiator Log)", data_r1)
    
    if data_r2 and len(data_r2['distance']) > 0:
        res_r2 = analyze_dataset("ROBOT 2 (Responder / Mirror Log)", data_r2)

    if res_r1:
        print("\n" + "#" * 70)
        print("                 RECOMMENDED FIRMWARE CONFIGURATION")
        print("#" * 70)
        print(f"constexpr float UWB_CALIBRATION_OFFSET_M = {res_r1['offset']:.4f}f;")
        print(f"// Expected Measurement Accuracy: +/- {res_r1['stdev'] * 100.0:.1f} cm (1-Sigma)")
        print("#" * 70 + "\n")

if __name__ == "__main__":
    main()
