#!/usr/bin/env python3
"""
Anjoman Swarm Robot Platform - UWB Multivariate Calibration Analyzer
Designed for leaderless, decentralized multi-robot formation control.
This script performs high-precision multivariate linear regression on raw UWB logs
to extract fixed antenna delays (offsets) and thermal/CFO drift parameters,
generating production-ready C++ configuration blocks for the ESP32-S3 firmware.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_calibration(file_path, ground_truth, output_plot):
    print(f"[UWB-CALIB] Loading log file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)
    
    # Strip any whitespaces from columns
    df.columns = df.columns.str.strip()
    
    required_cols = ['range_raw_m', 'temp_uwb_2', 'cfo_ppm', 'rssi_dbm', 'fpp_dbm']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the CSV file. Available columns: {list(df.columns)}")

    # Clean data (drop NaNs and extreme outliers)
    initial_len = len(df)
    df = df.dropna(subset=required_cols)
    
    # Simple outlier rejection (e.g. range_raw_m must be within reasonable limits, e.g., > 0 and < 100)
    df = df[(df['range_raw_m'] > 0) & (df['range_raw_m'] < 100)]
    cleaned_len = len(df)
    
    print(f"[UWB-CALIB] Loaded {cleaned_len} valid rows (dropped {initial_len - cleaned_len} invalid rows)")

    # Define variables
    d_raw = df['range_raw_m'].values
    t_uwb2 = df['temp_uwb_2'].values
    cfo = df['cfo_ppm'].values
    rssi = df['rssi_dbm'].values
    fpp = df['fpp_dbm'].values
    
    # Calculate raw error
    error_raw = d_raw - ground_truth
    
    # Calculate some helper features
    rssi_fpp_diff = rssi - fpp

    # MULTIVARIATE REGRESSION MODEL
    # Model 1: Temperature + CFO
    # Error = beta_0 + beta_1 * temp_uwb_2 + beta_2 * cfo_ppm
    A = np.vstack([np.ones(len(df)), t_uwb2, cfo]).T
    betas, residuals, rank, s = np.linalg.lstsq(A, error_raw, rcond=None)
    
    beta_0, beta_temp, beta_cfo = betas
    
    # Apply calibration Model 1
    d_calibrated = d_raw - (beta_0 + beta_temp * t_uwb2 + beta_cfo * cfo)
    error_calibrated = d_calibrated - ground_truth

    # Model 2: Temperature + CFO + RSSI/FPP (LOS/NLOS power-induced bias)
    A_full = np.vstack([np.ones(len(df)), t_uwb2, cfo, rssi_fpp_diff]).T
    betas_full, _, _, _ = np.linalg.lstsq(A_full, error_raw, rcond=None)
    b0_f, b_temp_f, b_cfo_f, b_power_f = betas_full
    
    d_cal_full = d_raw - (b0_f + b_temp_f * t_uwb2 + b_cfo_f * cfo + b_power_f * rssi_fpp_diff)
    error_cal_full = d_cal_full - ground_truth

    # Statistical Evaluation
    mae_raw = np.mean(np.abs(error_raw))
    rmse_raw = np.sqrt(np.mean(error_raw**2))
    std_raw = np.std(d_raw)
    
    mae_cal = np.mean(np.abs(error_calibrated))
    rmse_cal = np.sqrt(np.mean(error_calibrated**2))
    std_cal = np.std(d_calibrated)

    mae_cal_f = np.mean(np.abs(error_cal_full))
    rmse_cal_f = np.sqrt(np.mean(error_cal_full**2))
    std_cal_f = np.std(d_cal_full)

    # R-squared calculation
    ss_tot = np.sum((error_raw - np.mean(error_raw))**2)
    ss_res_m1 = np.sum(error_calibrated**2)
    r2_m1 = 1 - (ss_res_m1 / ss_tot) if ss_tot > 0 else 0.0

    print("\n" + "="*80)
    print("                 UWB CALIBRATION SYSTEMATIC ANALYSIS REPORT")
    print("="*80)
    print(f"Log File Analyzed          : {os.path.basename(file_path)}")
    print(f"Ground Truth Reference     : {ground_truth:.4f} m")
    print(f"Total Samples Parsed       : {cleaned_len}")
    print("-"*80)
    print("METRIC                    |   RAW LOG DATA   |  CALIBRATED (M1)  | CALIBRATED (M2-FULL)")
    print("-"*80)
    print(f"Mean Measured Distance     |    {np.mean(d_raw):11.4f} m |    {np.mean(d_calibrated):11.4f} m |    {np.mean(d_cal_full):11.4f} m")
    print(f"Mean Error (Bias)          |    {np.mean(error_raw)*100:11.4f} cm|    {np.mean(error_calibrated)*100:11.4f} cm|    {np.mean(error_cal_full)*100:11.4f} cm")
    print(f"Mean Absolute Error (MAE)  |    {mae_raw*100:11.4f} cm|    {mae_cal*100:11.4f} cm|    {mae_cal_f*100:11.4f} cm")
    print(f"Root Mean Sq. Error (RMSE) |    {rmse_raw*100:11.4f} cm|    {rmse_cal*100:11.4f} cm|    {rmse_cal_f*100:11.4f} cm")
    print(f"Standard Deviation (Noise) |    {std_raw*100:11.4f} cm|    {std_cal*100:11.4f} cm|    {std_cal_f*100:11.4f} cm")
    print(f"Model R-squared (Fit R^2)  |         N/A      |    {r2_m1:11.4f}   |         N/A")
    print("="*80)

    # Output C++ Firmware Block
    print("\n" + "#"*80)
    print("                 RECOMMENDED ESP32-S3 FIRMWARE CONFIGURATION")
    print("#"*80)
    print(f"// Copy-paste this configuration into your 'Anjoman' ESP32-S3 PlatformIO project.")
    print(f"// File: include/UwbCalibration.h (or apply relative to the node pair)")
    print(f"namespace UwbCalib {{")
    print(f"    // Calibration Model 1: d_cal = d_raw - (OFFSET + K_TEMP * temp_uwb_2 + K_CFO * cfo_ppm)")
    print(f"    constexpr float UWB_FIXED_OFFSET_M  = {beta_0:.8f}f;  // Fixed Antenna Delay + Multipath Offset")
    print(f"    constexpr float UWB_K_TEMP          = {beta_temp:.8f}f;  // Thermal Drift Coefficient (m/C)")
    print(f"    constexpr float UWB_K_CFO           = {beta_cfo:.12f}f;  // Clock Drift Frequency Coefficient (m/ppm)")
    print(f"")
    print(f"    // Model 2 (Includes RSSI-FPP Power Correction for NLOS paths)")
    print(f"    constexpr float UWB_M2_OFFSET_M     = {b0_f:.8f}f;")
    print(f"    constexpr float UWB_M2_K_TEMP       = {b_temp_f:.8f}f;")
    print(f"    constexpr float UWB_M2_K_CFO        = {b_cfo_f:.12f}f;")
    print(f"    constexpr float UWB_M2_K_POWER      = {b_power_f:.8f}f;  // RSSI-FPP Difference Factor (m/dB)")
    print(f"}}")
    print("#"*80 + "\n")

    # Generate Plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Timeline Error Comparison
    axs[0, 0].plot(error_raw * 100, label='Raw Error (Uncalibrated)', color='red', alpha=0.6)
    axs[0, 0].plot(error_calibrated * 100, label='Calibrated Error (Model 1)', color='blue', alpha=0.8)
    axs[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axs[0, 0].set_title('Ranging Error Comparison Over Time')
    axs[0, 0].set_xlabel('Sample Index')
    axs[0, 0].set_ylabel('Ranging Error (cm)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Temperature vs Error correlation
    axs[0, 1].scatter(t_uwb2, error_raw * 100, color='red', alpha=0.3, label='Raw Error')
    # Plot regression trendline
    t_range = np.linspace(np.min(t_uwb2), np.max(t_uwb2), 100)
    # Project assuming mean CFO
    trend = beta_0 + beta_temp * t_range + beta_cfo * np.mean(cfo)
    axs[0, 1].plot(t_range, trend * 100, color='blue', linewidth=2.5, label='Thermal Calibration Trend')
    axs[0, 1].set_title('UWB Chip Temp 2 vs. Ranging Error')
    axs[0, 1].set_xlabel('UWB Sensor 2 Temperature (°C)')
    axs[0, 1].set_ylabel('Ranging Error (cm)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 3. CFO vs Error correlation
    axs[1, 0].scatter(cfo, error_raw * 100, color='orange', alpha=0.3, label='Raw Error vs CFO')
    axs[1, 0].set_title('Carrier Frequency Offset vs. Ranging Error')
    axs[1, 0].set_xlabel('CFO (ppm)')
    axs[1, 0].set_ylabel('Ranging Error (cm)')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Error Distribution Histogram
    axs[1, 1].hist(error_raw * 100, bins=50, color='red', alpha=0.5, label='Raw Error Std')
    axs[1, 1].hist(error_calibrated * 100, bins=50, color='blue', alpha=0.7, label='Calibrated Error Std')
    axs[1, 1].set_title('Error Distribution Histogram')
    axs[1, 1].set_xlabel('Error Magnitude (cm)')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"[UWB-CALIB] Statistical diagnostic plot saved to: {output_plot}")

def main():
    parser = argparse.ArgumentParser(description='UWB Log Calibration and Analysis Script for Anjoman Swarm Project')
    parser.add_argument('file', type=str, nargs='?', default='r1_r2_3m.txt', help='Path to UWB log CSV file (default: r1_r2_3m.txt)')
    parser.add_argument('--gt', type=float, default=3.0, help='Ground truth distance in meters (default: 3.0 m)')
    parser.add_argument('--plot', type=str, default='uwb_calibration_diagnostics.png', help='Path to save output diagnostic plot PNG')
    
    args = parser.parse_args()
    
    # Check if file exists, if not, write a dummy dataset for verification/example
    if not os.path.exists(args.file):
        print(f"[UWB-CALIB] Warning: Log file '{args.file}' not found. Generating a mock dataset matching Anjoman telemetry spec for demonstration.")
        # Generate representative mock data matching r1_r2_3m.txt parameters
        np.random.seed(42)
        n_samples = 2000
        
        # Simulating crystal heating from 30C to 36C
        temp_uwb2 = np.linspace(30.0, 36.0, n_samples) + np.random.normal(0, 0.1, n_samples)
        temp_uwb1 = temp_uwb2 - 1.2
        # CFO correlates with temperature (negative correlation)
        cfo = 8200.0 - (temp_uwb2 - 30.0) * 50.0 + np.random.normal(0, 10, n_samples)
        
        # Real distance is exactly ground truth
        d_real = args.gt
        
        # Fixed offset is around 22.27 m (as in r1_r2)
        offset = 22.271960
        # Positive thermal drift (+190.006 mm/C as in r1_r2 log)
        temp_drift = (temp_uwb2 - 34.225) * 0.190006
        # Noise (Std = 24.04 cm as in r1_r2)
        noise = np.random.normal(0, 0.24040, n_samples)
        
        d_raw = d_real + offset + temp_drift + noise
        
        mock_data = pd.DataFrame({
            'timestamp': np.arange(n_samples),
            'sequence': np.arange(n_samples),
            'range_raw_m': d_raw,
            'range_calibrated_m': d_raw - offset,
            'rssi_dbm': -68.0 + np.random.normal(0, 0.8, n_samples),
            'fpp_dbm': -80.5 + np.random.normal(0, 1.4, n_samples),
            'cfo_ppm': cfo,
            'tTx1': np.zeros(n_samples),
            'tRx1': np.zeros(n_samples),
            'tRx2': np.zeros(n_samples),
            'tTx2': np.zeros(n_samples),
            'temp_uwb_1': temp_uwb1,
            'temp_uwb_2': temp_uwb2,
            'temp_esp_1': temp_uwb1 + 1.5,
            'temp_esp_2': temp_uwb2 + 1.5,
            'response_received': np.ones(n_samples),
            'status': np.ones(n_samples)
        })
        mock_data.to_csv(args.file, index=False)
        print(f"[UWB-CALIB] Mock dataset written to: {args.file}")

    try:
        analyze_calibration(args.file, args.gt, args.plot)
    except Exception as e:
        print(f"[UWB-CALIB] Error executing analysis: {e}")

if __name__ == '__main__':
    main()
