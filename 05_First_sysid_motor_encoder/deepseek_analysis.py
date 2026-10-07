#!/usr/bin/env python3
"""
Actuator SysID Analysis Script - Phase-Aware Version

Reads sysid_rX.csv files from Arduino-based motor/encoder data,
analyzes each phase separately according to the Arduino FSM,
computes motor parameters per phase,
generates plots, and prints a detailed report.

Usage:
    python3 analyze_sysid.py
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import warnings

# Suppress specific warnings during fitting
warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------
# Constants
# -------------------------------
ENCODER_RESOLUTION = 4096          # 12-bit AS5600
DEG_PER_STEP = 360.0 / ENCODER_RESOLUTION
RAD_PER_STEP = np.deg2rad(DEG_PER_STEP)
SAMPLE_RATE_HZ = 100.0

# Phase definitions matching Arduino code exactly
PHASE_DEFINITIONS = {
    0: {'name': 'REST', 'start_ms': 0, 'end_ms': 5000, 'description': 'Baseline static sensor noise'},
    1: {'name': 'LEFT_FWD_STAIRCASE', 'start_ms': 5000, 'end_ms': 23000, 'description': 'Left motor forward staircase (9 levels)'},
    2: {'name': 'LEFT_REV_STAIRCASE', 'start_ms': 23000, 'end_ms': 41000, 'description': 'Left motor reverse staircase (9 levels)'},
    3: {'name': 'LEFT_STEP_RESPONSE', 'start_ms': 41000, 'end_ms': 59000, 'description': 'Left motor step responses (0.4, 0.7, 1.0)'},
    4: {'name': 'RIGHT_FWD_STAIRCASE', 'start_ms': 59000, 'end_ms': 77000, 'description': 'Right motor forward staircase (9 levels)'},
    5: {'name': 'RIGHT_REV_STAIRCASE', 'start_ms': 77000, 'end_ms': 95000, 'description': 'Right motor reverse staircase (9 levels)'},
    6: {'name': 'RIGHT_STEP_RESPONSE', 'start_ms': 95000, 'end_ms': 113000, 'description': 'Right motor step responses (0.4, 0.7, 1.0)'},
    7: {'name': 'DUAL_SYNC_STEPS', 'start_ms': 113000, 'end_ms': 137000, 'description': 'Dual motor synchronous steps (0.3, 0.5, 0.7, 0.9)'},
    8: {'name': 'COAST_DOWN', 'start_ms': 137000, 'end_ms': 157000, 'description': 'Coast-down friction test'},
    9: {'name': 'TERMINATION', 'start_ms': 157000, 'end_ms': 160000, 'description': 'Brake and termination'}
}

# Staircase levels
STAIRCASE_FWD = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80, 1.00]
STAIRCASE_REV = [-0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.60, -0.80, -1.00]

# Step response commands
STEP_COMMANDS = [0.40, 0.70, 1.00]

# Dual motor commands
DUAL_COMMANDS = [0.30, 0.50, 0.70, 0.90]

# -------------------------------
# Helper Functions
# -------------------------------
def load_sysid_file(filepath):
    """Load CSV file, skipping non-data lines."""
    valid_lines = []
    header = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 12:
                continue
            try:
                float(parts[0])
                if header is not None:
                    valid_lines.append(line)
            except ValueError:
                if 'TimeUs' in parts[0]:
                    header = line
    if header is None:
        raise ValueError("No valid CSV header found")
    csv_text = header + '\n' + '\n'.join(valid_lines)
    df = pd.read_csv(io.StringIO(csv_text))
    t0_us = df['TimeUs'].iloc[0]
    df['ElapsedMs'] = (df['TimeUs'] - t0_us) / 1000.0
    return df

def get_phase_mask(df, phase_num):
    """Get mask for a specific phase."""
    phase = PHASE_DEFINITIONS[phase_num]
    return (df['ElapsedMs'] >= phase['start_ms']) & (df['ElapsedMs'] < phase['end_ms'])

def compute_velocities(df):
    """Compute instantaneous velocities for both motors."""
    dt = np.diff(df['TimeUs']) / 1e6
    df['VelL'] = 0.0
    df['VelR'] = 0.0
    df.loc[1:, 'VelL'] = df['DeltaL'].iloc[1:].values / dt
    df.loc[1:, 'VelR'] = df['DeltaR'].iloc[1:].values / dt
    return df

def moving_average(data, window=5):
    """Simple moving average."""
    if window < 1 or len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='same')

def fit_linear(x, y):
    """Linear regression."""
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value**2

def fit_step_response(t, v, tau_guess=0.5):
    """Fit first-order step response: v(t) = v_ss * (1 - exp(-t/tau))."""
    def model(t, v_ss, tau):
        return v_ss * (1 - np.exp(-t / tau))
    try:
        popt, _ = curve_fit(model, t, v, p0=[v[-1] if len(v)>0 else 1.0, tau_guess],
                            maxfev=10000, bounds=([0, 0.01], [np.inf, 5.0]))
        return popt[0], popt[1]
    except:
        return np.nan, np.nan

def fit_coast_down(t, v, tau_guess=0.3):
    """Fit exponential decay: v(t) = v0 * exp(-t/tau)."""
    def model(t, v0, tau):
        return v0 * np.exp(-t / tau)
    try:
        popt, _ = curve_fit(model, t, v, p0=[v[0] if len(v)>0 else 1.0, tau_guess],
                            maxfev=10000, bounds=([0, 0.01], [np.inf, 5.0]))
        return popt[0], popt[1]
    except:
        return np.nan, np.nan

# -------------------------------
# Analysis Functions per Phase
# -------------------------------
def analyze_staircase(df, phase_num, motor, direction):
    """Analyze staircase response for a motor in a specific direction."""
    phase = PHASE_DEFINITIONS[phase_num]
    mask = get_phase_mask(df, phase_num)
    df_phase = df[mask].copy()
    
    if direction == 'forward':
        commands = STAIRCASE_FWD
    else:
        commands = STAIRCASE_REV
    
    results = {'commands': [], 'velocities': [], 'velocities_std': []}
    
    for i, cmd in enumerate(commands):
        seg_start = phase['start_ms'] + i * 2000
        seg_end = seg_start + 2000
        mask_seg = (df_phase['ElapsedMs'] >= seg_start) & (df_phase['ElapsedMs'] < seg_end)
        df_seg = df_phase[mask_seg]
        
        if len(df_seg) < 10:
            continue
        
        vel_col = f'Vel{motor}'
        if direction == 'reverse':
            # For reverse, use absolute velocity
            vel = np.abs(df_seg[vel_col].median())
        else:
            vel = df_seg[vel_col].median()
        
        results['commands'].append(abs(cmd))
        results['velocities'].append(vel)
        results['velocities_std'].append(df_seg[vel_col].std())
    
    return results

def compute_deadband_and_gain(staircase_results):
    """Compute deadband and gain from staircase data."""
    commands = np.array(staircase_results['commands'])
    velocities = np.array(staircase_results['velocities'])
    
    # Find moving segments (velocity > threshold)
    threshold = 0.5  # steps/sec
    moving_idx = velocities > threshold
    
    if np.sum(moving_idx) < 2:
        return np.nan, np.nan, np.nan  # deadband, gain, r_squared
    
    slope, intercept, r2 = fit_linear(commands[moving_idx], velocities[moving_idx])
    
    if slope <= 0:
        return np.nan, np.nan, r2
    
    # Deadband = -intercept/slope
    deadband = -intercept / slope
    if deadband < 0:
        deadband = 0.0
    
    return deadband, slope, r2

def analyze_step_response(df, phase_num, motor):
    """Analyze step responses for a motor."""
    phase = PHASE_DEFINITIONS[phase_num]
    results = []
    
    for i, cmd in enumerate(STEP_COMMANDS):
        step_start = phase['start_ms'] + i * 6000  # 3s on + 3s off = 6s per cycle
        step_end = step_start + 3000  # 3s on
        
        mask = (df['ElapsedMs'] >= step_start) & (df['ElapsedMs'] < step_end)
        df_step = df[mask].copy()
        
        if len(df_step) < 20:
            continue
        
        t = (df_step['ElapsedMs'] - step_start) / 1000.0
        v = df_step[f'Vel{motor}'].values
        v_smooth = moving_average(v, 5)
        
        # Fit only the rising portion (first 80%)
        fit_end = int(0.8 * len(t))
        if fit_end > 20:
            v_ss, tau = fit_step_response(t[:fit_end], v_smooth[:fit_end])
            results.append({
                'command': cmd,
                'v_ss': v_ss,
                'tau': tau,
                'v_final': v_smooth[-1] if len(v_smooth)>0 else np.nan
            })
    
    return results

def analyze_dual_steps(df, phase_num):
    """Analyze dual motor synchronous steps."""
    phase = PHASE_DEFINITIONS[phase_num]
    results = []
    
    for i, cmd in enumerate(DUAL_COMMANDS):
        seg_start = phase['start_ms'] + i * 6000
        seg_end = seg_start + 6000
        mask = (df['ElapsedMs'] >= seg_start) & (df['ElapsedMs'] < seg_end)
        df_seg = df[mask]
        
        if len(df_seg) < 50:
            continue
        
        velL = df_seg['VelL'].median()
        velR = df_seg['VelR'].median()
        velL_std = df_seg['VelL'].std()
        velR_std = df_seg['VelR'].std()
        
        ratio = velL / velR if abs(velR) > 1e-6 else np.nan
        
        results.append({
            'command': cmd,
            'velL': velL,
            'velR': velR,
            'velL_std': velL_std,
            'velR_std': velR_std,
            'ratio': ratio
        })
    
    return results

def analyze_coast_down(df, phase_num):
    """Analyze coast-down behavior."""
    phase = PHASE_DEFINITIONS[phase_num]
    results = {'L': [], 'R': []}
    
    # Two coast-down segments:
    # 1. After forward drive (5s to 10s within phase)
    # 2. After reverse drive (15s to 20s within phase)
    coast_segments = [
        (phase['start_ms'] + 5000, phase['start_ms'] + 10000, 'forward'),
        (phase['start_ms'] + 15000, phase['start_ms'] + 20000, 'reverse')
    ]
    
    for motor in ['L', 'R']:
        for seg_start, seg_end, direction in coast_segments:
            mask = (df['ElapsedMs'] >= seg_start) & (df['ElapsedMs'] < seg_end)
            df_seg = df[mask]
            
            if len(df_seg) < 20:
                continue
            
            t = (df_seg['ElapsedMs'] - seg_start) / 1000.0
            v = df_seg[f'Vel{motor}'].values
            
            # Use absolute velocity for reverse coast
            if direction == 'reverse':
                v = np.abs(v)
            
            v_smooth = moving_average(v, 5)
            v0, tau = fit_coast_down(t, v_smooth)
            
            results[motor].append({
                'direction': direction,
                'v0': v0,
                'tau': tau
            })
    
    return results

# -------------------------------
# Plotting Functions
# -------------------------------
def plot_staircase(staircase_fwd, staircase_rev, motor, robot_id, plot_dir):
    """Plot staircase response."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Forward
    ax1.errorbar(staircase_fwd['commands'], staircase_fwd['velocities'],
                 yerr=staircase_fwd['velocities_std'], fmt='bo-', capsize=3)
    ax1.set_xlabel('Command (duty cycle)')
    ax1.set_ylabel('Velocity (steps/sec)')
    ax1.set_title(f'Forward Staircase - Robot {robot_id} Motor {motor}')
    ax1.grid(True)
    
    # Reverse
    ax2.errorbar(staircase_rev['commands'], staircase_rev['velocities'],
                 yerr=staircase_rev['velocities_std'], fmt='rs-', capsize=3)
    ax2.set_xlabel('Command (duty cycle)')
    ax2.set_ylabel('Velocity (steps/sec)')
    ax2.set_title(f'Reverse Staircase - Robot {robot_id} Motor {motor}')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'staircase_motor_{motor}.png'))
    plt.close()

def plot_step_responses(step_results, motor, robot_id, plot_dir):
    """Plot step responses."""
    plt.figure(figsize=(10, 6))
    
    # We need the raw data for plotting
    # This will be done separately in the main analysis
    plt.close()

def plot_dual_steps(dual_results, robot_id, plot_dir):
    """Plot dual motor steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    commands = [r['command'] for r in dual_results]
    velL = [r['velL'] for r in dual_results]
    velR = [r['velR'] for r in dual_results]
    ratios = [r['ratio'] for r in dual_results]
    
    ax1.plot(commands, velL, 'bo-', label='Left')
    ax1.plot(commands, velR, 'rs-', label='Right')
    ax1.set_xlabel('Command (duty cycle)')
    ax1.set_ylabel('Velocity (steps/sec)')
    ax1.set_title(f'Dual Motor Velocities - Robot {robot_id}')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(commands, ratios, 'go-')
    ax2.set_xlabel('Command (duty cycle)')
    ax2.set_ylabel('Left/Right Ratio')
    ax2.set_title(f'Asymmetry Ratio - Robot {robot_id}')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'dual_steps.png'))
    plt.close()

def plot_coast_down(df, phase_num, coast_results, robot_id, plot_dir):
    """Plot coast-down curves."""
    phase = PHASE_DEFINITIONS[phase_num]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # First coast-down (after forward)
    seg_start = phase['start_ms'] + 5000
    seg_end = phase['start_ms'] + 10000
    mask = (df['ElapsedMs'] >= seg_start) & (df['ElapsedMs'] < seg_end)
    df_seg = df[mask]
    t = (df_seg['ElapsedMs'] - seg_start) / 1000.0
    ax1.plot(t, df_seg['VelL'], label='Left')
    ax1.plot(t, df_seg['VelR'], label='Right')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (steps/sec)')
    ax1.set_title('Coast-down after Forward')
    ax1.legend()
    ax1.grid(True)
    
    # Second coast-down (after reverse)
    seg_start = phase['start_ms'] + 15000
    seg_end = phase['start_ms'] + 20000
    mask = (df['ElapsedMs'] >= seg_start) & (df['ElapsedMs'] < seg_end)
    df_seg = df[mask]
    t = (df_seg['ElapsedMs'] - seg_start) / 1000.0
    ax2.plot(t, np.abs(df_seg['VelL']), label='Left (abs)')
    ax2.plot(t, np.abs(df_seg['VelR']), label='Right (abs)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (steps/sec)')
    ax2.set_title('Coast-down after Reverse')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'coast_down.png'))
    plt.close()

def plot_full_run(df, robot_id, plot_dir):
    """Plot full run with phase boundaries."""
    plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['ElapsedMs']/1000.0, df['StepsL'], label='Left steps')
    plt.plot(df['ElapsedMs']/1000.0, df['StepsR'], label='Right steps')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative steps')
    plt.title(f'Robot {robot_id} - Full Run')
    plt.legend()
    plt.grid(True)
    
    # Add phase boundaries
    for phase_num, phase in PHASE_DEFINITIONS.items():
        plt.axvline(phase['start_ms']/1000.0, color='gray', linestyle='--', alpha=0.5)
        if phase_num < 9:
            plt.text(phase['start_ms']/1000.0 + 0.5, plt.ylim()[1]*0.9, 
                     f"P{phase_num}", fontsize=8)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['ElapsedMs']/1000.0, df['VelL'], label='Left velocity')
    plt.plot(df['ElapsedMs']/1000.0, df['VelR'], label='Right velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (steps/sec)')
    plt.legend()
    plt.grid(True)
    
    for phase_num, phase in PHASE_DEFINITIONS.items():
        plt.axvline(phase['start_ms']/1000.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'full_run.png'))
    plt.close()

# -------------------------------
# Main Analysis per Robot
# -------------------------------
def analyze_robot(robot_id, filepath):
    print(f"\n{'='*80}")
    print(f"Analyzing Robot {robot_id}: {filepath}")
    print(f"{'='*80}")
    
    df = load_sysid_file(filepath)
    df = compute_velocities(df)
    
    plot_dir = f"plots_robot{robot_id}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Print encoder info
    print(f"\nEncoder resolution: {ENCODER_RESOLUTION} counts/rev")
    print(f"Degrees per step: {DEG_PER_STEP:.5f} deg")
    print(f"Radians per step: {RAD_PER_STEP:.6f} rad")
    
    results = {}
    
    # ==============================
    # Phase 1: Left Forward Staircase
    # ==============================
    print(f"\n--- Phase 1: Left Motor Forward Staircase ---")
    staircase_L_fwd = analyze_staircase(df, 1, 'L', 'forward')
    db_L_fwd, gain_L_fwd, r2_L_fwd = compute_deadband_and_gain(staircase_L_fwd)
    results['L_fwd'] = {'deadband': db_L_fwd, 'gain': gain_L_fwd, 'r2': r2_L_fwd}
    print(f"Deadband: {db_L_fwd:.4f} (duty cycle)" if not np.isnan(db_L_fwd) else "Deadband: N/A")
    print(f"Gain: {gain_L_fwd:.2f} steps/s per duty" if not np.isnan(gain_L_fwd) else "Gain: N/A")
    print(f"R²: {r2_L_fwd:.4f}" if not np.isnan(r2_L_fwd) else "R²: N/A")
    
    # ==============================
    # Phase 2: Left Reverse Staircase
    # ==============================
    print(f"\n--- Phase 2: Left Motor Reverse Staircase ---")
    staircase_L_rev = analyze_staircase(df, 2, 'L', 'reverse')
    db_L_rev, gain_L_rev, r2_L_rev = compute_deadband_and_gain(staircase_L_rev)
    results['L_rev'] = {'deadband': db_L_rev, 'gain': gain_L_rev, 'r2': r2_L_rev}
    print(f"Deadband: {db_L_rev:.4f} (duty cycle)" if not np.isnan(db_L_rev) else "Deadband: N/A")
    print(f"Gain: {gain_L_rev:.2f} steps/s per duty" if not np.isnan(gain_L_rev) else "Gain: N/A")
    print(f"R²: {r2_L_rev:.4f}" if not np.isnan(r2_L_rev) else "R²: N/A")
    
    plot_staircase(staircase_L_fwd, staircase_L_rev, 'L', robot_id, plot_dir)
    
    # ==============================
    # Phase 3: Left Step Response
    # ==============================
    print(f"\n--- Phase 3: Left Motor Step Response ---")
    step_L = analyze_step_response(df, 3, 'L')
    results['L_steps'] = step_L
    for i, step in enumerate(step_L):
        print(f"Step {step['command']:.2f}: v_ss={step['v_ss']:.2f} steps/s, tau={step['tau']:.3f} s")
    
    # Plot step responses for left motor
    fig, ax = plt.subplots(figsize=(10, 6))
    phase = PHASE_DEFINITIONS[3]
    for i, cmd in enumerate(STEP_COMMANDS):
        step_start = phase['start_ms'] + i * 6000
        step_end = step_start + 3000
        mask = (df['ElapsedMs'] >= step_start) & (df['ElapsedMs'] < step_end)
        df_step = df[mask]
        t = (df_step['ElapsedMs'] - step_start) / 1000.0
        ax.plot(t, df_step['VelL'], label=f'Step {cmd}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (steps/sec)')
    ax.set_title(f'Left Motor Step Responses - Robot {robot_id}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'left_step_responses.png'))
    plt.close()
    
    # ==============================
    # Phase 4: Right Forward Staircase
    # ==============================
    print(f"\n--- Phase 4: Right Motor Forward Staircase ---")
    staircase_R_fwd = analyze_staircase(df, 4, 'R', 'forward')
    db_R_fwd, gain_R_fwd, r2_R_fwd = compute_deadband_and_gain(staircase_R_fwd)
    results['R_fwd'] = {'deadband': db_R_fwd, 'gain': gain_R_fwd, 'r2': r2_R_fwd}
    print(f"Deadband: {db_R_fwd:.4f} (duty cycle)" if not np.isnan(db_R_fwd) else "Deadband: N/A")
    print(f"Gain: {gain_R_fwd:.2f} steps/s per duty" if not np.isnan(gain_R_fwd) else "Gain: N/A")
    print(f"R²: {r2_R_fwd:.4f}" if not np.isnan(r2_R_fwd) else "R²: N/A")
    
    # ==============================
    # Phase 5: Right Reverse Staircase
    # ==============================
    print(f"\n--- Phase 5: Right Motor Reverse Staircase ---")
    staircase_R_rev = analyze_staircase(df, 5, 'R', 'reverse')
    db_R_rev, gain_R_rev, r2_R_rev = compute_deadband_and_gain(staircase_R_rev)
    results['R_rev'] = {'deadband': db_R_rev, 'gain': gain_R_rev, 'r2': r2_R_rev}
    print(f"Deadband: {db_R_rev:.4f} (duty cycle)" if not np.isnan(db_R_rev) else "Deadband: N/A")
    print(f"Gain: {gain_R_rev:.2f} steps/s per duty" if not np.isnan(gain_R_rev) else "Gain: N/A")
    print(f"R²: {r2_R_rev:.4f}" if not np.isnan(r2_R_rev) else "R²: N/A")
    
    plot_staircase(staircase_R_fwd, staircase_R_rev, 'R', robot_id, plot_dir)
    
    # ==============================
    # Phase 6: Right Step Response
    # ==============================
    print(f"\n--- Phase 6: Right Motor Step Response ---")
    step_R = analyze_step_response(df, 6, 'R')
    results['R_steps'] = step_R
    for i, step in enumerate(step_R):
        print(f"Step {step['command']:.2f}: v_ss={step['v_ss']:.2f} steps/s, tau={step['tau']:.3f} s")
    
    # Plot step responses for right motor
    fig, ax = plt.subplots(figsize=(10, 6))
    phase = PHASE_DEFINITIONS[6]
    for i, cmd in enumerate(STEP_COMMANDS):
        step_start = phase['start_ms'] + i * 6000
        step_end = step_start + 3000
        mask = (df['ElapsedMs'] >= step_start) & (df['ElapsedMs'] < step_end)
        df_step = df[mask]
        t = (df_step['ElapsedMs'] - step_start) / 1000.0
        ax.plot(t, df_step['VelR'], label=f'Step {cmd}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (steps/sec)')
    ax.set_title(f'Right Motor Step Responses - Robot {robot_id}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'right_step_responses.png'))
    plt.close()
    
    # ==============================
    # Phase 7: Dual Motor Synchronous Steps
    # ==============================
    print(f"\n--- Phase 7: Dual Motor Synchronous Steps ---")
    dual_results = analyze_dual_steps(df, 7)
    results['dual'] = dual_results
    for res in dual_results:
        print(f"Command {res['command']:.2f}: VelL={res['velL']:.2f}, VelR={res['velR']:.2f}, "
              f"Ratio={res['ratio']:.3f}")
    
    plot_dual_steps(dual_results, robot_id, plot_dir)
    
    # ==============================
    # Phase 8: Coast-Down
    # ==============================
    print(f"\n--- Phase 8: Coast-Down ---")
    coast_results = analyze_coast_down(df, 8)
    results['coast'] = coast_results
    for motor in ['L', 'R']:
        for res in coast_results[motor]:
            print(f"Motor {motor} ({res['direction']}): v0={res['v0']:.2f} steps/s, "
                  f"tau={res['tau']:.3f} s")
    
    plot_coast_down(df, 8, coast_results, robot_id, plot_dir)
    
    # ==============================
    # Full Run Plot
    # ==============================
    plot_full_run(df, robot_id, plot_dir)
    
    return results

# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == "__main__":
    all_results = {}
    
    for robot_id in range(1, 5):
        filepath = f"sysid_r{robot_id}.csv"
        if os.path.exists(filepath):
            all_results[robot_id] = analyze_robot(robot_id, filepath)
        else:
            print(f"File {filepath} not found, skipping Robot {robot_id}")
    
    # Generate summary report
    summary_file = "sysid_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Actuator SysID Summary (Phase-Aware Analysis)\n")
        f.write("==============================================\n\n")
        
        for robot_id, results in all_results.items():
            f.write(f"Robot {robot_id}:\n")
            
            # Left motor
            f.write("  Left Motor:\n")
            f.write(f"    Forward: Deadband={results['L_fwd']['deadband']:.4f}, "
                   f"Gain={results['L_fwd']['gain']:.2f}\n")
            f.write(f"    Reverse: Deadband={results['L_rev']['deadband']:.4f}, "
                   f"Gain={results['L_rev']['gain']:.2f}\n")
            
            # Right motor
            f.write("  Right Motor:\n")
            f.write(f"    Forward: Deadband={results['R_fwd']['deadband']:.4f}, "
                   f"Gain={results['R_fwd']['gain']:.2f}\n")
            f.write(f"    Reverse: Deadband={results['R_rev']['deadband']:.4f}, "
                   f"Gain={results['R_rev']['gain']:.2f}\n")
            
            # Step response time constants
            if results['L_steps']:
                avg_tau_L = np.mean([s['tau'] for s in results['L_steps']])
                f.write(f"    Left avg tau: {avg_tau_L:.3f} s\n")
            if results['R_steps']:
                avg_tau_R = np.mean([s['tau'] for s in results['R_steps']])
                f.write(f"    Right avg tau: {avg_tau_R:.3f} s\n")
            
            # Asymmetry
            if results['dual']:
                avg_ratio = np.mean([r['ratio'] for r in results['dual']])
                f.write(f"    Asymmetry ratio: {avg_ratio:.3f}\n")
            
            f.write("\n")
    
    print(f"\nSummary written to {summary_file}")
