#!/usr/bin/env python3
"""
DKF vs EKF Performance Analysis
=================================
Reads CSV log from dkf_logger.py and generates comparison graphs.

Usage:
  python3 analyze_dkf_vs_ekf.py ~/dkf_logs/log_YYYYMMDD_HHMMSS.csv
  python3 analyze_dkf_vs_ekf.py  (auto-selects latest log)

Generates 4 graphs:
  1. Image coordinate time series (u axis): YOLO raw, EKF, DKF, Ground Truth
  2. Estimation error (pixels) over time: EKF vs DKF
  3. Control performance: ex (image center error) convergence
  4. Summary statistics bar chart
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 120


def load_csv(path):
    """Load and clean CSV log."""
    df = pd.read_csv(path)
    # Replace 'nan' strings with actual NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def find_active_region(df, min_err_col='err_dkf_px'):
    """Find time region where target is visible (non-NaN ground truth)."""
    mask = df['u_gt'].notna() & df['v_gt'].notna()
    if mask.sum() == 0:
        return df
    first = df.loc[mask, 'timestamp_s'].iloc[0]
    last = df.loc[mask, 'timestamp_s'].iloc[-1]
    # Add 0.5s padding
    return df[(df['timestamp_s'] >= first - 0.5) & (df['timestamp_s'] <= last + 0.5)]


def plot_image_coordinates(df, save_path):
    """
    Graph 1: Image coordinate (u-axis) time series.
    Shows YOLO raw, EKF estimate, DKF estimate, and Ground Truth.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    t = df['timestamp_s']

    # U axis
    ax = axes[0]
    ax.scatter(t, df['u_yolo'], s=8, alpha=0.4, color='gray', label='YOLO raw', zorder=2)
    ax.plot(t, df['u_ekf'], linewidth=1.2, color='#e74c3c', label='EKF', alpha=0.8)
    ax.plot(t, df['u_dkf'], linewidth=1.2, color='#2980b9', label='DKF')
    ax.plot(t, df['u_gt'], linewidth=1.5, color='#2ecc71', linestyle='--', label='Ground Truth')
    ax.axhline(y=424, color='orange', linestyle=':', alpha=0.5, label='Image center (cx=424)')
    ax.set_ylabel('u (pixels)')
    ax.set_title('Image Coordinate Estimation: u-axis')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # V axis
    ax = axes[1]
    ax.scatter(t, df['v_yolo'], s=8, alpha=0.4, color='gray', label='YOLO raw', zorder=2)
    ax.plot(t, df['v_ekf'], linewidth=1.2, color='#e74c3c', label='EKF', alpha=0.8)
    ax.plot(t, df['v_dkf'], linewidth=1.2, color='#2980b9', label='DKF')
    ax.plot(t, df['v_gt'], linewidth=1.5, color='#2ecc71', linestyle='--', label='Ground Truth')
    ax.axhline(y=240, color='orange', linestyle=':', alpha=0.5, label='Image center (cy=240)')
    ax.set_ylabel('v (pixels)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Image Coordinate Estimation: v-axis')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_estimation_error(df, save_path):
    """
    Graph 2: Estimation error (pixels) over time.
    Shows Euclidean distance from estimate to ground truth for both filters.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    t = df['timestamp_s']

    # Time series
    ax = axes[0]
    ax.plot(t, df['err_ekf_px'], linewidth=1.2, color='#e74c3c', label='EKF error', alpha=0.8)
    ax.plot(t, df['err_dkf_px'], linewidth=1.2, color='#2980b9', label='DKF error')
    ax.set_ylabel('Error (pixels)')
    ax.set_title('Estimation Error: Distance to Ground Truth')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Rolling average
    ax = axes[1]
    window = 25  # 0.5s at 50Hz
    err_ekf_smooth = df['err_ekf_px'].rolling(window, min_periods=1).mean()
    err_dkf_smooth = df['err_dkf_px'].rolling(window, min_periods=1).mean()
    ax.plot(t, err_ekf_smooth, linewidth=2, color='#e74c3c', label=f'EKF (rolling avg {window})')
    ax.plot(t, err_dkf_smooth, linewidth=2, color='#2980b9', label=f'DKF (rolling avg {window})')
    ax.fill_between(t, err_ekf_smooth, err_dkf_smooth, alpha=0.15, color='green',
                     where=err_ekf_smooth > err_dkf_smooth, label='DKF advantage')
    ax.set_ylabel('Error (pixels)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Smoothed Estimation Error (0.5s rolling average)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_control_performance(df, save_path):
    """
    Graph 3: Control performance - image center error (ex) convergence.
    Shows how quickly each filter brings the target to image center.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    t = df['timestamp_s']

    # ex (horizontal error from image center)
    ax = axes[0]
    ax.plot(t, df['ex_ekf'], linewidth=1.2, color='#e74c3c', label='ex (EKF)', alpha=0.8)
    ax.plot(t, df['ex_dkf'], linewidth=1.2, color='#2980b9', label='ex (DKF)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axhspan(-20, 20, alpha=0.1, color='green', label='±20px tolerance')
    ax.set_ylabel('ex (pixels)')
    ax.set_title('Control Performance: Horizontal Image Error (ex = u - cx)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Drone yaw
    ax = axes[1]
    ax.plot(t, np.degrees(df['drone_yaw']), linewidth=1.5, color='#8e44ad', label='Yaw')
    ax.set_ylabel('Yaw (degrees)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Drone Yaw Angle')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_summary_stats(df, save_path):
    """
    Graph 4: Summary statistics bar chart.
    Mean error, median error, max error, convergence time.
    """
    # Compute stats (only where GT is valid)
    valid = df.dropna(subset=['err_ekf_px', 'err_dkf_px'])

    stats = {
        'Mean Error (px)': [valid['err_ekf_px'].mean(), valid['err_dkf_px'].mean()],
        'Median Error (px)': [valid['err_ekf_px'].median(), valid['err_dkf_px'].median()],
        'Std Dev (px)': [valid['err_ekf_px'].std(), valid['err_dkf_px'].std()],
        'Max Error (px)': [valid['err_ekf_px'].max(), valid['err_dkf_px'].max()],
        '90th %ile (px)': [valid['err_ekf_px'].quantile(0.9), valid['err_dkf_px'].quantile(0.9)],
    }

    # Convergence time: first time |ex| < 20px for 0.5s continuously
    for label, col in [('EKF', 'ex_ekf'), ('DKF', 'ex_dkf')]:
        s = valid[col].abs()
        converged = (s < 20).rolling(25, min_periods=25).sum() == 25
        first_converge = converged[converged].index
        if len(first_converge) > 0:
            t_conv = valid.loc[first_converge[0], 'timestamp_s']
        else:
            t_conv = float('nan')
        stats[f'Convergence Time (s)'] = stats.get('Convergence Time (s)', [])
        stats['Convergence Time (s)'].append(t_conv)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = list(stats.keys())
    ekf_vals = [stats[k][0] for k in labels]
    dkf_vals = [stats[k][1] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, ekf_vals, width, label='EKF', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, dkf_vals, width, label='DKF', color='#2980b9', alpha=0.8)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Value')
    ax.set_title('DKF vs EKF: Summary Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # YOLO Hz info
    yolo_hz = df['yolo_hz'].iloc[-1] if 'yolo_hz' in df.columns else 0
    ax.text(0.98, 0.98, f'YOLO: {yolo_hz:.1f} Hz\nController: 50 Hz',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Improvement percentage
    mean_ekf = stats['Mean Error (px)'][0]
    mean_dkf = stats['Mean Error (px)'][1]
    if mean_ekf > 0:
        improvement = (mean_ekf - mean_dkf) / mean_ekf * 100
        ax.text(0.02, 0.98, f'DKF improvement: {improvement:+.1f}%',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=12, fontweight='bold',
                color='green' if improvement > 0 else 'red',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def print_text_summary(df):
    """Print text summary to console."""
    valid = df.dropna(subset=['err_ekf_px', 'err_dkf_px'])
    if len(valid) == 0:
        print('  No valid data with ground truth!')
        return

    print('\n' + '='*60)
    print('  DKF vs EKF Performance Summary')
    print('='*60)

    yolo_hz = df['yolo_hz'].iloc[-1] if 'yolo_hz' in df.columns else 0
    print(f'  YOLO Detection Rate:    {yolo_hz:.1f} Hz')
    print(f'  Controller Rate:        50 Hz')
    print(f'  Total logged samples:   {len(df)}')
    print(f'  Valid GT samples:       {len(valid)}')
    print(f'  Duration:               {df["timestamp_s"].iloc[-1]:.1f} s')
    print()

    for name, col in [('EKF', 'err_ekf_px'), ('DKF', 'err_dkf_px')]:
        s = valid[col]
        print(f'  {name}:')
        print(f'    Mean error:     {s.mean():.2f} px')
        print(f'    Median error:   {s.median():.2f} px')
        print(f'    Std dev:        {s.std():.2f} px')
        print(f'    Max error:      {s.max():.2f} px')
        print(f'    90th %%ile:      {s.quantile(0.9):.2f} px')
        print()

    mean_ekf = valid['err_ekf_px'].mean()
    mean_dkf = valid['err_dkf_px'].mean()
    if mean_ekf > 0:
        pct = (mean_ekf - mean_dkf) / mean_ekf * 100
        print(f'  DKF vs EKF improvement: {pct:+.1f}%')
    print('='*60)


def main():
    # Find CSV file
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Auto-find latest log
        log_dir = os.path.expanduser('~/dkf_logs')
        files = sorted(glob.glob(os.path.join(log_dir, 'log_*.csv')))
        if not files:
            print(f'No log files found in {log_dir}')
            print('Usage: python3 analyze_dkf_vs_ekf.py <path_to_csv>')
            sys.exit(1)
        csv_path = files[-1]
        print(f'Auto-selected latest log: {csv_path}')

    if not os.path.exists(csv_path):
        print(f'File not found: {csv_path}')
        sys.exit(1)

    # Load
    print(f'Loading: {csv_path}')
    df = load_csv(csv_path)
    print(f'  Loaded {len(df)} rows')

    # Focus on active region
    df_active = find_active_region(df)
    print(f'  Active region: {len(df_active)} rows')

    # Output directory
    out_dir = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]

    # Generate graphs
    print('\nGenerating graphs...')
    plot_image_coordinates(df_active, os.path.join(out_dir, f'{base}_1_coordinates.png'))
    plot_estimation_error(df_active, os.path.join(out_dir, f'{base}_2_error.png'))
    plot_control_performance(df_active, os.path.join(out_dir, f'{base}_3_control.png'))
    plot_summary_stats(df_active, os.path.join(out_dir, f'{base}_4_summary.png'))

    # Text summary
    print_text_summary(df_active)

    print(f'\nAll outputs saved to: {out_dir}/')


if __name__ == '__main__':
    main()