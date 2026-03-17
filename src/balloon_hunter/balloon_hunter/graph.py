#!/usr/bin/env python3
"""
DKF vs EKF Comparison Analysis v2
===================================
v2 Changes:
  - Velocity and angular velocity columns properly handled
  - GT values clamped in logger, so no more 1M pixel spikes
  - Better axis scaling (auto-ylim based on 99th percentile)
  - Speed graph uses actual velocity data

Usage:
  python3 graph_v2.py                          # auto-find latest pair
  python3 graph_v2.py log_DKF.csv log_EKF.csv  # manual
"""

import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 130
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def load(path):
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def find_logs():
    d = os.path.expanduser('~/dkf_logs')
    dkf = sorted(glob.glob(os.path.join(d, 'log_*_DKF.csv')))
    ekf = sorted(glob.glob(os.path.join(d, 'log_*_EKF.csv')))
    if not dkf:
        print('No DKF log found!')
        sys.exit(1)
    if not ekf:
        print('No EKF log found!')
        sys.exit(1)
    return dkf[-1], ekf[-1]


def active_region(df):
    mask = df['u_gt'].notna() & df['v_gt'].notna()
    if mask.sum() == 0:
        return df
    t0 = df.loc[mask, 'timestamp_s'].iloc[0] - 0.5
    t1 = df.loc[mask, 'timestamp_s'].iloc[-1] + 0.5
    return df[(df['timestamp_s'] >= t0) & (df['timestamp_s'] <= t1)].copy()


def smart_ylim(ax, data_list, margin=0.1):
    """Set ylim based on 1st-99th percentile to avoid outlier spikes."""
    all_vals = []
    for d in data_list:
        clean = d[np.isfinite(d)]
        if len(clean) > 0:
            all_vals.extend(clean)
    if len(all_vals) == 0:
        return
    arr = np.array(all_vals)
    lo = np.percentile(arr, 1)
    hi = np.percentile(arr, 99)
    rng = hi - lo
    ax.set_ylim(lo - rng * margin, hi + rng * margin)


def graph1_coordinates(dkf, ekf, out):
    """Image coordinate (u-axis) time series comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # DKF
    ax = axes[0]
    ax.set_title('DKF Experiment: Image Coordinate (u-axis)', fontsize=13)
    ax.scatter(dkf['timestamp_s'].values, dkf['u_yolo'].values, s=6, alpha=0.3, color='gray', label='YOLO raw')
    ax.plot(dkf['timestamp_s'].values, dkf['u_gt'].values, linewidth=1.5, color='#2ecc71', linestyle='--', label='Ground Truth')
    ax.plot(dkf['timestamp_s'].values, dkf['u_filt'].values, linewidth=1.2, color='#2980b9', label='DKF estimate')
    ax.axhline(y=424, color='orange', linestyle=':', alpha=0.5, label='Image center')
    ax.set_ylabel('u (pixels)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    smart_ylim(ax, [dkf['u_yolo'].values, dkf['u_gt'].values, dkf['u_filt'].values])

    # EKF
    ax = axes[1]
    ax.set_title('EKF Experiment: Image Coordinate (u-axis)', fontsize=13)
    ax.scatter(ekf['timestamp_s'].values, ekf['u_yolo'].values, s=6, alpha=0.3, color='gray', label='YOLO raw')
    ax.plot(ekf['timestamp_s'].values, ekf['u_gt'].values, linewidth=1.5, color='#2ecc71', linestyle='--', label='Ground Truth')
    ax.plot(ekf['timestamp_s'].values, ekf['u_filt'].values, linewidth=1.2, color='#e74c3c', label='EKF estimate')
    ax.axhline(y=424, color='orange', linestyle=':', alpha=0.5, label='Image center')
    ax.set_ylabel('u (pixels)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    smart_ylim(ax, [ekf['u_yolo'].values, ekf['u_gt'].values, ekf['u_filt'].values])

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


def graph2_error(dkf, ekf, out):
    """Estimation error comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    w = 25  # 0.5s rolling average at 50Hz

    ax = axes[0]
    ax.set_title('Estimation Error: Distance to Ground Truth (Smoothed)', fontsize=13)
    dkf_smooth = dkf['err_filt_px'].rolling(w, min_periods=1).mean().values
    ekf_smooth = ekf['err_filt_px'].rolling(w, min_periods=1).mean().values
    ax.plot(dkf['timestamp_s'].values, dkf_smooth, linewidth=2, color='#2980b9', label='DKF (0.5s avg)')
    ax.plot(ekf['timestamp_s'].values, ekf_smooth, linewidth=2, color='#e74c3c', label='EKF (0.5s avg)')
    ax.set_ylabel('Error (pixels)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    smart_ylim(ax, [dkf_smooth, ekf_smooth])
    # Ensure bottom is 0
    yl = ax.get_ylim()
    ax.set_ylim(0, yl[1])

    ax = axes[1]
    ax.set_title('Raw Estimation Error', fontsize=13)
    ax.plot(dkf['timestamp_s'].values, dkf['err_filt_px'].values, linewidth=0.8, alpha=0.5, color='#2980b9', label='DKF')
    ax.plot(ekf['timestamp_s'].values, ekf['err_filt_px'].values, linewidth=0.8, alpha=0.5, color='#e74c3c', label='EKF')
    ax.set_ylabel('Error (pixels)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    smart_ylim(ax, [dkf['err_filt_px'].values, ekf['err_filt_px'].values])
    yl = ax.get_ylim()
    ax.set_ylim(0, yl[1])

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


def graph3_control(dkf, ekf, out):
    """Control performance: ex convergence, yaw, speed."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # ex
    ax = axes[0]
    ax.set_title('Control Performance: ex (horizontal distance from image center)', fontsize=13)
    ax.plot(dkf['timestamp_s'].values, dkf['ex_filt'].values, linewidth=1.2, color='#2980b9', label='DKF', alpha=0.8)
    ax.plot(ekf['timestamp_s'].values, ekf['ex_filt'].values, linewidth=1.2, color='#e74c3c', label='EKF', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    ax.axhspan(-20, 20, alpha=0.1, color='green', label='±20px Tolerance')
    ax.set_ylabel('ex (pixels)')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Yaw
    ax = axes[1]
    ax.set_title('Drone Yaw Angle', fontsize=13)
    ax.plot(dkf['timestamp_s'].values, np.degrees(dkf['drone_yaw'].values), linewidth=1.5, color='#2980b9', label='DKF exp')
    ax.plot(ekf['timestamp_s'].values, np.degrees(ekf['drone_yaw'].values), linewidth=1.5, color='#e74c3c', label='EKF exp', linestyle='--')
    ax.set_ylabel('Yaw (deg)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Speed
    ax = axes[2]
    ax.set_title('Drone Speed', fontsize=13)
    dkf_speed = np.sqrt(dkf['drone_vx'].values**2 + dkf['drone_vy'].values**2 + dkf['drone_vz'].values**2)
    ekf_speed = np.sqrt(ekf['drone_vx'].values**2 + ekf['drone_vy'].values**2 + ekf['drone_vz'].values**2)
    ax.plot(dkf['timestamp_s'].values, dkf_speed, linewidth=1.5, color='#2980b9', label='DKF exp')
    ax.plot(ekf['timestamp_s'].values, ekf_speed, linewidth=1.5, color='#e74c3c', label='EKF exp', linestyle='--')
    ax.set_ylabel('Speed (m/s)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


def graph4_summary(dkf, ekf, out):
    """Summary statistics bar chart."""
    def get_stats(df, name):
        v = df['err_filt_px'].dropna()
        ex = df['ex_filt'].dropna().abs()
        conv_mask = (ex < 20).rolling(25, min_periods=25).sum() == 25
        ci = conv_mask[conv_mask].index
        ct = df.loc[ci[0], 'timestamp_s'] - df['timestamp_s'].iloc[0] if len(ci) > 0 else float('nan')
        return {
            'name': name, 'mean': v.mean(), 'median': v.median(),
            'std': v.std(), 'max': v.max(), 'p90': v.quantile(0.9), 'conv_t': ct
        }

    sd = get_stats(dkf, 'DKF')
    se = get_stats(ekf, 'EKF')

    labels = ['Mean\nError', 'Median\nError', 'Std Dev', 'Max\nError', '90th\n%ile', 'Converge\nTime (s)']
    dv = [sd['mean'], sd['median'], sd['std'], sd['max'], sd['p90'], sd['conv_t']]
    ev = [se['mean'], se['median'], se['std'], se['max'], se['p90'], se['conv_t']]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, ev, w, label='EKF', color='#e74c3c', alpha=0.8)
    b2 = ax.bar(x + w/2, dv, w, label='DKF', color='#2980b9', alpha=0.8)

    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x() + b.get_width()/2, h + 0.2, f'{h:.1f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=12)
    ax.set_title('DKF vs EKF: Performance Summary Statistics', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    # Improvement
    if se['mean'] > 0:
        imp = (se['mean'] - sd['mean']) / se['mean'] * 100
        color = 'green' if imp > 0 else 'red'
        ax.text(0.02, 0.95, f'DKF Improvement: {imp:+.1f}% (Mean Error)',
                transform=ax.transAxes, fontsize=13, fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # YOLO Hz
    hz_d = dkf['yolo_hz'].iloc[-1] if 'yolo_hz' in dkf.columns else 0
    hz_e = ekf['yolo_hz'].iloc[-1] if 'yolo_hz' in ekf.columns else 0
    ax.text(0.98, 0.95, f'YOLO: ~{(hz_d + hz_e)/2:.0f} Hz\nController: 50 Hz',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


def text_summary(dkf, ekf):
    print('\n' + '='*60)
    print('        DKF vs EKF Performance Comparison Results')
    print('='*60)
    for name, df in [('DKF', dkf), ('EKF', ekf)]:
        v = df['err_filt_px'].dropna()
        hz = df['yolo_hz'].iloc[-1] if 'yolo_hz' in df.columns else 0
        speed = np.sqrt(df['drone_vx']**2 + df['drone_vy']**2 + df['drone_vz']**2)
        print(f'\n  [{name}]  ({len(df)} samples, YOLO {hz:.1f}Hz)')
        print(f'    Mean Error:      {v.mean():.2f} px')
        print(f'    Median Error:    {v.median():.2f} px')
        print(f'    90th Percentile: {v.quantile(0.9):.2f} px')
        print(f'    Max Error:       {v.max():.2f} px')
        print(f'    Avg Speed:       {speed.mean():.2f} m/s')
        print(f'    Max AngVel:      {df["omega_z"].abs().max():.2f} rad/s')

    md = dkf['err_filt_px'].dropna().mean()
    me = ekf['err_filt_px'].dropna().mean()
    if me > 0:
        print(f'\n  >>> DKF Improvement: {(me - md)/me*100:+.1f}% <<<')
    print('='*60)


def main():
    if len(sys.argv) >= 3:
        dkf_path, ekf_path = sys.argv[1], sys.argv[2]
    else:
        dkf_path, ekf_path = find_logs()
        print(f'Using DKF: {dkf_path}')
        print(f'Using EKF: {ekf_path}')

    dkf = active_region(load(dkf_path))
    ekf = active_region(load(ekf_path))
    print(f'DKF: {len(dkf)} rows, EKF: {len(ekf)} rows')

    out_dir = os.path.dirname(dkf_path)
    base = 'comparison'

    print('\nGenerating graphs...')
    graph1_coordinates(dkf, ekf, os.path.join(out_dir, f'{base}_1_coordinates.png'))
    graph2_error(dkf, ekf, os.path.join(out_dir, f'{base}_2_error.png'))
    graph3_control(dkf, ekf, os.path.join(out_dir, f'{base}_3_control.png'))
    graph4_summary(dkf, ekf, os.path.join(out_dir, f'{base}_4_summary.png'))

    text_summary(dkf, ekf)
    print(f'\nAll outputs in: {out_dir}/')


if __name__ == '__main__':
    main()