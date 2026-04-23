#!/usr/bin/env python3
"""
DKF / EKF / GT Performance Analysis
=====================================
Loads CSV logs and generates:
  1. 3D trajectory + target position
  2. Image coordinate (u-axis) time series
  3. Estimation error comparison
  4. Control performance (ex, yaw, speed, distance)
  5. Summary statistics (CEP, mean error, convergence time, etc.)

Usage:
  python3 graph.py                              # auto-find latest logs
  python3 graph.py log_DKF.csv log_EKF.csv      # explicit two-file compare
  python3 graph.py log_GT.csv                    # single file analysis
"""

import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.rcParams['figure.dpi'] = 130
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Color map for filter types
COLORS = {
    'DKF': '#2980b9',
    'EKF': '#e74c3c',
    'GT':  '#27ae60',
}


def load(path):
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in ('filter_type', 'mission_state'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def find_logs():
    """Find all available log files (DKF, EKF, GT) — return dict of type→path."""
    d = os.path.expanduser('~/dkf_logs')
    result = {}
    for ftype in ('DKF', 'EKF', 'GT'):
        files = sorted(glob.glob(os.path.join(d, f'log_*_{ftype}.csv')))
        if files:
            result[ftype] = files[-1]
    return result


def active_region(df):
    """Extract region where target is visible (GT pixel valid)."""
    mask = df['u_gt'].notna() & df['v_gt'].notna()
    if mask.sum() == 0:
        return df
    t0 = df.loc[mask, 'timestamp_s'].iloc[0] - 0.5
    t1 = df.loc[mask, 'timestamp_s'].iloc[-1] + 0.5
    return df[(df['timestamp_s'] >= t0) & (df['timestamp_s'] <= t1)].copy()


def get_filter_type(df):
    """Get filter type from the first valid row."""
    if 'filter_type' in df.columns:
        vals = df['filter_type'].dropna().unique()
        # Could be numeric if coerced, check string column
        for v in vals:
            if isinstance(v, str) and v in ('DKF', 'EKF', 'GT'):
                return v
    return 'UNKNOWN'


# ═══════════════════════════════════════════════════════════════
#  Graph 1: 3D Trajectory + Target Position
# ═══════════════════════════════════════════════════════════════
def graph_3d_trajectory(datasets, out):
    """3D plot: drone trajectory (NED→ENU for display) + target position."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Drone Trajectory & Target Position', fontsize=14)

    for name, df in datasets.items():
        color = COLORS.get(name, 'gray')
        # NED to ENU for display: east=y_ned, north=x_ned, up=-z_ned
        x_enu = df['drone_y'].values     # East
        y_enu = df['drone_x'].values     # North
        z_enu = -df['drone_z'].values    # Up
        ax.plot(x_enu, y_enu, z_enu, linewidth=1.5, color=color, label=f'{name} trajectory', alpha=0.8)
        # Start marker
        ax.scatter(x_enu[0], y_enu[0], z_enu[0], s=80, color=color, marker='^', zorder=5)
        # End marker
        ax.scatter(x_enu[-1], y_enu[-1], z_enu[-1], s=80, color=color, marker='v', zorder=5)

    # Target position (from first dataset that has it)
    # target_world is Gazebo ENU: X=East, Y=North, Z=Up
    # Display axes: East=X, North=Y, Up=Z
    BALLOON_RADIUS = 0.3  # metres — from model.sdf sphere radius
    for name, df in datasets.items():
        if 'target_x' in df.columns and df['target_x'].notna().any():
            t_east  = df['target_x'].values  # Gazebo X = East
            t_north = df['target_y'].values  # Gazebo Y = North
            t_up    = df['target_z'].values  # Gazebo Z = Up (sphere centre)
            ax.plot(t_east, t_north, t_up, linewidth=2.5, color='red',
                    label='Target path', linestyle='--')
            mid = len(t_east) // 2
            # Draw balloon sphere at midpoint using scatter with size scaled to radius
            # Approximate screen-size in points: use a large marker to represent 0.3m sphere
            ax.scatter(t_east[mid], t_north[mid], t_up[mid],
                       s=1500, color='red', alpha=0.35,
                       marker='o', edgecolors='darkred', linewidths=2,
                       zorder=10, label=f'Target (r={BALLOON_RADIUS}m)')
            ax.scatter(t_east[mid], t_north[mid], t_up[mid],
                       s=40, color='darkred', marker='+',
                       zorder=11)
            break

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


# ═══════════════════════════════════════════════════════════════
#  Graph 2: Image Coordinate (u-axis) Time Series
# ═══════════════════════════════════════════════════════════════
def graph_coordinates(datasets, out):
    """Image coordinate (u-axis) for each filter type."""
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, datasets.items()):
        color = COLORS.get(name, 'gray')
        ax.set_title(f'{name}: Image Coordinate (u-axis)', fontsize=13)
        ax.scatter(df['timestamp_s'].values, df['u_yolo'].values,
                   s=6, alpha=0.3, color='gray', label='YOLO raw')
        ax.plot(df['timestamp_s'].values, df['u_gt'].values,
                linewidth=1.5, color='#2ecc71', linestyle='--', label='Ground Truth')
        ax.plot(df['timestamp_s'].values, df['u_est'].values,
                linewidth=1.2, color=color, label=f'{name} estimate')
        ax.axhline(y=424, color='orange', linestyle=':', alpha=0.5, label='Image center')
        ax.set_ylabel('u (pixels)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


# ═══════════════════════════════════════════════════════════════
#  Graph 3: Estimation Error
# ═══════════════════════════════════════════════════════════════
def graph_error(datasets, out):
    """Estimation error (pixel distance to GT)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    w = 25  # 0.5s moving average at 50Hz

    ax = axes[0]
    ax.set_title('Estimation Error: Distance to Ground Truth (Smoothed)', fontsize=13)
    for name, df in datasets.items():
        color = COLORS.get(name, 'gray')
        smooth = df['err_est_px'].rolling(w, min_periods=1).mean().values
        ax.plot(df['timestamp_s'].values, smooth, linewidth=2, color=color,
                label=f'{name} (0.5s avg)')
    ax.set_ylabel('Error (pixels)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.set_title('Raw Estimation Error', fontsize=13)
    for name, df in datasets.items():
        color = COLORS.get(name, 'gray')
        ax.plot(df['timestamp_s'].values, df['err_est_px'].values,
                linewidth=0.8, alpha=0.5, color=color, label=name)
    ax.set_ylabel('Error (pixels)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


# ═══════════════════════════════════════════════════════════════
#  Graph 4: Control Performance
# ═══════════════════════════════════════════════════════════════
def graph_control(datasets, out):
    """Control: ex, yaw, speed, distance to target."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # ex (image center error)
    ax = axes[0]
    ax.set_title('ex: Horizontal Distance from Image Center', fontsize=13)
    for name, df in datasets.items():
        color = COLORS.get(name, 'gray')
        ax.plot(df['timestamp_s'].values, df['ex_from_center'].values,
                linewidth=1.2, color=color, label=name, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    ax.axhspan(-20, 20, alpha=0.1, color='green', label='±20px')
    ax.set_ylabel('ex (pixels)')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Yaw
    ax = axes[1]
    ax.set_title('Drone Yaw Angle', fontsize=13)
    for name, df in datasets.items():
        color = COLORS.get(name, 'gray')
        ax.plot(df['timestamp_s'].values, np.degrees(df['drone_yaw'].values),
                linewidth=1.5, color=color, label=name)
    ax.set_ylabel('Yaw (deg)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Speed
    ax = axes[2]
    ax.set_title('Drone Speed', fontsize=13)
    for name, df in datasets.items():
        color = COLORS.get(name, 'gray')
        spd = np.sqrt(df['drone_vx']**2 + df['drone_vy']**2 + df['drone_vz']**2).values
        ax.plot(df['timestamp_s'].values, spd, linewidth=1.5, color=color, label=name)
    ax.set_ylabel('Speed (m/s)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Distance to target
    ax = axes[3]
    ax.set_title('Distance to Target', fontsize=13)
    for name, df in datasets.items():
        color = COLORS.get(name, 'gray')
        if 'dist_to_target' in df.columns:
            ax.plot(df['timestamp_s'].values, df['dist_to_target'].values,
                    linewidth=1.5, color=color, label=name)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Collision threshold')
    ax.set_ylabel('Distance (m)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


# ═══════════════════════════════════════════════════════════════
#  Graph 5: Summary Statistics + CEP
# ═══════════════════════════════════════════════════════════════
def compute_stats(df, name):
    """Compute performance statistics for one experiment."""
    err = df['err_est_px'].dropna()
    ex = df['ex_from_center'].dropna().abs()

    # Convergence time: |ex| < 20px sustained for 0.5s
    conv_t = float('nan')
    if len(ex) >= 25:
        conv_mask = (ex < 20).rolling(25, min_periods=25).sum() == 25
        ci = conv_mask[conv_mask].index
        if len(ci) > 0:
            conv_t = df.loc[ci[0], 'timestamp_s'] - df['timestamp_s'].iloc[0]

    # CEP (Circular Error Probable): radius within which 50% of estimates fall
    if len(err) > 0:
        cep = float(err.quantile(0.5))
    else:
        cep = float('nan')

    # Mission time (first INTERCEPT to end or collision)
    intercept_mask = df['mission_state'] == 'INTERCEPT'
    if intercept_mask.any():
        t_start = df.loc[intercept_mask, 'timestamp_s'].iloc[0]
        t_end = df.loc[intercept_mask, 'timestamp_s'].iloc[-1]
        mission_time = t_end - t_start
    else:
        mission_time = float('nan')

    # Final distance to target
    if 'dist_to_target' in df.columns:
        final_dist = df['dist_to_target'].dropna().iloc[-1] if len(df) > 0 else float('nan')
        min_dist = df['dist_to_target'].dropna().min()
    else:
        final_dist = float('nan')
        min_dist = float('nan')

    return {
        'name': name,
        'mean_err': err.mean() if len(err) > 0 else float('nan'),
        'median_err': err.median() if len(err) > 0 else float('nan'),
        'std_err': err.std() if len(err) > 0 else float('nan'),
        'max_err': err.max() if len(err) > 0 else float('nan'),
        'p90_err': err.quantile(0.9) if len(err) > 0 else float('nan'),
        'cep': cep,
        'conv_t': conv_t,
        'mission_time': mission_time,
        'min_dist': min_dist,
        'final_dist': final_dist,
        'samples': len(df),
    }


def graph_summary(datasets, out):
    """Summary bar chart with CEP, errors, convergence time, mission time."""
    all_stats = {name: compute_stats(df, name) for name, df in datasets.items()}

    labels = ['CEP\n(px)', 'Mean\nError', 'Median\nError',
              'Converge\nTime (s)', 'Mission\nTime (s)', 'Min\nDist (m)']

    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(all_stats)
    x = np.arange(len(labels))
    w = 0.8 / n

    for i, (name, s) in enumerate(all_stats.items()):
        color = COLORS.get(name, 'gray')
        vals = [s['cep'], s['mean_err'], s['median_err'],
                s['conv_t'], s['mission_time'], s['min_dist']]
        bars = ax.bar(x + (i - n/2 + 0.5) * w, vals, w,
                      label=name, color=color, alpha=0.8)
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x() + b.get_width()/2, h + 0.2,
                        f'{h:.1f}', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=12)
    ax.set_title('Performance Summary (lower is better)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    # Improvement text (if 2+ datasets)
    names = list(all_stats.keys())
    if len(names) >= 2:
        s0, s1 = all_stats[names[0]], all_stats[names[1]]
        if s1['mean_err'] > 0 and not np.isnan(s0['mean_err']):
            imp = (s1['mean_err'] - s0['mean_err']) / s1['mean_err'] * 100
            better = names[0] if imp > 0 else names[1]
            color = 'green' if imp > 0 else 'red'
            # ax.text(0.02, 0.95,
            #         f'{better} improvement: {abs(imp):.1f}% (Mean Error)',
            #         transform=ax.transAxes, fontsize=13, fontweight='bold',
            #         color=color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


def text_summary(datasets):
    """Print text summary to console."""
    print('\n' + '=' * 65)
    print('         Filter Performance Comparison Results')
    print('=' * 65)

    all_stats = {}
    for name, df in datasets.items():
        s = compute_stats(df, name)
        all_stats[name] = s
        print(f'\n  [{name}]  ({s["samples"]} samples)')
        print(f'    CEP (50%ile):    {s["cep"]:.2f} px')
        print(f'    Mean Error:      {s["mean_err"]:.2f} px')
        print(f'    90th Percentile: {s["p90_err"]:.2f} px')
        print(f'    Max Error:       {s["max_err"]:.2f} px')
        print(f'    Convergence:     {s["conv_t"]:.2f} s')
        print(f'    Mission Time:    {s["mission_time"]:.2f} s')
        print(f'    Min Distance:    {s["min_dist"]:.3f} m')
        print(f'    Final Distance:  {s["final_dist"]:.3f} m')

    # Pairwise comparison
    names = list(all_stats.keys())
    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = all_stats[names[i]], all_stats[names[j]]
                if a['mean_err'] > 0 and b['mean_err'] > 0:
                    imp = (b['mean_err'] - a['mean_err']) / b['mean_err'] * 100
                    print(f'\n  >>> {names[i]} vs {names[j]}: '
                          f'{names[i]} {"better" if imp > 0 else "worse"} '
                          f'by {abs(imp):.1f}% (Mean Error) <<<')
                    imp_cep = (b['cep'] - a['cep']) / b['cep'] * 100 if b['cep'] > 0 else 0
                    print(f'  >>> CEP: {names[i]} {"better" if imp_cep > 0 else "worse"} '
                          f'by {abs(imp_cep):.1f}% <<<')

    print('=' * 65)


def main():
    # Determine which logs to load
    datasets = {}

    if len(sys.argv) >= 3:
        # Explicit paths
        for path in sys.argv[1:]:
            df = load(path)
            ftype = get_filter_type(df)
            if ftype == 'UNKNOWN':
                ftype = os.path.basename(path).split('_')[-1].replace('.csv', '')
            datasets[ftype] = active_region(df)
            print(f'Loaded {ftype}: {path}')
    elif len(sys.argv) == 2:
        path = sys.argv[1]
        df = load(path)
        ftype = get_filter_type(df)
        if ftype == 'UNKNOWN':
            ftype = os.path.basename(path).split('_')[-1].replace('.csv', '')
        datasets[ftype] = active_region(df)
        print(f'Loaded {ftype}: {path}')
    else:
        found = find_logs()
        if not found:
            print('No log files found in ~/dkf_logs/')
            sys.exit(1)
        for ftype, path in found.items():
            datasets[ftype] = active_region(load(path))
            print(f'Using {ftype}: {path}')

    out_dir = os.path.expanduser('~/dkf_logs')
    base = 'analysis'

    print('\nGenerating graphs...')
    graph_3d_trajectory(datasets, os.path.join(out_dir, f'{base}_1_trajectory3d.png'))
    graph_coordinates(datasets, os.path.join(out_dir, f'{base}_2_coordinates.png'))
    graph_error(datasets, os.path.join(out_dir, f'{base}_3_error.png'))
    graph_control(datasets, os.path.join(out_dir, f'{base}_4_control.png'))
    graph_summary(datasets, os.path.join(out_dir, f'{base}_5_summary.png'))

    text_summary(datasets)
    print(f'\nAll result images saved in: {out_dir}/')


if __name__ == '__main__':
    main()
