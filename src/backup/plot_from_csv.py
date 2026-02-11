#!/usr/bin/env python3
"""
Plot from CSV - 궤적 시각화 스크립트
사용법: python3 plot_from_csv.py <drone_csv> <target_csv> [GT_X GT_Y GT_Z]
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def load_csv_data(filepath):
    data = []
    if not os.path.exists(filepath):
        return np.array([])
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append([float(row['timestamp']), float(row['pos_x']), float(row['pos_y']), float(row['pos_z'])])
    return np.array(data)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 plot_from_csv.py <drone_csv> <target_csv> [gt_x gt_y gt_z]")
        return

    drone_data = load_csv_data(sys.argv[1])
    target_data = load_csv_data(sys.argv[2])

    if len(drone_data) == 0:
        print("No drone data found.")
        return

    # 상대 시간 계산
    t0 = drone_data[0, 0]
    d_time = drone_data[:, 0] - t0
    d_x, d_y, d_z = drone_data[:, 1], drone_data[:, 2], -drone_data[:, 3] # NED to Up

    # Ground Truth 설정 (월드 파일 기준 기본값 0, 7, 4)
    gt_pos = None
    if len(sys.argv) >= 6:
        gt_pos = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]

    fig = plt.figure(figsize=(15, 10))

    # 1. 3D Trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(d_x, d_y, d_z, label='Drone Path', color='blue')
    if len(target_data) > 0:
        t_x, t_y, t_z = target_data[:, 1], target_data[:, 2], -target_data[:, 3]
        ax1.scatter(t_x, t_y, t_z, c='orange', s=10, label='Estimated Target', alpha=0.5)
    if gt_pos:
        ax1.scatter(gt_pos[0], gt_pos[1], gt_pos[2], c='red', marker='*', s=200, label='True Target')
    ax1.set_title("3D Trajectory")
    ax1.legend()

    # 2. Top View (XY)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(d_x, d_y, label='Drone')
    if len(target_data) > 0:
        ax2.scatter(target_data[:, 1], target_data[:, 2], c='orange', s=5, alpha=0.5)
    if gt_pos:
        ax2.scatter(gt_pos[0], gt_pos[1], c='red', marker='*')
    ax2.set_xlabel("X (North)"); ax2.set_ylabel("Y (East)")
    ax2.set_title("Top View (XY)")
    ax2.grid(True)

    # 3. Altitude (Z over time)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(d_time, d_z, label='Drone Altitude')
    if gt_pos:
        ax3.axhline(y=gt_pos[2], color='red', linestyle='--', label='Target Height')
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Height (m)")
    ax3.set_title("Altitude over Time")
    ax3.legend()

    # 4. Error Statistics (거리 오차)
    ax4 = fig.add_subplot(2, 2, 4)
    if gt_pos and len(target_data) > 0:
        errors = np.linalg.norm(target_data[:, 1:4] - np.array([gt_pos[0], gt_pos[1], -gt_pos[2]]), axis=1)
        ax4.plot(target_data[:, 0] - t0, errors, color='purple')
        ax4.set_title(f"Estimation Error (Avg: {np.mean(errors):.2f}m)")
        ax4.set_ylabel("Error (m)")
    else:
        ax4.text(0.5, 0.5, "GT required for Error Plot", ha='center')

    plt.tight_layout()
    plot_name = sys.argv[1].replace('.csv', '.png')
    plt.savefig(plot_name)
    print(f"Plot saved as {plot_name}")
    plt.show()

if __name__ == '__main__':
    main()