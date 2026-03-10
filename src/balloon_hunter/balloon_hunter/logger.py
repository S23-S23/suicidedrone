#!/usr/bin/env python3
"""
DKF vs EKF Performance Logger
===============================
Runs alongside ibvs_png_controller to log data for comparing
DKF (delay-compensated) vs EKF (no delay compensation).

Subscribes to:
- /Yolov8_Inference_1  : YOLO raw detections
- /drone1/fmu/out/monitoring : drone state (pos, vel, attitude, angular vel)

Computes internally:
- Ground truth image coordinates (from known target world position)
- Simple EKF estimate (no delay compensation, for comparison)
- DKF estimate (read from controller's published data or re-run internally)

Logs to CSV: ~/dkf_logs/log_YYYYMMDD_HHMMSS.csv

Usage:
  ros2 run balloon_hunter dkf_logger
"""

import math
import os
import csv
import numpy as np
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import Monitoring
from yolov8_msgs.msg import Yolov8Inference


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────
def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)

def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)


# ──────────────────────────────────────────────
# Simple EKF (no delay compensation, for comparison)
# ──────────────────────────────────────────────
class SimpleEKF:
    """
    Standard Extended Kalman Filter WITHOUT delay compensation.
    Uses constant-velocity model. This is the baseline to compare against DKF.
    
    When YOLO measurement arrives, it treats it as "current" (ignoring delay).
    """

    def __init__(self, dt=0.02):
        self.dt = dt
        # State: [u, v, u_dot, v_dot] in pixels
        self.x = np.zeros(4)

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        self.Q = np.diag([5.0, 5.0, 50.0, 50.0])
        self.R = np.diag([10.0, 10.0])
        self.P = np.eye(4) * 100.0
        self.initialized = False

    def predict(self):
        if not self.initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """z = [u, v] in pixels. Treats as current (no delay handling)."""
        if not self.initialized:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.initialized = True
            return
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_estimate(self):
        if not self.initialized:
            return None
        return self.x.copy()  # [u, v, u_dot, v_dot]


# ──────────────────────────────────────────────
# Paper DKF (copy from controller for independent logging)
# ──────────────────────────────────────────────
import collections

class DKFForLogging:
    """
    Same DKF as in the controller, duplicated here for independent estimation.
    This ensures the logger can compute DKF estimates without depending on
    the controller's internal state.
    """

    def __init__(self, foc, R_b_c, dt=0.02, delay_steps=3):
        self.foc = foc
        self.R_b_c = R_b_c
        self.R_c_b = R_b_c.T
        self.dt = dt
        self.D = delay_steps
        self.x = np.zeros(4)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.diag([1e-4, 1e-4, 1e-2, 1e-2])
        r = (5.0 / foc) ** 2
        self.R = np.diag([r, r])
        self.P = np.eye(4) * 0.1
        self.history = collections.deque(maxlen=30)
        self.initialized = False

    def _imu_image_motion(self, px, py, omega_body, vel_ned, R_e_b):
        dt = self.dt
        omega_cam = self.R_c_b @ omega_body
        vel_cam = self.R_c_b @ (R_e_b.T @ vel_ned)
        pz_c = 15.0
        wxc, wyc, wzc = omega_cam
        vxc, vyc, vzc = vel_cam
        dp_rot = np.array([
            px*py*wxc - (1+px**2)*wyc + py*wzc,
            (1+py**2)*wxc - px*py*wyc - px*wzc
        ]) * dt
        dp_trans = np.array([
            -vxc/pz_c + px*vzc/pz_c,
            -vyc/pz_c + py*vzc/pz_c
        ]) * dt
        return dp_rot + dp_trans

    def predict(self, omega_body, vel_ned, R_e_b):
        if not self.initialized:
            return
        px, py = self.x[0], self.x[1]
        dp = self._imu_image_motion(px, py, omega_body, vel_ned, R_e_b)
        
        omega_cam = self.R_c_b @ omega_body
        vel_cam = self.R_c_b @ (R_e_b.T @ vel_ned)
        wxc, wyc, wzc = omega_cam
        vzc = vel_cam[2]
        pzc = 15.0
        dt = self.dt
        
        F_pp = np.eye(2) + np.array([
            [vzc/pzc + py*wxc - 2*px*wyc, px*wxc + wzc],
            [-py*wyc - wzc, vzc/pzc + 2*py*wxc - px*wyc]
        ]) * dt
        F = np.eye(4)
        F[0:2, 0:2] = F_pp
        F[0:2, 2:4] = np.eye(2) * dt

        self.history.append({
            'x': self.x.copy(), 'P': self.P.copy(),
            'omega_body': omega_body.copy(), 'vel_ned': vel_ned.copy(),
            'R_e_b': R_e_b.copy(),
        })
        self.x[0] += dp[0] + self.x[2] * dt
        self.x[1] += dp[1] + self.x[3] * dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_pixel):
        z_bar = np.array([z_pixel[0]/self.foc, z_pixel[1]/self.foc])
        if not self.initialized:
            self.x[:2] = z_bar
            self.x[2:] = 0.0
            self.P = np.eye(4) * 0.01
            self.initialized = True
            return
        D = self.D
        hl = len(self.history)
        if hl < D:
            H = self.H
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x += K @ (z_bar - H @ self.x)
            self.P = (np.eye(4) - K @ H) @ self.P
            return
        idx = hl - D
        xd = self.history[idx]['x'].copy()
        Pd = self.history[idx]['P'].copy()
        H = self.H
        S = H @ Pd @ H.T + self.R
        K = Pd @ H.T @ np.linalg.inv(S)
        xc = xd + K @ (z_bar - H @ xd)
        Pc = (np.eye(4) - K @ H) @ Pd
        for i in range(idx, hl):
            h = self.history[i]
            dp = self._imu_image_motion(xc[0], xc[1], h['omega_body'], h['vel_ned'], h['R_e_b'])
            xc[0] += dp[0] + xc[2]*self.dt
            xc[1] += dp[1] + xc[3]*self.dt
            
            omega_cam = self.R_c_b @ h['omega_body']
            vel_cam = self.R_c_b @ (h['R_e_b'].T @ h['vel_ned'])
            wxc,wyc,wzc = omega_cam
            vzc = vel_cam[2]
            pzc = 15.0
            F_pp = np.eye(2) + np.array([
                [vzc/pzc+xc[1]*wxc-2*xc[0]*wyc, xc[0]*wxc+wzc],
                [-xc[1]*wyc-wzc, vzc/pzc+2*xc[1]*wxc-xc[0]*wyc]
            ]) * self.dt
            F = np.eye(4)
            F[0:2,0:2] = F_pp
            F[0:2,2:4] = np.eye(2)*self.dt
            Pc = F @ Pc @ F.T + self.Q
        self.x = xc
        self.P = Pc

    def get_pixel(self):
        if not self.initialized:
            return None
        return np.array([self.x[0]*self.foc, self.x[1]*self.foc,
                         self.x[2]*self.foc, self.x[3]*self.foc])


# ──────────────────────────────────────────────
# Logger Node
# ──────────────────────────────────────────────
class DKFPerformanceLogger(Node):
    def __init__(self):
        super().__init__('dkf_performance_logger')

        # ── Parameters ──
        self.declare_parameter('system_id', 1)
        # Ground truth target position in GAZEBO frame (ENU)
        # Gazebo ENU (7, 10, 2) → NED: x_ned = x_enu, y_ned = -y_enu... 
        # Actually for PX4 SITL with Gazebo Classic:
        #   NED x = Gazebo x (forward/north)  -- depends on setup
        #   Need to match whatever coordinate the Monitoring topic uses
        self.declare_parameter('target_gazebo_x', 7.0)
        self.declare_parameter('target_gazebo_y', 10.0)
        self.declare_parameter('target_gazebo_z', 2.0)

        # Camera intrinsics (must match controller)
        self.declare_parameter('fx', 454.8)
        self.declare_parameter('fy', 454.8)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('cam_pitch_deg', 0.0)

        self.declare_parameter('detection_topic', '/Yolov8_Inference_1')
        self.declare_parameter('monitoring_topic', '/drone1/fmu/out/monitoring')

        # Get params
        self.system_id = self.get_parameter('system_id').value
        gz_x = self.get_parameter('target_gazebo_x').value
        gz_y = self.get_parameter('target_gazebo_y').value
        gz_z = self.get_parameter('target_gazebo_z').value

        # Convert Gazebo (ENU) to NED for PX4
        # PX4 SITL Gazebo Classic: NED_x = Gazebo_x, NED_y = Gazebo_y, NED_z = -Gazebo_z
        # (This depends on your spawn orientation; adjust if needed)
        self.target_ned = np.array([gz_x, gz_y, -gz_z])

        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.cam_pitch = math.radians(self.get_parameter('cam_pitch_deg').value)
        self.foc = self.fx

        # Camera transform
        self.R_b_c = np.array([
            [0, 0, 1], [1, 0, 0], [0, 1, 0]
        ], dtype=float) @ rot_x(-self.cam_pitch)

        # Drone state
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_yaw = 0.0
        self.drone_pitch = 0.0
        self.drone_roll = 0.0
        self.drone_omega = np.zeros(3)

        # YOLO raw
        self.latest_yolo_u = float('nan')
        self.latest_yolo_v = float('nan')
        self.yolo_count = 0
        self.yolo_first_time = None

        # ── Filters ──
        self.ekf = SimpleEKF(dt=0.02)
        self.dkf = DKFForLogging(
            foc=self.foc, R_b_c=self.R_b_c, dt=0.02, delay_steps=3
        )

        # ── CSV setup ──
        log_dir = os.path.expanduser('~/dkf_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(log_dir, f'log_{timestamp_str}.csv')

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp_s',
            'u_yolo', 'v_yolo',
            'u_ekf', 'v_ekf',
            'u_dkf', 'v_dkf',
            'u_gt', 'v_gt',
            'drone_x', 'drone_y', 'drone_z',
            'drone_yaw', 'drone_pitch', 'drone_roll',
            'drone_vx', 'drone_vy', 'drone_vz',
            'omega_x', 'omega_y', 'omega_z',
            'err_ekf_px', 'err_dkf_px',
            'ex_ekf', 'ex_dkf',
            'yolo_hz',
        ])

        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.row_count = 0

        # ── Subscribers ──
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(
            Monitoring,
            self.get_parameter('monitoring_topic').value,
            self.monitoring_cb, qos_be)

        self.create_subscription(
            Yolov8Inference,
            self.get_parameter('detection_topic').value,
            self.detection_cb, 10)

        # 50Hz logging timer (matches controller rate)
        self.create_timer(0.02, self.log_timer_cb)

        self.get_logger().info(f'DKF Logger started → {self.csv_path}')
        self.get_logger().info(f'Target NED: {self.target_ned}')

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f'Logged {self.row_count} rows to {self.csv_path}')
        super().destroy_node()

    # ──────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────
    def monitoring_cb(self, msg: Monitoring):
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        if hasattr(msg, 'vel_x'):
            self.drone_vel = np.array([msg.vel_x, msg.vel_y, msg.vel_z])
        self.drone_yaw = msg.head
        self.drone_pitch = getattr(msg, 'pitch', 0.0)
        self.drone_roll = getattr(msg, 'roll', 0.0)
        if hasattr(msg, 'rollspeed'):
            self.drone_omega = np.array([
                msg.rollspeed, msg.pitchspeed, msg.yawspeed])

    def detection_cb(self, msg: Yolov8Inference):
        if not msg.yolov8_inference:
            return
        det = msg.yolov8_inference[0]
        u = (det.left + det.right) * 0.5
        v = (det.top + det.bottom) * 0.5

        self.latest_yolo_u = u
        self.latest_yolo_v = v

        # Feed to both filters
        self.ekf.update(np.array([u, v]))
        self.dkf.update(np.array([u, v]))

        # Count for Hz calculation
        now = self.get_clock().now().nanoseconds / 1e9
        if self.yolo_first_time is None:
            self.yolo_first_time = now
        self.yolo_count += 1

    # ──────────────────────────────────────────
    # Ground truth projection
    # ──────────────────────────────────────────
    def compute_ground_truth_uv(self):
        """
        Project known 3D target position to image pixel coordinates.
        This is the "perfect" answer for where the target should appear.
        
        Returns (u_gt, v_gt) or (nan, nan) if target is behind camera.
        """
        # Relative position: target - drone, in NED
        p_rel_ned = self.target_ned - self.drone_pos

        # NED → Body
        R_e_b = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch) @ rot_x(self.drone_roll)
        p_rel_body = R_e_b.T @ p_rel_ned

        # Body → Camera
        R_c_b = self.R_b_c.T
        p_cam = R_c_b @ p_rel_body

        # Camera convention: Z=forward, X=right, Y=down
        if p_cam[2] <= 0:
            return float('nan'), float('nan')  # Behind camera

        # Pinhole projection
        u_gt = self.fx * (p_cam[0] / p_cam[2]) + self.cx
        v_gt = self.fy * (p_cam[1] / p_cam[2]) + self.cy

        return u_gt, v_gt

    # ──────────────────────────────────────────
    # Main logging loop (50Hz)
    # ──────────────────────────────────────────
    def log_timer_cb(self):
        now = self.get_clock().now().nanoseconds / 1e9
        t = now - self.start_time

        # Build rotation matrix for DKF predict
        R_e_b = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch) @ rot_x(self.drone_roll)

        # Predict both filters
        self.ekf.predict()
        self.dkf.predict(self.drone_omega, self.drone_vel, R_e_b)

        # Get estimates
        ekf_est = self.ekf.get_estimate()
        dkf_est = self.dkf.get_pixel()

        u_ekf = ekf_est[0] if ekf_est is not None else float('nan')
        v_ekf = ekf_est[1] if ekf_est is not None else float('nan')
        u_dkf = dkf_est[0] if dkf_est is not None else float('nan')
        v_dkf = dkf_est[1] if dkf_est is not None else float('nan')

        # Ground truth
        u_gt, v_gt = self.compute_ground_truth_uv()

        # Error metrics (pixels)
        if not math.isnan(u_gt) and not math.isnan(u_ekf):
            err_ekf = math.sqrt((u_ekf - u_gt)**2 + (v_ekf - v_gt)**2)
        else:
            err_ekf = float('nan')

        if not math.isnan(u_gt) and not math.isnan(u_dkf):
            err_dkf = math.sqrt((u_dkf - u_gt)**2 + (v_dkf - v_gt)**2)
        else:
            err_dkf = float('nan')

        # Image center error (for control performance)
        ex_ekf = (u_ekf - self.cx) if not math.isnan(u_ekf) else float('nan')
        ex_dkf = (u_dkf - self.cx) if not math.isnan(u_dkf) else float('nan')

        # YOLO Hz
        elapsed = now - self.yolo_first_time if self.yolo_first_time else 0.0
        yolo_hz = self.yolo_count / elapsed if elapsed > 1.0 else 0.0

        # Write row
        self.csv_writer.writerow([
            f'{t:.4f}',
            f'{self.latest_yolo_u:.1f}', f'{self.latest_yolo_v:.1f}',
            f'{u_ekf:.1f}', f'{v_ekf:.1f}',
            f'{u_dkf:.1f}', f'{v_dkf:.1f}',
            f'{u_gt:.1f}', f'{v_gt:.1f}',
            f'{self.drone_pos[0]:.3f}', f'{self.drone_pos[1]:.3f}', f'{self.drone_pos[2]:.3f}',
            f'{self.drone_yaw:.4f}', f'{self.drone_pitch:.4f}', f'{self.drone_roll:.4f}',
            f'{self.drone_vel[0]:.3f}', f'{self.drone_vel[1]:.3f}', f'{self.drone_vel[2]:.3f}',
            f'{self.drone_omega[0]:.4f}', f'{self.drone_omega[1]:.4f}', f'{self.drone_omega[2]:.4f}',
            f'{err_ekf:.2f}', f'{err_dkf:.2f}',
            f'{ex_ekf:.1f}', f'{ex_dkf:.1f}',
            f'{yolo_hz:.1f}',
        ])
        self.row_count += 1

        if self.row_count % 250 == 0:  # ~5s
            self.csv_file.flush()
            self.get_logger().info(
                f'Logged {self.row_count} rows | '
                f'err_ekf={err_ekf:.1f}px err_dkf={err_dkf:.1f}px | '
                f'YOLO {yolo_hz:.1f}Hz')


def main(args=None):
    rclpy.init(args=args)
    node = DKFPerformanceLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()