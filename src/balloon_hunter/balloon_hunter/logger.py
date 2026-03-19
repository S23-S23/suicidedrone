#!/usr/bin/env python3
"""
Filter Performance Logger for Hover+Yaw scenario
==================================================
Subscribes to:
  - /filter_estimate (Float32MultiArray from controller: [u, v, u_dot, v_dot])
  - /Yolov8_Inference_1 (YOLO raw detections)
  - Monitoring (drone position/attitude)
  - VehicleLocalPosition (drone velocity)
  - VehicleAngularVelocity (drone angular velocity)

Computes ground truth by projecting known target position to image.

CSV columns:
  timestamp_s, filter_type,
  u_yolo, v_yolo,          ← YOLO raw detection
  u_filt, v_filt,           ← actual filter output (DKF or EKF)
  u_gt, v_gt,               ← ground truth projection
  drone_x, drone_y, drone_z,
  drone_yaw, drone_pitch, drone_roll,
  drone_vx, drone_vy, drone_vz,
  omega_x, omega_y, omega_z,
  err_filt_px,               ← |filter - GT| in pixels
  ex_filt,                   ← filter_u - cx (distance from center)
  ex_yolo,                   ← yolo_u - cx
  yolo_hz

Usage:
  ros2 run balloon_hunter logger --ros-args -p filter_type:=DKF
"""

import math, os, csv
import numpy as np
from datetime import datetime
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import Monitoring, VehicleLocalPosition, VehicleAngularVelocity
from yolov8_msgs.msg import Yolov8Inference
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point


def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=float)
def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=float)
def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=float)


class FilterLogger(Node):
    def __init__(self):
        super().__init__('logger')

        self.declare_parameter('filter_type', 'DKF')
        self.declare_parameter('system_id', 1)
        self.declare_parameter('target_gazebo_x', 7.0)
        self.declare_parameter('target_gazebo_y', 10.0)
        self.declare_parameter('target_gazebo_z', 2.0)
        self.declare_parameter('fx', 454.8)
        self.declare_parameter('fy', 454.8)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('cam_pitch_deg', 0.0)
        self.declare_parameter('detection_topic', '/Yolov8_Inference_1')
        self.declare_parameter('monitoring_topic', '/drone1/fmu/out/monitoring')

        self.filter_type = self.get_parameter('filter_type').value.upper()
        self.system_id = self.get_parameter('system_id').value
        gz_x = self.get_parameter('target_gazebo_x').value
        gz_y = self.get_parameter('target_gazebo_y').value
        gz_z = self.get_parameter('target_gazebo_z').value
        self.target_ned = np.array([gz_x, gz_y, -gz_z])   # updated by /target_world_pos
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.cam_pitch = math.radians(self.get_parameter('cam_pitch_deg').value)
        self.foc = self.fx

        self.R_b_c = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float) @ rot_x(-self.cam_pitch)

        # State
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_yaw = 0.0
        self.drone_pitch_val = 0.0
        self.drone_roll = 0.0
        self.drone_omega = np.zeros(3)

        # YOLO raw
        self.yolo_u = float('nan')
        self.yolo_v = float('nan')
        self.yolo_count = 0
        self.yolo_first_t = None

        # Filter estimate (from controller's /filter_estimate topic)
        self.filt_u = float('nan')
        self.filt_v = float('nan')

        # CSV
        log_dir = os.path.expanduser('~/dkf_logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(log_dir, f'log_{ts}_{self.filter_type}.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.w = csv.writer(self.csv_file)
        self.w.writerow([
            'timestamp_s', 'filter_type',
            'u_yolo', 'v_yolo',
            'u_filt', 'v_filt',
            'u_gt', 'v_gt',
            'drone_x', 'drone_y', 'drone_z',
            'drone_yaw', 'drone_pitch', 'drone_roll',
            'drone_vx', 'drone_vy', 'drone_vz',
            'omega_x', 'omega_y', 'omega_z',
            'err_filt_px', 'ex_filt', 'ex_yolo',
            'yolo_hz',
        ])
        self.start_t = self.get_clock().now().nanoseconds / 1e9
        self.rows = 0

        # QoS
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=10)
        topic_prefix = f"drone{self.system_id}/fmu/"

        # Subscribers
        self.create_subscription(Monitoring, self.get_parameter('monitoring_topic').value, self.mon_cb, qos)
        self.create_subscription(VehicleLocalPosition, f'{topic_prefix}out/vehicle_local_position', self.vlp_cb, qos)
        self.create_subscription(VehicleAngularVelocity, f'{topic_prefix}out/vehicle_angular_velocity', self.avel_cb, qos)
        self.create_subscription(Yolov8Inference, self.get_parameter('detection_topic').value, self.det_cb, 10)

        # Subscribe to controller's filter estimate
        self.create_subscription(Float32MultiArray, '/filter_estimate', self.filt_est_cb, 10)

        # Subscribe to moving target position (from target_mover)
        self.create_subscription(Point, '/target_world_pos', self._target_pos_cb, 10)

        # Timer 50Hz
        self.create_timer(0.02, self.log_cb)

        self.get_logger().info(f'Logger started [{self.filter_type}] -> {self.csv_path}')
        self.get_logger().info(f'Target NED: {self.target_ned}')
        self.get_logger().info(f'Subscribing to /filter_estimate for actual filter output')

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f'Saved {self.rows} rows to {self.csv_path}')
        super().destroy_node()

    # ── Callbacks ──
    def mon_cb(self, msg: Monitoring):
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        self.drone_yaw = msg.head
        self.drone_pitch_val = msg.pitch
        self.drone_roll = msg.roll

    def vlp_cb(self, msg: VehicleLocalPosition):
        if msg.v_xy_valid and msg.v_z_valid:
            self.drone_vel = np.array([msg.vx, msg.vy, msg.vz])

    def avel_cb(self, msg: VehicleAngularVelocity):
        self.drone_omega = np.array([msg.xyz[0], msg.xyz[1], msg.xyz[2]])

    def det_cb(self, msg: Yolov8Inference):
        if not msg.yolov8_inference:
            return
        d = msg.yolov8_inference[0]
        self.yolo_u = (d.left + d.right) * 0.5
        self.yolo_v = (d.top + d.bottom) * 0.5
        now = self.get_clock().now().nanoseconds / 1e9
        if self.yolo_first_t is None:
            self.yolo_first_t = now
        self.yolo_count += 1

    def _target_pos_cb(self, msg: Point):
        """Update target NED position from moving target (Gazebo frame → NED)."""
        self.target_ned = np.array([msg.x, msg.y, -msg.z])

    def filt_est_cb(self, msg: Float32MultiArray):
        """Receive actual filter estimate from controller."""
        if len(msg.data) >= 2:
            self.filt_u = msg.data[0]
            self.filt_v = msg.data[1]

    # ── Ground Truth ──
    def gt_uv(self):
        p_rel = self.target_ned - self.drone_pos
        R_e_b = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch_val) @ rot_x(self.drone_roll)
        p_cam = self.R_b_c.T @ (R_e_b.T @ p_rel)
        if p_cam[2] < 0.5:
            return float('nan'), float('nan')
        u = self.fx * (p_cam[0] / p_cam[2]) + self.cx
        v = self.fy * (p_cam[1] / p_cam[2]) + self.cy
        if abs(u) > 2000 or abs(v) > 2000:
            return float('nan'), float('nan')
        return u, v

    # ── Logging (50Hz) ──
    def log_cb(self):
        t = self.get_clock().now().nanoseconds / 1e9 - self.start_t

        u_gt, v_gt = self.gt_uv()

        # Filter estimate error (actual filter output vs GT)
        if not math.isnan(u_gt) and not math.isnan(self.filt_u):
            err_filt = math.sqrt((self.filt_u - u_gt)**2 + (self.filt_v - v_gt)**2)
        else:
            err_filt = float('nan')

        ex_filt = (self.filt_u - self.cx) if not math.isnan(self.filt_u) else float('nan')
        ex_yolo = (self.yolo_u - self.cx) if not math.isnan(self.yolo_u) else float('nan')

        elapsed = (self.get_clock().now().nanoseconds / 1e9 - self.yolo_first_t) if self.yolo_first_t else 0
        hz = self.yolo_count / elapsed if elapsed > 1 else 0

        self.w.writerow([
            f'{t:.4f}', self.filter_type,
            f'{self.yolo_u:.1f}', f'{self.yolo_v:.1f}',
            f'{self.filt_u:.1f}', f'{self.filt_v:.1f}',
            f'{u_gt:.1f}', f'{v_gt:.1f}',
            f'{self.drone_pos[0]:.3f}', f'{self.drone_pos[1]:.3f}', f'{self.drone_pos[2]:.3f}',
            f'{self.drone_yaw:.4f}', f'{self.drone_pitch_val:.4f}', f'{self.drone_roll:.4f}',
            f'{self.drone_vel[0]:.3f}', f'{self.drone_vel[1]:.3f}', f'{self.drone_vel[2]:.3f}',
            f'{self.drone_omega[0]:.4f}', f'{self.drone_omega[1]:.4f}', f'{self.drone_omega[2]:.4f}',
            f'{err_filt:.2f}', f'{ex_filt:.1f}', f'{ex_yolo:.1f}',
            f'{hz:.1f}',
        ])
        self.rows += 1
        if self.rows % 250 == 0:
            self.csv_file.flush()
            self.get_logger().info(
                f'[{self.filter_type}] {self.rows} rows | '
                f'filt=({self.filt_u:.0f},{self.filt_v:.0f}) gt=({u_gt:.0f},{v_gt:.0f}) '
                f'err={err_filt:.0f}px | YOLO {hz:.1f}Hz'
            )


def main(args=None):
    rclpy.init(args=args)
    node = FilterLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()