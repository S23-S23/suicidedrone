#!/usr/bin/env python3
"""
Filter Performance Logger (DKF / EKF / GT)
============================================
Records per-tick data for post-experiment analysis:
  - YOLO raw detection (u, v)
  - Filter estimate (u, v, u_dot, v_dot, delay_steps) from /filter_estimate
  - Ground truth pixel (projected from target world position)
  - Drone 3D state (pos, vel, yaw, pitch, roll, omega) — from direct PX4 topics
  - Target 3D world position (from /target_world_pos)
  - Distance to target, mission elapsed time, delay D

Subscriptions:
  /droneN/fmu/out/monitoring          — pos, yaw, pitch, roll
  /droneN/fmu/out/vehicle_local_position — vel (NED)
  /droneN/fmu/out/vehicle_angular_velocity — omega (body-FRD)
  /filter_estimate                    — [u, v, u_dot, v_dot, delay_steps]

Auto-terminates when drone–target distance < collision_distance.

Usage:
  ros2 run balloon_hunter logger --ros-args -p filter_type:=GT
"""

import math, os, csv, time
import numpy as np
from datetime import datetime
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import Monitoring, VehicleLocalPosition, VehicleAngularVelocity
from suicide_drone_msgs.msg import TargetInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray, String


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
        super().__init__('filter_logger')

        # Parameters
        self.declare_parameter('filter_type', 'DKF')
        self.declare_parameter('system_id', 1)
        self.declare_parameter('target_gazebo_x', 0.0)
        self.declare_parameter('target_gazebo_y', 5.0)
        self.declare_parameter('target_gazebo_z', 2.0)
        self.declare_parameter('fx', 454.8)
        self.declare_parameter('fy', 454.8)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('cam_pitch_deg', 0.0)
        self.declare_parameter('detection_topic', '/target_info')
        self.declare_parameter('collision_distance', 1.0)

        self.filter_type = self.get_parameter('filter_type').value.upper()
        self.system_id   = self.get_parameter('system_id').value
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.cam_pitch = math.radians(self.get_parameter('cam_pitch_deg').value)
        self.foc = self.fx
        self.collision_dist = self.get_parameter('collision_distance').value

        prefix = f'drone{self.system_id}/fmu/out/'

        # Fallback static target position (NED)
        gz_x = self.get_parameter('target_gazebo_x').value
        gz_y = self.get_parameter('target_gazebo_y').value
        gz_z = self.get_parameter('target_gazebo_z').value
        # Gazebo ENU: X=East, Y=North, Z=Up → PX4 NED: X=North, Y=East, Z=Down
        self.target_ned = np.array([gz_y, gz_x, -gz_z])
        self._target_from_topic = False

        # Camera body→camera rotation
        self.R_b_c = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float) @ rot_x(-self.cam_pitch)

        # Drone state (all from direct PX4 topics)
        self.drone_pos = np.zeros(3)   # NED [m]
        self.drone_vel = np.zeros(3)   # NED [m/s]
        self.drone_yaw = 0.0
        self.drone_pitch = 0.0
        self.drone_roll  = 0.0
        self.drone_omega = np.zeros(3) # body-FRD [rad/s]

        # Detection / estimate
        self.yolo_u = self.yolo_v = float('nan')
        self.est_u = self.est_v = float('nan')
        self.est_udot = self.est_vdot = float('nan')
        self.delay_steps = 0
        self.yolo_count = 0
        self.yolo_first_t = None

        # Target world position (Gazebo ENU: x, y, z)
        self.target_world = np.array([gz_x, gz_y, gz_z])

        # Mission state
        self.mission_state = 'IDLE'
        self._finished = False

        # CSV setup
        log_dir = os.path.expanduser('~/dkf_logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(log_dir, f'log_{ts}_{self.filter_type}.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.w = csv.writer(self.csv_file)
        self.w.writerow([
            'timestamp_s', 'filter_type',
            'u_yolo', 'v_yolo',
            'u_est', 'v_est',
            'udot_est', 'vdot_est',
            'u_gt', 'v_gt',
            'drone_x', 'drone_y', 'drone_z',
            'drone_yaw', 'drone_pitch', 'drone_roll',
            'drone_vx', 'drone_vy', 'drone_vz',
            'omega_x', 'omega_y', 'omega_z',
            'target_x', 'target_y', 'target_z',
            'dist_to_target',
            'delay_steps',
            'err_est_px', 'ex_from_center',
            'yolo_hz', 'mission_state',
        ])
        self.start_t = self.get_clock().now().nanoseconds / 1e9
        self.rows = 0

        # QoS — match PX4 BEST_EFFORT sensor topics
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10
        )

        # ── Subscriptions ──
        self.create_subscription(
            Monitoring, f'{prefix}monitoring',
            self.mon_cb, qos
        )
        self.create_subscription(
            VehicleLocalPosition, f'{prefix}vehicle_local_position',
            self.vlp_cb, qos
        )
        self.create_subscription(
            VehicleAngularVelocity, f'{prefix}vehicle_angular_velocity',
            self.avel_cb, qos
        )
        self.create_subscription(
            TargetInfo,
            self.get_parameter('detection_topic').value,
            self.det_cb, 10
        )
        self.create_subscription(
            Float32MultiArray, '/filter_estimate',
            self.est_cb, 10
        )
        self.create_subscription(
            Point, '/target_world_pos',
            self.target_cb, 10
        )
        self.create_subscription(
            String, '/mission_state',
            self.state_cb, 10
        )

        # 50Hz logging timer
        self.create_timer(0.02, self.log_cb)

        self.get_logger().info(f'Logger started [{self.filter_type}] -> {self.csv_path}')

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f'Saved {self.rows} rows to {self.csv_path}')
        super().destroy_node()

    # ── Callbacks ──────────────────────────────────────────────
    def mon_cb(self, msg: Monitoring):
        """Position + attitude from Monitoring (yaw/pitch/roll available here)."""
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        self.drone_yaw   = msg.head
        self.drone_pitch = msg.pitch
        self.drone_roll  = msg.roll

    def vlp_cb(self, msg: VehicleLocalPosition):
        """Velocity from VehicleLocalPosition (Monitoring has no vel fields)."""
        if msg.v_xy_valid and msg.v_z_valid:
            self.drone_vel = np.array([msg.vx, msg.vy, msg.vz])

    def avel_cb(self, msg: VehicleAngularVelocity):
        """Angular velocity in body-FRD from VehicleAngularVelocity."""
        self.drone_omega = np.array([msg.xyz[0], msg.xyz[1], msg.xyz[2]])

    def det_cb(self, msg: TargetInfo):
        self.yolo_u = (msg.left + msg.right) * 0.5
        self.yolo_v = (msg.top + msg.bottom) * 0.5
        now = self.get_clock().now().nanoseconds / 1e9
        if self.yolo_first_t is None:
            self.yolo_first_t = now
        self.yolo_count += 1

    def est_cb(self, msg: Float32MultiArray):
        """Filter estimate: [u, v, u_dot, v_dot, delay_steps]."""
        if len(msg.data) >= 2:
            self.est_u    = msg.data[0]
            self.est_v    = msg.data[1]
        if len(msg.data) >= 4:
            self.est_udot = msg.data[2]
            self.est_vdot = msg.data[3]
        if len(msg.data) >= 5:
            self.delay_steps = int(msg.data[4])

    def target_cb(self, msg: Point):
        """Receive target world position from /target_world_pos (Gazebo ENU)."""
        self.target_world = np.array([msg.x, msg.y, msg.z])
        # Gazebo ENU: X=East, Y=North, Z=Up → PX4 NED: X=North, Y=East, Z=Down
        self.target_ned = np.array([msg.y, msg.x, -msg.z])
        self._target_from_topic = True

    def state_cb(self, msg: String):
        self.mission_state = msg.data

    # ── Ground truth projection ───────────────────────────────
    def gt_uv(self):
        """Project known 3D target to image pixel (u, v)."""
        p = self.target_ned - self.drone_pos
        R = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch) @ rot_x(self.drone_roll)
        pc = self.R_b_c.T @ (R.T @ p)
        if pc[2] <= 0:
            return float('nan'), float('nan')
        return self.fx * (pc[0] / pc[2]) + self.cx, self.fy * (pc[1] / pc[2]) + self.cy

    # ── Distance to target ────────────────────────────────────
    def dist_to_target(self):
        return float(np.linalg.norm(self.drone_pos - self.target_ned))

    # ── Main logging (50Hz) ───────────────────────────────────
    def log_cb(self):
        if self._finished:
            return

        t = self.get_clock().now().nanoseconds / 1e9 - self.start_t

        u_gt, v_gt = self.gt_uv()
        u_est, v_est = self.est_u, self.est_v
        dist = self.dist_to_target()

        # Estimation error
        if not math.isnan(u_gt) and not math.isnan(u_est):
            err = math.sqrt((u_est - u_gt)**2 + (v_est - v_gt)**2)
        else:
            err = float('nan')

        ex = (u_est - self.cx) if not math.isnan(u_est) else float('nan')

        elapsed = (self.get_clock().now().nanoseconds / 1e9 - self.yolo_first_t) if self.yolo_first_t else 0
        hz = self.yolo_count / elapsed if elapsed > 1 else 0

        self.w.writerow([
            f'{t:.4f}', self.filter_type,
            f'{self.yolo_u:.1f}', f'{self.yolo_v:.1f}',
            f'{u_est:.1f}', f'{v_est:.1f}',
            f'{self.est_udot:.1f}', f'{self.est_vdot:.1f}',
            f'{u_gt:.1f}', f'{v_gt:.1f}',
            f'{self.drone_pos[0]:.3f}', f'{self.drone_pos[1]:.3f}', f'{self.drone_pos[2]:.3f}',
            f'{self.drone_yaw:.4f}', f'{self.drone_pitch:.4f}', f'{self.drone_roll:.4f}',
            f'{self.drone_vel[0]:.3f}', f'{self.drone_vel[1]:.3f}', f'{self.drone_vel[2]:.3f}',
            f'{self.drone_omega[0]:.4f}', f'{self.drone_omega[1]:.4f}', f'{self.drone_omega[2]:.4f}',
            f'{self.target_world[0]:.3f}', f'{self.target_world[1]:.3f}', f'{self.target_world[2]:.3f}',
            f'{dist:.3f}',
            f'{self.delay_steps}',
            f'{err:.2f}', f'{ex:.1f}',
            f'{hz:.1f}', self.mission_state,
        ])
        self.rows += 1

        if self.rows % 250 == 0:
            self.csv_file.flush()
            self.get_logger().info(
                f'[{self.filter_type}] {self.rows} rows | '
                f'err={err:.1f}px | dist={dist:.1f}m | D={self.delay_steps} | {self.mission_state}'
            )

        # Auto-terminate on collision
        if dist < self.collision_dist and self.mission_state == 'TRACKING':
            self.get_logger().info(
                f'[{self.filter_type}] COLLISION detected (dist={dist:.2f}m) '
                f'at t={t:.2f}s — saving and shutting down'
            )
            self._finished = True
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info(f'Saved {self.rows} rows to {self.csv_path}')
            raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = FilterLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
