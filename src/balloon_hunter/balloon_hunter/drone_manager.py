#!/usr/bin/env python3
"""
Hover + Yaw-Only Controller for DKF vs EKF comparison
=======================================================
Scenario:
  1. Takeoff and hover at fixed position
  2. Target is to the right of the drone
  3. Drone rotates yaw only (no translation) to center the target in image
  4. Publishes filter estimate on /filter_estimate for logger

This isolates the filter performance from navigation/guidance issues.
No PNG, no velocity control - pure yaw PD with position hold.

Parameter 'filter_type': 'DKF' or 'EKF'

Usage:
  ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py filter_type:=DKF
  ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py filter_type:=EKF
"""

import math
import os
import signal
import time
import collections
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    Monitoring,
    VehicleLocalPosition,
    VehicleAngularVelocity,
)
from geometry_msgs.msg import PoseStamped, Point
from yolov8_msgs.msg import Yolov8Inference
from std_msgs.msg import Float32MultiArray
from enum import Enum


# ── Utility ──
def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)

def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ══════════════════════════════════════════════
# DKF (Paper-faithful, Yang et al. 2025)
# ══════════════════════════════════════════════
class DelayedKalmanFilter:
    def __init__(self, foc, R_b_c, dt=0.02, delay_steps=3, max_history=30):
        self.foc = foc
        self.R_b_c = R_b_c
        self.R_c_b = R_b_c.T
        self.dt = dt
        self.D = delay_steps
        self.x = np.zeros(4)  # [p_bar_x, p_bar_y, p_bar_x_dot, p_bar_y_dot]
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.diag([1e-4, 1e-4, 1e-2, 1e-2])
        self.R = np.diag([(5.0 / foc) ** 2] * 2)
        self.P = np.eye(4) * 0.1
        self.history = collections.deque(maxlen=max_history)
        self.initialized = False

    def _imu_image_motion(self, px, py, omega_body, vel_ned, R_e_b):
        dt = self.dt
        wc = self.R_c_b @ omega_body
        vc = self.R_c_b @ (R_e_b.T @ vel_ned)
        pz = 5.0
        wxc, wyc, wzc = wc
        vxc, vyc, vzc = vc
        dp_rot = np.array([
            px * py * wxc - (1 + px**2) * wyc + py * wzc,
            (1 + py**2) * wxc - px * py * wyc - px * wzc
        ]) * dt
        dp_trans = np.array([
            -vxc / pz + px * vzc / pz,
            -vyc / pz + py * vzc / pz
        ]) * dt
        return dp_rot + dp_trans, wc, vzc, pz

    def _build_F(self, px, py, wc, vzc, pz):
        dt = self.dt
        wxc, wyc, wzc_val = wc
        F_pp = np.eye(2) + np.array([
            [vzc / pz + py * wxc - 2 * px * wyc, px * wxc + wzc_val],
            [-py * wyc - wzc_val, vzc / pz + 2 * py * wxc - px * wyc]
        ]) * dt
        F = np.eye(4)
        F[0:2, 0:2] = F_pp
        F[0:2, 2:4] = np.eye(2) * dt
        return F

    def predict(self, omega_body, vel_ned, R_e_b):
        if not self.initialized:
            return
        px, py = self.x[0], self.x[1]
        dp, wc, vzc, pz = self._imu_image_motion(px, py, omega_body, vel_ned, R_e_b)
        F = self._build_F(px, py, wc, vzc, pz)

        self.history.append({
            'x': self.x.copy(), 'P': self.P.copy(),
            'w': omega_body.copy(), 'v': vel_ned.copy(), 'R': R_e_b.copy(),
        })

        self.x[0] += dp[0] + self.x[2] * self.dt
        self.x[1] += dp[1] + self.x[3] * self.dt
        self.P = F @ self.P @ F.T + self.Q

        self.x[0] = np.clip(self.x[0], -3.0, 3.0)
        self.x[1] = np.clip(self.x[1], -3.0, 3.0)
        self.x[2] = np.clip(self.x[2], -2.0, 2.0)
        self.x[3] = np.clip(self.x[3], -2.0, 2.0)
        if np.any(np.diag(self.P) > 10.0):
            self.P = np.eye(4) * 0.1

    def update(self, z_pixel):
        zb = np.array([z_pixel[0] / self.foc, z_pixel[1] / self.foc])
        if not self.initialized:
            self.x[:2] = zb
            self.x[2:] = 0
            self.P = np.eye(4) * 0.01
            self.initialized = True
            return
        D = self.D
        hl = len(self.history)
        if hl < D:
            self._std_correct(zb)
            return
        idx = hl - D
        xd, Pd = self.history[idx]['x'].copy(), self.history[idx]['P'].copy()
        S = self.H @ Pd @ self.H.T + self.R
        K = Pd @ self.H.T @ np.linalg.inv(S)
        xc = xd + K @ (zb - self.H @ xd)
        Pc = (np.eye(4) - K @ self.H) @ Pd

        # delayed 시점부터 현재까지 re-propagate
        for i in range(idx, hl):
            h = self.history[i]
            dp, wc, vzc, pz = self._imu_image_motion(xc[0], xc[1], h['w'], h['v'], h['R'])
            F = self._build_F(xc[0], xc[1], wc, vzc, pz)
            xc[0] += dp[0] + xc[2] * self.dt
            xc[1] += dp[1] + xc[3] * self.dt
            Pc = F @ Pc @ F.T + self.Q

        self.x, self.P = xc, Pc

    def _std_correct(self, zb):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ (zb - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_pixel(self):
        if not self.initialized:
            return None
        return np.array([self.x[0] * self.foc, self.x[1] * self.foc,
                         self.x[2] * self.foc, self.x[3] * self.foc])


# ══════════════════════════════════════════════
# EKF (Simple, no delay compensation)
# ══════════════════════════════════════════════d
class SimpleEKF:
    """
    EKF (no delay compensation) - fair comparison with DKF.
    Same IMU-driven motion model as DKF, but no delayed correction:
    when a YOLO measurement arrives, update immediately with current state.
    """
    def __init__(self, foc, R_b_c, dt=0.02):
        self.foc = foc
        self.R_b_c = R_b_c
        self.R_c_b = R_b_c.T
        self.dt = dt
        self.x = np.zeros(4)   # [p_bar_x, p_bar_y, p_bar_x_dot, p_bar_y_dot]
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.diag([1e-4, 1e-4, 1e-2, 1e-2])
        self.R = np.diag([(5.0 / foc) ** 2] * 2)
        self.P = np.eye(4) * 0.1
        self.initialized = False

    def _imu_image_motion(self, px, py, omega_body, vel_ned, R_e_b):
        """DKF와 완전히 동일한 motion model."""
        dt = self.dt
        wc = self.R_c_b @ omega_body
        vc = self.R_c_b @ (R_e_b.T @ vel_ned)
        pz = 5.0
        wxc, wyc, wzc = wc
        vxc, vyc, vzc = vc
        dp_rot = np.array([
            px * py * wxc - (1 + px**2) * wyc + py * wzc,
            (1 + py**2) * wxc - px * py * wyc - px * wzc
        ]) * dt
        dp_trans = np.array([
            -vxc / pz + px * vzc / pz,
            -vyc / pz + py * vzc / pz
        ]) * dt
        return dp_rot + dp_trans, wc, vzc, pz

    def _build_F(self, px, py, wc, vzc, pz):
        """DKF와 완전히 동일한 Jacobian 계산."""
        dt = self.dt
        wxc, wyc, wzc_val = wc
        F_pp = np.eye(2) + np.array([
            [vzc / pz + py * wxc - 2 * px * wyc, px * wxc + wzc_val],
            [-py * wyc - wzc_val, vzc / pz + 2 * py * wxc - px * wyc]
        ]) * dt
        F = np.eye(4)
        F[0:2, 0:2] = F_pp
        F[0:2, 2:4] = np.eye(2) * dt
        return F

    def predict(self, omega_body, vel_ned, R_e_b):
        if not self.initialized:
            return
        px, py = self.x[0], self.x[1]
        dp, wc, vzc, pz = self._imu_image_motion(px, py, omega_body, vel_ned, R_e_b)
        F = self._build_F(px, py, wc, vzc, pz)

        self.x[0] += dp[0] + self.x[2] * self.dt
        self.x[1] += dp[1] + self.x[3] * self.dt
        self.P = F @ self.P @ F.T + self.Q

        self.x[0] = np.clip(self.x[0], -3.0, 3.0)
        self.x[1] = np.clip(self.x[1], -3.0, 3.0)
        self.x[2] = np.clip(self.x[2], -2.0, 2.0)
        self.x[3] = np.clip(self.x[3], -2.0, 2.0)
        if np.any(np.diag(self.P) > 10.0):
            self.P = np.eye(4) * 0.1

    def update(self, z_pixel):
        """딜레이 보상 없이 바로 update — DKF와의 유일한 차이점."""
        zb = np.array([z_pixel[0] / self.foc, z_pixel[1] / self.foc])
        if not self.initialized:
            self.x[:2] = zb
            self.x[2:] = 0
            self.P = np.eye(4) * 0.01
            self.initialized = True
            return
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ (zb - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_pixel(self):
        if not self.initialized:
            return None
        return np.array([self.x[0] * self.foc, self.x[1] * self.foc,
                         self.x[2] * self.foc, self.x[3] * self.foc])


# ══════════════════════════════════════════════
# State Machine
# ══════════════════════════════════════════════
class State(Enum):
    IDLE = 0
    TAKEOFF = 1
    YAW_TRACK = 2   # hover + yaw only


# ══════════════════════════════════════════════
# Controller Node
# ══════════════════════════════════════════════
class HoverYawController(Node):
    def __init__(self):
        super().__init__('drone_manager')

        # Parameters
        self.declare_parameter('system_id', 1)
        self.declare_parameter('filter_type', 'DKF')
        self.declare_parameter('takeoff_height', 6.0)
        self.declare_parameter('fx', 454.8)
        self.declare_parameter('fy', 454.8)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('cam_pitch_deg', 0.0)
        self.declare_parameter('kp_yaw', 0.5)    # normalized error [-1,1] → rad/step (max ~7.5 rad/s at 50Hz)#0.5
        self.declare_parameter('kd_yaw', 0.0003)  # normalized error rate → rad/step #0.0003
        self.declare_parameter('dkf_dt', 0.02)
        self.declare_parameter('dkf_delay_steps', 10)  # 200ms / 20ms = 10 steps
        self.declare_parameter('detection_topic', '/Yolov8_Inference_1')
        self.declare_parameter('monitoring_topic', '/drone1/fmu/out/monitoring')

        self.system_id = self.get_parameter('system_id').value
        self.filter_type = self.get_parameter('filter_type').value.upper()
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.fx = self.get_parameter('fx').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.cam_pitch = math.radians(self.get_parameter('cam_pitch_deg').value)
        self.foc = self.fx
        self.kp_yaw = self.get_parameter('kp_yaw').value
        self.kd_yaw = self.get_parameter('kd_yaw').value
        dkf_dt = self.get_parameter('dkf_dt').value
        dkf_delay = self.get_parameter('dkf_delay_steps').value
        self.topic_prefix = f"drone{self.system_id}/fmu/"

        # Camera transform
        self.R_b_c = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float) @ rot_x(-self.cam_pitch)

        # Filter
        if self.filter_type == 'DKF':
            self.filt = DelayedKalmanFilter(self.foc, self.R_b_c, dkf_dt, dkf_delay)
        else:
            self.filt = SimpleEKF(self.foc, self.R_b_c, dkf_dt)

        # State
        self.state = State.IDLE
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_yaw = 0.0
        self.drone_pitch_val = 0.0
        self.drone_roll = 0.0
        self.drone_omega = np.zeros(3)
        self.nav_state = 0
        self.arming_state = 0
        self.last_cmd_time = 0.0
        self.hover_pos = None  # position to hold
        self.prev_ex_norm = 0.0
        self.target_detected = False
        self.last_det_time = 0.0
        self._mission_start_t = None   # wall-clock time when YAW_TRACK begins
        self._mission_duration = 50.0  # seconds until auto-finish
        self.target_lost_timeout = 2.0  # seconds without detection → stop yawing

        # Publishers
        self.ocm_pub = self.create_publisher(OffboardControlMode, f'{self.topic_prefix}in/offboard_control_mode', qos_profile_sensor_data)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, f'{self.topic_prefix}in/trajectory_setpoint', qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(VehicleCommand, f'{self.topic_prefix}in/vehicle_command', qos_profile_sensor_data)

        # NEW: Publish filter estimate for logger
        self.est_pub = self.create_publisher(Float32MultiArray, '/filter_estimate', 10)

        # Subscribers
        self.create_subscription(VehicleStatus, f'{self.topic_prefix}out/vehicle_status', self.status_cb, qos_profile_sensor_data)
        self.create_subscription(Monitoring, self.get_parameter('monitoring_topic').value, self.monitoring_cb, qos_profile_sensor_data)
        self.create_subscription(VehicleLocalPosition, f'{self.topic_prefix}out/vehicle_local_position', self.vlp_cb, qos_profile_sensor_data)
        self.create_subscription(VehicleAngularVelocity, f'{self.topic_prefix}out/vehicle_angular_velocity', self.avel_cb, qos_profile_sensor_data)
        self.create_subscription(Yolov8Inference, self.get_parameter('detection_topic').value, self.det_cb, 10)

        # Timers
        self.create_timer(0.1, self.ocm_cb)
        self.create_timer(0.02, self.control_cb)  # 50Hz
        self.create_timer(5.0, self.start_mission, callback_group=None)  # one-shot via cancel

        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info(f'  Hover + Yaw-Only Controller')
        self.get_logger().info(f'  Filter: *** {self.filter_type} ***')
        self.get_logger().info(f'  Yaw PD: kp={self.kp_yaw}, kd={self.kd_yaw}')
        self.get_logger().info(f'  Scenario: hover + yaw tracking only')
        self.get_logger().info('═══════════════════════════════════════')

    # ── Callbacks ──
    def status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def monitoring_cb(self, msg: Monitoring):
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
        det = msg.yolov8_inference[0]
        u = (det.left + det.right) * 0.5
        v = (det.top + det.bottom) * 0.5
        self.filt.update(np.array([u, v]))
        self.target_detected = True
        self.last_det_time = self.get_clock().now().nanoseconds / 1e9
        # Update prev_ex_norm on actual detection (not on filter prediction)
        # so that ex_dot reflects real measurement rate, not 50Hz prediction noise
        self.prev_ex_norm = (u - self.cx) / self.cx

    def start_mission(self):
        # Cancel timer (one-shot)
        # Note: in ROS2, we need to store timer to cancel it
        if self.state == State.IDLE:
            self.get_logger().info(f'Mission start ({self.filter_type}) → TAKEOFF')
            self.state = State.TAKEOFF

    # ── Offboard heartbeat ──
    def ocm_cb(self):
        msg = OffboardControlMode()
        msg.position = True   # Always position mode (hover)
        msg.velocity = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.ocm_pub.publish(msg)

    # ── Main control loop (50Hz) ──
    def control_cb(self):
        R_e_b = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch_val) @ rot_x(self.drone_roll)

        # Filter predict
        if self.filt.initialized:
            self.filt.predict(
                omega_body=self.drone_omega,
                vel_ned=self.drone_vel,
                R_e_b=R_e_b
            )

        # Publish filter estimate (for logger)
        est = self.filt.get_pixel()
        if est is not None:
            msg = Float32MultiArray()
            msg.data = [float(est[0]), float(est[1]), float(est[2]), float(est[3])]
            self.est_pub.publish(msg)

        # State machine
        if self.state == State.IDLE:
            safe_z = max(self.drone_pos[2], -0.1)
            self._pub_pos([self.drone_pos[0], self.drone_pos[1], safe_z], 0.0)
        elif self.state == State.TAKEOFF:
            self._handle_takeoff()
        elif self.state == State.YAW_TRACK:
            self._handle_yaw_track()

    def _handle_takeoff(self):
        alt = -self.takeoff_height
        now = self.get_clock().now().nanoseconds / 1e9

        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED or \
           self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            safe_z = max(self.drone_pos[2], -0.1)
            self._pub_pos([self.drone_pos[0], self.drone_pos[1], safe_z], 0.0)
        else:
            self._pub_pos([self.drone_pos[0], self.drone_pos[1], alt], self.drone_yaw)

        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if now - self.last_cmd_time > 1.0:
                self._pub_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.get_logger().info('ARM requested')
                self.last_cmd_time = now
            return

        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if now - self.last_cmd_time > 1.0:
                self._pub_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.get_logger().info('OFFBOARD requested')
                self.last_cmd_time = now
            return

        if abs(self.drone_pos[2] - alt) < 0.3:
            self.hover_pos = self.drone_pos.copy()
            self.hover_pos[2] = alt  # exact target altitude
            self._mission_start_t = time.time()
            self.get_logger().info(f'Takeoff complete → YAW_TRACK (hover at {self.hover_pos}) | 50s mission timer started')
            self.state = State.YAW_TRACK

    def _handle_yaw_track(self):
        """
        Core: hover in place, rotate yaw to center target in image.
        Only yaw changes. Position is held fixed.

        Yaw PD controller:
          ex_norm = (u_est - cx) / cx  in [-1, 1]
          delta_yaw = kp * ex_norm + kd * ex_norm_dot
          desired_yaw = current_yaw + delta_yaw
        """
        now = self.get_clock().now().nanoseconds / 1e9

        # Check if target is still being detected (timeout)
        if not self.target_detected or (now - self.last_det_time) > self.target_lost_timeout:
            self.target_detected = False
            self._stable_since = None   # reset stability on target loss
            self._pub_pos(self.hover_pos.tolist(), self.drone_yaw)
            self.get_logger().info(
                f'[{self.filter_type}] Target lost — holding yaw',
                throttle_duration_sec=2.0
            )
            return

        est = self.filt.get_pixel()
        if est is None:
            self._pub_pos(self.hover_pos.tolist(), self.drone_yaw)
            return

        # ── 50-second mission timeout ──────────────────────────────────────
        if self._mission_start_t and (time.time() - self._mission_start_t) >= self._mission_duration:
            self._auto_finish()
            return
        # ───────────────────────────────────────────────────────────────────

        u_est = est[0]
        # Normalize error to [-1, 1]
        ex_norm = (u_est - self.cx) / self.cx  # positive = target to the right

        # Deadband: ignore tiny errors (< 3% of half-width ≈ 13px)
        if abs(ex_norm) < 0.03:
            self._pub_pos(self.hover_pos.tolist(), self.drone_yaw)
            self.get_logger().info(
                f'[{self.filter_type}] Centered ex_norm={ex_norm:.3f} — holding',
                throttle_duration_sec=1.0
            )
            return

        # D term: derivative of normalized error (updated only on real detections in det_cb)
        ex_norm_dot = (ex_norm - self.prev_ex_norm) / max(now - self.last_det_time, 0.02)
        delta_yaw = self.kp_yaw * ex_norm + self.kd_yaw * ex_norm_dot
        desired_yaw = wrap_angle(self.drone_yaw + delta_yaw)

        # Hover at fixed position, only yaw changes
        self._pub_pos(self.hover_pos.tolist(), desired_yaw)

        self.get_logger().info(
            f'[{self.filter_type}] ex={ex_norm * self.cx:.0f}px({ex_norm:.2f}) '
            f'Δyaw={math.degrees(delta_yaw):.2f}° yaw={math.degrees(self.drone_yaw):.1f}°',
            throttle_duration_sec=0.5
        )

    def _auto_finish(self):
        """Target centered for 5 s → save logs and shut down the launch."""
        self.get_logger().info(
            f'[{self.filter_type}] ✓ Target centered ≥5 s — saving logs and shutting down'
        )
        # SIGINT to this process propagates through ros2 launch → kills all nodes + Gazebo
        os.kill(os.getpid(), signal.SIGINT)

    # ── PX4 helpers ──
    def _pub_pos(self, pos, yaw):
        msg = TrajectorySetpoint()
        msg.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pub.publish(msg)

    def _pub_cmd(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.param1, msg.param2, msg.command = p1, p2, cmd
        msg.target_system, msg.target_component = self.system_id, 1
        msg.source_system, msg.source_component, msg.from_external = 1, 1, True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HoverYawController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
