#!/usr/bin/env python3
"""
IBVS + PNG Interception Controller — DKF vs EKF comparison
============================================================
Merged from:
  - bad_manager.py: IBVS+PNG interception logic (Yan et al. 2025)
  - drone_manager.py: DKF/EKF filters with stability guards, logger integration

Scenario:
  1. Takeoff → Search (fly forward) → Intercept (IBVS+PNG toward target)
  2. DKF compensates image processing delay via IMU re-propagation
  3. EKF uses same IMU motion model but no delay compensation
  4. Publishes filter estimate on /filter_estimate for logger

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
from rclpy.qos import qos_profile_sensor_data
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
from std_msgs.msg import Float32MultiArray, String
from enum import Enum


# ── Utility ──────────────────────────────────────────────────
def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)

def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ══════════════════════════════════════════════════════════════
# DKF (Paper-faithful, Yang et al. 2025, Algorithm 2)
# ══════════════════════════════════════════════════════════════
class DelayedKalmanFilter:
    """
    State: x = [p_bar_x, p_bar_y, p_bar_x_dot, p_bar_y_dot]^T
           where p_bar = pixel / foc (normalized image coordinates)

    IMU-based prediction via image Jacobian (Eq. 51).
    Delayed measurement correction + IMU re-propagation (Algorithm 2).
    Stability guards: state clipping + covariance reset.
    """

    def __init__(self, foc, R_b_c, dt=0.02, delay_steps=10, assumed_depth=10.0,
                 max_history=50):
        self.foc = foc
        self.R_b_c = R_b_c
        self.R_c_b = R_b_c.T
        self.dt = dt
        self.D = delay_steps
        self.pzc = assumed_depth

        self.x = np.zeros(4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.diag([1e-4, 1e-4, 1e-2, 1e-2])
        self.R = np.diag([(5.0 / foc) ** 2] * 2)
        self.P = np.eye(4) * 0.1

        self.history = collections.deque(maxlen=max_history)
        self.initialized = False

    def _imu_image_motion(self, px, py, omega_body, vel_ned, R_e_b):
        """Eq. 51: image coordinate change from drone motion."""
        dt = self.dt
        wc = self.R_c_b @ omega_body
        vc = self.R_c_b @ (R_e_b.T @ vel_ned)
        pz = self.pzc
        wxc, wyc, wzc = wc
        vxc, vyc, vzc = vc

        # Rotation contribution (dominant during attitude changes)
        dp_rot = np.array([
            px * py * wxc - (1 + px**2) * wyc + py * wzc,
            (1 + py**2) * wxc - px * py * wyc - px * wzc
        ]) * dt

        # Translation contribution (dominant during high-speed flight)
        dp_trans = np.array([
            -vxc / pz + px * vzc / pz,
            -vyc / pz + py * vzc / pz
        ]) * dt

        return dp_rot + dp_trans, wc, vzc, pz

    def _build_F(self, px, py, wc, vzc, pz):
        """Eq. 54: state transition matrix."""
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
        """Algorithm 2, line 4: propagate state with IMU data."""
        if not self.initialized:
            return
        px, py = self.x[0], self.x[1]
        dp, wc, vzc, pz = self._imu_image_motion(px, py, omega_body, vel_ned, R_e_b)
        F = self._build_F(px, py, wc, vzc, pz)

        # Save state + IMU BEFORE prediction (for delayed re-propagation)
        self.history.append({
            'x': self.x.copy(), 'P': self.P.copy(),
            'w': omega_body.copy(), 'v': vel_ned.copy(), 'R': R_e_b.copy(),
        })

        # State prediction
        self.x[0] += dp[0] + self.x[2] * self.dt
        self.x[1] += dp[1] + self.x[3] * self.dt
        self.P = F @ self.P @ F.T + self.Q

        # Stability guards
        self.x[0] = np.clip(self.x[0], -3.0, 3.0)
        self.x[1] = np.clip(self.x[1], -3.0, 3.0)
        self.x[2] = np.clip(self.x[2], -2.0, 2.0)
        self.x[3] = np.clip(self.x[3], -2.0, 2.0)
        if np.any(np.diag(self.P) > 10.0):
            self.P = np.eye(4) * 0.1

    def update(self, z_pixel):
        """Algorithm 2, lines 5-9: correct with DELAYED measurement."""
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

        # Step 1: Retrieve state at time t-D
        idx = hl - D
        xd, Pd = self.history[idx]['x'].copy(), self.history[idx]['P'].copy()

        # Step 2: Correct state at t-D with current measurement
        S = self.H @ Pd @ self.H.T + self.R
        K = Pd @ self.H.T @ np.linalg.inv(S)
        xc = xd + K @ (zb - self.H @ xd)
        Pc = (np.eye(4) - K @ self.H) @ Pd

        # Step 3: Re-propagate from t-D to now using stored IMU data
        for i in range(idx, hl):
            h = self.history[i]
            dp, wc, vzc, pz = self._imu_image_motion(
                xc[0], xc[1], h['w'], h['v'], h['R']
            )
            F = self._build_F(xc[0], xc[1], wc, vzc, pz)
            xc[0] += dp[0] + xc[2] * self.dt
            xc[1] += dp[1] + xc[3] * self.dt
            Pc = F @ Pc @ F.T + self.Q

        # Step 4: Replace current state with re-propagated result
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


# ══════════════════════════════════════════════════════════════
# EKF (IMU-driven, no delay compensation)
# ══════════════════════════════════════════════════════════════
class SimpleEKF:
    """
    Same IMU-driven motion model as DKF for fair comparison.
    Only difference: treats delayed measurement as current (no re-propagation).
    """

    def __init__(self, foc, R_b_c, dt=0.02, assumed_depth=10.0):
        self.foc = foc
        self.R_b_c = R_b_c
        self.R_c_b = R_b_c.T
        self.dt = dt
        self.pzc = assumed_depth

        self.x = np.zeros(4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.diag([1e-4, 1e-4, 1e-2, 1e-2])
        self.R = np.diag([(5.0 / foc) ** 2] * 2)
        self.P = np.eye(4) * 0.1
        self.initialized = False

    def _imu_image_motion(self, px, py, omega_body, vel_ned, R_e_b):
        dt = self.dt
        wc = self.R_c_b @ omega_body
        vc = self.R_c_b @ (R_e_b.T @ vel_ned)
        pz = self.pzc
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
        """No delay compensation — treats delayed measurement as current."""
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


# ══════════════════════════════════════════════════════════════
# State Machine
# ══════════════════════════════════════════════════════════════
class State(Enum):
    IDLE = 0
    TAKEOFF = 1
    SEARCH = 2
    INTERCEPT = 3
    DONE = 4


# ══════════════════════════════════════════════════════════════
# IBVS + PNG Interception Controller
# ══════════════════════════════════════════════════════════════
class InterceptionController(Node):
    def __init__(self):
        super().__init__('drone_manager')

        # ── Parameters ──
        self.declare_parameter('system_id', 1)
        self.declare_parameter('filter_type', 'DKF')
        self.declare_parameter('takeoff_height', 6.0)

        # Camera intrinsics
        self.declare_parameter('fx', 454.8)
        self.declare_parameter('fy', 454.8)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('cam_pitch_deg', 0.0)

        # PNG parameters (Eq. 9, Yan et al.)
        self.declare_parameter('K_y', 3.0)
        self.declare_parameter('K_z', 3.0)
        self.declare_parameter('k_a', 2.0)        # velocity gain (Eq. 14)

        # Yaw PD (Eq. 13)
        self.declare_parameter('kp_yaw', 0.01)
        self.declare_parameter('kd_yaw', 0.0003)

        # Speed limits
        self.declare_parameter('max_speed', 10.0)
        self.declare_parameter('collision_distance', 1.0)

        # Filter parameters
        self.declare_parameter('dkf_dt', 0.02)
        self.declare_parameter('dkf_delay_steps', 10)  # 200ms / 20ms = 10 steps
        self.declare_parameter('assumed_depth', 7.0)   # pzc for image Jacobian

        # Mission
        self.declare_parameter('mission_timeout', 60.0)

        # Topics
        self.declare_parameter('detection_topic', '/Yolov8_Inference_1')
        self.declare_parameter('monitoring_topic', '/drone1/fmu/out/monitoring')

        # ── Get parameters ──
        self.system_id = self.get_parameter('system_id').value
        self.filter_type = self.get_parameter('filter_type').value.upper()
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.cam_pitch = math.radians(self.get_parameter('cam_pitch_deg').value)
        self.foc = self.fx

        self.Ky = self.get_parameter('K_y').value
        self.Kz = self.get_parameter('K_z').value
        self.ka = self.get_parameter('k_a').value
        self.kp_yaw = self.get_parameter('kp_yaw').value
        self.kd_yaw = self.get_parameter('kd_yaw').value
        self.max_speed = self.get_parameter('max_speed').value
        self.collision_dist = self.get_parameter('collision_distance').value
        self.mission_timeout = self.get_parameter('mission_timeout').value

        dkf_dt = self.get_parameter('dkf_dt').value
        dkf_delay = self.get_parameter('dkf_delay_steps').value
        assumed_depth = self.get_parameter('assumed_depth').value

        self.topic_prefix = f"drone{self.system_id}/fmu/"

        # ── Camera frame transform ──
        # body→camera: body_x=cam_z(forward), body_y=cam_x(right), body_z=cam_y(down)
        self.R_b_c = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=float) @ rot_x(-self.cam_pitch)

        # ── Create filter ──
        if self.filter_type == 'GT':
            self.filt = None
            self._gt_pixel = None       # [u, v, u_dot, v_dot]
            self._gt_prev_uv = None     # previous [u, v] for numerical diff
            self._gt_dt = dkf_dt        # 0.02s (50Hz)
        elif self.filter_type == 'DKF':
            self.filt = DelayedKalmanFilter(
                self.foc, self.R_b_c, dkf_dt, dkf_delay,
                assumed_depth=assumed_depth, max_history=dkf_delay + 20
            )
        else:
            self.filt = SimpleEKF(
                self.foc, self.R_b_c, dkf_dt,
                assumed_depth=assumed_depth
            )

        # ── State variables ──
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
        self._mission_start_t = None

        # Target tracking
        self.target_detected = False
        self.target_lost_count = 0
        self.target_lost_threshold = 50   # 50 ticks at 50Hz = 1s
        self._det_tick = 0                # detection freshness counter

        # PNG state (Eq. 9)
        self.prev_qy = None
        self.prev_qz = None
        self.prev_sigma_y = None
        self.prev_sigma_z = None
        self.prev_ex = 0.0

        # ── Velocity buffering (PNG → PX4 safety layer) ──────────
        # Limits acceleration so PX4 doesn't command extreme attitudes
        self._prev_cmd_vel = np.zeros(3)   # last velocity sent to PX4
        self.MAX_ACCEL = 4.0               # m/s² max velocity change rate
        self.INITIAL_SPEED = 3.5           # m/s  starting speed for ramp-up

        # Target position (from target_mover or known)
        self.target_world_pos = None

        # ── Publishers ──
        self.ocm_pub = self.create_publisher(
            OffboardControlMode,
            f'{self.topic_prefix}in/offboard_control_mode',
            qos_profile_sensor_data
        )
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint,
            f'{self.topic_prefix}in/trajectory_setpoint',
            qos_profile_sensor_data
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            f'{self.topic_prefix}in/vehicle_command',
            qos_profile_sensor_data
        )
        self.est_pub = self.create_publisher(Float32MultiArray, '/filter_estimate', 10)
        self.target_pos_pub = self.create_publisher(PoseStamped, '/ibvs_target_position', 10)
        self.state_pub = self.create_publisher(String, '/mission_state', 10)

        # ── Subscribers ──
        self.create_subscription(
            VehicleStatus, f'{self.topic_prefix}out/vehicle_status',
            self.status_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Monitoring, self.get_parameter('monitoring_topic').value,
            self.monitoring_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            VehicleLocalPosition, f'{self.topic_prefix}out/vehicle_local_position',
            self.vlp_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            VehicleAngularVelocity, f'{self.topic_prefix}out/vehicle_angular_velocity',
            self.avel_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Yolov8Inference, self.get_parameter('detection_topic').value,
            self.det_cb, 10
        )
        self.create_subscription(Point, '/target_world_pos', self._target_pos_cb, 10)

        # ── Timers ──
        self.create_timer(0.1, self.ocm_cb)       # 10Hz offboard heartbeat
        self.create_timer(0.02, self.control_cb)   # 50Hz main control
        self.create_timer(5.0, self.start_mission)  # one-shot

        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info(f'  IBVS + PNG Interception Controller')
        self.get_logger().info(f'  Filter: *** {self.filter_type} ***')
        self.get_logger().info(f'  Delay: {dkf_delay} steps ({dkf_delay * dkf_dt * 1000:.0f}ms)')
        self.get_logger().info(f'  Depth: {assumed_depth}m')
        self.get_logger().info(f'  PNG: Ky={self.Ky}, Kz={self.Kz}, ka={self.ka}')
        self.get_logger().info(f'  Yaw PD: kp={self.kp_yaw}, kd={self.kd_yaw}')
        self.get_logger().info(f'  Max speed: {self.max_speed} m/s')
        self.get_logger().info('═══════════════════════════════════════')

    # ── Callbacks ────────────────────────────────────────────
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

        if self.filter_type == 'GT':
            uv = np.array([u, v])
            if self._gt_prev_uv is not None:
                duv = (uv - self._gt_prev_uv) / self._gt_dt
            else:
                duv = np.zeros(2)
            self._gt_prev_uv = uv.copy()
            self._gt_pixel = np.array([u, v, duv[0], duv[1]])
        else:
            self.filt.update(np.array([u, v]))

        self.target_detected = True
        self.target_lost_count = 0
        self._det_tick = 0  # reset freshness counter

    def _target_pos_cb(self, msg: Point):
        self.target_world_pos = np.array([msg.x, msg.y, msg.z])

    def start_mission(self):
        if self.state == State.IDLE:
            self.get_logger().info(f'Mission start ({self.filter_type}) -> TAKEOFF')
            self.state = State.TAKEOFF

    # ── Offboard heartbeat (10Hz) ────────────────────────────
    def ocm_cb(self):
        msg = OffboardControlMode()
        if self.state == State.INTERCEPT:
            msg.position = False
            msg.velocity = True
        else:
            msg.position = True
            msg.velocity = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.ocm_pub.publish(msg)

    # ── Main control loop (50Hz) ─────────────────────────────
    def control_cb(self):
        state_msg = String()
        state_msg.data = self.state.name
        self.state_pub.publish(state_msg)

        R_e_b = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch_val) @ rot_x(self.drone_roll)

        # Filter predict step (DKF/EKF use IMU; GT bypasses filter entirely)
        if self.filter_type == 'GT':
            est = self._gt_pixel
        else:
            if self.filt.initialized:
                self.filt.predict(
                    omega_body=self.drone_omega,
                    vel_ned=self.drone_vel,
                    R_e_b=R_e_b
                )
            est = self.filt.get_pixel()

        # Publish estimate for logger
        if est is not None:
            msg = Float32MultiArray()
            msg.data = [float(est[0]), float(est[1]), float(est[2]), float(est[3])]
            self.est_pub.publish(msg)

        # Track target loss: _det_tick increments every control tick,
        # reset to 0 in det_cb when a new detection arrives
        self._det_tick += 1
        if self._det_tick > 2:  # no detection for >2 ticks (~40ms)
            self.target_lost_count += 1

        # State machine
        if self.state == State.IDLE:
            self._idle()
        elif self.state == State.TAKEOFF:
            self._takeoff()
        elif self.state == State.SEARCH:
            self._search()
        elif self.state == State.INTERCEPT:
            self._intercept()
        elif self.state == State.DONE:
            self._done()

    # ── State handlers ───────────────────────────────────────
    def _idle(self):
        safe_z = max(self.drone_pos[2], -0.1)
        self._pub_pos([self.drone_pos[0], self.drone_pos[1], safe_z])

    def _takeoff(self):
        alt = -self.takeoff_height
        now = self.get_clock().now().nanoseconds / 1e9

        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED or \
           self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            safe_z = max(self.drone_pos[2], -0.1)
            self._pub_pos([self.drone_pos[0], self.drone_pos[1], safe_z])
        else:
            self._pub_pos([self.drone_pos[0], self.drone_pos[1], alt])

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
            self.get_logger().info('Takeoff complete -> SEARCH')
            self.state = State.SEARCH

    def _search(self):
        """Fly forward until target is detected."""
        filter_ready = (self._gt_pixel is not None) if self.filter_type == 'GT' else self.filt.initialized
        if self.target_detected and filter_ready:
            self.get_logger().info(f'Target acquired ({self.filter_type}) -> INTERCEPT')
            self._init_png_state()
            self._mission_start_t = time.time()
            self.state = State.INTERCEPT
            return

        # Fly forward (positive Y in NED = forward in typical Gazebo setup)
        self._pub_pos([5.0, 0.0, -self.takeoff_height])

    def _intercept(self):
        """Core IBVS + PNG controller (Yan et al. 2025, Eq. 3-14)."""
        DT = 0.02  # control period (50Hz)

        # ── Mission timeout ──
        if self._mission_start_t and \
           (time.time() - self._mission_start_t) >= self.mission_timeout:
            self.get_logger().info(f'[{self.filter_type}] Mission timeout -> DONE')
            self._finish()
            return

        # ── Collision detection ──
        if self.target_world_pos is not None:
            target_ned = np.array([
                self.target_world_pos[0],
                self.target_world_pos[1],
                -self.target_world_pos[2]
            ])
            dist = np.linalg.norm(self.drone_pos - target_ned)
            if dist < self.collision_dist:
                self.get_logger().info(
                    f'[{self.filter_type}] COLLISION at dist={dist:.2f}m -> DONE'
                )
                self._finish()
                return

        # ── Check target lost ──
        if self.target_lost_count > self.target_lost_threshold:
            self.get_logger().warn('Target lost! -> SEARCH')
            self.target_detected = False
            self.prev_qy = None
            self._prev_cmd_vel = np.zeros(3)
            self.state = State.SEARCH
            return

        if self.filter_type == 'GT':
            est = self._gt_pixel
        else:
            est = self.filt.get_pixel()
        if est is None:
            # ── [완충] 데이터 없으면 감속하며 대기 ──
            self._prev_cmd_vel *= 0.9
            self._pub_vel(self._prev_cmd_vel, self.drone_yaw)
            return

        u_est, v_est = est[0], est[1]

        # ── Image error (Eq. 3) ──
        ex = u_est - self.cx
        ey = v_est - self.cy

        # ── [완충] stale detection 무시: FOV 밖 pixel이면 PNG 업데이트 금지 ──
        stale = self._det_tick > 10  # 200ms 이상 감지 없음 (20Hz det 기준 ~4회 미수신)
        if stale:
            # 마지막 명령 속도를 유지 (방향 유지, 거의 감속 없음)
            self._prev_cmd_vel *= 0.995
            self._pub_vel(self._prev_cmd_vel, self.drone_yaw)
            self.get_logger().info(
                f'[{self.filter_type}] stale det ({self._det_tick} ticks), coasting',
                throttle_duration_sec=0.5
            )
            return

        # ── LOS direction from image (Eq. 5) ──
        ray_cam = np.array([ex, ey, self.foc])
        ray_body = self.R_b_c @ ray_cam
        R_e_b = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch_val)
        ray_ned = R_e_b @ ray_body
        nt = normalize(ray_ned)

        # ── LOS angles (Eq. 7) ──
        nt_xy = math.sqrt(nt[0]**2 + nt[1]**2)
        qy = math.atan2(nt[2], nt_xy) if nt_xy > 1e-9 else 0.0
        qz = math.atan2(nt[0], nt[1]) if abs(nt[1]) > 1e-9 else 0.0

        # ── Velocity angles (Eq. 8) ──
        speed = np.linalg.norm(self.drone_vel)
        if speed > 0.5:
            nv = normalize(self.drone_vel)
            nv_xy = math.sqrt(nv[0]**2 + nv[1]**2)
            sigma_y = math.atan2(nv[2], nv_xy) if nv_xy > 1e-9 else 0.0
            sigma_z = math.atan2(nv[0], nv[1]) if abs(nv[1]) > 1e-9 else 0.0
        else:
            sigma_y = qy
            sigma_z = qz

        # ── PNG desired velocity angle (Eq. 9) ──
        if self.prev_qy is not None:
            delta_qy = wrap_angle(qy - self.prev_qy)
            delta_qz = wrap_angle(qz - self.prev_qz)
            sigma_yd = self.Ky * delta_qy + self.prev_sigma_y
            sigma_zd = self.Kz * delta_qz + self.prev_sigma_z
        else:
            sigma_yd = sigma_y
            sigma_zd = sigma_z

        self.prev_qy = qy
        self.prev_qz = qz
        self.prev_sigma_y = sigma_yd
        self.prev_sigma_z = sigma_zd

        # ── Desired velocity direction (Eq. 10) ──
        cos_sy = math.cos(sigma_yd)
        nvd = normalize(np.array([
            cos_sy * math.sin(sigma_zd),
            cos_sy * math.cos(sigma_zd),
            math.sin(sigma_yd)
        ]))

        # ── Desired speed (Eq. 14) ──
        # ── [완충] 초기 ramp-up: 처음에는 낮은 속도부터 시작 ──
        vd_mag = clamp(speed + self.ka, self.INITIAL_SPEED, self.max_speed)
        vd_raw = vd_mag * nvd

        # ══════════════════════════════════════════════════════════
        # ── [완충] Acceleration limiting (논문 Eq.17-23 attitude
        #    controller 대체) ──
        #    PX4 velocity controller가 극단적 attitude를 생성하지
        #    않도록 매 틱 velocity 변화량을 MAX_ACCEL * DT로 제한.
        #    논문의 sat(ω, ωm) (Eq. 23)와 동일한 역할.
        # ══════════════════════════════════════════════════════════
        dv = vd_raw - self._prev_cmd_vel
        dv_norm = np.linalg.norm(dv)
        max_dv = self.MAX_ACCEL * DT  # 최대 허용 속도 변화 (m/s per tick)
        if dv_norm > max_dv:
            dv = dv * (max_dv / dv_norm)
        vd = self._prev_cmd_vel + dv
        self._prev_cmd_vel = vd.copy()

        # ── Yaw: LOS 방향으로 직접 설정 ──
        # ── [완충] pixel error 기반 PD 대신, NED LOS 방향으로 yaw 설정.
        #    PX4가 yaw 변화를 자체 rate limit으로 부드럽게 처리. ──
        desired_yaw = math.atan2(nt[1], nt[0])  # NED에서 LOS 방향

        # ── Publish velocity command ──
        self._pub_vel(vd, desired_yaw)

        # ── Debug ──
        self.get_logger().info(
            f'[{self.filter_type}] e=({ex:.0f},{ey:.0f}) '
            f'q=({math.degrees(qy):.1f},{math.degrees(qz):.1f})deg '
            f'v={speed:.1f} vd={np.linalg.norm(vd):.1f}m/s '
            f'w=({self.drone_omega[0]:.2f},{self.drone_omega[1]:.2f},{self.drone_omega[2]:.2f})',
            throttle_duration_sec=0.5
        )

        self._publish_debug_target(nt)

    def _done(self):
        self._pub_pos(self.drone_pos.tolist(), yaw=self.drone_yaw)
        self.get_logger().info('Mission DONE, hovering.', throttle_duration_sec=5.0)

    # ── Helpers ──────────────────────────────────────────────
    def _init_png_state(self):
        self.prev_qy = None
        self.prev_qz = None
        self.prev_sigma_y = None
        self.prev_sigma_z = None
        self.prev_ex = 0.0
        self.target_lost_count = 0
        self._prev_cmd_vel = self.drone_vel.copy()  # 현재 속도부터 시작
        self._det_tick = 0

    def _finish(self):
        self.state = State.DONE
        self.get_logger().info(f'[{self.filter_type}] Finishing — shutting down in 3s')
        self.create_timer(3.0, lambda: os.kill(os.getpid(), signal.SIGINT))

    def _publish_debug_target(self, nt):
        assumed_dist = 10.0
        target_est = self.drone_pos + assumed_dist * nt
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(target_est[0])
        msg.pose.position.y = float(target_est[1])
        msg.pose.position.z = float(target_est[2])
        self.target_pos_pub.publish(msg)

    # ── PX4 command helpers ──────────────────────────────────
    def _pub_pos(self, pos, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pub.publish(msg)

    def _pub_vel(self, vel, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [float(vel[0]), float(vel[1]), float(vel[2])]
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
    node = InterceptionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
