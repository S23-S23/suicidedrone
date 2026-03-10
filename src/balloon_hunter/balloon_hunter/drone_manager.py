#!/usr/bin/env python3
"""
IBVS + PNG Interception Controller (v2 - Paper-faithful DKF)
=============================================================
Implementation of:
  [1] Yan et al. (2025) "Precise Interception Flight Targets by IBVS of Multicopter"
  [2] Yang et al. (2025) "High-Speed Interception Multicopter Control by IBVS"

DKF is based on [2] Algorithm 2 and Eq. 51:
- State includes normalized image coordinates
- IMU angular velocity drives image coordinate prediction via image Jacobian
- Delayed image measurements are corrected and re-propagated using stored IMU history

PNG + FOV Holding + Yaw PD based on [1] Eq. 6-14.
"""

import math
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
)
from geometry_msgs.msg import PoseStamped
from yolov8_msgs.msg import Yolov8Inference
from enum import Enum


# ──────────────────────────────────────────────
# Utility functions
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

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ──────────────────────────────────────────────
# Paper-faithful Delayed Kalman Filter (DKF)
# Based on Yang et al. (2025) Algorithm 2, Eq. 51
# ──────────────────────────────────────────────
class DelayedKalmanFilter:
    """
    Delayed Kalman Filter for image-space target position estimation.

    State: x = [p_bar_x, p_bar_y, p_bar_x_dot, p_bar_y_dot]^T
           where p_bar = [px/foc, py/foc] (normalized image coordinates)

    Key difference from simple KF:
    1. Prediction uses IMU angular velocity via image Jacobian (Eq. 51)
       - Drone rotation causes known image motion -> precise prediction
    2. Delayed measurement correction (Algorithm 2):
       - When YOLO measurement arrives (delayed by D steps),
         correct state at time t-D, then re-propagate to current time t
         using stored IMU history
    """

    def __init__(self, foc, R_b_c, dt=0.02, delay_steps=3, max_history=20):
        """
        Args:
            foc: focal length in pixels
            R_b_c: rotation matrix from camera frame to body frame (3x3)
            dt: prediction timestep (controller period)
            delay_steps: D, number of steps of image processing delay
            max_history: maximum IMU history buffer size
        """
        self.foc = foc
        self.R_b_c = R_b_c
        self.R_c_b = R_b_c.T  # body -> camera
        self.dt = dt
        self.D = delay_steps

        # State: [p_bar_x, p_bar_y, p_bar_x_dot, p_bar_y_dot]
        self.x = np.zeros(4)

        # Measurement matrix: observe [p_bar_x, p_bar_y]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise (tunable)
        q_pos = 1e-4
        q_vel = 1e-2
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        # Measurement noise (YOLO in normalized coords: ~5px / foc)
        r_meas = (5.0 / foc) ** 2
        self.R = np.diag([r_meas, r_meas])

        # Error covariance
        self.P = np.eye(4) * 0.1

        # History buffer for delay compensation (Algorithm 2)
        self.history = collections.deque(maxlen=max_history)

        self.initialized = False

    def _build_F(self, px_bar, py_bar, omega_cam, vz_c, pz_c):
        """
        Build state transition matrix with IMU-based image dynamics.
        Based on Eq. 54 (Yang et al.).
        """
        dt = self.dt
        wxc, wyc, wzc = omega_cam

        # Eq. 54: image coord dynamics from rotation + translation
        F_pp = np.eye(2) + np.array([
            [vz_c / pz_c + py_bar * wxc - 2 * px_bar * wyc,
             px_bar * wxc + wzc],
            [-py_bar * wyc - wzc,
             vz_c / pz_c + 2 * py_bar * wxc - px_bar * wyc]
        ]) * dt

        F = np.eye(4)
        F[0:2, 0:2] = F_pp
        F[0:2, 2:4] = np.eye(2) * dt
        return F

    def _compute_image_motion_from_imu(self, px_bar, py_bar, omega_body, vel_ned, R_e_b):
        """
        Compute image coordinate change from drone motion via image Jacobian (Eq. 51).

        The key insight: IMU tells us exactly how the drone rotated,
        so we can predict how much the target moved in the image
        even without a new YOLO measurement.
        """
        dt = self.dt

        # Transform: body -> camera
        omega_cam = self.R_c_b @ omega_body
        vel_body = R_e_b.T @ vel_ned
        vel_cam = self.R_c_b @ vel_body

        # Assumed depth (approximate; errors are second-order)
        pz_c = 15.0

        wxc, wyc, wzc = omega_cam
        vxc, vyc, vzc = vel_cam

        # Eq. 51: rotation contribution (dominant during yaw maneuvers)
        dp_rot = np.array([
            px_bar * py_bar * wxc - (1 + px_bar**2) * wyc + py_bar * wzc,
            (1 + py_bar**2) * wxc - px_bar * py_bar * wyc - px_bar * wzc
        ]) * dt

        # Eq. 51: translation contribution
        dp_trans = np.array([
            -1.0 / pz_c * vxc + px_bar / pz_c * vzc,
            -1.0 / pz_c * vyc + py_bar / pz_c * vzc
        ]) * dt

        return dp_rot + dp_trans, omega_cam, vzc, pz_c

    def predict(self, omega_body, vel_ned, R_e_b):
        """
        Prediction step using IMU data (Algorithm 2, line 4).

        Uses image Jacobian + IMU angular velocity for prediction,
        far more accurate than constant-velocity model during maneuvers.
        """
        if not self.initialized:
            return

        px_bar, py_bar = self.x[0], self.x[1]

        delta_p, omega_cam, vzc, pzc = self._compute_image_motion_from_imu(
            px_bar, py_bar, omega_body, vel_ned, R_e_b
        )

        F = self._build_F(px_bar, py_bar, omega_cam, vzc, pzc)

        # Save state + IMU BEFORE prediction (for re-propagation)
        self.history.append({
            'x': self.x.copy(),
            'P': self.P.copy(),
            'omega_body': omega_body.copy(),
            'vel_ned': vel_ned.copy(),
            'R_e_b': R_e_b.copy(),
        })

        # State prediction with IMU-based image motion + target velocity
        self.x[0] += delta_p[0] + self.x[2] * self.dt
        self.x[1] += delta_p[1] + self.x[3] * self.dt

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_pixel):
        """
        Correction with DELAYED measurement (Algorithm 2, lines 5-9).

        1. Roll back to state at time t-D
        2. Correct with measurement
        3. Re-propagate to now using stored IMU history
        """
        z_bar = np.array([z_pixel[0] / self.foc, z_pixel[1] / self.foc])

        if not self.initialized:
            self.x[0] = z_bar[0]
            self.x[1] = z_bar[1]
            self.x[2] = 0.0
            self.x[3] = 0.0
            self.P = np.eye(4) * 0.01
            self.initialized = True
            return

        D = self.D
        hist_len = len(self.history)

        if hist_len < D:
            self._standard_correction(z_bar)
            return

        # Step 1: Retrieve state at t-D
        idx_delayed = hist_len - D
        x_delayed = self.history[idx_delayed]['x'].copy()
        P_delayed = self.history[idx_delayed]['P'].copy()

        # Step 2: Correct at t-D (Eq. 34-36)
        H = self.H
        S = H @ P_delayed @ H.T + self.R
        K = P_delayed @ H.T @ np.linalg.inv(S)
        x_corrected = x_delayed + K @ (z_bar - H @ x_delayed)
        P_corrected = (np.eye(4) - K @ H) @ P_delayed

        # Step 3: Re-propagate t-D -> now using IMU history (Alg 2, lines 7-8)
        x_re = x_corrected.copy()
        P_re = P_corrected.copy()

        for i in range(idx_delayed, hist_len):
            h = self.history[i]
            px_bar, py_bar = x_re[0], x_re[1]

            delta_p, omega_cam, vzc, pzc = self._compute_image_motion_from_imu(
                px_bar, py_bar,
                h['omega_body'], h['vel_ned'], h['R_e_b']
            )
            F = self._build_F(px_bar, py_bar, omega_cam, vzc, pzc)

            x_re[0] += delta_p[0] + x_re[2] * self.dt
            x_re[1] += delta_p[1] + x_re[3] * self.dt
            P_re = F @ P_re @ F.T + self.Q

        # Step 4: Replace current state
        self.x = x_re
        self.P = P_re

    def _standard_correction(self, z_bar):
        H = self.H
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z_bar - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_estimate_pixel(self):
        """Get current estimate in pixel coordinates."""
        if not self.initialized:
            return None
        return np.array([
            self.x[0] * self.foc,
            self.x[1] * self.foc,
            self.x[2] * self.foc,
            self.x[3] * self.foc,
        ])


# ──────────────────────────────────────────────
# Mission State Machine
# ──────────────────────────────────────────────
class MissionState(Enum):
    IDLE = 0
    TAKEOFF = 1
    SEARCH = 2
    INTERCEPT = 3
    DONE = 4


# ──────────────────────────────────────────────
# Main IBVS + PNG Controller Node
# ──────────────────────────────────────────────
class IBVSPNGController(Node):
    def __init__(self):
        super().__init__('ibvs_png_controller')

        # ── Parameters ──
        self.declare_parameter('system_id', 1)
        self.declare_parameter('takeoff_height', 6.0)
        self.declare_parameter('img_width', 848)
        self.declare_parameter('img_height', 480)
        self.declare_parameter('fx', 454.8)
        self.declare_parameter('fy', 454.8)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('cam_pitch_deg', 0.0)
        self.declare_parameter('K_y', 3.0)
        self.declare_parameter('K_z', 3.0)
        self.declare_parameter('k_a', 2.0)
        self.declare_parameter('kp_yaw', 0.03)
        self.declare_parameter('kd_yaw', 0.01)
        self.declare_parameter('max_speed', 10.0)
        self.declare_parameter('search_speed', 3.0)
        self.declare_parameter('collision_distance', 0.5)
        self.declare_parameter('dkf_dt', 0.02)
        self.declare_parameter('dkf_delay_steps', 3)
        self.declare_parameter('detection_topic', '/Yolov8_Inference_1')
        self.declare_parameter('monitoring_topic', '/drone1/fmu/out/monitoring')

        # ── Get parameters ──
        self.system_id = self.get_parameter('system_id').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.img_w = self.get_parameter('img_width').value
        self.img_h = self.get_parameter('img_height').value
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
        self.search_speed = self.get_parameter('search_speed').value
        self.collision_dist = self.get_parameter('collision_distance').value

        dkf_dt = self.get_parameter('dkf_dt').value
        dkf_delay = self.get_parameter('dkf_delay_steps').value

        self.detection_topic = self.get_parameter('detection_topic').value
        self.monitoring_topic = self.get_parameter('monitoring_topic').value
        self.topic_prefix = f"drone{self.system_id}/fmu/"

        # ── Camera frame transform ──
        self.R_b_c = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=float)
        self.R_b_c = self.R_b_c @ rot_x(-self.cam_pitch)

        # ── State variables ──
        self.state = MissionState.IDLE
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_yaw = 0.0
        self.drone_pitch = 0.0
        self.drone_roll = 0.0
        self.drone_omega_body = np.zeros(3)
        self.nav_state = 0
        self.arming_state = 0
        self.last_cmd_time = 0.0
        self.search_start_pos = None
        self.search_distance_limit = 15.0

        # PNG state
        self.prev_qy = None
        self.prev_qz = None
        self.prev_sigma_y = None
        self.prev_sigma_z = None
        self.prev_ex = 0.0

        # Target tracking
        self.target_detected = False
        self.target_lost_count = 0
        self.target_lost_threshold = 50

        # ── Paper-faithful DKF ──
        self.dkf = DelayedKalmanFilter(
            foc=self.foc,
            R_b_c=self.R_b_c,
            dt=dkf_dt,
            delay_steps=dkf_delay,
            max_history=30
        )

        # ── Publishers ──
        self.ocm_pub = self.create_publisher(
            OffboardControlMode, f'{self.topic_prefix}in/offboard_control_mode', qos_profile_sensor_data)
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, f'{self.topic_prefix}in/trajectory_setpoint', qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(
            VehicleCommand, f'{self.topic_prefix}in/vehicle_command', qos_profile_sensor_data)
        self.target_pos_pub = self.create_publisher(
            PoseStamped, '/ibvs_target_position', 10)

        # ── Subscribers ──
        self.create_subscription(
            VehicleStatus, f'{self.topic_prefix}out/vehicle_status',
            self.status_cb, qos_profile_sensor_data)
        self.create_subscription(
            Monitoring, self.monitoring_topic,
            self.monitoring_cb, qos_profile_sensor_data)
        self.create_subscription(
            Yolov8Inference, self.detection_topic,
            self.detection_cb, 10)

        # ── Timers ──
        self.create_timer(0.1, self.ocm_timer_cb)
        self.create_timer(0.02, self.control_timer_cb)
        self.start_timer = self.create_timer(5.0, self.start_mission)

        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info('  IBVS + PNG Controller (Paper DKF v2)')
        self.get_logger().info(f'  PNG: Ky={self.Ky}, Kz={self.Kz}, ka={self.ka}')
        self.get_logger().info(f'  Yaw PD: kp={self.kp_yaw}, kd={self.kd_yaw}')
        self.get_logger().info(f'  DKF: dt={dkf_dt}, delay={dkf_delay} steps')
        self.get_logger().info(f'  Camera: {self.img_w}x{self.img_h}, foc={self.foc}')
        self.get_logger().info('═══════════════════════════════════════')

    # ──────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────
    def status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def monitoring_cb(self, msg: Monitoring):
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        if hasattr(msg, 'vel_x'):
            self.drone_vel = np.array([msg.vel_x, msg.vel_y, msg.vel_z])
        self.drone_yaw = msg.head
        self.drone_pitch = getattr(msg, 'pitch', 0.0)
        self.drone_roll = getattr(msg, 'roll', 0.0)
        if hasattr(msg, 'rollspeed'):
            self.drone_omega_body = np.array([
                msg.rollspeed, msg.pitchspeed, msg.yawspeed
            ])

    def detection_cb(self, msg: Yolov8Inference):
        if not msg.yolov8_inference:
            return
        det = msg.yolov8_inference[0]
        u = (det.left + det.right) * 0.5
        v = (det.top + det.bottom) * 0.5

        # DKF handles delay internally: correct past, re-propagate to now
        self.dkf.update(np.array([u, v]))
        self.target_detected = True
        self.target_lost_count = 0

        self.get_logger().info(f'[DET] bbox=({u:.0f},{v:.0f})', throttle_duration_sec=1.0)

    def start_mission(self):
        self.start_timer.cancel()
        if self.state == MissionState.IDLE:
            self.get_logger().info('Mission start -> TAKEOFF')
            self.state = MissionState.TAKEOFF

    # ──────────────────────────────────────────
    # Offboard heartbeat
    # ──────────────────────────────────────────
    def ocm_timer_cb(self):
        msg = OffboardControlMode()
        if self.state == MissionState.INTERCEPT:
            msg.position = False
            msg.velocity = True
        else:
            msg.position = True
            msg.velocity = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.ocm_pub.publish(msg)

    # ──────────────────────────────────────────
    # Main control loop (50Hz)
    # ──────────────────────────────────────────
    def control_timer_cb(self):
        R_e_b = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch) @ rot_x(self.drone_roll)

        # DKF predict with IMU (Algorithm 2, line 4)
        if self.dkf.initialized:
            self.dkf.predict(
                omega_body=self.drone_omega_body,
                vel_ned=self.drone_vel,
                R_e_b=R_e_b
            )

        if not self.target_detected:
            self.target_lost_count += 1

        if self.state == MissionState.IDLE:
            self._idle()
        elif self.state == MissionState.TAKEOFF:
            self._takeoff()
        elif self.state == MissionState.SEARCH:
            self._search()
        elif self.state == MissionState.INTERCEPT:
            self._intercept()
        elif self.state == MissionState.DONE:
            self._done()

    # ──────────────────────────────────────────
    # State handlers
    # ──────────────────────────────────────────
    def _idle(self):
        safe_z = max(self.drone_pos[2], -0.1)
        self._pub_position([self.drone_pos[0], self.drone_pos[1], safe_z])

    def _takeoff(self):
        target_alt = -self.takeoff_height
        now = self.get_clock().now().nanoseconds / 1e9

        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED or \
           self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            safe_z = max(self.drone_pos[2], -0.1)
            self._pub_position([self.drone_pos[0], self.drone_pos[1], safe_z])
        else:
            self._pub_position([self.drone_pos[0], self.drone_pos[1], target_alt])

        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if now - self.last_cmd_time > 1.0:
                self._pub_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.get_logger().info('ARM requested')
                self.last_cmd_time = now
            return

        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if now - self.last_cmd_time > 1.0:
                self._pub_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.get_logger().info('OFFBOARD requested')
                self.last_cmd_time = now
            return

        if abs(self.drone_pos[2] - target_alt) < 0.3:
            self.get_logger().info('Takeoff complete -> SEARCH')
            self.search_start_pos = self.drone_pos.copy()
            self.state = MissionState.SEARCH

    def _search(self):
        if self.search_start_pos is None:
            self.search_start_pos = self.drone_pos.copy()

        if self.target_detected and self.dkf.initialized:
            self.get_logger().info('Target acquired -> INTERCEPT')
            self._init_png_state()
            self.state = MissionState.INTERCEPT
            return

        dist = np.linalg.norm(self.drone_pos[:2] - self.search_start_pos[:2])
        if dist >= self.search_distance_limit:
            self._pub_position([self.drone_pos[0], self.drone_pos[1], -self.takeoff_height])
            self.get_logger().info('Search limit reached', throttle_duration_sec=3.0)
            return

        self._pub_position([5.0, 0.0, -self.takeoff_height])

    def _intercept(self):
        """Core IBVS + PNG controller (Yan et al. Eq. 3-14)."""
        if self.target_lost_count > self.target_lost_threshold:
            self.get_logger().warn('Target lost! -> SEARCH')
            self.target_detected = False
            self.prev_qy = None
            self.prev_qz = None
            self.state = MissionState.SEARCH
            return

        est = self.dkf.get_estimate_pixel()
        if est is None:
            self._pub_velocity([0.0, 0.0, 0.0], self.drone_yaw)
            return

        u_est, v_est = est[0], est[1]

        # Image error (Eq. 3)
        ex = u_est - self.cx
        ey = v_est - self.cy

        # LOS direction (Eq. 5)
        ray_cam = np.array([ex, ey, self.foc])
        ray_body = self.R_b_c @ ray_cam
        R_e_b = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch)
        nt = normalize(R_e_b @ ray_body)

        # LOS angles (Eq. 7)
        nt_xy = math.sqrt(nt[0]**2 + nt[1]**2)
        qy = math.atan2(nt[2], nt_xy) if nt_xy > 1e-9 else 0.0
        qz = math.atan2(nt[0], nt[1]) if abs(nt[1]) > 1e-9 else 0.0

        # Velocity angles (Eq. 8)
        speed = np.linalg.norm(self.drone_vel)
        if speed > 0.5:
            nv = normalize(self.drone_vel)
            nv_xy = math.sqrt(nv[0]**2 + nv[1]**2)
            sigma_y = math.atan2(nv[2], nv_xy) if nv_xy > 1e-9 else 0.0
            sigma_z = math.atan2(nv[0], nv[1]) if abs(nv[1]) > 1e-9 else 0.0
        else:
            sigma_y, sigma_z = qy, qz

        # PNG (Eq. 9)
        if self.prev_qy is not None:
            sigma_yd = self.Ky * wrap_angle(qy - self.prev_qy) + self.prev_sigma_y
            sigma_zd = self.Kz * wrap_angle(qz - self.prev_qz) + self.prev_sigma_z
        else:
            sigma_yd, sigma_zd = qy, qz

        self.prev_qy, self.prev_qz = qy, qz
        self.prev_sigma_y, self.prev_sigma_z = sigma_yd, sigma_zd

        # Desired velocity (Eq. 10, 14)
        cos_sy = math.cos(sigma_yd)
        nvd = normalize(np.array([
            cos_sy * math.sin(sigma_zd),
            cos_sy * math.cos(sigma_zd),
            math.sin(sigma_yd)
        ]))
        vd = clamp(speed + self.ka, 1.0, self.max_speed) * nvd

        # Yaw PD (Eq. 13)
        ex_dot = (ex - self.prev_ex) / 0.02
        self.prev_ex = ex
        desired_yaw = wrap_angle(
            self.drone_yaw + (self.kp_yaw * ex + self.kd_yaw * ex_dot) * 0.02
        )

        self._pub_velocity(vd, desired_yaw)

        self.get_logger().info(
            f'[IBVS] e=({ex:.0f},{ey:.0f}) q=({math.degrees(qy):.1f},{math.degrees(qz):.1f}) '
            f'v={speed:.1f} vd={np.linalg.norm(vd):.1f}',
            throttle_duration_sec=0.5
        )
        self._publish_debug_target(nt)

    def _done(self):
        self._pub_position(self.drone_pos.tolist(), yaw=self.drone_yaw)
        self.get_logger().info('DONE, hovering.', throttle_duration_sec=5.0)

    def _init_png_state(self):
        self.prev_qy = self.prev_qz = None
        self.prev_sigma_y = self.prev_sigma_z = None
        self.prev_ex = 0.0
        self.target_lost_count = 0

    def _publish_debug_target(self, nt):
        t = self.drone_pos + 10.0 * nt
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = float(t[0]), float(t[1]), float(t[2])
        self.target_pos_pub.publish(msg)

    # ── PX4 commands ──
    def _pub_position(self, pos, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pub.publish(msg)

    def _pub_velocity(self, vel, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [float(vel[0]), float(vel[1]), float(vel[2])]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pub.publish(msg)

    def _pub_command(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.param1, msg.param2, msg.command = p1, p2, command
        msg.target_system, msg.target_component = self.system_id, 1
        msg.source_system, msg.source_component, msg.from_external = 1, 1, True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = IBVSPNGController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()