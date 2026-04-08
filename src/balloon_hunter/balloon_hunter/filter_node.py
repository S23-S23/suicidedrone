#!/usr/bin/env python3
"""
Filter Node — DKF / EKF / DKF18 / EKF18 / GT pixel estimation
================================================================
Extracted from the monolithic drone_manager.py.

Subscribes:
  /target_info                              — TargetInfo (bbox from detector)
  drone{id}/fmu/out/monitoring              — Monitoring (pos, attitude)
  drone{id}/fmu/out/vehicle_local_position  — VehicleLocalPosition (vel)
  drone{id}/fmu/out/vehicle_angular_velocity— VehicleAngularVelocity (omega)
  drone{id}/fmu/out/vehicle_acceleration    — VehicleAcceleration (accel, 18-state only)
  /mission_state                            — String (state name)

Publishes:
  /filter_estimate  — Float32MultiArray [u, v, u_dot, v_dot, delay_steps]

Parameter 'filter_type': 'DKF', 'EKF', 'DKF18', 'EKF18', or 'GT'
"""

import math
import collections
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from px4_msgs.msg import (
    Monitoring,
    VehicleLocalPosition,
    VehicleAngularVelocity,
    VehicleAcceleration,
)
from suicide_drone_msgs.msg import TargetInfo
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


# ══════════════════════════════════════════════════════════════
# DKF (Paper-faithful, Yang et al. 2025, Algorithm 2)
# ══════════════════════════════════════════════════════════════
class DelayedKalmanFilter:
    """
    State: x = [p_bar_x, p_bar_y, p_bar_x_dot, p_bar_y_dot]^T
           where p_bar = pixel / foc (normalized image coordinates)

    IMU-based prediction via image Jacobian (Eq. 51).
    Delayed measurement correction + IMU re-propagation (Algorithm 2).

    delay_steps=0: standard EKF (no re-propagation).
    delay_steps>0: full DKF with history buffer and re-propagation.
    """

    def __init__(self, foc, R_b_c, dt=0.02, delay_steps=10, assumed_depth=10.0,
                 max_history=50, chi2_threshold=9.21):
        self.foc = foc
        self.R_b_c = R_b_c
        self.R_c_b = R_b_c.T
        self.dt = dt
        self.D = delay_steps
        self.pzc = assumed_depth
        self.chi2_threshold = chi2_threshold

        self.x = np.zeros(4)
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

        self.history.append({
            'x': self.x.copy(), 'P': self.P.copy(),
            'w': omega_body.copy(), 'v': vel_ned.copy(), 'R': R_e_b.copy(),
        })

        self.x[0] += dp[0] + self.x[2] * self.dt
        self.x[1] += dp[1] + self.x[3] * self.dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_pixel, delay_steps=None):
        zb = np.array([z_pixel[0] / self.foc, z_pixel[1] / self.foc])

        if not self.initialized:
            self.x[:2] = zb
            self.x[2:] = 0
            self.P = np.eye(4) * 0.01
            self.initialized = True
            return

        D = int(delay_steps) if delay_steps is not None else self.D
        D = max(0, min(D, self.D + 5))

        if D == 0:
            self._std_correct(zb)
            return

        hl = len(self.history)
        if hl < D:
            self._std_correct(zb)
            return

        idx = hl - D
        xd, Pd = self.history[idx]['x'].copy(), self.history[idx]['P'].copy()

        innovation = zb - self.H @ xd
        S = self.H @ Pd @ self.H.T + self.R
        d2 = float(innovation @ np.linalg.inv(S) @ innovation)
        if d2 > self.chi2_threshold:
            return

        K = Pd @ self.H.T @ np.linalg.inv(S)
        xc = xd + K @ innovation
        IKH = np.eye(4) - K @ self.H
        Pc = IKH @ Pd @ IKH.T + K @ self.R @ K.T

        for i in range(idx, hl):
            h = self.history[i]
            dp, wc, vzc, pz = self._imu_image_motion(
                xc[0], xc[1], h['w'], h['v'], h['R']
            )
            F = self._build_F(xc[0], xc[1], wc, vzc, pz)
            xc[0] += dp[0] + xc[2] * self.dt
            xc[1] += dp[1] + xc[3] * self.dt
            Pc = F @ Pc @ F.T + self.Q

        self.x, self.P = xc, Pc

    def _std_correct(self, zb):
        innovation = zb - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        d2 = float(innovation @ np.linalg.inv(S) @ innovation)
        if d2 > self.chi2_threshold:
            return
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ innovation
        IKH = np.eye(4) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

    def get_pixel(self):
        if not self.initialized:
            return None
        return np.array([self.x[0] * self.foc, self.x[1] * self.foc,
                         self.x[2] * self.foc, self.x[3] * self.foc])


# ══════════════════════════════════════════════════════════════
# DKF-18 (Paper-faithful, Yang et al. 2025, full 18-state)
# ══════════════════════════════════════════════════════════════
class DelayedKalmanFilter18:
    """
    Full 18-state Delayed Kalman Filter (Yang et al. 2025, Algorithm 2).

    State vector x (18,):
      q(0:4)   — quaternion [qw qx qy qz] (NED <- body-FRD)
      p_r(4:7) — relative position NED [m]
      v_r(7:10)— relative velocity NED [m/s]
      ip(10:12)— normalised image coords [u/f, v/f]
      bgyr(12:15) — gyro bias [rad/s]
      bacc(15:18) — accel bias [m/s^2]

    delay_steps=0 -> standard EKF; >0 -> full DKF with history replay.
    """

    SQ    = slice(0, 4)
    SPR   = slice(4, 7)
    SVR   = slice(7, 10)
    SIP   = slice(10, 12)
    SBGYR = slice(12, 15)
    SBACC = slice(15, 18)
    N     = 18

    def __init__(self, foc, R_b_c, dt=0.02, delay_steps=10,
                 assumed_depth=10.0, max_history=50, chi2_threshold=9.21):
        self.foc            = foc
        self.R_b_c          = R_b_c
        self.R_c_b          = R_b_c.T
        self.dt             = dt
        self.D              = delay_steps
        self.assumed_depth  = assumed_depth
        self.chi2_threshold = chi2_threshold

        self.g_ned = np.array([0.0, 0.0, 9.81])

        self.x = np.zeros(self.N)
        self.x[0] = 1.0

        self.H = np.zeros((2, self.N))
        self.H[0, 10] = 1.0
        self.H[1, 11] = 1.0

        self.Q = np.diag([
            1e-5, 1e-5, 1e-5, 1e-5,
            1e-4, 1e-4, 1e-4,
            1e-2, 1e-2, 1e-2,
            1e-4, 1e-4,
            1e-6, 1e-6, 1e-6,
            1e-4, 1e-4, 1e-4,
        ])

        px_noise = 5.0
        self.R_meas = np.diag([(px_noise / foc) ** 2] * 2)

        self.P = np.diag([
            1e-4, 1e-4, 1e-4, 1e-4,
            100., 100., 100.,
            1.0,  1.0,  1.0,
            1.0,  1.0,
            1e-4, 1e-4, 1e-4,
            0.01, 0.01, 0.01,
        ])

        self.history     = collections.deque(maxlen=max_history)
        self.initialized = False
        self._chi2_reject_count = 0

        self._last_omega = np.zeros(3)
        self._last_accel = np.zeros(3)

    @staticmethod
    def _q_to_R(q):
        qw, qx, qy, qz = q
        return np.array([
            [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qw*qz),    2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),      1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),      2*(qy*qz + qw*qx),    1 - 2*(qx**2 + qy**2)],
        ])

    @staticmethod
    def _euler_to_q(roll, pitch, yaw):
        cr, sr = math.cos(roll  / 2), math.sin(roll  / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw   / 2), math.sin(yaw   / 2)
        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
        ])

    def _f(self, x, omega_imu, accel_imu):
        dt = self.dt
        q    = x[self.SQ].copy()
        p_r  = x[self.SPR].copy()
        v_r  = x[self.SVR].copy()
        ip   = x[self.SIP].copy()
        bgyr = x[self.SBGYR].copy()
        bacc = x[self.SBACC].copy()

        omega_c = omega_imu - bgyr
        accel_c = accel_imu - bacc

        wx, wy, wz = omega_c
        Omega = 0.5 * np.array([
            [ 0,  -wx, -wy, -wz],
            [ wx,   0,   wz, -wy],
            [ wy,  -wz,   0,  wx],
            [ wz,   wy,  -wx,  0],
        ])
        q_new = q + Omega @ q * dt
        q_new /= (np.linalg.norm(q_new) + 1e-12)

        R_e_b = self._q_to_R(q)

        p_r_new = p_r + v_r * dt

        a_drone_ned = R_e_b @ accel_c + self.g_ned
        v_r_new = v_r - a_drone_ned * dt

        p_c  = self.R_c_b @ (R_e_b.T @ p_r)
        pzc  = max(abs(float(p_c[2])), 0.5)

        wc = self.R_c_b @ omega_c
        vc = self.R_c_b @ (R_e_b.T @ (-v_r))

        px_n, py_n = float(ip[0]), float(ip[1])
        wxc, wyc, wzc = wc
        vxc, vyc, vzc = vc

        dp_rot = np.array([
            px_n*py_n*wxc - (1 + px_n**2)*wyc + py_n*wzc,
            (1 + py_n**2)*wxc - px_n*py_n*wyc - px_n*wzc,
        ]) * dt

        dp_trans = np.array([
            -vxc/pzc + px_n*vzc/pzc,
            -vyc/pzc + py_n*vzc/pzc,
        ]) * dt

        ip_new = ip + dp_rot + dp_trans

        x_new = np.zeros(self.N)
        x_new[self.SQ]    = q_new
        x_new[self.SPR]   = p_r_new
        x_new[self.SVR]   = v_r_new
        x_new[self.SIP]   = ip_new
        x_new[self.SBGYR] = bgyr
        x_new[self.SBACC] = bacc
        return x_new

    def _numerical_F(self, x, omega_imu, accel_imu):
        eps = 1e-5
        f0  = self._f(x, omega_imu, accel_imu)
        F   = np.zeros((self.N, self.N))
        for i in range(self.N):
            xp = x.copy()
            xp[i] += eps
            xp[self.SQ] /= (np.linalg.norm(xp[self.SQ]) + 1e-12)
            F[:, i] = (self._f(xp, omega_imu, accel_imu) - f0) / eps
        return F

    def predict(self, omega_body, accel_body, R_e_b=None):
        if not self.initialized:
            return

        self._last_omega = omega_body.copy()
        self._last_accel = accel_body.copy()

        self.history.append({
            'x':     self.x.copy(),
            'P':     self.P.copy(),
            'omega': omega_body.copy(),
            'accel': accel_body.copy(),
        })

        F = self._numerical_F(self.x, omega_body, accel_body)
        self.x = self._f(self.x, omega_body, accel_body)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_pixel, delay_steps=None, roll=0.0, pitch=0.0, yaw=0.0):
        zb = np.array([z_pixel[0] / self.foc, z_pixel[1] / self.foc])

        if not self.initialized:
            self._initialize(zb, roll, pitch, yaw)
            return

        D = int(delay_steps) if delay_steps is not None else self.D
        D = max(0, min(D, self.D + 5))

        if D == 0 or len(self.history) < D:
            self._std_correct(zb)
            return

        hl  = len(self.history)
        idx = hl - D
        xd  = self.history[idx]['x'].copy()
        Pd  = self.history[idx]['P'].copy()

        innov = zb - self.H @ xd
        S     = self.H @ Pd @ self.H.T + self.R_meas
        d2    = float(innov @ np.linalg.solve(S, innov))
        if d2 > self.chi2_threshold:
            self._chi2_reject_count += 1
            if self._chi2_reject_count >= 3:
                self.initialized = False
            return
        self._chi2_reject_count = 0

        K   = Pd @ self.H.T @ np.linalg.inv(S)
        xc  = xd + K @ innov
        xc[self.SQ] /= (np.linalg.norm(xc[self.SQ]) + 1e-12)
        IKH = np.eye(self.N) - K @ self.H
        Pc  = IKH @ Pd @ IKH.T + K @ self.R_meas @ K.T

        for i in range(idx, hl):
            h   = self.history[i]
            F   = self._numerical_F(xc, h['omega'], h['accel'])
            xc  = self._f(xc, h['omega'], h['accel'])
            xc[self.SQ] /= (np.linalg.norm(xc[self.SQ]) + 1e-12)
            Pc  = F @ Pc @ F.T + self.Q

        self.x, self.P = xc, Pc

    def _initialize(self, zb, roll, pitch, yaw):
        q0 = self._euler_to_q(roll, pitch, yaw)
        self.x[self.SQ] = q0
        R_e_b = self._q_to_R(q0)

        bearing_cam  = np.array([zb[0], zb[1], 1.0])
        bearing_body = self.R_b_c @ bearing_cam
        bearing_ned  = R_e_b @ bearing_body
        self.x[self.SPR] = bearing_ned * self.assumed_depth

        self.x[self.SVR]   = np.zeros(3)
        self.x[self.SIP]   = zb
        self.x[self.SBGYR] = np.zeros(3)
        self.x[self.SBACC] = np.zeros(3)

        self.P = np.diag([
            1e-4, 1e-4, 1e-4, 1e-4,
            100., 100., 100.,
            1.0,  1.0,  1.0,
            1.0,  1.0,
            1e-4, 1e-4, 1e-4,
            0.01, 0.01, 0.01,
        ])

        self._chi2_reject_count = 0
        self.initialized = True

    def _std_correct(self, zb):
        innov = zb - self.H @ self.x
        S     = self.H @ self.P @ self.H.T + self.R_meas
        d2    = float(innov @ np.linalg.solve(S, innov))
        if d2 > self.chi2_threshold:
            self._chi2_reject_count += 1
            if self._chi2_reject_count >= 3:
                self.initialized = False
            return
        self._chi2_reject_count = 0
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ innov
        self.x[self.SQ] /= (np.linalg.norm(self.x[self.SQ]) + 1e-12)
        IKH = np.eye(self.N) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_meas @ K.T

    def get_pixel(self):
        if not self.initialized:
            return None

        ip   = self.x[self.SIP]
        q    = self.x[self.SQ]
        p_r  = self.x[self.SPR]
        v_r  = self.x[self.SVR]
        bgyr = self.x[self.SBGYR]

        R_e_b = self._q_to_R(q)

        p_c  = self.R_c_b @ (R_e_b.T @ p_r)
        pzc  = max(abs(float(p_c[2])), 0.5)

        omega_c = self.R_c_b @ (self._last_omega - bgyr)
        vc      = self.R_c_b @ (R_e_b.T @ (-v_r))

        px_n, py_n = float(ip[0]), float(ip[1])
        wxc, wyc, wzc = omega_c
        vxc, vyc, vzc = vc

        ip_dot = np.array([
            px_n*py_n*wxc - (1 + px_n**2)*wyc + py_n*wzc
            - vxc/pzc + px_n*vzc/pzc,
            (1 + py_n**2)*wxc - px_n*py_n*wyc - px_n*wzc
            - vyc/pzc + py_n*vzc/pzc,
        ])

        return np.array([
            ip[0] * self.foc, ip[1] * self.foc,
            ip_dot[0] * self.foc, ip_dot[1] * self.foc,
        ])


# ══════════════════════════════════════════════════════════════
# Filter ROS2 Node
# ══════════════════════════════════════════════════════════════
class FilterNode(Node):
    def __init__(self):
        super().__init__('filter_node')

        # ── Parameters ──
        self.declare_parameter('system_id', 1)
        self.declare_parameter('filter_type', 'DKF')
        self.declare_parameter('fx', 454.8)
        self.declare_parameter('fy', 454.8)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('cam_pitch_deg', 0.0)
        self.declare_parameter('dkf_dt', 0.02)
        self.declare_parameter('dkf_delay_steps', 2)
        self.declare_parameter('assumed_depth', 10.0)

        self.system_id    = self.get_parameter('system_id').value
        self.filter_type  = self.get_parameter('filter_type').value.upper()
        self.fx           = self.get_parameter('fx').value
        self.fy           = self.get_parameter('fy').value
        self.cx           = self.get_parameter('cx').value
        self.cy           = self.get_parameter('cy').value
        self.cam_pitch    = math.radians(self.get_parameter('cam_pitch_deg').value)
        self.foc          = self.fx
        dkf_dt            = self.get_parameter('dkf_dt').value
        dkf_delay         = self.get_parameter('dkf_delay_steps').value
        assumed_depth     = self.get_parameter('assumed_depth').value

        self.topic_prefix = f"drone{self.system_id}/fmu/"

        # ── Camera frame transform (iris depth camera) ──
        self.R_b_c = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=float) @ rot_x(-self.cam_pitch)

        # ── Create filter ──
        if self.filter_type == 'GT':
            self.filt = None
            self._gt_pixel = None
            self._gt_prev_uv = None
            self._gt_dt = dkf_dt
        elif self.filter_type == 'DKF18':
            self.filt = DelayedKalmanFilter18(
                self.foc, self.R_b_c, dkf_dt, dkf_delay,
                assumed_depth=assumed_depth, max_history=dkf_delay + 20
            )
        elif self.filter_type == 'EKF18':
            self.filt = DelayedKalmanFilter18(
                self.foc, self.R_b_c, dkf_dt, delay_steps=0,
                assumed_depth=assumed_depth, max_history=20
            )
        elif self.filter_type == 'DKF':
            self.filt = DelayedKalmanFilter(
                self.foc, self.R_b_c, dkf_dt, dkf_delay,
                assumed_depth=assumed_depth, max_history=dkf_delay + 20
            )
        else:  # EKF
            self.filt = DelayedKalmanFilter(
                self.foc, self.R_b_c, dkf_dt, delay_steps=0,
                assumed_depth=assumed_depth, max_history=20
            )

        # ── State variables ──
        self.drone_yaw = 0.0
        self.drone_pitch_val = 0.0
        self.drone_roll = 0.0
        self.drone_vel = np.zeros(3)
        self.drone_omega = np.zeros(3)
        self.drone_accel = np.zeros(3)
        self._last_delay_steps = 0
        self._last_meas_time = None          # track measurement age
        self._predict_frozen = False          # freeze prediction when no measurement
        self._max_predict_age = 0.5          # seconds without measurement before freezing
        self.mission_state = 'IDLE'

        # ── Publishers ──
        self.est_pub = self.create_publisher(Float32MultiArray, '/filter_estimate', 10)

        # ── Subscribers ──
        self.create_subscription(
            Monitoring, f'{self.topic_prefix}out/monitoring',
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
            VehicleAcceleration, f'{self.topic_prefix}out/vehicle_acceleration',
            self.accel_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            TargetInfo, '/target_info',
            self.det_cb, 10
        )
        self.create_subscription(
            String, '/mission_state',
            self.state_cb, 10
        )

        # ── Timer: 50Hz predict + publish ──
        self.create_timer(0.02, self.predict_and_publish)

        self.get_logger().info(
            f'FilterNode started: filter={self.filter_type}, '
            f'delay={dkf_delay} steps ({dkf_delay * dkf_dt * 1000:.0f}ms), '
            f'depth_prior={assumed_depth}m'
        )

    # ── Callbacks ──
    def monitoring_cb(self, msg: Monitoring):
        self.drone_yaw = msg.head
        self.drone_pitch_val = msg.pitch
        self.drone_roll = msg.roll

    def vlp_cb(self, msg: VehicleLocalPosition):
        if msg.v_xy_valid and msg.v_z_valid:
            self.drone_vel = np.array([msg.vx, msg.vy, msg.vz])

    def avel_cb(self, msg: VehicleAngularVelocity):
        self.drone_omega = np.array([msg.xyz[0], msg.xyz[1], msg.xyz[2]])

    def accel_cb(self, msg: VehicleAcceleration):
        self.drone_accel = np.array([msg.xyz[0], msg.xyz[1], msg.xyz[2]])

    def state_cb(self, msg: String):
        self.mission_state = msg.data

    def det_cb(self, msg: TargetInfo):
        active = self.mission_state in ('HOVER_INIT', 'TRACKING')
        if not active:
            return

        u = (msg.left + msg.right) * 0.5
        v = (msg.top + msg.bottom) * 0.5

        if self.filter_type == 'GT':
            uv = np.array([u, v])
            if self._gt_prev_uv is not None:
                duv = (uv - self._gt_prev_uv) / self._gt_dt
            else:
                duv = np.zeros(2)
            self._gt_prev_uv = uv.copy()
            self._gt_pixel = np.array([u, v, duv[0], duv[1]])
        else:
            delay_steps = None
            if self.filter_type in ('DKF18', 'EKF18'):
                self.filt.update(
                    np.array([u, v]),
                    delay_steps=delay_steps,
                    roll=self.drone_roll,
                    pitch=self.drone_pitch_val,
                    yaw=self.drone_yaw,
                )
            else:
                self.filt.update(np.array([u, v]), delay_steps=delay_steps)
            self._last_meas_time = self.get_clock().now()
            self._predict_frozen = False

    def predict_and_publish(self):
        active = self.mission_state in ('HOVER_INIT', 'TRACKING')
        if not active:
            return

        R_e_b = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch_val) @ rot_x(self.drone_roll)

        if self.filter_type == 'GT':
            est = self._gt_pixel
        else:
            # Freeze prediction when no measurement received for too long
            if (self._last_meas_time is not None and not self._predict_frozen):
                age = (self.get_clock().now() - self._last_meas_time).nanoseconds / 1e9
                if age > self._max_predict_age:
                    self._predict_frozen = True

            if self.filt.initialized and not self._predict_frozen:
                if self.filter_type in ('DKF18', 'EKF18'):
                    self.filt.predict(
                        omega_body=self.drone_omega,
                        accel_body=self.drone_accel,
                    )
                else:
                    self.filt.predict(
                        omega_body=self.drone_omega,
                        vel_ned=self.drone_vel,
                        R_e_b=R_e_b
                    )
            est = self.filt.get_pixel()

        if est is not None and active:
            _F32 = 3.4028234e+38
            def _safe_f32(v):
                try:
                    f = float(v)
                    return max(-_F32, min(_F32, f)) if math.isfinite(f) else 0.0
                except Exception:
                    return 0.0
            msg = Float32MultiArray()
            msg.data = [_safe_f32(est[0]), _safe_f32(est[1]),
                        _safe_f32(est[2]), _safe_f32(est[3]),
                        float(self._last_delay_steps)]
            self.est_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
