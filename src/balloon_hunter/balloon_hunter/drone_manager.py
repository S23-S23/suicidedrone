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
    VehicleAcceleration,        # added for 18-state DKF: specific force in FRD body
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

    Follows Yang et al. 2025 (Algorithm 2) without engineering hacks:
      - No state clamping
      - No covariance reset
      - Innovation gating (chi-squared test) for outlier rejection
      - Joseph form covariance update for numerical stability

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
        # chi^2 threshold for innovation gating (2 DOF, 99% → 9.21)
        self.chi2_threshold = chi2_threshold

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
        """Algorithm 2, line 4: propagate state with IMU data (Eq. 30-31)."""
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

        # State prediction (Eq. 30)
        self.x[0] += dp[0] + self.x[2] * self.dt
        self.x[1] += dp[1] + self.x[3] * self.dt
        # Covariance prediction (Eq. 31)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_pixel, delay_steps=None):
        """Algorithm 2, lines 5-9: correct with DELAYED measurement (Eq. 34-36).

        delay_steps: actual measured delay in predict-steps. If None, falls
                     back to the fixed self.D. Caller computes this from
                     ROS timestamps: round((t_now - t_image) / dt).
        """
        zb = np.array([z_pixel[0] / self.foc, z_pixel[1] / self.foc])

        if not self.initialized:
            self.x[:2] = zb
            self.x[2:] = 0
            self.P = np.eye(4) * 0.01
            self.initialized = True
            return

        D = int(delay_steps) if delay_steps is not None else self.D
        D = max(0, min(D, self.D + 5))  # 0 = EKF mode; cap at D+5 steps

        if D == 0:
            self._std_correct(zb)
            return

        hl = len(self.history)
        if hl < D:
            self._std_correct(zb)
            return

        # Step 1: Retrieve state at the estimated capture time (Eq. 34)
        idx = hl - D
        xd, Pd = self.history[idx]['x'].copy(), self.history[idx]['P'].copy()

        # Innovation gating at t-D: reject measurement if statistically inconsistent
        innovation = zb - self.H @ xd
        S = self.H @ Pd @ self.H.T + self.R
        d2 = float(innovation @ np.linalg.inv(S) @ innovation)
        if d2 > self.chi2_threshold:
            return  # outlier — discard measurement

        # Step 2: Correct state at t-D (Eq. 34-36), Joseph form for numerical stability
        K = Pd @ self.H.T @ np.linalg.inv(S)
        xc = xd + K @ innovation
        IKH = np.eye(4) - K @ self.H
        Pc = IKH @ Pd @ IKH.T + K @ self.R @ K.T

        # Step 3: Re-propagate from t-D to now using stored IMU data (Eq. 30-31)
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
        """Standard EKF correction (no delay). Used when D=0 or history insufficient."""
        innovation = zb - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        # Innovation gating: reject if Mahalanobis distance exceeds chi^2 threshold
        d2 = float(innovation @ np.linalg.inv(S) @ innovation)
        if d2 > self.chi2_threshold:
            return  # outlier — discard measurement
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ innovation
        # Joseph form: P = (I-KH)*P*(I-KH)^T + K*R*K^T (positive definiteness guarantee)
        IKH = np.eye(4) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

    def get_pixel(self):
        if not self.initialized:
            return None
        return np.array([self.x[0] * self.foc, self.x[1] * self.foc,
                         self.x[2] * self.foc, self.x[3] * self.foc])


# ══════════════════════════════════════════════════════════════
# DKF-18  (Paper-faithful, Yang et al. 2025, full 18-state)
# ══════════════════════════════════════════════════════════════
class DelayedKalmanFilter18:
    """
    Full 18-state Delayed Kalman Filter (Yang et al. 2025, Algorithm 2).

    State vector x (18,):
    ┌──────────┬───────┬────────────────────────────────────────────────────┐
    │ Component│ Index │ Description                                        │
    ├──────────┼───────┼────────────────────────────────────────────────────┤
    │ q        │  0:4  │ Unit quaternion [qw qx qy qz]  (NED ← body-FRD)   │
    │ p_r      │  4:7  │ Relative position NED = target − drone  [m]        │
    │ v_r      │  7:10 │ Relative velocity NED = target − drone  [m/s]      │
    │ ī_p      │ 10:12 │ Normalised image coords [u/f, v/f]                 │
    │ b_gyr    │ 12:15 │ Gyroscope bias in body-FRD  [rad/s]                │
    │ b_acc    │ 15:18 │ Accelerometer bias in body-FRD  [m/s²]             │
    └──────────┴───────┴────────────────────────────────────────────────────┘

    Why 18-state over the previous 4-state model
    ─────────────────────────────────────────────
    • pzc (depth) is NO longer a fixed assumed constant — it is tracked as
      the Z-component of p_r projected to the camera frame.  The 4-state
      model fixed pzc=7 m which degraded the image Jacobian at range.
    • Attitude is propagated via exact quaternion kinematics (q̇ = ½Ω(ω)q),
      NOT through the linearised image Jacobian.  This eliminates the
      break-down seen at ω_x ≈ 0.3–0.4 rad/s during intercept.
    • Gyro and accelerometer biases are estimated online, reducing drift
      in re-propagation during the D-step replay.
    • Re-propagation stores the actual VehicleAcceleration reading so the
      replay integrates real IMU, not a linearised approximation.

    Jacobian note
    ─────────────
    F is computed numerically (forward-difference, ε=1 e-5, 18+1 calls to
    _f per step).  This avoids the ~200-term analytical 18×18 derivation
    and the off-by-one sign errors that often plague quaternion-rate
    coupling expressions.  At 50 Hz prediction + 10 Hz measurement, the
    overhead is acceptable in simulation.

    delay_steps=0 → standard EKF (no re-propagation, no history).
    delay_steps>0 → full DKF with history replay (Algorithm 2, Eq. 34-36).
    """

    SQ    = slice(0, 4)    # quaternion  [qw qx qy qz]
    SPR   = slice(4, 7)    # relative position NED [m]
    SVR   = slice(7, 10)   # relative velocity NED [m/s]
    SIP   = slice(10, 12)  # normalised image coords
    SBGYR = slice(12, 15)  # gyro bias [rad/s]
    SBACC = slice(15, 18)  # accel bias [m/s²]
    N     = 18

    def __init__(self, foc, R_b_c, dt=0.02, delay_steps=10,
                 assumed_depth=10.0, max_history=50, chi2_threshold=9.21):
        # Reference: Yang et al. 2025, Section III-B filter design
        self.foc            = foc
        self.R_b_c          = R_b_c      # body-FRD → camera-optical
        self.R_c_b          = R_b_c.T    # camera-optical → body-FRD
        self.dt             = dt
        self.D              = delay_steps
        self.assumed_depth  = assumed_depth   # used ONLY at first-measurement init
        self.chi2_threshold = chi2_threshold  # χ²(2 d.o.f., 99%) = 9.21

        # NED gravity vector [m/s²] (down = positive in NED)
        self.g_ned = np.array([0.0, 0.0, 9.81])

        # Initial state: identity quaternion, all else zero
        self.x = np.zeros(self.N)
        self.x[0] = 1.0   # qw = 1

        # Measurement matrix H: directly observe normalised image coords ī_p
        # (Eq. 53 in Yang et al. — linear measurement model is the key advantage
        #  of including ī_p as an explicit state)
        self.H = np.zeros((2, self.N))
        self.H[0, 10] = 1.0
        self.H[1, 11] = 1.0

        # Process noise Q — tuned per component
        # q:    small (IMU-driven, model error is quaternion normalisation only)
        # p_r:  small random walk (target moves slowly relative to prediction step)
        # v_r:  larger (target manoeuvres + constant-velocity model error)
        # ī_p:  medium (image Jacobian approximation residual)
        # bgyr: very small (bias is nearly constant between calibrations)
        # bacc: small (bias drift slower than signal)
        self.Q = np.diag([
            1e-5, 1e-5, 1e-5, 1e-5,   # q
            1e-4, 1e-4, 1e-4,          # p_r
            1e-2, 1e-2, 1e-2,          # v_r
            1e-4, 1e-4,                # ī_p
            1e-6, 1e-6, 1e-6,          # b_gyr
            1e-4, 1e-4, 1e-4,          # b_acc
        ])

        # Measurement noise R: 5 pixel std-dev → normalised coords
        px_noise = 5.0
        self.R_meas = np.diag([(px_noise / foc) ** 2] * 2)

        # Initial covariance
        # p_r is highly uncertain (depth unknown), q is well-known from IMU
        self.P = np.diag([
            1e-4, 1e-4, 1e-4, 1e-4,   # q   — tight
            100., 100., 100.,           # p_r — large: unknown range
            1.0,  1.0,  1.0,           # v_r
            1.0,  1.0,                 # ī_p — wide init; converges after first real update
            1e-4, 1e-4, 1e-4,          # b_gyr
            0.01, 0.01, 0.01,          # b_acc
        ])

        self.history     = collections.deque(maxlen=max_history)
        self.initialized = False
        self._chi2_reject_count = 0    # consecutive chi2 gate rejections

        # Cache last IMU input for get_pixel() velocity computation
        self._last_omega = np.zeros(3)
        self._last_accel = np.zeros(3)

    # ── Quaternion / rotation utilities ──────────────────────

    @staticmethod
    def _q_to_R(q):
        """
        Unit quaternion [qw qx qy qz] → 3×3 rotation matrix R (NED ← body).
        Standard formula: R_ij from quaternion components.
        """
        qw, qx, qy, qz = q
        return np.array([
            [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qw*qz),    2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),      1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),      2*(qy*qz + qw*qx),    1 - 2*(qx**2 + qy**2)],
        ])

    @staticmethod
    def _euler_to_q(roll, pitch, yaw):
        """
        Euler ZYX angles → quaternion [qw qx qy qz] (NED ← body).
        Used only at filter initialisation to seed q from drone attitude.
        """
        cr, sr = math.cos(roll  / 2), math.sin(roll  / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw   / 2), math.sin(yaw   / 2)
        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
        ])

    # ── Nonlinear propagation f(x, u) ────────────────────────

    def _f(self, x, omega_imu, accel_imu):
        """
        One-step nonlinear state propagation: x_{k+1} = f(x_k, u_k).

        omega_imu : body-FRD angular velocity [rad/s]  (VehicleAngularVelocity.xyz)
        accel_imu : body-FRD specific force   [m/s²]   (VehicleAcceleration.xyz)
                    "Bias corrected acceleration including gravity" = what the IMU
                    physically measures = actual_accel_body − g_body.

        Equations implemented
        ─────────────────────
        1. Quaternion kinematics (exact, no small-angle approx):
               q_dot = ½ Ω(ω_c) q   where ω_c = ω_imu − b_gyr  (Eq. 28)
        2. Relative position:
               ṗ_r = v_r                                          (Eq. 29a)
        3. Relative velocity (target stationary assumption):
               v̇_r = −a_drone_ned                                (Eq. 29b)
               a_drone_ned = R_e_b @ accel_c + g_ned
               (accel_c = specific force; add g_ned to recover motion accel)
        4. Image coord propagation via image Jacobian (Eq. 51),
               now using pzc derived from state p_r instead of fixed constant.
        5. Bias random walk: ḃ = 0  (modelled via Q)
        """
        dt = self.dt
        q    = x[self.SQ].copy()
        p_r  = x[self.SPR].copy()
        v_r  = x[self.SVR].copy()
        ip   = x[self.SIP].copy()
        bgyr = x[self.SBGYR].copy()
        bacc = x[self.SBACC].copy()

        omega_c = omega_imu - bgyr   # bias-corrected gyro
        accel_c = accel_imu - bacc   # bias-corrected accel (still includes gravity)

        # 1. Quaternion kinematics: q_dot = ½ Ω(ω_c) q
        #    Ω(ω) is the 4×4 skew matrix for quaternion left-multiplication
        wx, wy, wz = omega_c
        Omega = 0.5 * np.array([
            [ 0,  -wx, -wy, -wz],
            [ wx,   0,   wz, -wy],
            [ wy,  -wz,   0,  wx],
            [ wz,   wy,  -wx,  0],
        ])
        q_new = q + Omega @ q * dt
        q_new /= (np.linalg.norm(q_new) + 1e-12)   # keep unit quaternion

        # Current rotation matrix from pre-update q
        R_e_b = self._q_to_R(q)

        # 2. Relative position (kinematic: dp_r/dt = v_r)
        p_r_new = p_r + v_r * dt

        # 3. Relative velocity
        #    accel_c = specific force (IMU output) = a_body − g_body
        #    → actual NED accel: a_ned = R_e_b @ accel_c + g_ned
        #    Ref: standard IMU mechanisation, e.g. Titterton & Weston Ch.3
        a_drone_ned = R_e_b @ accel_c + self.g_ned
        v_r_new = v_r - a_drone_ned * dt   # target stationary: a_rel = −a_drone

        # 4. Image coordinate propagation (Eq. 51, Yang et al.)
        #    KEY: pzc is now from the current state p_r, not a fixed constant.
        #    Project p_r into the camera frame; take Z as depth.
        p_c  = self.R_c_b @ (R_e_b.T @ p_r)
        pzc  = max(abs(float(p_c[2])), 0.5)   # guard: min 0.5 m depth

        wc = self.R_c_b @ omega_c             # angular velocity in camera frame
        # Camera translational velocity ≈ drone velocity (target stationary)
        # v_r = v_target − v_drone → v_drone ≈ −v_r
        vc = self.R_c_b @ (R_e_b.T @ (-v_r))

        px_n, py_n = float(ip[0]), float(ip[1])
        wxc, wyc, wzc = wc
        vxc, vyc, vzc = vc

        # Rotation part of image Jacobian (Eq. 51, rotation interaction matrix)
        dp_rot = np.array([
            px_n*py_n*wxc - (1 + px_n**2)*wyc + py_n*wzc,
            (1 + py_n**2)*wxc - px_n*py_n*wyc - px_n*wzc,
        ]) * dt

        # Translation part of image Jacobian (Eq. 51, translation interaction matrix)
        dp_trans = np.array([
            -vxc/pzc + px_n*vzc/pzc,
            -vyc/pzc + py_n*vzc/pzc,
        ]) * dt

        ip_new = ip + dp_rot + dp_trans

        # 5. Biases: constant model (random walk captured in Q)
        x_new = np.zeros(self.N)
        x_new[self.SQ]    = q_new
        x_new[self.SPR]   = p_r_new
        x_new[self.SVR]   = v_r_new
        x_new[self.SIP]   = ip_new
        x_new[self.SBGYR] = bgyr
        x_new[self.SBACC] = bacc
        return x_new

    def _numerical_F(self, x, omega_imu, accel_imu):
        """
        Numerical Jacobian F = ∂f/∂x  (forward differences, ε = 1e-5).

        Called once per predict step and once per re-propagation step.
        18+1 = 19 calls to _f per invocation.  Each column perturbs one
        state dimension; quaternion column is re-normalised to stay on S³.

        This replaces an analytical 18×18 derivation that would require
        explicit expressions for ∂(R_e_b @ accel)/∂q and ∂(image Jacobian)/∂q,
        which are prone to sign errors in the quaternion–rotation coupling.
        Ref: numerical differentiation standard practice in EKF implementations,
        e.g. Joan Solà "A micro Lie theory" (2018), Appendix D.
        """
        eps = 1e-5
        f0  = self._f(x, omega_imu, accel_imu)
        F   = np.zeros((self.N, self.N))
        for i in range(self.N):
            xp = x.copy()
            xp[i] += eps
            # Re-normalise quaternion after perturbation to remain on unit sphere
            xp[self.SQ] /= (np.linalg.norm(xp[self.SQ]) + 1e-12)
            F[:, i] = (self._f(xp, omega_imu, accel_imu) - f0) / eps
        return F

    # ── EKF / DKF public interface ────────────────────────────

    def predict(self, omega_body, accel_body, R_e_b=None):
        """
        Algorithm 2, line 4: IMU-driven state propagation.

        omega_body : VehicleAngularVelocity.xyz  [rad/s]  body-FRD
        accel_body : VehicleAcceleration.xyz     [m/s²]   body-FRD
                     (bias-corrected specific force, gravity included)
        R_e_b      : optional, not used in steady-state (q carries attitude)
        """
        if not self.initialized:
            return

        self._last_omega = omega_body.copy()
        self._last_accel = accel_body.copy()

        # Save state BEFORE propagation so update() can retrieve x(t-D), P(t-D)
        # History entry also stores IMU so re-propagation uses real measurements
        self.history.append({
            'x':     self.x.copy(),
            'P':     self.P.copy(),
            'omega': omega_body.copy(),
            'accel': accel_body.copy(),
        })

        # EKF predict: linearise f around current x
        F = self._numerical_F(self.x, omega_body, accel_body)

        # Propagate state (nonlinear, no approximation)
        self.x = self._f(self.x, omega_body, accel_body)

        # Propagate covariance (Eq. 31: P = F P Fᵀ + Q)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_pixel, delay_steps=None, roll=0.0, pitch=0.0, yaw=0.0):
        """
        Algorithm 2, lines 5-9: delayed measurement correction.

        z_pixel     : raw pixel detection (u, v)
        delay_steps : D = round((t_now − t_img) / dt) from ROS timestamps;
                      None → fall back to self.D (fixed parameter)
        roll/pitch/yaw : Euler angles used ONLY at first-call initialisation
                         to seed the quaternion from current drone attitude.
        """
        zb = np.array([z_pixel[0] / self.foc, z_pixel[1] / self.foc])

        if not self.initialized:
            self._initialize(zb, roll, pitch, yaw)
            return

        D = int(delay_steps) if delay_steps is not None else self.D
        D = max(0, min(D, self.D + 5))   # cap at D+5 to prevent runaway

        if D == 0 or len(self.history) < D:
            # Not enough history or EKF mode: standard single-step correction
            self._std_correct(zb)
            return

        hl  = len(self.history)
        idx = hl - D
        xd  = self.history[idx]['x'].copy()   # state at capture time t-D
        Pd  = self.history[idx]['P'].copy()   # covariance at t-D

        # Innovation gating at t-D: χ²(2 d.o.f.) test
        # Ref: Bar-Shalom et al. "Estimation with Applications" §5.4
        innov = zb - self.H @ xd
        S     = self.H @ Pd @ self.H.T + self.R_meas
        d2    = float(innov @ np.linalg.solve(S, innov))
        if d2 > self.chi2_threshold:
            self._chi2_reject_count += 1
            if self._chi2_reject_count >= 3:
                self.initialized = False   # filter has diverged; re-init on next det
            return   # statistical outlier — discard measurement
        self._chi2_reject_count = 0

        # Correct state at t-D (Eq. 34-36, Joseph form for numerical P.D.)
        K   = Pd @ self.H.T @ np.linalg.inv(S)
        xc  = xd + K @ innov
        xc[self.SQ] /= (np.linalg.norm(xc[self.SQ]) + 1e-12)
        IKH = np.eye(self.N) - K @ self.H
        Pc  = IKH @ Pd @ IKH.T + K @ self.R_meas @ K.T

        # Re-propagate from t-D to present using STORED IMU (Algorithm 2, line 8)
        # Each step uses the actual omega/accel recorded at that instant,
        # so the replay is physically faithful (no fixed-depth assumption).
        for i in range(idx, hl):
            h   = self.history[i]
            F   = self._numerical_F(xc, h['omega'], h['accel'])
            xc  = self._f(xc, h['omega'], h['accel'])
            xc[self.SQ] /= (np.linalg.norm(xc[self.SQ]) + 1e-12)
            Pc  = F @ Pc @ F.T + self.Q

        self.x, self.P = xc, Pc

    def _initialize(self, zb, roll, pitch, yaw):
        """
        Initialise state from first measurement.

        q    ← drone attitude converted from Euler ZYX
        p_r  ← target bearing × assumed_depth  (best guess at range)
        v_r  ← 0  (unknown initial velocity)
        ī_p  ← zb (measurement provides normalised image coords)
        biases ← 0  (assume zero at startup)
        """
        # Quaternion from current drone Euler attitude
        q0 = self._euler_to_q(roll, pitch, yaw)
        self.x[self.SQ] = q0
        R_e_b = self._q_to_R(q0)

        # Target bearing in camera optical frame: direction = [u/f, v/f, 1]
        # Scale by assumed_depth along the camera Z-axis to get initial p_r [NED]
        # Ref: standard monocular depth initialisation with a range prior
        bearing_cam  = np.array([zb[0], zb[1], 1.0])
        bearing_body = self.R_b_c @ bearing_cam          # body frame
        bearing_ned  = R_e_b @ bearing_body              # NED frame
        # bearing_cam[2] = 1 → camera-Z component = 1 unit = assumed_depth m
        self.x[self.SPR] = bearing_ned * self.assumed_depth

        self.x[self.SVR]   = np.zeros(3)     # unknown velocity → 0
        self.x[self.SIP]   = zb              # seed image state from measurement
        self.x[self.SBGYR] = np.zeros(3)
        self.x[self.SBACC] = np.zeros(3)

        # Keep p_r large (depth truly unknown); ī_p wide so first real measurement
        # can correct a false-positive init without being gated out by chi2
        self.P = np.diag([
            1e-4, 1e-4, 1e-4, 1e-4,   # q
            100., 100., 100.,           # p_r  (±10 m uncertainty)
            1.0,  1.0,  1.0,           # v_r
            1.0,  1.0,                 # ī_p — wide; tightens quickly after real updates
            1e-4, 1e-4, 1e-4,          # b_gyr
            0.01, 0.01, 0.01,          # b_acc
        ])

        self._chi2_reject_count = 0
        self.initialized = True

    def _std_correct(self, zb):
        """
        Standard (D=0) EKF update: innovation gating + Joseph form.
        Same statistical approach as the full DKF but without re-propagation.
        """
        innov = zb - self.H @ self.x
        S     = self.H @ self.P @ self.H.T + self.R_meas
        d2    = float(innov @ np.linalg.solve(S, innov))
        if d2 > self.chi2_threshold:
            self._chi2_reject_count += 1
            if self._chi2_reject_count >= 3:
                self.initialized = False   # filter has diverged; re-init on next det
            return
        self._chi2_reject_count = 0
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ innov
        self.x[self.SQ] /= (np.linalg.norm(self.x[self.SQ]) + 1e-12)
        IKH = np.eye(self.N) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_meas @ K.T

    def get_pixel(self):
        """
        Return [u, v, u_dot, v_dot] in pixel coords for the control loop.

        u, v      : from ī_p state × foc
        u_dot, v_dot : on-the-fly from image Jacobian using current state
                       (p_r provides real depth; _last_omega/accel used)
        """
        if not self.initialized:
            return None

        ip   = self.x[self.SIP]
        q    = self.x[self.SQ]
        p_r  = self.x[self.SPR]
        v_r  = self.x[self.SVR]
        bgyr = self.x[self.SBGYR]

        R_e_b = self._q_to_R(q)

        # Depth from state (no longer a fixed assumption)
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
        self.declare_parameter('collision_distance', 2.0)

        # Filter parameters
        self.declare_parameter('dkf_dt', 0.02)
        self.declare_parameter('dkf_delay_steps', 10)  # 200ms / 20ms = 10 steps
        self.declare_parameter('assumed_depth', 10.0)   # pzc for image Jacobian

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
        elif self.filter_type == 'DKF18':
            # Full 18-state DKF (Yang et al. 2025, Algorithm 2)
            # Uses quaternion kinematics + VehicleAcceleration for re-propagation
            self.filt = DelayedKalmanFilter18(
                self.foc, self.R_b_c, dkf_dt, dkf_delay,
                assumed_depth=assumed_depth, max_history=dkf_delay + 20
            )
        elif self.filter_type == 'EKF18':
            # 18-state EKF (same class, delay_steps=0 disables re-propagation)
            self.filt = DelayedKalmanFilter18(
                self.foc, self.R_b_c, dkf_dt, delay_steps=0,
                assumed_depth=assumed_depth, max_history=20
            )
        elif self.filter_type == 'DKF':
            self.filt = DelayedKalmanFilter(
                self.foc, self.R_b_c, dkf_dt, dkf_delay,
                assumed_depth=assumed_depth, max_history=dkf_delay + 20
            )
        else:  # EKF: same 4-state filter with delay_steps=0
            self.filt = DelayedKalmanFilter(
                self.foc, self.R_b_c, dkf_dt, delay_steps=0,
                assumed_depth=assumed_depth, max_history=20
            )

        # ── State variables ──
        self.state = State.IDLE
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_yaw = 0.0
        self.drone_pitch_val = 0.0
        self.drone_roll = 0.0
        self.drone_omega = np.zeros(3)
        self.drone_accel = np.zeros(3)   # body-FRD specific force [m/s²] (VehicleAcceleration)
        self._last_delay_steps = 0       # most recent measured delay D (for logger)
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
        # VehicleAcceleration: body-FRD bias-corrected specific force (gravity included)
        # Required by 18-state DKF for accurate IMU re-propagation
        self.create_subscription(
            VehicleAcceleration, f'{self.topic_prefix}out/vehicle_acceleration',
            self.accel_cb, qos_profile_sensor_data
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
        if self.filter_type in ('DKF18', 'EKF18'):
            self.get_logger().info(f'  Mode: 18-state (quaternion kinematics, depth from p_r)')
        else:
            self.get_logger().info(f'  Mode: 4-state (simplified image Jacobian)')
        self.get_logger().info(f'  Delay: {dkf_delay} steps ({dkf_delay * dkf_dt * 1000:.0f}ms)')
        self.get_logger().info(f'  Depth prior: {assumed_depth}m (18-state: init only)')
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

    def accel_cb(self, msg: VehicleAcceleration):
        # VehicleAcceleration.xyz: bias-corrected specific force in body-FRD [m/s²]
        # "Bias corrected acceleration (including gravity)" — as per .msg definition
        # This is exactly what the 18-state filter expects for accel_body
        self.drone_accel = np.array([msg.xyz[0], msg.xyz[1], msg.xyz[2]])

    def det_cb(self, msg: Yolov8Inference):
        if not msg.yolov8_inference:
            return
        if self.state not in (State.SEARCH, State.INTERCEPT):
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
            # Use fixed delay_steps from parameter (dynamic computation unreliable in sim
            # due to /clock publish latency; actual delay ≈ 1 YOLO frame ≈ 2 control steps)
            delay_steps = None

            if self.filter_type in ('DKF18', 'EKF18'):
                # 18-state filter also needs Euler angles for first-call quaternion seed
                self.filt.update(
                    np.array([u, v]),
                    delay_steps=delay_steps,
                    roll=self.drone_roll,
                    pitch=self.drone_pitch_val,
                    yaw=self.drone_yaw,
                )
            else:
                self.filt.update(np.array([u, v]), delay_steps=delay_steps)

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
            if self.filt.initialized and self.state in (State.SEARCH, State.INTERCEPT):
                if self.filter_type in ('DKF18', 'EKF18'):
                    # 18-state: pass angular velocity + specific force
                    # accel_body = VehicleAcceleration.xyz (bias-corrected, gravity included)
                    self.filt.predict(
                        omega_body=self.drone_omega,
                        accel_body=self.drone_accel,
                    )
                else:
                    # 4-state: original interface uses vel_ned + R_e_b
                    self.filt.predict(
                        omega_body=self.drone_omega,
                        vel_ned=self.drone_vel,
                        R_e_b=R_e_b
                    )
            est = self.filt.get_pixel()

        # Publish estimate for logger — data: [u, v, u_dot, v_dot, delay_steps]
        if est is not None:
            _F32 = 3.4028234e+38  # just under float32 max (3.4028234663852886e+38)
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
        # Require pitch rate to be low before entering INTERCEPT.
        # During the SEARCH yaw/pitch rotation omega_y can be ±3 rad/s; if we
        # switch mid-rotation the filter's first predict step uses that large
        # omega to push v_est far from the actual detection, causing a spurious
        # ey=-200 upward IBVS+PNG command on the very first control tick.
        pitch_stable = abs(self.drone_omega[1]) < 0.5   # rad/s
        if self.target_detected and filter_ready and pitch_stable:
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
            # 1. 타겟의 위치를 Gazebo ENU에서 PX4 NED로 변환
            # Gazebo (ENU): X = East, Y = North, Z = Up
            # PX4 (NED): X = North, Y = East, Z = Down
            # Gazebo ENU: X=East, Y=North, Z=Up → PX4 NED: X=North, Y=East, Z=Down
            target_pos_ned = np.array([
                self.target_world_pos[1],  # North = Gazebo Y
                self.target_world_pos[0],  # East  = Gazebo X
                -self.target_world_pos[2]  # Down  = -Gazebo Z
            ])

            # 2. 드론의 현재 위치(이미 NED)와 변환된 타겟 위치(NED) 사이의 거리 계산
            dist = np.linalg.norm(self.drone_pos - target_pos_ned)

            # 3. 충돌 조건 확인
            if dist < self.collision_dist:
                self.get_logger().info(f'COLLISION at dist={dist:.2f}m -> DONE')
                self._finish()
                return
    
        # ── Check target lost ──
        if self.target_lost_count > self.target_lost_threshold:
            self.get_logger().warn('Target lost! -> SEARCH')
            self.target_detected = False
            self.prev_qy = None
            self._prev_cmd_vel = np.zeros(3)
            if self.filt is not None:
                self.filt.initialized = False  # prevent predict-only divergence
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
        # Modified formula: sigma_yd = qy + Ky * dqy  (instead of prev_sigma + Ky*dqy)
        #
        # Why: the standard accumulation (prev_sigma + Ky*dqy) has two failure modes
        # when INTERCEPT starts from hover (sigma ≈ 0):
        #   1. Deadlock — if target is already below (qy < 0) but dqy ≈ 0,
        #      sigma_yd = 0 + 0 = 0 → drone never descends.
        #   2. Filter-convergence corruption — v_est jumps from init value to real
        #      position in 2-3 steps, making dqy transiently large-positive, so
        #      prev_sigma accumulates an upward command even though target is below.
        # Using qy as the baseline anchors sigma_yd to the actual LOS angle,
        # so even during filter transients the commanded direction stays downward.
        if self.prev_qy is not None:
            delta_qy = wrap_angle(qy - self.prev_qy)
            delta_qz = wrap_angle(qz - self.prev_qz)
            sigma_yd = qy + self.Ky * delta_qy
            sigma_zd = qz + self.Kz * delta_qz
        else:
            # First INTERCEPT step: seed directly from LOS angle.
            # sigma_y (velocity angle) ≈ 0 at hover — using it here would give a
            # horizontal command even when the target is already below.
            sigma_yd = qy
            sigma_zd = qz

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
