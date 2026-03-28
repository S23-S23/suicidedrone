#!/usr/bin/env python3
"""
PNG Guidance Node
Proportional Navigation Guidance for balloon interception.

Reference: "Precise Interception Flight Targets by Image-Based Visual Servoing
           of Multicopter", IEEE TIE 2025

Equations implemented:
  Eq.(8):  Current velocity direction angles (sigma_y, sigma_z)
             – from actual NED velocity vector (VehicleLocalPosition.vx/vy/vz)
             – falls back to body forward axis when speed < v_min_sigma
  Eq.(9):  Discrete PNG: sigma_d = K*(q_now − q_prev) + sigma_current
  Eq.(10): Desired velocity unit vector n_vd in NED spherical coordinates
  Eq.(14): Speed update: v = v + ka/rate  (clamped to v_max)

Subscriptions:
  /ibvs/los_angles                         – LOS angles from IBVSController
  /ibvs/target_detected                    – reset speed on re-acquisition
  drone{id}/fmu/out/vehicle_attitude       – quaternion FRD body → NED (fallback)
  drone{id}/fmu/out/vehicle_local_position – NED velocity for Eq.(8)

Publication:
  /png/velocity_cmd  (geometry_msgs/Twist) – NED velocity command
    linear.x = v_North [m/s]
    linear.y = v_East  [m/s]
    linear.z = v_Down  [m/s]
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from geometry_msgs.msg import Vector3, Twist
from std_msgs.msg import Bool


def quat_to_R(q):
    """
    Quaternion [w, x, y, z] → 3x3 rotation matrix.
    Returns R such that v_ned = R @ v_body  (FRD body → NED world).
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)


class PNGGuidance(Node):
    def __init__(self):
        super().__init__('png_guidance')

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter('system_id', 1)
        # PNG navigation gains (Eq.9) – typically 3 to 5
        self.declare_parameter('Ky', 1.0)     # elevation gain
        self.declare_parameter('Kz', 1.0)     # azimuth gain
        # Speed ramp (Eq.14)
        self.declare_parameter('ka',    0.2)  # acceleration increment [m/s per second]
        self.declare_parameter('v_max',  2.0) # maximum speed [m/s]
        self.declare_parameter('v_init', 0.5) # initial speed on intercept start [m/s]
        # Guidance loop rate [Hz] – should match camera and drone_manager mission timer
        self.declare_parameter('rate', 50.0)
        # Minimum NED speed [m/s] to trust velocity-derived sigma (Eq.8)
        # Below this threshold, fall back to body forward axis
        self.declare_parameter('v_min_sigma', 0.5)

        system_id    = self.get_parameter('system_id').value
        self.Ky      = self.get_parameter('Ky').value
        self.Kz      = self.get_parameter('Kz').value
        self.ka      = self.get_parameter('ka').value
        self.v_max   = self.get_parameter('v_max').value
        self.v_init  = self.get_parameter('v_init').value
        self.rate        = self.get_parameter('rate').value
        self.v_min_sigma = self.get_parameter('v_min_sigma').value

        # ── Runtime state ───────────────────────────────────────────────────
        self.R_e_b       = np.eye(3)  # FRD body → NED rotation (fallback for Eq.8)
        self.v_ned       = np.zeros(3)  # actual NED velocity [vN, vE, vD] m/s
        self.q_y_prev    = 0.0        # previous LOS elevation [rad]
        self.q_z_prev    = 0.0        # previous LOS azimuth   [rad]
        self.q_y_now     = 0.0        # current  LOS elevation [rad]
        self.q_z_now     = 0.0        # current  LOS azimuth   [rad]
        self.v_now       = self.v_init  # current speed magnitude [m/s]
        self.los_received      = False
        self.prev_detected     = False
        # On the very first guidance step after (re-)entry into INTERCEPT,
        # sigma_current from body heading diverges from the LOS direction
        # because the drone was hovering level while the balloon is below.
        # Using body heading would give sigma_yd ≈ 0 → no descent, or
        # worse, sigma_y > 0 (backward tilt from PX4 oscillation) → climb.
        # Fix: for the first step, aim sigma directly at the LOS angles.
        self.first_guidance_step = True

        # ── Subscriptions ───────────────────────────────────────────────────
        self.create_subscription(
            Vector3,
            '/ibvs/los_angles',
            self.los_callback,
            10,
        )
        self.create_subscription(
            Bool,
            '/ibvs/target_detected',
            self.target_detected_callback,
            10,
        )
        self.create_subscription(
            VehicleAttitude,
            f'drone{system_id}/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleLocalPosition,
            f'drone{system_id}/fmu/out/vehicle_local_position',
            self.local_position_callback,
            qos_profile_sensor_data,
        )

        # ── Publisher ───────────────────────────────────────────────────────
        self.vel_cmd_pub    = self.create_publisher(Twist,   '/png/velocity_cmd',      10)

        # PNG diagnostic topics (for analysis)
        self.pub_sigma_cur  = self.create_publisher(Vector3, '/png/sigma_current_deg', 10)
        self.pub_sigma_des  = self.create_publisher(Vector3, '/png/sigma_desired_deg', 10)
        self.pub_los_rate   = self.create_publisher(Vector3, '/png/los_rate_deg',      10)
        self.pub_speed_info = self.create_publisher(Vector3, '/png/speed_info',        10)

        # Guidance timer
        self.create_timer(1.0 / self.rate, self.guidance_loop)

        self.get_logger().info(
            f'PNGGuidance started: Ky={self.Ky}, Kz={self.Kz}, '
            f'ka={self.ka}, v_max={self.v_max}'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def attitude_callback(self, msg: VehicleAttitude):
        """Update R_e_b from VehicleAttitude quaternion (FRD body → NED)."""
        self.R_e_b = quat_to_R(msg.q)

    def local_position_callback(self, msg: VehicleLocalPosition):
        """Store actual NED velocity for Eq.(8) sigma computation."""
        self.v_ned = np.array([float(msg.vx), float(msg.vy), float(msg.vz)])

    def los_callback(self, msg: Vector3):
        """
        Receive LOS angles from IBVSController.
        Shift current → previous before updating (for discrete PNG Eq.9).

        On the very first measurement (or after re-acquisition reset),
        initialize prev == now so the initial LOS rate is zero.
        This prevents a large spurious Eq.(9) correction on the first step
        caused by (q_now − 0) when prev was never set.
        """
        if not self.los_received:
            # First valid measurement: no history yet → rate = 0
            self.q_y_prev = msg.x
            self.q_z_prev = msg.y
        else:
            self.q_y_prev = self.q_y_now
            self.q_z_prev = self.q_z_now
        self.q_y_now  = msg.x   # elevation
        self.q_z_now  = msg.y   # azimuth
        self.los_received = True

    def target_detected_callback(self, msg: Bool):
        """Reset LOS history and first-step flag on target re-acquisition.
        Speed is intentionally NOT reset so the drone immediately resumes
        at the speed it had when the target was lost, avoiding a slow restart
        that lets a moving balloon escape."""
        if msg.data and not self.prev_detected:
            # Reset LOS history so the first PNG step after re-acquisition
            # uses rate = 0 instead of stale prev values from before target loss
            self.los_received = False
            # Force first-step direct-aim behavior on re-entry
            self.first_guidance_step = True
            self.get_logger().info(
                f'PNG: target re-acquired, speed maintained={self.v_now:.2f} m/s'
            )
        self.prev_detected = msg.data

    # ── Guidance loop ─────────────────────────────────────────────────────────

    def guidance_loop(self):
        """
        Compute and publish NED velocity command using discrete PNG.

        Eq.(8):  sigma_y, sigma_z from body forward axis in NED
        Eq.(9):  sigma_yd = Ky*(q_y_now − q_y_prev) + sigma_y
                 sigma_zd = Kz*(q_z_now − q_z_prev) + sigma_z
        Eq.(10): n_vd = [cos(σ_yd)cos(σ_zd), cos(σ_yd)sin(σ_zd), −sin(σ_yd)]
        Eq.(14): v_now = min(v_now + ka/rate, v_max)
        """
        if not self.los_received:
            return

        # Eq.(8): Current velocity direction angles in NED
        #   Use actual NED velocity vector when speed is sufficient.
        #   Velocity control lets the drone move in any direction regardless of
        #   body heading, so body forward axis (R_e_b[:,0]) diverges from the
        #   true velocity direction during turns or sideslip → use v_ned instead.
        #   Fall back to body forward axis only when nearly stationary (speed < v_min_sigma).
        speed = np.linalg.norm(self.v_ned)
        if speed >= self.v_min_sigma:
            n_v = self.v_ned / speed   # unit vector of actual velocity
        else:
            n_v = self.R_e_b[:, 0]    # fallback: body forward axis in NED
        sigma_z = math.atan2(n_v[1], n_v[0])                               # azimuth [rad]
        sigma_y = math.atan2(-n_v[2], math.sqrt(n_v[0]**2 + n_v[1]**2))   # elevation [rad]

        # Eq.(9): Discrete PNG – desired velocity direction angles
        #
        # First-step override: on INTERCEPT entry the drone is hovering level
        # (sigma_y ≈ 0) while the balloon may be well below (q_y ≈ -15°).
        # Using sigma_current = 0 gives sigma_yd ≈ 0 → no initial descent.
        # Worse, any backward body tilt from PX4 oscillation yields sigma_y > 0
        # → positive sigma_yd → upward velocity command ("goes up first" symptom).
        #
        # Fix: on the first guidance step, set sigma_d = q_now so the drone
        # immediately aims at the LOS direction.  Subsequent steps use the
        # normal PNG law which tracks LOS rate changes from this baseline.
        # Eq.(9): Discrete PNG – desired velocity direction angles
        #
        # Baseline: q_now (current LOS angle) instead of sigma_current.
        #
        # Original formulation uses sigma_current (actual velocity direction) as
        # the baseline.  This fails for a multirotor in velocity control because:
        #   - sigma_y ≈ 0 (horizontal flight) when the balloon is clearly below
        #   - LOS rate ≈ 0 (q_y stable at -15°, stationary balloon on steady approach)
        #   - → sigma_yd = Ky*0 + 0 = 0 → no descent command (deadlock)
        #
        # Fix: use q_now as the baseline so the drone always aims at the current
        # LOS direction.  The Ky/Kz rate term still provides lead for moving targets.
        #   sigma_yd = q_y_now + Ky*(q_y_now - q_y_prev)
        #   sigma_zd = q_z_now + Kz*(q_z_now - q_z_prev)
        # For a stationary target: sigma_d = q_now → always aims at the balloon.
        # For a moving target: rate correction shifts sigma ahead of the LOS.
        if self.first_guidance_step:
            # First step: rate = 0 (prev == now), so sigma_d = q_now exactly
            self.first_guidance_step = False
            self.get_logger().info('PNG first step: baseline = q_now (rate=0)')

        sigma_yd = self.q_y_now + self.Ky * (self.q_y_now - self.q_y_prev)
        sigma_zd = self.q_z_now + self.Kz * (self.q_z_now - self.q_z_prev)

        # Eq.(14): Speed update with constant acceleration increment
        self.v_now = min(self.v_now + self.ka / self.rate, self.v_max)

        # Eq.(10): Desired velocity unit vector in NED
        #   NED x = North, y = East, z = Down
        #   n_vd_z = -sin(sigma_yd) because positive elevation = upward = negative NED-z
        n_vd = np.array([
            math.cos(sigma_yd) * math.cos(sigma_zd),   # North
            math.cos(sigma_yd) * math.sin(sigma_zd),   # East
            -math.sin(sigma_yd),                        # Down (negative = up)
        ])

        v_cmd = self.v_now * n_vd   # NED velocity command [m/s]

        # Publish as Twist (linear = NED velocity)
        twist           = Twist()
        twist.linear.x  = float(v_cmd[0])   # North
        twist.linear.y  = float(v_cmd[1])   # East
        twist.linear.z  = float(v_cmd[2])   # Down
        self.vel_cmd_pub.publish(twist)

        # ── Publish diagnostic topics ─────────────────────────────────────
        # Current velocity direction angles Eq.(8) [deg]
        msg_sc   = Vector3()
        msg_sc.x = math.degrees(sigma_y)
        msg_sc.y = math.degrees(sigma_z)
        self.pub_sigma_cur.publish(msg_sc)

        # Desired velocity direction angles Eq.(9) [deg]
        msg_sd   = Vector3()
        msg_sd.x = math.degrees(sigma_yd)
        msg_sd.y = math.degrees(sigma_zd)
        self.pub_sigma_des.publish(msg_sd)

        # LOS rate (q_now - q_prev) [deg/step]
        msg_lr   = Vector3()
        msg_lr.x = math.degrees(self.q_y_now - self.q_y_prev)
        msg_lr.y = math.degrees(self.q_z_now - self.q_z_prev)
        self.pub_los_rate.publish(msg_lr)

        # Speed info: v_now[m/s], actual_speed[m/s], sigma_source(0=vel, 1=body)
        msg_si   = Vector3()
        msg_si.x = float(self.v_now)
        msg_si.y = float(speed)
        msg_si.z = 0.0 if (speed >= self.v_min_sigma) else 1.0
        self.pub_speed_info.publish(msg_si)

        self.get_logger().info(
            f'PNG: σ_y={math.degrees(sigma_yd):.1f}°, σ_z={math.degrees(sigma_zd):.1f}°, '
            f'v={self.v_now:.2f} m/s, NED=({v_cmd[0]:.2f},{v_cmd[1]:.2f},{v_cmd[2]:.2f}), '
            f'speed={speed:.2f} m/s ({"vel" if speed >= self.v_min_sigma else "body"})',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = PNGGuidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
