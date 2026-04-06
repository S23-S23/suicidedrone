#!/usr/bin/env python3
"""
PNG Guidance Node
Proportional Navigation Guidance for balloon interception.

Reference: "Precise Interception Flight Targets by Image-Based Visual Servoing
           of Multicopter", IEEE TIE 2025

Equations implemented:
  Eq.(8):  Current velocity direction angles (sigma_y, sigma_z)
  Eq.(9):  Discrete PNG with time-normalized LOS rate (camera-rate independent)
  Eq.(10): Desired velocity unit vector n_vd in NED spherical coordinates
  Eq.(14): Speed update: v = v + ka/rate  (clamped to v_max)

Subscriptions:
  /ibvs/output                             — IBVSOutput (detected, LOS angles, fov_yaw_rate, fov_vel_z)
  drone{id}/fmu/out/vehicle_attitude       — quaternion FRD body -> NED (fallback)
  drone{id}/fmu/out/vehicle_local_position — NED velocity for Eq.(8)

Publication:
  /png/guidance_cmd  (suicide_drone_msgs/GuidanceCmd)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from suicide_drone_msgs.msg import IBVSOutput, GuidanceCmd


def quat_to_R(q):
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix R (NED <- body-FRD)."""
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
        self.declare_parameter('Ky', 3.0)
        self.declare_parameter('Kz', 3.4)
        self.declare_parameter('ka',    2.0)
        self.declare_parameter('v_max',  10.0)
        self.declare_parameter('v_init', 3.5)
        self.declare_parameter('rate', 50.0)
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
        self.R_e_b       = np.eye(3)
        self.v_ned       = np.zeros(3)
        self.q_y_prev    = 0.0
        self.q_z_prev    = 0.0
        self.q_y_now     = 0.0
        self.q_z_now     = 0.0
        # Time-normalized LOS rate [rad/s] — camera-rate-independent PNG lead
        self.los_rate_y  = 0.0
        self.los_rate_z  = 0.0
        self._los_prev_time = None
        self.v_now       = self.v_init
        self.los_received        = False
        self.prev_detected       = False
        self.first_guidance_step = True
        self.fov_yaw_rate        = 0.0
        self.fov_vel_z           = 0.0
        self._last_detected_time = None      # for speed-reset hysteresis

        # ── Subscriptions ───────────────────────────────────────────────────
        self.create_subscription(
            IBVSOutput,
            '/ibvs/output',
            self.ibvs_output_callback,
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
        self.guidance_pub = self.create_publisher(GuidanceCmd, '/png/guidance_cmd', 25)

        # Guidance timer
        self.create_timer(1.0 / self.rate, self.guidance_loop)

        self.get_logger().info(
            f'PNGGuidance started: Ky={self.Ky}, Kz={self.Kz}, '
            f'ka={self.ka}, v_max={self.v_max}, v_init={self.v_init}'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def attitude_callback(self, msg: VehicleAttitude):
        self.R_e_b = quat_to_R(msg.q)

    def local_position_callback(self, msg: VehicleLocalPosition):
        self.v_ned = np.array([float(msg.vx), float(msg.vy), float(msg.vz)])

    def ibvs_output_callback(self, msg: IBVSOutput):
        """
        Receive IBVSOutput from IBVSController.

        On target re-acquisition (detected: False -> True), reset LOS history.
        Speed is intentionally NOT reset so the drone immediately resumes
        at the speed it had when the target was lost.

        Time-normalized LOS rate: los_rate [rad/s] = (q_now - q_prev) / dt_los
        """
        if msg.detected and not self.prev_detected:
            self.los_received        = False
            self.first_guidance_step = True
            self.los_rate_y          = 0.0
            self.los_rate_z          = 0.0
            # Only reset speed after prolonged loss (>3s), not brief detection gaps
            now = self.get_clock().now()
            if self._last_detected_time is not None:
                gap = (now - self._last_detected_time).nanoseconds / 1e9
            else:
                gap = float('inf')
            if gap > 3.0:
                self.v_now = self.v_init
                self.get_logger().info(
                    f'PNG: target re-acquired after {gap:.1f}s, speed reset to {self.v_now:.2f} m/s'
                )
            else:
                self.get_logger().info(
                    f'PNG: target re-acquired after {gap:.1f}s, speed maintained={self.v_now:.2f} m/s'
                )
        self.prev_detected = msg.detected

        if not msg.detected:
            return

        now = self.get_clock().now()
        if not self.los_received:
            self.q_y_prev   = msg.q_y
            self.q_z_prev   = msg.q_z
            self.los_rate_y = 0.0
            self.los_rate_z = 0.0
        else:
            dt = (now - self._los_prev_time).nanoseconds / 1e9
            if dt > 1e-4:
                self.los_rate_y = (msg.q_y - self.q_y_now) / dt
                self.los_rate_z = (msg.q_z - self.q_z_now) / dt
            self.q_y_prev = self.q_y_now
            self.q_z_prev = self.q_z_now
        self._los_prev_time      = now
        self._last_detected_time = now
        self.q_y_now         = msg.q_y
        self.q_z_now         = msg.q_z
        self.fov_yaw_rate    = msg.fov_yaw_rate
        self.fov_vel_z       = msg.fov_vel_z
        self.los_received    = True

    # ── Guidance loop ─────────────────────────────────────────────────────────

    def guidance_loop(self):
        """
        Compute and publish NED velocity command using discrete PNG.

        Eq.(8):  sigma_y, sigma_z from actual NED velocity
        Eq.(9):  sigma_yd = q_y_now + Ky * los_rate_y * dt_guidance
        Eq.(10): n_vd = [cos(s_yd)*cos(s_zd), cos(s_yd)*sin(s_zd), -sin(s_yd)]
        Eq.(14): v_now = min(v_now + ka/rate, v_max)
        """
        if not self.los_received:
            return

        # Eq.(8): Current velocity direction angles in NED
        speed = np.linalg.norm(self.v_ned)
        if speed >= self.v_min_sigma:
            n_v = self.v_ned / speed
        else:
            n_v = self.R_e_b[:, 0]  # fallback: body forward axis in NED
        sigma_z = math.atan2(n_v[1], n_v[0])
        sigma_y = math.atan2(-n_v[2], math.sqrt(n_v[0]**2 + n_v[1]**2))

        # Eq.(9): Discrete PNG — time-normalized LOS rate
        if self.first_guidance_step:
            self.first_guidance_step = False
            self.get_logger().info('PNG first step: los_rate=0, sigma_d = q_now')

        dt_guidance = 1.0 / self.rate
        sigma_yd = self.q_y_now + self.Ky * self.los_rate_y * dt_guidance
        sigma_zd = self.q_z_now + self.Kz * self.los_rate_z * dt_guidance

        # Eq.(14): Speed update
        self.v_now = min(self.v_now + self.ka / self.rate, self.v_max)

        # Eq.(10): Desired velocity unit vector in NED
        n_vd = np.array([
            math.cos(sigma_yd) * math.cos(sigma_zd),   # North
            math.cos(sigma_yd) * math.sin(sigma_zd),   # East
            -math.sin(sigma_yd),                        # Down (negative = up)
        ])

        v_cmd = self.v_now * n_vd

        # Publish GuidanceCmd: PNG velocity + IBVS fov corrections
        cmd                  = GuidanceCmd()
        cmd.header.stamp     = self.get_clock().now().to_msg()
        cmd.target_detected  = self.prev_detected
        cmd.vel_n            = float(v_cmd[0])
        cmd.vel_e            = float(v_cmd[1])
        cmd.vel_d            = float(v_cmd[2])  # PNG sigma_yd handles vertical; fov_vel_z removed (was double-acting)
        cmd.yaw_rate         = self.fov_yaw_rate
        self.guidance_pub.publish(cmd)

        self.get_logger().info(
            f'PNG: s_y={math.degrees(sigma_yd):.1f}deg, s_z={math.degrees(sigma_zd):.1f}deg, '
            f'v={self.v_now:.2f} m/s, NED=({cmd.vel_n:.2f},{cmd.vel_e:.2f},{cmd.vel_d:.2f})',
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
