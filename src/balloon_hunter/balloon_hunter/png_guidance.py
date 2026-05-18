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

Formation-Referenced Dynamic Bias (FRDB):
  Follower drones (system_id != leader_id) compute az_bias from inter-drone geometry
  instead of a hardcoded constant.  The leader broadcasts its LOS direction; each
  follower measures its signed lateral offset from the leader's LOS line and converts
  that to a bias angle:

    d_perp  = dot(my_pos - leader_pos, e_perp_right)   [m, RTK-based, always accurate]
    az_bias = atan2(d_perp, r_estimated) * bias_gain_K  [rad, decays naturally to 0]

  Properties:
  • Direction (left/right) is purely RTK-derived → no target-position dependency
  • Magnitude depends on r_estimated, but a 2× range error only changes curve sharpness
  • Bias goes to 0 geometrically as drones converge → no separate decay function needed
  • w still multiplies the full bias, keeping the smooth S-curve shape

Subscriptions (all drones):
  ibvs/output                              — IBVSOutput
  drone{id}/fmu/out/vehicle_attitude       — quaternion FRD -> NED
  drone{id}/fmu/out/vehicle_local_position — NED velocity + position

Leader-only publication:
  /drone{leader_id}/leader_los             — Float32MultiArray [d_n, d_e, d_d] (NED LOS unit vec)

Follower-only subscriptions:
  /drone{leader_id}/fmu/out/vehicle_local_position — leader NED position (RTK)
  /drone{leader_id}/leader_los                     — leader LOS direction

Publication (all drones):
  png/guidance_cmd  (suicide_drone_msgs/GuidanceCmd)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Float32MultiArray
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
        self.declare_parameter('Kz', 3.0)
        self.declare_parameter('ka',    2.0)
        self.declare_parameter('v_max',  10.0)
        self.declare_parameter('v_init', 3.5)
        self.declare_parameter('rate', 50.0)
        self.declare_parameter('v_min_sigma', 0.5)
        # Fallback fixed bias — used only when leader LOS data is unavailable.
        # Set to 0 for all drones when FRDB is active.
        self.declare_parameter('az_bias_deg', 0.0)
        self.declare_parameter('el_bias_deg', 0.0)
        self.declare_parameter('r0', 10.0)
        self.declare_parameter('bias_decay_alpha', 1.0)
        # Formation-Referenced Dynamic Bias (FRDB)
        self.declare_parameter('leader_id', 1)       # which drone is the formation leader
        self.declare_parameter('bias_gain_K', 3.0)   # amplification of geometric angle

        system_id    = self.get_parameter('system_id').value
        self.Ky      = self.get_parameter('Ky').value
        self.Kz      = self.get_parameter('Kz').value
        self.ka      = self.get_parameter('ka').value
        self.v_max   = self.get_parameter('v_max').value
        self.v_init  = self.get_parameter('v_init').value
        self.rate        = self.get_parameter('rate').value
        self.v_min_sigma = self.get_parameter('v_min_sigma').value
        self.az_bias_rad      = math.radians(self.get_parameter('az_bias_deg').value)
        self.el_bias_rad      = math.radians(self.get_parameter('el_bias_deg').value)
        self.r0               = self.get_parameter('r0').value
        self.bias_decay_alpha = self.get_parameter('bias_decay_alpha').value
        leader_id             = self.get_parameter('leader_id').value
        self.bias_gain_K      = self.get_parameter('bias_gain_K').value

        self._is_leader   = (system_id == leader_id)
        self._is_follower = not self._is_leader
        self._leader_id   = leader_id

        # ── Runtime state ───────────────────────────────────────────────────
        self.R_e_b       = np.eye(3)
        self.v_ned       = np.zeros(3)
        self.my_pos_ned  = np.zeros(3)          # own NED position (RTK)
        self.r_estimated = self.r0
        self.q_y_prev    = 0.0
        self.q_z_prev    = 0.0
        self.q_y_now     = 0.0
        self.q_z_now     = 0.0
        self.los_rate_y  = 0.0
        self.los_rate_z  = 0.0
        self._los_prev_time = None
        self.v_now       = self.v_init
        self.los_received        = False
        self.prev_detected       = False
        self.first_guidance_step = True
        self.fov_yaw_rate        = 0.0
        self.fov_vel_z           = 0.0
        self._last_detected_time = None

        # FRDB follower state
        self.leader_pos_ned  = None   # NED position of leader (from RTK topic)
        self.leader_d1_ned   = None   # LOS unit vector broadcast by leader
        self._leader_los_time = None  # timestamp of last leader LOS message

        # ── Subscriptions ───────────────────────────────────────────────────
        self.create_subscription(
            IBVSOutput, 'ibvs/output', self.ibvs_output_callback, 10,
        )
        self.create_subscription(
            VehicleAttitude,
            f'/drone{system_id}/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleLocalPosition,
            f'/drone{system_id}/fmu/out/vehicle_local_position',
            self.local_position_callback,
            qos_profile_sensor_data,
        )

        # ── Leader: broadcast own LOS direction ─────────────────────────────
        if self._is_leader:
            self.leader_los_pub = self.create_publisher(
                Float32MultiArray,
                f'/drone{system_id}/leader_los',
                10,
            )

        # ── Follower: subscribe to leader position + LOS ────────────────────
        if self._is_follower:
            self.create_subscription(
                VehicleLocalPosition,
                f'/drone{leader_id}/fmu/out/vehicle_local_position',
                self.leader_position_callback,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Float32MultiArray,
                f'/drone{leader_id}/leader_los',
                self.leader_los_callback,
                10,
            )

        # ── Publisher ───────────────────────────────────────────────────────
        self.guidance_pub = self.create_publisher(GuidanceCmd, 'png/guidance_cmd', 25)

        self.create_timer(1.0 / self.rate, self.guidance_loop)

        role = 'leader' if self._is_leader else f'follower (leader=drone{leader_id})'
        self.get_logger().info(
            f'PNGGuidance started [{role}]: Ky={self.Ky}, Kz={self.Kz}, '
            f'ka={self.ka}, v_max={self.v_max}, v_init={self.v_init}, '
            f'bias_gain_K={self.bias_gain_K}, '
            f'az_bias_fallback={math.degrees(self.az_bias_rad):.1f}deg'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def attitude_callback(self, msg: VehicleAttitude):
        self.R_e_b = quat_to_R(msg.q)

    def local_position_callback(self, msg: VehicleLocalPosition):
        self.v_ned      = np.array([float(msg.vx), float(msg.vy), float(msg.vz)])
        self.my_pos_ned = np.array([float(msg.x),  float(msg.y),  float(msg.z)])

    def leader_position_callback(self, msg: VehicleLocalPosition):
        """Follower only: capture leader's NED position from its PX4 topic."""
        self.leader_pos_ned = np.array([float(msg.x), float(msg.y), float(msg.z)])

    def leader_los_callback(self, msg: Float32MultiArray):
        """Follower only: receive leader's LOS unit vector [d_n, d_e, d_d]."""
        if len(msg.data) >= 3:
            self.leader_d1_ned   = np.array([msg.data[0], msg.data[1], msg.data[2]])
            self._leader_los_time = self.get_clock().now()

    def ibvs_output_callback(self, msg: IBVSOutput):
        if msg.detected and not self.prev_detected:
            self.los_received        = False
            self.first_guidance_step = True
            self.los_rate_y          = 0.0
            self.los_rate_z          = 0.0
            now = self.get_clock().now()
            if self._last_detected_time is not None:
                gap = (now - self._last_detected_time).nanoseconds / 1e9
            else:
                gap = float('inf')
            if gap > 3.0:
                self.v_now       = self.v_init
                self.r_estimated = self.r0
                self.get_logger().info(
                    f'PNG: target re-acquired after {gap:.1f}s, speed/range reset'
                )
            else:
                self.get_logger().info(
                    f'PNG: target re-acquired after {gap:.1f}s, speed maintained={self.v_now:.2f} m/s'
                )
        self.prev_detected = msg.detected

        if not msg.detected:
            self.fov_yaw_rate = 0.0
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
        self.q_y_now      = msg.q_y
        self.q_z_now      = msg.q_z
        self.fov_yaw_rate = msg.fov_yaw_rate
        self.fov_vel_z    = msg.fov_vel_z
        self.los_received = True

        # Leader broadcasts its LOS direction so followers can compute FRDB bias.
        if self._is_leader:
            d1 = np.array([
                math.cos(self.q_y_now) * math.cos(self.q_z_now),
                math.cos(self.q_y_now) * math.sin(self.q_z_now),
                -math.sin(self.q_y_now),
            ])
            los_msg = Float32MultiArray()
            los_msg.data = [float(d1[0]), float(d1[1]), float(d1[2])]
            self.leader_los_pub.publish(los_msg)

    # ── FRDB: Formation-Referenced Dynamic Bias ───────────────────────────────

    def _compute_dynamic_az_bias(self) -> float:
        """
        Compute az_bias from formation geometry.

        Uses the signed lateral offset of this drone from the leader's LOS line
        to derive a geometrically-motivated approach angle bias:

            e_perp_right = [-d1_e, d1_n]   (rightward perp to d1 in NED horizontal)
            d_perp       = dot(my_pos - leader_pos, e_perp_right)  [m]
            az_bias      = atan2(d_perp, r_estimated) * K          [rad]

        Positive d_perp → drone is to the RIGHT of leader's LOS → positive bias.
        Negative d_perp → drone is to the LEFT                  → negative bias.

        The bias naturally converges to 0 as d_perp → 0 (drones approach target
        from different angles and converge on it), requiring no artificial decay.
        The w weight in the caller still shapes the S-curve profile.

        Falls back to fixed az_bias_rad if leader data is missing or stale (>2s).
        """
        if self.leader_pos_ned is None or self.leader_d1_ned is None:
            return self.az_bias_rad

        if self._leader_los_time is not None:
            age = (self.get_clock().now() - self._leader_los_time).nanoseconds / 1e9
            if age > 2.0:
                self.get_logger().warn(
                    f'Leader LOS stale ({age:.1f}s), falling back to fixed bias',
                    throttle_duration_sec=2.0,
                )
                return self.az_bias_rad

        p12 = (self.my_pos_ned - self.leader_pos_ned)[:2]   # [N, E]
        d1  = self.leader_d1_ned[:2]
        d1_norm = np.linalg.norm(d1)
        if d1_norm < 1e-6:
            return self.az_bias_rad
        d1 = d1 / d1_norm

        # Rightward perpendicular to d1 in NED horizontal plane:
        # cross([d1_n, d1_e, 0], [0, 0, -1]) = [-d1_e, d1_n, 0]
        e_perp_right = np.array([-d1[1], d1[0]])
        d_perp = float(np.dot(p12, e_perp_right))

        r = max(self.r_estimated, 0.5)
        return math.atan2(d_perp, r) * self.bias_gain_K

    # ── Guidance loop ─────────────────────────────────────────────────────────

    def guidance_loop(self):
        """
        Compute and publish NED velocity command using discrete PNG + FRDB.

        Eq.(8):  sigma_y, sigma_z from actual NED velocity
        Eq.(9):  sigma_yd = q_y_now + Ky * los_rate_y * dt  (+el_bias*w)
                 sigma_zd = q_z_now + Kz * los_rate_z * dt  (+az_bias_eff*w)
        Eq.(10): n_vd = [cos(s_yd)*cos(s_zd), cos(s_yd)*sin(s_zd), -sin(s_yd)]
        Eq.(14): v_now = min(v_now + ka/rate, v_max)

        az_bias_eff:
          Leader   → 0 (direct approach)
          Follower → FRDB dynamic bias from formation geometry
                     (fallback: fixed az_bias_rad when leader data unavailable)

        fov_yaw_rate scaling by (1-w):
          During biased approach the balloon is intentionally off-center.
          Applying full yaw correction would fight the PNG and cause the drone
          to rotate past the target. Scaling by (1-w) disables yaw correction
          when the bias is active and re-enables it smoothly at close range.
        """
        if not self.los_received:
            return

        # Eq.(8)
        speed = np.linalg.norm(self.v_ned)
        if speed >= self.v_min_sigma:
            n_v = self.v_ned / speed
        else:
            n_v = self.R_e_b[:, 0]
        sigma_z = math.atan2(n_v[1], n_v[0])
        sigma_y = math.atan2(-n_v[2], math.sqrt(n_v[0]**2 + n_v[1]**2))

        if self.first_guidance_step:
            self.first_guidance_step = False
            self.get_logger().info('PNG first step: los_rate=0, sigma_d = q_now')

        dt_guidance = 1.0 / self.rate

        self.r_estimated = max(self.r_estimated - self.v_now / self.rate, 0.0)
        w = min(self.r_estimated / self.r0, 1.0) ** self.bias_decay_alpha

        # Effective azimuth bias
        if self._is_follower:
            az_bias_eff = self._compute_dynamic_az_bias()
        else:
            az_bias_eff = self.az_bias_rad   # 0 for leader

        sigma_yd = self.q_y_now + self.Ky * self.los_rate_y * dt_guidance + self.el_bias_rad * w
        sigma_zd = self.q_z_now + self.Kz * self.los_rate_z * dt_guidance + az_bias_eff * w

        # Eq.(14)
        self.v_now = min(self.v_now + self.ka / self.rate, self.v_max)

        # Eq.(10)
        n_vd = np.array([
            math.cos(sigma_yd) * math.cos(sigma_zd),
            math.cos(sigma_yd) * math.sin(sigma_zd),
            -math.sin(sigma_yd),
        ])
        v_cmd = self.v_now * n_vd

        cmd              = GuidanceCmd()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.target_detected = self.prev_detected
        cmd.vel_n        = float(v_cmd[0])
        cmd.vel_e        = float(v_cmd[1])
        cmd.vel_d        = float(v_cmd[2])
        cmd.yaw_rate     = self.fov_yaw_rate * (1.0 - w)
        self.guidance_pub.publish(cmd)

        self.get_logger().info(
            f'PNG: s_z={math.degrees(sigma_zd):.1f}deg '
            f'bias={math.degrees(az_bias_eff * w):.1f}deg(w={w:.2f} r={self.r_estimated:.1f}m) '
            f'v={self.v_now:.2f}m/s NED=({cmd.vel_n:.2f},{cmd.vel_e:.2f},{cmd.vel_d:.2f})',
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
