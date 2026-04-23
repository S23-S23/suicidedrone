#!/usr/bin/env python3
"""
IBVS Controller Node
Image-Based Visual Servoing for balloon interception.

Reference: "Precise Interception Flight Targets by Image-Based Visual Servoing
           of Multicopter", IEEE TIE 2025

Equations implemented:
  Eq.(3):  Normalized image error (ex, ey)
  Eq.(5):  LOS unit vector in NED frame via R_e_b @ R_b_c @ ray
  Eq.(7):  LOS angles (elevation q_y, azimuth q_z) from LOS unit vector
  Eq.(13): FOV yaw rate controller using IMU angular velocity (avoids numerical diff)

Subscriptions:
  /filter_estimate                             — Float32MultiArray [u, v, u_dot, v_dot, delay]
  drone{id}/fmu/out/vehicle_attitude           — quaternion FRD body -> NED
  drone{id}/fmu/out/vehicle_angular_velocity   — FRD body angular velocity

Publications:
  /ibvs/output  (suicide_drone_msgs/IBVSOutput)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleAttitude, VehicleAngularVelocity
from suicide_drone_msgs.msg import IBVSOutput


def quat_to_R(q):
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix R (NED <- body-FRD)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)


class IBVSController(Node):
    def __init__(self):
        super().__init__('ibvs_controller')

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter('system_id', 1)
        # Camera intrinsics (iris_depth_camera: 848x480, hfov=1.5009831567)
        self.declare_parameter('fx', 454.8)
        self.declare_parameter('fy', 454.8)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)
        # FOV yaw controller gains (Eq.13)
        self.declare_parameter('fov_kp',   1.5)
        self.declare_parameter('fov_kd',   0.1)
        # FOV vertical (ey) velocity controller gains — analogous to Eq.13 for Z axis
        self.declare_parameter('fov_kp_z', 1.5)
        self.declare_parameter('fov_kd_z', 0.1)
        # Seconds without detection before target_detected -> False
        self.declare_parameter('target_timeout', 0.5)

        system_id          = self.get_parameter('system_id').value
        self.fx            = self.get_parameter('fx').value
        self.fy            = self.get_parameter('fy').value
        self.cx            = self.get_parameter('cx').value
        self.cy            = self.get_parameter('cy').value
        self.fov_kp        = self.get_parameter('fov_kp').value
        self.fov_kd        = self.get_parameter('fov_kd').value
        self.fov_kp_z      = self.get_parameter('fov_kp_z').value
        self.fov_kd_z      = self.get_parameter('fov_kd_z').value
        self.target_timeout = self.get_parameter('target_timeout').value

        # ── Camera (OpenCV) -> Body (FRD) rotation matrix ───────────────────
        # OpenCV: X=right, Y=down, Z=forward
        # FRD:    X=forward, Y=right, Z=down
        # body_x = cam_z, body_y = cam_x, body_z = cam_y
        self.R_b_c = np.array([
            [0, 0, 1],   # body_x <- cam_z  (forward)
            [1, 0, 0],   # body_y <- cam_x  (right)
            [0, 1, 0],   # body_z <- cam_y  (down)
        ], dtype=float)

        # ── Runtime state ───────────────────────────────────────────────────
        self.R_e_b             = np.eye(3)   # FRD -> NED rotation (Eq.5)
        self.b_omega_y         = 0.0         # body pitch rate [rad/s] (ey controller)
        self.b_omega_z         = 0.0         # body yaw rate [rad/s] (Eq.13)
        self.last_detect_time  = None

        # ── Subscriptions ───────────────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray,
            '/filter_estimate',
            self.filter_callback,
            10,
        )
        self.create_subscription(
            VehicleAttitude,
            f'drone{system_id}/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleAngularVelocity,
            f'drone{system_id}/fmu/out/vehicle_angular_velocity',
            self.angular_velocity_callback,
            qos_profile_sensor_data,
        )

        # ── Publishers ──────────────────────────────────────────────────────
        self.pub_ibvs = self.create_publisher(IBVSOutput, '/ibvs/output', 10)

        # Timeout checker at 10 Hz
        self.create_timer(0.1, self._timeout_check)

        self.get_logger().info('IBVSController started: /filter_estimate -> /ibvs/output')

    # ── IMU callbacks ────────────────────────────────────────────────────────

    def attitude_callback(self, msg: VehicleAttitude):
        self.R_e_b = quat_to_R(msg.q)

    def angular_velocity_callback(self, msg: VehicleAngularVelocity):
        self.b_omega_y = float(msg.xyz[1])
        self.b_omega_z = float(msg.xyz[2])

    # ── Filter estimate callback ────────────────────────────────────────────

    def filter_callback(self, msg: Float32MultiArray):
        """
        Process DKF/EKF filter estimate and compute IBVS outputs.

        Input: Float32MultiArray [u_px, v_px, u_dot_px, v_dot_px, delay_steps]
          u_px, v_px:         filtered pixel coordinates
          u_dot_px, v_dot_px: filtered pixel velocities [px/s]

        Eq.(3):  ex = (u - cx) / fx,  ey = (v - cy) / fy
        Eq.(5):  n_t = R_e_b @ R_b_c @ [ex, ey, 1]^T  (normalized)
        Eq.(7):  q_y = atan2(-n_t_z, ||n_t_xy||),  q_z = atan2(n_t_y, n_t_x)
        Eq.(13): w_yaw = kp*ex + kd*ex_dot   (ex_dot from filter)
        """
        if len(msg.data) < 4:
            return

        u     = msg.data[0]   # filtered pixel x
        v     = msg.data[1]   # filtered pixel y
        u_dot = msg.data[2]   # filtered pixel velocity x [px/s]
        v_dot = msg.data[3]   # filtered pixel velocity y [px/s]

        # Eq.(3): normalized image-plane error
        ex = (u - self.cx) / self.fx
        ey = (v - self.cy) / self.fy

        # Eq.(5): LOS unit vector in NED frame
        ray_cam  = np.array([ex, ey, 1.0])
        ray_body = self.R_b_c @ ray_cam
        ray_ned  = self.R_e_b @ ray_body
        norm = np.linalg.norm(ray_ned)
        if norm < 1e-6:
            return
        n_t = ray_ned / norm

        # Eq.(7): LOS angles in NED spherical coordinates
        q_z = math.atan2(n_t[1], n_t[0])
        q_y = math.atan2(-n_t[2], math.sqrt(n_t[0]**2 + n_t[1]**2))

        # Eq.(13): FOV holding yaw rate controller
        #   ex_dot from filter estimate (pixel velocity / focal length)
        ex_dot       = u_dot / self.fx
        fov_yaw_rate = self.fov_kp * ex + self.fov_kd * ex_dot

        # Vertical (ey) controller — analogous to Eq.13 for Z axis
        ey_dot    = v_dot / self.fy
        fov_vel_z = self.fov_kp_z * ey + self.fov_kd_z * ey_dot

        # ── Publish ──────────────────────────────────────────────────────────
        self.last_detect_time = self.get_clock().now()

        ibvs_msg                = IBVSOutput()
        ibvs_msg.header.stamp   = self.get_clock().now().to_msg()
        ibvs_msg.detected       = True
        ibvs_msg.q_y            = q_y
        ibvs_msg.q_z            = q_z
        ibvs_msg.fov_yaw_rate   = float(fov_yaw_rate)
        ibvs_msg.fov_vel_z      = float(fov_vel_z)
        self.pub_ibvs.publish(ibvs_msg)

        self.get_logger().info(
            f'IBVS: u={u:.0f}px, ex={ex:.3f}, ey={ey:.3f}, '
            f'q_y={math.degrees(q_y):.1f}deg, q_z={math.degrees(q_z):.1f}deg, '
            f'yaw_rate={fov_yaw_rate:.3f} rad/s',
            throttle_duration_sec=1.0,
        )

    # ── Timeout ──────────────────────────────────────────────────────────────

    def _timeout_check(self):
        if self.last_detect_time is None:
            return
        elapsed = (self.get_clock().now() - self.last_detect_time).nanoseconds / 1e9
        if elapsed > self.target_timeout:
            ibvs_msg          = IBVSOutput()
            ibvs_msg.header.stamp = self.get_clock().now().to_msg()
            ibvs_msg.detected = False
            ibvs_msg.q_y      = 0.0
            ibvs_msg.q_z      = 0.0
            self.pub_ibvs.publish(ibvs_msg)


def main(args=None):
    rclpy.init(args=args)
    node = IBVSController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
