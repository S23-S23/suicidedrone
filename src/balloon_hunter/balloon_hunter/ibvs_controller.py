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
  /target_info                        – gt_balloon_detector output
  drone{id}/fmu/out/vehicle_attitude  – quaternion FRD body → NED
  drone{id}/fmu/out/vehicle_angular_velocity – FRD body angular velocity

Publications:
  /ibvs/target_detected  (std_msgs/Bool)
  /ibvs/los_angles       (geometry_msgs/Vector3: x=q_y elevation, y=q_z azimuth)
  /ibvs/image_error      (geometry_msgs/Vector3: x=ex, y=ey)
  /ibvs/fov_yaw_rate     (std_msgs/Float64) – commanded yaw rate for drone
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import VehicleAttitude, VehicleAngularVelocity
from suicide_drone_msgs.msg import TargetInfo
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool, Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


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


class IBVSController(Node):
    def __init__(self):
        super().__init__('ibvs_controller')

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter('system_id', 1)
        # Camera intrinsics (typhoon_h480 @ 640×360, FOV=2.0 rad)
        self.declare_parameter('fx', 205.5)
        self.declare_parameter('fy', 205.5)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 180.0)
        # FOV yaw controller gains (Eq.13)
        self.declare_parameter('fov_kp',   1.5)
        self.declare_parameter('fov_kd',   0.1)
        # FOV vertical (ey) velocity controller gains — analogous to Eq.13 for Z axis
        self.declare_parameter('fov_kp_z', 1.5)
        self.declare_parameter('fov_kd_z', 0.1)
        # Seconds without detection before target_detected → False
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

        # ── Camera (OpenCV) → Body (FRD) rotation matrix ───────────────────
        # OpenCV: X=right, Y=down,    Z=forward
        # FRD:    X=forward, Y=right, Z=down
        # Mapping: body_x = cam_z, body_y = cam_x, body_z = cam_y
        # Used in Eq.(5): ray_body = R_b_c @ ray_cam
        self.R_b_c = np.array([
            [0, 0, 1],   # body_x ← cam_z  (forward)
            [1, 0, 0],   # body_y ← cam_x  (right)
            [0, 1, 0],   # body_z ← cam_y  (down)
        ], dtype=float)

        # ── Runtime state ───────────────────────────────────────────────────
        self.R_e_b             = np.eye(3)   # FRD → NED rotation (Eq.5)
        self.b_omega_y         = 0.0         # body pitch rate [rad/s] (ey controller)
        self.b_omega_z         = 0.0         # body yaw rate [rad/s] (Eq.13)
        self._last_q           = np.array([1.0, 0.0, 0.0, 0.0])  # attitude quaternion [w,x,y,z]
        self.last_detect_time  = None        # for timeout check
        self._bridge           = CvBridge()
        self._latest_img       = None        # cache of the latest inference_result image

        # ── Subscriptions ───────────────────────────────────────────────────
        self.create_subscription(
            TargetInfo,
            '/target_info',
            self.detection_callback,
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

        # Subscribe to inference_result image
        self.create_subscription(
            Image,
            f'/inference_result_{system_id}',
            self._image_callback,
            10,
        )

        # ── Publishers ──────────────────────────────────────────────────────
        self.pub_detected   = self.create_publisher(Bool,    '/ibvs/target_detected',  10)
        self.pub_los        = self.create_publisher(Vector3, '/ibvs/los_angles',        10)
        self.pub_yaw_rate   = self.create_publisher(Float64, '/ibvs/fov_yaw_rate',      10)
        self.pub_fov_vel_z  = self.create_publisher(Float64, '/ibvs/fov_vel_z',         10)

        # Publish IBVS debug image
        self.pub_debug_img  = self.create_publisher(Image,   '/ibvs/debug_image',       10)

        # Timeout checker at 10 Hz
        self.create_timer(0.1, self._timeout_check)

        self.get_logger().info('IBVSController started: detection topic=/target_info')

    # ── Image callback ───────────────────────────────────────────────────────

    def _image_callback(self, msg: Image):
        """Cache the latest inference_result image."""
        self._latest_img = msg

    # ── IMU callbacks ────────────────────────────────────────────────────────

    def attitude_callback(self, msg: VehicleAttitude):
        """Store R_e_b (FRD body → NED) from VehicleAttitude quaternion."""
        self.R_e_b   = quat_to_R(msg.q)
        self._last_q = np.array([float(msg.q[0]), float(msg.q[1]),
                                  float(msg.q[2]), float(msg.q[3])])

    def angular_velocity_callback(self, msg: VehicleAngularVelocity):
        """Store body angular rates from IMU (FRD frame).
        xyz[1] = pitch rate (positive = nose down), xyz[2] = yaw rate.
        """
        self.b_omega_y = float(msg.xyz[1])
        self.b_omega_z = float(msg.xyz[2])

    # ── Detection callback ───────────────────────────────────────────────────

    def detection_callback(self, msg: TargetInfo):
        """
        Process the first detection and compute IBVS outputs.

        Eq.(3):  ex = (u - cx) / fx,  ey = (v - cy) / fy
        Eq.(5):  n_t = R_e_b @ R_b_c @ [ex, ey, 1]^T  (normalized)
        Eq.(7):  q_y = atan2(-n_t_z, ||n_t_xy||),  q_z = atan2(n_t_y, n_t_x)
        Eq.(13): ω_yaw = kp*ex + kd*(−(1+ex²)*b_ω_z)
        """
        det = msg

        # Bounding box center in pixels
        u = (det.left + det.right)  * 0.5
        v = (det.top  + det.bottom) * 0.5

        # Eq.(3): normalized image-plane error (image center = principal point)
        ex = (u - self.cx) / self.fx
        ey = (v - self.cy) / self.fy

        # Eq.(5): LOS unit vector in NED frame
        #   ray in camera frame (OpenCV)
        ray_cam  = np.array([ex, ey, 1.0])
        #   transform to FRD body frame
        ray_body = self.R_b_c @ ray_cam
        #   transform to NED world frame using current attitude
        ray_ned  = self.R_e_b @ ray_body
        norm = np.linalg.norm(ray_ned)
        if norm < 1e-6:
            return
        n_t = ray_ned / norm

        # Eq.(7): LOS angles in NED spherical coordinates
        #   azimuth  q_z: angle from North in horizontal plane (clockwise positive)
        q_z = math.atan2(n_t[1], n_t[0])
        #   elevation q_y: angle above horizontal (NED z=Down, so -z = Up)
        q_y = math.atan2(-n_t[2], math.sqrt(n_t[0]**2 + n_t[1]**2))

        # Eq.(13): FOV holding yaw rate controller
        #   From interaction matrix (Eq.4) for pure yaw motion:
        #     ė_x ≈ −(1 + ex²) · b_ω_z          (Eq.26 in paper)
        #   IMU measurement replaces noisy numerical differentiation of ex
        ex_dot       = -(1.0 + ex**2) * self.b_omega_z
        fov_yaw_rate = self.fov_kp * ex + self.fov_kd * ex_dot

        # Vertical (ey) image-plane controller — analogous to Eq.13 for Z axis
        #   ė_y ≈ −(1 + ey²) · b_ω_y  (pitch rate, FRD y-axis, cam x-axis rotation)
        #   Output: NED Z velocity correction [m/s]
        #     ey > 0 → balloon below image center → drone descends (+NED z) → balloon rises ✓
        ey_dot    = -(1.0 + ey**2) * self.b_omega_y
        fov_vel_z = self.fov_kp_z * ey + self.fov_kd_z * ey_dot

        # ── Publish ──────────────────────────────────────────────────────────
        self.last_detect_time = self.get_clock().now()

        det_msg      = Bool()
        det_msg.data = True
        self.pub_detected.publish(det_msg)

        los_msg   = Vector3()
        los_msg.x = q_y   # elevation  (used by PNG guidance Eq.9)
        los_msg.y = q_z   # azimuth
        los_msg.z = 0.0
        self.pub_los.publish(los_msg)

        yaw_msg      = Float64()
        yaw_msg.data = float(fov_yaw_rate)
        self.pub_yaw_rate.publish(yaw_msg)

        vel_z_msg      = Float64()
        vel_z_msg.data = float(fov_vel_z)
        self.pub_fov_vel_z.publish(vel_z_msg)

        self.get_logger().info(
            f'IBVS: u={u:.0f}px, ex={ex:.3f}, ey={ey:.3f}, '
            f'q_y={math.degrees(q_y):.1f}°, q_z={math.degrees(q_z):.1f}°, '
            f'yaw_rate={fov_yaw_rate:.3f} rad/s',
            throttle_duration_sec=1.0,
        )

        # ── Debug image overlay ───────────────────────────────────────────
        if self._latest_img is not None:
            try:
                cv_img = self._bridge.imgmsg_to_cv2(self._latest_img, 'bgr8')
            except Exception:
                cv_img = None
            if cv_img is not None:
                h, w   = cv_img.shape[:2]
                cx_img = int(self.cx)
                cy_img = int(self.cy)
                u_int  = int(round(u))
                v_int  = int(round(v))

                # ── 1. Balloon bounding box ─────────────────────────
                bb_l = int(det.left);  bb_t = int(det.top)
                bb_r = int(det.right); bb_b = int(det.bottom)
                cv2.rectangle(cv_img, (bb_l, bb_t), (bb_r, bb_b), (0, 255, 0), 2)

                # ── 2. Principal-point crosshair (image center) ─────
                cv2.line(cv_img, (cx_img - 20, cy_img),
                                  (cx_img + 20, cy_img), (0, 255, 255), 2)
                cv2.line(cv_img, (cx_img, cy_img - 20),
                                  (cx_img, cy_img + 20), (0, 255, 255), 2)
                cv2.circle(cv_img, (cx_img, cy_img), 4, (0, 255, 255), 1)

                # ── 3. Balloon center dot ───────────────────────────
                cv2.circle(cv_img, (u_int, v_int), 7, (0, 60, 255), 2)
                cv2.circle(cv_img, (u_int, v_int), 2, (0, 60, 255), -1)

                # ── 4. Error vector arrow (center → balloon) ────────
                # Shows LOS image error direction and magnitude
                cv2.arrowedLine(cv_img, (cx_img, cy_img), (u_int, v_int),
                                (0, 140, 255), 2, tipLength=0.2)

                # ── 5. Horizontal error bar  ex  (bottom-center) ────
                BAR_W  = w // 3
                BAR_H  = 12
                bx     = (w - BAR_W) // 2
                by     = h - 28
                cv2.rectangle(cv_img, (bx, by),
                              (bx + BAR_W, by + BAR_H), (40, 40, 40), -1)
                mid_bx = bx + BAR_W // 2
                cv2.line(cv_img, (mid_bx, by - 2), (mid_bx, by + BAR_H + 2),
                         (200, 200, 200), 1)
                fill_ex = int(ex * self.fx)
                fill_ex = max(-BAR_W // 2, min(BAR_W // 2, fill_ex))
                col_ex  = (30, 200, 30) if abs(ex) < 0.1 else (0, 100, 255)
                cv2.rectangle(cv_img, (mid_bx, by),
                              (mid_bx + fill_ex, by + BAR_H), col_ex, -1)
                cv2.rectangle(cv_img, (bx, by),
                              (bx + BAR_W, by + BAR_H), (180, 180, 180), 1)
                cv2.putText(cv_img, f'ex={ex:+.3f}', (bx, by - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

                # ── 6. Vertical error bar  ey  (right edge) ─────────
                VB_W  = 12
                VB_H  = h // 3
                vbx   = w - 24
                vby   = (h - VB_H) // 2
                cv2.rectangle(cv_img, (vbx, vby),
                              (vbx + VB_W, vby + VB_H), (40, 40, 40), -1)
                mid_vy = vby + VB_H // 2
                cv2.line(cv_img, (vbx - 2, mid_vy), (vbx + VB_W + 2, mid_vy),
                         (200, 200, 200), 1)
                fill_ey = int(ey * self.fy)
                fill_ey = max(-VB_H // 2, min(VB_H // 2, fill_ey))
                col_ey  = (30, 200, 30) if abs(ey) < 0.1 else (0, 100, 255)
                cv2.rectangle(cv_img, (vbx, mid_vy),
                              (vbx + VB_W, mid_vy + fill_ey), col_ey, -1)
                cv2.rectangle(cv_img, (vbx, vby),
                              (vbx + VB_W, vby + VB_H), (180, 180, 180), 1)
                cv2.putText(cv_img, f'ey', (vbx - 2, vby - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                cv2.putText(cv_img, f'{ey:+.2f}', (vbx - 10, vby + VB_H + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                # ── 7. Azimuth compass  q_z  (top-right circle) ─────
                # Arrow points in the horizontal LOS direction (NED projected)
                COMP_R  = 42
                comp_cx = w - COMP_R - 12
                comp_cy = COMP_R + 12
                cv2.circle(cv_img, (comp_cx, comp_cy), COMP_R, (40, 40, 40), -1)
                cv2.circle(cv_img, (comp_cx, comp_cy), COMP_R, (160, 160, 160), 1)
                # North reference tick
                cv2.line(cv_img,
                         (comp_cx, comp_cy - COMP_R + 3),
                         (comp_cx, comp_cy - COMP_R + 9), (200, 200, 200), 1)
                az_ex = int(comp_cx + COMP_R * 0.75 * math.sin(q_z))
                az_ey = int(comp_cy - COMP_R * 0.75 * math.cos(q_z))
                cv2.arrowedLine(cv_img, (comp_cx, comp_cy),
                                (az_ex, az_ey), (0, 255, 255), 2, tipLength=0.3)
                cv2.putText(cv_img, f'q_z',
                            (comp_cx - 10, comp_cy + COMP_R + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                cv2.putText(cv_img, f'{math.degrees(q_z):+.1f}',
                            (comp_cx - 14, comp_cy + COMP_R + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

                # ── 8. Elevation gauge  q_y  (below azimuth compass) ─
                # Bar left of center = looking down, right = looking up
                G_W   = COMP_R * 2
                G_H   = 10
                gx    = w - G_W - 12
                gy    = comp_cy * 2 + 18
                cv2.rectangle(cv_img, (gx, gy),
                              (gx + G_W, gy + G_H), (40, 40, 40), -1)
                mid_gx = gx + G_W // 2
                cv2.line(cv_img, (mid_gx, gy - 2), (mid_gx, gy + G_H + 2),
                         (200, 200, 200), 1)
                MAX_EL   = math.radians(45.0)
                fill_el  = int((q_y / MAX_EL) * (G_W // 2))
                fill_el  = max(-G_W // 2, min(G_W // 2, fill_el))
                col_el   = (50, 220, 50) if q_y > 0 else (0, 130, 255)
                cv2.rectangle(cv_img, (mid_gx, gy),
                              (mid_gx + fill_el, gy + G_H), col_el, -1)
                cv2.rectangle(cv_img, (gx, gy),
                              (gx + G_W, gy + G_H), (160, 160, 160), 1)
                cv2.putText(cv_img, f'q_y {math.degrees(q_y):+.1f}',
                            (gx, gy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

                # ── 9. Yaw-rate indicator  (top-left circle) ────────
                # Arrow angle proportional to commanded yaw rate
                YAW_R  = 32
                yc_cx  = YAW_R + 12
                yc_cy  = YAW_R + 12
                cv2.circle(cv_img, (yc_cx, yc_cy), YAW_R, (40, 40, 40), -1)
                cv2.circle(cv_img, (yc_cx, yc_cy), YAW_R, (160, 160, 160), 1)
                MAX_YR = 3.0
                yr_ang = (fov_yaw_rate / MAX_YR) * math.pi
                yr_ang = max(-math.pi, min(math.pi, yr_ang))
                yr_ex  = int(yc_cx + YAW_R * 0.75 * math.sin(yr_ang))
                yr_ey  = int(yc_cy - YAW_R * 0.75 * math.cos(yr_ang))
                yr_col = (220, 0, 200) if abs(fov_yaw_rate) > 0.1 else (100, 100, 100)
                cv2.arrowedLine(cv_img, (yc_cx, yc_cy),
                                (yr_ex, yr_ey), yr_col, 2, tipLength=0.35)
                cv2.putText(cv_img, 'YAW',
                            (yc_cx - 13, yc_cy + YAW_R + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 0, 200), 1)
                cv2.putText(cv_img, f'{fov_yaw_rate:+.2f}',
                            (yc_cx - 16, yc_cy + YAW_R + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

                # ── 10. Compact text summary (bottom-left) ──────────
                lines = [
                    f'q_y={math.degrees(q_y):+.1f}  q_z={math.degrees(q_z):+.1f} deg',
                    f'yr={fov_yaw_rate:+.3f} rad/s',
                ]
                ty = h - 48
                for line in lines:
                    cv2.putText(cv_img, line, (8, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3)
                    cv2.putText(cv_img, line, (8, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                    ty += 18

                try:
                    debug_msg = self._bridge.cv2_to_imgmsg(cv_img, 'bgr8')
                    debug_msg.header = self._latest_img.header
                    self.pub_debug_img.publish(debug_msg)
                except Exception:
                    pass

    # ── Timeout ──────────────────────────────────────────────────────────────

    def _timeout_check(self):
        """Publish target_detected=False when no detection for target_timeout seconds."""
        if self.last_detect_time is None:
            return
        elapsed = (self.get_clock().now() - self.last_detect_time).nanoseconds / 1e9
        if elapsed > self.target_timeout:
            msg      = Bool()
            msg.data = False
            self.pub_detected.publish(msg)


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
