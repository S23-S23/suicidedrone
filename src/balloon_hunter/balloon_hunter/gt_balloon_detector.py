#!/usr/bin/env python3
"""
Ground Truth Balloon Detector Node
Drop-in replacement for balloon_detector (YOLO).

Reads the balloon's true 3D position from /gazebo/model_states and the
camera's true pose (position + orientation) from /gazebo/link_states,
then projects the balloon onto the image plane using a pinhole camera model.

Key design decision (Method B):
  The projection ray originates at the camera optical center and uses the
  camera link's world orientation directly.  No NED conversion or drone-body
  attitude reconstruction is needed - the gimbal state, drone pitch/roll/yaw
  are all automatically accounted for through the camera link pose.

Projection pipeline:
  1. rel_ENU  = balloon_ENU - camera_ENU           (vector in Gazebo world ENU)
  2. r_cam    = R_ENU_to_cam @ rel_ENU             (rotate into camera frame)
  3. u = fx * r_cam[0]/r_cam[2] + cx              (pinhole projection)
     v = fy * r_cam[1]/r_cam[2] + cy

Camera frame (OpenCV / ROS convention):
  X = right,  Y = down,  Z = forward

The depth_camera_link in Gazebo link_states uses the Gazebo SDF FLU convention:
  link-X = forward,  link-Y = left,  link-Z = up

A frame-alignment matrix R_FLU_to_opencv is required to convert from the
Gazebo SDF link frame to the OpenCV camera frame:
  opencv-X (right)   = -link-Y  →  row: [0, -1,  0]
  opencv-Y (down)    = -link-Z  →  row: [0,  0, -1]
  opencv-Z (forward) =  link-X  →  row: [1,  0,  0]

Input  (identical to balloon_detector):
  /drone{id}/camera/image_raw   - timing sync + visualization base image
  /gazebo/model_states          - balloon 3D ground-truth position
  /gazebo/link_states           - camera link 6-DOF pose (position + orientation)

Output (identical to balloon_detector):
  /Yolov8_Inference_{id}        - yolov8_msgs/Yolov8Inference (bounding box)
  /inference_result_{id}        - sensor_msgs/Image (annotated visualization)
"""

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates, LinkStates
from geometry_msgs.msg import Point
from yolov8_msgs.msg import InferenceResult, Yolov8Inference
from cv_bridge import CvBridge

bridge = CvBridge()

# Visualization colors (BGR)
COLOR_DETECTED   = (0, 255,   0)   # green  – balloon in FOV and detected
COLOR_OUT_OF_FOV = (0, 165, 255)   # orange – balloon projected outside image
COLOR_BEHIND     = (0,   0, 255)   # red    – balloon behind camera
COLOR_WAITING    = (180, 180, 180) # grey   – waiting for Gazebo data


# --------------------------------------------------------------------------- #
#  Math helpers                                                                 #
# --------------------------------------------------------------------------- #
def quat_to_R_world_to_frame(q) -> np.ndarray:
    """
    Build a 3x3 rotation matrix that transforms vectors FROM world ENU
    INTO the local frame represented by quaternion q (q = frame in world).

    q describes how the frame is oriented in the world:
      R_frame_to_world (columns = frame axes in world coords)
    We want the inverse: R_world_to_frame = R_frame_to_world.T
    """
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    R_frame_to_world = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
    ], dtype=float)
    return R_frame_to_world.T   # world -> frame


# --------------------------------------------------------------------------- #
#  Node                                                                         #
# --------------------------------------------------------------------------- #
class GtBalloonDetector(Node):
    def __init__(self):
        super().__init__('balloon_detector')  # keep same node name as YOLO version

        # --- Parameters (same structure as balloon_detector) ----------------
        self.declare_parameter('system_id', 1)
        self.declare_parameter('camera_topic', '/drone1/camera/image_raw')
        # Camera intrinsics – iris_depth_camera: 848x480, hfov=1.5009831567
        self.declare_parameter('width',  848)
        self.declare_parameter('height', 480)
        self.declare_parameter('fx', 454.8)
        self.declare_parameter('fy', 454.8)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)
        # GT-specific parameters
        self.declare_parameter('balloon_model_name',  'target_balloon')
        self.declare_parameter('balloon_radius',      0.3)
        # balloon_link <pose>0 0 1.5 0 0 0</pose> in model.sdf:
        # model_states returns the model root pose; add this for the sphere center.
        self.declare_parameter('balloon_link_z_offset', 1.5)
        # Camera link name in /gazebo/link_states: "<model>::<link>"
        self.declare_parameter('camera_link_name', 'drone1::depth_camera_link')
        # Sensor offset from link origin in link-local frame [x, y, z]
        # iris_depth_camera: sensor is at link origin (no offset)
        self.declare_parameter('sensor_offset_x', 0.0)
        self.declare_parameter('sensor_offset_y', 0.0)
        self.declare_parameter('sensor_offset_z', 0.0)

        sid                   = self.get_parameter('system_id').value
        camera_topic          = self.get_parameter('camera_topic').value
        self.width            = self.get_parameter('width').value
        self.height           = self.get_parameter('height').value
        self.fx               = self.get_parameter('fx').value
        self.fy               = self.get_parameter('fy').value
        self.cx               = self.get_parameter('cx').value
        self.cy               = self.get_parameter('cy').value
        self.balloon_model    = self.get_parameter('balloon_model_name').value
        self.balloon_r        = self.get_parameter('balloon_radius').value
        self.balloon_z_offset = self.get_parameter('balloon_link_z_offset').value
        self.camera_link_name = self.get_parameter('camera_link_name').value
        self.sensor_offset    = np.array([
            self.get_parameter('sensor_offset_x').value,
            self.get_parameter('sensor_offset_y').value,
            self.get_parameter('sensor_offset_z').value,
        ])

        # --- State (updated by Gazebo callbacks) ----------------------------
        # All positions stored in Gazebo ENU world frame
        self.balloon_enu:  np.ndarray | None = None  # balloon sphere center
        self.camera_enu:   np.ndarray | None = None  # camera optical center
        self.R_enu_to_cam: np.ndarray | None = None  # rotation: ENU -> camera

        # --- QoS ------------------------------------------------------------
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscriptions --------------------------------------------------
        self.create_subscription(Image,       camera_topic,          self.camera_callback,       cam_qos)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_states_callback, 10)
        self.create_subscription(LinkStates,  '/gazebo/link_states',  self.link_states_callback,  10)

        # --- Publishers (identical topics/types to balloon_detector) --------
        self.yolov8_pub = self.create_publisher(Yolov8Inference, f'/Yolov8_Inference_{sid}', 10)
        self.img_pub    = self.create_publisher(Image,           f'/inference_result_{sid}',  10)
        self.pos_pub    = self.create_publisher(Point,           '/target_world_pos',         10)

        # --- 20Hz timer: publish detection independent of slow camera topic ---
        self.create_timer(0.05, self._timer_detect)  # 20Hz

        self.get_logger().info(
            f'GtBalloonDetector started | camera={camera_topic} '
            f'balloon_model="{self.balloon_model}" '
            f'camera_link="{self.camera_link_name}"'
        )

    # ----------------------------------------------------------------------- #
    #  Gazebo callbacks                                                         #
    # ----------------------------------------------------------------------- #
    def model_states_callback(self, msg: ModelStates):
        """Extract balloon sphere center position in Gazebo ENU."""
        if self.balloon_model not in msg.name:
            self.get_logger().warn(
                f'Balloon model "{self.balloon_model}" not found. '
                f'Available: {list(msg.name)}',
                throttle_duration_sec=5.0,
            )
            return
        idx = msg.name.index(self.balloon_model)
        p   = msg.pose[idx].position
        # Apply balloon_link Z offset from model.sdf (<pose>0 0 1.5 0 0 0</pose>)
        self.balloon_enu = np.array([float(p.x), float(p.y), float(p.z) + self.balloon_z_offset])
        # Publish target 3D world position (Gazebo ENU) for logger/drone_manager
        pt = Point()
        pt.x, pt.y, pt.z = self.balloon_enu[0], self.balloon_enu[1], self.balloon_enu[2]
        self.pos_pub.publish(pt)

    def link_states_callback(self, msg: LinkStates):
        """
        Extract camera optical-center position AND orientation in Gazebo ENU.
        The full 6-DOF pose from link_states automatically reflects drone
        pitch, roll, yaw and any gimbal state - no separate attitude source needed.
        """
        if self.camera_link_name not in msg.name:
            self.get_logger().warn(
                f'Camera link "{self.camera_link_name}" not found. '
                f'Available: {list(msg.name)}',
                throttle_duration_sec=5.0,
            )
            return
        idx  = msg.name.index(self.camera_link_name)
        pose = msg.pose[idx]

        # Orientation: build ENU -> camera rotation matrix from quaternion
        self.R_enu_to_cam = quat_to_R_world_to_frame(pose.orientation)

        # Position: link origin in ENU + sensor offset (in link-local frame)
        p = pose.position
        link_origin_enu = np.array([float(p.x), float(p.y), float(p.z)])
        R_link_to_world = self.R_enu_to_cam.T
        self.camera_enu = link_origin_enu + R_link_to_world @ self.sensor_offset

    # ----------------------------------------------------------------------- #
    #  20Hz timer: publish Yolov8Inference without waiting for camera image     #
    # ----------------------------------------------------------------------- #
    def _timer_detect(self):
        """Project balloon and publish detection at 20Hz (camera-independent)."""
        if self.balloon_enu is None or self.camera_enu is None or self.R_enu_to_cam is None:
            return

        rel_enu = self.balloon_enu - self.camera_enu
        r_flu = self.R_enu_to_cam @ rel_enu
        R_FLU_to_opencv = np.array([[0, -1,  0],
                                    [0,  0, -1],
                                    [1,  0,  0]], dtype=float)
        r_cam = R_FLU_to_opencv @ r_flu
        z_cam = r_cam[2]
        if z_cam <= 0.01:
            return

        u = self.fx * (r_cam[0] / z_cam) + self.cx
        v = self.fy * (r_cam[1] / z_cam) + self.cy
        pix_r  = max(int(self.fx * self.balloon_r / z_cam), 4)
        left   = int(u - pix_r)
        top    = int(v - pix_r)
        right  = int(u + pix_r)
        bottom = int(v + pix_r)

        in_fov = (right > 0 and left < self.width and
                  bottom > 0 and top < self.height)
        if not in_fov:
            return

        yolo_msg        = Yolov8Inference()
        yolo_msg.header.stamp = self.get_clock().now().to_msg()
        det            = InferenceResult()
        det.class_name = 'balloon'
        det.left       = max(left,   0)
        det.top        = max(top,    0)
        det.right      = min(right,  self.width  - 1)
        det.bottom     = min(bottom, self.height - 1)
        yolo_msg.yolov8_inference.append(det)
        self.yolov8_pub.publish(yolo_msg)

    # ----------------------------------------------------------------------- #
    #  Main callback (visualization only — detection published by timer)       #
    # ----------------------------------------------------------------------- #
    def camera_callback(self, msg: Image):
        """Project balloon GT position onto the image and publish Yolov8Inference."""
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # Wait until all Gazebo data has been received at least once
        missing = []
        if self.balloon_enu   is None: missing.append(f'model:{self.balloon_model}')
        if self.camera_enu    is None: missing.append(f'link:{self.camera_link_name}')
        if self.R_enu_to_cam  is None: missing.append('camera orientation')
        if missing:
            self._put_status(cv_img, f'Waiting for Gazebo: {missing}', COLOR_WAITING)
            self._publish_image(cv_img, msg)
            self.get_logger().warn(f'[GT] Waiting: {missing}', throttle_duration_sec=3.0)
            return

        # Step 1: Relative vector in Gazebo ENU (balloon w.r.t. camera center)
        rel_enu = self.balloon_enu - self.camera_enu

        # Step 2: Rotate ENU vector into Gazebo SDF link frame (FLU convention).
        #   R_enu_to_cam is built from the camera link quaternion in link_states,
        #   which already encodes drone pitch/roll/yaw and gimbal state.
        r_flu = self.R_enu_to_cam @ rel_enu

        # Step 2b: Align from Gazebo FLU link frame to OpenCV camera frame.
        #   Gazebo SDF link frame: X=forward, Y=left, Z=up
        #   OpenCV camera frame:   X=right,   Y=down, Z=forward
        R_FLU_to_opencv = np.array([[0, -1,  0],
                                    [0,  0, -1],
                                    [1,  0,  0]], dtype=float)
        r_cam = R_FLU_to_opencv @ r_flu
        z_cam = r_cam[2]   # depth along camera optical axis (OpenCV Z = forward)

        # Balloon is behind the camera
        if z_cam <= 0.01:
            status = f'Balloon BEHIND camera | z_cam={z_cam:.2f}m'
            self._put_status(cv_img, status, COLOR_BEHIND)
            self._publish_image(cv_img, msg)
            self.get_logger().info(f'[GT] {status}', throttle_duration_sec=2.0)
            return

        # Step 3: Pinhole projection -> pixel (u, v)
        u = self.fx * (r_cam[0] / z_cam) + self.cx
        v = self.fy * (r_cam[1] / z_cam) + self.cy

        # Step 4: Pixel radius proportional to balloon sphere radius
        pix_r  = max(int(self.fx * self.balloon_r / z_cam), 4)
        left   = int(u - pix_r)
        top    = int(v - pix_r)
        right  = int(u + pix_r)
        bottom = int(v + pix_r)
        iu, iv = int(u), int(v)

        in_fov = (right > 0 and left < self.width and
                  bottom > 0 and top < self.height)

        # --- Publish Yolov8Inference when balloon is within the FOV ---------
        yolo_msg        = Yolov8Inference()
        yolo_msg.header = msg.header
        if in_fov:
            det            = InferenceResult()
            det.class_name = 'balloon'
            det.left       = max(left,   0)
            det.top        = max(top,    0)
            det.right      = min(right,  self.width  - 1)
            det.bottom     = min(bottom, self.height - 1)
            yolo_msg.yolov8_inference.append(det)
            self.yolov8_pub.publish(yolo_msg)
            self.get_logger().info(
                f'[GT] DETECTED | bbox=({det.left},{det.top},{det.right},{det.bottom}) '
                f'center=({iu},{iv}) z={z_cam:.2f}m',
                throttle_duration_sec=1.0,
            )

        # --- Visualization --------------------------------------------------
        if in_fov:
            cv2.rectangle(cv_img, (left, top), (right, bottom), COLOR_DETECTED, 3)
            cv2.drawMarker(cv_img, (iu, iv), COLOR_DETECTED,
                           markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
            label = f'balloon GT  z={z_cam:.1f}m'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lx = max(left, 0)
            ly = max(top - 8, th + 4)
            cv2.rectangle(cv_img, (lx, ly - th - 4), (lx + tw + 4, ly + 2), (0, 0, 0), -1)
            cv2.putText(cv_img, label, (lx + 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_DETECTED, 2)
        else:
            # Arrow pointing toward the balloon's projected direction
            cx_img, cy_img = self.width // 2, self.height // 2
            dx, dy = iu - cx_img, iv - cy_img
            norm   = math.hypot(dx, dy) or 1.0
            margin = 40
            ex = int(cx_img + dx / norm * (min(cx_img, cy_img) - margin))
            ey = int(cy_img + dy / norm * (min(cx_img, cy_img) - margin))
            cv2.arrowedLine(cv_img, (cx_img, cy_img), (ex, ey),
                            COLOR_OUT_OF_FOV, 2, tipLength=0.2)
            status = f'Balloon out of FOV | proj=({iu},{iv}) z={z_cam:.1f}m'
            self._put_status(cv_img, status, COLOR_OUT_OF_FOV)
            self.get_logger().info(f'[GT] {status}', throttle_duration_sec=2.0)

        # Debug overlay (bottom-left)
        debug_lines = [
            f'camera ENU:  ({self.camera_enu[0]:.1f}, {self.camera_enu[1]:.1f}, {self.camera_enu[2]:.1f})',
            f'balloon ENU: ({self.balloon_enu[0]:.1f}, {self.balloon_enu[1]:.1f}, {self.balloon_enu[2]:.1f})',
            f'r_cam: ({r_cam[0]:.2f}, {r_cam[1]:.2f}, {r_cam[2]:.2f})',
            f'in_fov={in_fov}  z={z_cam:.2f}m',
        ]
        for i, line in enumerate(debug_lines):
            cv2.putText(cv_img, line, (6, self.height - 10 - i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        self._publish_image(cv_img, msg)

    # ----------------------------------------------------------------------- #
    #  Helpers                                                                  #
    # ----------------------------------------------------------------------- #
    def _put_status(self, img: np.ndarray, text: str, color):
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _publish_image(self, cv_img: np.ndarray, original_msg: Image):
        try:
            out        = bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            out.header = original_msg.header
            self.img_pub.publish(out)
        except Exception as e:
            self.get_logger().error(f'Image publish error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = GtBalloonDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
