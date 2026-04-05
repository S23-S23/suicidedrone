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

The cgo3_camera_link in Gazebo link_states uses the Gazebo SDF FLU convention
(when drone is level, facing North with yaw=1.5708):
  link-X = ENU-Y (North  = forward)
  link-Y = -ENU-X (West  = left)
  link-Z = ENU-Z  (Up    = up)

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
  /target_info                  - suicide_drone_msgs/TargetInfo (bounding box)
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates, LinkStates
from suicide_drone_msgs.msg import TargetInfo


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
        super().__init__('target_detector')  # keep same node name as YOLO version

        # --- Parameters (same structure as balloon_detector) ----------------
        self.declare_parameter('system_id', 1)
        self.declare_parameter('camera_topic', '/drone1/camera/image_raw')
        # Camera intrinsics – must match position_estimator defaults
        self.declare_parameter('width',  640)
        self.declare_parameter('height', 360)
        self.declare_parameter('fx', 205.5)
        self.declare_parameter('fy', 205.5)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 180.0)
        # GT-specific parameters
        self.declare_parameter('balloon_model_name',  'target_balloon')
        self.declare_parameter('balloon_radius',      0.3)
        # balloon_link <pose>0 0 1.5 0 0 0</pose> in model.sdf:
        # model_states returns the model root pose; add this for the sphere center.
        self.declare_parameter('balloon_link_z_offset', 1.5)
        # Camera link name in /gazebo/link_states: "<model>::<link>"
        self.declare_parameter('camera_link_name', 'drone1::cgo3_camera_link')

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

        # --- Publisher ----------------------------------------------------------
        self.target_pub = self.create_publisher(TargetInfo, '/target_info', 10)

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

        # Position: true pinhole (optical center) in ENU
        # /gazebo/link_states gives the cgo3_camera_link origin, NOT the sensor.
        # The SDF defines the camera sensor at <pose>0.0 0 -0.162 0 0 0</pose>
        # relative to the link origin in the link's local frame.
        # Convert that offset to world ENU and add it to the link origin.
        p = pose.position
        link_origin_enu = np.array([float(p.x), float(p.y), float(p.z)])
        sensor_offset_link = np.array([0.0, 0.0, -0.162])
        R_link_to_world = self.R_enu_to_cam.T
        self.camera_enu = link_origin_enu + R_link_to_world @ sensor_offset_link

    # ----------------------------------------------------------------------- #
    #  Main callback                                                            #
    # ----------------------------------------------------------------------- #
    def camera_callback(self, msg: Image):
        """Project balloon GT position onto the image plane and publish TargetInfo."""
        # Wait until all Gazebo data has been received at least once
        if self.balloon_enu is None or self.camera_enu is None or self.R_enu_to_cam is None:
            self.get_logger().warn('[GT] Waiting for Gazebo data', throttle_duration_sec=3.0)
            return

        # Step 1: Relative vector in Gazebo ENU (balloon w.r.t. camera center)
        rel_enu = self.balloon_enu - self.camera_enu

        # Step 2: Rotate ENU -> Gazebo SDF FLU link frame -> OpenCV camera frame
        r_flu = self.R_enu_to_cam @ rel_enu
        R_FLU_to_opencv = np.array([[0, -1,  0],
                                    [0,  0, -1],
                                    [1,  0,  0]], dtype=float)
        r_cam = R_FLU_to_opencv @ r_flu
        z_cam = r_cam[2]

        if z_cam <= 0.01:
            self.get_logger().info('[GT] Balloon behind camera', throttle_duration_sec=2.0)
            return

        # Step 3: Pinhole projection
        u = self.fx * (r_cam[0] / z_cam) + self.cx
        v = self.fy * (r_cam[1] / z_cam) + self.cy

        # Step 4: Pixel bounding box from projected sphere radius
        pix_r  = max(int(self.fx * self.balloon_r / z_cam), 4)
        left   = int(u - pix_r);  top    = int(v - pix_r)
        right  = int(u + pix_r);  bottom = int(v + pix_r)

        in_fov = (right > 0 and left < self.width and
                  bottom > 0 and top < self.height)

        if in_fov:
            target_msg            = TargetInfo()
            target_msg.header     = msg.header
            target_msg.class_name = 'balloon'
            target_msg.left       = max(left,   0)
            target_msg.top        = max(top,    0)
            target_msg.right      = min(right,  self.width  - 1)
            target_msg.bottom     = min(bottom, self.height - 1)
            self.target_pub.publish(target_msg)
            self.get_logger().info(
                f'[GT] DETECTED | bbox=({target_msg.left},{target_msg.top},'
                f'{target_msg.right},{target_msg.bottom}) z={z_cam:.2f}m',
                throttle_duration_sec=1.0,
            )
        else:
            self.get_logger().info(
                f'[GT] Out of FOV | proj=({int(u)},{int(v)}) z={z_cam:.1f}m',
                throttle_duration_sec=2.0,
            )


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
