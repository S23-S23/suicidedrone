#!/usr/bin/env python3
"""
Position Estimator Node
Estimates 3D position of detected balloons based on camera geometry and drone position
Based on box2image approach from box2image_ref_image_backup.py
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from yolov8_msgs.msg import Yolov8Inference
from px4_msgs.msg import Monitoring


def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)


def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)


def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)


class PositionEstimator(Node):
    def __init__(self):
        super().__init__('position_estimator')

        # Camera intrinsic parameters (typhoon_h480)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 360)
        self.declare_parameter('fx', 205.5)
        self.declare_parameter('fy', 205.5)
        self.declare_parameter('cx', 320)
        self.declare_parameter('cy', 180)
        # self.declare_parameter('width', 1280)
        # self.declare_parameter('height', 720)
        # self.declare_parameter('fx', 678.8712179620)
        # self.declare_parameter('fy', 676.5923040326)
        # self.declare_parameter('cx', 600.7451721112)
        # self.declare_parameter('cy', 363.7283523432)
        # Camera pitch from SDF: 1.0 radian = 57.3 degrees (pointing down)
        # In our convention, positive pitch = down
        self.declare_parameter('cam_pitch_deg', 57.3)

        # Topics
        self.declare_parameter('system_id', 1)
        self.declare_parameter('detection_topic', '/Yolov8_Inference_1')
        self.declare_parameter('position_topic', '/drone1/fmu/out/vehicle_local_position')
        self.declare_parameter('monitoring_topic', '/drone1/fmu/out/monitoring')
        self.declare_parameter('target_position_topic', '/balloon_target_position')

        # Get parameters
        self.system_id = self.get_parameter('system_id').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.cam_pitch = math.radians(self.get_parameter('cam_pitch_deg').value)

        self.detection_topic = self.get_parameter('detection_topic').value
        self.position_topic = self.get_parameter('position_topic').value
        self.monitoring_topic = self.get_parameter('monitoring_topic').value
        self.target_position_topic = self.get_parameter('target_position_topic').value

        # Drone state
        self.drone_pos = np.array([0.0, 0.0, 0.0])  # NED frame
        self.drone_yaw = 0.0
        self.drone_pitch = 0.0

        # Camera rotation matrices
        # ROS/OpenCV Camera frame: X=right, Y=down, Z=forward
        # Drone Body frame (FRD): X=forward, Y=right, Z=down
        # NED World frame: X=North, Y=East, Z=Down
        #
        # Camera aligned with body (no pitch):
        # cam_z (forward) -> body_x (forward)
        # cam_x (right) -> body_y (right)
        # cam_y (down) -> body_z (down)
        self.R_b_c_align = np.array([[0, 0, 1],  # row 0: body_x from camera
                                      [1, 0, 0],  # row 1: body_y from camera
                                      [0, 1, 0]], # row 2: body_z from camera
                                      dtype=float)
        # Apply pitch rotation BEFORE alignment
        # Camera pitched down by cam_pitch degrees (57.3 deg from SDF)
        # # Positive rotation around X-axis rotates Y->Z (pointing down)
        # self.R_b_cam_fixed = self.R_b_c_align @ rot_x(self.cam_pitch)
        self.R_b_cam_fixed = self.R_b_c_align @ rot_x(-self.cam_pitch)
    

        # QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.detection_sub = self.create_subscription(
            Yolov8Inference,
            self.detection_topic,
            self.detection_callback,
            10
        )

        # Use Monitoring topic only (VehicleLocalPosition not reliable in this setup)
        self.monitoring_sub = self.create_subscription(
            Monitoring,
            self.monitoring_topic,
            self.monitoring_callback,
            qos_profile
        )

        # Publisher
        self.target_pub = self.create_publisher(
            PoseStamped,
            self.target_position_topic,
            10
        )

        # self.get_logger().info('Position Estimator with Pitch-compensated initialized')
        # self.get_logger().info(f'Camera: fx={self.fx:.2f}, fy={self.fy:.2f}, pitch={self.get_parameter("cam_pitch_deg").value}°')
        # self.get_logger().info(f'Subscribing to: {self.detection_topic}')

    def monitoring_callback(self, msg: Monitoring):
        """Update drone position from PX4 Monitoring"""
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        self.drone_yaw = msg.head  # radians
        self.drone_pitch = msg.pitch 
        self.get_logger().info(f'[DEBUG] Pos: ({msg.pos_x:.2f}, {msg.pos_y:.2f}, {msg.pos_z:.2f}), Y={msg.head:.3f}, P={msg.pitch:.3f}', throttle_duration_sec=5.0)

    def detection_callback(self, msg: Yolov8Inference):
        """Process detections and estimate 3D position"""
        self.get_logger().info(f'[DEBUG] Position estimator: Detection callback triggered, detections={len(msg.yolov8_inference)}', throttle_duration_sec=2.0)

        if not msg.yolov8_inference:
            self.get_logger().warn('[DEBUG] Position estimator: No detections in message', throttle_duration_sec=2.0)
            return

        # Process first detection only
        det = msg.yolov8_inference[0]

        # Use bottom center of bounding box
        u = (det.left + det.right) * 0.5
        v = det.bottom

        self.get_logger().info(f'[DEBUG] Bbox: u={u:.1f}, v={v:.1f}, drone_pos=({self.drone_pos[0]:.2f},{self.drone_pos[1]:.2f},{self.drone_pos[2]:.2f}), yaw_rad={self.drone_yaw:.3f}', throttle_duration_sec=2.0)

        # Convert pixel to normalized camera coordinates
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy

        # Ray in camera frame (ROS/OpenCV convention: X=right, Y=down, Z=forward)
        r_cam = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])

        # Transform to body frame
        r_body = self.R_b_cam_fixed @ r_cam

        # Transform to NED frame (only yaw, ignore roll/pitch)
        R_n_b = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch)
        r_ned = R_n_b @ r_body

        # Normalize ray
        if np.linalg.norm(r_ned) < 1e-6:
            self.get_logger().warn('[DEBUG] Ray norm too small, skipping', throttle_duration_sec=2.0)
            return
        r_ned_norm = r_ned / np.linalg.norm(r_ned)

        # Calculate intersection with balloon height
        C_z = float(self.drone_pos[2])
        r_z = float(r_ned_norm[2])

        self.get_logger().info(f'[DEBUG] Ray: r_ned_norm=({r_ned_norm[0]:.3f},{r_ned_norm[1]:.3f},{r_ned_norm[2]:.3f}), C_z={C_z:.2f}, r_z={r_z:.3f}', throttle_duration_sec=2.0)

        if abs(r_z) < 1e-6:
            self.get_logger().warn(f'[DEBUG] r_z too small ({r_z:.6f}), skipping', throttle_duration_sec=2.0)
            return

        # Target height in NED frame
        # Gazebo world (ENU): balloon at z=2m
        # NED: z = -world_z = -2m (2m above ground, which is negative in NED Down)
        target_height = -2.0  # Balloon height in NED frame (2m up = -2 in NED)
        t = (target_height - C_z) / r_z

        self.get_logger().info(f'[DEBUG] t={(target_height - C_z):.2f}/{r_z:.3f}={t:.2f}', throttle_duration_sec=2.0)

        if t <= 0.0:
            self.get_logger().warn(f'[DEBUG] t={t:.2f} <= 0, skipping (ray pointing away from target)', throttle_duration_sec=2.0)
            return

        # Calculate global NED position
        target_pos = self.drone_pos + t * r_ned_norm

        # Calculate distance
        relative_pos = target_pos - self.drone_pos
        distance = np.linalg.norm(relative_pos[:2])

        # Publish target position
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(target_pos[0])
        pose_msg.pose.position.y = float(target_pos[1])
        pose_msg.pose.position.z = float(target_pos[2])

        self.target_pub.publish(pose_msg)

        self.get_logger().info(
            f'[DEBUG] Publishing target position: NED=({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}m), Dist={distance:.2f}m, topic={self.target_position_topic}',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = PositionEstimator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()