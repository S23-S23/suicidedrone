#!/usr/bin/env python3
"""
GCS Station Node - Ground Control Station with ROI Selection and Depth Camera
Displays depth camera feed, allows operator to select target ROI, uses depth data, and publishes 3D position
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, RegionOfInterest
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from px4_msgs.msg import Monitoring


class GCSStation(Node):
    """Ground Control Station for manual target selection with depth camera"""

    def __init__(self):
        super().__init__('gcs_station')

        # Parameters
        self.declare_parameter('system_id', 1)
        self.declare_parameter('rgb_camera_topic', '/camera/image_raw')
        self.declare_parameter('depth_camera_topic', '/camera/depth/image_raw')
        self.declare_parameter('target_position_topic', '/balloon_target_position')
        self.declare_parameter('monitoring_topic', '/drone1/fmu/out/monitoring')
        self.declare_parameter('display_fps', 10)

        # Depth camera parameters
        self.declare_parameter('focal_length', 424.0)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('cam_pitch_deg', 0.0)

        self.system_id = self.get_parameter('system_id').value
        rgb_topic = self.get_parameter('rgb_camera_topic').value
        depth_topic = self.get_parameter('depth_camera_topic').value
        target_topic = self.get_parameter('target_position_topic').value
        monitoring_topic = self.get_parameter('monitoring_topic').value
        display_fps = self.get_parameter('display_fps').value

        self.fx = self.get_parameter('focal_length').value
        self.fy = self.fx
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.cam_pitch_rad = np.radians(self.get_parameter('cam_pitch_deg').value)

        # CV Bridge for ROS-OpenCV conversion
        self.bridge = CvBridge()

        # Camera image storage
        self.rgb_image = None
        self.depth_image = None
        self.display_image = None

        # ROI selection state
        self.roi_start = None
        self.roi_end = None
        self.roi_selecting = False
        self.roi_selected = False
        self.current_roi = None  # (x, y, w, h)

        # Drone state
        self.drone_position = None
        self.drone_yaw = None
        self.drone_pitch = None

        # Tracking state
        self.tracker = None
        self.is_tracking = False

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self.rgb_image_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_image_callback, 10
        )

        # QoS profile for PX4 topics (BEST_EFFORT to match PX4 publisher)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.monitoring_sub = self.create_subscription(
            Monitoring, monitoring_topic, self.monitoring_callback, qos_profile
        )

        # Publisher for target position
        self.target_pub = self.create_publisher(PoseStamped, target_topic, 10)

        # Display and processing timers
        self.display_timer = self.create_timer(1.0 / display_fps, self.display_callback)
        self.process_timer = self.create_timer(0.1, self.process_callback)  # 10Hz

        # OpenCV window setup
        self.window_name = f'GCS Station - Drone {self.system_id}'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 848, 480)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.get_logger().info(f'GCS Station initialized for Drone {self.system_id} (Depth Camera Mode)')
        self.get_logger().info('Instructions: Drag mouse to select ROI, press "r" to reset, "q" to quit')

    def rgb_image_callback(self, msg):
        """Receive RGB camera image"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert RGB image: {e}')

    def depth_image_callback(self, msg):
        """Receive depth camera image"""
        try:
            # Depth images are typically 32FC1 (float32) or 16UC1 (uint16)
            # Both RGB and Depth are 848x480 from RealSense D455
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.get_logger().info(
                f'Depth image received: {self.depth_image.shape}, '
                f'range=[{self.depth_image.min():.2f}, {self.depth_image.max():.2f}]',
                throttle_duration_sec=5.0
            )
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')

    def monitoring_callback(self, msg):
        """Receive drone position and orientation"""
        self.drone_position = (msg.pos_x, msg.pos_y, msg.pos_z)
        self.drone_yaw = msg.head
        self.drone_pitch = msg.pitch

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
            self.roi_selecting = True
            self.roi_selected = False
            self.get_logger().info(f'ROI selection started at ({x}, {y})')

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.roi_selecting:
                self.roi_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.roi_selecting:
                self.roi_end = (x, y)
                self.roi_selecting = False

                if self.roi_start and self.roi_end:
                    x1, y1 = self.roi_start
                    x2, y2 = self.roi_end

                    x_min = min(x1, x2)
                    y_min = min(y1, y2)
                    x_max = max(x1, x2)
                    y_max = max(y1, y2)

                    width = x_max - x_min
                    height = y_max - y_min

                    if width > 10 and height > 10:
                        self.current_roi = (x_min, y_min, width, height)
                        self.roi_selected = True
                        self.initialize_tracker()
                        self.get_logger().info(
                            f'ROI selected: x={x_min}, y={y_min}, w={width}, h={height}'
                        )
                    else:
                        self.get_logger().warn('ROI too small, selection cancelled')
                        self.roi_start = None
                        self.roi_end = None

    def initialize_tracker(self):
        """Initialize tracker with selected ROI"""
        if self.current_roi and self.rgb_image is not None:
            # For Gazebo simulation with static targets, tracking is optional
            # Try to initialize CSRT tracker, but don't fail if it doesn't work
            try:
                # Try different OpenCV tracker APIs (4.5.1+ changed location)
                if hasattr(cv2, 'TrackerCSRT_create'):
                    self.tracker = cv2.TrackerCSRT_create()
                elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                    self.tracker = cv2.legacy.TrackerCSRT_create()
                else:
                    # OpenCV tracking not available
                    self.get_logger().info('OpenCV tracker not available, using fixed ROI mode')
                    self.is_tracking = False
                    self.tracker = None
                    return

                success = self.tracker.init(self.rgb_image, self.current_roi)
                if success:
                    self.is_tracking = True
                    self.get_logger().info('Tracker initialized successfully')
                else:
                    self.get_logger().warn('Tracker init failed, using fixed ROI mode')
                    self.is_tracking = False
                    self.tracker = None
            except Exception as e:
                self.get_logger().warn(f'Tracker creation failed: {e}, using fixed ROI mode')
                self.is_tracking = False
                self.tracker = None

    def process_callback(self):
        """Main processing: track ROI, get depth from depth image, publish position"""
        if not self.roi_selected:
            return

        if self.rgb_image is None:
            self.get_logger().warn('RGB image not available', throttle_duration_sec=5.0)
            return

        if self.depth_image is None:
            self.get_logger().warn('Depth image not available', throttle_duration_sec=5.0)
            return

        # Update tracker (if successful, update ROI; if failed, keep using last ROI)
        if self.is_tracking and self.tracker:
            success, bbox = self.tracker.update(self.rgb_image)
            if success:
                self.current_roi = tuple(map(int, bbox))
                self.get_logger().info('Tracking update successful', throttle_duration_sec=2.0)
            else:
                self.get_logger().warn('Tracking lost, using last known ROI', throttle_duration_sec=2.0)
                self.is_tracking = False
                # Don't return - continue with last known ROI position

        if not self.current_roi:
            self.get_logger().warn('No ROI selected', throttle_duration_sec=5.0)
            return

        if self.drone_position is None:
            self.get_logger().warn('Drone position not available', throttle_duration_sec=5.0)
            return

        try:
            # Get depth of ROI from depth image
            depth = self.compute_roi_depth(self.depth_image, self.current_roi)
            if depth is None or depth <= 0 or np.isnan(depth) or np.isinf(depth):
                self.get_logger().warn('Invalid depth value', throttle_duration_sec=2.0)
                return

            # Convert to 3D position in NED frame
            ned_pos = self.compute_ned_position(self.current_roi, depth)
            if ned_pos is None:
                return

            # Publish target position
            pose_msg = PoseStamped()
            pose_msg.header = Header()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = ned_pos[0]
            pose_msg.pose.position.y = ned_pos[1]
            pose_msg.pose.position.z = ned_pos[2]
            pose_msg.pose.orientation.w = 1.0

            self.target_pub.publish(pose_msg)

            self.get_logger().info(
                f'[TARGET] Depth={depth:.2f}m, NED=({ned_pos[0]:.2f}, {ned_pos[1]:.2f}, {ned_pos[2]:.2f})',
                throttle_duration_sec=1.0
            )

        except Exception as e:
            self.get_logger().error(f'Processing error: {e}')

    def compute_roi_depth(self, depth_map, roi):
        """Compute median depth of ROI from depth image"""
        x, y, w, h = roi
        x = max(0, min(x, depth_map.shape[1] - 1))
        y = max(0, min(y, depth_map.shape[0] - 1))
        w = min(w, depth_map.shape[1] - x)
        h = min(h, depth_map.shape[0] - y)

        if w <= 0 or h <= 0:
            self.get_logger().warn(f'Invalid ROI dimensions: w={w}, h={h}')
            return None

        # Extract ROI depth values
        roi_depth = depth_map[y:y+h, x:x+w]

        # Filter out invalid depth values (0, inf, nan)
        valid_depths = roi_depth[(roi_depth > 0) & np.isfinite(roi_depth)]

        if len(valid_depths) < 10:
            self.get_logger().warn(
                f'Insufficient valid depth values: {len(valid_depths)}/10 required. '
                f'ROI: ({x},{y},{w},{h}), depth range: [{roi_depth.min():.2f}, {roi_depth.max():.2f}]',
                throttle_duration_sec=2.0
            )
            return None

        # Return median depth
        median_depth = np.median(valid_depths)
        self.get_logger().info(
            f'ROI depth - median: {median_depth:.2f}m, valid pixels: {len(valid_depths)}, '
            f'min: {valid_depths.min():.2f}m, max: {valid_depths.max():.2f}m',
            throttle_duration_sec=2.0
        )
        return median_depth

    def compute_ned_position(self, roi, depth):
        """Convert ROI + depth to NED world position"""
        x, y, w, h = roi
        u = x + w / 2.0
        v = y + h / 2.0

        # Camera frame
        X_cam = (u - self.cx) * depth / self.fx
        Y_cam = (v - self.cy) * depth / self.fy
        Z_cam = depth

        # Body frame (pitch rotation)
        cos_p = np.cos(self.cam_pitch_rad)
        sin_p = np.sin(self.cam_pitch_rad)
        X_body = Z_cam * cos_p - Y_cam * sin_p
        Y_body = X_cam
        Z_body = Z_cam * sin_p + Y_cam * cos_p

        # NED frame (yaw rotation + translation)
        if self.drone_yaw is None:
            return None

        cos_yaw = np.cos(self.drone_yaw)
        sin_yaw = np.sin(self.drone_yaw)
        X_ned_rel = X_body * cos_yaw - Y_body * sin_yaw
        Y_ned_rel = X_body * sin_yaw + Y_body * cos_yaw
        Z_ned_rel = Z_body

        drone_x, drone_y, drone_z = self.drone_position
        return (drone_x + X_ned_rel, drone_y + Y_ned_rel, drone_z + Z_ned_rel)

    def display_callback(self):
        """Display RGB camera feed with ROI overlay"""
        if self.rgb_image is None:
            return

        # Display RGB image only
        self.display_image = self.rgb_image.copy()

        # Add camera label
        cv2.putText(self.display_image, 'RGB Camera (848x480)', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw ROI
        if self.roi_selected and self.current_roi:
            x, y, w, h = self.current_roi
            color = (0, 255, 0) if self.is_tracking else (0, 255, 255)
            thickness = 3
            cv2.rectangle(self.display_image, (x, y), (x + w, y + h), color, thickness)

            # Add tracking status label
            if self.is_tracking:
                label = f'TRACKING: {w}x{h}'
                label_color = (0, 255, 0)  # Green
            else:
                label = f'LOCKED: {w}x{h}'
                label_color = (0, 255, 255)  # Cyan

            # Background for label
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(self.display_image, (x, y - label_h - 10), (x + label_w + 10, y), (0, 0, 0), -1)
            cv2.putText(self.display_image, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

        elif self.roi_selecting and self.roi_start and self.roi_end:
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(self.display_image, 'SELECTING...', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Instructions panel (top right area)
        instructions = [
            'CONTROLS:',
            '  Mouse Drag: Select Target',
            '  R: Reset Selection',
            '  Q: Quit'
        ]
        y_start = 60
        for i, text in enumerate(instructions):
            font_scale = 0.6 if i == 0 else 0.5
            thickness = 2 if i == 0 else 1
            cv2.putText(self.display_image, text, (10, y_start + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Status bar at bottom
        status_bar_height = 40
        status_y = self.display_image.shape[0] - status_bar_height

        # Semi-transparent background for status
        overlay = self.display_image.copy()
        cv2.rectangle(overlay, (0, status_y), (self.display_image.shape[1], self.display_image.shape[0]),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, self.display_image, 0.4, 0, self.display_image)

        # Status text
        if self.is_tracking:
            status = 'STATUS: TRACKING TARGET (ACTIVE)'
            status_color = (0, 255, 0)
        elif self.roi_selected:
            status = 'STATUS: TARGET LOCKED (FIXED MODE)'
            status_color = (0, 255, 255)
        else:
            status = 'STATUS: WAITING FOR TARGET SELECTION'
            status_color = (100, 100, 255)

        cv2.putText(self.display_image, status,
                   (10, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        cv2.imshow(self.window_name, self.display_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quit requested')
            rclpy.shutdown()
        elif key == ord('r'):
            self.reset_roi()

    def reset_roi(self):
        """Reset ROI selection and tracking"""
        self.roi_start = None
        self.roi_end = None
        self.roi_selecting = False
        self.roi_selected = False
        self.current_roi = None
        self.tracker = None
        self.is_tracking = False
        self.get_logger().info('ROI selection reset')

    def destroy_node(self):
        """Cleanup on shutdown"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GCSStation()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()