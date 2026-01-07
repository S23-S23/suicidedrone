#!/usr/bin/env python3
"""
GCS Station Node - Ground Control Station with ROI Selection and Stereo Depth Estimation
Displays stereo camera feed, allows operator to select target ROI, computes depth, and publishes 3D position
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, RegionOfInterest
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from px4_msgs.msg import Monitoring


class GCSStation(Node):
    """Ground Control Station for manual target selection with stereo depth estimation"""

    def __init__(self):
        super().__init__('gcs_station')

        # Parameters
        self.declare_parameter('system_id', 1)
        self.declare_parameter('left_camera_topic', '/left_camera/image_raw')
        self.declare_parameter('right_camera_topic', '/right_camera/image_raw')
        self.declare_parameter('target_position_topic', '/balloon_target_position')
        self.declare_parameter('monitoring_topic', '/drone1/fmu/out/monitoring')
        self.declare_parameter('display_fps', 10)

        # Stereo parameters
        self.declare_parameter('baseline', 0.07)
        self.declare_parameter('focal_length', 205.5)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('cam_pitch_deg', 57.3)

        self.system_id = self.get_parameter('system_id').value
        left_topic = self.get_parameter('left_camera_topic').value
        right_topic = self.get_parameter('right_camera_topic').value
        target_topic = self.get_parameter('target_position_topic').value
        monitoring_topic = self.get_parameter('monitoring_topic').value
        display_fps = self.get_parameter('display_fps').value

        self.baseline = self.get_parameter('baseline').value
        self.fx = self.get_parameter('focal_length').value
        self.fy = self.fx
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.cam_pitch_rad = np.radians(self.get_parameter('cam_pitch_deg').value)

        # CV Bridge for ROS-OpenCV conversion
        self.bridge = CvBridge()

        # Camera image storage
        self.left_image = None
        self.right_image = None
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

        # Stereo matcher
        self.stereo_matcher = cv2.StereoBM_create(numDisparities=64, blockSize=15)

        # Tracking state
        self.tracker = None
        self.is_tracking = False

        # Subscribers
        self.left_sub = self.create_subscription(
            Image, left_topic, self.left_image_callback, 10
        )
        self.right_sub = self.create_subscription(
            Image, right_topic, self.right_image_callback, 10
        )
        self.monitoring_sub = self.create_subscription(
            Monitoring, monitoring_topic, self.monitoring_callback, 10
        )

        # Publisher for target position
        self.target_pub = self.create_publisher(PoseStamped, target_topic, 10)

        # Display and processing timers
        self.display_timer = self.create_timer(1.0 / display_fps, self.display_callback)
        self.process_timer = self.create_timer(0.1, self.process_callback)  # 10Hz

        # OpenCV window setup
        self.window_name = f'GCS Station - Drone {self.system_id}'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 480)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.get_logger().info(f'GCS Station initialized for Drone {self.system_id}')
        self.get_logger().info('Instructions: Drag mouse to select ROI, press "r" to reset, "q" to quit')

    def left_image_callback(self, msg):
        """Receive left camera image"""
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert left image: {e}')

    def right_image_callback(self, msg):
        """Receive right camera image"""
        try:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert right image: {e}')

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
        if self.current_roi and self.left_image is not None:
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

                success = self.tracker.init(self.left_image, self.current_roi)
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
        """Main processing: track ROI, compute depth, publish position"""
        if not self.roi_selected or self.left_image is None or self.right_image is None:
            return

        # Update tracker (if successful, update ROI; if failed, keep using last ROI)
        if self.is_tracking and self.tracker:
            success, bbox = self.tracker.update(self.left_image)
            if success:
                self.current_roi = tuple(map(int, bbox))
                self.get_logger().info('Tracking update successful', throttle_duration_sec=2.0)
            else:
                self.get_logger().warn('Tracking lost, using last known ROI', throttle_duration_sec=2.0)
                self.is_tracking = False
                # Don't return - continue with last known ROI position

        if not self.current_roi or self.drone_position is None:
            return

        try:
            # Compute stereo depth
            left_gray = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(self.right_image, cv2.COLOR_BGR2GRAY)
            disparity = self.stereo_matcher.compute(left_gray, right_gray)

            # Get depth of ROI
            depth = self.compute_roi_depth(disparity, self.current_roi)
            if depth is None:
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

    def compute_roi_depth(self, disparity_map, roi):
        """Compute median depth of ROI"""
        x, y, w, h = roi
        x = max(0, min(x, disparity_map.shape[1] - 1))
        y = max(0, min(y, disparity_map.shape[0] - 1))
        w = min(w, disparity_map.shape[1] - x)
        h = min(h, disparity_map.shape[0] - y)

        if w <= 0 or h <= 0:
            return None

        roi_disparity = disparity_map[y:y+h, x:x+w].astype(np.float32) / 16.0
        valid_disparities = roi_disparity[roi_disparity > 0]

        if len(valid_disparities) < 10:
            return None

        depths = (self.baseline * self.fx) / valid_disparities
        return np.median(depths)

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
        """Display camera feed with ROI overlay"""
        if self.left_image is None:
            return

        # Create side-by-side display with labels
        left_display = self.left_image.copy()

        # Add camera labels
        cv2.putText(left_display, 'LEFT CAMERA', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        if self.right_image is not None:
            right_display = self.right_image.copy()
            cv2.putText(right_display, 'RIGHT CAMERA', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            # Stack horizontally with separator line
            separator = np.ones((left_display.shape[0], 3, 3), dtype=np.uint8) * 255
            self.display_image = np.hstack([left_display, separator, right_display])
        else:
            self.display_image = left_display

        # Draw ROI on left side only
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