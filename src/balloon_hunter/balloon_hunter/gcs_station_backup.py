#!/usr/bin/env python3
"""
GCS Station Node - 원본 해상도 UI 및 추적 전용 모드
특징점 검출 성공 시에만 타겟 좌표를 발행하며, 실패 시 드론은 FORWARD 상태를 유지합니다.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from px4_msgs.msg import Monitoring

class GCSStation(Node):
    def __init__(self):
        super().__init__('gcs_station')

        # 1. 파라미터 선언 및 설정
        self.declare_parameter('system_id', 1)
        self.declare_parameter('rgb_camera_topic', '/camera/image_raw')
        self.declare_parameter('depth_camera_topic', '/camera/depth/image_raw')
        self.declare_parameter('target_position_topic', '/balloon_target_position')
        self.declare_parameter('monitoring_topic', '/drone1/fmu/out/monitoring')
        self.declare_parameter('display_fps', 10)
        self.declare_parameter('focal_length', 424.0)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)

        self.system_id = self.get_parameter('system_id').value
        self.fx = self.get_parameter('focal_length').value
        self.fy = self.fx
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value

        # 2. 내부 변수 초기화
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.drone_position = None
        self.drone_yaw = None
        
        self.roi_selecting = False
        self.roi_selected = False
        self.roi_start = None
        self.roi_end = None
        self.current_roi = None
        self.tracker = None
        self.is_tracking = False

        # 3. 구독자 및 발행자 설정 (Latency 최소화를 위해 depth=1)
        self.rgb_sub = self.create_subscription(
            Image, self.get_parameter('rgb_camera_topic').value, self.rgb_image_callback, 1)
        self.depth_sub = self.create_subscription(
            Image, self.get_parameter('depth_camera_topic').value, self.depth_image_callback, 1)
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.monitoring_sub = self.create_subscription(
            Monitoring, self.get_parameter('monitoring_topic').value, self.monitoring_callback, qos_profile)
        self.target_pub = self.create_publisher(PoseStamped, self.get_parameter('target_position_topic').value, 10)

        # 4. 타이머 및 윈도우 설정
        self.display_timer = self.create_timer(1.0 / self.get_parameter('display_fps').value, self.display_callback)
        self.process_timer = self.create_timer(0.1, self.process_callback) # 10Hz

        self.window_name = f'GCS Station - Drone {self.system_id}'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 848, 480)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.get_logger().info('GCS Node: Full Scale UI & Selective Tracking Active')

    # --- Callbacks ---
    def rgb_image_callback(self, msg):
        try: self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except: pass

    def depth_image_callback(self, msg):
        try: self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except: pass

    def monitoring_callback(self, msg):
        self.drone_position = (msg.pos_x, msg.pos_y, msg.pos_z)
        self.drone_yaw = msg.head

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 좌표를 원본 해상도(848x480)와 1:1 대응하여 처리"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
            self.roi_selecting = True
            self.roi_selected = False
        elif event == cv2.EVENT_MOUSEMOVE and self.roi_selecting:
            self.roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.roi_selecting:
            self.roi_selecting = False
            self.roi_end = (x, y)
            w, h = abs(self.roi_start[0] - x), abs(self.roi_start[1] - y)
            if w > 10 and h > 10:
                self.current_roi = (min(self.roi_start[0], x), min(self.roi_start[1], y), w, h)
                self.roi_selected = True
                self.initialize_tracker()

    # --- Core Processing ---
    def initialize_tracker(self):
        """CSRT 트래커 초기화. 실패 시 즉시 차단하여 드론의 FORWARD 상태 유지"""
        if self.current_roi and self.rgb_image is not None:
            # OpenCV 4.8.x 전용 CSRT 생성
            self.tracker = cv2.TrackerCSRT_create()
            success = self.tracker.init(self.rgb_image, tuple(map(int, self.current_roi)))
            
            if success:
                self.is_tracking = True
                self.get_logger().info('Tracker Init Success: Tracking Active')
            else:
                self.is_tracking = False
                self.tracker = None
                self.roi_selected = False
                self.get_logger().warn('Feature Detection Failed: Return to SCAN mode')

    def process_callback(self):
        """추적 중일 때만 좌표 발행. 실패/분실 시 발행 중단"""
        if not self.is_tracking or self.tracker is None:
            return

        success, bbox = self.tracker.update(self.rgb_image)
        if not success:
            self.get_logger().warn('Target Lost: Stopping transmission')
            self.reset_roi()
            return

        self.current_roi = tuple(map(int, bbox))
        depth = self.get_roi_depth(self.depth_image, self.current_roi)
        
        if depth and self.drone_position:
            ned_pos = self.calculate_ned(self.current_roi, depth)
            if ned_pos:
                msg = PoseStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'map'
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = ned_pos
                self.target_pub.publish(msg)

    def get_roi_depth(self, depth_map, roi):
        x, y, w, h = map(int, roi)
        area = depth_map[y:y+h, x:x+w]
        valid = area[(area > 0) & np.isfinite(area)]
        return np.median(valid) if len(valid) > 10 else None

    def calculate_ned(self, roi, depth):
        x, y, w, h = roi
        u, v = x + w/2.0, y + h/2.0
        # Camera to Body
        Xc, Yc, Zc = (u - self.cx)*depth/self.fx, (v - self.cy)*depth/self.fy, depth
        Xb, Yb, Zb = Zc, Xc, Yc 
        # Body to NED
        if self.drone_yaw is None: return None
        cy, sy = np.cos(self.drone_yaw), np.sin(self.drone_yaw)
        Xn = Xb * cy - Yb * sy + self.drone_position[0]
        Ye = Xb * sy + Yb * cy + self.drone_position[1]
        D = Zb + self.drone_position[2]
        return (Xn, Ye, D)

    # --- UI Rendering (원본 해상도 복구) ---
    def display_callback(self):
        if self.rgb_image is None: return
        
        disp = self.rgb_image.copy()
        h, w = disp.shape[:2]

        # 1. UI 안내 및 라벨
        cv2.putText(disp, f'Drone {self.system_id} - 848x480 View', (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 2. ROI 드로잉
        if self.roi_selected or self.roi_selecting:
            if self.roi_selecting:
                cv2.rectangle(disp, self.roi_start, self.roi_end, (0, 255, 255), 2)
            elif self.current_roi:
                rx, ry, rw, rh = map(int, self.current_roi)
                color = (0, 255, 0) if self.is_tracking else (0, 0, 255)
                cv2.rectangle(disp, (rx, ry), (rx+rw, ry+rh), color, 2)
                txt = "TRACKING" if self.is_tracking else "LOST"
                cv2.putText(disp, txt, (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. 우측 상단 가이드
        guide = ["Drag: Select Target", "R: Reset ROI", "Q: Quit"]
        for i, t in enumerate(guide):
            cv2.putText(disp, t, (w - 200, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 4. 하단 상태 바 (반투명)
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, disp, 0.4, 0, disp)

        status = "STATE: TRACKING" if self.is_tracking else "STATE: SCANNING (FORWARD)"
        color = (0, 255, 0) if self.is_tracking else (0, 255, 255)
        cv2.putText(disp, status, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(self.window_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): rclpy.shutdown()
        elif key == ord('r'): self.reset_roi()

    def reset_roi(self):
        self.is_tracking = False
        self.roi_selected = False
        self.tracker = None
        self.get_logger().info('Target Released: Returning to SCAN')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GCSStation()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()