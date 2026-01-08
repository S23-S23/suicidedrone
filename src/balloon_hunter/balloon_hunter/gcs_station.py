#!/usr/bin/env python3
"""
GCS Station Node - Depth-based Foreground Tracking Mode
전통적인 트래커 대신 Depth 정보를 사용하여 사용자가 선택한 거리의 '물체 덩어리'를 추적합니다.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from px4_msgs.msg import Monitoring

class GCSStation(Node):
    def __init__(self):
        super().__init__('gcs_station')

        # 1. 파라미터 선언
        self.declare_parameter('system_id', 1)
        self.declare_parameter('rgb_camera_topic', '/camera/image_raw')
        self.declare_parameter('depth_camera_topic', '/camera/depth/image_raw')
        self.declare_parameter('target_position_topic', '/balloon_target_position')
        self.declare_parameter('monitoring_topic', '/drone1/fmu/out/monitoring')
        self.declare_parameter('display_fps', 5)
        self.declare_parameter('focal_length', 544.6)
        self.declare_parameter('cx', 424.0)
        self.declare_parameter('cy', 240.0)

        self.system_id = self.get_parameter('system_id').value
        self.fx = self.get_parameter('focal_length').value
        self.fy = self.fx
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value

        # 2. 내부 변수
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.drone_position = None
        self.drone_yaw = None
        
        # 추적 관련
        self.is_tracking = False
        self.roi_selecting = False
        self.roi_selected = False
        self.roi_start = None
        self.roi_end = None
        self.target_depth = None  # 추적 중인 물체의 거리
        self.target_center_uv = None # 추적 중인 물체의 2D 중심 (u, v)
        self.depth_threshold = 1.0 # 전경 분리를 위한 거리 허용 오차 (m)

        # 3. 구독/발행
        self.rgb_sub = self.create_subscription(Image, self.get_parameter('rgb_camera_topic').value, self.rgb_image_callback, 1)
        self.depth_sub = self.create_subscription(Image, self.get_parameter('depth_camera_topic').value, self.depth_image_callback, 1)
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.monitoring_sub = self.create_subscription(Monitoring, self.get_parameter('monitoring_topic').value, self.monitoring_callback, qos)
        self.target_pub = self.create_publisher(PoseStamped, self.get_parameter('target_position_topic').value, 10)

        # 4. 타이머 및 UI
        self.display_timer = self.create_timer(1.0 / self.get_parameter('display_fps').value, self.display_callback)
        self.process_timer = self.create_timer(0.05, self.process_callback) # 20Hz로 더 빠르게 추적

        self.window_name = f'GCS Station - Depth Tracking Mode'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 848, 480)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.get_logger().info('GCS Node: Depth-based Segmentation Tracking Active')

    def rgb_image_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_image_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')

    def monitoring_callback(self, msg):
        self.drone_position = (msg.pos_x, msg.pos_y, msg.pos_z)
        self.drone_yaw = msg.head

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
            self.roi_selecting = True
            self.roi_selected = False
            self.is_tracking = False
        elif event == cv2.EVENT_MOUSEMOVE and self.roi_selecting:
            self.roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.roi_selecting:
            self.roi_selecting = False
            self.roi_end = (x, y)
            w, h = abs(self.roi_start[0] - x), abs(self.roi_start[1] - y)
            if w > 5 and h > 5:
                roi = (min(self.roi_start[0], x), min(self.roi_start[1], y), w, h)
                self.start_depth_tracking(roi)

    def start_depth_tracking(self, roi):
        """선택된 영역의 깊이 정보를 기반으로 추적 초기화"""
        if self.depth_image is None: return
        
        x, y, w, h = roi
        roi_depth = self.depth_image[y:y+h, x:x+w]
        valid_depths = roi_depth[(roi_depth > 0.1) & (np.isfinite(roi_depth))]
        
        if len(valid_depths) > 0:
            self.target_depth = np.median(valid_depths)
            self.target_center_uv = (x + w/2, y + h/2)
            self.is_tracking = True
            self.roi_selected = True
            self.get_logger().info(f'Depth Tracking Initialized: Dist={self.target_depth:.2f}m')
        else:
            self.get_logger().warn('Invalid Depth in ROI: Cannot start tracking')

    def process_callback(self):
        """Depth Segmentation 기반 타겟 위치 갱신 및 발행"""
        if not self.is_tracking or self.depth_image is None or self.drone_position is None:
            return

        # 1. 현재 타겟 깊이 주변의 마스크 생성
        # 드론이 다가감에 따라 타겟 깊이는 실시간으로 변함
        min_d = self.target_depth - self.depth_threshold
        max_d = self.target_depth + self.depth_threshold
        
        # 0.5m ~ 50m 사이의 유효 거리만 고려
        mask = cv2.inRange(self.depth_image, max(0.3, min_d), min(50.0, max_d))
        
        # 2. 현재 추적 중인 위치 주변에서 가장 큰 덩어리 찾기 (Centroid)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None
        min_dist = float('inf')
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50: continue # 너무 작은 노이즈 제거
            
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 이전 중심점과 가장 가까운 덩어리를 타겟으로 선택
            dist = np.sqrt((cx - self.target_center_uv[0])**2 + (cy - self.target_center_uv[1])**2)
            if dist < min_dist and dist < 150: # 갑자기 너무 멀리 점프하는 것 방지
                min_dist = dist
                best_cnt = cnt
                self.target_center_uv = (cx, cy)

        if best_cnt is not None:
            # 타겟 깊이 업데이트 (추적 안정성 향상)
            new_depth = self.depth_image[int(self.target_center_uv[1]), int(self.target_center_uv[0])]
            if np.isfinite(new_depth) and new_depth > 0:
                self.target_depth = new_depth
            
            # NED 좌표 계산 및 발행
            ned_pos = self.calculate_ned(self.target_center_uv, self.target_depth)
            if ned_pos:
                msg = PoseStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'map'
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = ned_pos
                self.target_pub.publish(msg)
        else:
            # 추적 실패 시 즉시 중단하여 드론을 FORWARD 상태로 돌림
            self.get_logger().warn('Depth Object Lost: Target out of range')
            self.is_tracking = False
            self.roi_selected = False

    def calculate_ned(self, uv, depth):
        u, v = uv
        Xc, Yc, Zc = (u - self.cx)*depth/self.fx, (v - self.cy)*depth/self.fy, depth
        Xb, Yb, Zb = Zc, Xc, Yc 
        if self.drone_yaw is None: return None
        cy, sy = np.cos(self.drone_yaw), np.sin(self.drone_yaw)
        Xn = Xb * cy - Yb * sy + self.drone_position[0]
        Ye = Xb * sy + Yb * cy + self.drone_position[1]
        D = Zb + self.drone_position[2]
        return (Xn, Ye, D)

    def display_callback(self):
        if self.rgb_image is None: return
        
        disp = self.rgb_image.copy()
        h, w = disp.shape[:2]

        # UI 가이드
        cv2.putText(disp, f'DEPTH TRACKING MODE - {self.system_id}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.roi_selecting:
            cv2.rectangle(disp, self.roi_start, self.roi_end, (0, 255, 255), 2)
        
        if self.is_tracking and self.target_center_uv:
            # 추적 중인 지점에 조준선 표시
            u, v = map(int, self.target_center_uv)
            cv2.drawMarker(disp, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(disp, (u, v), 10, (0, 255, 0), 2)
            cv2.putText(disp, f'DIST: {self.target_depth:.2f}m', (u + 15, v - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 하단 상태 바
        status = "TRACKING: ACTIVE" if self.is_tracking else "STATUS: SCANNING (FORWARD)"
        color = (0, 255, 0) if self.is_tracking else (0, 255, 255)
        cv2.rectangle(disp, (0, h-35), (w, h), (0,0,0), -1)
        cv2.putText(disp, status, (15, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow(self.window_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'): self.is_tracking = False; self.roi_selected = False
        elif key == ord('q'): rclpy.shutdown()

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
