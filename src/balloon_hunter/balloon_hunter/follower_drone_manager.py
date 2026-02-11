#!/usr/bin/env python3
"""
Follower Drone Manager (Drone 2, 3)
수정 사항:
- 디지털 줌(Digital Zoom) 기능 추가: 타겟 감지 시 200x200 크롭 후 640x480으로 확대 출력
- 로깅 강화 및 이륙 판정 로직 안정화
- 동적 ROI 크기 계산 및 시각화 개선
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    Monitoring
)
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
from enum import Enum

class FollowerState(Enum):
    IDLE = 0
    TAKEOFF = 1
    FORMATION_FLIGHT = 2
    HOVERING = 3

class FollowerDroneManager(Node):
    def __init__(self):
        super().__init__("follower_drone_manager")

        # Parameters
        self.declare_parameter('drone_id', 2) #이거 왜 2지... 
        self.declare_parameter('takeoff_height', 2.0)
        self.declare_parameter('formation_angle', 30.0)
        self.declare_parameter('formation_distance', 4.0)
        self.declare_parameter('leader_drone_id', 1)

        # Camera intrinsics
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('focal_length', 203.7)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)

        # ROI & Zoom parameters
        self.declare_parameter('target_physical_size', 0.5)
        self.declare_parameter('roi_offset_ratio', 1.5)
        self.zoom_crop_size = 200  # 디지털 줌을 위한 크롭 영역 크기 (200x200)

        self.drone_id = self.get_parameter('drone_id').value
        self.system_id = self.drone_id
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.formation_angle_rad = math.radians(self.get_parameter('formation_angle').value)
        self.formation_distance = self.get_parameter('formation_distance').value
        self.leader_drone_id = self.get_parameter('leader_drone_id').value
        self.target_physical_size = self.get_parameter('target_physical_size').value
        self.roi_offset_ratio = self.get_parameter('roi_offset_ratio').value

        # Camera Setup
        self.img_width = self.get_parameter('image_width').value
        self.img_height = self.get_parameter('image_height').value
        fx = self.get_parameter('focal_length').value
        self.K = np.array([[fx, 0, self.get_parameter('cx').value],
                          [0, fx, self.get_parameter('cy').value],
                          [0, 0, 1]], dtype=np.float32)
        self.camera_info_received = False

        self.topic_prefix_fmu = f"drone{self.drone_id}/fmu/"
        self.state = FollowerState.IDLE
        self.drone_pos = np.array([0.0, 0.0, 0.0])
        self.drone_yaw = 0.0
        self.drone_pitch = 0.0
        self.leader_pos = None
        self.leader_yaw = None
        self.target_global_pos = None
        self.predicted_uv = None
        self.predicted_roi_size = 50
        self.nav_state = 0
        self.arming_state = 0
        self.last_cmd_time = 0
        self.current_image = None
        self.bridge = CvBridge()

        # Publishers
        self.ocm_publisher = self.create_publisher(OffboardControlMode, f'{self.topic_prefix_fmu}in/offboard_control_mode', qos_profile_sensor_data)
        self.traj_setpoint_publisher = self.create_publisher(TrajectorySetpoint, f'{self.topic_prefix_fmu}in/trajectory_setpoint', qos_profile_sensor_data)
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, f'{self.topic_prefix_fmu}in/vehicle_command', qos_profile_sensor_data)

        # Subscribers
        self.status_sub = self.create_subscription(VehicleStatus, f'{self.topic_prefix_fmu}out/vehicle_status', self.status_callback, qos_profile_sensor_data)
        self.monitoring_sub = self.create_subscription(Monitoring, f'{self.topic_prefix_fmu}out/monitoring', self.monitoring_callback, qos_profile_sensor_data)
        
        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.leader_monitoring_sub = self.create_subscription(Monitoring, f'/drone{self.leader_drone_id}/fmu/out/monitoring', self.leader_monitoring_callback, qos_best_effort)
        self.target_sub = self.create_subscription(PoseStamped, '/balloon_target_position', self.target_callback, 10)
        self.image_sub = self.create_subscription(Image, f'/drone{self.drone_id}/camera/image_undistorted', self.image_callback, 1)
        self.camera_info_sub = self.create_subscription(CameraInfo, f'/drone{self.drone_id}/camera/camera_info', self.camera_info_callback, 1)

        # Timers
        self.create_timer(0.1, self.timer_ocm_callback)
        self.create_timer(0.04, self.timer_mission_callback)
        self.start_mission_timer = self.create_timer(10.0, self.start_mission)

        self.window_name = f'Follower {self.drone_id} Digital Zoom'
        self.window_created = False

    def status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def monitoring_callback(self, msg):
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        self.drone_yaw = msg.head
        self.drone_pitch = msg.pitch

    def leader_monitoring_callback(self, msg):
        self.leader_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        self.leader_yaw = msg.head

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.K = np.array([[msg.k[0], msg.k[1], msg.k[2]],
                              [msg.k[3], msg.k[4], msg.k[5]],
                              [msg.k[6], msg.k[7], msg.k[8]]], dtype=np.float32)
            self.camera_info_received = True
            self.get_logger().info(f'[Follower {self.drone_id}] CameraInfo Fixed: fx={msg.k[0]:.2f}')

    def target_callback(self, msg):
        self.target_global_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.get_logger().info(f'[Follower {self.drone_id}] Received Global Target: NED({self.target_global_pos[0]:.2f}, {self.target_global_pos[1]:.2f}, {self.target_global_pos[2]:.2f})')
        result = self.project_global_to_uv(self.target_global_pos)
        if result:
            self.predicted_uv, self.predicted_roi_size = result
            u, v = self.predicted_uv
            # 타겟이 화면 안에 들어오고 Formation 상태일 때만 Hovering 전환
            if 0 <= u < self.img_width and 0 <= v < self.img_height and self.state == FollowerState.FORMATION_FLIGHT:
                self.get_logger().info(f'[Follower {self.drone_id}] Target in Sight! Switching to HOVERING.')
                self.state = FollowerState.HOVERING
        else:
            self.predicted_uv = None

    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.visualize_reprojection()
        except Exception as e: 
            self.get_logger().error(f'Image error: {e}')

    def project_global_to_uv(self, target_global):
        dx, dy, dz = target_global - self.drone_pos

        # 1. Yaw 회전 (NED → Body)
        cos_yaw, sin_yaw = np.cos(self.drone_yaw), np.sin(self.drone_yaw)
        Xb = dx * cos_yaw + dy * sin_yaw
        Yb = -dx * sin_yaw + dy * cos_yaw
        Zb = dz

        # 2. Pitch 회전 (Body frame에서 Y축 기준)
        cos_pitch, sin_pitch = np.cos(self.drone_pitch), np.sin(self.drone_pitch)
        Xb_rot = Xb * cos_pitch + Zb * sin_pitch
        Zb_rot = -Xb * sin_pitch + Zb * cos_pitch

        # 3. Camera frame으로 변환 (Xc=right, Yc=down, Zc=forward)
        Xc, Yc, Zc = Yb, Zb_rot, Xb_rot
        if Zc <= 0.1: return None

        u = self.K[0, 0] * (Xc / Zc) + self.K[0, 2]
        v = self.K[1, 1] * (Yc / Zc) + self.K[1, 2]

        roi_size_pixels = (self.target_physical_size * self.K[0, 0]) / Zc
        roi_size_with_offset = int(roi_size_pixels * self.roi_offset_ratio)
        roi_size_with_offset = max(30, min(roi_size_with_offset, 200))

        return ((u, v), roi_size_with_offset)

    def visualize_reprojection(self):
        """디지털 줌이 적용된 시각화 함수 - 타겟이 있을 때만 창 표시"""
        if self.current_image is None or self.predicted_uv is None:
            return  # 타겟이 없으면 아무것도 하지 않음

        # 타겟이 있을 때만 창 생성
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.window_created = True

        raw_img = self.current_image.copy()
        h, w = raw_img.shape[:2]

        u, v = map(int, self.predicted_uv)

        # 1. 크롭 영역 계산 (200x200)
        half_zoom = self.zoom_crop_size // 2
        x1 = max(0, min(u - half_zoom, w - self.zoom_crop_size))
        y1 = max(0, min(v - half_zoom, h - self.zoom_crop_size))
        x2 = x1 + self.zoom_crop_size
        y2 = y1 + self.zoom_crop_size

        # 2. 이미지 크롭 및 확대
        crop_img = raw_img[y1:y2, x1:x2]
        display_img = cv2.resize(crop_img, (w, h), interpolation=cv2.INTER_LINEAR)

        # 3. 확대된 이미지에서의 좌표 보정 및 표시
        zoom_u = int((u - x1) * (w / self.zoom_crop_size))
        zoom_v = int((v - y1) * (h / self.zoom_crop_size))

        # ROI 박스 (확대된 크기에 맞춰 그림)
        half_roi = int((self.predicted_roi_size // 2) * (w / self.zoom_crop_size))
        cv2.rectangle(display_img, (zoom_u - half_roi, zoom_v - half_roi),
                     (zoom_u + half_roi, zoom_v + half_roi), (0, 255, 0), 3)
        cv2.drawMarker(display_img, (zoom_u, zoom_v), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

        cv2.putText(display_img, "DIGITAL ZOOM ACTIVE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(self.window_name, display_img)
        cv2.waitKey(1)

    def start_mission(self):
        self.start_mission_timer.cancel()
        if self.state == FollowerState.IDLE:
            self.state = FollowerState.TAKEOFF

    def timer_ocm_callback(self):
        msg = OffboardControlMode()
        msg.position, msg.velocity, msg.timestamp = True, False, int(self.get_clock().now().nanoseconds / 1000)
        self.ocm_publisher.publish(msg)

    def timer_mission_callback(self):
        if self.state == FollowerState.IDLE:
            self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], max(self.drone_pos[2], -0.1)])
        elif self.state == FollowerState.TAKEOFF: self.handle_takeoff()
        elif self.state == FollowerState.FORMATION_FLIGHT: self.handle_formation_flight()
        elif self.state == FollowerState.HOVERING: self.handle_hovering()

    def handle_takeoff(self):
        target_alt = -self.takeoff_height
        now = self.get_clock().now().nanoseconds / 1e9

        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED or self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], max(self.drone_pos[2], -0.1)])
        else:
            self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], target_alt])

        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if now - self.last_cmd_time > 1.0:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.last_cmd_time = now
            return

        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if now - self.last_cmd_time > 1.0:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.last_cmd_time = now
            return

        if abs(self.drone_pos[2] - target_alt) < 0.3:
            self.state = FollowerState.FORMATION_FLIGHT

    def handle_formation_flight(self):
        if self.leader_pos is None or self.leader_yaw is None:
            self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], -self.takeoff_height])
            return

        if self.drone_id % 2 == 0:
            layer = self.drone_id // 2
            theta = self.formation_angle_rad  # 왼쪽
        else:
            layer = (self.drone_id - 1) // 2
            theta = -self.formation_angle_rad   # 오른쪽

        distance = self.formation_distance * layer
        target_radian = self.leader_yaw + np.pi + theta
        target_x = self.leader_pos[0] + distance * np.cos(target_radian)
        target_y = self.leader_pos[1] + distance * np.sin(target_radian)

        # 헤딩: follower에서 leader를 바라보는 방향 (target_radian의 반대 방향) <이거 아님

        # target_yaw = target_radian - np.pi
        target_yaw = self.leader_yaw + theta

        self.publish_trajectory_setpoint([target_x, target_y, -self.takeoff_height], yaw=target_yaw)

    def handle_hovering(self):
        self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], -self.takeoff_height])

    def publish_trajectory_setpoint(self, pos, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        msg.yaw, msg.timestamp = float(yaw), int(self.get_clock().now().nanoseconds / 1000)
        self.traj_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.param1, msg.param2, msg.command = p1, p2, command
        msg.target_system, msg.target_component, msg.source_system, msg.source_component, msg.from_external = self.system_id, 1, self.system_id, 1, True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FollowerDroneManager()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()