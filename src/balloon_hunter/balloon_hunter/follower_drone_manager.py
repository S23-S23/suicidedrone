#!/usr/bin/env python3
"""
Follower Drone Manager (Drone 2, 3)
수정 사항: 로깅 강화 및 이륙 판정 로직 안정화
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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from enum import Enum
import numpy as np
import cv2
import math

class FollowerState(Enum):
    IDLE = 0
    TAKEOFF = 1
    FORMATION_FLIGHT = 2
    HOVERING = 3

class FollowerDroneManager(Node):
    def __init__(self):
        super().__init__("follower_drone_manager")

        # Parameters
        self.declare_parameter('drone_id', 2)
        self.declare_parameter('takeoff_height', 2.0)
        self.declare_parameter('formation_offset_x', 0.0)
        self.declare_parameter('formation_offset_y', 2.0)
        self.declare_parameter('leader_drone_id', 1)

        # Camera intrinsics (Fisheye)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('focal_length', 203.7)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)

        self.drone_id = self.get_parameter('drone_id').value
        self.system_id = self.drone_id  # PX4 MAVLink System ID (drone_id와 동일)
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.formation_offset_x = self.get_parameter('formation_offset_x').value
        self.formation_offset_y = self.get_parameter('formation_offset_y').value
        self.leader_drone_id = self.get_parameter('leader_drone_id').value

        # Camera Setup
        self.img_width = self.get_parameter('image_width').value
        self.img_height = self.get_parameter('image_height').value
        fx = self.get_parameter('focal_length').value
        self.K = np.array([[fx, 0, self.get_parameter('cx').value],
                          [0, fx, self.get_parameter('cy').value],
                          [0, 0, 1]], dtype=np.float32)

        self.topic_prefix_fmu = f"drone{self.drone_id}/fmu/"
        self.state = FollowerState.IDLE
        self.drone_pos = np.array([0.0, 0.0, 0.0])
        self.drone_yaw = 0.0
        self.leader_pos = None
        self.target_global_pos = None
        self.predicted_uv = None
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

        # Timers
        self.create_timer(0.1, self.timer_ocm_callback)
        self.create_timer(0.04, self.timer_mission_callback)
        # PX4 준비 대기 시간을 10초로 증가 (Leader보다 늦게 시작하므로)
        self.start_mission_timer = self.create_timer(10.0, self.start_mission)

        self.window_name = f'Follower {self.drone_id} Reprojection'
        self.window_created = False
        #self.get_logger().info(f'[Follower {self.drone_id}] Ready and waiting for mission (will start in 10s)...')

    def status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state
        #self.get_logger().info(f'[Follower {self.drone_id}] Status: nav={self.nav_state}, arm={self.arming_state}', throttle_duration_sec=5.0)

    def monitoring_callback(self, msg):
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        self.drone_yaw = msg.head

    def leader_monitoring_callback(self, msg):
        self.leader_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])

    def target_callback(self, msg):
        self.target_global_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.predicted_uv = self.project_global_to_uv(self.target_global_pos)
        if self.predicted_uv:
            u, v = self.predicted_uv
            if 0 <= u < self.img_width and 0 <= v < self.img_height and self.state == FollowerState.FORMATION_FLIGHT:
                self.get_logger().info(f'[Follower {self.drone_id}] Target detected in camera! Switching to HOVERING.')
                self.state = FollowerState.HOVERING

    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.predicted_uv is not None: self.visualize_reprojection()
        except Exception as e: self.get_logger().error(f'Image error: {e}')

    def project_global_to_uv(self, target_global):
        dx, dy, dz = target_global - self.drone_pos
        cy, sy = np.cos(self.drone_yaw), np.sin(self.drone_yaw)
        Xb = dx * cy + dy * sy
        Yb = -dx * sy + dy * cy
        Zb = dz
        Xc, Yc, Zc = Yb, Zb, Xb # Body to Camera (Forward-looking)
        if Zc <= 0: return None
        u = self.K[0, 0] * (Xc / Zc) + self.K[0, 2]
        v = self.K[1, 1] * (Yc / Zc) + self.K[1, 2]
        return (u, v)

    def visualize_reprojection(self):
        if self.current_image is None: return
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.window_created = True
        disp = self.current_image.copy()
        if self.predicted_uv:
            u, v = map(int, self.predicted_uv)
            if 0 <= u < self.img_width and 0 <= v < self.img_height:
                cv2.rectangle(disp, (u-25, v-25), (u+25, v+25), (0, 255, 0), 2)
                cv2.drawMarker(disp, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow(self.window_name, disp)
        cv2.waitKey(1)

    def start_mission(self):
        self.start_mission_timer.cancel()
        if self.state == FollowerState.IDLE:
            #self.get_logger().info(f'[Follower {self.drone_id}] Starting Takeoff sequence...')
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

        # Continuous setpoint streaming for Offboard
        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED or self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], max(self.drone_pos[2], -0.1)])
        else:
            self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], target_alt])

        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if now - self.last_cmd_time > 1.0:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.get_logger().info(f'[Follower {self.drone_id}] Sending ARM command...')
                self.last_cmd_time = now
            return

        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if now - self.last_cmd_time > 1.0:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.get_logger().info(f'[Follower {self.drone_id}] Sending OFFBOARD command...')
                self.last_cmd_time = now
            return

        if abs(self.drone_pos[2] - target_alt) < 0.3:
            self.get_logger().info(f'[Follower {self.drone_id}] Takeoff Success! Switching to Formation.')
            self.state = FollowerState.FORMATION_FLIGHT

    def handle_formation_flight(self):
        if self.leader_pos is None:
            self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], -self.takeoff_height])
            return
        target_x = self.leader_pos[0] + self.formation_offset_x
        target_y = self.leader_pos[1] + self.formation_offset_y
        self.publish_trajectory_setpoint([target_x, target_y, -self.takeoff_height])

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
        msg.target_system, msg.target_component, msg.source_system, msg.source_component, msg.from_external = self.system_id, 1, 1, 1, True
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