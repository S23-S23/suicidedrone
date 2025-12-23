#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    Monitoring
)
from geometry_msgs.msg import PoseStamped
from enum import Enum
import numpy as np
import math

class MissionState(Enum):
    IDLE = 0
    TAKEOFF = 1
    FORWARD = 2
    TRACKING = 3
    CHARGING = 4
    DONE = 5

class BalloonHunterDroneManager(Node):
    def __init__(self):
        super().__init__("balloon_hunter_drone_manager")

        # Parameters
        self.declare_parameter('system_id', 1)
        self.declare_parameter('takeoff_height', 6.0)
        self.declare_parameter('forward_speed', 15.0)
        self.declare_parameter('tracking_speed', 20.0)
        self.declare_parameter('charge_speed', 20.0)
        self.declare_parameter('charge_distance', 3.0)
        self.declare_parameter('collision_distance', 0.5)

        self.system_id = self.get_parameter('system_id').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.forward_speed = self.get_parameter('forward_speed').value
        self.tracking_speed = self.get_parameter('tracking_speed').value
        self.charge_speed = self.get_parameter('charge_speed').value
        self.charge_distance = self.get_parameter('charge_distance').value
        self.collision_distance = self.get_parameter('collision_distance').value

        self.get_logger().info(f"Drone Manager {self.system_id} Initializing with Monitoring...")

        self.topic_prefix_fmu = f"drone{self.system_id}/fmu/"

        # State variables
        self.state = MissionState.IDLE
        self.drone_pos = np.array([0.0, 0.0, 0.0])
        self.drone_yaw = 0.0
        self.target_pos = None
        self.nav_state = 0
        self.arming_state = 0
        self.monitoring_msg = Monitoring() # Monitoring 변수 초기화
        self.last_cmd_time = 0
        self.forward_start_pos = None  # Track starting position for forward flight
        self.forward_distance_limit = 10.0  # Maximum forward distance in meters

        # Publishers
        self.ocm_publisher = self.create_publisher(OffboardControlMode, f'{self.topic_prefix_fmu}in/offboard_control_mode', qos_profile_sensor_data)
        self.traj_setpoint_publisher = self.create_publisher(TrajectorySetpoint, f'{self.topic_prefix_fmu}in/trajectory_setpoint', qos_profile_sensor_data)
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, f'{self.topic_prefix_fmu}in/vehicle_command', qos_profile_sensor_data)

        # Subscribers
        # Use Monitoring for position data (VehicleLocalPosition not reliable in this setup)
        self.status_sub = self.create_subscription(VehicleStatus, f'{self.topic_prefix_fmu}out/vehicle_status', self.status_callback, qos_profile_sensor_data)
        self.monitoring_sub = self.create_subscription(Monitoring, f'{self.topic_prefix_fmu}out/monitoring', self.monitoring_callback, qos_profile_sensor_data)
        self.target_sub = self.create_subscription(PoseStamped, '/balloon_target_position', self.target_callback, 10)

        # Timers
        self.create_timer(0.1, self.timer_ocm_callback)
        self.create_timer(0.04, self.timer_mission_callback) # 25Hz

        # Wait for PX4 to be ready before starting mission (single shot timer)
        self.start_mission_timer = self.create_timer(5.0, self.start_mission)

    def status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def monitoring_callback(self, msg):
        """Monitoring 메시지로부터 드론 위치 및 상태 업데이트"""
        self.monitoring_msg = msg
        # Update drone position from monitoring
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        self.drone_yaw = msg.head  # Already in radians despite msg definition saying degrees
        self.get_logger().info(f'[DEBUG] Monitoring callback: pos=({msg.pos_x:.2f}, {msg.pos_y:.2f}, {msg.pos_z:.2f})', throttle_duration_sec=5.0)

    def target_callback(self, msg):
        self.target_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.get_logger().info(f'[DEBUG] Drone manager: Target callback triggered, pos=({self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}, {self.target_pos[2]:.2f}), current_state={self.state}', throttle_duration_sec=1.0)

        if self.state == MissionState.FORWARD:
            self.get_logger().info('Balloon detected! TRACKING start.')
            self.state = MissionState.TRACKING

    def start_mission(self):
        # Cancel the timer after first run (single-shot behavior)
        self.start_mission_timer.cancel()

        if self.state == MissionState.IDLE:
            self.get_logger().info('Switching to TAKEOFF state...')
            self.state = MissionState.TAKEOFF

    def timer_ocm_callback(self):
        msg = OffboardControlMode()
        msg.position, msg.velocity, msg.timestamp = True, False, int(self.get_clock().now().nanoseconds / 1000)
        self.ocm_publisher.publish(msg)

    def timer_mission_callback(self):
        # Log current state periodically
        self.get_logger().info(f'[DEBUG] Current state: {self.state}, pos=({self.drone_pos[0]:.2f},{self.drone_pos[1]:.2f},{self.drone_pos[2]:.2f}), armed={self.arming_state}, nav={self.nav_state}', throttle_duration_sec=3.0)

        if self.state == MissionState.IDLE:
            # Send safe initial position (current or slightly above ground)
            # Prevent sending [0,0,0] which commands drone to go underground
            safe_z = max(self.drone_pos[2], -0.1)  # Never command below -0.1m
            self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], safe_z])
            return

        if self.state == MissionState.TAKEOFF: self.handle_takeoff()
        elif self.state == MissionState.FORWARD: self.handle_forward()
        elif self.state == MissionState.TRACKING: self.handle_tracking()
        elif self.state == MissionState.CHARGING: self.handle_charging()
        elif self.state == MissionState.DONE: self.handle_done()

    def handle_takeoff(self):
        # Key: Must send trajectory setpoints continuously BEFORE arming/mode change
        # This satisfies PX4's offboard streaming requirement
        target_alt = -self.takeoff_height

        # Always publish setpoint first (critical for offboard mode)
        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED or self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            # Before armed/offboard: publish safe ground-level position
            safe_z = max(self.drone_pos[2], -0.1)  # Never command below -0.1m
            self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], safe_z])
        else:
            # After armed/offboard: publish target altitude
            self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], target_alt])

        now = self.get_clock().now().nanoseconds / 1e9

        # 1. Check and request ARM (retry every 1 second if failed)
        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if now - self.last_cmd_time > 1.0:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.get_logger().info('Attempting to ARM...')
                self.last_cmd_time = now
            return

        # 2. Check and request OFFBOARD mode (retry every 1 second if failed)
        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if now - self.last_cmd_time > 1.0:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.get_logger().info('Requesting OFFBOARD mode...')
                self.last_cmd_time = now
            return

        # 3. Check if target altitude reached
        if abs(self.drone_pos[2] - target_alt) < 0.3:
            self.get_logger().info('Takeoff Success! Starting forward flight.')
            self.forward_start_pos = self.drone_pos.copy()  # Save starting position
            self.state = MissionState.FORWARD
            return

    def handle_forward(self):
        # Initialize start position if not set
        if self.forward_start_pos is None:
            self.forward_start_pos = self.drone_pos.copy()
            self.get_logger().info(f'[DEBUG] Forward flight started from position: ({self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f})')

        # Check if forward distance limit reached
        distance_traveled = np.linalg.norm(self.drone_pos[:2] - self.forward_start_pos[:2])

        self.get_logger().info(f'[DEBUG] FORWARD state: pos=({self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f}, {self.drone_pos[2]:.2f}), traveled={distance_traveled:.2f}m', throttle_duration_sec=2.0)

        if distance_traveled >= self.forward_distance_limit:
            self.get_logger().info(f'Forward distance limit reached ({distance_traveled:.2f}m). Hovering in place.', throttle_duration_sec=2.0)
            # Hover in current position
            self.publish_trajectory_setpoint([self.drone_pos[0], self.drone_pos[1], -self.takeoff_height])
            return

        # Continue forward flight - fixed target position: x=0, y=+10, z=maintain altitude
        self.publish_trajectory_setpoint([5.0, 0.0, -self.takeoff_height])

    def handle_tracking(self):
        self.get_logger().info(f'[DEBUG] TRACKING state: drone=({self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f}), target={self.target_pos if self.target_pos is not None else "None"}', throttle_duration_sec=1.0)

        if self.target_pos is None:
            self.get_logger().warn('[DEBUG] Target lost, returning to FORWARD')
            self.state = MissionState.FORWARD
            return
        diff = self.target_pos - self.drone_pos
        if np.linalg.norm(diff[:2]) < self.charge_distance:
            self.state = MissionState.CHARGING
            return
        direction = diff / (np.linalg.norm(diff) + 1e-6)
        next_pos = self.drone_pos + direction * self.tracking_speed * 0.5
        self.publish_trajectory_setpoint([next_pos[0], next_pos[1], -self.takeoff_height], yaw=math.atan2(diff[1], diff[0]))

    def handle_charging(self):
        self.get_logger().info(f'[DEBUG] CHARGING state: drone=({self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f}), target={self.target_pos if self.target_pos is not None else "None"}', throttle_duration_sec=1.0)

        if self.target_pos is None:
            self.state = MissionState.DONE
            return
        diff = self.target_pos - self.drone_pos
        if np.linalg.norm(diff) < self.collision_distance:
            self.get_logger().info('HIT!')
            self.state = MissionState.DONE
            return
        direction = diff / (np.linalg.norm(diff) + 1e-6)
        next_pos = self.drone_pos + direction * self.charge_speed * 0.8
        self.publish_trajectory_setpoint(next_pos, yaw=math.atan2(diff[1], diff[0]))

    def handle_done(self):
        self.publish_trajectory_setpoint(self.drone_pos, yaw=self.drone_yaw)

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

def main(args=None):
    rclpy.init(args=args)
    node = BalloonHunterDroneManager()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
