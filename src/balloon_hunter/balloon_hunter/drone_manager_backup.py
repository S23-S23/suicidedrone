#!/usr/bin/env python3
"""
Balloon Hunter Drone Manager
Simplified drone manager for balloon hunting mission
Based on drone_manager.py structure
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus
)
from geometry_msgs.msg import PoseStamped
from enum import Enum
import numpy as np
import math


class MissionState(Enum):
    """Mission states for balloon hunting"""
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
        self.declare_parameter('takeoff_height', 5.0)
        self.declare_parameter('forward_speed', 2.0)
        self.declare_parameter('tracking_speed', 3.0)
        self.declare_parameter('charge_speed', 5.0)
        self.declare_parameter('charge_distance', 3.0)
        self.declare_parameter('collision_distance', 0.5)

        self.system_id = self.get_parameter('system_id').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.forward_speed = self.get_parameter('forward_speed').value
        self.tracking_speed = self.get_parameter('tracking_speed').value
        self.charge_speed = self.get_parameter('charge_speed').value
        self.charge_distance = self.get_parameter('charge_distance').value
        self.collision_distance = self.get_parameter('collision_distance').value

        self.get_logger().info(f"Balloon Hunter Drone Manager {self.system_id}")

        # Topic prefixes
        self.topic_prefix_fmu = f"drone{self.system_id}/fmu/"

        # State variables
        self.state = MissionState.IDLE
        self.drone_pos = np.array([0.0, 0.0, 0.0])  # NED
        self.drone_yaw = 0.0
        self.target_pos = None
        self.takeoff_start_pos = None
        self.nav_state = 0

        # OCM message
        self.ocm_msg = OffboardControlMode()
        self.ocm_msg.position = True
        self.ocm_msg.velocity = False
        self.ocm_msg.acceleration = False
        self.ocm_msg.attitude = False
        self.ocm_msg.body_rate = False

        # Publishers
        self.ocm_publisher = self.create_publisher(
            OffboardControlMode,
            f'{self.topic_prefix_fmu}in/offboard_control_mode',
            qos_profile_sensor_data
        )

        self.traj_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint,
            f'{self.topic_prefix_fmu}in/trajectory_setpoint',
            qos_profile_sensor_data
        )

        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand,
            f'{self.topic_prefix_fmu}in/vehicle_command',
            qos_profile_sensor_data
        )

        # Subscribers
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            f'{self.topic_prefix_fmu}out/vehicle_local_position',
            self.position_callback,
            qos_profile_sensor_data
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            f'{self.topic_prefix_fmu}out/vehicle_status',
            self.status_callback,
            qos_profile_sensor_data
        )

        self.target_sub = self.create_subscription(
            PoseStamped,
            '/balloon_target_position',
            self.target_callback,
            10
        )

        # Timers
        self.timer_ocm = self.create_timer(0.1, self.timer_ocm_callback)
        self.timer_mission = self.create_timer(0.04, self.timer_mission_callback)  # 25 Hz

        self.get_logger().info("Balloon Hunter Drone Manager initialized")

        # Auto-start mission after 3 seconds
        self.create_timer(10.0, self.start_mission)

    def position_callback(self, msg: VehicleLocalPosition):
        """Update drone position"""
        self.drone_pos = np.array([msg.x, msg.y, msg.z])
        self.drone_yaw = msg.heading

    def status_callback(self, msg: VehicleStatus):
        """Update vehicle status"""
        self.nav_state = msg.nav_state

    def target_callback(self, msg: PoseStamped):
        """Update balloon target position"""
        self.target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        # Transition to TRACKING if currently in FORWARD
        if self.state == MissionState.FORWARD:
            self.get_logger().info('Balloon detected! Switching to TRACKING')
            self.state = MissionState.TRACKING

    def start_mission(self):
        if self.state == MissionState.IDLE:
            self.get_logger().info('Starting mission sequence...')
            # 1. 먼저 상태를 TAKEOFF로 바꿔서 Setpoint가 나가게 함
            self.state = MissionState.TAKEOFF 
            self.takeoff_start_pos = self.drone_pos.copy()

            # 2. 약간의 시간차(0.5s)를 두고 시동(Arm) 명령
            self.create_timer(3.0, lambda: self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0))
            
            # 3. 다시 시간차(1.0s)를 두고 오프보드 모드 전환
            self.create_timer(3.0, lambda: self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0))

    def timer_ocm_callback(self):
        """Publish offboard control mode"""
        self.ocm_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.ocm_publisher.publish(self.ocm_msg)

    def timer_mission_callback(self):
        """Main mission control loop"""
        if self.state == MissionState.IDLE:
            self.publish_trajectory_setpoint(position=[0.0, 0.0, 0.0])
            return
        elif self.state == MissionState.TAKEOFF:
            self.handle_takeoff()
        elif self.state == MissionState.FORWARD:
            self.handle_forward()
        elif self.state == MissionState.TRACKING:
            self.handle_tracking()
        elif self.state == MissionState.CHARGING:
            self.handle_charging()
        elif self.state == MissionState.DONE:
            self.handle_done()

    def handle_takeoff(self):
        """Handle takeoff state"""
        target_alt = -self.takeoff_height  # NED: negative is up

        if abs(self.drone_pos[2] - target_alt) < 0.3:
            self.get_logger().info('Takeoff complete. Starting FORWARD flight')
            self.state = MissionState.FORWARD
            return

        self.publish_trajectory_setpoint(
            position=[self.takeoff_start_pos[0], self.takeoff_start_pos[1], target_alt],
            yaw=0.0
        )

    def handle_forward(self):
        """Handle forward flight"""
        # Move forward at constant speed
        next_pos = self.drone_pos.copy()
        next_pos[0] += self.forward_speed * 0.04  # dt = 0.04s

        self.publish_trajectory_setpoint(
            position=[next_pos[0], next_pos[1], -self.takeoff_height],
            yaw=0.0
        )

    def handle_tracking(self):
        """Handle target tracking"""
        if self.target_pos is None:
            self.get_logger().warn('Lost target, returning to FORWARD')
            self.state = MissionState.FORWARD
            return

        # Calculate distance
        diff = self.target_pos - self.drone_pos
        distance = np.linalg.norm(diff[:2])

        # Switch to CHARGING if close enough
        if distance < self.charge_distance:
            self.get_logger().info(f'Target in range ({distance:.2f}m)! Starting CHARGE')
            self.state = MissionState.CHARGING
            return

        # Move towards target
        direction = diff / (np.linalg.norm(diff) + 1e-6)
        next_pos = self.drone_pos + direction * self.tracking_speed * 0.04

        yaw = math.atan2(diff[1], diff[0])

        self.publish_trajectory_setpoint(
            position=[next_pos[0], next_pos[1], -self.takeoff_height],
            yaw=yaw
        )

    def handle_charging(self):
        """Handle charging at target"""
        if self.target_pos is None:
            self.get_logger().warn('Lost target during charge')
            self.state = MissionState.DONE
            return

        # Calculate distance
        diff = self.target_pos - self.drone_pos
        distance = np.linalg.norm(diff)

        # Check collision
        if distance < self.collision_distance:
            self.get_logger().info('BALLOON HIT! Mission complete!')
            self.state = MissionState.DONE
            return

        # Full speed ahead!
        direction = diff / (np.linalg.norm(diff) + 1e-6)
        next_pos = self.drone_pos + direction * self.charge_speed * 0.04

        yaw = math.atan2(diff[1], diff[0])

        self.publish_trajectory_setpoint(
            position=[next_pos[0], next_pos[1], next_pos[2]],
            yaw=yaw
        )

    def handle_done(self):
        """Handle mission complete"""
        # Hover in place
        self.publish_trajectory_setpoint(
            position=[self.drone_pos[0], self.drone_pos[1], self.drone_pos[2]],
            yaw=self.drone_yaw
        )

    def publish_trajectory_setpoint(self, position, yaw=float('nan')):
        """Publish trajectory setpoint"""
        msg = TrajectorySetpoint()
        msg.position = [float(position[0]), float(position[1]), float(position[2])]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        """Publish vehicle command"""
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = self.system_id
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BalloonHunterDroneManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
