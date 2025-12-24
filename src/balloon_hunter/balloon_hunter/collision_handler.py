#!/usr/bin/env python3
"""
Collision Handler Node
Monitors distance between drone and balloon, triggers balloon disappearance on collision
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleLocalPosition
from std_msgs.msg import Bool
import numpy as np


class CollisionHandler(Node):
    def __init__(self):
        super().__init__('collision_handler')

        # Parameters
        self.declare_parameter('collision_distance', 0.5)  # meters
        self.declare_parameter('drone_id', 1)

        self.collision_distance = self.get_parameter('collision_distance').value
        self.drone_id = self.get_parameter('drone_id').value

        # State
        self.drone_pos = None
        self.balloon_pos = None
        self.collision_detected = False

        # QoS
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.drone_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            f'drone{self.drone_id}/fmu/out/vehicle_local_position',
            self.drone_pos_callback,
            best_effort
        )

        self.balloon_pos_sub = self.create_subscription(
            PoseStamped,
            '/balloon_target_position',
            self.balloon_pos_callback,
            10
        )

        # Publisher
        self.collision_pub = self.create_publisher(
            Bool,
            '/balloon_collision',
            10
        )

        # Timer to check collision
        self.create_timer(0.1, self.check_collision)

        self.get_logger().info('Collision Handler initialized')

    def drone_pos_callback(self, msg: VehicleLocalPosition):
        """Update drone position"""
        self.drone_pos = np.array([msg.x, msg.y, msg.z])

    def balloon_pos_callback(self, msg: PoseStamped):
        """Update balloon position"""
        self.balloon_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

    def check_collision(self):
        """Check if drone collided with balloon"""
        if self.drone_pos is None or self.balloon_pos is None:
            return

        if self.collision_detected:
            return

        # Calculate distance
        distance = np.linalg.norm(self.drone_pos - self.balloon_pos)

        if distance < self.collision_distance:
            self.get_logger().info(f'COLLISION DETECTED! Distance: {distance:.3f}m')
            self.collision_detected = True

            # Publish collision event
            msg = Bool()
            msg.data = True
            self.collision_pub.publish(msg)

            # In a real Gazebo plugin, this would remove the balloon model
            # For now, just log the event
            self.get_logger().info('Balloon popped! (In real simulation, balloon model would be removed)')


def main(args=None):
    rclpy.init(args=args)
    node = CollisionHandler()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()