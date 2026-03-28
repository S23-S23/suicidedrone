#!/usr/bin/env python3
"""
Collision Handler Node
Monitors distance between drone and balloon using:
  - Drone position from PX4 Monitoring (NED frame)
  - Balloon position from Gazebo model_states (ENU frame, converted to NED)

Publishes /balloon_collision (Bool) when distance < collision_distance.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import Monitoring
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool
import numpy as np


class CollisionHandler(Node):
    def __init__(self):
        super().__init__('collision_handler')

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter('collision_distance',      0.5)
        self.declare_parameter('drone_id',                1)
        self.declare_parameter('balloon_model_name',      'target_balloon')
        # Z offset of balloon_link inside the model (model.sdf: <pose>0 0 1.5 0 0 0</pose>)
        self.declare_parameter('balloon_link_z_offset',   1.5)

        self.collision_distance    = self.get_parameter('collision_distance').value
        self.drone_id              = self.get_parameter('drone_id').value
        self.balloon_model_name    = self.get_parameter('balloon_model_name').value
        self.balloon_link_z_offset = self.get_parameter('balloon_link_z_offset').value

        # ── State ───────────────────────────────────────────────────────────
        self.drone_pos_ned   = None   # NED [m] from Monitoring
        self.balloon_pos_ned = None   # NED [m] converted from Gazebo ENU
        self.collision_detected = False

        # ── QoS ─────────────────────────────────────────────────────────────
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Subscriptions ────────────────────────────────────────────────────
        # Drone position from PX4 Monitoring (NED local frame)
        self.create_subscription(
            Monitoring,
            f'drone{self.drone_id}/fmu/out/monitoring',
            self.monitoring_callback,
            best_effort,
        )
        # Balloon position from Gazebo model_states (ENU world frame)
        self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_states_callback,
            10,
        )

        # ── Publisher ────────────────────────────────────────────────────────
        self.collision_pub = self.create_publisher(Bool, '/balloon_collision', 10)

        # Collision check at 10 Hz
        self.create_timer(0.1, self.check_collision)

        self.get_logger().info(
            f'CollisionHandler: drone{self.drone_id}, '
            f'balloon="{self.balloon_model_name}", '
            f'threshold={self.collision_distance}m'
        )

    def monitoring_callback(self, msg: Monitoring):
        """Update drone position from PX4 Monitoring (NED local frame)."""
        self.drone_pos_ned = np.array([msg.pos_x, msg.pos_y, msg.pos_z])

    def model_states_callback(self, msg: ModelStates):
        """
        Update balloon position from Gazebo model_states.
        Gazebo uses ENU world frame; convert to NED for comparison with Monitoring.
        ENU → NED: ned_x = enu_y, ned_y = enu_x, ned_z = -enu_z
        """
        if self.balloon_model_name not in msg.name:
            return

        idx = msg.name.index(self.balloon_model_name)
        p   = msg.pose[idx].position

        # Balloon sphere center in ENU (model root + link Z offset)
        enu_x = float(p.x)
        enu_y = float(p.y)
        enu_z = float(p.z) + self.balloon_link_z_offset

        # Convert ENU → NED
        # NED home assumed at Gazebo world origin (spawn point ≈ ENU (0,0,0))
        ned_x =  enu_y    # North = ENU-Y
        ned_y =  enu_x    # East  = ENU-X
        ned_z = -enu_z    # Down  = −ENU-Z

        self.balloon_pos_ned = np.array([ned_x, ned_y, ned_z])

    def check_collision(self):
        """Publish collision event when drone is within collision_distance of balloon."""
        if self.drone_pos_ned is None or self.balloon_pos_ned is None:
            return
        if self.collision_detected:
            return

        distance = np.linalg.norm(self.drone_pos_ned - self.balloon_pos_ned)

        if distance < self.collision_distance:
            self.get_logger().info(
                f'COLLISION DETECTED! Distance: {distance:.3f}m'
            )
            self.collision_detected = True

            msg      = Bool()
            msg.data = True
            self.collision_pub.publish(msg)


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
