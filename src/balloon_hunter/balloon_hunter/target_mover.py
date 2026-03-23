#!/usr/bin/env python3
"""
Target Mover — constant-velocity left-right motion for DKF vs EKF comparison
==============================================================================
Moves 'target_balloon' in Gazebo X at constant speed, reversing direction
at ±amplitude. This creates a triangle-wave trajectory:

    speed = 2.0 m/s (constant)
    range = nominal_x ± amplitude

Updated at 50 Hz for smooth, continuous motion.
Motion is deterministic (same speed, same start direction every run).

Publishes current world position on /target_world_pos for logger GT.
"""

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Point


class TargetMover(Node):
    def __init__(self):
        super().__init__('target_mover')

        self.declare_parameter('target_name',  'target_balloon')
        self.declare_parameter('nominal_x',     0.0)   # Gazebo X centre
        self.declare_parameter('nominal_y',     5.0)   # Gazebo Y (depth, fixed)
        self.declare_parameter('nominal_z',     5.0)   # Gazebo Z (height, fixed)
        self.declare_parameter('amplitude',     4.0)   # ±4 m range in Gazebo X
        self.declare_parameter('speed',         0.3)   # m/s constant speed

        self.target_name = self.get_parameter('target_name').value
        self.nominal_x   = self.get_parameter('nominal_x').value
        self.nominal_y   = self.get_parameter('nominal_y').value
        self.nominal_z   = self.get_parameter('nominal_z').value
        self.amplitude   = self.get_parameter('amplitude').value
        self.speed       = self.get_parameter('speed').value

        # State: current offset from nominal_x and direction
        self._offset = 0.0        # current X offset from nominal
        self._direction = 0.5     # +1 = moving in +X, -1 = moving in -X
        self._dt = 0.5           # 50 Hz

        # Gazebo service — try both ROS2 and ROS1-style names
        self._cli = None
        for svc_name in ['/set_entity_state', '/gazebo/set_entity_state']:
            cli = self.create_client(SetEntityState, svc_name)
            if cli.wait_for_service(timeout_sec=5.0):
                self._cli = cli
                self.get_logger().info(f'Connected to Gazebo service: {svc_name}')
                break
            else:
                self.get_logger().warn(f'Service {svc_name} not available, trying next…')
                self.destroy_client(cli)

        if self._cli is None:
            self.get_logger().error(
                'No set_entity_state service found! '
                'Make sure libgazebo_ros_state.so is loaded in the world file.'
            )
            return

        # Position publisher for logger GT computation
        self._pos_pub = self.create_publisher(Point, '/target_world_pos', 10)

        # 50 Hz timer — every step moves exactly speed * dt = constant distance
        self.create_timer(self._dt, self._update)

        half_period = 2.0 * self.amplitude / self.speed
        self.get_logger().info(
            f'TargetMover started | centre=({self.nominal_x},{self.nominal_y},{self.nominal_z}) '
            f'| range=±{self.amplitude}m | speed={self.speed}m/s | half-period={half_period:.1f}s'
        )

    def _update(self):
        if self._cli is None:
            return

        # Constant velocity step
        self._offset += self._direction * self.speed * self._dt

        # Reverse at boundaries
        if self._offset >= self.amplitude:
            self._offset = self.amplitude
            self._direction = -1.0
        elif self._offset <= -self.amplitude:
            self._offset = -self.amplitude
            self._direction = 1.0

        x = self.nominal_x + self._offset
        y = self.nominal_y
        z = self.nominal_z

        # Move balloon in Gazebo
        req = SetEntityState.Request()
        req.state.name = self.target_name
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = z
        req.state.reference_frame = 'world'
        self._cli.call_async(req)

        # Publish for logger
        pt = Point()
        pt.x = x
        pt.y = y
        pt.z = z
        self._pos_pub.publish(pt)


def main(args=None):
    rclpy.init(args=args)
    node = TargetMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
