#!/usr/bin/env python3
"""
Drone Visualizer for Real Flight
==================================
Same as drone_visualizer.py but without Gazebo dependencies.
No balloon ground-truth marker (no /gazebo/model_states).

Publishes:
  - TF:                    map -> drone{id}
  - nav_msgs/Path:         drone trajectory
  - geometry_msgs/PoseStamped: current drone pose
  - visualization_msgs/Marker: mission state text
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import Monitoring
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster

_STATE_COLOR = {
    'INIT':       (0.6, 0.6, 0.6),
    'HOVER_INIT': (1.0, 1.0, 0.0),
    'TRACKING':   (0.0, 1.0, 0.0),
}


def ned_to_enu(x_ned, y_ned, z_ned):
    return y_ned, x_ned, -z_ned


def ned_yaw_to_enu_quat(ned_yaw_rad: float) -> Quaternion:
    enu_yaw = math.pi / 2.0 - ned_yaw_rad
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(enu_yaw / 2.0)
    q.w = math.cos(enu_yaw / 2.0)
    return q


class DroneVisualizerReal(Node):
    def __init__(self):
        super().__init__('drone_visualizer')

        self.declare_parameter('system_id', 1)
        self.declare_parameter('max_path_points', 5000)

        system_id            = self.get_parameter('system_id').value
        self.max_path_points = self.get_parameter('max_path_points').value
        self.frame_id        = 'map'
        self.drone_frame     = f'drone{system_id}'

        self.tf_broadcaster = TransformBroadcaster(self)

        self.path_pub       = self.create_publisher(Path,        '/drone/trajectory',  10)
        self.pose_pub       = self.create_publisher(PoseStamped, '/drone/pose',        10)
        self.state_text_pub = self.create_publisher(Marker,      '/drone/state_text',  10)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id

        self.drone_enu = (0.0, 0.0, 0.0)
        self.mission_state = 'INIT'

        self.create_subscription(
            Monitoring,
            f'drone{system_id}/fmu/out/monitoring',
            self.monitoring_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, '/mission_state',
            self.mission_state_callback, 10,
        )

        self.get_logger().info(f'DroneVisualizerReal started: frame={self.frame_id}')

    def monitoring_callback(self, msg: Monitoring):
        now = self.get_clock().now().to_msg()
        x_enu, y_enu, z_enu = ned_to_enu(msg.pos_x, msg.pos_y, msg.pos_z)
        self.drone_enu = (x_enu, y_enu, z_enu)

        quat = ned_yaw_to_enu_quat(msg.head)

        tf_msg = TransformStamped()
        tf_msg.header.stamp    = now
        tf_msg.header.frame_id = self.frame_id
        tf_msg.child_frame_id  = self.drone_frame
        tf_msg.transform.translation.x = float(x_enu)
        tf_msg.transform.translation.y = float(y_enu)
        tf_msg.transform.translation.z = float(z_enu)
        tf_msg.transform.rotation = quat
        self.tf_broadcaster.sendTransform(tf_msg)

        pose = PoseStamped()
        pose.header.stamp    = now
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(x_enu)
        pose.pose.position.y = float(y_enu)
        pose.pose.position.z = float(z_enu)
        pose.pose.orientation = quat
        self.pose_pub.publish(pose)

        self.path_msg.header.stamp = now
        self.path_msg.poses.append(pose)
        if len(self.path_msg.poses) > self.max_path_points:
            self.path_msg.poses.pop(0)
        self.path_pub.publish(self.path_msg)

        self._publish_state_marker()

    def mission_state_callback(self, msg: String):
        self.mission_state = msg.data
        self._publish_state_marker()

    def _publish_state_marker(self):
        x_enu, y_enu, z_enu = self.drone_enu
        r, g, b = _STATE_COLOR.get(self.mission_state, (1.0, 1.0, 1.0))

        marker = Marker()
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns              = 'mission_state'
        marker.id              = 0
        marker.type            = Marker.TEXT_VIEW_FACING
        marker.action          = Marker.ADD
        marker.text            = self.mission_state
        marker.pose.position.x    = x_enu
        marker.pose.position.y    = y_enu
        marker.pose.position.z    = z_enu + 1.5
        marker.pose.orientation.w = 1.0
        marker.scale.z = 5.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        marker.lifetime.sec     = 1
        marker.lifetime.nanosec = 0
        self.state_text_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = DroneVisualizerReal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
