#!/usr/bin/env python3
"""
Drone Visualizer Node
Converts PX4 NED position to ROS map frame (ENU) and publishes:
  - TF:                    map -> drone{id}
  - nav_msgs/Path:         drone trajectory
  - geometry_msgs/PoseStamped: current drone pose
  - visualization_msgs/Marker: balloon target as a red sphere
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import Monitoring
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster

# Mission state text color (R, G, B)  – matches MissionState enum in drone_manager
_STATE_COLOR = {
    'IDLE':     (0.6, 0.6, 0.6),
    'TAKEOFF':  (1.0, 1.0, 0.0),
    'FORWARD':  (0.0, 0.8, 1.0),
    'TRACKING': (0.0, 1.0, 0.0),
    'CHARGING': (1.0, 0.5, 0.0),
    'DONE':     (1.0, 1.0, 1.0),
}


def ned_to_enu(x_ned, y_ned, z_ned):
    """Convert NED (North-East-Down) to ENU (East-North-Up)."""
    return y_ned, x_ned, -z_ned


def ned_yaw_to_enu_quat(ned_yaw_rad: float) -> Quaternion:
    """
    Convert NED heading (rad) to ENU quaternion (rotation around Z).
    NED: 0 = North, clockwise positive
    ENU: 0 = East,  counter-clockwise positive
    enu_yaw = pi/2 - ned_yaw
    """
    enu_yaw = math.pi / 2.0 - ned_yaw_rad
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(enu_yaw / 2.0)
    q.w = math.cos(enu_yaw / 2.0)
    return q


class DroneVisualizer(Node):
    def __init__(self):
        super().__init__('drone_visualizer')

        self.declare_parameter('system_id', 1)
        self.declare_parameter('max_path_points', 5000)
        # Balloon sphere radius (m) – must match model.sdf
        self.declare_parameter('balloon_radius', 0.3)
        # Balloon model name in Gazebo
        self.declare_parameter('balloon_model_name', 'target_balloon')
        # Z offset of balloon_link inside the model (model.sdf: <pose>0 0 1.5 0 0 0</pose>)
        self.declare_parameter('balloon_link_z_offset', 1.5)

        system_id                = self.get_parameter('system_id').value
        self.max_path_points     = self.get_parameter('max_path_points').value
        self.balloon_radius      = self.get_parameter('balloon_radius').value
        self.balloon_model_name  = self.get_parameter('balloon_model_name').value
        self.balloon_link_z_offset = self.get_parameter('balloon_link_z_offset').value
        self.frame_id            = 'map'
        self.drone_frame         = f'drone{system_id}'
        monitoring_topic         = f'drone{system_id}/fmu/out/monitoring'

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publishers
        self.path_pub       = self.create_publisher(Path,        '/drone/trajectory',    10)
        self.pose_pub       = self.create_publisher(PoseStamped, '/drone/pose',          10)
        self.balloon_pub    = self.create_publisher(Marker,      '/balloon/marker',      10)
        self.state_text_pub = self.create_publisher(Marker,      '/drone/state_text',    10)

        # Accumulated path
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id

        # Current drone ENU position (updated in monitoring_callback)
        self.drone_enu = (0.0, 0.0, 0.0)
        # Current mission state name
        self.mission_state = 'IDLE'

        # Subscriptions
        self.create_subscription(
            Monitoring,
            monitoring_topic,
            self.monitoring_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_states_callback,
            10,
        )
        self.create_subscription(
            String,
            '/mission_state',
            self.mission_state_callback,
            10,
        )

        self.get_logger().info(
            f'DroneVisualizer started: monitoring={monitoring_topic}, frame={self.frame_id}'
        )

    # ----------------------------------------------------------------------- #
    #  Drone monitoring callback                                                #
    # ----------------------------------------------------------------------- #
    def monitoring_callback(self, msg: Monitoring):
        now = self.get_clock().now().to_msg()

        # NED -> ENU
        x_enu, y_enu, z_enu = ned_to_enu(msg.pos_x, msg.pos_y, msg.pos_z)
        self.drone_enu = (x_enu, y_enu, z_enu)

        # NOTE: msg.head is in radians (confirmed by drone_manager.py convention)
        quat = ned_yaw_to_enu_quat(msg.head)

        # TF: map -> drone{id}
        tf_msg = TransformStamped()
        tf_msg.header.stamp      = now
        tf_msg.header.frame_id   = self.frame_id
        tf_msg.child_frame_id    = self.drone_frame
        tf_msg.transform.translation.x = float(x_enu)
        tf_msg.transform.translation.y = float(y_enu)
        tf_msg.transform.translation.z = float(z_enu)
        tf_msg.transform.rotation = quat
        self.tf_broadcaster.sendTransform(tf_msg)

        # Current pose
        pose = PoseStamped()
        pose.header.stamp    = now
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(x_enu)
        pose.pose.position.y = float(y_enu)
        pose.pose.position.z = float(z_enu)
        pose.pose.orientation = quat
        self.pose_pub.publish(pose)

        # Trajectory path (accumulated)
        self.path_msg.header.stamp = now
        self.path_msg.poses.append(pose)
        if len(self.path_msg.poses) > self.max_path_points:
            self.path_msg.poses.pop(0)
        self.path_pub.publish(self.path_msg)

    # ----------------------------------------------------------------------- #
    #  Mission state callback                                                   #
    # ----------------------------------------------------------------------- #
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

        # Display 1.5 m above the drone
        marker.pose.position.x    = x_enu
        marker.pose.position.y    = y_enu
        marker.pose.position.z    = z_enu + 1.5
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.8   # text height in metres

        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        marker.lifetime.sec     = 1   # auto-clear after 1 s if no update
        marker.lifetime.nanosec = 0

        self.state_text_pub.publish(marker)

    # ----------------------------------------------------------------------- #
    #  Balloon ground-truth callback                                            #
    # ----------------------------------------------------------------------- #
    def model_states_callback(self, msg: ModelStates):
        """
        Read the balloon's true position from Gazebo model_states.
        Gazebo ENU -> RViz2 map (ENU): direct 1-to-1 mapping.
        Apply balloon_link Z offset from model.sdf.
        """
        if self.balloon_model_name not in msg.name:
            return

        idx = msg.name.index(self.balloon_model_name)
        p   = msg.pose[idx].position

        # Gazebo frame is already ENU (X=East, Y=North, Z=Up) = map frame
        # Just apply the internal link Z offset so we point at the sphere center
        x_enu = float(p.x)
        y_enu = float(p.y)
        z_enu = float(p.z) + self.balloon_link_z_offset

        marker = Marker()
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns              = 'balloon'
        marker.id              = 0
        marker.type            = Marker.SPHERE
        marker.action          = Marker.ADD

        marker.pose.position.x    = x_enu
        marker.pose.position.y    = y_enu
        marker.pose.position.z    = z_enu
        marker.pose.orientation.w = 1.0

        d = self.balloon_radius * 2.0
        marker.scale.x = d
        marker.scale.y = d
        marker.scale.z = d

        # Red, semi-transparent
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        marker.lifetime.sec     = 0  # persist until replaced
        marker.lifetime.nanosec = 0

        self.balloon_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = DroneVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
