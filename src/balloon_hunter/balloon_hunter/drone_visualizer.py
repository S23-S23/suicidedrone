#!/usr/bin/env python3
"""
Drone Visualizer Node
Publishes to RViz2:
  - TF:                         map -> drone{id}          (from /gazebo/model_states GT, 50 Hz timer)
  - geometry_msgs/PoseStamped:  current drone pose        (from /gazebo/model_states GT, 50 Hz timer)
  - Marker LINE_STRIP (green):  Gazebo GT trajectory      (from /gazebo/model_states)
  - Marker LINE_STRIP (yellow): PX4 estimated trajectory  (from VehicleLocalPosition)
  - Marker SPHERE:              balloon target             (from /gazebo/model_states)
  - Marker TEXT_VIEW_FACING:    mission state label        (from /mission_state)
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import VehicleLocalPosition
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, Quaternion
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster

# Mission state text color (R, G, B)  – matches MissionState enum in drone_manager
_STATE_COLOR = {
    'IDLE':      (0.6, 0.6, 0.6),
    'TAKEOFF':   (1.0, 1.0, 0.0),
    'FORWARD':   (0.0, 0.8, 1.0),
    'INTERCEPT': (1.0, 0.3, 0.0),  # orange-red: IBVS+PNG active
    'DONE':      (1.0, 1.0, 1.0),
}

# Trajectory colors
_COLOR_GT   = (0.0, 1.0, 0.0, 1.0)   # green  – Gazebo ground truth
_COLOR_PX4  = (1.0, 1.0, 0.0, 0.8)   # yellow – PX4 VehicleLocalPosition estimate


def ned_to_enu(x_ned, y_ned, z_ned):
    """Convert NED (North-East-Down) to ENU (East-North-Up)."""
    return y_ned, x_ned, -z_ned


def gazebo_quat_to_enu_quat(gx: float, gy: float, gz: float, gw: float) -> Quaternion:
    """
    Extract ENU yaw from Gazebo orientation (ENU world frame quaternion)
    and return a pure-yaw ENU quaternion.
    yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    """
    yaw = math.atan2(2.0 * (gw * gz + gx * gy),
                     1.0 - 2.0 * (gy * gy + gz * gz))
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class DroneVisualizer(Node):
    def __init__(self):
        super().__init__('drone_visualizer')

        self.declare_parameter('system_id', 1)
        self.declare_parameter('max_path_points', 5000)
        self.declare_parameter('balloon_radius', 0.3)
        self.declare_parameter('balloon_model_name', 'target_balloon')
        self.declare_parameter('balloon_link_z_offset', 1.5)

        system_id                  = self.get_parameter('system_id').value
        self.max_path_points       = self.get_parameter('max_path_points').value
        self.balloon_radius        = self.get_parameter('balloon_radius').value
        self.balloon_model_name    = self.get_parameter('balloon_model_name').value
        self.balloon_link_z_offset = self.get_parameter('balloon_link_z_offset').value
        self.frame_id              = 'map'
        self.drone_frame           = f'drone{system_id}'
        self.drone_model_name      = f'drone{system_id}'
        local_pos_topic            = f'drone{system_id}/fmu/out/vehicle_local_position'

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publishers
        self.pose_pub       = self.create_publisher(PoseStamped, '/drone/pose',          10)
        self.gt_traj_pub    = self.create_publisher(Marker,      '/drone/gt_trajectory', 10)
        self.px4_traj_pub   = self.create_publisher(Marker,      '/drone/px4_trajectory',10)
        self.balloon_pub    = self.create_publisher(Marker,      '/balloon/marker',      10)
        self.state_text_pub = self.create_publisher(Marker,      '/drone/state_text',    10)

        # Accumulated trajectory point lists (ENU)
        self.gt_points:  list[Point] = []   # Gazebo ground truth
        self.px4_points: list[Point] = []   # PX4 VehicleLocalPosition estimate

        # Latest GT pose from Gazebo model_states (ENU)
        self.drone_gt_enu  = (0.0, 0.0, 0.0)
        self.drone_gt_quat: Quaternion = Quaternion()
        self.drone_gt_quat.w = 1.0
        self.drone_gt_valid = False

        # Current mission state name
        self.mission_state = 'IDLE'

        # Subscriptions
        self.create_subscription(
            VehicleLocalPosition,
            local_pos_topic,
            self.local_position_callback,
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

        # 50 Hz timer: republish TF + PoseStamped from latest GT pose
        self.create_timer(1.0 / 50.0, self.pose_timer_callback)

        self.get_logger().info(
            f'DroneVisualizer started | GT=green(/drone/gt_trajectory) '
            f'PX4=yellow(/drone/px4_trajectory) frame={self.frame_id}'
        )

    # ----------------------------------------------------------------------- #
    #  50 Hz timer – TF + PoseStamped from latest Gazebo GT pose              #
    # ----------------------------------------------------------------------- #
    def pose_timer_callback(self):
        if not self.drone_gt_valid:
            return

        now = self.get_clock().now().to_msg()
        x_enu, y_enu, z_enu = self.drone_gt_enu
        quat = self.drone_gt_quat

        # TF: map -> drone{id}
        tf_msg = TransformStamped()
        tf_msg.header.stamp               = now
        tf_msg.header.frame_id            = self.frame_id
        tf_msg.child_frame_id             = self.drone_frame
        tf_msg.transform.translation.x    = float(x_enu)
        tf_msg.transform.translation.y    = float(y_enu)
        tf_msg.transform.translation.z    = float(z_enu)
        tf_msg.transform.rotation         = quat
        self.tf_broadcaster.sendTransform(tf_msg)

        # PoseStamped
        pose = PoseStamped()
        pose.header.stamp       = now
        pose.header.frame_id    = self.frame_id
        pose.pose.position.x    = float(x_enu)
        pose.pose.position.y    = float(y_enu)
        pose.pose.position.z    = float(z_enu)
        pose.pose.orientation   = quat
        self.pose_pub.publish(pose)

    # ----------------------------------------------------------------------- #
    #  PX4 VehicleLocalPosition – estimated trajectory (yellow)                #
    # ----------------------------------------------------------------------- #
    def local_position_callback(self, msg: VehicleLocalPosition):
        """Accumulate PX4 EKF2 estimated position and publish yellow trajectory."""
        # VehicleLocalPosition x/y/z is NED [m]
        x_enu, y_enu, z_enu = ned_to_enu(float(msg.x), float(msg.y), float(msg.z))

        pt = Point()
        pt.x, pt.y, pt.z = x_enu, y_enu, z_enu
        self.px4_points.append(pt)
        if len(self.px4_points) > self.max_path_points:
            self.px4_points.pop(0)

        self._publish_line_strip(
            self.px4_traj_pub,
            self.px4_points,
            ns='px4_trajectory',
            marker_id=1,
            color=_COLOR_PX4,
            line_width=0.08,
        )

    # ----------------------------------------------------------------------- #
    #  Gazebo model_states – GT pose storage + trajectory (green) + balloon   #
    # ----------------------------------------------------------------------- #
    def model_states_callback(self, msg: ModelStates):
        now = self.get_clock().now().to_msg()

        # ── Drone ground-truth trajectory (green) ──────────────────────────
        if self.drone_model_name in msg.name:
            idx  = msg.name.index(self.drone_model_name)
            p    = msg.pose[idx].position
            ori  = msg.pose[idx].orientation

            # Gazebo frame is ENU = map frame directly
            x_enu, y_enu, z_enu = float(p.x), float(p.y), float(p.z)

            # Store latest GT pose for 50 Hz timer
            self.drone_gt_enu   = (x_enu, y_enu, z_enu)
            self.drone_gt_quat  = gazebo_quat_to_enu_quat(
                float(ori.x), float(ori.y), float(ori.z), float(ori.w)
            )
            self.drone_gt_valid = True

            pt = Point()
            pt.x, pt.y, pt.z = x_enu, y_enu, z_enu
            self.gt_points.append(pt)
            if len(self.gt_points) > self.max_path_points:
                self.gt_points.pop(0)

            self._publish_line_strip(
                self.gt_traj_pub,
                self.gt_points,
                ns='gt_trajectory',
                marker_id=0,
                color=_COLOR_GT,
                line_width=0.12,
            )

        # ── Balloon sphere marker (red) ─────────────────────────────────────
        if self.balloon_model_name not in msg.name:
            return

        idx = msg.name.index(self.balloon_model_name)
        p   = msg.pose[idx].position
        x_enu = float(p.x)
        y_enu = float(p.y)
        z_enu = float(p.z) + self.balloon_link_z_offset

        balloon = Marker()
        balloon.header.stamp        = now
        balloon.header.frame_id     = self.frame_id
        balloon.ns                  = 'balloon'
        balloon.id                  = 0
        balloon.type                = Marker.SPHERE
        balloon.action              = Marker.ADD
        balloon.pose.position.x     = x_enu
        balloon.pose.position.y     = y_enu
        balloon.pose.position.z     = z_enu
        balloon.pose.orientation.w  = 1.0
        d = self.balloon_radius * 2.0
        balloon.scale.x = balloon.scale.y = balloon.scale.z = d
        balloon.color.r = 1.0
        balloon.color.g = 0.0
        balloon.color.b = 0.0
        balloon.color.a = 0.8
        balloon.lifetime.sec = balloon.lifetime.nanosec = 0
        self.balloon_pub.publish(balloon)

    # ----------------------------------------------------------------------- #
    #  Mission state label                                                      #
    # ----------------------------------------------------------------------- #
    def mission_state_callback(self, msg: String):
        self.mission_state = msg.data
        self._publish_state_marker()

    def _publish_state_marker(self):
        x_enu, y_enu, z_enu = self.drone_gt_enu
        r, g, b = _STATE_COLOR.get(self.mission_state, (1.0, 1.0, 1.0))

        marker = Marker()
        marker.header.stamp       = self.get_clock().now().to_msg()
        marker.header.frame_id    = self.frame_id
        marker.ns                 = 'mission_state'
        marker.id                 = 0
        marker.type               = Marker.TEXT_VIEW_FACING
        marker.action             = Marker.ADD
        marker.text               = self.mission_state
        marker.pose.position.x    = x_enu
        marker.pose.position.y    = y_enu
        marker.pose.position.z    = z_enu + 1.5
        marker.pose.orientation.w = 1.0
        marker.scale.z            = 0.8
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        marker.lifetime.sec     = 1
        marker.lifetime.nanosec = 0
        self.state_text_pub.publish(marker)

    # ----------------------------------------------------------------------- #
    #  Helper: publish accumulated points as a LINE_STRIP marker               #
    # ----------------------------------------------------------------------- #
    def _publish_line_strip(self, publisher, points: list, ns: str,
                            marker_id: int, color: tuple, line_width: float):
        if len(points) < 2:
            return
        r, g, b, a = color
        marker = Marker()
        marker.header.stamp       = self.get_clock().now().to_msg()
        marker.header.frame_id    = self.frame_id
        marker.ns                 = ns
        marker.id                 = marker_id
        marker.type               = Marker.LINE_STRIP
        marker.action             = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x            = line_width   # line width [m]
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        marker.points             = list(points)
        marker.lifetime.sec = marker.lifetime.nanosec = 0
        publisher.publish(marker)


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
