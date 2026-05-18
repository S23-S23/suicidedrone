#!/usr/bin/env python3
"""
Swarm Visualizer Node
Single node that handles all drones in the swarm.

Subscriptions (per drone N):
  /drone{N}/fmu/out/monitoring  → TF + PX4 estimated trajectory
  /drone{N}/mission_state       → state text marker
  /gazebo/model_states          → Gazebo GT trajectories + balloon sphere

Published (per drone N):
  /drone{N}/gt_trajectory   Marker LINE_STRIP  (Gazebo ground truth)
  /drone{N}/px4_trajectory  Marker LINE_STRIP  (PX4 Monitoring estimate)
  /drone{N}/state_text      Marker TEXT_VIEW_FACING

Published (shared):
  /balloon/marker           Marker SPHERE

Color scheme per drone:
  drone1: GT=green    / PX4=yellow
  drone2: GT=cyan     / PX4=orange
  drone3: GT=magenta  / PX4=purple
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import Monitoring
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, TransformStamped, Quaternion
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster

# (GT_rgba, PX4_rgba) per system_id
_DRONE_COLORS = {
    1: ((0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 0.8)),   # green  / yellow
    2: ((0.0, 0.8, 1.0, 1.0), (1.0, 0.5, 0.0, 0.8)),   # cyan   / orange
    3: ((1.0, 0.0, 1.0, 1.0), (0.6, 0.0, 1.0, 0.8)),   # magenta / purple
}

_STATE_COLOR = {
    'IDLE':      (0.6, 0.6, 0.6),
    'TAKEOFF':   (1.0, 1.0, 0.0),
    'SEARCH':    (0.0, 0.8, 1.0),
    'INTERCEPT': (0.0, 1.0, 0.0),
    'DONE':      (1.0, 1.0, 1.0),
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


class _DroneState:
    """Per-drone mutable state held by SwarmVisualizer."""

    def __init__(self, system_id: int, max_points: int, gt_pub, px4_pub, state_pub):
        self.system_id   = system_id
        self.max_points  = max_points
        self.gt_pub      = gt_pub
        self.px4_pub     = px4_pub
        self.state_pub   = state_pub

        self.gt_points:  list[Point] = []
        self.px4_points: list[Point] = []
        self.enu         = (0.0, 0.0, 0.0)
        self.mission_state = 'IDLE'

        colors = _DRONE_COLORS.get(system_id, _DRONE_COLORS[1])
        self.gt_color  = colors[0]
        self.px4_color = colors[1]
        self.model_name    = f'drone{system_id}'
        self.drone_frame   = f'drone{system_id}'


class SwarmVisualizer(Node):
    def __init__(self):
        super().__init__('swarm_visualizer')

        self.declare_parameter('system_ids',            [1, 2, 3])
        self.declare_parameter('max_path_points',       5000)
        self.declare_parameter('balloon_radius',        0.3)
        self.declare_parameter('balloon_model_name',    'target_balloon')
        self.declare_parameter('balloon_link_z_offset', 1.5)

        system_ids              = self.get_parameter('system_ids').value
        max_points              = self.get_parameter('max_path_points').value
        self.balloon_radius     = self.get_parameter('balloon_radius').value
        self.balloon_model_name = self.get_parameter('balloon_model_name').value
        self.balloon_z_offset   = self.get_parameter('balloon_link_z_offset').value

        self.frame_id      = 'map'
        self.tf_broadcaster = TransformBroadcaster(self)

        self.balloon_pub = self.create_publisher(Marker, '/balloon/marker', 10)

        self._drones: dict[int, _DroneState] = {}

        for sid in system_ids:
            ns      = f'drone{sid}'
            gt_pub  = self.create_publisher(Marker, f'/{ns}/gt_trajectory',  10)
            px4_pub = self.create_publisher(Marker, f'/{ns}/px4_trajectory', 10)
            st_pub  = self.create_publisher(Marker, f'/{ns}/state_text',     10)

            state = _DroneState(sid, max_points, gt_pub, px4_pub, st_pub)
            self._drones[sid] = state

            self.create_subscription(
                Monitoring,
                f'/{ns}/fmu/out/monitoring',
                lambda msg, s=state: self._monitoring_cb(msg, s),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                String,
                f'/{ns}/mission_state',
                lambda msg, s=state: self._mission_state_cb(msg, s),
                10,
            )

        self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self._model_states_cb,
            10,
        )

        gt_colors  = ', '.join(f'D{s}={"green/cyan/magenta".split("/")[i]}' for i, s in enumerate(system_ids))
        self.get_logger().info(
            f'SwarmVisualizer started | drones={list(system_ids)} | '
            f'GT: green/cyan/magenta  PX4: yellow/orange/purple'
        )

    # ------------------------------------------------------------------ #
    #  PX4 Monitoring → TF + PX4 trajectory                               #
    # ------------------------------------------------------------------ #
    def _monitoring_cb(self, msg: Monitoring, state: _DroneState):
        now = self.get_clock().now().to_msg()
        x_enu, y_enu, z_enu = ned_to_enu(msg.pos_x, msg.pos_y, msg.pos_z)
        state.enu = (x_enu, y_enu, z_enu)
        quat = ned_yaw_to_enu_quat(msg.head)

        # TF: map -> drone{N}
        tf_msg = TransformStamped()
        tf_msg.header.stamp             = now
        tf_msg.header.frame_id          = self.frame_id
        tf_msg.child_frame_id           = state.drone_frame
        tf_msg.transform.translation.x  = float(x_enu)
        tf_msg.transform.translation.y  = float(y_enu)
        tf_msg.transform.translation.z  = float(z_enu)
        tf_msg.transform.rotation       = quat
        self.tf_broadcaster.sendTransform(tf_msg)

        # PX4 trajectory
        pt = Point()
        pt.x, pt.y, pt.z = float(x_enu), float(y_enu), float(z_enu)
        state.px4_points.append(pt)
        if len(state.px4_points) > state.max_points:
            state.px4_points.pop(0)

        self._publish_line_strip(
            state.px4_pub, state.px4_points,
            ns=f'px4_{state.system_id}',
            marker_id=state.system_id,
            color=state.px4_color,
            line_width=0.12,
        )
        self._publish_state_marker(state)

    # ------------------------------------------------------------------ #
    #  Mission state                                                        #
    # ------------------------------------------------------------------ #
    def _mission_state_cb(self, msg: String, state: _DroneState):
        state.mission_state = msg.data
        self._publish_state_marker(state)

    def _publish_state_marker(self, state: _DroneState):
        x, y, z = state.enu
        r, g, b = _STATE_COLOR.get(state.mission_state, (1.0, 1.0, 1.0))

        marker = Marker()
        marker.header.stamp       = self.get_clock().now().to_msg()
        marker.header.frame_id    = self.frame_id
        marker.ns                 = f'state_{state.system_id}'
        marker.id                 = state.system_id
        marker.type               = Marker.TEXT_VIEW_FACING
        marker.action             = Marker.ADD
        marker.text               = f'D{state.system_id}: {state.mission_state}'
        marker.pose.position.x    = x
        marker.pose.position.y    = y
        marker.pose.position.z    = z + 1.5
        marker.pose.orientation.w = 1.0
        marker.scale.z            = 3.0
        marker.color.r            = r
        marker.color.g            = g
        marker.color.b            = b
        marker.color.a            = 1.0
        marker.lifetime.sec       = 1
        marker.lifetime.nanosec   = 0

        state.state_pub.publish(marker)

    # ------------------------------------------------------------------ #
    #  Gazebo model_states → GT trajectories + balloon                     #
    # ------------------------------------------------------------------ #
    def _model_states_cb(self, msg: ModelStates):
        now = self.get_clock().now().to_msg()

        for sid, state in self._drones.items():
            if state.model_name not in msg.name:
                continue
            idx = msg.name.index(state.model_name)
            p   = msg.pose[idx].position
            pt  = Point()
            pt.x, pt.y, pt.z = float(p.x), float(p.y), float(p.z)
            state.gt_points.append(pt)
            if len(state.gt_points) > state.max_points:
                state.gt_points.pop(0)
            self._publish_line_strip(
                state.gt_pub, state.gt_points,
                ns=f'gt_{sid}',
                marker_id=sid,
                color=state.gt_color,
                line_width=0.12,
            )

        # Balloon sphere
        if self.balloon_model_name not in msg.name:
            return
        idx   = msg.name.index(self.balloon_model_name)
        p     = msg.pose[idx].position
        x_enu = float(p.x)
        y_enu = float(p.y)
        z_enu = float(p.z) + self.balloon_z_offset

        marker = Marker()
        marker.header.stamp       = now
        marker.header.frame_id    = self.frame_id
        marker.ns                 = 'balloon'
        marker.id                 = 0
        marker.type               = Marker.SPHERE
        marker.action             = Marker.ADD
        marker.pose.position.x    = x_enu
        marker.pose.position.y    = y_enu
        marker.pose.position.z    = z_enu
        marker.pose.orientation.w = 1.0
        d = self.balloon_radius * 2.0
        marker.scale.x = d
        marker.scale.y = d
        marker.scale.z = d
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker.lifetime.sec = marker.lifetime.nanosec = 0

        self.balloon_pub.publish(marker)

    # ------------------------------------------------------------------ #
    #  Helper                                                              #
    # ------------------------------------------------------------------ #
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
        marker.scale.x            = line_width
        marker.color.r            = r
        marker.color.g            = g
        marker.color.b            = b
        marker.color.a            = a
        marker.points             = list(points)
        marker.lifetime.sec = marker.lifetime.nanosec = 0
        publisher.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = SwarmVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
