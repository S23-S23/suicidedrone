#!/usr/bin/env python3
"""
Balloon Mover Node

Moves the target balloon in Gazebo once the drone enters FORWARD state.
Uses /gazebo/set_entity_state service (provided by libgazebo_ros_state.so).

The balloon model must be non-static + kinematic (model.sdf):
  <static>false</static>
  <link ...>
    <kinematic>true</kinematic>

With kinematic=true, Gazebo ODE integrates the balloon position at physics
rate (1000 Hz) using the velocity set via set_entity_state. The service is
only called on events (start / direction-change / stop), not every frame.

Movement patterns (movement_pattern parameter):
  left   – West  (Gazebo -X)  : balloon drifts left in the drone's camera view
  right  – East  (Gazebo +X)
  up     – Up    (Gazebo +Z)
  down   – Down  (Gazebo -Z)
  random – random horizontal direction, changes every random_interval seconds
  none   – balloon stays still
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String, Bool
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist
from px4_msgs.msg import Monitoring
import numpy as np


# ── Supported movement pattern names ──────────────────────────────────────────
class Pattern:
    LEFT   = 'left'
    RIGHT  = 'right'
    UP     = 'up'
    DOWN   = 'down'
    RANDOM = 'random'
    NONE   = 'none'


class BalloonMover(Node):
    def __init__(self):
        super().__init__('balloon_mover')

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter('balloon_model_name', 'target_balloon')
        self.declare_parameter('movement_pattern', 'left')
        self.declare_parameter('speed', 0.1)
        # Timer rate for collision check + random direction tracking [Hz]
        self.declare_parameter('update_rate', 20.0)
        self.declare_parameter('initial_x', 3.0)
        self.declare_parameter('initial_y', 15.0)
        self.declare_parameter('initial_z', 2.0)
        self.declare_parameter('random_interval', 3.0)
        self.declare_parameter('collision_distance', 0.6)
        self.declare_parameter('system_id', 1)
        self.declare_parameter('balloon_link_z_offset', 1.5)

        self.balloon_name       = self.get_parameter('balloon_model_name').value
        self.pattern            = self.get_parameter('movement_pattern').value
        self.speed              = self.get_parameter('speed').value
        self.update_rate        = self.get_parameter('update_rate').value
        self.random_interval    = self.get_parameter('random_interval').value
        self.collision_distance = self.get_parameter('collision_distance').value
        system_id               = self.get_parameter('system_id').value
        self.balloon_z_offset   = self.get_parameter('balloon_link_z_offset').value

        # Tracked position in Gazebo ENU [m] (integrated locally for collision)
        self.pos_x = self.get_parameter('initial_x').value
        self.pos_y = self.get_parameter('initial_y').value
        self.pos_z = self.get_parameter('initial_z').value

        self.moving         = False   # starts moving on FORWARD entry
        self.collision_done = False   # latch: collision published once
        self.drone_pos_ned  = None    # NED [m] from Monitoring

        # Current velocity applied to Gazebo [m/s]
        self._vel = (0.0, 0.0, 0.0)

        # Random pattern state
        self._random_dx    = 0.0
        self._random_dy    = 0.0
        self._random_timer = 0.0

        # ── Service client ──────────────────────────────────────────────────
        self.set_state_cli = self.create_client(
            SetEntityState, '/gazebo/set_entity_state'
        )

        # ── Publishers ──────────────────────────────────────────────────────
        self.collision_pub = self.create_publisher(Bool, '/balloon_collision', 10)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(
            String, '/mission_state', self._mission_state_cb, 10
        )
        self.create_subscription(
            Monitoring,
            f'drone{system_id}/fmu/out/monitoring',
            self._monitoring_cb,
            qos_profile_sensor_data,
        )

        # ── Timer (collision check + random direction tracking) ─────────────
        self.create_timer(1.0 / self.update_rate, self._update)

        self.get_logger().info(
            f'BalloonMover started: pattern={self.pattern}, '
            f'speed={self.speed} m/s'
        )

    # ── Drone position callback ────────────────────────────────────────────

    def _monitoring_cb(self, msg: Monitoring):
        self.drone_pos_ned = np.array([msg.pos_x, msg.pos_y, msg.pos_z])

    # ── Mission state callback ─────────────────────────────────────────────

    def _mission_state_cb(self, msg: String):
        if (msg.data == 'FORWARD' or msg.data == 'INTERCEPT') and not self.moving:
            self.get_logger().info(
                f'FORWARD state entered — balloon starts moving ({self.pattern})'
            )
            self.moving = True
            self._start_motion()

    # ── Motion start ───────────────────────────────────────────────────────

    def _start_motion(self):
        """Compute initial velocity and send once to Gazebo."""
        if self.pattern == Pattern.RANDOM:
            self._pick_random_direction()
        vx, vy, vz = self._compute_velocity()
        self._apply_velocity(vx, vy, vz)

    # ── Random direction ───────────────────────────────────────────────────

    def _pick_random_direction(self):
        angle = np.random.uniform(0.0, 2.0 * math.pi)
        self._random_dx    = math.cos(angle)
        self._random_dy    = math.sin(angle)
        self._random_timer = self.random_interval
        self.get_logger().info(
            f'BalloonMover random: new angle={math.degrees(angle):.1f}°'
        )

    # ── Velocity computation ───────────────────────────────────────────────

    def _compute_velocity(self):
        """Return (vx, vy, vz) in Gazebo ENU [m/s] for the current pattern."""
        v = self.speed
        if self.pattern == Pattern.LEFT:
            return (-v, 0.0, 0.0)
        elif self.pattern == Pattern.RIGHT:
            return (v, 0.0, 0.0)
        elif self.pattern == Pattern.UP:
            return (0.0, 0.0, v)
        elif self.pattern == Pattern.DOWN:
            return (0.0, 0.0, -v)
        elif self.pattern == Pattern.RANDOM:
            return (self._random_dx * v, self._random_dy * v, 0.0)
        return (0.0, 0.0, 0.0)

    # ── Send velocity to Gazebo (event-driven, not every frame) ───────────

    def _apply_velocity(self, vx, vy, vz):
        """
        Call set_entity_state with the desired velocity.
        Gazebo kinematic physics will integrate this at 1000 Hz until next call.
        Also anchors the position to our locally-tracked value to prevent drift.
        """
        self._vel = (vx, vy, vz)

        if not self.set_state_cli.service_is_ready():
            self.get_logger().warn('set_entity_state not ready — skipping')
            return

        req = SetEntityState.Request()
        state = EntityState()
        state.name = self.balloon_name
        state.pose = Pose()
        state.pose.position.x = float(self.pos_x)
        state.pose.position.y = float(self.pos_y)
        state.pose.position.z = float(self.pos_z)
        state.pose.orientation.w = 1.0
        state.twist = Twist()
        state.twist.linear.x = float(vx)
        state.twist.linear.y = float(vy)
        state.twist.linear.z = float(vz)
        state.reference_frame = 'world'
        req.state = state

        self.set_state_cli.call_async(req)

    # ── Collision check ────────────────────────────────────────────────────

    def _check_collision(self):
        if self.collision_done or self.drone_pos_ned is None:
            return

        # Balloon ENU → NED: (x=East,y=North,z=Up) → (x=North,y=East,z=Down)
        balloon_ned = np.array([
             self.pos_y,
             self.pos_x,
            -(self.pos_z + self.balloon_z_offset),
        ])

        distance = np.linalg.norm(self.drone_pos_ned - balloon_ned)

        if distance < self.collision_distance:
            self.get_logger().info(
                f'COLLISION! distance={distance:.3f}m — balloon stopped'
            )
            self.moving         = False
            self.collision_done = True
            self._apply_velocity(0.0, 0.0, 0.0)   # stop Gazebo kinematic body
            msg      = Bool()
            msg.data = True
            self.collision_pub.publish(msg)

    # ── Timer callback ─────────────────────────────────────────────────────

    def _update(self):
        # Integrate local position for collision detection
        vx, vy, vz = self._vel
        dt = 1.0 / self.update_rate
        self.pos_x += vx * dt
        self.pos_y += vy * dt
        self.pos_z += vz * dt

        self._check_collision()

        if not self.moving:
            return

        # Random: check if direction should change; send new velocity if so
        if self.pattern == Pattern.RANDOM:
            self._random_timer -= dt
            if self._random_timer <= 0.0:
                self._pick_random_direction()
                nvx, nvy, nvz = self._compute_velocity()
                self._apply_velocity(nvx, nvy, nvz)

        self.get_logger().info(
            f'Balloon pos=({self.pos_x:.2f},{self.pos_y:.2f},{self.pos_z:.2f})',
            throttle_duration_sec=2.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = BalloonMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
