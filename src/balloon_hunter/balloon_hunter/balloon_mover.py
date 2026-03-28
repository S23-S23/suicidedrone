#!/usr/bin/env python3
"""
Balloon Mover Node

Moves the target balloon in Gazebo once the drone enters FORWARD state.
Uses /gazebo/set_entity_state service (provided by libgazebo_ros_state.so).

Movement patterns (movement_pattern parameter):
  left   – West  (Gazebo -X)  : balloon drifts left in the drone's camera view
  right  – East  (Gazebo +X)
  up     – Up    (Gazebo +Z)
  down   – Down  (Gazebo -Z)
  random – random horizontal direction, changes every random_interval seconds

Additional patterns can be added by extending _get_delta().
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist
import numpy as np


# ── Supported movement pattern names ──────────────────────────────────────────
class Pattern:
    LEFT   = 'left'
    RIGHT  = 'right'
    UP     = 'up'
    DOWN   = 'down'
    RANDOM = 'random'


class BalloonMover(Node):
    def __init__(self):
        super().__init__('balloon_mover')

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter('balloon_model_name', 'target_balloon')
        # Movement pattern: left | right | up | down | random
        self.declare_parameter('movement_pattern', 'left')
        # Balloon speed [m/s]
        self.declare_parameter('speed', 0.5)
        # Position update frequency [Hz]
        self.declare_parameter('update_rate', 20.0)
        # Initial balloon position in Gazebo ENU world frame [m]
        # (must match the world-file spawn pose)
        self.declare_parameter('initial_x', 3.0)
        self.declare_parameter('initial_y', 15.0)
        self.declare_parameter('initial_z', 2.0)
        # For 'random' pattern: direction change interval [s]
        self.declare_parameter('random_interval', 3.0)

        self.balloon_name     = self.get_parameter('balloon_model_name').value
        self.pattern          = self.get_parameter('movement_pattern').value
        self.speed            = self.get_parameter('speed').value
        self.update_rate      = self.get_parameter('update_rate').value
        self.random_interval  = self.get_parameter('random_interval').value

        # Current position in Gazebo ENU [m]
        self.pos_x = self.get_parameter('initial_x').value
        self.pos_y = self.get_parameter('initial_y').value
        self.pos_z = self.get_parameter('initial_z').value

        self.moving = False   # starts moving on first FORWARD entry

        # State for random pattern
        self._random_dx    = 0.0
        self._random_dy    = 0.0
        self._random_timer = 0.0   # countdown to next direction change [s]

        # ── Service client ──────────────────────────────────────────────────
        self.set_state_cli = self.create_client(
            SetEntityState, '/gazebo/set_entity_state'
        )

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(
            String, '/mission_state', self._mission_state_cb, 10
        )

        # ── Timer ───────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.update_rate, self._update)

        self.get_logger().info(
            f'BalloonMover started: pattern={self.pattern}, '
            f'speed={self.speed} m/s, rate={self.update_rate} Hz'
        )

    # ── Mission state callback ─────────────────────────────────────────────

    def _mission_state_cb(self, msg: String):
        if (msg.data == 'FORWARD' or msg.data == 'INTERCEPT') and not self.moving:
            self.get_logger().info(
                f'FORWARD state entered — balloon starts moving ({self.pattern})'
            )
            self.moving = True
            self._init_pattern()

    # ── Pattern initialisation ─────────────────────────────────────────────

    def _init_pattern(self):
        """Called once when FORWARD is entered to set up pattern-specific state."""
        if self.pattern == Pattern.RANDOM:
            self._pick_random_direction()

    def _pick_random_direction(self):
        """Choose a new random horizontal direction and reset countdown."""
        angle = np.random.uniform(0.0, 2.0 * math.pi)
        self._random_dx   = math.cos(angle)
        self._random_dy   = math.sin(angle)
        self._random_timer = self.random_interval
        self.get_logger().info(
            f'BalloonMover random: new direction angle={math.degrees(angle):.1f}°'
        )

    # ── Delta computation ──────────────────────────────────────────────────

    def _get_delta(self):
        """
        Return (dx, dy, dz) position increment for one update step
        in Gazebo ENU world frame.

        Drone faces North (+Y in Gazebo ENU) by default (yaw=π/2 at spawn).
        Camera view directions from drone's perspective:
          left  → West  (Gazebo -X)
          right → East  (Gazebo +X)
          up    → Up    (Gazebo +Z)
          down  → Down  (Gazebo -Z)
        """
        step = self.speed / self.update_rate

        if self.pattern == Pattern.LEFT:
            return (-step, 0.0, 0.0)

        elif self.pattern == Pattern.RIGHT:
            return (step, 0.0, 0.0)

        elif self.pattern == Pattern.UP:
            return (0.0, 0.0, step)

        elif self.pattern == Pattern.DOWN:
            return (0.0, 0.0, -step)

        elif self.pattern == Pattern.RANDOM:
            # Tick countdown; pick new direction when it expires
            self._random_timer -= 1.0 / self.update_rate
            if self._random_timer <= 0.0:
                self._pick_random_direction()
            return (
                self._random_dx * step,
                self._random_dy * step,
                0.0,
            )

        return (0.0, 0.0, 0.0)

    # ── Position update timer ──────────────────────────────────────────────

    def _update(self):
        if not self.moving:
            return

        dx, dy, dz = self._get_delta()
        self.pos_x += dx
        self.pos_y += dy
        self.pos_z += dz

        if not self.set_state_cli.service_is_ready():
            return

        req = SetEntityState.Request()
        state = EntityState()
        state.name = self.balloon_name
        state.pose = Pose()
        state.pose.position.x = float(self.pos_x)
        state.pose.position.y = float(self.pos_y)
        state.pose.position.z = float(self.pos_z)
        state.pose.orientation.w = 1.0
        state.twist = Twist()          # zero velocity (static model)
        state.reference_frame = 'world'
        req.state = state

        self.set_state_cli.call_async(req)

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
