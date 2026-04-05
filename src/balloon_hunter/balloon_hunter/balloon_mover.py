#!/usr/bin/env python3
"""
Balloon Mover Node

Moves the target balloon in Gazebo once the drone enters FORWARD state.
Uses /gazebo/set_entity_state service (provided by libgazebo_ros_state.so).

The balloon model must be non-static + kinematic (model.sdf):
  <static>false</static>
  <link ...>
    <kinematic>true</kinematic>

Collision detection uses ground-truth ENU positions from /gazebo/model_states
for both the balloon and the drone — no NED conversion or position estimation.

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
from std_msgs.msg import String, Bool
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState, ModelStates
from geometry_msgs.msg import Pose, Twist
import numpy as np


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
        self.declare_parameter('update_rate', 20.0)
        self.declare_parameter('initial_x', 3.0)
        self.declare_parameter('initial_y', 15.0)
        self.declare_parameter('initial_z', 2.0)
        self.declare_parameter('random_interval', 3.0)
        self.declare_parameter('collision_distance', 1.6)
        self.declare_parameter('system_id', 1)
        # balloon_link <pose>0 0 1.5 0 0 0</pose> in model.sdf:
        # model_states returns the model root pose; add this for the sphere center.
        self.declare_parameter('balloon_link_z_offset', 1.5)

        self.balloon_name       = self.get_parameter('balloon_model_name').value
        self.pattern            = self.get_parameter('movement_pattern').value
        self.speed              = self.get_parameter('speed').value
        self.update_rate        = self.get_parameter('update_rate').value
        self.random_interval    = self.get_parameter('random_interval').value
        self.collision_distance = self.get_parameter('collision_distance').value
        self.system_id          = self.get_parameter('system_id').value
        self.balloon_z_offset   = self.get_parameter('balloon_link_z_offset').value

        # Balloon model-origin pose (ENU) — updated from /gazebo/model_states.
        # Falls back to initial_x/y/z until first model_states message arrives.
        self.pos_x = self.get_parameter('initial_x').value
        self.pos_y = self.get_parameter('initial_y').value
        self.pos_z = self.get_parameter('initial_z').value

        # GT positions in Gazebo ENU [m] — set by _model_states_cb
        self.balloon_enu: np.ndarray | None = None   # balloon sphere center
        self.drone_enu:   np.ndarray | None = None   # drone model origin

        self.moving         = False
        self.collision_done = False

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
        # Ground-truth positions for both balloon and drone
        self.create_subscription(
            ModelStates, '/gazebo/model_states', self._model_states_cb, 10
        )

        # ── Timer ───────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.update_rate, self._update)

        self.get_logger().info(
            f'BalloonMover started: pattern={self.pattern}, '
            f'speed={self.speed} m/s, collision_distance={self.collision_distance} m'
        )

    # ── Ground-truth callback ──────────────────────────────────────────────

    def _model_states_cb(self, msg: ModelStates):
        """Extract balloon sphere center and drone origin from Gazebo ENU."""
        drone_name = f'drone{self.system_id}'

        if self.balloon_name in msg.name:
            idx = msg.name.index(self.balloon_name)
            p = msg.pose[idx].position
            # model root pose + link z-offset = sphere center
            self.balloon_enu = np.array([
                float(p.x),
                float(p.y),
                float(p.z) + self.balloon_z_offset,
            ])
            # Keep pos_x/y/z in sync with actual Gazebo model-origin position
            # so _apply_velocity anchors to the true position.
            self.pos_x = float(p.x)
            self.pos_y = float(p.y)
            self.pos_z = float(p.z)

        if drone_name in msg.name:
            idx = msg.name.index(drone_name)
            p = msg.pose[idx].position
            self.drone_enu = np.array([float(p.x), float(p.y), float(p.z)])

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

    # ── Send state to Gazebo ───────────────────────────────────────────────

    def _apply_velocity(self, vx, vy, vz):
        """
        Call set_entity_state to set balloon velocity (and anchor position).
        Kinematic body: Gazebo integrates at physics rate until next call.
        pos_x/y/z is kept up-to-date from model_states, so the anchor is accurate.
        """
        self._vel = (vx, vy, vz)

        if not self.set_state_cli.service_is_ready():
            self.get_logger().warn('set_entity_state not ready — retrying next tick')
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

    # ── Collision check (GT ENU) ───────────────────────────────────────────

    def _check_collision(self):
        """
        Compare balloon sphere center and drone origin, both in Gazebo ENU.
        No NED conversion — direct 3-D Euclidean distance in world frame.
        """
        if self.collision_done:
            return
        if self.balloon_enu is None or self.drone_enu is None:
            return

        distance = np.linalg.norm(self.drone_enu - self.balloon_enu)

        self.get_logger().info(
            f'[collision] dist={distance:.2f}m '
            f'drone=({self.drone_enu[0]:.1f},{self.drone_enu[1]:.1f},{self.drone_enu[2]:.1f}) '
            f'balloon=({self.balloon_enu[0]:.1f},{self.balloon_enu[1]:.1f},{self.balloon_enu[2]:.1f})',
            throttle_duration_sec=1.0,
        )

        if distance < self.collision_distance:
            self.get_logger().info(
                f'COLLISION! distance={distance:.3f}m — balloon stopped'
            )
            self.moving         = False
            self.collision_done = True
            self._apply_velocity(0.0, 0.0, 0.0)
            msg      = Bool()
            msg.data = True
            self.collision_pub.publish(msg)

    # ── Timer callback ─────────────────────────────────────────────────────

    def _update(self):
        self._check_collision()

        if not self.moving:
            return

        # Random pattern: check direction change
        if self.pattern == Pattern.RANDOM:
            self._random_timer -= 1.0 / self.update_rate
            if self._random_timer <= 0.0:
                self._pick_random_direction()
                vx, vy, vz = self._compute_velocity()
                self._apply_velocity(vx, vy, vz)

        self.get_logger().info(
            f'Balloon ENU=({self.pos_x:.2f},{self.pos_y:.2f},{self.pos_z:.2f})',
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
