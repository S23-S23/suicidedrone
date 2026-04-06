#!/usr/bin/env python3
"""
Drone Manager Node — Thin FSM
FSM: IDLE -> TAKEOFF -> SEARCH -> INTERCEPT -> DONE

All IBVS/PNG/filter logic is in separate nodes.
This node only handles:
  - State machine transitions
  - PX4 offboard control mode & setpoint publishing
  - Collision detection
  - Mission timeout

Subscriptions:
  drone{id}/fmu/out/vehicle_status      — arming/nav state
  drone{id}/fmu/out/monitoring          — position, attitude
  /png/guidance_cmd                     — GuidanceCmd from PNG
  /target_world_pos                     — target position for collision check
  /mission_state                        — (publishes, not subscribes)

Publications:
  drone{id}/fmu/in/offboard_control_mode
  drone{id}/fmu/in/trajectory_setpoint
  drone{id}/fmu/in/vehicle_command
  /mission_state                        — String (state name for other nodes)
"""

import os
import signal
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    Monitoring,
)
from geometry_msgs.msg import Point
from std_msgs.msg import String
from suicide_drone_msgs.msg import GuidanceCmd
from enum import Enum


class State(Enum):
    IDLE = 0
    TAKEOFF = 1
    SEARCH = 2
    INTERCEPT = 3
    DONE = 4


class DroneManager(Node):
    def __init__(self):
        super().__init__('drone_manager')

        # ── Parameters ──
        self.declare_parameter('system_id', 1)
        self.declare_parameter('takeoff_height', 6.0)
        self.declare_parameter('forward_distance_limit', 50.0)
        self.declare_parameter('collision_distance', 2.0)
        self.declare_parameter('mission_timeout', 60.0)
        self.declare_parameter('max_speed', 10.0)

        self.system_id              = self.get_parameter('system_id').value
        self.takeoff_height         = self.get_parameter('takeoff_height').value
        self.forward_distance_limit = self.get_parameter('forward_distance_limit').value
        self.collision_dist         = self.get_parameter('collision_distance').value
        self.mission_timeout        = self.get_parameter('mission_timeout').value
        self.max_speed              = self.get_parameter('max_speed').value

        self.topic_prefix = f"drone{self.system_id}/fmu/"

        self.get_logger().info(f'DroneManager {self.system_id} initializing...')

        # ── State variables ──
        self.state            = State.IDLE
        self.drone_pos        = np.zeros(3)
        self.drone_yaw        = 0.0
        self.nav_state        = 0
        self.arming_state     = 0
        self.last_cmd_time    = 0.0
        self.forward_start_pos = None
        self._mission_start_t = None

        # INTERCEPT inputs (from PNG guidance_cmd)
        self.guidance_cmd     = None
        self.target_world_pos = None

        # ── Publishers ──
        self.ocm_pub = self.create_publisher(
            OffboardControlMode,
            f'{self.topic_prefix}in/offboard_control_mode',
            qos_profile_sensor_data,
        )
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint,
            f'{self.topic_prefix}in/trajectory_setpoint',
            qos_profile_sensor_data,
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            f'{self.topic_prefix}in/vehicle_command',
            qos_profile_sensor_data,
        )
        self.state_pub = self.create_publisher(String, '/mission_state', 10)

        # ── Subscribers ──
        self.create_subscription(
            VehicleStatus,
            f'{self.topic_prefix}out/vehicle_status',
            self.status_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Monitoring,
            f'{self.topic_prefix}out/monitoring',
            self.monitoring_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            GuidanceCmd,
            '/png/guidance_cmd',
            self.guidance_cmd_cb,
            10,
        )
        self.create_subscription(
            Point, '/target_world_pos',
            self.target_pos_cb, 10
        )

        # ── Timers ──
        self.create_timer(0.1,  self.ocm_cb)          # 10 Hz offboard heartbeat
        self.create_timer(0.02, self.control_cb)       # 50 Hz main control
        self.create_timer(5.0,  self.start_mission)    # one-shot

        self.get_logger().info('DroneManager started: IDLE -> TAKEOFF -> SEARCH -> INTERCEPT -> DONE')

    # ── Callbacks ──
    def status_cb(self, msg: VehicleStatus):
        self.nav_state    = msg.nav_state
        self.arming_state = msg.arming_state

    def monitoring_cb(self, msg: Monitoring):
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        self.drone_yaw = msg.head

    def guidance_cmd_cb(self, msg: GuidanceCmd):
        self.guidance_cmd = msg

        # SEARCH -> INTERCEPT when target detected
        if msg.target_detected and self.state == State.SEARCH:
            self.get_logger().info('Target detected! SEARCH -> INTERCEPT')
            self._mission_start_t = time.time()
            self.state = State.INTERCEPT

        # INTERCEPT -> SEARCH when target lost
        if not msg.target_detected and self.state == State.INTERCEPT:
            self.get_logger().warn('Target lost! INTERCEPT -> SEARCH')
            self.state = State.SEARCH

    def target_pos_cb(self, msg: Point):
        self.target_world_pos = np.array([msg.x, msg.y, msg.z])

    def start_mission(self):
        if self.state == State.IDLE:
            self.get_logger().info('Mission start -> TAKEOFF')
            self.state = State.TAKEOFF

    # ── Offboard heartbeat (10Hz) ──
    def ocm_cb(self):
        msg = OffboardControlMode()
        if self.state == State.INTERCEPT:
            msg.position = False
            msg.velocity = True
        else:
            msg.position = True
            msg.velocity = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.ocm_pub.publish(msg)

    # ── Main control loop (50Hz) ──
    def control_cb(self):
        # Publish state for other nodes
        state_msg = String()
        state_msg.data = self.state.name
        self.state_pub.publish(state_msg)

        if self.state == State.IDLE:
            self._idle()
        elif self.state == State.TAKEOFF:
            self._takeoff()
        elif self.state == State.SEARCH:
            self._search()
        elif self.state == State.INTERCEPT:
            self._intercept()
        elif self.state == State.DONE:
            self._done()

    # ── State handlers ──
    def _idle(self):
        safe_z = max(self.drone_pos[2], -0.1)
        self._pub_pos([self.drone_pos[0], self.drone_pos[1], safe_z])

    def _takeoff(self):
        alt = -self.takeoff_height
        now = self.get_clock().now().nanoseconds / 1e9

        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED or \
           self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            safe_z = max(self.drone_pos[2], -0.1)
            self._pub_pos([self.drone_pos[0], self.drone_pos[1], safe_z])
        else:
            self._pub_pos([self.drone_pos[0], self.drone_pos[1], alt])

        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if now - self.last_cmd_time > 1.0:
                self._pub_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.get_logger().info('ARM requested')
                self.last_cmd_time = now
            return

        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if now - self.last_cmd_time > 1.0:
                self._pub_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.get_logger().info('OFFBOARD requested')
                self.last_cmd_time = now
            return

        if abs(self.drone_pos[2] - alt) < 0.3:
            self.get_logger().info('Takeoff complete -> SEARCH')
            self.forward_start_pos = self.drone_pos.copy()
            self.state = State.SEARCH

    def _search(self):
        """Fly forward until target is detected (via guidance_cmd_cb)."""
        if self.forward_start_pos is None:
            self.forward_start_pos = self.drone_pos.copy()

        distance_traveled = np.linalg.norm(
            self.drone_pos[:2] - self.forward_start_pos[:2]
        )

        if distance_traveled >= self.forward_distance_limit:
            self._pub_pos([self.drone_pos[0], self.drone_pos[1], -self.takeoff_height])
            self.get_logger().info(
                f'Forward limit reached ({distance_traveled:.1f}m). Hovering.',
                throttle_duration_sec=2.0,
            )
            return

        # Fly forward (positive X in NED)
        self._pub_pos([5.0, 0.0, -self.takeoff_height])

    def _intercept(self):
        """Velocity control using PNG GuidanceCmd."""
        # ── Mission timeout ──
        if self._mission_start_t and \
           (time.time() - self._mission_start_t) >= self.mission_timeout:
            self.get_logger().info('Mission timeout -> DONE')
            self._finish()
            return

        # ── Collision detection ──
        if self.target_world_pos is not None:
            target_pos_ned = np.array([
                self.target_world_pos[1],   # North = Gazebo Y
                self.target_world_pos[0],   # East  = Gazebo X
                -self.target_world_pos[2]   # Down  = -Gazebo Z
            ])
            dist = np.linalg.norm(self.drone_pos - target_pos_ned)
            if dist < self.collision_dist:
                self.get_logger().info(f'COLLISION at dist={dist:.2f}m -> DONE')
                self._finish()
                return

        # ── Apply guidance command ──
        if self.guidance_cmd is None:
            self._pub_vel(np.zeros(3), self.drone_yaw)
            return

        cmd = self.guidance_cmd
        vel = np.array([cmd.vel_n, cmd.vel_e, cmd.vel_d])
        self._pub_vel(vel, yaw_rate=cmd.yaw_rate)

        self.get_logger().info(
            f'INTERCEPT: v=({cmd.vel_n:.2f},{cmd.vel_e:.2f},{cmd.vel_d:.2f}) '
            f'yr={cmd.yaw_rate:.3f}',
            throttle_duration_sec=1.0,
        )

    def _done(self):
        self._pub_pos(self.drone_pos.tolist(), yaw=self.drone_yaw)
        self.get_logger().info('Mission DONE, hovering.', throttle_duration_sec=5.0)

    def _finish(self):
        self.state = State.DONE
        self.get_logger().info('Finishing -- shutting down in 3s')
        self.create_timer(3.0, lambda: os.kill(os.getpid(), signal.SIGINT))

    # ── PX4 command helpers ──
    def _pub_pos(self, pos, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pub.publish(msg)

    def _pub_vel(self, vel, yaw=None, yaw_rate=None):
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [float(vel[0]), float(vel[1]), float(vel[2])]
        if yaw_rate is not None:
            msg.yaw = float('nan')
            msg.yawspeed = float(yaw_rate)
        else:
            msg.yaw = float(yaw) if yaw is not None else 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pub.publish(msg)

    def _pub_cmd(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.param1, msg.param2, msg.command = p1, p2, cmd
        msg.target_system, msg.target_component = self.system_id, 1
        msg.source_system, msg.source_component, msg.from_external = 1, 1, True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DroneManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
