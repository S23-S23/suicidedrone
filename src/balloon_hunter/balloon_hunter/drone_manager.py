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
  png/guidance_cmd                      — GuidanceCmd from PNG
  ibvs/output                           — IBVSOutput (q_z for SEARCH yaw)
  /target_world_pos                     — target position for collision check
  /mission_state                        — (publishes, not subscribes)

Publications:
  drone{id}/fmu/in/offboard_control_mode
  drone{id}/fmu/in/trajectory_setpoint
  drone{id}/fmu/in/vehicle_command
  /mission_state                        — String (state name for other nodes)
"""

import math
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
from suicide_drone_msgs.msg import GuidanceCmd, IBVSOutput
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
        # Per-drone NED search waypoint: fly here during SEARCH until target detected.
        # Must differ per drone in swarm to avoid all converging on the same point.
        self.declare_parameter('search_target_n', 5.0)
        self.declare_parameter('search_target_e', 0.0)
        # Proximity gate: drone stays in SEARCH until within this radius of the
        # search waypoint, even if the target is already detected.
        # Prevents flanking drones (2, 3) from intercepting before reaching their
        # pre-planned azimuth position.  Set to 0.0 to disable (drone1 default).
        self.declare_parameter('search_arrival_dist', 1.0)
        # Auto-start delay [s]. Set to a large value in swarm launches to disable
        # the self-start timer and rely on /swarm/start instead.
        self.declare_parameter('start_delay_s', 5.0)
        # Drone spawn position in Gazebo ENU (for target_world_pos → local NED conversion)
        self.declare_parameter('spawn_gazebo_x', 0.0)
        self.declare_parameter('spawn_gazebo_y', 0.0)

        self.system_id              = self.get_parameter('system_id').value
        self.takeoff_height         = self.get_parameter('takeoff_height').value
        self.forward_distance_limit = self.get_parameter('forward_distance_limit').value
        self.collision_dist         = self.get_parameter('collision_distance').value
        self.mission_timeout        = self.get_parameter('mission_timeout').value
        self.max_speed              = self.get_parameter('max_speed').value
        self.search_target_n        = self.get_parameter('search_target_n').value
        self.search_target_e        = self.get_parameter('search_target_e').value
        self.search_arrival_dist    = self.get_parameter('search_arrival_dist').value
        self.start_delay_s          = self.get_parameter('start_delay_s').value
        self.spawn_gazebo_x         = self.get_parameter('spawn_gazebo_x').value
        self.spawn_gazebo_y         = self.get_parameter('spawn_gazebo_y').value

        self.topic_prefix = f"/drone{self.system_id}/fmu/"

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
        self._killed          = False   # True when collision kill has been sent

        # INTERCEPT inputs (from PNG guidance_cmd)
        self.guidance_cmd     = None
        self.target_world_pos = None
        self.ibvs_output      = None

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
        self.state_pub = self.create_publisher(String, 'mission_state', 10)
        self.ready_pub = self.create_publisher(String, '/swarm/ready', 10)
        self._ready_published = False

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
            'png/guidance_cmd',
            self.guidance_cmd_cb,
            10,
        )
        self.create_subscription(
            IBVSOutput,
            'ibvs/output',
            self.ibvs_output_cb,
            10,
        )
        self.create_subscription(
            Point, '/target_world_pos',
            self.target_pos_cb, 10
        )
        self.create_subscription(
            String, '/swarm/start',
            self._swarm_start_cb, 10
        )

        # ── Timers ──
        self.create_timer(0.1,  self.ocm_cb)          # 10 Hz offboard heartbeat
        self.create_timer(0.02, self.control_cb)       # 50 Hz main control
        self.create_timer(self.start_delay_s, self.start_mission)  # auto-start

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

        # SEARCH -> INTERCEPT when target detected AND within proximity gate
        if msg.target_detected and self.state == State.SEARCH:
            search_wp = np.array([self.search_target_n, self.search_target_e,
                                  -self.takeoff_height])
            dist_to_wp = np.linalg.norm(self.drone_pos - search_wp)
            if self.search_arrival_dist <= 0.0 or dist_to_wp < self.search_arrival_dist: #edit
                self.get_logger().info(
                    f'Target detected at wp_dist={dist_to_wp:.1f}m -> INTERCEPT'
                )
                self._mission_start_t = time.time()
                self.state = State.INTERCEPT
            else:
                self.get_logger().info(
                    f'Target detected but not at flanking wp yet '
                    f'({dist_to_wp:.1f}m > {self.search_arrival_dist:.1f}m), holding SEARCH',
                    throttle_duration_sec=2.0,
                )

        # INTERCEPT -> SEARCH when target lost
        if not msg.target_detected and self.state == State.INTERCEPT:
            self.get_logger().warn('Target lost! INTERCEPT -> SEARCH')
            self.state = State.SEARCH

    def ibvs_output_cb(self, msg: IBVSOutput):
        self.ibvs_output = msg

    def target_pos_cb(self, msg: Point):
        self.target_world_pos = np.array([msg.x, msg.y, msg.z])

    def start_mission(self):
        if self.state == State.IDLE:
            self.get_logger().info('Mission start (auto-timer) -> TAKEOFF')
            self.state = State.TAKEOFF

    def _swarm_start_cb(self, msg: String):
        if self.state == State.IDLE:
            self.get_logger().info(f'Swarm start received ({msg.data!r}) -> TAKEOFF')
            self.state = State.TAKEOFF

    # ── Offboard heartbeat (10Hz) ──
    def ocm_cb(self):
        # Stop heartbeat after collision kill — PX4 will failsafe to land/disarm.
        if self._killed:
            return
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
        """Hold ground position and pre-arm so all drones are ready before /swarm/start."""
        now = self.get_clock().now().nanoseconds / 1e9
        safe_z = max(self.drone_pos[2], -0.1)
        self._pub_pos([self.drone_pos[0], self.drone_pos[1], safe_z])

        # Pre-arm during IDLE: by the time /swarm/start fires, every drone is
        # already armed + in offboard mode → simultaneous climb on the signal.
        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if now - self.last_cmd_time > 1.0:
                self._pub_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.get_logger().info('Pre-arm: ARM requested')
                self.last_cmd_time = now
        elif self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if now - self.last_cmd_time > 1.0:
                self._pub_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.get_logger().info('Pre-arm: OFFBOARD requested')
                self.last_cmd_time = now
        else:
            # Armed + Offboard in IDLE → signal coordinator that this drone is ready.
            if not self._ready_published:
                msg = String()
                msg.data = str(self.system_id)
                self.ready_pub.publish(msg)
                self.get_logger().info('Pre-arm complete: published /swarm/ready')
                self._ready_published = True

    def _takeoff(self):
        alt = -self.takeoff_height
        now = self.get_clock().now().nanoseconds / 1e9

        # Safety net: finish arming if pre-arm didn't complete before the signal.
        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            safe_z = max(self.drone_pos[2], -0.1)
            self._pub_pos([self.drone_pos[0], self.drone_pos[1], safe_z])
            if now - self.last_cmd_time > 1.0:
                self._pub_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.get_logger().info('ARM requested')
                self.last_cmd_time = now
            return

        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            safe_z = max(self.drone_pos[2], -0.1)
            self._pub_pos([self.drone_pos[0], self.drone_pos[1], safe_z])
            if now - self.last_cmd_time > 1.0:
                self._pub_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.get_logger().info('OFFBOARD requested')
                self.last_cmd_time = now
            return

        # Armed + offboard → climb.
        self._pub_pos([self.drone_pos[0], self.drone_pos[1], alt])
        if abs(self.drone_pos[2] - alt) < 0.3:
            self.get_logger().info('Takeoff complete -> SEARCH')
            self.forward_start_pos = self.drone_pos.copy()
            self.state = State.SEARCH

    def _search(self):
        """Fly to search_target_n/e (flanking waypoint), yawing toward target throughout.

        Yaw priority:
          1. IBVS q_z (camera-derived LOS azimuth) when target is detected — no prior knowledge.
          2. Fallback: face toward search waypoint when target not yet visible.
        """
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

        if self.ibvs_output is not None and self.ibvs_output.detected:
            yaw = self.ibvs_output.q_z
        else:
            yaw = math.atan2(
                self.search_target_e - self.drone_pos[1],
                self.search_target_n - self.drone_pos[0],
            )
        self._pub_pos(
            [self.search_target_n, self.search_target_e, -self.takeoff_height],
            yaw=yaw,
        )

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
            # Convert absolute Gazebo ENU → drone-local NED by subtracting spawn offset.
            # drone_pos (from PX4 Monitoring) is relative to the drone's own spawn point.
            target_pos_ned = np.array([
                self.target_world_pos[1] - self.spawn_gazebo_y,  # North = Gazebo Y - spawn_Y
                self.target_world_pos[0] - self.spawn_gazebo_x,  # East  = Gazebo X - spawn_X
                -self.target_world_pos[2]                         # Down  = -Gazebo Z
            ])
            dist = np.linalg.norm(self.drone_pos - target_pos_ned)
            if dist < self.collision_dist:
                self.get_logger().info(f'COLLISION at dist={dist:.2f}m -> kill drone')
                self._collision_kill()
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
        if self._killed:
            self.get_logger().info('Collision landing.', throttle_duration_sec=2.0)
        else:
            self._pub_pos(self.drone_pos.tolist(), yaw=self.drone_yaw)
            self.get_logger().info('Mission DONE, hovering.', throttle_duration_sec=5.0)

    def _collision_kill(self):
        """Stop OCM heartbeat and command landing on collision."""
        self._killed = True
        self.state   = State.DONE
        self._pub_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('Collision: NAV_LAND sent, OCM stopped -> DONE')

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
