#!/usr/bin/env python3
"""
Drone Manager for Real Flight
==============================
Simplified FSM for real-world IBVS+PNG balloon tracking.

Flow:
  1. Drone is flying under RC control (POSCTL / ALTCTL)
  2. User launches this node
  3. INIT: Capture current position, send offboard heartbeat + position hold
  4. HOVER_INIT: After OFFBOARD mode is active, hold position for 2s (filter init)
  5. TRACKING: Follow PNG guidance velocity commands
     - Target detected: velocity control from PNG
     - Target lost: hold last known position
  6. User regains control by switching RC to non-OFFBOARD mode

No collision detection, no mission timeout auto-shutdown.
The user manually takes over with RC when desired.

Subscriptions:
  drone{id}/fmu/out/vehicle_status          — arming/nav state
  drone{id}/fmu/out/monitoring              — position, attitude
  /png/guidance_cmd                         — GuidanceCmd from PNG

Publications:
  drone{id}/fmu/in/offboard_control_mode
  drone{id}/fmu/in/trajectory_setpoint
  drone{id}/fmu/in/vehicle_command
  /mission_state                            — String (state name)
"""

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
from std_msgs.msg import String
from suicide_drone_msgs.msg import GuidanceCmd
from enum import Enum


class State(Enum):
    INIT = 0
    HOVER_INIT = 1
    TRACKING = 2


class DroneManagerReal(Node):
    def __init__(self):
        super().__init__('drone_manager')

        # ── Parameters ──
        self.declare_parameter('system_id', 1)
        self.declare_parameter('hover_init_duration', 2.0)  # seconds to hover for filter init
        self.declare_parameter('max_speed', 10.0)

        self.system_id          = self.get_parameter('system_id').value
        self.hover_init_dur     = self.get_parameter('hover_init_duration').value
        self.max_speed          = self.get_parameter('max_speed').value

        self.topic_prefix = f"drone{self.system_id}/fmu/"

        # ── State variables ──
        self.state            = State.INIT
        self.drone_pos        = np.zeros(3)  # NED
        self.drone_yaw        = 0.0
        self.nav_state        = 0
        self.arming_state     = 0
        self.last_cmd_time    = 0.0
        self.ocm_count        = 0            # offboard heartbeat count
        self.pos_received     = False        # have we received at least one position?

        # Position to hold during INIT/HOVER_INIT
        self.hold_pos         = np.zeros(3)
        self.hold_yaw         = 0.0

        # Hover init timer
        self._hover_start_t   = None

        # TRACKING inputs (from PNG guidance_cmd)
        self.guidance_cmd     = None
        self._last_hold_pos   = np.zeros(3)  # position when target was last lost

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

        # ── Timers ──
        self.create_timer(0.1,  self.ocm_cb)      # 10 Hz offboard heartbeat
        self.create_timer(0.02, self.control_cb)   # 50 Hz main control

        self.get_logger().info(
            f'DroneManagerReal started: INIT -> HOVER_INIT ({self.hover_init_dur}s) -> TRACKING'
        )

    # ── Callbacks ──
    def status_cb(self, msg: VehicleStatus):
        self.nav_state    = msg.nav_state
        self.arming_state = msg.arming_state

    def monitoring_cb(self, msg: Monitoring):
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        self.drone_yaw = msg.head
        if not self.pos_received:
            self.pos_received = True
            self.hold_pos = self.drone_pos.copy()
            self.hold_yaw = self.drone_yaw
            self.get_logger().info(
                f'Initial position captured: NED=({msg.pos_x:.2f}, {msg.pos_y:.2f}, {msg.pos_z:.2f})'
            )

    def guidance_cmd_cb(self, msg: GuidanceCmd):
        self.guidance_cmd = msg

    # ── Offboard heartbeat (10Hz) ──
    def ocm_cb(self):
        msg = OffboardControlMode()
        if self.state == State.TRACKING and self.guidance_cmd is not None \
                and self.guidance_cmd.target_detected:
            msg.position = False
            msg.velocity = True
        else:
            msg.position = True
            msg.velocity = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.ocm_pub.publish(msg)
        self.ocm_count += 1

    # ── Main control loop (50Hz) ──
    def control_cb(self):
        # Publish state for other nodes
        state_msg = String()
        state_msg.data = self.state.name
        self.state_pub.publish(state_msg)

        if self.state == State.INIT:
            self._init()
        elif self.state == State.HOVER_INIT:
            self._hover_init()
        elif self.state == State.TRACKING:
            self._tracking()

    # ── State handlers ──
    def _init(self):
        """Wait for position data, send position hold, request OFFBOARD."""
        if not self.pos_received:
            return

        # Always publish position hold at captured position
        self._pub_pos(self.hold_pos.tolist(), yaw=self.hold_yaw)

        now = self.get_clock().now().nanoseconds / 1e9

        # Need at least ~20 OCM messages before requesting OFFBOARD
        if self.ocm_count < 20:
            return

        # Request OFFBOARD mode
        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if now - self.last_cmd_time > 1.0:
                self._pub_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.get_logger().info('OFFBOARD mode requested (switch RC to OFFBOARD if needed)')
                self.last_cmd_time = now
            return

        # OFFBOARD mode active -> transition to HOVER_INIT
        self.get_logger().info('OFFBOARD active -> HOVER_INIT')
        self._hover_start_t = time.time()
        self.state = State.HOVER_INIT

    def _hover_init(self):
        """Hold position for hover_init_duration seconds (filter initialization)."""
        self._pub_pos(self.hold_pos.tolist(), yaw=self.hold_yaw)

        elapsed = time.time() - self._hover_start_t
        if elapsed >= self.hover_init_dur:
            self.get_logger().info(
                f'Filter init complete ({self.hover_init_dur}s) -> TRACKING'
            )
            self._last_hold_pos = self.drone_pos.copy()
            self.state = State.TRACKING

        self.get_logger().info(
            f'HOVER_INIT: {elapsed:.1f}/{self.hover_init_dur:.1f}s',
            throttle_duration_sec=0.5,
        )

    def _tracking(self):
        """Follow PNG guidance when target detected, hold position otherwise."""
        if self.guidance_cmd is not None and self.guidance_cmd.target_detected:
            # Velocity control from PNG
            cmd = self.guidance_cmd
            vel = np.array([cmd.vel_n, cmd.vel_e, cmd.vel_d])
            self._pub_vel(vel, yaw_rate=cmd.yaw_rate)

            # Update hold position for when target is lost
            self._last_hold_pos = self.drone_pos.copy()

            self.get_logger().info(
                f'TRACKING: v=({cmd.vel_n:.2f},{cmd.vel_e:.2f},{cmd.vel_d:.2f}) '
                f'yr={cmd.yaw_rate:.3f}',
                throttle_duration_sec=1.0,
            )
        else:
            # Target not detected -> hold position
            self._pub_pos(self._last_hold_pos.tolist(), yaw=self.drone_yaw)
            self.get_logger().info(
                'TRACKING: target lost, holding position',
                throttle_duration_sec=2.0,
            )

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
    node = DroneManagerReal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
