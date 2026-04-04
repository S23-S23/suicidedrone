#!/usr/bin/env python3
"""
Drone Manager Node
FSM: IDLE → TAKEOFF → FORWARD → INTERCEPT → DONE

INTERCEPT replaces the previous TRACKING+CHARGING states.
  - Uses velocity control (OffboardControlMode.velocity=True)
  - Velocity setpoint from PNG guidance (/png/velocity_cmd)
  - Yaw rate setpoint from IBVS FOV controller (/ibvs/fov_yaw_rate)
  - Exits to FORWARD on target lost, DONE on collision
"""

import math
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
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool, Float64
from suicide_drone_msgs.msg import IBVSOutput
from enum import Enum
import numpy as np


class MissionState(Enum):
    IDLE      = 0
    TAKEOFF   = 1
    FORWARD   = 2
    INTERCEPT = 3   # merged TRACKING + CHARGING (velocity control + PNG guidance)
    DONE      = 4


class BalloonHunterDroneManager(Node):
    def __init__(self):
        super().__init__('drone_manager')

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter('system_id',          1)
        self.declare_parameter('takeoff_height',     2.0)
        self.declare_parameter('forward_speed',      2.0)
        self.declare_parameter('forward_distance_limit', 50.0)

        self.system_id              = self.get_parameter('system_id').value
        self.takeoff_height         = self.get_parameter('takeoff_height').value
        self.forward_speed          = self.get_parameter('forward_speed').value
        self.forward_distance_limit = self.get_parameter('forward_distance_limit').value

        self.get_logger().info(
            f'Drone Manager {self.system_id} initializing (IBVS+PNG mode)...'
        )

        self.topic_prefix_fmu = f'drone{self.system_id}/fmu/'

        # ── State variables ─────────────────────────────────────────────────
        self.state            = MissionState.IDLE
        self.drone_pos        = np.array([0.0, 0.0, 0.0])   # NED [m]
        self.drone_yaw        = 0.0                           # [rad]
        self.nav_state        = 0
        self.arming_state     = 0
        self.last_cmd_time    = 0.0
        self.forward_start_pos = None

        # INTERCEPT inputs (from IBVS + PNG)
        self.target_detected  = False
        self.vel_cmd          = None    # NED velocity [m/s] from PNG
        self.fov_yaw_rate     = 0.0    # [rad/s] from IBVS (horizontal, Eq.13)
        self.fov_vel_z        = 0.0    # [m/s] NED Z correction from IBVS ey
        self.collision_done   = False

        # Set when INTERCEPT is entered for the first time.
        # After that, FORWARD hovers in place instead of returning to origin.
        self.intercept_entered_once = False
        self.hover_pos = None           # position to hold when target is lost

        # ── Publishers ──────────────────────────────────────────────────────
        self.ocm_publisher = self.create_publisher(
            OffboardControlMode,
            f'{self.topic_prefix_fmu}in/offboard_control_mode',
            qos_profile_sensor_data,
        )
        self.traj_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint,
            f'{self.topic_prefix_fmu}in/trajectory_setpoint',
            qos_profile_sensor_data,
        )
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand,
            f'{self.topic_prefix_fmu}in/vehicle_command',
            qos_profile_sensor_data,
        )
        self.mission_state_pub = self.create_publisher(String, '/mission_state', 10)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(
            VehicleStatus,
            f'{self.topic_prefix_fmu}out/vehicle_status',
            self.status_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Monitoring,
            f'{self.topic_prefix_fmu}out/monitoring',
            self.monitoring_callback,
            qos_profile_sensor_data,
        )

        # IBVS: detection flag + LOS angles (merged)
        self.create_subscription(
            IBVSOutput,
            '/ibvs/output',
            self.ibvs_output_callback,
            10,
        )
        # IBVS: FOV yaw rate command (Eq.13, horizontal)
        self.create_subscription(
            Float64,
            '/ibvs/fov_yaw_rate',
            self.fov_yaw_rate_callback,
            10,
        )
        # IBVS: FOV Z velocity correction (ey-based, vertical)
        self.create_subscription(
            Float64,
            '/ibvs/fov_vel_z',
            self.fov_vel_z_callback,
            10,
        )
        # PNG: velocity command (Eq.10)
        self.create_subscription(
            Twist,
            '/png/velocity_cmd',
            self.vel_cmd_callback,
            10,
        )
        # Collision event from collision_handler
        self.create_subscription(
            Bool,
            '/balloon_collision',
            self.collision_callback,
            10,
        )

        # ── Timers ──────────────────────────────────────────────────────────
        self.create_timer(0.02,  self.timer_ocm_callback)      # 50 Hz
        self.create_timer(0.02, self.timer_mission_callback)  # 50 Hz

        # Single-shot: wait 5 s for PX4 before starting mission
        self.start_mission_timer = self.create_timer(5.0, self.start_mission)

    # ── PX4 status callbacks ──────────────────────────────────────────────────

    def status_callback(self, msg: VehicleStatus):
        self.nav_state    = msg.nav_state
        self.arming_state = msg.arming_state

    def monitoring_callback(self, msg: Monitoring):
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        self.drone_yaw = msg.head
        self.get_logger().info(
            f'[DEBUG] pos=({msg.pos_x:.2f},{msg.pos_y:.2f},{msg.pos_z:.2f})',
            throttle_duration_sec=5.0,
        )

    # ── IBVS / PNG / collision callbacks ─────────────────────────────────────

    def ibvs_output_callback(self, msg: IBVSOutput):
        self.target_detected = msg.detected

        # Level-triggered: FORWARD → INTERCEPT whenever target is visible
        # Edge-triggered (not was_detected) is intentionally avoided:
        # the rising edge can occur during TAKEOFF before FORWARD is entered,
        # causing the transition to be permanently missed.
        if msg.detected and self.state == MissionState.FORWARD:
            self.get_logger().info('Target detected! Switching to INTERCEPT.')
            self.intercept_entered_once = True
            self.state = MissionState.INTERCEPT

        if not msg.detected and self.state == MissionState.INTERCEPT:
            self.get_logger().warn('Target lost! Hovering in place (FORWARD).')
            self.hover_pos = self.drone_pos.copy()   # freeze current position
            self.state = MissionState.FORWARD

    def fov_yaw_rate_callback(self, msg: Float64):
        """FOV yaw rate command from IBVS (Eq.13, horizontal)."""
        self.fov_yaw_rate = msg.data

    def fov_vel_z_callback(self, msg: Float64):
        """NED Z velocity correction from IBVS ey (vertical centering)."""
        self.fov_vel_z = msg.data

    def vel_cmd_callback(self, msg: Twist):
        """NED velocity command from PNG guidance (Eq.10)."""
        self.vel_cmd = np.array([msg.linear.x, msg.linear.y, msg.linear.z])

    def collision_callback(self, msg: Bool):
        if msg.data and self.state == MissionState.INTERCEPT:
            self.get_logger().info('Balloon collision! Mission DONE.')
            self.collision_done = True
            self.state = MissionState.DONE

    # ── Mission start ─────────────────────────────────────────────────────────

    def start_mission(self):
        self.start_mission_timer.cancel()
        if self.state == MissionState.IDLE:
            self.get_logger().info('Starting mission: IDLE → TAKEOFF')
            self.state = MissionState.TAKEOFF

    # ── OCM timer: must match current control mode ────────────────────────────

    def timer_ocm_callback(self):
        msg           = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        # position=True + velocity=True: position feedback (altitude/position hold) +
        # velocity feedforward (fast response). In INTERCEPT, the current position is
        # passed as the position setpoint so that the position error = 0, and movement
        # is driven solely by velocity feedforward (PNG). In FORWARD hover, hover_pos
        # is passed as the position setpoint so position feedback actively corrects drift.
        if self.intercept_entered_once or self.state == MissionState.INTERCEPT:
            msg.position = True
            msg.velocity = True
        else:
            msg.position = True
            msg.velocity = False
        self.ocm_publisher.publish(msg)

    # ── Mission FSM timer ─────────────────────────────────────────────────────

    def timer_mission_callback(self):
        # Publish state name for drone_visualizer
        state_msg      = String()
        state_msg.data = self.state.name
        self.mission_state_pub.publish(state_msg)

        self.get_logger().info(
            f'[DEBUG] state={self.state.name}, '
            f'pos=({self.drone_pos[0]:.2f},{self.drone_pos[1]:.2f},{self.drone_pos[2]:.2f}), '
            f'armed={self.arming_state}, nav={self.nav_state}',
            throttle_duration_sec=3.0,
        )

        if self.state == MissionState.IDLE:
            safe_z = max(self.drone_pos[2], -0.1)
            self._publish_position_setpoint([self.drone_pos[0], self.drone_pos[1], safe_z])
        elif self.state == MissionState.TAKEOFF:
            self.handle_takeoff()
        elif self.state == MissionState.FORWARD:
            self.handle_forward()
        elif self.state == MissionState.INTERCEPT:
            self.handle_intercept()
        elif self.state == MissionState.DONE:
            self.handle_done()

    # ── State handlers ────────────────────────────────────────────────────────

    def handle_takeoff(self):
        target_alt = -self.takeoff_height  # NED Down → negative = up

        # Always publish a setpoint first (PX4 offboard streaming requirement)
        if (self.arming_state != VehicleStatus.ARMING_STATE_ARMED
                or self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD):
            safe_z = max(self.drone_pos[2], -0.1)
            self._publish_position_setpoint(
                [self.drone_pos[0], self.drone_pos[1], safe_z]
            )
        else:
            self._publish_position_setpoint(
                [self.drone_pos[0], self.drone_pos[1], target_alt]
            )

        now = self.get_clock().now().nanoseconds / 1e9

        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if now - self.last_cmd_time > 1.0:
                self.publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0
                )
                self.get_logger().info('Attempting to ARM...')
                self.last_cmd_time = now
            return

        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if now - self.last_cmd_time > 1.0:
                self.publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0
                )
                self.get_logger().info('Requesting OFFBOARD mode...')
                self.last_cmd_time = now
            return

        if abs(self.drone_pos[2] - target_alt) < 0.3:
            self.get_logger().info('Takeoff complete! Switching to FORWARD.')
            self.forward_start_pos = self.drone_pos.copy()
            self.state = MissionState.FORWARD

    def handle_forward(self):
        # ── Case A: target was lost during INTERCEPT → hover where we are ──
        # In position=True+velocity=True mode, hover_pos is passed as the position
        # setpoint and velocity=[0,0,0] as feedforward. Position feedback actively
        # corrects altitude drift while velocity mode is maintained so the drone
        # can quickly return to INTERCEPT when the target is re-acquired.
        if self.intercept_entered_once:
            hover = self.hover_pos if self.hover_pos is not None else self.drone_pos
            self._publish_velocity_setpoint([0.0, 0.0, 0.0], pos=hover, yawspeed=0.0)
            self.get_logger().info(
                f'[DEBUG] FORWARD (hover, pos=({hover[0]:.2f},{hover[1]:.2f},{hover[2]:.2f}))',
                throttle_duration_sec=2.0,
            )
            return

        # ── Case B: initial forward flight before first INTERCEPT ──
        if self.forward_start_pos is None:
            self.forward_start_pos = self.drone_pos.copy()

        distance_traveled = np.linalg.norm(
            self.drone_pos[:2] - self.forward_start_pos[:2]
        )

        self.get_logger().info(
            f'[DEBUG] FORWARD: traveled={distance_traveled:.2f}m',
            throttle_duration_sec=2.0,
        )

        if distance_traveled >= self.forward_distance_limit:
            self.get_logger().info(
                f'Forward limit reached ({distance_traveled:.2f}m). Hovering.',
                throttle_duration_sec=2.0,
            )
            self._publish_position_setpoint(
                [self.drone_pos[0], self.drone_pos[1], -self.takeoff_height]
            )
            return

        # Fly forward (North in NED = PX4 x-axis)
        self._publish_position_setpoint([1.0, 0.0, -self.takeoff_height])

    def handle_intercept(self):
        """
        Position+Velocity feedforward intercept using PNG guidance + IBVS ey correction.
        Position setpoint: current drone position (position error=0 → position feedback neutralized)
        Velocity setpoint: PNG Eq.10 + IBVS fov_vel_z (Z image-center correction)
        Yaw rate setpoint: IBVS fov_yaw_rate Eq.13 (horizontal image-center correction)
        """
        if self.vel_cmd is None:
            # No velocity command yet – hover until PNG publishes
            self._publish_velocity_setpoint([0.0, 0.0, 0.0], pos=self.drone_pos, yawspeed=0.0)
            return

        # PNG velocity + IBVS ey Z-correction
        vel = self.vel_cmd.copy()
        vel[2] += self.fov_vel_z

        self._publish_velocity_setpoint(
            vel, pos=self.drone_pos, yawspeed=self.fov_yaw_rate
        )

        self.get_logger().info(
            f'[DEBUG] INTERCEPT: png_v=({self.vel_cmd[0]:.2f},{self.vel_cmd[1]:.2f},'
            f'{self.vel_cmd[2]:.2f}) fov_vz={self.fov_vel_z:.3f} yr={self.fov_yaw_rate:.3f}',
            throttle_duration_sec=1.0,
        )

    def handle_done(self):
        # Hover in place
        self._publish_position_setpoint(
            [self.drone_pos[0], self.drone_pos[1], self.drone_pos[2]],
            yaw=self.drone_yaw,
        )

    # ── Setpoint helpers ──────────────────────────────────────────────────────

    def _publish_position_setpoint(self, pos, yaw=0.0):
        """Send position setpoint in NED frame."""
        msg          = TrajectorySetpoint()
        msg.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        msg.yaw      = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_setpoint_publisher.publish(msg)

    def _publish_velocity_setpoint(self, vel, pos=None, yawspeed=0.0):
        """
        Send velocity (+ optional position feedforward) setpoint in NED frame.
        pos: position setpoint [N, E, D]. None → NaN (velocity-only).
             INTERCEPT: pos=drone_pos (error=0, only velocity feedforward acts)
             FORWARD hover: pos=hover_pos (position feedback corrects drift)
        """
        nan = float('nan')
        msg          = TrajectorySetpoint()
        if pos is not None:
            msg.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        else:
            msg.position = [nan, nan, nan]
        msg.velocity = [float(vel[0]), float(vel[1]), float(vel[2])]
        msg.yaw      = nan
        msg.yawspeed = float(yawspeed)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, p1=0.0, p2=0.0):
        msg                 = VehicleCommand()
        msg.param1          = p1
        msg.param2          = p2
        msg.command         = command
        msg.target_system   = self.system_id
        msg.target_component = 1
        msg.source_system   = 1
        msg.source_component = 1
        msg.from_external   = True
        msg.timestamp       = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BalloonHunterDroneManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
