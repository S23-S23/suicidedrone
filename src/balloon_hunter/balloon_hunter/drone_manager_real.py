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
    VehicleLandDetected,
    Monitoring,
)
from std_msgs.msg import String, Float32MultiArray
from suicide_drone_msgs.msg import GuidanceCmd, TargetInfo
from enum import Enum


class State(Enum):
    INIT = 0
    HOVER_INIT = 1
    TRACKING = 2
    LANDING = 3      # 직충돌 감지 -> AUTO.LAND
    DISARMED = 4     # 착지 완료 -> DISARM 송신 후 종료


class DroneManagerReal(Node):
    def __init__(self):
        super().__init__('drone_manager')

        # ── Parameters ──
        self.declare_parameter('system_id', 1)
        self.declare_parameter('hover_init_duration', 2.0)  # seconds to hover for filter init
        self.declare_parameter('max_speed', 10.0)

        # ── 직충돌 감지 (bbox 면적비율 + 검출 소실) ──
        # 충돌 판정: "세션 peak 면적비율 >= collision_area_frac"인 상태에서
        #            "collision_lost_time 초 이상 미검출"이면 직충돌/도달로 보고 LAND+DISARM.
        self.declare_parameter('collision_area_frac', 0.50)  # 화면 대비 bbox 면적비율 임계 (0~1)
        self.declare_parameter('collision_lost_time', 1.0)   # 임계 초과 후 미검출 지속시간 [s]
        self.declare_parameter('collision_min_edges', 0)     # peak 프레임 경계접촉 변 수 하한(0=미사용)
        self.declare_parameter('image_width', 1280)          # /target_info 기준 영상 폭
        self.declare_parameter('image_height', 720)          # /target_info 기준 영상 높이
        self.declare_parameter('enable_collision_land', True)  # 충돌 시 자동 착륙 on/off

        self.system_id          = self.get_parameter('system_id').value
        self.hover_init_dur     = self.get_parameter('hover_init_duration').value
        self.max_speed          = self.get_parameter('max_speed').value
        self.collision_area_frac = self.get_parameter('collision_area_frac').value
        self.collision_lost_time = self.get_parameter('collision_lost_time').value
        self.collision_min_edges = self.get_parameter('collision_min_edges').value
        self.img_w              = int(self.get_parameter('image_width').value)
        self.img_h              = int(self.get_parameter('image_height').value)
        self.enable_collision_land = self.get_parameter('enable_collision_land').value
        self.img_area           = float(self.img_w * self.img_h)

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

        # ── 직충돌 감지 상태 ──
        self._bbox_last_frac  = 0.0          # 가장 최근 프레임 면적비율
        self._bbox_peak_frac  = 0.0          # 현재 추적 세션의 최대 면적비율
        self._bbox_peak_edges = 0            # peak 프레임에서의 경계접촉 변 수
        self._last_target_time = None        # 마지막 /target_info 수신 시각
        self._collision_latched = False      # 충돌 판정 래치 (한 번 걸리면 유지)
        self._land_cmd_time   = 0.0          # AUTO.LAND 명령 재송신 throttle
        self.landed           = False        # VehicleLandDetected.landed

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
        # 충돌 감지 디버그/로깅용: [last_frac, peak_frac, peak_edges, lost_time, latched]
        self.collision_pub = self.create_publisher(Float32MultiArray, '/collision_info', 10)

        # ── Subscribers ──
        self.create_subscription(
            VehicleStatus,
            f'{self.topic_prefix}out/vehicle_status_v1',
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
        # 직충돌 감지용: 탐지 bbox 크기 추적
        self.create_subscription(
            TargetInfo,
            '/target_info',
            self.target_info_cb,
            10,
        )
        # 착지 완료 확인용 (DISARM 트리거)
        self.create_subscription(
            VehicleLandDetected,
            f'{self.topic_prefix}out/vehicle_land_detected',
            self.land_detected_cb,
            qos_profile_sensor_data,
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

    def land_detected_cb(self, msg: VehicleLandDetected):
        self.landed = msg.landed

    def target_info_cb(self, msg: TargetInfo):
        """탐지 bbox 크기를 추적해 충돌 판정용 세션 peak 면적비율을 갱신."""
        now = self.get_clock().now()
        L, T, R, B = msg.left, msg.top, msg.right, msg.bottom
        w = max(0, R - L)
        h = max(0, B - T)
        frac = (w * h) / self.img_area
        edges = self._edges_touched(L, T, R, B)

        # 장시간 끊겼다가 다시 잡히면 새 세션 -> peak 리셋
        if self._last_target_time is not None:
            gap = (now - self._last_target_time).nanoseconds / 1e9
            if gap > self.collision_lost_time:
                self._bbox_peak_frac = 0.0
                self._bbox_peak_edges = 0

        self._bbox_last_frac = frac
        if frac > self._bbox_peak_frac:
            self._bbox_peak_frac = frac
            self._bbox_peak_edges = edges
        self._last_target_time = now

    def _edges_touched(self, L, T, R, B):
        """bbox가 화면 경계에 접한 변 개수 (가까울수록 클리핑되어 증가)."""
        n = 0
        if L <= 0:              n += 1
        if T <= 0:              n += 1
        if R >= self.img_w - 1: n += 1
        if B >= self.img_h - 1: n += 1
        return n

    def _check_collision(self):
        """직충돌/도달 판정: peak 면적비율 충족 + 일정시간 미검출."""
        if self._collision_latched:
            return True
        if not self.enable_collision_land:
            return False
        if self._last_target_time is None:
            return False
        if self._bbox_peak_frac < self.collision_area_frac:
            return False
        if self._bbox_peak_edges < self.collision_min_edges:
            return False
        lost = (self.get_clock().now() - self._last_target_time).nanoseconds / 1e9
        if lost >= self.collision_lost_time:
            self._collision_latched = True
            self.get_logger().warn(
                f'COLLISION 감지: peak_frac={self._bbox_peak_frac*100:.1f}% '
                f'(edges={self._bbox_peak_edges}), 소실={lost:.1f}s -> LANDING'
            )
            return True
        return False

    # ── Offboard heartbeat (10Hz) ──
    def ocm_cb(self):
        # 착륙/종료 단계에서는 OFFBOARD heartbeat 중단 -> AUTO.LAND가 인계받게 함
        if self.state in (State.LANDING, State.DISARMED):
            return
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

        # 충돌 감지 디버그 정보 발행 (rosbag 로깅용)
        if self._last_target_time is not None:
            lost = (self.get_clock().now() - self._last_target_time).nanoseconds / 1e9
        else:
            lost = -1.0  # 아직 한 번도 탐지 안 됨
        ci = Float32MultiArray()
        ci.data = [
            float(self._bbox_last_frac),
            float(self._bbox_peak_frac),
            float(self._bbox_peak_edges),
            float(lost),
            1.0 if self._collision_latched else 0.0,
        ]
        self.collision_pub.publish(ci)

        if self.state == State.INIT:
            self._init()
        elif self.state == State.HOVER_INIT:
            self._hover_init()
        elif self.state == State.TRACKING:
            if self._check_collision():
                self.state = State.LANDING
                self._land_cmd_time = 0.0
            else:
                self._tracking()
        elif self.state == State.LANDING:
            self._landing()
        elif self.state == State.DISARMED:
            pass  # 종료 상태: 아무 setpoint도 발행하지 않음

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

    def _landing(self):
        """직충돌 후 AUTO.LAND로 현재 위치 착륙 -> 착지 확인 후 DISARM.

        - OFFBOARD setpoint/heartbeat는 발행하지 않는다(ocm_cb에서 차단).
        - AUTO.LAND 진입 명령을 착지 전까지 1초 간격으로 재송신(누락 대비).
        - VehicleLandDetected.landed=True 면 DISARM(강제 아님) 후 종료.
          (land_detected가 안 와도 PX4가 착지 후 자동 disarm 하므로 안전망 존재)
        """
        now = self.get_clock().now().nanoseconds / 1e9

        if not self.landed:
            if now - self._land_cmd_time > 1.0:
                # DO_SET_MODE -> AUTO.LAND : base=custom(1), main=AUTO(4), sub=LAND(6)
                self._pub_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 6.0)
                self._land_cmd_time = now
                self.get_logger().info('AUTO.LAND 명령 송신 (착지 대기)', throttle_duration_sec=1.0)
            return

        # 착지 완료 -> DISARM (param1=0: disarm, 강제 플래그 미사용)
        self._pub_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.get_logger().warn('착지 확인 -> DISARM 송신, 임무 종료')
        self.state = State.DISARMED

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

    def _pub_cmd(self, cmd, p1=0.0, p2=0.0, p3=0.0):
        msg = VehicleCommand()
        msg.param1, msg.param2, msg.param3, msg.command = p1, p2, p3, cmd
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
