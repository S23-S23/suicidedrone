#!/usr/bin/env python3
"""
Mission Logger Node
Records mission data to a CSV file.
File: /tmp/mission_log_<YYYY-MM-DD_HH-MM-SS>.csv

Additional log:
  /tmp/mission_events_<YYYY-MM-DD_HH-MM-SS>.log
    - State transition events (with timestamps)
    - Per-state summary statistics on mission end
"""

import csv
import math
import os
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String, Bool, Float64
from geometry_msgs.msg import Vector3, Twist
from gazebo_msgs.msg import ModelStates
from px4_msgs.msg import VehicleLocalPosition


CSV_HEADER = [
    # ── Basic state ────────────────────────────────────────────────────
    'timestamp_sec',
    'mission_state',
    'target_detected',
    # ── Drone position (Gazebo GT, ENU) ────────────────────────────────
    'drone_gt_x_enu',
    'drone_gt_y_enu',
    'drone_gt_z_enu',
    # ── Drone position/velocity (PX4, NED) ─────────────────────────────
    'drone_px4_x_ned',
    'drone_px4_y_ned',
    'drone_px4_z_ned',
    'drone_px4_vx_ned',
    'drone_px4_vy_ned',
    'drone_px4_vz_ned',
    # ── Balloon position / distance ────────────────────────────────────
    'balloon_x_enu',
    'balloon_y_enu',
    'balloon_z_enu',
    'dist_to_balloon_m',
    # ── IBVS: Image error (Eq.3) ───────────────────────────────────────
    'ex',
    'ey',
    'ex_dot',
    'pixel_u',
    'pixel_v',
    # ── IBVS: LOS angles (Eq.7) ────────────────────────────────────────
    'q_y_deg',
    'q_z_deg',
    # ── IBVS: FOV controller (Eq.13) ───────────────────────────────────
    'fov_kp_term',
    'fov_kd_term',
    'fov_yaw_rate',
    # ── IBVS: Drone attitude ────────────────────────────────────────────
    'roll_deg',
    'pitch_deg',
    'yaw_deg',
    'body_yaw_rate_rad_s',
    # ── PNG: Velocity direction angles (Eq.8/9) ─────────────────────────
    'sigma_y_cur_deg',
    'sigma_z_cur_deg',
    'sigma_y_des_deg',
    'sigma_z_des_deg',
    # ── PNG: LOS rate ───────────────────────────────────────────────────
    'los_rate_qy_deg',
    'los_rate_qz_deg',
    # ── PNG: Speed info (Eq.14) ─────────────────────────────────────────
    'png_v_now_m_s',
    'png_speed_actual_m_s',
    'png_sigma_source',        # 0=velocity-based, 1=body-based
    # ── PNG: Velocity command (Eq.10) ───────────────────────────────────
    'v_cmd_n',
    'v_cmd_e',
    'v_cmd_d',
    'v_cmd_total',
]

# Per-state statistics initial values
def _new_state_stats():
    return {
        'visit_count':    0,
        'tick_count':     0,
        'total_duration': 0.0,
        'dist_min':       float('inf'),
        'dist_max':       float('-inf'),
        'dist_sum':       0.0,
        'dist_cnt':       0,
        'det_true':       0,
        'det_total':      0,
        'vcmd_sum':       0.0,
        'vcmd_cnt':       0,
    }


class MissionLogger(Node):
    def __init__(self):
        super().__init__('mission_logger')

        # Parameters
        self.declare_parameter('system_id', 1)
        self.declare_parameter('drone_model_name', '')
        self.declare_parameter('balloon_model_name', 'target_balloon')
        self.declare_parameter('balloon_link_z_offset', 1.5)
        self.declare_parameter('log_rate', 10.0)

        system_id = self.get_parameter('system_id').value
        drone_model_name_param = self.get_parameter('drone_model_name').value
        self._drone_model_name = drone_model_name_param if drone_model_name_param else f'drone{system_id}'
        self._balloon_model_name = self.get_parameter('balloon_model_name').value
        self._balloon_link_z_offset = self.get_parameter('balloon_link_z_offset').value
        log_rate = self.get_parameter('log_rate').value

        # ── Time-series state variables ────────────────────────────────────
        self._mission_state = None
        self._target_detected = None
        # IBVS image error
        self._ex = None
        self._ey = None
        self._ex_dot = None
        self._pixel_u = None
        self._pixel_v = None
        # IBVS LOS
        self._q_y_rad = None
        self._q_z_rad = None
        # IBVS FOV controller
        self._fov_kp_term = None
        self._fov_kd_term = None
        self._fov_yaw_rate = None
        # IBVS attitude
        self._roll_deg = None
        self._pitch_deg = None
        self._yaw_deg = None
        self._body_yaw_rate = None
        # PNG sigma
        self._sigma_y_cur = None
        self._sigma_z_cur = None
        self._sigma_y_des = None
        self._sigma_z_des = None
        # PNG LOS rate
        self._los_rate_qy = None
        self._los_rate_qz = None
        # PNG speed
        self._png_v_now = None
        self._png_speed_actual = None
        self._png_sigma_source = None
        # PNG velocity cmd
        self._v_cmd_n = None
        self._v_cmd_e = None
        self._v_cmd_d = None
        # Drone/balloon position
        self._drone_gt_x = None
        self._drone_gt_y = None
        self._drone_gt_z = None
        self._balloon_x = None
        self._balloon_y = None
        self._balloon_z = None
        self._px4_x = None
        self._px4_y = None
        self._px4_z = None
        self._px4_vx = None
        self._px4_vy = None
        self._px4_vz = None

        # ── State transition / event tracking variables ────────────────────
        self._prev_mission_state = None   # previous state (for transition detection)
        self._mission_start_time = None   # time of the first mission tick
        self._state_entry_time   = None   # time of current state entry
        self._collision_occurred = False  # whether a collision event was received
        self._state_stats        = {}     # state_name → statistics dict

        # ── Open CSV file ──────────────────────────────────────────────────
        timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_path = f'/tmp/mission_log_{timestamp_str}.csv'
        self._csv_file = open(log_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(CSV_HEADER)

        # ── Open event log file ───────────────────────────────────────────
        event_log_path = f'/tmp/mission_events_{timestamp_str}.log'
        self._event_file = open(event_log_path, 'w', buffering=1)  # line-buffered
        self._event_file.write(f'Mission Log Started : {timestamp_str}\n')
        self._event_file.write(f'CSV file            : {log_path}\n')
        self._event_file.write('=' * 60 + '\n')

        self.get_logger().info(f'Mission log     : {log_path}')
        self.get_logger().info(f'Mission events  : {event_log_path}')

        # ── QoS ───────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(String, '/mission_state',             self._cb_mission_state,      10)
        self.create_subscription(Bool,   '/ibvs/target_detected',      self._cb_target_detected,    10)
        # IBVS image error
        self.create_subscription(Vector3,'/ibvs/image_error',          self._cb_image_error,        10)
        self.create_subscription(Float64,'/ibvs/ex_dot',               self._cb_ex_dot,             10)
        self.create_subscription(Vector3,'/ibvs/pixel_center',         self._cb_pixel_center,       10)
        # IBVS LOS / FOV controller
        self.create_subscription(Vector3,'/ibvs/los_angles',           self._cb_los_angles,         10)
        self.create_subscription(Float64,'/ibvs/fov_kp_term',          self._cb_fov_kp_term,        10)
        self.create_subscription(Float64,'/ibvs/fov_kd_term',          self._cb_fov_kd_term,        10)
        self.create_subscription(Float64,'/ibvs/fov_yaw_rate',         self._cb_fov_yaw_rate,       10)
        # IBVS attitude
        self.create_subscription(Vector3,'/ibvs/attitude_rpy_deg',     self._cb_attitude_rpy,       10)
        self.create_subscription(Float64,'/ibvs/body_yaw_rate',        self._cb_body_yaw_rate,      10)
        # PNG diagnostics
        self.create_subscription(Vector3,'/png/sigma_current_deg',     self._cb_sigma_current,      10)
        self.create_subscription(Vector3,'/png/sigma_desired_deg',     self._cb_sigma_desired,      10)
        self.create_subscription(Vector3,'/png/los_rate_deg',          self._cb_los_rate,           10)
        self.create_subscription(Vector3,'/png/speed_info',            self._cb_speed_info,         10)
        # Velocity command / collision
        self.create_subscription(Twist,  '/png/velocity_cmd',          self._cb_velocity_cmd,       10)
        self.create_subscription(Bool,   '/balloon_collision',         self._cb_collision,          10)
        self.create_subscription(
            VehicleLocalPosition,
            f'drone{system_id}/fmu/out/vehicle_local_position',
            self._cb_vehicle_local_position,
            sensor_qos,
        )
        self.create_subscription(ModelStates, '/gazebo/model_states', self._cb_model_states, 10)

        # ── Timer ─────────────────────────────────────────────────────────
        self.create_timer(1.0 / log_rate, self._log_timer_cb)

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _cb_mission_state(self, msg: String):
        self._mission_state = msg.data

    def _cb_target_detected(self, msg: Bool):
        self._target_detected = msg.data

    def _cb_image_error(self, msg: Vector3):
        self._ex = msg.x
        self._ey = msg.y

    def _cb_ex_dot(self, msg: Float64):
        self._ex_dot = msg.data

    def _cb_pixel_center(self, msg: Vector3):
        self._pixel_u = msg.x
        self._pixel_v = msg.y

    def _cb_los_angles(self, msg: Vector3):
        self._q_y_rad = msg.x
        self._q_z_rad = msg.y

    def _cb_fov_kp_term(self, msg: Float64):
        self._fov_kp_term = msg.data

    def _cb_fov_kd_term(self, msg: Float64):
        self._fov_kd_term = msg.data

    def _cb_fov_yaw_rate(self, msg: Float64):
        self._fov_yaw_rate = msg.data

    def _cb_attitude_rpy(self, msg: Vector3):
        self._roll_deg  = msg.x
        self._pitch_deg = msg.y
        self._yaw_deg   = msg.z

    def _cb_body_yaw_rate(self, msg: Float64):
        self._body_yaw_rate = msg.data

    def _cb_sigma_current(self, msg: Vector3):
        self._sigma_y_cur = msg.x
        self._sigma_z_cur = msg.y

    def _cb_sigma_desired(self, msg: Vector3):
        self._sigma_y_des = msg.x
        self._sigma_z_des = msg.y

    def _cb_los_rate(self, msg: Vector3):
        self._los_rate_qy = msg.x
        self._los_rate_qz = msg.y

    def _cb_speed_info(self, msg: Vector3):
        self._png_v_now        = msg.x
        self._png_speed_actual = msg.y
        self._png_sigma_source = msg.z

    def _cb_velocity_cmd(self, msg: Twist):
        self._v_cmd_n = msg.linear.x
        self._v_cmd_e = msg.linear.y
        self._v_cmd_d = msg.linear.z

    def _cb_vehicle_local_position(self, msg: VehicleLocalPosition):
        self._px4_x  = msg.x
        self._px4_y  = msg.y
        self._px4_z  = msg.z
        self._px4_vx = msg.vx
        self._px4_vy = msg.vy
        self._px4_vz = msg.vz

    def _cb_model_states(self, msg: ModelStates):
        for i, name in enumerate(msg.name):
            if name == self._drone_model_name:
                pose = msg.pose[i]
                self._drone_gt_x = pose.position.x
                self._drone_gt_y = pose.position.y
                self._drone_gt_z = pose.position.z
            elif name == self._balloon_model_name:
                pose = msg.pose[i]
                self._balloon_x = pose.position.x
                self._balloon_y = pose.position.y
                self._balloon_z = pose.position.z + self._balloon_link_z_offset

    def _cb_collision(self, msg: Bool):
        if msg.data and not self._collision_occurred:
            self._collision_occurred = True

    # ── Timer callback: CSV recording + state transition detection ────────

    def _log_timer_cb(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        if self._mission_start_time is None:
            self._mission_start_time = now

        elapsed = now - self._mission_start_time

        # Compute dist_to_balloon_m
        if (self._drone_gt_x is not None and self._balloon_x is not None):
            drone_pos   = np.array([self._drone_gt_x, self._drone_gt_y, self._drone_gt_z])
            balloon_pos = np.array([self._balloon_x,  self._balloon_y,  self._balloon_z])
            dist = float(np.linalg.norm(drone_pos - balloon_pos))
        else:
            dist = None

        # Compute v_cmd_total
        if (self._v_cmd_n is not None
                and self._v_cmd_e is not None
                and self._v_cmd_d is not None):
            v_cmd_total = math.sqrt(self._v_cmd_n**2 + self._v_cmd_e**2 + self._v_cmd_d**2)
        else:
            v_cmd_total = None

        # LOS angles in degrees
        q_y_deg = math.degrees(self._q_y_rad) if self._q_y_rad is not None else None
        q_z_deg = math.degrees(self._q_z_rad) if self._q_z_rad is not None else None

        # ── State transition detection ────────────────────────────────────
        if self._mission_state != self._prev_mission_state:
            self._handle_state_transition(elapsed, now, dist)

        # ── Accumulate current state statistics ───────────────────────────
        state = self._mission_state
        if state is not None:
            stats = self._state_stats.setdefault(state, _new_state_stats())
            stats['tick_count'] += 1
            if dist is not None:
                stats['dist_min']  = min(stats['dist_min'], dist)
                stats['dist_max']  = max(stats['dist_max'], dist)
                stats['dist_sum'] += dist
                stats['dist_cnt'] += 1
            if self._target_detected is not None:
                stats['det_total'] += 1
                if self._target_detected:
                    stats['det_true'] += 1
            if v_cmd_total is not None:
                stats['vcmd_sum'] += v_cmd_total
                stats['vcmd_cnt'] += 1

        # ── Write CSV row ──────────────────────────────────────────────────
        def _f(val):
            return '' if val is None else val

        row = [
            # Basic state
            f'{now:.6f}',
            _f(self._mission_state),
            _f(self._target_detected),
            # Drone position (GT ENU)
            _f(self._drone_gt_x),
            _f(self._drone_gt_y),
            _f(self._drone_gt_z),
            # Drone position/velocity (PX4 NED)
            _f(self._px4_x),
            _f(self._px4_y),
            _f(self._px4_z),
            _f(self._px4_vx),
            _f(self._px4_vy),
            _f(self._px4_vz),
            # Balloon position / distance
            _f(self._balloon_x),
            _f(self._balloon_y),
            _f(self._balloon_z),
            '' if dist is None else dist,
            # IBVS image error (Eq.3)
            _f(self._ex),
            _f(self._ey),
            _f(self._ex_dot),
            _f(self._pixel_u),
            _f(self._pixel_v),
            # IBVS LOS angles (Eq.7)
            '' if q_y_deg is None else q_y_deg,
            '' if q_z_deg is None else q_z_deg,
            # IBVS FOV controller (Eq.13)
            _f(self._fov_kp_term),
            _f(self._fov_kd_term),
            _f(self._fov_yaw_rate),
            # IBVS drone attitude
            _f(self._roll_deg),
            _f(self._pitch_deg),
            _f(self._yaw_deg),
            _f(self._body_yaw_rate),
            # PNG sigma (Eq.8/9)
            _f(self._sigma_y_cur),
            _f(self._sigma_z_cur),
            _f(self._sigma_y_des),
            _f(self._sigma_z_des),
            # PNG LOS rate
            _f(self._los_rate_qy),
            _f(self._los_rate_qz),
            # PNG speed info (Eq.14)
            _f(self._png_v_now),
            _f(self._png_speed_actual),
            _f(self._png_sigma_source),
            # PNG velocity command (Eq.10)
            _f(self._v_cmd_n),
            _f(self._v_cmd_e),
            _f(self._v_cmd_d),
            '' if v_cmd_total is None else v_cmd_total,
        ]
        self._csv_writer.writerow(row)

    # ── State transition handler ───────────────────────────────────────────

    def _handle_state_transition(self, elapsed: float, now: float, dist):
        """Record state transition to event log and finalize previous state statistics."""
        prev  = self._prev_mission_state
        cur   = self._mission_state

        # Compute and accumulate previous state duration
        if prev is not None and self._state_entry_time is not None:
            duration = now - self._state_entry_time
            stats    = self._state_stats.setdefault(prev, _new_state_stats())
            stats['total_duration'] += duration

            dist_str = f'dist={dist:.2f}m' if dist is not None else 'dist=N/A'
            self._event_file.write(
                f'[{elapsed:8.2f}s] EXIT  {prev:<12s}'
                f'  (stayed {duration:.2f}s, {dist_str})\n'
            )

        # Enter new state
        if cur is not None:
            stats = self._state_stats.setdefault(cur, _new_state_stats())
            stats['visit_count'] += 1

            dist_str = f'dist={dist:.2f}m' if dist is not None else ''
            self._event_file.write(
                f'[{elapsed:8.2f}s] ENTER {cur:<12s}  {dist_str}\n'
            )

        self._prev_mission_state = cur
        self._state_entry_time   = now

    # ── Mission end summary ────────────────────────────────────────────────

    def _write_summary(self, elapsed: float):
        """Write per-state summary statistics to the event log file on mission end."""
        f = self._event_file

        # Finalize duration of the last state
        if self._mission_state is not None and self._state_entry_time is not None:
            end_now  = (self.get_clock().now().nanoseconds * 1e-9)
            duration = end_now - self._state_entry_time
            stats    = self._state_stats.setdefault(self._mission_state, _new_state_stats())
            stats['total_duration'] += duration

        # Determine outcome
        outcome = 'COLLISION (SUCCESS)' if self._collision_occurred else 'NO_COLLISION'

        # Minimum approach distance (across all states)
        global_dist_min = float('inf')
        for s in self._state_stats.values():
            if s['dist_cnt'] > 0:
                global_dist_min = min(global_dist_min, s['dist_min'])
        dist_min_str = f'{global_dist_min:.3f}m' if global_dist_min < float('inf') else 'N/A'

        f.write('\n')
        f.write('=' * 60 + '\n')
        f.write('MISSION SUMMARY\n')
        f.write('=' * 60 + '\n')
        f.write(f'Outcome       : {outcome}\n')
        f.write(f'Total duration: {elapsed:.2f}s\n')
        f.write(f'Min dist to balloon: {dist_min_str}\n')
        f.write('\n')

        # Per-state table header
        header = (
            f'{"State":<12s}  {"Duration":>9s}  {"Visits":>6s}  '
            f'{"Ticks":>6s}  {"DetRate":>8s}  {"DistMin":>8s}  '
            f'{"DistMax":>8s}  {"DistAvg":>8s}  {"VcmdAvg":>8s}'
        )
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')

        # Sort by FSM state order
        state_order = ['IDLE', 'TAKEOFF', 'FORWARD', 'INTERCEPT', 'DONE']
        all_states  = state_order + [s for s in self._state_stats if s not in state_order]

        for state in all_states:
            if state not in self._state_stats:
                continue
            s = self._state_stats[state]

            dur_str  = f'{s["total_duration"]:.2f}s'
            det_str  = (f'{100.0 * s["det_true"] / s["det_total"]:.1f}%'
                        if s['det_total'] > 0 else '  -  ')
            dmin_str = (f'{s["dist_min"]:.2f}m' if s['dist_cnt'] > 0 else '  -  ')
            dmax_str = (f'{s["dist_max"]:.2f}m' if s['dist_cnt'] > 0 else '  -  ')
            davg_str = (f'{s["dist_sum"] / s["dist_cnt"]:.2f}m'
                        if s['dist_cnt'] > 0 else '  -  ')
            vcmd_str = (f'{s["vcmd_sum"] / s["vcmd_cnt"]:.2f}m/s'
                        if s['vcmd_cnt'] > 0 else '  -  ')

            f.write(
                f'{state:<12s}  {dur_str:>9s}  {s["visit_count"]:>6d}  '
                f'{s["tick_count"]:>6d}  {det_str:>8s}  {dmin_str:>8s}  '
                f'{dmax_str:>8s}  {davg_str:>8s}  {vcmd_str:>8s}\n'
            )

        f.write('=' * 60 + '\n')
        self.get_logger().info(
            f'Mission summary written. Outcome={outcome}, min_dist={dist_min_str}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MissionLogger()
    try:
        rclpy.spin(node)
    finally:
        # Mission shutdown handling
        if node._mission_start_time is not None:
            elapsed = node.get_clock().now().nanoseconds * 1e-9 - node._mission_start_time
        else:
            elapsed = 0.0

        node._event_file.write(
            f'\n[{elapsed:8.2f}s] MISSION END  state={node._mission_state}\n'
        )
        node._write_summary(elapsed)

        node._csv_file.close()
        node._event_file.close()
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
