#!/usr/bin/env python3
"""
IBVS + PNG Interception Controller
===================================
Implementation of "Precise Interception Flight Targets by
Image-based Visual Servoing of Multicopter" (Yan et al., 2025)

This node combines:
1. Delayed Kalman Filter (DKF) for image processing delay compensation
2. Proportional Navigation Guidance (PNG) for smooth trajectory generation
3. FOV Holding Controller for stable target tracking
4. Lyapunov-based attitude controller

Replaces: position_estimator.py + drone_manager.py

Reference equations are noted as (Eq. N) matching the paper.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    Monitoring,
    VehicleAttitudeSetpoint,
)
from geometry_msgs.msg import PoseStamped
from yolov8_msgs.msg import Yolov8Inference
from enum import Enum


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────
def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)

def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def wrap_angle(a):
    """Wrap angle to [-pi, pi]"""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ──────────────────────────────────────────────
# Delayed Kalman Filter (DKF)
# ──────────────────────────────────────────────
class DelayedKalmanFilter:
    """
    Delayed Kalman Filter for image-space target position estimation.
    
    State: [u, v, u_dot, v_dot]^T  (pixel position and velocity)
    Measurement: [u, v]^T from YOLO detection (delayed)
    
    Compensates for image processing delay by predicting the current
    state from delayed measurements. Based on [28] in the paper.
    """
    def __init__(self, dt=0.02, delay_steps=3):
        """
        Args:
            dt: prediction time step (1/frequency of controller, e.g. 50Hz -> 0.02s)
            delay_steps: number of time steps of delay to compensate
                         (e.g. YOLO at 20Hz with ~50ms delay, controller at 50Hz -> ~3 steps)
        """
        self.dt = dt
        self.delay_steps = delay_steps
        
        # State: [u, v, u_dot, v_dot]
        self.x = np.zeros(4)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        
        # Process noise covariance
        q_pos = 5.0    # pixel position noise
        q_vel = 50.0   # pixel velocity noise
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        
        # Measurement noise covariance
        r_meas = 10.0  # pixel measurement noise from YOLO
        self.R = np.diag([r_meas, r_meas])
        
        # Error covariance
        self.P = np.eye(4) * 100.0
        
        self.initialized = False
    
    def predict(self):
        """Predict step: advance state by dt"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        """
        Update step with delayed measurement.
        
        Args:
            z: [u, v] measurement (pixel coordinates)
        """
        if not self.initialized:
            self.x[0] = z[0]
            self.x[1] = z[1]
            self.x[2] = 0.0
            self.x[3] = 0.0
            self.initialized = True
            return
        
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
    
    def get_current_estimate(self):
        """
        Get delay-compensated current estimate.
        Propagates state forward by delay_steps to compensate for
        image processing latency.
        
        Returns:
            (u, v, u_dot, v_dot) - estimated current pixel position and velocity
        """
        # Forward-propagate to compensate delay
        x_pred = self.x.copy()
        for _ in range(self.delay_steps):
            x_pred = self.F @ x_pred
        
        return x_pred


# ──────────────────────────────────────────────
# Mission State Machine
# ──────────────────────────────────────────────
class MissionState(Enum):
    IDLE = 0
    TAKEOFF = 1
    SEARCH = 2       # Forward flight, searching for target
    INTERCEPT = 3    # IBVS + PNG active
    DONE = 4


# ──────────────────────────────────────────────
# Main IBVS + PNG Controller Node
# ──────────────────────────────────────────────
class IBVSPNGController(Node):
    def __init__(self):
        super().__init__('ibvs_png_controller')
        
        # ── Parameters ──
        self.declare_parameter('system_id', 1)
        self.declare_parameter('takeoff_height', 6.0)     # meters (NED: negative z)
        
        # Camera intrinsics (iris_depth_camera from SDF)
        self.declare_parameter('img_width', 848)
        self.declare_parameter('img_height', 480)
        self.declare_parameter('fx', 454.8)
        self.declare_parameter('fy', 454.8)
        self.declare_parameter('cx', 424.0)               # image center u
        self.declare_parameter('cy', 240.0)               # image center v
        self.declare_parameter('cam_pitch_deg', 0.0)       # camera pitch (degrees)
        
        # PNG parameters (Eq. 9)
        self.declare_parameter('K_y', 3.0)                 # PNG constant vertical
        self.declare_parameter('K_z', 3.0)                 # PNG constant horizontal
        self.declare_parameter('k_a', 2.0)                 # velocity gain (Eq. 14)
        
        # FOV Holding / Yaw PD controller (Eq. 13)
        self.declare_parameter('kp_yaw', 0.03)
        self.declare_parameter('kd_yaw', 0.01)
        
        # Speed limits
        self.declare_parameter('max_speed', 10.0)          # m/s
        self.declare_parameter('search_speed', 3.0)        # m/s during search
        self.declare_parameter('collision_distance', 0.5)   # m, mission done threshold
        
        # DKF parameters
        self.declare_parameter('dkf_dt', 0.02)             # 50Hz controller rate
        self.declare_parameter('dkf_delay_steps', 3)       # ~60ms delay compensation
        
        # Topics
        self.declare_parameter('detection_topic', '/Yolov8_Inference_1')
        self.declare_parameter('monitoring_topic', '/drone1/fmu/out/monitoring')
        
        # ── Get parameters ──
        self.system_id = self.get_parameter('system_id').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.img_w = self.get_parameter('img_width').value
        self.img_h = self.get_parameter('img_height').value
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.cam_pitch = math.radians(self.get_parameter('cam_pitch_deg').value)
        self.foc = self.fx  # focal length in pixels (used in LOS computation, Eq. 5)
        
        self.Ky = self.get_parameter('K_y').value
        self.Kz = self.get_parameter('K_z').value
        self.ka = self.get_parameter('k_a').value
        self.kp_yaw = self.get_parameter('kp_yaw').value
        self.kd_yaw = self.get_parameter('kd_yaw').value
        self.max_speed = self.get_parameter('max_speed').value
        self.search_speed = self.get_parameter('search_speed').value
        self.collision_dist = self.get_parameter('collision_distance').value
        
        dkf_dt = self.get_parameter('dkf_dt').value
        dkf_delay = self.get_parameter('dkf_delay_steps').value
        
        self.detection_topic = self.get_parameter('detection_topic').value
        self.monitoring_topic = self.get_parameter('monitoring_topic').value
        self.topic_prefix = f"drone{self.system_id}/fmu/"
        
        # ── Camera frame transform ──
        # Camera (ROS/OpenCV): X=right, Y=down, Z=forward
        # Body (FRD): X=forward, Y=right, Z=down
        self.R_b_c = np.array([
            [0, 0, 1],   # body_x = cam_z (forward)
            [1, 0, 0],   # body_y = cam_x (right)
            [0, 1, 0],   # body_z = cam_y (down)
        ], dtype=float)
        # Apply camera pitch offset
        self.R_b_c = self.R_b_c @ rot_x(-self.cam_pitch)
        
        # ── State variables ──
        self.state = MissionState.IDLE
        self.drone_pos = np.zeros(3)      # NED position [m]
        self.drone_vel = np.zeros(3)      # NED velocity [m/s]
        self.drone_yaw = 0.0              # yaw [rad]
        self.drone_pitch = 0.0            # pitch [rad]
        self.drone_roll = 0.0             # roll [rad]
        self.nav_state = 0
        self.arming_state = 0
        self.last_cmd_time = 0.0
        self.search_start_pos = None
        self.search_distance_limit = 15.0  # meters
        
        # PNG state (previous step values for discrete integration, Eq. 9)
        self.prev_qy = None               # previous LOS angle vertical
        self.prev_qz = None               # previous LOS angle horizontal
        self.prev_sigma_y = None           # previous velocity angle vertical
        self.prev_sigma_z = None           # previous velocity angle horizontal
        
        # Yaw PD state
        self.prev_ex = 0.0                # previous image error for derivative
        
        # Target tracking
        self.target_detected = False
        self.target_lost_count = 0
        self.target_lost_threshold = 50   # ~1 second at 50Hz
        
        # ── DKF ──
        self.dkf = DelayedKalmanFilter(dt=dkf_dt, delay_steps=dkf_delay)
        
        # ── QoS ──
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # ── Publishers ──
        self.ocm_pub = self.create_publisher(
            OffboardControlMode,
            f'{self.topic_prefix}in/offboard_control_mode',
            qos_profile_sensor_data
        )
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint,
            f'{self.topic_prefix}in/trajectory_setpoint',
            qos_profile_sensor_data
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            f'{self.topic_prefix}in/vehicle_command',
            qos_profile_sensor_data
        )
        # Debug: publish estimated target position for visualization
        self.target_pos_pub = self.create_publisher(
            PoseStamped,
            '/ibvs_target_position',
            10
        )
        
        # ── Subscribers ──
        self.create_subscription(
            VehicleStatus,
            f'{self.topic_prefix}out/vehicle_status',
            self.status_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Monitoring,
            self.monitoring_topic,
            self.monitoring_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Yolov8Inference,
            self.detection_topic,
            self.detection_cb,
            10
        )
        
        # ── Timers ──
        self.create_timer(0.1, self.ocm_timer_cb)       # 10Hz offboard heartbeat
        self.create_timer(0.02, self.control_timer_cb)   # 50Hz main control loop
        self.start_timer = self.create_timer(5.0, self.start_mission)
        
        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info('  IBVS + PNG Interception Controller')
        self.get_logger().info(f'  Ky={self.Ky}, Kz={self.Kz}, ka={self.ka}')
        self.get_logger().info(f'  kp={self.kp_yaw}, kd={self.kd_yaw}')
        self.get_logger().info(f'  Camera: {self.img_w}x{self.img_h}, fx={self.fx}')
        self.get_logger().info('═══════════════════════════════════════')
    
    # ──────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────
    def status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state
    
    def monitoring_cb(self, msg: Monitoring):
        self.drone_pos = np.array([msg.pos_x, msg.pos_y, msg.pos_z])
        self.drone_vel = np.array([msg.vel_x, msg.vel_y, msg.vel_z]) if hasattr(msg, 'vel_x') else self.drone_vel
        self.drone_yaw = msg.head
        self.drone_pitch = msg.pitch if hasattr(msg, 'pitch') else 0.0
        self.drone_roll = msg.roll if hasattr(msg, 'roll') else 0.0
    
    def detection_cb(self, msg: Yolov8Inference):
        """Process YOLO detections → feed into DKF"""
        if not msg.yolov8_inference:
            return
        
        det = msg.yolov8_inference[0]
        u = (det.left + det.right) * 0.5
        v = (det.top + det.bottom) * 0.5  # Use center of bbox (not bottom)
        
        # Feed measurement to DKF
        self.dkf.update(np.array([u, v]))
        self.target_detected = True
        self.target_lost_count = 0
        
        self.get_logger().info(
            f'[DET] bbox_center=({u:.0f},{v:.0f})',
            throttle_duration_sec=1.0
        )
    
    def start_mission(self):
        self.start_timer.cancel()
        if self.state == MissionState.IDLE:
            self.get_logger().info('Mission start → TAKEOFF')
            self.state = MissionState.TAKEOFF
    
    # ──────────────────────────────────────────
    # Offboard heartbeat
    # ──────────────────────────────────────────
    def ocm_timer_cb(self):
        msg = OffboardControlMode()
        # During INTERCEPT we use velocity control for PNG
        if self.state == MissionState.INTERCEPT:
            msg.position = False
            msg.velocity = True
        else:
            msg.position = True
            msg.velocity = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.ocm_pub.publish(msg)
    
    # ──────────────────────────────────────────
    # Main control loop (50Hz)
    # ──────────────────────────────────────────
    def control_timer_cb(self):
        # DKF predict step every cycle (regardless of new measurement)
        if self.dkf.initialized:
            self.dkf.predict()
        
        # Track target loss
        if self.target_detected:
            pass  # reset happens in detection_cb
        else:
            self.target_lost_count += 1
        
        # State machine
        if self.state == MissionState.IDLE:
            self._idle()
        elif self.state == MissionState.TAKEOFF:
            self._takeoff()
        elif self.state == MissionState.SEARCH:
            self._search()
        elif self.state == MissionState.INTERCEPT:
            self._intercept()
        elif self.state == MissionState.DONE:
            self._done()
    
    # ──────────────────────────────────────────
    # State handlers
    # ──────────────────────────────────────────
    def _idle(self):
        safe_z = max(self.drone_pos[2], -0.1)
        self._pub_position([self.drone_pos[0], self.drone_pos[1], safe_z])
    
    def _takeoff(self):
        target_alt = -self.takeoff_height
        now = self.get_clock().now().nanoseconds / 1e9
        
        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED or \
           self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            safe_z = max(self.drone_pos[2], -0.1)
            self._pub_position([self.drone_pos[0], self.drone_pos[1], safe_z])
        else:
            self._pub_position([self.drone_pos[0], self.drone_pos[1], target_alt])
        
        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if now - self.last_cmd_time > 1.0:
                self._pub_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.get_logger().info('ARM requested')
                self.last_cmd_time = now
            return
        
        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if now - self.last_cmd_time > 1.0:
                self._pub_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.get_logger().info('OFFBOARD requested')
                self.last_cmd_time = now
            return
        
        if abs(self.drone_pos[2] - target_alt) < 0.3:
            self.get_logger().info('Takeoff complete → SEARCH')
            self.search_start_pos = self.drone_pos.copy()
            self.state = MissionState.SEARCH
    
    def _search(self):
        """Fly forward until target is detected"""
        if self.search_start_pos is None:
            self.search_start_pos = self.drone_pos.copy()
        
        # Check if target found
        if self.target_detected and self.dkf.initialized:
            self.get_logger().info('Target acquired → INTERCEPT')
            self._init_png_state()
            self.state = MissionState.INTERCEPT
            return
        
        # Distance limit
        dist = np.linalg.norm(self.drone_pos[:2] - self.search_start_pos[:2])
        if dist >= self.search_distance_limit:
            self._pub_position([self.drone_pos[0], self.drone_pos[1], -self.takeoff_height])
            self.get_logger().info('Search limit reached, hovering', throttle_duration_sec=3.0)
            return
        
        # Forward flight (same as original: fly to x=5, y=0)
        self._pub_position([5.0, 0.0, -self.takeoff_height])
    
    def _intercept(self):
        """
        Core IBVS + PNG controller.
        
        Pipeline:
        1. Get DKF-compensated image coordinates
        2. Compute image error (Eq. 3)
        3. Compute LOS direction and angles (Eq. 5, 7)
        4. Compute velocity angles (Eq. 8)
        5. Apply PNG to get desired velocity direction (Eq. 9-10)
        6. Apply FOV holding for speed and yaw (Eq. 13-14)
        7. Output velocity command
        """
        # ── Check target status ──
        if self.target_lost_count > self.target_lost_threshold:
            self.get_logger().warn('Target lost! Returning to SEARCH')
            self.target_detected = False
            self.prev_qy = None
            self.prev_qz = None
            self.state = MissionState.SEARCH
            return
        
        if not self.dkf.initialized:
            self._pub_velocity([0.0, 0.0, 0.0], self.drone_yaw)
            return
        
        # ── Step 1: DKF estimate (delay-compensated) ──
        est = self.dkf.get_current_estimate()
        u_est, v_est = est[0], est[1]
        u_dot_est, v_dot_est = est[2], est[3]
        
        # ── Step 2: Image error (Eq. 3) ──
        # e = i_p - i_pc  (target pixel - image center)
        ex = u_est - self.cx   # horizontal error
        ey = v_est - self.cy   # vertical error
        
        # Normalized image error (Eq. 4 notation)
        ex_bar = ex / self.foc
        ey_bar = ey / self.foc
        
        # ── Step 3: LOS direction vector (Eq. 5) ──
        # n_t = R_e_b * R_b_c * [ex, ey, foc]^T / ||...||
        ray_cam = np.array([ex, ey, self.foc])
        ray_body = self.R_b_c @ ray_cam
        
        # Body to NED rotation (yaw + pitch)
        R_e_b = rot_z(self.drone_yaw) @ rot_y(self.drone_pitch)
        ray_ned = R_e_b @ ray_body
        nt = normalize(ray_ned)  # LOS unit vector in NED
        
        # ── Step 4: LOS angles (Eq. 7) ──
        # qy = arctan(ntz / sqrt(ntx^2 + nty^2))  (vertical)
        # qz = arctan(ntx / nty)                    (horizontal)
        nt_xy = math.sqrt(nt[0]**2 + nt[1]**2)
        qy = math.atan2(nt[2], nt_xy) if nt_xy > 1e-9 else 0.0
        qz = math.atan2(nt[0], nt[1]) if abs(nt[1]) > 1e-9 else 0.0
        
        # ── Step 5: Velocity direction angles (Eq. 8) ──
        speed = np.linalg.norm(self.drone_vel)
        if speed > 0.5:
            nv = normalize(self.drone_vel)
            nv_xy = math.sqrt(nv[0]**2 + nv[1]**2)
            sigma_y = math.atan2(nv[2], nv_xy) if nv_xy > 1e-9 else 0.0
            sigma_z = math.atan2(nv[0], nv[1]) if abs(nv[1]) > 1e-9 else 0.0
        else:
            # Low speed: use LOS direction as initial velocity direction
            sigma_y = qy
            sigma_z = qz
        
        # ── Step 6: PNG law (Eq. 9) ──
        if self.prev_qy is not None and self.prev_qz is not None:
            delta_qy = wrap_angle(qy - self.prev_qy)
            delta_qz = wrap_angle(qz - self.prev_qz)
            
            sigma_yd = self.Ky * delta_qy + self.prev_sigma_y
            sigma_zd = self.Kz * delta_qz + self.prev_sigma_z
        else:
            # First iteration: desired velocity direction = LOS direction
            sigma_yd = qy
            sigma_zd = qz
        
        # Store for next iteration
        self.prev_qy = qy
        self.prev_qz = qz
        self.prev_sigma_y = sigma_yd
        self.prev_sigma_z = sigma_zd
        
        # ── Step 7: Desired velocity direction (Eq. 10) ──
        # nvd = [cos(σyd)*sin(σzd), cos(σyd)*cos(σzd), sin(σyd)]
        cos_sy = math.cos(sigma_yd)
        nvd = np.array([
            cos_sy * math.sin(sigma_zd),
            cos_sy * math.cos(sigma_zd),
            math.sin(sigma_yd)
        ])
        nvd = normalize(nvd)
        
        # ── Step 8: Desired speed magnitude (Eq. 14, FOV holding) ──
        # vd = v_now + ka
        vd_mag = speed + self.ka
        vd_mag = clamp(vd_mag, 1.0, self.max_speed)
        
        # Desired velocity vector
        vd = vd_mag * nvd
        
        # ── Step 9: Yaw control via PD (Eq. 13, FOV holding) ──
        # bw_psi = kp * ex + kd * ex_dot
        ex_dot = (ex - self.prev_ex) / 0.02  # finite difference at 50Hz
        yaw_rate_cmd = self.kp_yaw * ex + self.kd_yaw * ex_dot
        self.prev_ex = ex
        
        # Convert yaw rate to desired yaw
        desired_yaw = self.drone_yaw + yaw_rate_cmd * 0.02
        desired_yaw = wrap_angle(desired_yaw)
        
        # ── Publish velocity command ──
        self._pub_velocity(vd, desired_yaw)
        
        # ── Debug logging ──
        self.get_logger().info(
            f'[IBVS] e=({ex:.0f},{ey:.0f}) q=({math.degrees(qy):.1f}°,{math.degrees(qz):.1f}°) '
            f'v={speed:.1f}m/s vd={vd_mag:.1f}m/s',
            throttle_duration_sec=0.5
        )
        
        # ── Publish debug target position (ray-based estimate for viz) ──
        self._publish_debug_target(nt)
    
    def _done(self):
        self._pub_position(self.drone_pos.tolist(), yaw=self.drone_yaw)
        self.get_logger().info('Mission DONE, hovering.', throttle_duration_sec=5.0)
    
    # ──────────────────────────────────────────
    # PNG state initialization
    # ──────────────────────────────────────────
    def _init_png_state(self):
        """Initialize PNG state when entering INTERCEPT mode"""
        self.prev_qy = None
        self.prev_qz = None
        self.prev_sigma_y = None
        self.prev_sigma_z = None
        self.prev_ex = 0.0
        self.target_lost_count = 0
        self.get_logger().info('PNG state initialized')
    
    # ──────────────────────────────────────────
    # Debug visualization
    # ──────────────────────────────────────────
    def _publish_debug_target(self, nt):
        """
        Publish estimated target position for RViz visualization.
        This is a rough estimate using a fixed assumed distance.
        The actual controller does NOT use this 3D position.
        """
        # Assume target is ~10m away along LOS (for visualization only)
        assumed_dist = 10.0
        target_est = self.drone_pos + assumed_dist * nt
        
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(target_est[0])
        msg.pose.position.y = float(target_est[1])
        msg.pose.position.z = float(target_est[2])
        self.target_pos_pub.publish(msg)
    
    # ──────────────────────────────────────────
    # PX4 command helpers
    # ──────────────────────────────────────────
    def _pub_position(self, pos, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pub.publish(msg)
    
    def _pub_velocity(self, vel, yaw=0.0):
        """Publish velocity setpoint in NED frame"""
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')]  # NaN = don't control position
        msg.velocity = [float(vel[0]), float(vel[1]), float(vel[2])]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pub.publish(msg)
    
    def _pub_command(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.param1 = p1
        msg.param2 = p2
        msg.command = command
        msg.target_system = self.system_id
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = IBVSPNGController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()