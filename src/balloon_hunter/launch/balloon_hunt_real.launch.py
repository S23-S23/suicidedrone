#!/usr/bin/env python3
"""
Launch file for real-world IBVS + PNG balloon tracking.

Architecture (same as simulation, minus Gazebo/SITL):
  balloon_detector (YOLO)   -> /target_info
  filter_node (DKF/EKF)     -> /filter_estimate
  ibvs_controller           -> /ibvs/output   (subscribes /filter_estimate)
  png_guidance              -> /png/guidance_cmd
  drone_manager_real (FSM)  -> PX4 setpoints

Flow:
  1. Drone flies under RC control
  2. User runs: ros2 launch balloon_hunter balloon_hunt_real.launch.py
  3. drone_manager captures current position, requests OFFBOARD
  4. Hovers 2s for filter initialization
  5. Starts IBVS+PNG tracking
  6. User takes back control via RC mode switch

Usage:
  ros2 launch balloon_hunter balloon_hunt_real.launch.py
  ros2 launch balloon_hunter balloon_hunt_real.launch.py camera_topic:=/camera/image_raw
  ros2 launch balloon_hunter balloon_hunt_real.launch.py filter_type:=EKF v_max:=8.0
"""
import os
from datetime import datetime
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, TimerAction,
)


def launch_setup(context, *args, **kwargs):
    # ── Resolve launch arguments ──
    system_id      = int(LaunchConfiguration('system_id').perform(context))
    camera_topic   = LaunchConfiguration('camera_topic').perform(context)
    model_path     = LaunchConfiguration('model_path').perform(context)
    filter_type    = LaunchConfiguration('filter_type').perform(context)
    target_class   = LaunchConfiguration('target_class').perform(context)
    conf           = float(LaunchConfiguration('conf').perform(context))

    # Camera intrinsics
    fx = float(LaunchConfiguration('fx').perform(context))
    fy = float(LaunchConfiguration('fy').perform(context))
    cx = float(LaunchConfiguration('cx').perform(context))
    cy = float(LaunchConfiguration('cy').perform(context))
    cam_pitch_deg = float(LaunchConfiguration('cam_pitch_deg').perform(context))

    # IBVS gains
    fov_kp   = float(LaunchConfiguration('fov_kp').perform(context))
    fov_kd   = float(LaunchConfiguration('fov_kd').perform(context))
    fov_kp_z = float(LaunchConfiguration('fov_kp_z').perform(context))
    fov_kd_z = float(LaunchConfiguration('fov_kd_z').perform(context))

    # PNG parameters
    Ky     = float(LaunchConfiguration('Ky').perform(context))
    Kz     = float(LaunchConfiguration('Kz').perform(context))
    ka     = float(LaunchConfiguration('ka').perform(context))
    v_max  = float(LaunchConfiguration('v_max').perform(context))
    v_init = float(LaunchConfiguration('v_init').perform(context))
    rate   = float(LaunchConfiguration('rate').perform(context))

    # Drone manager
    hover_init_duration = float(LaunchConfiguration('hover_init_duration').perform(context))

    # Collision-triggered landing
    collision_area_frac = float(LaunchConfiguration('collision_area_frac').perform(context))
    collision_lost_time = float(LaunchConfiguration('collision_lost_time').perform(context))
    collision_min_edges = int(LaunchConfiguration('collision_min_edges').perform(context))
    land_disarm_timeout = float(LaunchConfiguration('land_disarm_timeout').perform(context))

    # ── 1. YOLO Detector ──
    balloon_detector_node = Node(
        package='balloon_hunter',
        executable='balloon_detector',
        name='balloon_detector',
        output='screen',
        parameters=[{
            'system_id': system_id,
            'camera_topic': camera_topic,
            'model_path': model_path,
            'conf': conf,
            'target_class': target_class,
        }],
    )

    # ── 2. IBVS Controller ──
    ibvs_controller_node = Node(
        package='balloon_hunter',
        executable='ibvs_controller',
        name='ibvs_controller',
        output='screen',
        parameters=[{
            'system_id': system_id,
            'fx': fx, 'fy': fy,
            'cx': cx, 'cy': cy,
            'fov_kp': fov_kp, 'fov_kd': fov_kd,
            'fov_kp_z': fov_kp_z, 'fov_kd_z': fov_kd_z,
            'target_timeout': 0.5,
            'cam_pitch_deg': cam_pitch_deg,
        }],
    )

    # ── 3. PNG Guidance ──
    png_guidance_node = Node(
        package='balloon_hunter',
        executable='png_guidance',
        name='png_guidance',
        output='screen',
        parameters=[{
            'system_id': system_id,
            'Ky': Ky, 'Kz': Kz,
            'ka': ka,
            'v_max': v_max, 'v_init': v_init,
            'rate': rate,
            'v_min_sigma': 0.5,
        }],
    )

    # ── 4. Drone Manager (Real Flight FSM) ──
    drone_manager_node = Node(
        package='balloon_hunter',
        executable='drone_manager_real',
        name='drone_manager',
        output='screen',
        parameters=[{
            'system_id': system_id,
            'hover_init_duration': hover_init_duration,
            'max_speed': v_max,
            # 직충돌 감지 -> 자동 착륙/DISARM
            'collision_area_frac': collision_area_frac,
            'collision_lost_time': collision_lost_time,
            'collision_min_edges': collision_min_edges,
            'image_width': 1280,
            'image_height': 720,
            'enable_collision_land': True,
            'land_disarm_timeout': land_disarm_timeout,
        }],
    )

    # ── 5. Filter Node ──
    filter_node = Node(
        package='balloon_hunter',
        executable='filter_node',
        name='filter_node',
        output='screen',
        parameters=[{
            'system_id': system_id,
            'filter_type': filter_type,
            'fx': fx, 'fy': fy,
            'cx': cx, 'cy': cy,
            'cam_pitch_deg': cam_pitch_deg,
            'dkf_dt': 0.02,
            'dkf_delay_steps': 3,
            'assumed_depth': 10.0,
        }],
    )

    # ── 6. Logger ──
    logger_node = Node(
        package='balloon_hunter',
        executable='logger',
        name='logger',
        output='screen',
        parameters=[{
            'filter_type': filter_type,
            'system_id': system_id,
            'fx': fx, 'fy': fy,
            'cx': cx, 'cy': cy,
            'cam_pitch_deg': cam_pitch_deg,
        }],
    )

    # ── 7. Drone Visualizer (no Gazebo) ──
    drone_visualizer_node = Node(
        package='balloon_hunter',
        executable='drone_visualizer_real',
        name='drone_visualizer',
        output='screen',
        parameters=[{
            'system_id': system_id,
            'max_path_points': 5000,
        }],
    )

    # ── 8. Rosbag Record (이미지 제외, 제어/상태 토픽만) ──
    # 홈 디렉토리 기준 -> 사용자/머신 바뀌어도 동작 (BAG_DIR 환경변수로 덮어쓰기 가능)
    bag_dir = os.environ.get('BAG_DIR', os.path.expanduser('~/suicidedrone_log'))
    os.makedirs(bag_dir, exist_ok=True)
    bag_name = f'rosbag_{datetime.now().strftime("%Y%m%d_%H%M%S")}_drone{system_id}'
    rosbag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-o', os.path.join(bag_dir, bag_name),
            # 감지 / 필터 / 제어
            '/target_info',
            '/filter_estimate',
            '/ibvs/output',
            '/png/guidance_cmd',
            '/mission_state',
            '/collision_info',        # [last_frac, peak_frac, peak_edges, lost_time, latched]
            'inference_result_2',
            # PX4 출력
            f'drone{system_id}/fmu/out/monitoring',
            f'drone{system_id}/fmu/out/vehicle_local_position',
            f'drone{system_id}/fmu/out/vehicle_angular_velocity',
            f'drone{system_id}/fmu/out/vehicle_attitude',
            f'drone{system_id}/fmu/out/vehicle_status_v1',
            f'drone{system_id}/fmu/out/vehicle_acceleration',
            f'drone{system_id}/fmu/out/vehicle_land_detected',  # 착지 확인 (DISARM 트리거)
            # PX4 입력
            f'drone{system_id}/fmu/in/trajectory_setpoint',
            f'drone{system_id}/fmu/in/offboard_control_mode',
            f'drone{system_id}/fmu/in/vehicle_command',         # OFFBOARD/LAND/DISARM 명령 기록
        ],
        output='screen',
    )

    # Start mission nodes after a short delay for XRCE to initialize
    mission_nodes = TimerAction(
        period=3.0,
        actions=[
            rosbag_record,
            balloon_detector_node,
            ibvs_controller_node,
            png_guidance_node,
            filter_node,
            drone_manager_node,
            logger_node,
            #drone_visualizer_node,
        ],
    )

    return [
        mission_nodes,
    ]


def generate_launch_description():
    return LaunchDescription([
        # ── System ──
        DeclareLaunchArgument('system_id', default_value='2'), #edittt

        # ── Camera ──
        DeclareLaunchArgument('camera_topic', default_value='/camera/camera/color/image_raw',
                              description='Camera image topic (RealSense color stream)'),
        DeclareLaunchArgument('model_path',
                              default_value='/home/suvlab/ros2_ws/suicidedrone/src/balloon_hunter/models/balloon_yolov8n.pt',
                              description='YOLO model weights absolute path'),
        DeclareLaunchArgument('target_class', default_value='red-balloon',
                              description='YOLO class name to detect (substring match)'),
        DeclareLaunchArgument('conf', default_value='0.5',
                              description='YOLO confidence threshold'),

        # ── Camera Intrinsics (RealSense D455 color, 1280x720, from camera_info) ──
        DeclareLaunchArgument('fx', default_value='643.2935'),
        DeclareLaunchArgument('fy', default_value='642.6299'),
        DeclareLaunchArgument('cx', default_value='644.1669'),
        DeclareLaunchArgument('cy', default_value='355.8758'),
        DeclareLaunchArgument('cam_pitch_deg', default_value='45.0',
                              description='Camera mount pitch [deg], positive = tilted down from forward'),

        # ── IBVS Gains ──
        DeclareLaunchArgument('fov_kp',   default_value='0.5'),
        DeclareLaunchArgument('fov_kd',   default_value='0.05'),
        DeclareLaunchArgument('fov_kp_z', default_value='0.5'),
        DeclareLaunchArgument('fov_kd_z', default_value='0.05'),

        # ── PNG Parameters ──
        DeclareLaunchArgument('Ky', default_value='3.0'),
        DeclareLaunchArgument('Kz', default_value='3.0'),
        DeclareLaunchArgument('ka', default_value='0.5'),
        DeclareLaunchArgument('v_max', default_value='0.7',
                              description='Max intercept speed [m/s]'),
        DeclareLaunchArgument('v_init', default_value='0.5',
                              description='Initial intercept speed [m/s]'),
        DeclareLaunchArgument('rate', default_value='50.0',
                              description='Guidance loop rate [Hz]'),

        # ── Drone Manager ──
        DeclareLaunchArgument('hover_init_duration', default_value='5.0',
                              description='Hover time for filter initialization [s]'),

        # ── Collision-triggered Landing ──
        DeclareLaunchArgument('collision_area_frac', default_value='0.10',
                              description='bbox 면적비율(화면 대비) 임계. 너무 늦게 착륙하면 낮추고, 너무 일찍이면 높임'),
        DeclareLaunchArgument('collision_lost_time', default_value='1.0',
                              description='임계 초과 후 미검출 지속시간[s]. 오검출(조기착륙) 잦으면 키움'),
        DeclareLaunchArgument('collision_min_edges', default_value='0',
                              description='peak 프레임 경계접촉 변 수 하한(0=미사용, 1~2=더 엄격)'),
        DeclareLaunchArgument('land_disarm_timeout', default_value='5.0',
                              description='AUTO.LAND 후 landed 미수신 시 강제 DISARM까지 시간[s] (<=0=비활성)'),

        # ── Filter ──
        DeclareLaunchArgument('filter_type', default_value='DKF',
                              description='Filter: DKF (full 18-state, paper-faithful) or EKF (same model, no delay replay)'),

        OpaqueFunction(function=launch_setup),
    ])
