#!/usr/bin/env python3
"""
Launch file for real-world IBVS + PNG balloon tracking.

Architecture (same as simulation, minus Gazebo/SITL):
  balloon_detector (YOLO)   -> /target_info
  ibvs_controller           -> /ibvs/output
  png_guidance              -> /png/guidance_cmd
  drone_manager_real (FSM)  -> PX4 setpoints
  filter_node (DKF/EKF)     -> /filter_estimate (for logger)

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
from ament_index_python.packages import get_package_share_directory
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

    # XRCE-DDS
    xrce_port = LaunchConfiguration('xrce_port').perform(context)

    # ── MicroXRCE-DDS Agent ──
    xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', xrce_port],
        output='screen',
    )

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
            'cam_pitch_deg': 0.0,
            'dkf_dt': 0.02,
            'dkf_delay_steps': 2,
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
            'cam_pitch_deg': 0.0,
            'collision_distance': 999.0,  # effectively disable collision auto-shutdown
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

    # ── 8. RViz2 (optional) ──
    current_package_path = get_package_share_directory('balloon_hunter')
    rviz_config = os.path.join(current_package_path, 'config', 'drone_trajectory.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    # ── 9. Rosbag Record ──
    bag_dir = os.path.expanduser('~/dkf_logs')
    bag_name = f'rosbag_{datetime.now().strftime("%Y%m%d_%H%M%S")}_drone{system_id}'
    rosbag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-o', os.path.join(bag_dir, bag_name),
            # 이미지
            camera_topic,
            f'/inference_result_{system_id}',
            # 감지 / 필터 / 제어
            '/target_info',
            '/filter_estimate',
            '/ibvs/output',
            '/png/guidance_cmd',
            '/mission_state',
            # PX4 출력
            f'drone{system_id}/fmu/out/monitoring',
            f'drone{system_id}/fmu/out/vehicle_local_position',
            f'drone{system_id}/fmu/out/vehicle_angular_velocity',
            f'drone{system_id}/fmu/out/vehicle_attitude',
            f'drone{system_id}/fmu/out/vehicle_status',
            # PX4 입력
            f'drone{system_id}/fmu/in/trajectory_setpoint',
            f'drone{system_id}/fmu/in/offboard_control_mode',
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
            drone_visualizer_node,
        ],
    )

    return [
        xrce_agent,
        mission_nodes,
        rviz_node,
    ]


def generate_launch_description():
    return LaunchDescription([
        # ── System ──
        DeclareLaunchArgument('system_id', default_value='1'),
        DeclareLaunchArgument('xrce_port', default_value='8888',
                              description='MicroXRCE-DDS agent UDP port'),

        # ── Camera ──
        DeclareLaunchArgument('camera_topic', default_value='/drone1/camera/image_raw',
                              description='Camera image topic name'),
        DeclareLaunchArgument('model_path',
                              default_value='/home/kiki/suicidedrone/src/balloon_hunter/models/yolov8n.pt',
                              description='YOLO model weights path'),
        DeclareLaunchArgument('target_class', default_value='sports ball',
                              description='YOLO class name to detect'),
        DeclareLaunchArgument('conf', default_value='0.5',
                              description='YOLO confidence threshold'),

        # ── Camera Intrinsics (calibrate for your real camera!) ──
        DeclareLaunchArgument('fx', default_value='454.8'),
        DeclareLaunchArgument('fy', default_value='454.8'),
        DeclareLaunchArgument('cx', default_value='424.0'),
        DeclareLaunchArgument('cy', default_value='240.0'),

        # ── IBVS Gains ──
        DeclareLaunchArgument('fov_kp',   default_value='1.5'),
        DeclareLaunchArgument('fov_kd',   default_value='0.1'),
        DeclareLaunchArgument('fov_kp_z', default_value='1.5'),
        DeclareLaunchArgument('fov_kd_z', default_value='0.1'),

        # ── PNG Parameters ──
        DeclareLaunchArgument('Ky', default_value='3.0'),
        DeclareLaunchArgument('Kz', default_value='3.0'),
        DeclareLaunchArgument('ka', default_value='2.0'),
        DeclareLaunchArgument('v_max', default_value='3.0',
                              description='Max intercept speed [m/s]'),
        DeclareLaunchArgument('v_init', default_value='1.5',
                              description='Initial intercept speed [m/s]'),
        DeclareLaunchArgument('rate', default_value='50.0',
                              description='Guidance loop rate [Hz]'),

        # ── Drone Manager ──
        DeclareLaunchArgument('hover_init_duration', default_value='2.0',
                              description='Hover time for filter initialization [s]'),

        # ── Filter ──
        DeclareLaunchArgument('filter_type', default_value='DKF18',
                              description='Filter: DKF, EKF, DKF18, EKF18'),

        OpaqueFunction(function=launch_setup),
    ])
