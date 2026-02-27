#!/usr/bin/env python3
"""
Launch file for IBVS + PNG balloon interception.
Replaces: position_estimator + drone_manager with single ibvs_png_controll3r node.
Keeps: balloon_detector (YOLO) as-is.

Usage:
  ros2 launch balloon_hunter balloon_hunt_ibvs.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import (
    ExecuteProcess, DeclareLaunchArgument, OpaqueFunction,
    IncludeLaunchDescription, SetEnvironmentVariable, TimerAction,
    RegisterEventHandler,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit


def launch_setup(context, *args, **kwargs):
    current_package_path = get_package_share_directory('balloon_hunter')
    px4_src_path = LaunchConfiguration('px4_src_path').perform(context)
    gazebo_classic_path = f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic'
    model_path = LaunchConfiguration('model_path').perform(context)
    drone_id = 1

    # Environment
    resource_path_env = SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', '/usr/share/gazebo-11')
    model_path_env = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:'
        f'{current_package_path}/models:{gazebo_classic_path}/models'
    )
    plugin_path_env = SetEnvironmentVariable(
        'GAZEBO_PLUGIN_PATH',
        f'{px4_src_path}/build/px4_sitl_default/build_gazebo-classic/'
    )
    gz_ip_env = SetEnvironmentVariable('GZ_IP', '127.0.0.1')

    # Infrastructure
    xrce_agent_process = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'], output='screen'
    )

    world_file_path = os.path.join(current_package_path, 'worlds', 'balloon_hunt.world')
    gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch'),
            '/gazebo.launch.py'
        ]),
        launch_arguments={'world': world_file_path, 'verbose': 'false', 'gui': 'true'}.items()
    )

    # SDF generation
    def create_drone_sdf_cmd(idx, tcp_port, udp_port, model_name):
        template_file = os.path.join(current_package_path, 'models', model_name, f'{model_name}.sdf.jinja')
        return ExecuteProcess(
            cmd=[
                f'{gazebo_classic_path}/scripts/jinja_gen.py',
                template_file, current_package_path,
                '--mavlink_tcp_port', str(tcp_port),
                '--mavlink_udp_port', str(udp_port),
                '--mavlink_id', str(idx),
                '--output-file', f'/tmp/drone_{idx}.sdf'
            ],
            output='screen'
        )

    gen_sdf_drone1 = create_drone_sdf_cmd(drone_id, 4560, 14560, 'iris_depth_camera')

    # Spawn
    spawn_drone1 = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=[
            '-file', f'/tmp/drone_{drone_id}.sdf',
            '-entity', f'drone{drone_id}',
            '-x', '0.0', '-y', '0.0', '-z', '0.1', '-Y', '1.5708',
            '-robot_namespace', f'drone{drone_id}'
        ],
        output='screen'
    )

    # PX4 SITL
    px4_process_1 = ExecuteProcess(
        cmd=[
            f'{px4_src_path}/build/px4_sitl_default/bin/px4',
            '-i', '0', '-d', f'{px4_src_path}/build/px4_sitl_default/etc',
            '-w', f'{px4_src_path}/build/px4_sitl_default/ROMFS/instance0'
        ],
        additional_env={
            'PX4_SIM_MODEL': 'gazebo-classic_iris',
            'PX4_UXRCE_DDS_NS': f'drone{drone_id}',
            'PX4_UXRCE_DDS_PORT': '8888',
            'PX4_SYS_ID': str(drone_id),
            'PX4_SIM_SPEED_FACTOR': '1'
        },
        output='screen'
    )

    # ── Mission nodes ──
    # YOLO detector (unchanged)
    balloon_detector = Node(
        package='balloon_hunter',
        executable='balloon_detector',
        name='balloon_detector',
        output='screen',
        parameters=[{
            'system_id': drone_id,
            'camera_topic': f'/drone{drone_id}/camera/image_raw',
            'model_path': model_path,
            'conf': 0.5,
            'target_class': 'sports ball'
        }]
    )

    # NEW: IBVS + PNG controller (replaces position_estimator + drone_manager)
    drone_manager = Node(
        package='balloon_hunter',
        executable='drone_manager',
        name='drone_manager',
        output='screen',
        parameters=[{
            'system_id': drone_id,
            'takeoff_height': 6.0,
            # Camera intrinsics
            'img_width': 848,
            'img_height': 480,
            'fx': 454.8,
            'fy': 454.8,
            'cx': 424.0,
            'cy': 240.0,
            'cam_pitch_deg': 0.0,
            # PNG parameters (paper defaults)
            'K_y': 3.0,
            'K_z': 3.0,
            'k_a': 2.0,
            # Yaw PD (paper defaults)
            'kp_yaw': 0.03,
            'kd_yaw': 0.01,
            # Speed
            'max_speed': 10.0,
            'search_speed': 3.0,
            'collision_distance': 0.5,
            # DKF
            'dkf_dt': 0.02,
            'dkf_delay_steps': 3,
            # Topics
            'detection_topic': f'/Yolov8_Inference_{drone_id}',
            'monitoring_topic': f'/drone{drone_id}/fmu/out/monitoring',
        }]
    )

    collision_handler = Node(
        package='balloon_hunter',
        executable='collision_handler',
        name='collision_handler',
        output='screen',
        parameters=[{'collision_distance': 0.5, 'drone_id': drone_id}]
    )

    # Event chaining
    start_spawn_drone1 = TimerAction(period=2.0, actions=[spawn_drone1])

    px4_1_event = RegisterEventHandler(
        OnProcessExit(target_action=spawn_drone1,
                      on_exit=[TimerAction(period=2.0, actions=[px4_process_1])])
    )

    mission_nodes_event = RegisterEventHandler(
        OnProcessExit(target_action=spawn_drone1, on_exit=[
            TimerAction(period=10.0, actions=[
                balloon_detector,
                drone_manager,
                collision_handler
            ])
        ])
    )

    return [
        resource_path_env, model_path_env, plugin_path_env, gz_ip_env,
        xrce_agent_process, gazebo_node,
        gen_sdf_drone1,
        start_spawn_drone1,
        px4_1_event,
        mission_nodes_event
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('px4_src_path', default_value='/home/kiki/PX4Swarm'),
        DeclareLaunchArgument('model_path', default_value='/home/kiki/visionws/src/balloon_hunter/models/yolov8n.pt'),
        OpaqueFunction(function=launch_setup)
    ])