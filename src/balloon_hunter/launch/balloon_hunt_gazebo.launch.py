#!/usr/bin/env python3
"""
Balloon Hunter with Gazebo Simulation Launch File
Automatically starts Gazebo, PX4 SITL, MicroXRCE Agent, and all mission nodes
Based on uwb_sim/launch/gazebo_typhoon_gazebo_world_run.launch.py
"""

import os
from jinja2 import Environment, FileSystemLoader
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import (
    ExecuteProcess,
    DeclareLaunchArgument,
    OpaqueFunction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource


def launch_setup(context, *args, **kwargs):
    # Package paths
    current_package_path = get_package_share_directory('balloon_hunter')
    px4_src_path = LaunchConfiguration('px4_src_path').perform(context)
    gazebo_classic_path = f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic'

    # Get parameters
    drone_id = int(LaunchConfiguration('drone_id').perform(context))
    model_path = LaunchConfiguration('model_path').perform(context)

    # Environment variables for PX4
    px4_sim_env = SetEnvironmentVariable('PX4_SIM_MODEL', 'gazebo-classic_typhoon_h480')
    px4_lat = SetEnvironmentVariable('PX4_HOME_LAT', '36.6299')
    px4_lon = SetEnvironmentVariable('PX4_HOME_LON', '127.4588')

    # Environment variables for Gazebo
    resource_path_env = SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', '/usr/share/gazebo-11')
    model_path_env = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        f'{current_package_path}:{current_package_path}/models:{gazebo_classic_path}/models'
    )
    plugin_path_env = SetEnvironmentVariable(
        'GAZEBO_PLUGIN_PATH',
        f'{px4_src_path}/build/px4_sitl_default/build_gazebo-classic/'
    )

    # MicroXRCE Agent (udp4 -p 8888)
    xrce_agent_process = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
    )

    # Generate Gazebo world with red balloon
    env = Environment(loader=FileSystemLoader(os.path.join(current_package_path, 'worlds')))

    # Use existing world file (balloon_hunt.world)
    world_file_path = os.path.join(current_package_path, 'worlds', 'balloon_hunt.world')

    # Launch Gazebo
    gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch'),
            '/gazebo.launch.py'
        ]),
        launch_arguments={
            'world': world_file_path,
            'verbose': 'false',
            'gui': 'true'
        }.items()
    )

    # Generate drone model SDF using Jinja
    jinja_cmd = [
        f'{gazebo_classic_path}/scripts/jinja_gen.py',
        f'{current_package_path}/models/typhoon_h480/typhoon_h480.sdf.jinja',
        f'{current_package_path}',
        '--mavlink_tcp_port', f'{4560}',
        '--mavlink_udp_port', f'{14560}',
        '--mavlink_id', f'{drone_id}',
        '--gst_udp_port', f'{5600}',
        '--video_uri', f'{5600}',
        '--mavlink_cam_udp_port', f'{14530}',
        '--output-file', f'/tmp/balloon_hunter_drone.sdf'
    ]

    jinja_process = ExecuteProcess(
        cmd=jinja_cmd,
        output='screen',
    )

    # Spawn drone in Gazebo (facing the red balloon at x=0, y=4, z=2)
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', '/tmp/balloon_hunter_drone.sdf',
            '-entity', f'drone{drone_id}',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.1',
            '-Y', '1.5708'  # Yaw 90 degrees (1.5708 rad, facing +Y direction towards balloon)
        ],
        output='screen',
    )

    # PX4 SITL
    px4_cmd = [
        'env',
        'PX4_SIM_MODEL=gazebo-classic_typhoon_h480',
        f'{px4_src_path}/build/px4_sitl_default/bin/px4',
        '-i', '0',
        '-d', f'{px4_src_path}/build/px4_sitl_default/etc',
        '-w', f'{px4_src_path}/build/px4_sitl_default/ROMFS/instance0',
    ]

    px4_process = ExecuteProcess(
        cmd=px4_cmd,
        output='screen',
    )

    # Balloon Detector Node (YOLO)
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

    # Position Estimator Node
    position_estimator = Node(
        package='balloon_hunter',
        executable='position_estimator',
        name='position_estimator',
        output='screen',
        parameters=[{
            'system_id': drone_id,
            'detection_topic': f'/Yolov8_Inference_{drone_id}',
            'position_topic': f'/drone{drone_id}/fmu/out/vehicle_local_position',
            'monitoring_topic': f'/drone{drone_id}/fmu/out/monitoring',
            'target_position_topic': '/balloon_target_position'
        }]
    )

    # Drone Manager Node
    drone_manager = Node(
        package='balloon_hunter',
        executable='drone_manager',
        name='balloon_hunter_drone_manager',
        output='screen',
        parameters=[{
            'system_id': drone_id,
            'takeoff_height': 6.0,
            'forward_speed': 2.0,
            'tracking_speed': 3.0,
            'charge_speed': 5.0,
            'charge_distance': 3.0,
            'collision_distance': 0.5
        }]
    )

    # Collision Handler Node
    collision_handler = Node(
        package='balloon_hunter',
        executable='collision_handler',
        name='collision_handler',
        output='screen',
        parameters=[{
            'collision_distance': 0.5,
            'drone_id': drone_id
        }]
    )

    nodes_to_start = [
        px4_lat,
        px4_lon,
        resource_path_env,
        px4_sim_env,
        model_path_env,
        plugin_path_env,
        xrce_agent_process,
        gazebo_node,
        jinja_process,
        spawn_entity_node,
        px4_process,
        balloon_detector,
        position_estimator,
        drone_manager,
        collision_handler,
    ]

    return nodes_to_start


def generate_launch_description():
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            'px4_src_path',
            default_value='/home/kiki/PX4Swarm',
            description='PX4 source code path'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'drone_id',
            default_value='1',
            description='Drone ID'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/kiki/visionws/src/balloon_hunter/models/yolov8n.pt',
            description='Path to YOLO model'
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])