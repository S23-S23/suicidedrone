#!/usr/bin/env python3
"""
Balloon Hunter - Ground Truth Mode Launch File
Uses Gazebo Ground Truth instead of YOLO + position_estimator.
Removes the balloon_detector and position_estimator nodes and
replaces them with the ground_truth_target_provider node.
"""

import os
from datetime import datetime
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
    current_package_path = get_package_share_directory('balloon_hunter')
    px4_src_path = LaunchConfiguration('px4_src_path').perform(context)
    gazebo_classic_path = f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic'

    drone_id = int(LaunchConfiguration('drone_id').perform(context))

    # Environment variables
    px4_sim_env        = SetEnvironmentVariable('PX4_SIM_MODEL', 'gazebo-classic_typhoon_h480')
    px4_lat            = SetEnvironmentVariable('PX4_HOME_LAT', '36.6299')
    px4_lon            = SetEnvironmentVariable('PX4_HOME_LON', '127.4588')
    resource_path_env  = SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', '/usr/share/gazebo-11')
    model_path_env     = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        f'{current_package_path}:{current_package_path}/models:{gazebo_classic_path}/models'
    )
    plugin_path_env    = SetEnvironmentVariable(
        'GAZEBO_PLUGIN_PATH',
        f'{px4_src_path}/build/px4_sitl_default/build_gazebo-classic/'
    )

    # MicroXRCE Agent
    xrce_agent_process = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
    )

    # Gazebo
    world_file_path = os.path.join(current_package_path, 'worlds', 'balloon_hunt.world')
    gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch'),
            '/gazebo.launch.py'
        ]),
        launch_arguments={
            'world': world_file_path,
            'verbose': 'false',
            'gui': 'false'
        }.items()
    )

    # Generate drone SDF via Jinja
    jinja_process = ExecuteProcess(
        cmd=[
            f'{gazebo_classic_path}/scripts/jinja_gen.py',
            f'{current_package_path}/models/typhoon_h480/typhoon_h480.sdf.jinja',
            f'{current_package_path}',
            '--mavlink_tcp_port', '4560',
            '--mavlink_udp_port', '14560',
            '--mavlink_id', f'{drone_id}',
            '--gst_udp_port', '5600',
            '--video_uri', '5600',
            '--mavlink_cam_udp_port', '14530',
            '--output-file', '/tmp/balloon_hunter_drone.sdf',
        ],
        output='screen',
    )

    # Spawn drone
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', '/tmp/balloon_hunter_drone.sdf',
            '-entity', f'drone{drone_id}',
            '-x', '0.0', '-y', '0.0', '-z', '0.1',
            '-Y', '1.5708',
        ],
        output='screen',
    )

    # PX4 SITL
    px4_process = ExecuteProcess(
        cmd=[
            'env',
            'PX4_SIM_MODEL=gazebo-classic_typhoon_h480',
            f'{px4_src_path}/build/px4_sitl_default/bin/px4',
            '-i', '0',
            '-d', f'{px4_src_path}/build/px4_sitl_default/etc',
            '-w', f'{px4_src_path}/build/px4_sitl_default/ROMFS/instance0',
        ],
        output='screen',
    )

    # [GT] Ground Truth Balloon Detector (replaces YOLO-based balloon_detector)
    # Output: /Yolov8_Inference_{id}  →  received by position_estimator as-is
    gt_balloon_detector = Node(
        package='balloon_hunter',
        executable='gt_balloon_detector',
        name='balloon_detector',
        output='screen',
        parameters=[{
            'system_id': drone_id,
            'camera_topic': f'/drone{drone_id}/camera/image_raw',
            'balloon_model_name': 'target_balloon',
            'balloon_radius': 0.3,
            'balloon_link_z_offset': 1.5,
            'drone_model_name': f'drone{drone_id}',
            'camera_link_name': f'drone{drone_id}::cgo3_camera_link',
            'cam_pitch_deg': 0.0,
        }]
    )

    # [IBVS] Image-Based Visual Servoing Controller
    # Computes LOS angles (Eq.5,7), image error (Eq.3), FOV yaw/Z rate (Eq.13)
    ibvs_controller = Node(
        package='balloon_hunter',
        executable='ibvs_controller',
        name='ibvs_controller',
        output='screen',
        parameters=[{
            'system_id': drone_id,
            'fx': 205.5,
            'fy': 205.5,
            'cx': 320.0,
            'cy': 180.0,
            'fov_kp':   1.5,
            'fov_kd':   0.1,
            'fov_kp_z': LaunchConfiguration('ibvs_kp_z'),
            'fov_kd_z': LaunchConfiguration('ibvs_kd_z'),
            'target_timeout': 1.5,
        }]
    )

    # [PNG] Proportional Navigation Guidance
    # Computes NED velocity command (Eq.8,9,10,14)
    png_guidance = Node(
        package='balloon_hunter',
        executable='png_guidance',
        name='png_guidance',
        output='screen',
        parameters=[{
            'system_id': drone_id,
            'Ky': LaunchConfiguration('png_Ky'),
            'Kz': LaunchConfiguration('png_Kz'),
            'ka': LaunchConfiguration('png_ka'),
            'v_max': LaunchConfiguration('png_v_max'),
            'v_init': LaunchConfiguration('png_v_init'),
            'rate': 50.0,
        }]
    )

    # Drone Manager (INTERCEPT state uses PNG velocity + IBVS yaw rate)
    drone_manager = Node(
        package='balloon_hunter',
        executable='drone_manager',
        name='balloon_hunter_drone_manager',
        output='screen',
        parameters=[{
            'system_id': drone_id,
            'takeoff_height': 5.5,
            'forward_speed': 2.0,
            'forward_distance_limit': 50.0,
        }]
    )

    # Collision Handler (uses Gazebo model_states + Monitoring, no position_estimator)
    collision_handler = Node(
        package='balloon_hunter',
        executable='collision_handler',
        name='collision_handler',
        output='screen',
        parameters=[{
            'collision_distance': 0.5,
            'drone_id': drone_id,
            'balloon_model_name': 'target_balloon',
            'balloon_link_z_offset': 1.5,
        }]
    )

    # Drone Visualizer (TF/Path for RViz2)
    drone_visualizer = Node(
        package='balloon_hunter',
        executable='drone_visualizer',
        name='drone_visualizer',
        output='screen',
        parameters=[{
            'system_id': drone_id,
            'max_path_points': 5000,
        }]
    )

    # RViz2
    rviz_config = os.path.join(current_package_path, 'config', 'drone_trajectory.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    # Mission Logger
    mission_logger = Node(
        package='balloon_hunter',
        executable='mission_logger',
        name='mission_logger',
        output='screen',
        parameters=[{
            'system_id': drone_id,
            'balloon_model_name': 'target_balloon',
            'balloon_link_z_offset': 1.5,
            'log_rate': 10.0,
        }]
    )

    # ── ros2 bag recording (mcap) ────────────────────────────────────────────
    # Records all topics.  Output directory: <bag_dir>/balloon_hunt_<timestamp>
    # Storage format: mcap  (install: ros-$ROS_DISTRO-rosbag2-storage-mcap)
    bag_enable = LaunchConfiguration('bag_enable').perform(context).lower()
    bag_actions = []
    if bag_enable != 'false':
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        bag_dir   = LaunchConfiguration('bag_dir').perform(context)
        bag_path  = os.path.join(bag_dir, f'balloon_hunt_{timestamp}')
        os.makedirs(bag_dir, exist_ok=True)
        bag_actions.append(ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '--all',
                '--storage', 'mcap',
                '--output', bag_path,
            ],
            output='screen',
        ))

    # Balloon Mover
    # Moves the target balloon when the drone enters FORWARD state.
    # movement_pattern driven by the 'move' launch argument.
    balloon_mover = Node(
        package='balloon_hunter',
        executable='balloon_mover',
        name='balloon_mover',
        output='screen',
        parameters=[{
            'balloon_model_name': 'target_balloon',
            'movement_pattern': LaunchConfiguration('move'),
            'speed': 1.0,
            'update_rate': 20.0,
            'initial_x': 3.0,
            'initial_y': 15.0,
            'initial_z': 2.0,
            'random_interval': 3.0,
        }]
    )

    return [
        px4_lat, px4_lon,
        resource_path_env, px4_sim_env, model_path_env, plugin_path_env,
        xrce_agent_process,
        gazebo_node,
        jinja_process,
        spawn_entity_node,
        px4_process,
        gt_balloon_detector,   # → /Yolov8_Inference_{id}
        ibvs_controller,       # → /ibvs/target_detected, /ibvs/los_angles, /ibvs/fov_yaw_rate
        png_guidance,          # → /png/velocity_cmd
        drone_manager,         # consumes ibvs + png outputs
        collision_handler,     # → /balloon_collision
        drone_visualizer,
        rviz_node,
        mission_logger,
        balloon_mover,         # moves target balloon on FORWARD state entry
        *bag_actions,          # ros2 bag record (mcap) — empty list when bag_enable=false
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'px4_src_path',
            default_value='/home/user/Projects/PX4Swarm',
            description='PX4 source code path'
        ),
        DeclareLaunchArgument(
            'drone_id',
            default_value='1',
            description='Drone ID'
        ),
        DeclareLaunchArgument(
            'move',
            default_value='left',
            description='Balloon movement pattern: left | right | up | down | random | none'
        ),
        # IBVS vertical (ey) controller gains
        DeclareLaunchArgument(
            'ibvs_kp_z', default_value='1.5',
            description='IBVS ey Z-velocity gain (vertical image centering)'
        ),
        DeclareLaunchArgument(
            'ibvs_kd_z', default_value='0.1',
            description='IBVS ey_dot Z-velocity derivative gain'
        ),
        # PNG Guidance tuning parameters
        DeclareLaunchArgument(
            'png_v_max',  default_value='3.0',
            description='PNG max speed [m/s]'
        ),
        DeclareLaunchArgument(
            'png_v_init', default_value='1.5',
            description='PNG initial speed on intercept entry [m/s]'
        ),
        DeclareLaunchArgument(
            'png_ka',     default_value='2.0',
            description='PNG speed acceleration increment [m/s per second]'
        ),
        DeclareLaunchArgument(
            'png_Ky',     default_value='2.0',
            description='PNG elevation gain'
        ),
        DeclareLaunchArgument(
            'png_Kz',     default_value='4.0',
            description='PNG azimuth gain'
        ),
        # ros2 bag arguments
        DeclareLaunchArgument(
            'bag_enable',
            default_value='true',
            description='Enable ros2 bag recording: true | false'
        ),
        DeclareLaunchArgument(
            'bag_dir',
            # install/<pkg>/share/<pkg>  →  ../../../../log/rosbag  =  <ws>/log/rosbag
            default_value=os.path.normpath(os.path.join(
                get_package_share_directory('balloon_hunter'),
                '..', '..', '..', '..', 'log', 'rosbag')),
            description='Directory to save mcap bag files (default: <workspace>/log/rosbag)'
        ),
        OpaqueFunction(function=launch_setup),
    ])
