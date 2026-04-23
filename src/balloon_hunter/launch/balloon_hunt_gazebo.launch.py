#!/usr/bin/env python3
"""
Launch file for modular IBVS + PNG balloon interception.

Architecture:
  balloon_detector (or gt_balloon_detector)  -> /target_info
  ibvs_controller                            -> /ibvs/output
  png_guidance                               -> /png/guidance_cmd
  drone_manager (thin FSM)                   -> PX4 setpoints
  filter_node (DKF/EKF)                      -> /filter_estimate (for logger)

Usage:
  ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py filter_type:=DKF
  ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py filter_type:=EKF
  ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py detector_type:=GT filter_type:=GT
"""
import os
from datetime import datetime
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
    filter_type = LaunchConfiguration('filter_type').perform(context)
    detector_type = LaunchConfiguration('detector_type').perform(context)
    drone_id = 1

    # ── Environment ──
    resource_path_env = SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', '/usr/share/gazebo-11')
    model_path_env = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        f'{current_package_path}/models:'
        f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:'
        f'{gazebo_classic_path}/models'
    )
    plugin_path_env = SetEnvironmentVariable(
        'GAZEBO_PLUGIN_PATH',
        f'{px4_src_path}/build/px4_sitl_default/build_gazebo-classic/'
    )
    gz_ip_env = SetEnvironmentVariable('GZ_IP', '127.0.0.1')

    # ── Infrastructure ──
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

    # ── SDF generation ──
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

    # ── Spawn ──
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

    # ── PX4 SITL ──
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

    # ══════════════════════════════════════════════════════════════
    #  Mission nodes — all with use_sim_time: True
    # ══════════════════════════════════════════════════════════════

    # 1. Detector (YOLO or GT)
    balloon_detector_node = Node(
        package='balloon_hunter',
        executable='balloon_detector',
        name='balloon_detector',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'system_id': drone_id,
            'camera_topic': f'/drone{drone_id}/camera/image_raw',
            'model_path': model_path,
            'conf': 0.5,
            'target_class': 'sports ball',
        }]
    )

    gt_balloon_detector_node = Node(
        package='balloon_hunter',
        executable='gt_balloon_detector',
        name='balloon_detector',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'system_id': drone_id,
            'camera_topic': f'/drone{drone_id}/camera/image_raw',
            'width': 848,
            'height': 480,
            'fx': 454.8,
            'fy': 454.8,
            'cx': 424.0,
            'cy': 240.0,
            'balloon_model_name': 'target_balloon',
            'balloon_radius': 0.3,
            'balloon_link_z_offset': 1.5,
            'camera_link_name': f'drone{drone_id}::depth_camera_link',
            'sensor_offset_x': 0.0,
            'sensor_offset_y': 0.0,
            'sensor_offset_z': 0.0,
        }]
    )

    if detector_type == 'GT':
        detector_node = gt_balloon_detector_node
        filter_type = 'GT'
    else:
        detector_node = balloon_detector_node

    # 2. IBVS Controller
    ibvs_controller_node = Node(
        package='balloon_hunter',
        executable='ibvs_controller',
        name='ibvs_controller',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'system_id': drone_id,
            'fx': 454.8,
            'fy': 454.8,
            'cx': 424.0,
            'cy': 240.0,
            'fov_kp': 1.5,
            'fov_kd': 0.1,
            'fov_kp_z': 1.5,
            'fov_kd_z': 0.1,
            'target_timeout': 0.5,
        }]
    )

    # 3. PNG Guidance
    png_guidance_node = Node(
        package='balloon_hunter',
        executable='png_guidance',
        name='png_guidance',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'system_id': drone_id,
            'Ky': 3.0,
            'Kz': 3.0,
            'ka': 2.0,
            'v_max': 10.0,
            'v_init': 3.5,
            'rate': 50.0,
            'v_min_sigma': 0.5,
        }]
    )

    # 4. Drone Manager (thin FSM)
    drone_manager_node = Node(
        package='balloon_hunter',
        executable='drone_manager',
        name='drone_manager',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'system_id': drone_id,
            'takeoff_height': 6.0,
            'forward_distance_limit': 50.0,
            'collision_distance': 2.0,
            'mission_timeout': 60.0,
            'max_speed': 10.0,
        }]
    )

    # 5. Filter Node (DKF/EKF — for logger, runs alongside IBVS+PNG)
    filter_node = Node(
        package='balloon_hunter',
        executable='filter_node',
        name='filter_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'system_id': drone_id,
            'filter_type': filter_type,
            'fx': 454.8,
            'fy': 454.8,
            'cx': 424.0,
            'cy': 240.0,
            'cam_pitch_deg': 0.0,
            'dkf_dt': 0.02,
            'dkf_delay_steps': 2,
            'assumed_depth': 10.0,
        }]
    )

    # 6. Logger
    logger_node = Node(
        package='balloon_hunter',
        executable='logger',
        name='logger',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'filter_type': filter_type,
            'system_id': drone_id,
            'target_gazebo_x':  3.0,
            'target_gazebo_y':  10.0,
            'target_gazebo_z':  6.5,
            'fx': 454.8,
            'fy': 454.8,
            'cx': 424.0,
            'cy': 240.0,
            'cam_pitch_deg': 0.0,
            'collision_distance': 2.0,
        }]
    )

    # 7. Target Mover
    target_mover_node = Node(
        package='balloon_hunter',
        executable='target_mover',
        name='target_mover',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'target_name': 'target_balloon',
            'nominal_x':    3.0,
            'nominal_y':    13.0,
            'nominal_z':    5.0,
            'amplitude':    0.0,
            'speed':        0.0,
            'balloon_link_z_offset': 1.5,
        }]
    )

    # 8. Drone Visualizer
    drone_visualizer_node = Node(
        package='balloon_hunter',
        executable='drone_visualizer',
        name='drone_visualizer',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'system_id': drone_id,
            'max_path_points': 5000,
            'balloon_model_name': 'target_balloon',
            'balloon_radius': 0.3,
            'balloon_link_z_offset': 1.5,
            'fx': 454.8,
            'fy': 454.8,
            'cx': 424.0,
            'cy': 240.0,
            'camera_topic': f'/drone{drone_id}/camera/image_raw',
        }]
    )

    # 9. RViz2
    rviz_config = os.path.join(current_package_path, 'config', 'drone_trajectory.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    # ── ros2 bag recording (mcap) ────────────────────────────────────────────
    # Records all topics for later inspection in Foxglove Studio.
    # Storage format: mcap (install: ros-$ROS_DISTRO-rosbag2-storage-mcap)
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

    # ── Event chaining ──
    start_spawn_drone1 = TimerAction(period=2.0, actions=[spawn_drone1])

    px4_1_event = RegisterEventHandler(
        OnProcessExit(target_action=spawn_drone1,
                      on_exit=[TimerAction(period=2.0, actions=[px4_process_1])])
    )

    mission_nodes_event = RegisterEventHandler(
        OnProcessExit(target_action=spawn_drone1, on_exit=[
            TimerAction(period=10.0, actions=[
                target_mover_node,
                detector_node,
                ibvs_controller_node,
                png_guidance_node,
                filter_node,
                drone_manager_node,
                logger_node,
                drone_visualizer_node,
            ])
        ])
    )

    return [
        resource_path_env, model_path_env, plugin_path_env, gz_ip_env,
        xrce_agent_process, gazebo_node,
        gen_sdf_drone1,
        start_spawn_drone1,
        px4_1_event,
        mission_nodes_event,
        rviz_node,
        *bag_actions,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('px4_src_path', default_value='/home/kiki/PX4Swarm'),
        DeclareLaunchArgument(
            'model_path',
            default_value=os.path.join(
                get_package_share_directory('balloon_hunter'), 'models', 'yolov8n.pt'
            ),
        ),
        DeclareLaunchArgument('filter_type', default_value='DKF'),
        DeclareLaunchArgument('detector_type', default_value='YOLO'),
        DeclareLaunchArgument(
            'bag_enable',
            default_value='true',
            description='Enable ros2 bag mcap recording: true | false'
        ),
        DeclareLaunchArgument(
            'bag_dir',
            # install/<pkg>/share/<pkg>  →  ../../../../log/rosbag  =  <ws>/log/rosbag
            default_value=os.path.normpath(os.path.join(
                get_package_share_directory('balloon_hunter'),
                '..', '..', '..', '..', 'log', 'rosbag')),
            description='Directory to save mcap bag files (default: <workspace>/log/rosbag)'
        ),
        OpaqueFunction(function=launch_setup)
    ])
