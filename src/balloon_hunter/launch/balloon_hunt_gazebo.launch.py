#!/usr/bin/env python3
"""
Balloon Hunter with Iris Stereo Camera Launch File
"""

import os
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
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource


def launch_setup(context, *args, **kwargs):
    # 패키지 경로 설정
    current_package_path = get_package_share_directory('balloon_hunter')
    px4_src_path = LaunchConfiguration('px4_src_path').perform(context)
    gazebo_classic_path = f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic'

    # 파라미터 가져오기
    drone_id = int(LaunchConfiguration('drone_id').perform(context))
    model_path = LaunchConfiguration('model_path').perform(context)

    # 1. PX4 환경 변수 설정 (Typhoon -> Iris 변경)
    px4_sim_env = SetEnvironmentVariable('PX4_SIM_MODEL', 'gazebo-classic_iris')
    px4_lat = SetEnvironmentVariable('PX4_HOME_LAT', '36.6299')
    px4_lon = SetEnvironmentVariable('PX4_HOME_LON', '127.4588')

    # 2. Gazebo 환경 변수 설정
    resource_path_env = SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', '/usr/share/gazebo-11')
    model_path_env = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:'
        f'{current_package_path}/models:'
        f'{gazebo_classic_path}/models'
    )
    plugin_path_env = SetEnvironmentVariable(
        'GAZEBO_PLUGIN_PATH',
        f'{px4_src_path}/build/px4_sitl_default/build_gazebo-classic/'
    )

    # MicroXRCE Agent 실행
    xrce_agent_process = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
    )

    # Gazebo 월드 실행
    world_file_path = os.path.join(current_package_path, 'worlds', 'balloon_hunt.world')
    #world_file_path = os.path.join(px4_src_path, 'Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds', 'outdoor.world')
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

    # 3. 드론 모델 생성 (PX4의 iris_depth_camera 직접 사용)
    # PX4의 iris_depth_camera.sdf를 사용 (모든 의존성 모델 포함)
    drone_sdf_path = f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris_depth_camera/iris_depth_camera.sdf'

    # Gazebo에 드론 스폰 (PX4 iris_depth_camera 모델 사용)
    # Gazebo가 완전히 시작될 때까지 5초 대기
    spawn_entity_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-file', drone_sdf_path,
                    '-entity', f'drone{drone_id}',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '0.1',
                    '-Y', '1.5708'
                ],
                output='screen',
            )
        ]
    )

    # 4. PX4 SITL 실행 (Iris 모델 적용)
    px4_cmd = [
        'env',
        'PX4_SIM_MODEL=gazebo-classic_iris',
        f'{px4_src_path}/build/px4_sitl_default/bin/px4',
        '-i', '0',
        '-d', f'{px4_src_path}/build/px4_sitl_default/etc',
        '-w', f'{px4_src_path}/build/px4_sitl_default/ROMFS/instance0',
    ]

    px4_process = ExecuteProcess(
        cmd=px4_cmd,
        output='screen',
    )

    # 5. 미션 노드 설정 (GCS-based system - YOLO nodes disabled)
    # balloon_detector = Node(
    #     package='balloon_hunter',
    #     executable='balloon_detector',
    #     name='balloon_detector',
    #     output='screen',
    #     parameters=[{
    #         'system_id': drone_id,
    #         # 스테레오 카메라의 왼쪽 영상을 기본 분석용으로 사용
    #         'camera_topic': '/left_camera/image_raw',
    #         'model_path': model_path,
    #         'conf': 0.5,
    #         'target_class': 'sports ball'
    #     }]
    # )

    # position_estimator = Node(
    #     package='balloon_hunter',
    #     executable='position_estimator',
    #     name='position_estimator',
    #     output='screen',
    #     parameters=[{
    #         'system_id': drone_id,
    #         'detection_topic': f'/Yolov8_Inference_{drone_id}',
    #         'position_topic': f'/drone{drone_id}/fmu/out/vehicle_local_position',
    #         'monitoring_topic': f'/drone{drone_id}/fmu/out/monitoring',
    #         'target_position_topic': '/balloon_target_position'
    #     }]
    # )

    # New GCS-based system (all-in-one node) - Depth Camera Mode
    # Wait for drone spawn (5s) + camera initialization (3s) = 8s total delay
    gcs_station = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='balloon_hunter',
                executable='gcs_station',
                name='gcs_station',
                output='screen',
                parameters=[{
                    'system_id': drone_id,
                    'rgb_camera_topic': '/camera/image_raw',
                    'depth_camera_topic': '/camera/depth/image_raw',
                    'target_position_topic': '/balloon_target_position',
                    'monitoring_topic': f'/drone{drone_id}/fmu/out/monitoring',
                    'display_fps': 5,
                    # Depth camera parameters (RealSense D455: 848x480)
                    # Focal length calculated from FOV: f = width / (2 * tan(FOV/2))
                    # FOV = 1.5009831567 rad = 86 degrees
                    # f = 848 / (2 * tan(1.5009831567/2)) = 848 / 1.557 = 544.6
                    'focal_length': 544.6,
                    'cx': 424.0,  # Principal point x (848/2 = 424)
                    'cy': 240.0,  # Principal point y (480/2 = 240)
                    'cam_pitch_deg': 0.0
                }]
            )
        ]
    )

    drone_manager = Node(
        package='balloon_hunter',
        executable='drone_manager',
        name='balloon_hunter_drone_manager',
        output='screen',
        parameters=[{
            'system_id': drone_id,
            'takeoff_height': 2.0,
            'forward_speed': 2.0,
            'tracking_speed': 3.0,
            'charge_speed': 5.0,
            'charge_distance': 3.0,
            'collision_distance': 0.5
        }]
    )

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
        px4_lat, px4_lon, resource_path_env, px4_sim_env,
        model_path_env, plugin_path_env,
        xrce_agent_process, gazebo_node,
        spawn_entity_node, px4_process,
        # balloon_detector, position_estimator,  # Disabled for GCS-based system
        gcs_station,  # New GCS-based all-in-one node (Depth Camera Mode)
        drone_manager, collision_handler,
    ]

    return nodes_to_start


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument('px4_src_path', default_value='/home/kiki/PX4Swarm'),
        DeclareLaunchArgument('drone_id', default_value='1'),
        DeclareLaunchArgument('model_path', default_value='/home/kiki/visionws/src/balloon_hunter/models/yolov8n.pt')
    ]
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])