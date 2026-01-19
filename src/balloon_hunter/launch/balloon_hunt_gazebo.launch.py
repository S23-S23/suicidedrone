#!/usr/bin/env python3
"""
Balloon Hunter with Iris Stereo Camera Launch File (Fixed for Multi-Drone Communication)
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
    RegisterEventHandler,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit

def launch_setup(context, *args, **kwargs):
    # 패키지 경로 설정
    current_package_path = get_package_share_directory('balloon_hunter')
    px4_src_path = LaunchConfiguration('px4_src_path').perform(context)
    gazebo_classic_path = f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic'

    # 파라미터 가져오기
    model_path = LaunchConfiguration('model_path').perform(context)

    # 1. PX4 및 Gazebo 환경 변수 설정
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

    # MicroXRCE Agent 실행 (8888 포트)
    xrce_agent_process = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
    )

    # Gazebo 월드 실행
    world_file_path = os.path.join(current_package_path, 'worlds', 'balloon_hunt.world')
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

    # 드론 모델 경로
    drone1_sdf_path = f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris_depth_camera/iris_depth_camera.sdf'
    drone2_sdf_path = f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris_fisheye_lens_camera/iris_fisheye_lens_camera.sdf'
    drone3_sdf_path = f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris_fisheye_lens_camera/iris_fisheye_lens_camera.sdf'

    # Gazebo 드론 스폰 노드
    spawn_drone1 = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-file', drone1_sdf_path, '-entity', 'drone1', '-x', '0.0', '-y', '0.0', '-z', '0.1', '-Y', '1.5708', '-robot_namespace', 'drone1'],
        output='screen'
    )
    spawn_drone2 = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-file', drone2_sdf_path, '-entity', 'drone2', '-x', '2.0', '-y', '0.0', '-z', '0.1', '-Y', '1.5708', '-robot_namespace', 'drone2'],
        output='screen'
    )
    spawn_drone3 = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-file', drone3_sdf_path, '-entity', 'drone3', '-x', '-2.0', '-y', '0.0', '-z', '0.1', '-Y', '1.5708', '-robot_namespace', 'drone3'],
        output='screen'
    )

    # 순차 스폰 이벤트 핸들러 (3초 간격)
    spawn_drone2_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone1, on_exit=[TimerAction(period=3.0, actions=[spawn_drone2])]))
    spawn_drone3_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone2, on_exit=[TimerAction(period=3.0, actions=[spawn_drone3])]))

    # 4. PX4 SITL 실행 (드론 스폰 완료 후 시작)
    # Drone 1 PX4 (Drone1 spawn 완료 후 2초 대기)
    px4_process_1 = ExecuteProcess(
        cmd=[f'{px4_src_path}/build/px4_sitl_default/bin/px4', '-i', '0', '-d', f'{px4_src_path}/build/px4_sitl_default/etc', '-w', f'{px4_src_path}/build/px4_sitl_default/ROMFS/instance0'],
        additional_env={
            'PX4_SIM_MODEL': 'gazebo-classic_iris',
            'PX4_UXRCE_DDS_NS': 'drone1',
            'PX4_UXRCE_DDS_PORT': '8888',
            'PX4_SYS_ID': '1'
        },
        output='screen'
    )
    px4_1_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone1, on_exit=[TimerAction(period=2.0, actions=[px4_process_1])]))

    # Drone 2 PX4 (Drone2 spawn 완료 후 2초 대기)
    px4_process_2 = ExecuteProcess(
        cmd=[f'{px4_src_path}/build/px4_sitl_default/bin/px4', '-i', '1', '-d', f'{px4_src_path}/build/px4_sitl_default/etc', '-w', f'{px4_src_path}/build/px4_sitl_default/ROMFS/instance1'],
        additional_env={
            'PX4_SIM_MODEL': 'gazebo-classic_iris',
            'PX4_UXRCE_DDS_NS': 'drone2',
            'PX4_UXRCE_DDS_PORT': '8888',
            'PX4_SYS_ID': '2'
        },
        output='screen'
    )
    px4_2_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone2, on_exit=[TimerAction(period=2.0, actions=[px4_process_2])]))

    # Drone 3 PX4 (Drone3 spawn 완료 후 2초 대기)
    px4_process_3 = ExecuteProcess(
        cmd=[f'{px4_src_path}/build/px4_sitl_default/bin/px4', '-i', '2', '-d', f'{px4_src_path}/build/px4_sitl_default/etc', '-w', f'{px4_src_path}/build/px4_sitl_default/ROMFS/instance2'],
        additional_env={
            'PX4_SIM_MODEL': 'gazebo-classic_iris',
            'PX4_UXRCE_DDS_NS': 'drone3',
            'PX4_UXRCE_DDS_PORT': '8888',
            'PX4_SYS_ID': '3'
        },
        output='screen'
    )
    px4_3_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone3, on_exit=[TimerAction(period=2.0, actions=[px4_process_3])]))

    # 5. 미션 노드 설정 (스폰 완료 후 지연 시작)

    # GCS Station (Drone3 spawn 완료 후 5초 대기)
    gcs_station = Node(
        package='balloon_hunter', executable='gcs_station', name='gcs_station', output='screen',
        parameters=[{'system_id': 1, 'rgb_camera_topic': '/drone1/camera/image_raw', 'depth_camera_topic': '/drone1/camera/depth/image_raw', 'target_position_topic': '/balloon_target_position', 'monitoring_topic': '/drone1/fmu/out/monitoring', 'display_fps': 2, 'focal_length': 705.5, 'cx': 640.0, 'cy': 360.0}]
    )

    # Leader Drone Manager (Drone3 spawn 완료 후 5초 대기)
    leader_drone_manager = Node(
        package='balloon_hunter', executable='drone_manager', name='leader_drone_manager', output='screen',
        parameters=[{'system_id': 1, 'takeoff_height': 2.0, 'forward_speed': 4.0, 'tracking_speed': 3.0, 'charge_speed': 5.0, 'charge_distance': 3.0, 'collision_distance': 0.5}]
    )

    # Fisheye Undistort (Drone3 spawn 완료 후 5초 대기)
    undistort_drone2 = Node(package='balloon_hunter', executable='fisheye_undistort', name='fisheye_undistort_drone2', output='screen', parameters=[{'drone_id': 2, 'image_width': 640, 'image_height': 480}])
    undistort_drone3 = Node(package='balloon_hunter', executable='fisheye_undistort', name='fisheye_undistort_drone3', output='screen', parameters=[{'drone_id': 3, 'image_width': 640, 'image_height': 480}])

    # Follower Managers (Drone3 spawn 완료 후 5초 대기)
    follower_drone2_manager = Node(
        package='balloon_hunter', executable='follower_drone_manager', name='follower_drone2_manager', output='screen',
        parameters=[{'drone_id': 2, 'takeoff_height': 2.0, 'formation_offset_x': 0.0, 'formation_offset_y': 2.0, 'leader_drone_id': 1, 'image_width': 640, 'image_height': 480, 'focal_length': 203.7, 'cx': 320.0, 'cy': 240.0}]
    )
    follower_drone3_manager = Node(
        package='balloon_hunter', executable='follower_drone_manager', name='follower_drone3_manager', output='screen',
        parameters=[{'drone_id': 3, 'takeoff_height': 2.0, 'formation_offset_x': 0.0, 'formation_offset_y': -2.0, 'leader_drone_id': 1, 'image_width': 640, 'image_height': 480, 'focal_length': 203.7, 'cx': 320.0, 'cy': 240.0}]
    )

    # 모든 미션 노드를 Drone3 spawn 완료 후 5초 뒤에 시작
    mission_nodes_event = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_drone3,
            on_exit=[TimerAction(
                period=5.0,
                actions=[
                    gcs_station,
                    leader_drone_manager,
                    undistort_drone2,
                    undistort_drone3,
                    follower_drone2_manager,
                    follower_drone3_manager
                ]
            )]
        )
    )

    return [
        resource_path_env, model_path_env, plugin_path_env,
        xrce_agent_process, gazebo_node,
        TimerAction(period=10.0, actions=[spawn_drone1]),
        spawn_drone2_event, spawn_drone3_event,
        px4_1_event, px4_2_event, px4_3_event,  # PX4 프로세스를 드론 스폰 후 시작
        mission_nodes_event  # 모든 미션 노드를 하나의 이벤트로 통합
    ]

def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument('px4_src_path', default_value='/home/kiki/PX4Swarm'),
        DeclareLaunchArgument('model_path', default_value='/home/kiki/visionws/src/balloon_hunter/models/yolov8n.pt')
    ]
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])