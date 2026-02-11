#!/usr/bin/env python3
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
from launch.event_handlers import OnProcessExit, OnShutdown

def launch_setup(context, *args, **kwargs):
    # 1. 경로 및 파라미터 설정
    current_package_path = get_package_share_directory('balloon_hunter')
    px4_src_path = LaunchConfiguration('px4_src_path').perform(context)
    gazebo_classic_path = f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic'
    model_path = LaunchConfiguration('model_path').perform(context)

    # 2. 환경 변수 설정
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
    gz_ip_env = SetEnvironmentVariable('GZ_IP', '127.0.0.1')

    # # 2.5. 네트워크 격리 (외부 트래픽 차단, localhost만 허용)
    # block_external_traffic = ExecuteProcess(
    #     cmd=['sudo', 'iptables', '-I', 'OUTPUT', '1', '-o', 'lo', '-j', 'ACCEPT'],
    #     output='screen'
    # )
    # block_external_traffic2 = ExecuteProcess(
    #     cmd=['sudo', 'iptables', '-A', 'OUTPUT', '-m', 'state', '--state', 'ESTABLISHED,RELATED', '-j', 'ACCEPT'],
    #     output='screen'
    # )
    # block_external_traffic3 = ExecuteProcess(
    #     cmd=['sudo', 'iptables', '-A', 'OUTPUT', '-j', 'DROP'],
    #     output='screen'
    # )

    # # 종료 시 iptables 규칙 복구
    # restore_network = RegisterEventHandler(
    #     OnShutdown(on_shutdown=[
    #         ExecuteProcess(cmd=['sudo', 'iptables', '-D', 'OUTPUT', '-j', 'DROP'], output='screen'),
    #         ExecuteProcess(cmd=['sudo', 'iptables', '-D', 'OUTPUT', '-m', 'state', '--state', 'ESTABLISHED,RELATED', '-j', 'ACCEPT'], output='screen'),
    #         ExecuteProcess(cmd=['sudo', 'iptables', '-D', 'OUTPUT', '-o', 'lo', '-j', 'ACCEPT'], output='screen'),
    #     ])
    # )

    # 3. 인프라 실행 (Agent & Gazebo)
    xrce_agent_process = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
    )

    world_file_path = os.path.join(current_package_path, 'worlds', 'balloon_hunt.world')
    gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch'),
            '/gazebo.launch.py'
        ]),
        launch_arguments={'world': world_file_path, 'verbose': 'false', 'gui': 'true'}.items()
    )

    # 4. SDF 동적 생성 함수 (Jinja2 활용)
    # 각 드론별로 TCP/UDP 포트를 다르게 설정하여 충돌을 방지합니다.
    # Drone1: iris_depth_camera (Leader with depth camera)
    # Drone2/3: iris_fisheye_lens_camera (Followers with fisheye camera)
    def create_drone_sdf_cmd(drone_idx, tcp_port, udp_port, model_name):
        return ExecuteProcess(
            cmd=[
                f'{gazebo_classic_path}/scripts/jinja_gen.py',
                os.path.join(current_package_path, 'models', model_name, f'{model_name}.sdf.jinja'),
                f'{current_package_path}',
                '--mavlink_tcp_port', str(tcp_port),
                '--mavlink_udp_port', str(udp_port),
                '--mavlink_id', str(drone_idx),
                '--output-file', f'/tmp/drone_{drone_idx}.sdf'
            ],
            output='screen'
        )

    gen_sdf_drone1 = create_drone_sdf_cmd(1, 4560, 14560, 'iris_depth_camera')
    gen_sdf_drone2 = create_drone_sdf_cmd(2, 4561, 14561, 'iris_fisheye_lens_camera')
    gen_sdf_drone3 = create_drone_sdf_cmd(3, 4562, 14562, 'iris_fisheye_lens_camera')
    gen_sdf_drone4 = create_drone_sdf_cmd(4, 4563, 14563, 'iris_fisheye_lens_camera')
    gen_sdf_drone5 = create_drone_sdf_cmd(5, 4564, 14564, 'iris_fisheye_lens_camera')

    # 5. 드론 스폰 노드 설정
    # V자 대형에 맞게 스폰 위치 조정 (경로 교차 방지)
    # leader가 East(+Y)를 향하므로, follower들은 뒤쪽(-Y)에 V자 형태로 배치
    spawn_drone1 = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-file', '/tmp/drone_1.sdf', '-entity', 'drone1', '-x', '0.0', '-y', '0.0', '-z', '0.1', '-Y', '1.5708', '-robot_namespace', 'drone1'],
        output='screen'
    )
    spawn_drone2 = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-file', '/tmp/drone_2.sdf', '-entity', 'drone2', '-x', '-2.0', '-y', '0.0', '-z', '0.1', '-Y', '1.5708', '-robot_namespace', 'drone2'],
        output='screen'
    )
    spawn_drone3 = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-file', '/tmp/drone_3.sdf', '-entity', 'drone3', '-x', '2.0', '-y', '0.0', '-z', '0.1', '-Y', '1.5708', '-robot_namespace', 'drone3'],
        output='screen'
    )
    spawn_drone4 = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-file', '/tmp/drone_4.sdf', '-entity', 'drone4', '-x', '-4.0', '-y', '0.0', '-z', '0.1', '-Y', '1.5708', '-robot_namespace', 'drone4'],
        output='screen'
    )
    spawn_drone5 = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-file', '/tmp/drone_5.sdf', '-entity', 'drone5', '-x', '4.0', '-y', '0.0', '-z', '0.1', '-Y', '1.5708', '-robot_namespace', 'drone5'],
        output='screen'
    )

    # 6. PX4 SITL 프로세스 설정
    # COM_LOW_BAT_ACT=0: 배터리 failsafe 비활성화 (시뮬레이션 속도 저하 대응)
    def create_px4_process(instance, sys_id, ns):
        return ExecuteProcess(
            cmd=[f'{px4_src_path}/build/px4_sitl_default/bin/px4', '-i', str(instance), '-d', f'{px4_src_path}/build/px4_sitl_default/etc', '-w', f'{px4_src_path}/build/px4_sitl_default/ROMFS/instance{instance}'],
            additional_env={
                'PX4_SIM_MODEL': 'gazebo-classic_iris',
                'PX4_UXRCE_DDS_NS': ns,
                'PX4_UXRCE_DDS_PORT': '8888',
                'PX4_SYS_ID': str(sys_id),
                'PX4_SIM_SPEED_FACTOR': '1'
            },
            output='screen'
        )

    px4_process_1 = create_px4_process(0, 1, 'drone1')
    px4_process_2 = create_px4_process(1, 2, 'drone2')
    px4_process_3 = create_px4_process(2, 3, 'drone3')
    px4_process_4 = create_px4_process(3, 4, 'drone4')
    px4_process_5 = create_px4_process(4, 5, 'drone5')

    # 7. 미션 노드 설정
    gcs_station = Node(
        package='balloon_hunter', executable='gcs_station', name='gcs_station', output='screen',
        parameters=[{'system_id': 1, 'rgb_camera_topic': '/drone1/camera/image_raw', 'depth_camera_topic': '/drone1/camera/depth/image_raw', 'target_position_topic': '/balloon_target_position', 'monitoring_topic': '/drone1/fmu/out/monitoring', 'display_fps': 2, 'focal_length': 705.5, 'cx': 640.0, 'cy': 360.0}]
    )
    leader_drone_manager = Node(
        package='balloon_hunter', executable='drone_manager', name='leader_drone_manager', output='screen',
        parameters=[{'system_id': 1, 'takeoff_height': 2.0, 'forward_speed': 4.0, 'tracking_speed': 3.0, 'charge_speed': 5.0, 'charge_distance': 3.0, 'collision_distance': 0.5}]
    )
    undistort_drone2 = Node(package='balloon_hunter', executable='fisheye_undistort', name='fisheye_undistort_drone2', output='screen', parameters=[{'drone_id': 2, 'image_width': 640, 'image_height': 480}])
    undistort_drone3 = Node(package='balloon_hunter', executable='fisheye_undistort', name='fisheye_undistort_drone3', output='screen', parameters=[{'drone_id': 3, 'image_width': 640, 'image_height': 480}])
    follower_drone2_manager = Node(
        package='balloon_hunter', executable='follower_drone_manager', name='follower_drone2_manager', output='screen',
        parameters=[{'drone_id': 2, 'takeoff_height': 2.0, 'formation_angle': 30.0, 'formation_distance': 4.0, 'leader_drone_id': 1, 'image_width': 640, 'image_height': 480, 'focal_length': 203.7, 'cx': 320.0, 'cy': 240.0}]
    )
    follower_drone3_manager = Node(
        package='balloon_hunter', executable='follower_drone_manager', name='follower_drone3_manager', output='screen',
        parameters=[{'drone_id': 3, 'takeoff_height': 2.0, 'formation_angle': 30.0, 'formation_distance': 4.0, 'leader_drone_id': 1, 'image_width': 640, 'image_height': 480, 'focal_length': 203.7, 'cx': 320.0, 'cy': 240.0}]
    )
    undistort_drone4 = Node(package='balloon_hunter', executable='fisheye_undistort', name='fisheye_undistort_drone4', output='screen', parameters=[{'drone_id': 4, 'image_width': 640, 'image_height': 480}])
    undistort_drone5 = Node(package='balloon_hunter', executable='fisheye_undistort', name='fisheye_undistort_drone5', output='screen', parameters=[{'drone_id': 5, 'image_width': 640, 'image_height': 480}])
    follower_drone4_manager = Node(
        package='balloon_hunter', executable='follower_drone_manager', name='follower_drone4_manager', output='screen',
        parameters=[{'drone_id': 4, 'takeoff_height': 2.0, 'formation_angle': 30.0, 'formation_distance': 3.0, 'leader_drone_id': 1, 'image_width': 640, 'image_height': 480, 'focal_length': 203.7, 'cx': 320.0, 'cy': 240.0}]
    )
    follower_drone5_manager = Node(
        package='balloon_hunter', executable='follower_drone_manager', name='follower_drone5_manager', output='screen',
        parameters=[{'drone_id': 5, 'takeoff_height': 2.0, 'formation_angle': 30.0, 'formation_distance': 3.0, 'leader_drone_id': 1, 'image_width': 640, 'image_height': 480, 'focal_length': 203.7, 'cx': 320.0, 'cy': 240.0}]
    )

    # 8. 이벤트 체이닝 (순차 실행)
    # SDF 생성 지연 후 드론 1 스폰
    start_spawn_drone1 = TimerAction(period=2.0, actions=[spawn_drone1])
    
    # 드론 1 완료 후 드론 2 스폰 및 드론 1 PX4 실행
    spawn_drone2_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone1, on_exit=[TimerAction(period=3.0, actions=[spawn_drone2])]))
    px4_1_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone1, on_exit=[TimerAction(period=2.0, actions=[px4_process_1])]))

    # 드론 2 완료 후 드론 3 스폰 및 드론 2 PX4 실행
    spawn_drone3_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone2, on_exit=[TimerAction(period=3.0, actions=[spawn_drone3])]))
    px4_2_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone2, on_exit=[TimerAction(period=2.0, actions=[px4_process_2])]))

    # 드론 3 완료 후 드론 4 스폰 및 드론 3 PX4 실행
    spawn_drone4_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone3, on_exit=[TimerAction(period=3.0, actions=[spawn_drone4])]))
    px4_3_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone3, on_exit=[TimerAction(period=2.0, actions=[px4_process_3])]))

    # 드론 4 완료 후 드론 5 스폰 및 드론 4 PX4 실행
    spawn_drone5_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone4, on_exit=[TimerAction(period=3.0, actions=[spawn_drone5])]))
    px4_4_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone4, on_exit=[TimerAction(period=2.0, actions=[px4_process_4])]))

    # 드론 5 완료 후 드론 5 PX4 및 미션 노드 실행
    px4_5_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone5, on_exit=[TimerAction(period=2.0, actions=[px4_process_5])]))
    # 미션 노드 시작 시간을 20초로 증가 (5대 드론의 PX4와 DDS 연결이 안정화될 때까지 대기)
    mission_nodes_event = RegisterEventHandler(OnProcessExit(target_action=spawn_drone5, on_exit=[TimerAction(period=20.0, actions=[gcs_station, leader_drone_manager, undistort_drone2, undistort_drone3, undistort_drone4, undistort_drone5, follower_drone2_manager, follower_drone3_manager, follower_drone4_manager, follower_drone5_manager])]))

    return [
        # block_external_traffic, block_external_traffic2, block_external_traffic3,
        # restore_network,
        resource_path_env, model_path_env, plugin_path_env, gz_ip_env,
        xrce_agent_process, gazebo_node,
        gen_sdf_drone1, gen_sdf_drone2, gen_sdf_drone3, gen_sdf_drone4, gen_sdf_drone5,  # SDF 생성은 즉시 시작
        start_spawn_drone1,
        spawn_drone2_event, spawn_drone3_event, spawn_drone4_event, spawn_drone5_event,
        px4_1_event, px4_2_event, px4_3_event, px4_4_event, px4_5_event,
        mission_nodes_event
    ]

def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument('px4_src_path', default_value='/home/kiki/PX4Swarm'),
        DeclareLaunchArgument('model_path', default_value='/home/kiki/visionws/src/balloon_hunter/models/yolov8n.pt')
    ]
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
