import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, SetEnvironmentVariable, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context, *args, **kwargs):
    current_package_path = get_package_share_directory('balloon_hunter')
    px4_src_path = LaunchConfiguration('px4_src_path').perform(context)
    drone_id = LaunchConfiguration('drone_id').perform(context)

    # 1. 환경 변수 설정
    gz_sim_resource_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        f'{current_package_path}/models:{px4_src_path}/Tools/simulation/gz/models'
    )

    # 2. ROS-Gazebo Bridge (정상 작동 확인됨)
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/model/x500_depth/link/base_link/sensor/left/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/model/x500_depth/link/base_link/sensor/left/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/model/x500_depth/link/base_link/sensor/right/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/model/x500_depth/link/base_link/sensor/right/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/model/x500_depth/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu@gz.msgs.IMU'
        ],
        remappings=[
            ('/model/x500_depth/link/base_link/sensor/left/image', '/left_camera/image_raw'),
            ('/model/x500_depth/link/base_link/sensor/right/image', '/right_camera/image_raw'),
            ('/model/x500_depth/link/base_link/sensor/imu_sensor/imu', f'/drone{drone_id}/imu')
        ],
        output='screen'
    )

    # 3. Gazebo 및 드론 생성
    gz_sim = ExecuteProcess(
        cmd=['ign', 'gazebo', os.path.join(current_package_path, 'worlds', 'balloon_hunt.world'), '-r'],
        output='screen'
    )

    spawn_drone = ExecuteProcess(
        cmd=[
            'ign', 'service', '-s', '/world/balloon_hunt/create',
            '--reqtype', 'ignition.msgs.EntityFactory',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '5000',
            '--req', f'sdf: "<sdf version=\'1.9\'><include><uri>x500_depth</uri><name>x500_depth</name></include></sdf>"'
        ],
        output='screen'
    )

    # 4. PX4 SITL 실행 (가장 중요한 수정 부분)
    # -d 옵션 뒤의 경로를 ROMFS가 있는 실제 빌드 경로로 지정
    px4_dir = os.path.join(px4_src_path, 'build/px4_sitl_default')
    px4_process = ExecuteProcess(
        cmd=[
            os.path.join(px4_dir, 'bin/px4'),
            '-i', '0',
            '-d', os.path.join(px4_dir, 'etc'),
            '-w', os.path.join(px4_dir, 'ROMFS/instance0')
        ],
        cwd=px4_dir,  # 작업 디렉토리를 빌드 폴더로 지정하여 px4-alias.sh를 찾게 함
        env={
            'PX4_SIM_MODEL': 'x500',
            'PX4_GZ_SIM_DISABLED': '1',
        },
        output='screen'
    )

    return [gz_sim_resource_path, gz_sim, bridge_node, spawn_drone, px4_process]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('px4_src_path', default_value='/home/kiki/PX4Swarm'),
        DeclareLaunchArgument('drone_id', default_value='1'),
        OpaqueFunction(function=launch_setup)
    ])