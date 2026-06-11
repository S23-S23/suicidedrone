import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def launch_setup(context, *args, **kwargs):
    current_package_path = get_package_share_directory('follower_drone')
    balloon_hunter_package_path = get_package_share_directory('balloon_hunter')
    drone_id = LaunchConfiguration('drone_id').perform(context)
    formation_degree = LaunchConfiguration('formation_degree').perform(context)
    formation_distance = LaunchConfiguration('formation_distance').perform(context)

    port_name = LaunchConfiguration('port_name').perform(context)
    baud_rate = LaunchConfiguration('baud_rate').perform(context)

    xrce_agent_process = ExecuteProcess(
        cmd=[FindExecutable(name='MicroXRCEAgent'), '-p', '8888'],
        output='screen'
    )

    drone_manager_node = Node(
        package='follower_drone',
        executable='drone_manager',
        name=f'drone_manager_{drone_id}',
        output='screen',
        parameters=[{
            'system_id': int(drone_id),
            'formation_degree': float(formation_degree),
            'formation_distance': float(formation_distance)
        }]
    )

    balloon_hunt_manager = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(balloon_hunter_package_path, 'launch', 'balloon_hunt_real.launch.py')
        ),
        launch_arguments={
            # ── System ──
            'system_id': '2',

            # ── Camera ──
            'camera_topic': '/camera/camera/color/image_raw',
            'model_path': '/home/suvlab/ros2_ws/suicidedrone/src/balloon_hunter/models/balloon_yolov8n.pt',
            'target_class': 'red-balloon',
            'conf': '0.5',

            # ── Camera Intrinsics ──
            'fx': '643.2935',
            'fy': '642.6299',
            'cx': '644.1669',
            'cy': '355.8758',
            'cam_pitch_deg': '45.0',

            # ── IBVS Gains ──
            'fov_kp': '0.5',
            'fov_kd': '0.05',
            'fov_kp_z': '0.5',
            'fov_kd_z': '0.05',

            # ── PNG Parameters ──
            'Ky': '3.0',
            'Kz': '3.0',
            'ka': '0.5',
            'v_max': '0.7',
            'v_init': '0.5',
            'rate': '50.0',

            # ── Drone Manager ──
            'hover_init_duration': '5.0',

            # ── Filter ──
            'filter_type': 'DKF',
        }.items()
    )

    serial_node = Node(
        package="jfi_comm",
        executable="serial_comm_node",
        name="serial_comm_node",
        output="screen",
        parameters=[
            {"port_name": port_name},
            {"baud_rate": int(baud_rate)},
            {"system_id": int(drone_id)},
        ],
    )

    receiver_node = Node(
        package="jfi_comm",
        executable="pos_yaw_receiver_node",
        name="pos_yaw_receiver_node",
        output="screen",
        parameters=[
            {"drone_id": int(drone_id)},
        ],
    )


    node = [
        # xrce_agent_process,
        drone_manager_node,
        balloon_hunt_manager,
        serial_node,
        receiver_node
    ]

    return node

def generate_launch_description():
    declared_arguments = []
    declared_arguments.extend([
        DeclareLaunchArgument('drone_id',           default_value='2',              description='Drone ID'),
        DeclareLaunchArgument('formation_degree',   default_value='40.0',           description='Formation Degree'),
        DeclareLaunchArgument('formation_distance', default_value='3.0',            description='Formation Distance'),
        DeclareLaunchArgument('port_name',          default_value='/dev/ttyUSB0',   description='Serial Port Name for J-Fi Receiver'),
        DeclareLaunchArgument('baud_rate',          default_value='115200',         description='Baud Rate for J-Fi Receiver'),
    ])

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
