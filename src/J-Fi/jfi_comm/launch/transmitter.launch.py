from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch_ros.actions import Node

def generate_launch_description():
    # 파라미터
    port_arg = DeclareLaunchArgument(
        "port_name", default_value="/dev/ttyUSB0"
    )
    baud_arg = DeclareLaunchArgument(
        "baud_rate", default_value="115200"
    )

    port_name = LaunchConfiguration("port_name")
    baud_rate = LaunchConfiguration("baud_rate")

    # MicroXRCEAgent udp4 -p 8888
    xrce_agent_process = ExecuteProcess(
        cmd=[FindExecutable(name='MicroXRCEAgent'), 'udp4', '-p', '8888'],
        output='screen',
    )

    # 1. 시리얼 통신 노드 (sys_id=1)
    serial_node = Node(
        package="jfi_comm",
        executable="serial_comm_node",
        name="serial_comm_node",
        output="screen",
        parameters=[
            {"port_name": port_name},
            {"baud_rate": int(baud_rate)},
            {"system_id": 1},  # 송신 드론은 ID=1 고정
        ],
    )

    # 2. PX4 데이터 브릿지
    bridge_node = Node(
        package="jfi_comm",
        executable="monitoring_bridge_node",
        name="monitoring_bridge_node",
        output="screen",
    )

    return LaunchDescription([
        xrce_agent_process,
        port_arg,
        baud_arg,
        serial_node,
        bridge_node,
    ])
