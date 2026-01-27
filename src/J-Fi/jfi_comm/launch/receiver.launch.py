from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 파라미터
    port_arg = DeclareLaunchArgument(
        "port_name", default_value="/dev/ttyUSB0"
    )
    baud_arg = DeclareLaunchArgument(
        "baud_rate", default_value="115200"
    )
    system_id_arg = DeclareLaunchArgument(
        #TODO : 드론마다 수정할 것
        "system_id", default_value="2",
        description="System ID for this receiver (2, 3, 4, or 5)"
    )
    
    port_name = LaunchConfiguration("port_name")
    baud_rate = LaunchConfiguration("baud_rate")
    system_id = LaunchConfiguration("system_id")
    
    # 1. 시리얼 통신 노드
    serial_node = Node(
        package="jfi_comm",
        executable="serial_comm_node",
        name="serial_comm_node",
        output="screen",
        parameters=[
            {"port_name": port_name},
            {"baud_rate": baud_rate},
            {"system_id": system_id},  # 2, 3, 4, 5 중 하나
        ],
    )
    
    # 2. RTK 데이터 수신 노드
    receiver_node = Node(
        package="jfi_comm",
        executable="pos_yaw_receiver_node",
        name="pos_yaw_receiver_node",
        output="screen",
        parameters=[
            {"my_system_id": system_id},
        ],
    )
    
    return LaunchDescription([
        port_arg,
        baud_arg,
        system_id_arg,
        serial_node,
        receiver_node,
    ])