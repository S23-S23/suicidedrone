import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    current_package_path = get_package_share_directory('follower_drone')
    drone_id = LaunchConfiguration('drone_id').perform(context)
    formation_degree = LaunchConfiguration('formation_degree').perform(context)
    formation_distance = LaunchConfiguration('formation_distance').perform(context)

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

    node = [
        drone_manager_node
    ]

    return node


def generate_launch_description():
    current_package_path = get_package_share_directory('follower_drone')
    declared_arguments = []
    declared_arguments.extend([
        DeclareLaunchArgument(
            'drone_id',
            default_value='2',
            description='Drone ID'
        ),
        DeclareLaunchArgument(
            'formation_degree',
            default_value='40.0',
            description='Formation Degree'
        ),
        DeclareLaunchArgument(
            'formation_distance',
            default_value='3.0',
            description='Formation Distance'
        )
    ])

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
