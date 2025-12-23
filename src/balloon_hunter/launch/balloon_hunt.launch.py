#!/usr/bin/env python3
"""
Launch file for Balloon Hunter simulation
Based on yolov8 and drone_manager structure
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Launch arguments
    drone_id_arg = DeclareLaunchArgument(
        'drone_id',
        default_value='1',
        description='Drone ID'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/kiki/visionws/src/balloon_hunter/models/yolov8n.pt',
        description='Path to YOLO model'
    )

    # Balloon Detector Node (YOLO-based)
    balloon_detector = Node(
        package='balloon_hunter',
        executable='balloon_detector',
        name='balloon_detector',
        output='screen',
        parameters=[{
            'system_id': LaunchConfiguration('drone_id'),
            'camera_topic': '/drone1/camera/image_raw',
            'model_path': LaunchConfiguration('model_path'),
            'conf': 0.5,
            'target_class': 'sports ball'  # Red balloon approximation
        }]
    )

    # Position Estimator Node (box2image-based)
    position_estimator = Node(
        package='balloon_hunter',
        executable='position_estimator',
        name='position_estimator',
        output='screen',
        parameters=[{
            'system_id': LaunchConfiguration('drone_id'),
            'width': 1280,
            'height': 720,
            'fx': 678.8712179620,
            'fy': 676.5923040326,
            'cx': 600.7451721112,
            'cy': 363.7283523432,
            'cam_pitch_deg': -35.0,
            'detection_topic': '/Yolov8_Inference_1',
            'position_topic': '/drone1/fmu/out/vehicle_local_position',
            'target_position_topic': '/balloon_target_position'
        }]
    )

    # Drone Manager Node (mission control)
    drone_manager = Node(
        package='balloon_hunter',
        executable='drone_manager',
        name='balloon_hunter_drone_manager',
        output='screen',
        parameters=[{
            'system_id': LaunchConfiguration('drone_id'),
            'takeoff_height': 5.0,
            'forward_speed': 2.0,
            'tracking_speed': 3.0,
            'charge_speed': 5.0,
            'charge_distance': 3.0,
            'collision_distance': 0.5
        }]
    )

    # Collision Handler Node
    collision_handler = Node(
        package='balloon_hunter',
        executable='collision_handler',
        name='collision_handler',
        output='screen',
        parameters=[{
            'collision_distance': 0.5,
            'drone_id': LaunchConfiguration('drone_id')
        }]
    )

    return LaunchDescription([
        drone_id_arg,
        model_path_arg,
        balloon_detector,
        position_estimator,
        drone_manager,
        collision_handler
    ])
