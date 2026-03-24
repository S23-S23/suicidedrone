#!/usr/bin/env python3
"""
Ground Truth Target Provider Node
YOLO + position_estimator 파이프라인을 대체하는 노드.
Gazebo model_states에서 풍선의 실제 위치(Ground Truth)를 읽어
/balloon_target_position 토픽으로 퍼블리시한다.

좌표 변환:
  Gazebo world frame: X=East, Y=North, Z=Up
  PX4 NED frame:      X=North, Y=East,  Z=Down
  → NED_x = Gazebo_y
  → NED_y = Gazebo_x
  → NED_z = -Gazebo_z
"""

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
import numpy as np


def gazebo_to_ned(gx: float, gy: float, gz: float):
    """Gazebo ENU → PX4 local NED 변환"""
    return gy, gx, -gz


class GroundTruthTargetProvider(Node):
    def __init__(self):
        super().__init__('ground_truth_target_provider')

        self.declare_parameter('balloon_model_name', 'target_balloon')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('target_position_topic', '/balloon_target_position')

        self.balloon_model_name = self.get_parameter('balloon_model_name').value
        publish_rate = self.get_parameter('publish_rate').value
        target_topic = self.get_parameter('target_position_topic').value

        self.balloon_ned: np.ndarray | None = None

        # Publisher (position_estimator와 동일한 토픽/타입)
        self.target_pub = self.create_publisher(PoseStamped, target_topic, 10)

        # Gazebo model_states 구독
        self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_states_callback,
            10,
        )

        # 일정 주기로 퍼블리시
        self.create_timer(1.0 / publish_rate, self.publish_target)

        self.get_logger().info(
            f'GroundTruthTargetProvider started: '
            f'balloon_model="{self.balloon_model_name}", '
            f'publish_topic="{target_topic}", '
            f'rate={publish_rate} Hz'
        )

    def model_states_callback(self, msg: ModelStates):
        if self.balloon_model_name not in msg.name:
            self.get_logger().warn(
                f'Model "{self.balloon_model_name}" not found in Gazebo. '
                f'Available: {msg.name}',
                throttle_duration_sec=5.0,
            )
            return

        idx = msg.name.index(self.balloon_model_name)
        gx = msg.pose[idx].position.x
        gy = msg.pose[idx].position.y
        gz = msg.pose[idx].position.z

        x_ned, y_ned, z_ned = gazebo_to_ned(gx, gy, gz)
        self.balloon_ned = np.array([x_ned, y_ned, z_ned])

        self.get_logger().info(
            f'[GT] Balloon Gazebo=({gx:.2f}, {gy:.2f}, {gz:.2f}) '
            f'→ NED=({x_ned:.2f}, {y_ned:.2f}, {z_ned:.2f})',
            throttle_duration_sec=5.0,
        )

    def publish_target(self):
        if self.balloon_ned is None:
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(self.balloon_ned[0])
        msg.pose.position.y = float(self.balloon_ned[1])
        msg.pose.position.z = float(self.balloon_ned[2])
        self.target_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthTargetProvider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
