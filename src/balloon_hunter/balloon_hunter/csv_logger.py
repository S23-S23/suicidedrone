#!/usr/bin/env python3
"""
CSV Logger Node - 토픽 데이터를 CSV 파일로 저장
수정 사항: 타겟 토픽을 /balloon_target_position (PoseStamped)으로 변경
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped  # Vector3에서 PoseStamped로 변경
from px4_msgs.msg import Monitoring
from datetime import datetime
import os
import csv
import signal
import sys

class CSVLogger(Node):
    def __init__(self):
        super().__init__('csv_logger')

        # 파라미터 설정
        self.declare_parameter('log_directory', '/home/kiki/visionws/logs')
        self.log_dir = self.get_parameter('log_directory').value
        os.makedirs(self.log_dir, exist_ok=True)

        # CSV 파일명 생성 (실행 시점 기준)
        now = datetime.now()
        filename = now.strftime('%m%d_%H%M%S')
        self.drone_csv_path = os.path.join(self.log_dir, f'drone_positions_{filename}.csv')
        self.target_csv_path = os.path.join(self.log_dir, f'target_positions_{filename}.csv')

        # CSV 헤더 초기화
        with open(self.drone_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'pos_x', 'pos_y', 'pos_z'])

        with open(self.target_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'pos_x', 'pos_y', 'pos_z'])

        self.drone_count = 0
        self.target_count = 0

        # 드론 위치 구독 (Monitoring 메시지 사용)
        self.create_subscription(
            Monitoring,
            '/drone1/fmu/out/monitoring',
            self.drone_callback,
            qos_profile_sensor_data
        )

        # 타겟 위치 구독 (PoseStamped 메시지 사용)
        self.create_subscription(
            PoseStamped,
            '/balloon_target_position',
            self.target_callback,
            10
        )

        self.get_logger().info(f'CSV Logger started. Logging to {self.log_dir}')

    def drone_callback(self, msg):
        timestamp = self.get_clock().now().nanoseconds / 1e9
        with open(self.drone_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, msg.pos_x, msg.pos_y, msg.pos_z])
        
        self.drone_count += 1
        if self.drone_count % 100 == 0:
            self.get_logger().info(f'Logged {self.drone_count} drone points')

    def target_callback(self, msg):
        """PoseStamped 메시지에서 위치 정보 추출"""
        timestamp = self.get_clock().now().nanoseconds / 1e9
        
        # PoseStamped는 msg.pose.position 에 x, y, z가 있음
        with open(self.target_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, 
                msg.pose.position.x, 
                msg.pose.position.y, 
                msg.pose.position.z
            ])

        self.target_count += 1
        if self.target_count % 20 == 0:
            self.get_logger().info(f'Logged {self.target_count} target estimates')

    def destroy_node(self):
        self.get_logger().info(f'Final Count - Drone: {self.drone_count}, Target: {self.target_count}')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CSVLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()