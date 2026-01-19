#!/usr/bin/env python3
"""
Fisheye Lens Undistortion Node
어안렌즈로 들어온 image_raw를 일반 카메라처럼 펼쳐서 재발행합니다.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class FisheyeUndistort(Node):
    def __init__(self):
        super().__init__('fisheye_undistort')

        # 파라미터 선언
        self.declare_parameter('drone_id', 2)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        self.drone_id = self.get_parameter('drone_id').value
        self.img_width = self.get_parameter('image_width').value
        self.img_height = self.get_parameter('image_height').value

        # CvBridge
        self.bridge = CvBridge()

        # 어안렌즈 카메라 파라미터 (Gazebo iris_fisheye_lens_camera 기준)
        # horizontal_fov: 3.1415 rad (180도), 640x480 resolution
        # FOV = 180도일 때 tan(90도) = 무한대이므로 작은 FOV 사용
        # 실제로는 cutoff_angle = 1.5707 (90도)를 사용
        # 카메라 내부 파라미터 행렬 K
        # f = width / (2 * tan(FOV/2))
        # 180도 FOV를 위해서는 매우 작은 focal length 필요
        fx = self.img_width / (2 * np.tan(np.pi / 4))  # 실질적으로 90도 FOV 사용
        fy = fx
        cx = self.img_width / 2.0
        cy = self.img_height / 2.0

        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # 어안렌즈 왜곡 계수 (Equidistant 모델)
        # k1, k2, k3, k4 (Gazebo에서는 기본적으로 왜곡 없음으로 설정되어 있을 수 있음)
        self.D = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Undistortion을 위한 새로운 카메라 매트릭스 계산
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D,
            (self.img_width, self.img_height),
            np.eye(3),
            balance=0.0
        )

        # Undistortion 맵 계산 (한 번만 계산하여 재사용)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.new_K,
            (self.img_width, self.img_height), cv2.CV_16SC2
        )

        # 이미지 처리 제한 (통신 부하 감소)
        self.last_process_time = 0
        self.process_interval = 0.2  # 5Hz (200ms)

        # 구독 및 발행
        self.image_sub = self.create_subscription(
            Image,
            f'/drone{self.drone_id}/camera/image_raw',
            self.image_callback,
            1
        )

        self.undistorted_pub = self.create_publisher(
            Image,
            f'/drone{self.drone_id}/camera/image_undistorted',
            1
        )

        #self.get_logger().info(f'Fisheye Undistort Node Started for Drone {self.drone_id}')
        #self.get_logger().info(f'K matrix:\n{self.K}')
        #self.get_logger().info(f'New K matrix:\n{self.new_K}')

    def image_callback(self, msg):
        """어안렌즈 이미지를 받아서 왜곡 보정 후 재발행 (5Hz 제한)"""
        try:
            # Hz 제한 (5Hz = 200ms 간격)
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self.last_process_time < self.process_interval:
                return
            self.last_process_time = current_time

            # ROS Image -> OpenCV
            fisheye_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Undistortion 적용
            undistorted_img = cv2.remap(fisheye_img, self.map1, self.map2,
                                       interpolation=cv2.INTER_LINEAR)

            # OpenCV -> ROS Image
            undistorted_msg = self.bridge.cv2_to_imgmsg(undistorted_img, 'bgr8')
            undistorted_msg.header = msg.header

            # 발행
            self.undistorted_pub.publish(undistorted_msg)

        except Exception as e:
            self.get_logger().error(f'Undistortion failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = FisheyeUndistort()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
