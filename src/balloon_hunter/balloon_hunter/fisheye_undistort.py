#!/usr/bin/env python3
"""
Fisheye Lens Undistortion Node
어안렌즈로 들어온 image_raw를 일반 카메라처럼 펼쳐서 재발행합니다.

Gazebo Fisheye Camera 설정:
- horizontal_fov: 3.1415 rad (180도)
- cutoff_angle: 1.5707 rad (90도)
- type: equidistant
- resolution: 640x480

Equidistant fisheye 모델: r = f * theta
여기서 theta = cutoff_angle = π/2 일 때 r = width/2
따라서 f = (width/2) / (π/2) = width / π ≈ 203.7
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
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

        # Gazebo Equidistant Fisheye 모델의 원본 K 행렬
        # f = (width/2) / cutoff_angle = 320 / 1.5707 ≈ 203.7
        fx_orig = self.img_width / np.pi  # 203.7
        fy_orig = fx_orig
        cx = self.img_width / 2.0
        cy = self.img_height / 2.0

        self.K = np.array([
            [fx_orig, 0, cx],
            [0, fy_orig, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # 어안렌즈 왜곡 계수 (Gazebo에서는 기본적으로 왜곡 없음)
        self.D = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Undistortion을 위한 새로운 카메라 매트릭스 계산
        # balance=0.0: 유효 픽셀만 보존 (검은 영역 최소화)
        # balance=1.0: 모든 원본 픽셀 보존 (검은 영역 많음)
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D,
            (self.img_width, self.img_height),
            np.eye(3),
            balance=0.5  # 적절한 균형
        )

        # new_K의 focal length를 follower_drone_manager에서 사용해야 함
        self.new_fx = self.new_K[0, 0]
        self.new_fy = self.new_K[1, 1]
        self.new_cx = self.new_K[0, 2]
        self.new_cy = self.new_K[1, 2]

        self.get_logger().info(f'[Drone {self.drone_id}] Undistorted K: fx={self.new_fx:.2f}, fy={self.new_fy:.2f}, cx={self.new_cx:.2f}, cy={self.new_cy:.2f}')

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

        # CameraInfo 발행 (follower가 정확한 K 행렬 사용 가능하도록)
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            f'/drone{self.drone_id}/camera/camera_info',
            1
        )

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

            # CameraInfo 발행
            camera_info = CameraInfo()
            camera_info.header = msg.header
            camera_info.width = self.img_width
            camera_info.height = self.img_height
            # ROS2 CameraInfo.k는 array('d', ...) 형식이어야 함
            camera_info.k = [float(self.new_fx), 0.0, float(self.new_cx),
                            0.0, float(self.new_fy), float(self.new_cy),
                            0.0, 0.0, 1.0]
            camera_info.d = []  # undistorted (빈 배열)
            camera_info.distortion_model = 'plumb_bob'
            camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            camera_info.p = [float(self.new_fx), 0.0, float(self.new_cx), 0.0,
                            0.0, float(self.new_fy), float(self.new_cy), 0.0,
                            0.0, 0.0, 1.0, 0.0]
            self.camera_info_pub.publish(camera_info)

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
