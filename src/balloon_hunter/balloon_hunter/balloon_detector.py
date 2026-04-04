#!/usr/bin/env python3
"""
Balloon Detector Node using YOLO
Detects red balloons from camera feed using YOLOv8
Based on yolov8_ros2_pt.py
"""

import torch
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from suicide_drone_msgs.msg import TargetInfo
import cv2

bridge = CvBridge()


class BalloonDetector(Node):
    def __init__(self):
        super().__init__('balloon_detector')

        # Parameters
        self.declare_parameter('system_id', 1)
        self.declare_parameter('camera_topic', '/drone1/camera/image_raw')
        self.declare_parameter('model_path', '/home/kiki/visionws/src/balloon_hunter/models/yolov8n.pt')
        self.declare_parameter('conf', 0.5)  # Confidence threshold
        self.declare_parameter('target_class', 'sports ball')  # Red balloon class in COCO dataset

        self.system_id = self.get_parameter('system_id').value
        self.camera_topic = self.get_parameter('camera_topic').value
        model_path = self.get_parameter('model_path').value
        self.conf = self.get_parameter('conf').value
        self.target_class = self.get_parameter('target_class').value

        # Load YOLO model
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(model_path)

        # Try to use GPU if available
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            self.model.to('cuda:0')
            self.get_logger().info('Using GPU (CUDA)')
        else:
            self.device = 'cpu'
            self.get_logger().info('Using CPU (GPU not available)')

        # Get target class ID
        names = self.model.names
        if isinstance(names, dict):
            matching_classes = [k for k, v in names.items() if self.target_class.lower() in v.lower()]
            if matching_classes:
                self.target_cls_id = matching_classes[0]
            else:
                self.get_logger().warn(f'Target class "{self.target_class}" not found, using all classes')
                self.target_cls_id = None
        else:
            matching_classes = [i for i, v in enumerate(names) if self.target_class.lower() in v.lower()]
            if matching_classes:
                self.target_cls_id = matching_classes[0]
            else:
                self.target_cls_id = None

        self.get_logger().info(f'Target class: {self.target_class}, ID: {self.target_cls_id}, conf={self.conf}')

        # QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.camera_callback,
            qos_profile
        )

        # Publishers
        self.yolov8_pub = self.create_publisher(
            TargetInfo,
            '/target_info',
            10
        )

        self.img_pub = self.create_publisher(
            Image,
            f'/inference_result_{self.system_id}',
            10
        )

        self.get_logger().info('Balloon Detector initialized')
        self.get_logger().info(f'Subscribing to: {self.camera_topic}')

    def camera_callback(self, msg: Image):
        """Process camera images with YOLO"""
        try:
            self.get_logger().info('[DEBUG] Balloon detector: Camera callback triggered', throttle_duration_sec=5.0)
            # Convert ROS Image to OpenCV
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run YOLO inference
            results = self.model.predict(
                source=cv_image,
                conf=self.conf,
                device=self.device,
                verbose=False
            )

            # Process results — publish only the first matching detection
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())

                        # Filter by target class if specified
                        if self.target_cls_id is not None and cls_id != self.target_cls_id:
                            continue

                        # Publish first detection as TargetInfo
                        target_info            = TargetInfo()
                        target_info.header     = msg.header
                        target_info.class_name = self.model.names[cls_id]
                        target_info.left       = int(x1)
                        target_info.top        = int(y1)
                        target_info.right      = int(x2)
                        target_info.bottom     = int(y2)
                        self.yolov8_pub.publish(target_info)

                        # Draw on image
                        cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        label = f'{self.model.names[cls_id]}: {conf:.2f}'
                        cv2.putText(cv_image, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        self.get_logger().info(
                            f'[DEBUG] Publishing detection: bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)}), topic=/target_info',
                            throttle_duration_sec=1.0
                        )
                        break  # first target only

            # Publish visualization
            img_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            img_msg.header = msg.header
            self.img_pub.publish(img_msg)

        except Exception as e:
            self.get_logger().error(f'Error in camera callback: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = BalloonDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()