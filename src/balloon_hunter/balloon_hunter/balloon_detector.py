#!/usr/bin/env python3
"""
Balloon Detector Node (YOLOv8 TensorRT)
=======================================
Adapted from the in-drone tested RealsenseYoloNode.

Runs YOLOv8 with TensorRT + FP16 on CUDA (device=0, half=True) and publishes
the first matching detection as TargetInfo for the IBVS/DKF pipeline.

Subscriptions:
  /camera/camera/color/image_raw   (sensor_msgs/Image) — RealSense color stream

Publications:
  /target_info                     (suicide_drone_msgs/TargetInfo)
                                   — single best matching detection for the pipeline
  /inference_result_{system_id}    (sensor_msgs/Image) — annotated visualization
  /raw_img_{system_id}             (sensor_msgs/Image) — raw passthrough

Key notes:
  * Preserves the original camera header.stamp on /target_info so the DKF can
    compute true image latency (do NOT overwrite stamp with now()).
  * QoS depth=1 on the image subscriber: always process the freshest frame.
  * TensorRT warmup at startup to avoid first-frame lag.
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from suicide_drone_msgs.msg import TargetInfo
from ultralytics import YOLO

bridge = CvBridge()


class BalloonDetector(Node):
    def __init__(self):
        super().__init__('balloon_detector')

        # ── Parameters ──
        self.declare_parameter('system_id', 1)
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('model_path', '../yolo_pt/balloon_yolov8n.pt')
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('target_class', 'red-balloon')
        self.declare_parameter('warmup_h', 720)
        self.declare_parameter('warmup_w', 1280)

        self.system_id    = self.get_parameter('system_id').value
        camera_topic      = self.get_parameter('camera_topic').value
        model_name        = self.get_parameter('model_path').value
        self.conf         = self.get_parameter('conf').value
        self.target_class = self.get_parameter('target_class').value
        warmup_h          = self.get_parameter('warmup_h').value
        warmup_w          = self.get_parameter('warmup_w').value

        # ── Resolve model path: script-relative first, then as given ──
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_name)
        if not os.path.exists(model_path):
            model_path = model_name

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        try:
            self.model = YOLO(model_path, task='detect')
        except Exception as e:
            self.get_logger().error(f'Model load failed: {e}')
            raise

        # ── TensorRT / CUDA warmup ──
        self.get_logger().info('TensorRT warmup...')
        dummy = np.zeros((warmup_h, warmup_w, 3), dtype=np.uint8)
        self.model(dummy, device=0, half=True, verbose=False)
        self.get_logger().info('Warmup done')

        self.get_logger().info(
            f'Target class filter: "{self.target_class}"  |  conf={self.conf}'
        )

        # ── Publishers ──
        self.target_pub  = self.create_publisher(TargetInfo, '/target_info', 10)
        self.img_pub     = self.create_publisher(
            Image, f'/inference_result_{self.system_id}', 1
        )
        self.raw_img_pub = self.create_publisher(
            Image, f'/raw_img_{self.system_id}', 1
        )

        # ── Subscriber (depth=1: always the freshest frame) ──
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, qos,
        )
        self.get_logger().info(f'Subscribed to: {camera_topic}')

    # ── Callback ──
    def image_callback(self, msg: Image):
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge failed: {e}')
            return

        # Raw passthrough (keeps original header for logging)
        raw_out = bridge.cv2_to_imgmsg(img, encoding='bgr8')
        raw_out.header = msg.header
        self.raw_img_pub.publish(raw_out)

        # TensorRT FP16 inference
        results = self.model(
            img, device=0, half=True, conf=self.conf, verbose=False
        )

        # ── Pick first box whose class name contains target_class ──
        target_lower = self.target_class.lower().strip()
        best_box, best_class = None, None
        for box in results[0].boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            if target_lower and target_lower not in cls_name.lower():
                continue
            best_box, best_class = box, cls_name
            break  # first matching detection only

        if best_box is not None:
            x1, y1, x2, y2 = best_box.xyxy[0].tolist()
            tgt = TargetInfo()
            # IMPORTANT: preserve original camera timestamp for DKF delay comp.
            tgt.header.stamp    = msg.header.stamp
            tgt.header.frame_id = msg.header.frame_id
            tgt.class_name = best_class
            tgt.left   = int(x1)
            tgt.top    = int(y1)
            tgt.right  = int(x2)
            tgt.bottom = int(y2)
            self.target_pub.publish(tgt)

            self.get_logger().info(
                f'{best_class}: bbox=({int(x1)},{int(y1)})-({int(x2)},{int(y2)})',
                throttle_duration_sec=1.0,
            )

        # Annotated visualization (always publish)
        annotated = results[0].plot()
        ann_out = bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        ann_out.header = msg.header
        self.img_pub.publish(ann_out)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BalloonDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('Ctrl+C detected, shutting down...')
    except Exception as e:
        if node:
            node.get_logger().error(f'Unexpected error: {e}')
        raise
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
