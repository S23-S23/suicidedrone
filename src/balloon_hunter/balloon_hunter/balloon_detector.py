#!/usr/bin/env python3
"""
Balloon Detector Node using OpenCV HSV color detection
Detects red balloons from camera feed using color thresholding.
Output format is identical to the YOLO version (Yolov8Inference + bounding box).

Detection runs on a fixed 10 Hz ROS timer — completely decoupled from camera Hz.
The camera callback stores frames into a time-stamped queue.
The timer publishes a result for the frame that arrived ~delay_ms ago,
simulating realistic image-processing latency for DKF evaluation.
"""

import collections
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolov8_msgs.msg import InferenceResult, Yolov8Inference

bridge = CvBridge()

# Red color ranges in HSV (two ranges because red wraps around H=0/180)
RED_LOWER1 = np.array([0,   100, 60])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([160, 100, 60])
RED_UPPER2 = np.array([180, 255, 255])

# Minimum contour area to count as a detection (filters noise)
MIN_AREA = 200


class BalloonDetector(Node):
    def __init__(self):
        super().__init__('balloon_detector')

        self.declare_parameter('system_id', 1)
        self.declare_parameter('camera_topic', '/drone1/camera/image_raw')
        self.declare_parameter('detect_hz', 10.0)
        self.declare_parameter('processing_delay_ms', 200.0)  # simulated processing delay

        self.system_id    = self.get_parameter('system_id').value
        self.camera_topic = self.get_parameter('camera_topic').value
        detect_hz         = self.get_parameter('detect_hz').value
        self._delay_s     = self.get_parameter('processing_delay_ms').value / 1000.0

        # Ring buffer: deque of (wall_time_s, Image msg)
        self._frame_buf = collections.deque(maxlen=60)  # ~2 s at 30 Hz

        # QoS: depth=1, always keep only the latest frame
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self._store_frame, qos
        )

        self.yolov8_pub = self.create_publisher(
            Yolov8Inference, f'/Yolov8_Inference_{self.system_id}', 10
        )
        self.img_pub = self.create_publisher(
            Image, f'/inference_result_{self.system_id}', 10
        )

        # Fixed-rate detection timer — independent of camera publish rate
        self.create_timer(1.0 / detect_hz, self._detect)

        self.get_logger().info(
            f'Balloon Detector (OpenCV HSV) initialized | '
            f'topic={self.camera_topic} | detect_hz={detect_hz:.1f} | '
            f'processing_delay={self._delay_s*1000:.0f}ms'
        )

    # ── camera callback: push frame + arrival time into ring buffer ───────
    def _store_frame(self, msg: Image):
        now = self.get_clock().now().nanoseconds / 1e9
        self._frame_buf.append((now, msg))
        self.get_logger().info(
            '[DEBUG] Balloon detector: Camera callback triggered',
            throttle_duration_sec=5.0
        )

    # ── 10 Hz timer: publish detection for a frame captured ~delay_s ago ─
    def _detect(self):
        if not self._frame_buf:
            return   # no frame received yet

        now = self.get_clock().now().nanoseconds / 1e9

        # Find the newest frame that is at least _delay_s old
        msg = None
        for t, m in reversed(self._frame_buf):
            if now - t >= self._delay_s:
                msg = m
                break

        if msg is None:
            # All frames are too recent; nothing to publish yet (still warming up)
            return

        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # ── Red HSV detection ──────────────────────────────────────────
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
            mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
            mask  = cv2.bitwise_or(mask1, mask2)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
            # ──────────────────────────────────────────────────────────────

            yolov8_inference = Yolov8Inference()
            yolov8_inference.header = msg.header

            if valid:
                best = max(valid, key=cv2.contourArea)
                x1, y1, w, h = cv2.boundingRect(best)
                x2, y2 = x1 + w, y1 + h

                ir = InferenceResult()
                ir.class_name = 'red_balloon'
                ir.left   = x1
                ir.top    = y1
                ir.right  = x2
                ir.bottom = y2
                yolov8_inference.yolov8_inference.append(ir)

                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(cv_image, 'red_balloon', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if yolov8_inference.yolov8_inference:
                self.yolov8_pub.publish(yolov8_inference)
                det = yolov8_inference.yolov8_inference[0]
                self.get_logger().info(
                    f'[DEBUG] Publishing detection: 1 target(s), '
                    f'bbox=({det.left},{det.top},{det.right},{det.bottom}), '
                    f'topic=/Yolov8_Inference_{self.system_id}',
                    throttle_duration_sec=1.0
                )

            img_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            img_msg.header = msg.header
            self.img_pub.publish(img_msg)

        except Exception as e:
            self.get_logger().error(f'Error in _detect: {str(e)}')


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