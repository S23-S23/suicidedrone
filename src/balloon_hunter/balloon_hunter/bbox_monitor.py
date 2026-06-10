#!/usr/bin/env python3
"""
BBox Monitor (튜닝용)
=====================
balloon_detector가 발행하는 /target_info(TargetInfo)를 구독해서
현재 빨간 풍선 bounding box의 크기 정보를 터미널에 실시간 출력한다.

용도:
  충돌 판정(직충돌)에 쓸 "면적 비율(area_frac) 임계값"을 실측으로 정하기 위한 도구.
  실제 비행은 하지 않고, 카메라 + balloon_detector만 켜둔 상태에서 사용한다.

출력 항목:
  - bbox px (left, top, right, bottom)
  - w × h (px), area (px), area_frac (화면 대비 %)
  - center (u, v)
  - edge: 화면 경계에 접한 변 개수 (가까울수록 잘림 → 증가)
  - peak: 현재 탐지 세션에서의 최대 area_frac (소실 직전까지 얼마나 커졌나)
  - 소실 시: 마지막으로 본 값 + 경과시간 (1초 소실 로직 감 잡기용)

실행:
  # (터미널1) 카메라  : 사용자가 따로 실행
  # (터미널2) detector : ros2 run balloon_hunter balloon_detector --ros-args -p ...
  # (터미널3) 모니터   :
  ros2 run balloon_hunter bbox_monitor
  ros2 run balloon_hunter bbox_monitor --ros-args -p image_width:=1280 -p image_height:=720

주의:
  TargetInfo에는 영상 해상도 정보가 없으므로 image_width/height 파라미터로 준다.
  기본값은 run_suicide_drone.sh의 rgb_camera.profile(1280x720)에 맞춰져 있다.
"""

import rclpy
from rclpy.node import Node
from suicide_drone_msgs.msg import TargetInfo


class BBoxMonitor(Node):
    def __init__(self):
        super().__init__('bbox_monitor')

        # ── Parameters ──
        self.declare_parameter('target_info_topic', '/target_info')
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('lost_timeout', 1.0)   # 이 시간 이상 미검출 시 LOST 표시
        self.declare_parameter('print_rate', 5.0)     # 터미널 출력 주기 [Hz]

        topic            = self.get_parameter('target_info_topic').value
        self.img_w       = int(self.get_parameter('image_width').value)
        self.img_h       = int(self.get_parameter('image_height').value)
        self.lost_timeout = float(self.get_parameter('lost_timeout').value)
        print_rate       = float(self.get_parameter('print_rate').value)

        self.img_area = float(self.img_w * self.img_h)

        # ── Runtime state ──
        self.last_box     = None          # (L, T, R, B)
        self.last_time    = None          # rclpy Time of last detection
        self.session_peak = 0.0           # 현재 탐지 세션 최대 area_frac [%]
        self._was_lost    = True          # 직전에 LOST 상태였나 (세션 리셋용)

        # ── Subscriber (detector 발행 QoS와 동일: depth=10) ──
        self.create_subscription(TargetInfo, topic, self.target_cb, 10)

        # ── Print timer ──
        self.create_timer(1.0 / print_rate, self.print_status)

        self.get_logger().info(
            f'BBoxMonitor 시작: "{topic}" 구독  |  '
            f'image={self.img_w}x{self.img_h}  |  lost_timeout={self.lost_timeout}s'
        )
        print('-' * 92, flush=True)

    # ── Callback ──
    def target_cb(self, msg: TargetInfo):
        L, T, R, B = msg.left, msg.top, msg.right, msg.bottom

        # 새 탐지 세션 시작이면 피크 리셋
        if self._was_lost:
            self.session_peak = 0.0
            self._was_lost = False

        self.last_box  = (L, T, R, B)
        self.last_time = self.get_clock().now()

        frac = self._area_frac(L, T, R, B)
        if frac > self.session_peak:
            self.session_peak = frac

    # ── Helpers ──
    def _area_frac(self, L, T, R, B):
        w = max(0, R - L)
        h = max(0, B - T)
        return (w * h) / self.img_area * 100.0

    def _edges_touched(self, L, T, R, B):
        """bbox가 화면 경계에 접한 변 개수 (클리핑 정도 = 근접 정도)."""
        n = 0
        if L <= 0:              n += 1
        if T <= 0:              n += 1
        if R >= self.img_w - 1: n += 1
        if B >= self.img_h - 1: n += 1
        return n

    # ── Print (timer) ──
    def print_status(self):
        if self.last_time is None:
            print('[ -- ] 아직 탐지 없음 (detector/카메라 확인)', flush=True)
            return

        elapsed = (self.get_clock().now() - self.last_time).nanoseconds / 1e9
        L, T, R, B = self.last_box
        w, h = max(0, R - L), max(0, B - T)
        area = w * h
        frac = self._area_frac(L, T, R, B)
        cu, cv = (L + R) // 2, (T + B) // 2
        edges = self._edges_touched(L, T, R, B)

        if elapsed <= self.lost_timeout:
            print(
                f'[DET ] px=({L:4d},{T:4d},{R:4d},{B:4d})  '
                f'w×h={w:4d}×{h:4d}  area={area:7d}px  frac={frac:5.2f}%  '
                f'center=({cu:4d},{cv:4d})  edge={edges}  peak={self.session_peak:5.2f}%',
                flush=True,
            )
        else:
            self._was_lost = True   # 다음 탐지 때 세션 피크 리셋
            print(
                f'[LOST] {elapsed:4.1f}s 미검출  |  '
                f'마지막: w×h={w}×{h} area={area}px frac={frac:.2f}%  '
                f'(세션 peak={self.session_peak:.2f}%)',
                flush=True,
            )


def main(args=None):
    rclpy.init(args=args)
    node = BBoxMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
