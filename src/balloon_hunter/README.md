# Balloon Hunter

드론이 빨간색 풍선을 YOLOv8로 검출하고 추적하여 터트리는 PX4 + Gazebo 시뮬레이션 패키지

## 개요

이 패키지는 다음 시나리오를 **Gazebo 시뮬레이션**에서 자동으로 실행합니다:

1. **이륙 (TAKEOFF)**: 드론이 지정된 고도(5m)로 이륙
2. **전진 비행 (FORWARD)**: 드론이 직진으로 비행하며 타겟 탐색
3. **추적 (TRACKING)**: 빨간색 풍선 발견 시 위치 추정 및 추적
4. **돌진 (CHARGING)**: 풍선에 근접하면 전속력으로 돌진
5. **완료 (DONE)**: 풍선과 충돌 후 미션 완료

## 참고 코드

- **YOLO 검출**: `/home/kiki/joljak/src/uwb_reconn/src/yolobot_recognition/scripts/yolov8_ros2_pt.py`
- **위치 추정**: `/home/kiki/Downloads/Image2Pos/Image2Pos/box2image_ref_image_backup.py`
- **드론 제어**: `/home/kiki/joljak/src/uwb_reconn/src/drone_manager/drone_manager/drone_manager.py`
- **Gazebo 런치**: `/home/kiki/joljak/src/uwb_reconn/src/uwb_sim/launch/gazebo_typhoon_gazebo_world_run.launch.py`

## 🚀 빠른 시작

```bash
cd /home/kiki/visionws
source install/setup.bash
ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py
```

**한 줄 명령으로 Gazebo, PX4, MicroXRCE Agent, 모든 노드가 자동 실행됩니다!**

## 빌드

```bash
cd /home/kiki/visionws
colcon build --packages-select px4_msgs yolov8_msgs balloon_hunter --symlink-install
source install/setup.bash
```

## 실행 방법

### 방법 1: Gazebo 자동 실행 (권장) ⭐

```bash
ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py
```

파라미터:
- `px4_src_path`: PX4 경로 (기본: `/home/kiki/PX4-Autopilot`)
- `drone_id`: 드론 ID (기본: 1)
- `model_path`: YOLO 모델 경로

### 방법 2: 수동 실행

터미널 1 - PX4:
```bash
cd /home/kiki/PX4-Autopilot
make px4_sitl gazebo-classic_typhoon_h480
```

터미널 2 - MicroXRCE:
```bash
MicroXRCEAgent udp4 -p 8888
```

터미널 3 - 노드:
```bash
ros2 launch balloon_hunter balloon_hunt.launch.py
```

## 주요 토픽

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/Yolov8_Inference_1` | yolov8_msgs/Yolov8Inference | YOLO 검출 결과 |
| `/balloon_target_position` | geometry_msgs/PoseStamped | 풍선 위치 (NED) |
| `/inference_result_1` | sensor_msgs/Image | 검출 시각화 |
| `/balloon_collision` | std_msgs/Bool | 충돌 이벤트 |

## 모니터링

```bash
# 시각화
ros2 run rqt_image_view rqt_image_view /inference_result_1

# 토픽 확인
ros2 topic echo /balloon_target_position
ros2 topic echo /balloon_collision
```

## 상세 문서

자세한 내용은 패키지 내 README.md 참조
