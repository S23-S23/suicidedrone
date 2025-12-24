# Balloon Hunter

드론이 빨간색 풍선을 YOLOv8로 검출하고 추적하여 터트리는 PX4 + Gazebo 시뮬레이션 패키지

## 개요

이 패키지는 다음 시나리오를 **Gazebo 시뮬레이션**에서 자동으로 실행합니다:

1. **이륙 (TAKEOFF)**: 드론이 지정된 고도(5m)로 이륙
2. **전진 비행 (FORWARD)**: 드론이 직진으로 비행하며 타겟 탐색
3. **추적 (TRACKING)**: 빨간색 풍선 발견 시 위치 추정 및 추적
4. **돌진 (CHARGING)**: 풍선에 근접하면 전속력으로 돌진
5. **완료 (DONE)**: 풍선과 충돌 후 미션 완료


##  시작

```bash
ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py
```


## 설정 방법


파라미터:
- `px4_src_path`: PX4Swarm 경로 (예시: `default_value='/home/user/PX4Swarm'`)
- `drone_id`: 드론 ID (기본: 1)
- `model_path`: YOLO 모델 경로
