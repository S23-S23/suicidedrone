# Balloon Hunter

드론이 빨간색 풍선을 YOLOv8로 검출하고 추적하여 터트리는 PX4 + Gazebo + rviz 시뮬레이션 패키지


##  시작

```bash
ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py detector_type:=GT
```

```bash
ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py filter_type:=DKF
```

```bash
ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py filter_type:=EKF 
```


## 설정 방법

PX4 version : https://github.com/SUV-Lab/PX4Swarm/tree/v1.14/forest-recon
YOLOv8n 설치

파라미터:
- `px4_src_path`: PX4Swarm 경로 (예시: `default_value='/home/user/PX4Swarm'`)
- `drone_id`: 드론 ID (기본: 1)
- `model_path`: YOLO 모델 경로
