# Gazebo Ignition Migration Guide

이 프로젝트는 Gazebo Classic에서 Gazebo Ignition (Gazebo Sim)으로 마이그레이션되었습니다.

## 주요 변경 사항

### 1. 패키지 의존성
- `gazebo_ros` → `ros_gz_sim`, `ros_gz_bridge`, `ros_gz_image`
- Gazebo Ignition용 ROS 2 패키지 추가

### 2. Launch 파일
- 새로운 launch 파일: [balloon_hunt_gazebo_ignition.launch.py](launch/balloon_hunt_gazebo_ignition.launch.py)
- `gazebo_ros` → `ros_gz_sim` 사용
- 환경 변수 변경:
  - `GAZEBO_MODEL_PATH` → `GZ_SIM_RESOURCE_PATH`
  - `PX4_SIM_MODEL`: `gazebo-classic_iris` → `gz_x500`

### 3. World 파일
- SDF 버전: 1.6 → 1.9
- Physics 설정 업데이트
- Gazebo Ignition 플러그인 추가:
  - `gz-sim-physics-system`
  - `gz-sim-user-commands-system`
  - `gz-sim-scene-broadcaster-system`
  - `gz-sim-sensors-system`
  - `gz-sim-imu-system`

### 4. 모델 파일
- SDF 버전: 1.5/1.6 → 1.9
- `iris_stereo_camera` 모델 완전 재작성
  - 스테레오 카메라 센서 통합
  - IMU 센서 추가
  - Multicopter motor model 플러그인 사용
- `red_balloon` 모델 SDF 버전 업데이트

### 5. ROS-Gazebo Bridge
카메라 및 센서 데이터를 ROS 2로 브리징:
```python
# Left Camera
/left_camera/image_raw
/left_camera/camera_info

# Right Camera
/right_camera/image_raw
/right_camera/camera_info

# IMU
/drone{id}/imu
```

## 설치 요구사항

### Gazebo Ignition 설치
```bash
# Ignition Fortress (ROS 2 Humble과 호환)
sudo apt-get update
sudo apt-get install ignition-fortress

# ROS-Gazebo 브리지
sudo apt-get install ros-humble-ros-gz-sim ros-humble-ros-gz-bridge ros-humble-ros-gz-image
```

**참고**: 현재 구현은 Ignition Fortress (Gazebo Sim 6)를 사용합니다. Garden 이상 버전을 사용하는 경우 일부 명령어가 다를 수 있습니다.

### 패키지 빌드
```bash
cd ~/visionws
colcon build --packages-select balloon_hunter
source install/setup.bash
```

## 실행 방법

### Gazebo Ignition으로 실행
```bash
ros2 launch balloon_hunter balloon_hunt_gazebo_ignition.launch.py
```

### 옵션 파라미터
```bash
ros2 launch balloon_hunter balloon_hunt_gazebo_ignition.launch.py \
  px4_src_path:=/path/to/PX4Swarm \
  drone_id:=1 \
  model_path:=/path/to/yolov8n.pt
```

## 마이그레이션 세부사항

### World 파일 변경사항
- Physics engine: `ode` → `ignored` (Gazebo Ignition이 자동 선택)
- 플러그인 시스템 추가 (필수)
- Material scripts 제거 (Ignition은 직접 material 정의 사용)
- Model URI: `model://name` → `name` (간소화)

### 모델 변경사항

#### iris_stereo_camera
- 완전히 새로운 SDF 파일 작성
- 센서가 별도 링크로 정의됨
- `relative_to` 속성 사용하여 좌표계 명확화
- Camera와 IMU 센서 직접 통합
- Motor model 플러그인 통합

#### red_balloon
- SDF 버전만 1.9로 업데이트
- 기본 구조는 유지

### 환경 변수 매핑

| Gazebo Classic | Gazebo Ignition |
|----------------|-----------------|
| `GAZEBO_MODEL_PATH` | `GZ_SIM_RESOURCE_PATH` |
| `GAZEBO_PLUGIN_PATH` | (플러그인은 모델 SDF에 직접 정의) |
| `GAZEBO_RESOURCE_PATH` | (더 이상 필요 없음) |
| `gazebo-classic_iris` | `gz_x500` (PX4 모델) |

## PX4-Gazebo Ignition 통합

PX4는 Gazebo Ignition을 공식 지원합니다:
- PX4 v1.14 이상에서 Gazebo Ignition 지원
- 모델명: `gz_x500`, `gz_x500_vision` 등
- 환경 변수: `PX4_SIM_MODEL=gz_x500`

### PX4 빌드 확인
```bash
cd ~/PX4Swarm
make px4_sitl gz_x500
```

## 문제 해결

### 1. 모델을 찾을 수 없음
```bash
export GZ_SIM_RESOURCE_PATH=/home/kiki/visionws/src/balloon_hunter/models:/home/kiki/visionws/src/balloon_hunter/worlds
```

### 2. 카메라 토픽이 안 보임
```bash
# Bridge가 제대로 실행되는지 확인
ros2 topic list | grep camera

# Ignition topic 확인
ign topic -l | grep camera
```

### 3. PX4가 "Gazebo gz please install gz-garden" 오류
이는 정상입니다. 현재 구현은 PX4의 Gazebo Classic 모드를 사용하면서 Ignition Gazebo를 별도로 실행합니다:
- `PX4_GZ_SIM_DISABLED=1` 환경 변수로 PX4의 Gazebo 자동 실행 비활성화
- PX4는 MAVLink 통신만 담당
- Gazebo Ignition은 launch 파일에서 별도 실행

### 4. Balloon collision plugin 오류
Gazebo Classic용 플러그인은 Ignition에서 작동하지 않습니다. 현재는 플러그인이 제거되어 있습니다:
- 충돌 감지는 ROS 2의 `collision_handler` 노드가 처리
- 향후 Ignition용 플러그인 개발 필요

### 5. Iris 메쉬 파일을 찾을 수 없음
메쉬 파일 대신 간단한 박스 geometry를 사용하도록 변경되었습니다:
- 실제 Iris 모델이 필요한 경우 PX4의 Gazebo 모델을 복사하거나
- `model://iris/meshes/iris.dae` 경로가 올바른지 확인

### 6. 성능 문제
Gazebo Ignition은 Classic보다 리소스를 더 사용할 수 있습니다:
```bash
# Render engine 변경 (ogre2 → ogre)
# world 파일에서 수정
<render_engine>ogre</render_engine>
```

## 참고 자료

- [Gazebo Sim Documentation](https://gazebosim.org/docs/garden)
- [ROS 2 Gazebo Integration](https://github.com/gazebosim/ros_gz)
- [PX4 Gazebo Simulation](https://docs.px4.io/main/en/sim_gazebo_gz/)
- [SDF Format Specification](http://sdformat.org/)

## 기존 Gazebo Classic 사용

기존 Gazebo Classic을 계속 사용하려면:
```bash
ros2 launch balloon_hunter balloon_hunt_gazebo.launch.py
```

## 향후 작업

- [ ] Balloon collision 플러그인을 Gazebo Ignition용으로 포팅
- [ ] 성능 최적화
- [ ] 추가 센서 통합 (GPS, 거리 센서 등)
- [ ] Multi-drone 시뮬레이션 테스트
