#!/bin/bash
# ============================================================================
# Balloon Hunter — Real Flight Launch Script
# ----------------------------------------------------------------------------
# 1) RealSense 카메라 드라이버 실행
# 2) Balloon Hunter 파이프라인 (YOLO + DKF + IBVS + PNG + Drone Manager) 실행
#
# 사용법:
#   bash run_balloon_hunt.sh
#   bash run_balloon_hunt.sh v_max:=1.0 cam_pitch_deg:=30
# ============================================================================

set -e  # 에러 발생 시 스크립트 중단

# 스크립트 종료(Ctrl-C, 에러 등) 시 모든 자식 프로세스 함께 종료 (한 번만 실행)
_cleanup_done=0
cleanup() {
    [ "$_cleanup_done" = "1" ] && return
    _cleanup_done=1
    echo ''
    echo '[run_balloon_hunt] 종료 중... 모든 프로세스 정리'
    trap - EXIT INT TERM
    kill 0 2>/dev/null
}
trap cleanup EXIT INT TERM

# ── 경로 자동 탐지 ────────────────────────────────────────────────────────
# 이 스크립트는 <workspace>/src/run_suicide_drone.sh 에 위치한다고 가정.
# 따라서 워크스페이스 루트 = 스크립트 폴더의 상위. (어느 머신/경로에서든 동작)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# 카메라 워크스페이스는 환경변수 CAMERA_WS로 덮어쓸 수 있음 (기본: ~/ros2_ws/camera)
CAMERA_WS="${CAMERA_WS:-$HOME/ros2_ws/camera}"

if [ ! -f "${WS_ROOT}/install/setup.bash" ]; then
    echo "[run_balloon_hunt] ERROR: ${WS_ROOT}/install/setup.bash 없음. 먼저 빌드하세요:"
    echo "    cd ${WS_ROOT} && colcon build && source install/setup.bash"
    exit 1
fi

# ── 0. ROS2 Humble 기본 환경 소싱 ─────────────────────────────────────────
source /opt/ros/humble/setup.bash

# ── 1. RealSense 카메라 런치 (백그라운드) ──────────────────────────────────
echo "[run_balloon_hunt] 1) RealSense 카메라 실행 중... (camera_ws=${CAMERA_WS})"
source "${CAMERA_WS}/install/setup.bash"
ros2 launch realsense2_camera rs_launch.py \
    pointcloud.enable:=false \
    depth_module.profile:=848x480x30 \
    rgb_camera.profile:=1280x720x30 \
    &
RS_PID=$!

# 카메라 토픽이 올라올 때까지 대기
echo "[run_balloon_hunt]    카메라 스트림 대기 (최대 20초)..."
for i in {1..20}; do
    if ros2 topic list 2>/dev/null | grep -q "/camera/camera/color/image_raw"; then
        echo "[run_balloon_hunt]    카메라 OK (${i}초 후 감지)"
        break
    fi
    sleep 1
done

# ── 2. Balloon Hunter 파이프라인 런치 (포그라운드) ─────────────────────────
echo "[run_balloon_hunt] 2) Balloon Hunter 파이프라인 실행 (ws=${WS_ROOT})"
source "${WS_ROOT}/install/setup.bash"
ros2 launch balloon_hunter balloon_hunt_real.launch.py "$@"

# 위 ros2 launch가 Ctrl-C로 종료되면 trap이 걸려서 자동 정리됨
