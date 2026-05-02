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

# ── 0. ROS2 Humble 기본 환경 소싱 ─────────────────────────────────────────
source /opt/ros/humble/setup.bash

# ── 1. RealSense 카메라 런치 (백그라운드) ──────────────────────────────────
echo "[run_balloon_hunt] 1) RealSense 카메라 실행 중..."
source ~/ros2_ws/camera/install/setup.bash
ros2 launch realsense2_camera rs_launch.py \
    pointcloud.enable:=false \
    depth_module.profile:=848x480x30 \
    rgb_camera.profile:=1280x720x30 \
    &
RS_PID=$!

# 카메라 토픽이 올라올 때까지 대기
echo "[run_balloon_hunt]    카메라 스트림 대기 (최대 15초)..."
for i in {1..15}; do
    if ros2 topic list 2>/dev/null | grep -q "/camera/camera/color/image_raw"; then
        echo "[run_balloon_hunt]    카메라 OK (${i}초 후 감지)"
        break
    fi
    sleep 1
done

# ── 2. Balloon Hunter 파이프라인 런치 (포그라운드) ─────────────────────────
echo "[run_balloon_hunt] 2) Balloon Hunter 파이프라인 실행"
source ~/ros2_ws/suicidedrone/install/setup.bash
ros2 launch balloon_hunter balloon_hunt_real.launch.py "$@"

# 위 ros2 launch가 Ctrl-C로 종료되면 trap이 걸려서 자동 정리됨
