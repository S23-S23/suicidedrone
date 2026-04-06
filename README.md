# Balloon Hunter — Drone Interception Simulation

## Introduction

Balloon Hunter is a ROS 2 simulation of an autonomous drone that detects and physically intercepts a moving balloon target. The system is built on **PX4 SITL** with **Gazebo Classic** and implements a full perception–guidance–control pipeline based on Image-Based Visual Servoing (IBVS) and Proportional Navigation Guidance (PNG).

The drone operates through a five-stage finite state machine: **IDLE → TAKEOFF → FORWARD → INTERCEPT → DONE**. During FORWARD it searches for the balloon; once detected it transitions to INTERCEPT where IBVS computes line-of-sight (LOS) angles and FOV correction rates, which PNG converts into a 3D NED velocity command sent to PX4 in offboard mode. The mission ends when the drone physically collides with the balloon, detected via Gazebo ground-truth positions.

The system supports two detector back-ends:
- **GT mode** (`balloon_hunt_gt_gazebo.launch.py`): uses Gazebo model/link states for zero-latency ground-truth detection — ideal for guidance algorithm validation.
- **YOLO mode** (`balloon_hunt_gazebo.launch.py`): uses a YOLOv8 model for realistic onboard perception.

---

## Architecture

### Nodes

| Executable | Role |
|---|---|
| `gt_balloon_detector` | Projects balloon GT pose (Gazebo) onto image plane via pinhole model; publishes `TargetInfo` |
| `ibvs_controller` | Computes image-plane errors (ex, ey), LOS angles (q_y, q_z), FOV yaw/Z rates; publishes `IBVSOutput` |
| `png_guidance` | Discrete Proportional Navigation Guidance; produces NED velocity command `GuidanceCmd` |
| `drone_manager` | Mission FSM (IDLE→TAKEOFF→FORWARD→INTERCEPT→DONE); sends offboard setpoints to PX4 |
| `balloon_mover` | Moves the balloon in Gazebo on FORWARD entry; detects collision via GT ENU distance |
| `drone_visualizer` | RViz2 TF, trajectory markers, balloon marker (red→blue on hit), IBVS debug image overlay |

### Custom Messages (`suicide_drone_msgs`)

| Message | Fields | Flow |
|---|---|---|
| `TargetInfo` | `header`, `class_name`, `top`, `left`, `bottom`, `right` [px] | detector → ibvs, visualizer |
| `IBVSOutput` | `header`, `detected`, `q_y` [rad], `q_z` [rad], `fov_yaw_rate` [rad/s], `fov_vel_z` [m/s] | ibvs → png, visualizer |
| `GuidanceCmd` | `header`, `target_detected`, `vel_n/e/d` [m/s], `yaw_rate` [rad/s] | png → drone_manager |

### Mission FSM (`drone_manager`)

```
IDLE ──[5 s timer]──► TAKEOFF ──[|alt - target| < 0.3 m]──► FORWARD
                                                                 │  ▲
                                                    detected=True│  │detected=False
                                                                 ▼  │
                                                             INTERCEPT ──[/balloon_collision]──► DONE
```

| State | Control | Notes |
|---|---|---|
| IDLE | hover | waits 5 s for PX4 to boot |
| TAKEOFF | position setpoint `[x, y, -takeoff_height]` | arms drone, enables OFFBOARD |
| FORWARD | position hold | hovers at `hover_pos` if target previously acquired |
| INTERCEPT | position + velocity feedforward | pos setpoint = current pos (zero error); velocity = PNG output |
| DONE | position hold | collision latched; drone hovers indefinitely |

### IBVS + PNG Guidance (Reference: IEEE TIE 2025)

**IBVS (`ibvs_controller`):**

- **Eq. 3** — Image error:

$$e_x = \frac{u - c_x}{f_x}, \quad e_y = \frac{v - c_y}{f_y}$$

- **Eq. 5** — LOS unit vector:

$$\mathbf{n}_t = R_{eb} \, R_{bc} \begin{bmatrix} e_x \\ e_y \\ 1 \end{bmatrix}$$

- **Eq. 7** — LOS angles:

$$q_y = \text{atan2}(-n_{t,z},\ \|\mathbf{n}_{t,xy}\|), \quad q_z = \text{atan2}(n_{t,y},\ n_{t,x})$$

- **Eq. 13** — FOV yaw rate:

$$\dot{\psi}_{fov} = k_p \, e_x + k_d \left(-(1 + e_x^2)\, \omega_{z}^{body}\right)$$

- **Vertical** — FOV Z velocity:

$$v_{z,fov} = k_{p,z} \, e_y + k_{d,z} \left(-(1 + e_y^2)\, \omega_{y}^{body}\right)$$

**PNG (`png_guidance`):**

- **Eq. 8** — Current velocity direction $(\sigma_y,\, \sigma_z)$ derived from NED velocity $\mathbf{v}$ or body forward axis

- **Eq. 9** — Discrete PNG update:

$$\sigma_d = q_{now} + K \cdot \dot{q} \cdot \Delta t$$

- **Eq. 10** — Desired velocity unit vector:

$$\mathbf{n}_{vd} = \begin{bmatrix} \cos\sigma_{y,d}\cos\sigma_{z,d} \\ \cos\sigma_{y,d}\sin\sigma_{z,d} \\ -\sin\sigma_{y,d} \end{bmatrix}$$

- **Eq. 14** — Speed ramp:

$$v \leftarrow \min\!\left(v + \frac{k_a}{f_{rate}},\ v_{max}\right)$$

- **Final output:**

$$\mathbf{v}_{NED} = v\,\mathbf{n}_{vd}, \quad \dot{\psi} = \dot{\psi}_{fov}, \quad v_{d} \mathrel{+}= v_{z,fov}$$

---

## HOW TO RUN

### Prerequisites

- ROS 2 Humble
- PX4 SITL (built at `px4_src_path` with target `px4_sitl_default`)
- Gazebo Classic 11 + `ros-humble-gazebo-ros-pkgs`
- `MicroXRCEAgent` on PATH
- `ros-humble-rosbag2-storage-mcap` (for bag recording)

### 1. Build

```bash
cd ~/Projects/suicidedrone
colcon build --packages-select suicide_drone_msgs balloon_hunter
source install/setup.bash
```

### 2. Launch — Ground Truth mode

```bash
ros2 launch balloon_hunter balloon_hunt_gt_gazebo.launch.py
```

Starts (in order): MicroXRCE Agent, Gazebo, PX4 Jinja SDF generation, drone spawn, PX4 SITL, all ROS 2 nodes, RViz2, ros2 bag record.

### 3. Launch Arguments

| Argument | Default | Description |
|---|---|---|
| `px4_src_path` | `/home/user/Projects/PX4Swarm` | Path to the PX4 source tree |
| `drone_id` | `1` | Drone instance ID (affects topic prefix `/drone{id}/...`) |
| `move` | `left` | Balloon movement pattern: `left` \| `right` \| `up` \| `down` \| `random` \| `none` |
| `ibvs_kp_z` | `1.5` | IBVS vertical error (ey) proportional gain |
| `ibvs_kd_z` | `0.1` | IBVS vertical error derivative gain |
| `png_v_max` | `3.0` | PNG maximum intercept speed [m/s] |
| `png_v_init` | `1.5` | PNG speed at INTERCEPT entry [m/s] |
| `png_ka` | `2.0` | PNG speed ramp-up rate [m/s²] |
| `png_Ky` | `2.0` | PNG elevation navigation gain |
| `png_Kz` | `4.0` | PNG azimuth navigation gain |
| `bag_enable` | `true` | Enable rosbag2 recording in mcap format |
| `bag_dir` | `<workspace>/log/rosbag` | Output directory for bag files |

### 4. Examples

```bash
# Balloon moves right, no bag recording
ros2 launch balloon_hunter balloon_hunt_gt_gazebo.launch.py \
  move:=right bag_enable:=false

# Random balloon evasion, faster intercept
ros2 launch balloon_hunter balloon_hunt_gt_gazebo.launch.py \
  move:=random png_v_max:=5.0 png_v_init:=2.0 png_ka:=3.0

# Stationary balloon, tune IBVS vertical gain only
ros2 launch balloon_hunter balloon_hunt_gt_gazebo.launch.py \
  move:=none ibvs_kp_z:=2.0 ibvs_kd_z:=0.05 bag_enable:=false

# Custom PX4 path, save bags to /tmp
ros2 launch balloon_hunter balloon_hunt_gt_gazebo.launch.py \
  px4_src_path:=/opt/PX4-Autopilot bag_dir:=/tmp/bags
```

### 5. Bag Playback

Bags are saved in mcap format under `<workspace>/log/rosbag/balloon_hunt_<timestamp>/`.

```bash
ros2 bag info log/rosbag/balloon_hunt_2026-04-06_12-00-00
ros2 bag play log/rosbag/balloon_hunt_2026-04-06_12-00-00
```
