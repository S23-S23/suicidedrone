#!/usr/bin/env python3
"""
Swarm launch file — 3 drones in V formation, each attacking from a different angle.

Formation (top-down, balloon ahead at Gazebo (3,13)):

               🎈  balloon
                |
             drone1  (x=0,   y=0)       ← front center
            /        \\
        drone2         drone3
    (x=-2, y=-2.5)  (x=+2, y=-2.5)     ← back row

Drone spacing: ~3.2 m (drone1↔2, drone1↔3), 4 m (drone2↔3).

Each drone's initial yaw is set to point directly at the balloon,
giving each drone a slightly different LOS angle from the start.
Biased PNG (az_bias_deg) then reinforces the angular separation through
the terminal phase, so all three impact from measurably different azimuths.

  drone1: az_bias =   0°  (direct frontal approach)
  drone2: az_bias = -20°  (left-biased approach)
  drone3: az_bias = +20°  (right-biased approach)

Port assignments (no conflicts):
  drone1: TCP=4560, UDP=14560, PX4 instance=0, DDS=8888
  drone2: TCP=4561, UDP=14561, PX4 instance=1, DDS=8889
  drone3: TCP=4562, UDP=14562, PX4 instance=2, DDS=8890

Topic isolation: every mission node runs under namespace drone{N}:
  /droneN/target_info, /droneN/filter_estimate, /droneN/mission_state,
  /droneN/ibvs/output, /droneN/png/guidance_cmd
  (all relative topics in node code — no leading slash)

PX4 fmu topics namespaced by PX4_UXRCE_DDS_NS:
  droneN/fmu/out/..., droneN/fmu/in/...

Usage:
  ros2 launch balloon_hunter balloon_hunt_swarm.launch.py filter_type:=DKF
  ros2 launch balloon_hunter balloon_hunt_swarm.launch.py detector_type:=GT filter_type:=GT
"""
import os
from datetime import datetime
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import (
    ExecuteProcess, DeclareLaunchArgument, OpaqueFunction,
    IncludeLaunchDescription, SetEnvironmentVariable, TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource


# ── Per-drone static config ────────────────────────────────────────────────────
# Balloon at Gazebo ENU (x=3, y=13).
#
# V formation — each drone faces directly toward the balloon (individual yaw).
# Yaw in Gazebo ENU: measured from +X (East), CCW.  atan2(dy, dx) where
#   dy = balloon_y - drone_y (North component),  dx = balloon_x - drone_x (East).
#
#   drone1 (0, 0)    → (dx=3, dy=13)  → yaw = atan2(13, 3)  ≈ 1.346 rad
#   drone2 (-2,-2.5) → (dx=5, dy=15.5)→ yaw = atan2(15.5,5) ≈ 1.258 rad
#   drone3 (2, -2.5) → (dx=1, dy=15.5)→ yaw = atan2(15.5,1) ≈ 1.507 rad
#
# Drone spacing: drone1↔2 ≈ 3.2 m, drone1↔3 ≈ 3.2 m, drone2↔3 = 4.0 m.
#
# search_n/e: NED waypoint flown during SEARCH (midpoint between spawn and balloon).
# az_bias_deg: Biased PNG lateral offset — separates impact angles in terminal phase.
#
# (system_id, tcp_port, udp_port, px4_instance,
#  spawn_x, spawn_y, spawn_z, spawn_yaw_rad, dds_port,
#  search_n, search_e, az_bias_deg)

# ── Formation-Referenced Dynamic Bias (FRDB) swarm config ────────────────────
#
# Drone 1 is the leader: it flies straight and broadcasts its LOS direction.
# Drones 2 & 3 are followers: they compute az_bias from their signed lateral
# offset relative to drone 1's LOS line, using only RTK inter-drone positions:
#
#   e_perp_right = [-d1_e, d1_n]          (rightward perp to leader LOS in NED)
#   d_perp       = dot(my_pos - leader_pos, e_perp_right)   [m, RTK — exact]
#   az_bias      = atan2(d_perp, r_estimated) * K            [rad]
#
# V formation (drone2 left, drone3 right of drone1):
#   drone2: d_perp < 0  →  az_bias < 0  →  approaches from the left
#   drone3: d_perp > 0  →  az_bias > 0  →  approaches from the right
#
# Expected terminal impact pattern (top-down):
#
#      drone2 ↗         ↖ drone3
#              [balloon]
#                  ↑
#               drone1
#
# The bias decays geometrically to 0 as drones converge on the target.
# w (range weight) additionally shapes the S-curve profile.
# az_bias_deg is kept at 0 for all drones; it acts as a fallback only when
# leader LOS data is unavailable (e.g., leader hasn't detected the balloon yet).
#
# (system_id, tcp_port, udp_port, px4_instance,
#  spawn_x, spawn_y, spawn_z, spawn_yaw_rad, dds_port,
#  search_n, search_e, az_bias_deg)
DRONE_CONFIGS = [
    #                                                          search_n  search_e  az_bias_deg
    (1, 4560, 14560, 0,  0.0,  0.0,  0.1, 1.344, 8888,  7.0,   1.5,  0.0),
    (2, 4561, 14561, 1, -2.0, -2.5,  0.1, 1.258, 8889,  6.0,  -3.0, 0.0),  # flanked left
    (3, 4562, 14562, 2,  2.0, -2.5,  0.1, 1.507, 8890,  6.0,  +3.0, 0.0),  # flanked right
]

# FRDB + IACG parameters shared by all PNG nodes.
IACG_PARAMS = {
    'r0':               10.0,   # initial assumed range [m]
    'bias_decay_alpha':  2.0,   # 1 = linear decay; 0.5 = fast early; 2 = late snap
    'leader_id':         1,     # drone that broadcasts LOS direction
    'bias_gain_K':       3.5,   # amplification factor for geometric angle (~38° effective bias)
}


def make_drone_nodes(context, drone_id, filter_type, detector_type, model_path,
                     search_n, search_e, az_bias_deg, spawn_x, spawn_y):
    """Return the list of mission nodes for one drone, all under namespace drone{N}."""
    ns = f'drone{drone_id}'
    base_params = {'use_sim_time': True, 'system_id': drone_id}

    if detector_type == 'GT':
        detector_node = Node(
            namespace=ns,
            package='balloon_hunter',
            executable='gt_balloon_detector',
            name='balloon_detector',
            output='screen',
            parameters=[{
                **base_params,
                'camera_topic': f'/{ns}/camera/image_raw',
                'width': 848, 'height': 480,
                'fx': 454.8, 'fy': 454.8, 'cx': 424.0, 'cy': 240.0,
                'balloon_model_name': 'target_balloon',
                'balloon_radius': 0.3,
                'balloon_link_z_offset': 1.5,
                'camera_link_name': f'{ns}::depth_camera_link',
                'sensor_offset_x': 0.0,
                'sensor_offset_y': 0.0,
                'sensor_offset_z': 0.0,
            }]
        )
        effective_filter = 'GT'
    else:
        detector_node = Node(
            namespace=ns,
            package='balloon_hunter',
            executable='balloon_detector',
            name='balloon_detector',
            output='screen',
            parameters=[{
                **base_params,
                'camera_topic': f'/{ns}/camera/image_raw',
                'model_path': model_path,
                'conf': 0.5,
                'target_class': 'sports ball',
            }]
        )
        effective_filter = filter_type

    filter_node = Node(
        namespace=ns,
        package='balloon_hunter',
        executable='filter_node',
        name='filter_node',
        output='screen',
        parameters=[{
            **base_params,
            'filter_type': effective_filter,
            'fx': 454.8, 'fy': 454.8, 'cx': 424.0, 'cy': 240.0,
            'cam_pitch_deg': 0.0,
            'dkf_dt': 0.02,
            'dkf_delay_steps': 2,
            'assumed_depth': 10.0,   # matches IACG_PARAMS r0
        }]
    )

    ibvs_node = Node(
        namespace=ns,
        package='balloon_hunter',
        executable='ibvs_controller',
        name='ibvs_controller',
        output='screen',
        parameters=[{
            **base_params,
            'fx': 454.8, 'fy': 454.8, 'cx': 424.0, 'cy': 240.0,
            'fov_kp': 1.5, 'fov_kd': 0.1,
            'fov_kp_z': 1.5, 'fov_kd_z': 0.1,
            'target_timeout': 0.5,
        }]
    )

    png_node = Node(
        namespace=ns,
        package='balloon_hunter',
        executable='png_guidance',
        name='png_guidance',
        output='screen',
        parameters=[{
            **base_params,
            'Ky': 3.0, 'Kz': 3.0,
            'ka': 2.0,
            'v_max': 10.0,
            'v_init': 3.5,
            'rate': 50.0,
            'v_min_sigma': 0.5,
            'az_bias_deg': az_bias_deg,   # 0.0 for all; fallback when leader LOS unavailable
            'el_bias_deg': 0.0,
            **IACG_PARAMS,
        }]
    )

    manager_node = Node(
        namespace=ns,
        package='balloon_hunter',
        executable='drone_manager',
        name='drone_manager',
        output='screen',
        parameters=[{
            **base_params,
            'takeoff_height': 6.0,
            'forward_distance_limit': 50.0,
            'collision_distance': 1.5,
            'mission_timeout': 60.0,
            'max_speed': 10.0,
            # Per-drone NED search waypoint (midpoint; drone transitions to
            # INTERCEPT as soon as IBVS detects the target — no proximity gate).
            'search_target_n': search_n,
            'search_target_e': search_e,
            'search_arrival_dist': 3.0,
            # Disable auto-start timer; /swarm/start trigger fires all drones together.
            'start_delay_s': 9999.0,
            # Spawn position in Gazebo ENU — needed to convert /target_world_pos to local NED.
            'spawn_gazebo_x': spawn_x,
            'spawn_gazebo_y': spawn_y,
        }]
    )

    logger_node = Node(
        namespace=ns,
        package='balloon_hunter',
        executable='logger',
        name='logger',
        output='screen',
        parameters=[{
            **base_params,
            'filter_type': effective_filter,
            'spawn_gazebo_x': spawn_x,
            'spawn_gazebo_y': spawn_y,
            'fx': 454.8, 'fy': 454.8, 'cx': 424.0, 'cy': 240.0,
            'cam_pitch_deg': 0.0,
            'collision_distance': 1.5,
        }]
    )

    return [
        detector_node, filter_node, ibvs_node,
        png_node, manager_node, logger_node,
    ]


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('balloon_hunter')
    px4_src   = LaunchConfiguration('px4_src_path').perform(context)
    gz_classic = f'{px4_src}/Tools/simulation/gazebo-classic/sitl_gazebo-classic'
    model_path  = LaunchConfiguration('model_path').perform(context)
    filter_type = LaunchConfiguration('filter_type').perform(context)
    detector_type = LaunchConfiguration('detector_type').perform(context)
    bag_enable  = LaunchConfiguration('bag_enable').perform(context).lower()

    # ── Environment ──
    envs = [
        SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', '/usr/share/gazebo-11'),
        SetEnvironmentVariable(
            'GAZEBO_MODEL_PATH',
            f'{pkg}/models:{px4_src}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models'
            f':{gz_classic}/models'
        ),
        SetEnvironmentVariable(
            'GAZEBO_PLUGIN_PATH',
            f'{px4_src}/build/px4_sitl_default/build_gazebo-classic/'
        ),
        SetEnvironmentVariable('GZ_IP', '127.0.0.1'),
    ]

    # ── XRCE agents — one per PX4 instance, each on its own port ──
    # Using a single shared agent with multiple PX4 clients causes DDS session
    # conflicts in practice. Separate agents are the reliable approach.
    xrce_agents = [
        ExecuteProcess(
            cmd=['MicroXRCEAgent', 'udp4', '-p', str(cfg[8])],
            output='screen'
        )
        for cfg in DRONE_CONFIGS
    ]

    # ── Gazebo ──
    world_file = os.path.join(pkg, 'worlds', 'balloon_hunt.world')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch'),
            '/gazebo.launch.py'
        ]),
        launch_arguments={'world': world_file, 'verbose': 'false', 'gui': 'true'}.items()
    )

    # ── RViz ──
    rviz_cfg = os.path.join(pkg, 'config', 'drone_trajectory.rviz')
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_cfg], output='screen',
    )

    # ── Target mover (single balloon, global) ──
    target_mover = Node(
        package='balloon_hunter',
        executable='target_mover',
        name='target_mover',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'target_name': 'target_balloon',
            'nominal_x': 0.0, 'nominal_y': 15.0, 'nominal_z': 5.0,
            'amplitude': 0.0, 'speed': 0.0,
            'balloon_link_z_offset': 1.5,
        }]
    )

    # ── Single swarm visualizer (all 3 drones, color-coded trajectories) ──
    swarm_visualizer = Node(
        package='balloon_hunter',
        executable='swarm_visualizer',
        name='swarm_visualizer',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'system_ids': [cfg[0] for cfg in DRONE_CONFIGS],
            'max_path_points': 5000,
            'balloon_model_name': 'target_balloon',
            'balloon_radius': 0.3,
            'balloon_link_z_offset': 1.5,
        }]
    )

    # ── Per-drone: SDF gen, spawn, PX4, mission nodes ──
    sdf_gens   = []
    spawns     = []
    px4_procs  = []
    all_mission_nodes = []

    for (sys_id, tcp_port, udp_port, px4_inst,
         sx, sy, sz, syaw, dds_port,
         search_n, search_e, az_bias_deg) in DRONE_CONFIGS:
        ns = f'drone{sys_id}'

        # SDF generation from jinja template
        template = os.path.join(pkg, 'models', 'iris_depth_camera', 'iris_depth_camera.sdf.jinja')
        sdf_gen = ExecuteProcess(
            cmd=[
                f'{gz_classic}/scripts/jinja_gen.py',
                template, pkg,
                '--mavlink_tcp_port', str(tcp_port),
                '--mavlink_udp_port', str(udp_port),
                '--mavlink_id', str(sys_id),
                '--output-file', f'/tmp/drone_{sys_id}.sdf',
            ],
            output='screen'
        )
        sdf_gens.append(sdf_gen)

        # Gazebo spawn
        spawn = Node(
            package='gazebo_ros', executable='spawn_entity.py',
            arguments=[
                '-file', f'/tmp/drone_{sys_id}.sdf',
                '-entity', ns,
                '-x', str(sx), '-y', str(sy), '-z', str(sz),
                '-Y', str(syaw),
                '-robot_namespace', ns,
            ],
            output='screen'
        )
        spawns.append(spawn)

        # PX4 SITL instance — always create a fresh working dir (no stale params)
        romfs_base = f'{px4_src}/build/px4_sitl_default/ROMFS'
        romfs_dir  = f'{romfs_base}/instance{px4_inst}'
        mkdir_px4  = ExecuteProcess(
            cmd=['bash', '-c', f'rm -rf {romfs_dir} && mkdir -p {romfs_dir}'],
            output='screen'
        )
        sdf_gens.append(mkdir_px4)   # run immediately alongside SDF gen

        px4 = ExecuteProcess(
            cmd=[
                f'{px4_src}/build/px4_sitl_default/bin/px4',
                '-i', str(px4_inst),
                '-d', f'{px4_src}/build/px4_sitl_default/etc',
                '-w', romfs_dir,
            ],
            additional_env={
                'PX4_SIM_MODEL':        'gazebo-classic_iris',
                'PX4_UXRCE_DDS_NS':     ns,
                'PX4_UXRCE_DDS_PORT':   str(dds_port),
                'PX4_SYS_ID':           str(sys_id),
                'PX4_SIM_SPEED_FACTOR': '1',
            },
            output='screen'
        )
        px4_procs.append(px4)

        mission_nodes = make_drone_nodes(
            context, sys_id, filter_type, detector_type, model_path,
            search_n, search_e, az_bias_deg, sx, sy,
        )
        all_mission_nodes.extend(mission_nodes)

    # ── Timing ──
    #   t= 0s: Gazebo, XRCE agents, SDF gen, ROMFS mkdir — all immediate
    #   t= 3s: spawn drone1
    #   t= 5s: PX4 instance0 (drone1)
    #   t= 6s: spawn drone2
    #   t= 8s: PX4 instance1 (drone2)
    #   t= 9s: spawn drone3
    #   t=11s: PX4 instance2 (drone3)
    #   t=13s: target mover
    #   t=30s: mission nodes  ← extended to ensure all 3 EKFs converge in Gazebo
    timed_spawns = [
        TimerAction(period=3.0,  actions=[spawns[0]]),
        TimerAction(period=6.0,  actions=[spawns[1]]),
        TimerAction(period=9.0,  actions=[spawns[2]]),
    ]
    timed_px4 = [
        TimerAction(period=5.0,  actions=[px4_procs[0]]),
        TimerAction(period=8.0,  actions=[px4_procs[1]]),
        TimerAction(period=11.0, actions=[px4_procs[2]]),
    ]
    # swarm_coordinator: waits for all drone_managers to publish /swarm/ready
    # (ARM + OFFBOARD reached in IDLE), then fires /swarm/start automatically.
    # This replaces the fixed-timer approach and guarantees simultaneous departure.
    swarm_coordinator = Node(
        package='balloon_hunter',
        executable='swarm_coordinator',
        name='swarm_coordinator',
        output='screen',
        parameters=[{
            'expected_drone_ids': [cfg[0] for cfg in DRONE_CONFIGS],
            'start_publish_count': 10,
        }]
    )

    timed_target_mover = TimerAction(period=13.0, actions=[target_mover])
    timed_mission = TimerAction(
        period=30.0,
        actions=[*all_mission_nodes, swarm_visualizer, swarm_coordinator],
    )

    # ── ros2 bag (optional) ──
    bag_actions = []
    if bag_enable != 'false':
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        bag_dir   = LaunchConfiguration('bag_dir').perform(context)
        bag_path  = os.path.join(bag_dir, f'balloon_hunt_swarm_{timestamp}')
        os.makedirs(bag_dir, exist_ok=True)
        bag_actions.append(ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '--all', '--storage', 'mcap', '--output', bag_path],
            output='screen',
        ))

    return [
        *envs,
        *xrce_agents, gazebo, rviz,
        *sdf_gens,
        *timed_spawns,
        *timed_px4,
        timed_target_mover,
        timed_mission,
        *bag_actions,
    ]


def generate_launch_description():
    pkg = get_package_share_directory('balloon_hunter')
    return LaunchDescription([
        DeclareLaunchArgument('px4_src_path', default_value='/home/a/PX4Swarm'),
        DeclareLaunchArgument(
            'model_path',
            default_value=os.path.join(pkg, 'models', 'yolov8n.pt'),
        ),
        DeclareLaunchArgument('filter_type',   default_value='DKF'),
        DeclareLaunchArgument('detector_type', default_value='YOLO'),
        DeclareLaunchArgument(
            'bag_enable', default_value='true',
            description='Enable ros2 bag mcap recording: true | false'
        ),
        DeclareLaunchArgument(
            'bag_dir',
            default_value=os.path.normpath(os.path.join(
                pkg, '..', '..', '..', '..', 'log', 'rosbag')),
            description='Directory to save mcap bag files'
        ),
        OpaqueFunction(function=launch_setup)
    ])
