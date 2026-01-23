import os
from jinja2 import Environment, FileSystemLoader
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import FindExecutable
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)

def launch_setup(context, *args, **kwargs):
    # Configuration
    current_package_path = get_package_share_directory('follower_drone')
    leader_drone_package_path = get_package_share_directory('balloon_hunter')
    px4_src_path = LaunchConfiguration('px4_src_path').perform(context)
    gazebo_classic_path = f'{px4_src_path}/Tools/simulation/gazebo-classic/sitl_gazebo-classic'
    world_type = LaunchConfiguration('world_type').perform(context)
    px4_sim_env = SetEnvironmentVariable('PX4_SIM_MODEL', f'gazebo-classic_iris')
    px4_lat = SetEnvironmentVariable('PX4_HOME_LAT', f'36.6299')
    px4_lon = SetEnvironmentVariable('PX4_HOME_LON', f'127.4588')
    num_drone = int(LaunchConfiguration('num_drone').perform(context))

    # Environments
    # resource_path_env = SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', f'/usr/share/gazebo-11')
    # model_path_env = SetEnvironmentVariable('GAZEBO_MODEL_PATH',
    #                                         f'{current_package_path}:{current_package_path}/models:{gazebo_classic_path}/models')
    # plugin_path_env = SetEnvironmentVariable('GAZEBO_PLUGIN_PATH',
    #                                         f'{px4_src_path}/build/px4_sitl_default/build_gazebo-classic/')

    # MicroXRCEAgent udp4 -p 8888
    xrce_agent_process = ExecuteProcess(
        cmd=[FindExecutable(name='MicroXRCEAgent'), 'udp4', '-p', '8888'],
        #output='screen',
    )

    # # gazebo world
    # env = Environment(loader=FileSystemLoader(os.path.join(leader_drone_package_path, 'worlds')))
    # jinja_world = env.get_template(f'{world_type}.world.jinja')
    # simulation_world = jinja_world.render()
    # world_file_path = os.path.join('/tmp', 'output.world')
    # with open(world_file_path, 'w') as f:
    #     f.write(simulation_world)

    # gazebo_node = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([os.path.join(
    #         get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py']),
    #     launch_arguments={'world': world_file_path, 'verbose':'false', 'gui':'true' }.items()
    # )

    drone_process_list = []
    for i in range(num_drone):
        jinja_cmd = [
            f'{gazebo_classic_path}/scripts/jinja_gen.py',
            f'{current_package_path}/models/iris_fisheye_lens_camera/iris_fisheye_lens_camera.sdf.jinja',
            f'{current_package_path}',
            '--mavlink_tcp_port', f'{4560+i+1}',
            '--mavlink_udp_port', f'{14560+i+1}',
            '--mavlink_id', f'{1+i+1}',
            '--gst_udp_port' , f'{5600+i+1}' ,
            '--video_uri', f'{5600+i+1}',
            '--mavlink_cam_udp_port', f'{14530+i+1}',
            '--output-file', f'/tmp/model_{i}.sdf'
        ]
        jinja_process = ExecuteProcess(
            cmd=jinja_cmd,
            #output='screen',
        )
        drone_process_list.append(jinja_process)

        # node to spawn robot model in gazebo
        # ## random
        # agent_x = -12.0 + np.random.uniform(5, 7)*i
        # agent_y = -15.0 + np.random.uniform(-3, 3)

        ## one line without leader drone
        grid_idx = i if i < 2 else i + 1

        agent_x = -10.0 + 2.0 * grid_idx
        agent_y = -15.0

        # ## square
        # agent_x = 3.0 * (-1)**i
        # agent_y = 3.0 * (-1 if i%3==0 else 1)

        # ## diagonal
        # agent_x = -6.0 + 3*i
        # agent_y = -25.0 + 1*i

        if i == 0 :
            ref_agent_x = agent_x # rviz 용
            ref_agent_y = agent_y
        spawn_entity_node = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-file', f'/tmp/model_{i}.sdf', '-entity', f'robot_{i}', '-x', f'{agent_x}', '-y', f'{agent_y}','-z', '0.4'],
            #output='screen',
            )

        drone_process_list.append(spawn_entity_node)

        # PX4
        # build_path/bin/px4 -i $N -d "$build_path/etc" >out.log 2>err.log &
        cmd = [
            'env',
            'PX4_SIM_MODEL=gazebo-classic_iris',
            f'{px4_src_path}/build/px4_sitl_default/bin/px4',
            '-i', f'{i+1}',
            '-d',
            f'{px4_src_path}/build/px4_sitl_default/etc',
            '-w', f'{px4_src_path}/build/px4_sitl_default/ROMFS/instance{i+1}',
            '>out.log', '2>err.log',
        ]

        px4_process = ExecuteProcess(
            cmd=cmd,
            output='screen',
        )
        drone_process_list.append(px4_process)

        manager = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            current_package_path, 'launch'), '/follower_drone.launch.py']),
        launch_arguments={
            'drone_id': f'{i+2}',
            }.items()
        )

        drone_process_list.append(manager)

    node_run  = [
        px4_lat,
        px4_lon,
        # resource_path_env,
        px4_sim_env,
        # model_path_env,
        # plugin_path_env,
        # xrce_agent_process,
        # gazebo_node,
        *drone_process_list,
    ]

    return node_run

def generate_launch_description():
    current_package_path = get_package_share_directory('follower_drone')
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            'px4_src_path',
            default_value='/home/jun/PX4Swarm',
            description='px4 source code path'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_type',
            default_value='iris_fisheye_lens_camera',
            description='Type of the robot to spawn'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'world_type',
            default_value='balloon_hunt',
            description='Type of the world to spawn'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'num_drone',
            default_value='4',
            description='Number of drone to spawn'
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
