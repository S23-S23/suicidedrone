import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import VehicleCommand, OffboardControlMode, TrajectorySetpoint, Monitoring

from follower_drone.takeoff_mission import TakeoffMission
from follower_drone.formation import Formation
from follower_drone.mode_handler import ModeHandler, Mode

import math
import numpy as np

class DroneManager(Node):
    def __init__(self):
        super().__init__('drone_manager')
        self.declare_parameter('system_id', 1)
        self.declare_parameter('formation_degree', 30.0)
        self.declare_parameter('formation_distance', 3.0)

        self.system_id = self.get_parameter('system_id').get_parameter_value().integer_value
        self.formation_degree = self.get_parameter('formation_degree').get_parameter_value().double_value
        self.formation_distance = self.get_parameter('formation_distance').get_parameter_value().double_value

        self.monitoring_msg = Monitoring()
        self.leader_monitoring_msg = Monitoring()
        self.formation_radian = math.radians(self.formation_degree/2.0)

        self.mode_handler = ModeHandler(self)
        self.mode_handler.change_mode(Mode.TAKEOFF)
        self.takeoff_mission = None

        self.ocm_publisher_ = self.create_publisher(
            OffboardControlMode,
            f'/drone{self.system_id}/fmu/in/offboard_control_mode', qos_profile_sensor_data)
        self.traj_setpoint_publisher_ = self.create_publisher(
            TrajectorySetpoint,
            f'/drone{self.system_id}/fmu/in/trajectory_setpoint',
            qos_profile_sensor_data
        )

        self.monitoring_subscriber_ = self.create_subscription(
            Monitoring,
            f'/drone{self.system_id}/fmu/out/monitoring',
            self.monitoring_callback,
            qos_profile_sensor_data
        )
        # 임시
        self.leader_monitoring_subscriber_ = self.create_subscription(
            Monitoring,
            f'/drone1/fmu/out/monitoring',
            self.leader_monitoring_callback,
            qos_profile_sensor_data
        )

        self.timer_mission_ = self.create_timer(0.05, self.timer_mission_callback)
        self.timer_ocm_ = self.create_timer(0.1, self.timer_ocm_callback)

        self.get_logger().info(f'drone_manager_{self.system_id} initialized complete.')


    def timer_ocm_callback(self):
        ocm_msg = OffboardControlMode()
        ocm_msg.position = True
        ocm_msg.velocity = False
        ocm_msg.acceleration = False
        ocm_msg.attitude = False
        ocm_msg.body_rate = False

        self.ocm_publisher_.publish(ocm_msg)

    def timer_mission_callback(self):
        if self.mode_handler.is_in_mode(Mode.TAKEOFF):
            if self.takeoff_mission is None:
                # self.get_logger().info(f"drone{self.system_id} Takeoff Mission Start")
                self.takeoff_mission = TakeoffMission(self, self.system_id, 2.0) # Takeoff to 2.0 meters
        if self.mode_handler.is_in_mode(Mode.FORMATION):
            self.handle_formation()

    def handle_formation(self):
        takeoff_offset = LLH2NED([self.leader_monitoring_msg.ref_lat, self.leader_monitoring_msg.ref_lon, self.leader_monitoring_msg.ref_alt],
                                 [self.monitoring_msg.ref_lat, self.monitoring_msg.ref_lon, self.monitoring_msg.ref_alt])
        self.formation = Formation(
            self,
            self.system_id,
            self.formation_distance,
            self.formation_radian,
            takeoff_offset
            )
        setpoint = TrajectorySetpoint()
        setpoint.position = self.formation.calculate_position()
        self.traj_setpoint_publisher_.publish(setpoint)

    def monitoring_callback(self, msg):
        self.monitoring_msg = msg

    def leader_monitoring_callback(self, msg):
        self.leader_monitoring_msg = msg

# WGS-84
a = 6378137.0
f = 1.0 / 298.257223563
e2 = 2 * f - f * f

def NED2LLH(NED, ref_LLH):
    lat_ref = np.deg2rad(ref_LLH[0])
    lon_ref = np.deg2rad(ref_LLH[1])

    sin_lat_ref = np.sin(lat_ref)
    cos_lat_ref = np.cos(lat_ref)

    N_ref = a / np.sqrt(1 - e2 * sin_lat_ref**2)

    dlat = NED[0] / N_ref
    dlon = NED[1] / (N_ref * cos_lat_ref)

    lat = lat_ref + dlat
    lon = lon_ref + dlon
    h = ref_LLH[2] + NED[2]

    return [np.rad2deg(lat), np.rad2deg(lon), h]

def LLH2NED(LLH, ref_LLH):
    lat_ref = np.deg2rad(ref_LLH[0])
    lon_ref = np.deg2rad(ref_LLH[1])
    lat = np.deg2rad(LLH[0])
    lon = np.deg2rad(LLH[1])

    sin_lat_ref = np.sin(lat_ref)
    cos_lat_ref = np.cos(lat_ref)

    N_ref = a / np.sqrt(1 - e2 * sin_lat_ref**2)

    dlat = lat - lat_ref
    dlon = lon - lon_ref

    NED_N = dlat * N_ref
    NED_E = dlon * N_ref * cos_lat_ref
    NED_D = LLH[2] - ref_LLH[2]

    return [NED_N, NED_E, NED_D]

def main(args=None):
    rclpy.init(args=args)
    drone_manager = DroneManager()
    rclpy.spin(drone_manager)
    drone_manager.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
