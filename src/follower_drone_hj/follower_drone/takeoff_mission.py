import rclpy
from rclpy.qos import qos_profile_sensor_data
from px4_msgs.msg import VehicleCommand, TrajectorySetpoint
from follower_drone.mode_handler import Mode

from enum import Enum
import math

class MonitoringFlagType(Enum):
        SAFETY_LOCK_STATUS = 0
        ARM_STATUS = 1
        OFFBOARD_MODE = 2
        MANUAL_MODE = 3
        AUTO_MODE = 4

class ProgressStatus(Enum):
    DISARM=0
    ARM=1
    OFFBOARD=2
    TAKEOFF=3
    Done=4

class TakeoffMission():
    def __init__(self, node: rclpy.node.Node, system_id: int, takeoff_altitude: float):
        self.node = node
        self.system_id = system_id
        self.takeoff_altitude = takeoff_altitude
        self.disarmPos=[0,0]
        self.currentProgressStatus=ProgressStatus.DISARM

        self.vehicle_command_publisher_ = self.node.create_publisher(
            VehicleCommand,
            f'/drone{self.system_id}/fmu/in/vehicle_command',
            qos_profile_sensor_data
        )
        self.traj_setpoint_publisher_ = self.node.create_publisher(
            TrajectorySetpoint,
            f'/drone{self.system_id}/fmu/in/trajectory_setpoint',
            qos_profile_sensor_data
        )

        self.timer_mission_ = self.node.create_timer(0.5, self.timer_mission_callback)


    def timer_mission_callback(self):
        if not self.node.monitoring_msg.pos_x:
            return
        # print("Current Progress :", self.currentProgressStatus)
        if self.currentProgressStatus == ProgressStatus.DISARM:
            self.currentProgressStatus=ProgressStatus(self.currentProgressStatus.value + 1)
            self.disarmPos[0]=self.POSX()
            self.disarmPos[1]=self.POSY()

        if self.currentProgressStatus == ProgressStatus.ARM:
            if not self.isArmed():
                msg = VehicleCommand()
                msg.target_system = self.system_id
                msg.command = 400 # MAV_CMD_COMPONENT_ARM_DISARM=400,
                msg.param1 = 1.0
                msg.confirmation=True
                msg.from_external = True
                self.vehicle_command_publisher_.publish(msg)
            else:
                self.currentProgressStatus=ProgressStatus(self.currentProgressStatus.value + 1)

        if self.currentProgressStatus == ProgressStatus.OFFBOARD:
            if not self.isOffboard():
                msg = VehicleCommand()
                msg.target_system = self.system_id
                msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
                msg.param1 = 1.0
                msg.param2 = 6.0 # PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6,
                msg.from_external = True
                self.vehicle_command_publisher_.publish(msg)
            else:
                self.currentProgressStatus=ProgressStatus(self.currentProgressStatus.value + 1)

        if self.currentProgressStatus == ProgressStatus.TAKEOFF:
            setpoint=[self.disarmPos[0], self.disarmPos[1], -self.takeoff_altitude]
            success, distance = self.isOnSetpoint(setpoint)
            if not success:
                self.setpoint(setpoint)
                self.node.get_logger().info(f"distance: {distance}")
                # print(f"drone : {setpoint}")
            else:
                self.currentProgressStatus=ProgressStatus(self.currentProgressStatus.value + 1)
        if self.currentProgressStatus == ProgressStatus.Done:
            self.node.mode_handler.change_mode(Mode.FORMATION)
            # print("Current Progress :", self.currentProgressStatus)
            self.node.destroy_timer(self.timer_mission_)
            self.timer_mission_ = None

    def POSX(self):
        return self.node.monitoring_msg.pos_x

    def POSY(self):
        return self.node.monitoring_msg.pos_y

    def POS(self):
        return [self.node.monitoring_msg.pos_x,
                self.node.monitoring_msg.pos_y,
                self.node.monitoring_msg.pos_z]
    def distance(self, pos1, pos2):
        posDiff = [pos1[0]-pos2[0], pos1[1]-pos2[1], pos1[2]-pos2[2]]
        distance = math.sqrt(posDiff[0]**2 + posDiff[1]**2 +  posDiff[2]**2)
        return distance

    def isArmed(self):
        return self.monitoringFlag(self.node.monitoring_msg.status1, MonitoringFlagType.ARM_STATUS.value)

    def isOffboard(self):
        return self.monitoringFlag(self.node.monitoring_msg.status1, MonitoringFlagType.OFFBOARD_MODE.value)

    def isOnSetpoint(self, targetPOS):
        distance = self.distance(self.POS(), targetPOS)
        return (distance < 0.2), distance

    def setpoint(self, setpoint, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position[0] = setpoint[0]
        msg.position[1] = setpoint[1]
        msg.position[2] = setpoint[2]
        msg.yaw = yaw
        msg.yawspeed = 0.0
        self.traj_setpoint_publisher_.publish(msg)
        self.node.get_logger().info("Publishing Takeoff Setpoint")

    def monitoringFlag(self, aValue, aBit):
        return (aValue & (1<<aBit)) > 0
