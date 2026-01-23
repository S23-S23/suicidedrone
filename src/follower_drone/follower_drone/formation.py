import rclpy
import numpy as np

class Formation:
    def __init__(self, node: rclpy.node.Node, drone_id, distance, angle_radian, takeoff_offset):
        self.node = node
        self.drone_id = drone_id
        self.distance = distance
        self.angle_radian = angle_radian
        # test 때 삭제
        self.takeoff_offset = takeoff_offset

    def calculate_position(self):
        # test 때 rtk_ned로 변경 필요
        x_leader, y_leader, yaw_leader = self.node.leader_monitoring_msg.pos_x, self.node.leader_monitoring_msg.pos_y, self.node.leader_monitoring_msg.head

        if self.drone_id % 2 == 0:
            layer = self.drone_id // 2
            theta = -self.angle_radian
        else:
            layer = (self.drone_id - 1) // 2
            theta = self.angle_radian

        distance = self.distance * layer
        target_radian = yaw_leader + np.pi + theta

        target_x = (x_leader + distance * np.cos(target_radian)) + self.takeoff_offset[0]
        target_y = (y_leader + distance * np.sin(target_radian)) + self.takeoff_offset[1]

        return [target_x, target_y, self.node.leader_monitoring_msg.pos_z]
