import rclpy
import numpy as np

class Formation:
    def __init__(self, node: rclpy.node.Node, drone_id, distance, angle_radian):
        self.node = node
        self.drone_id = drone_id
        self.distance = distance
        self.angle_radian = angle_radian

    def calculate_position(self):
        x_leader, y_leader, yaw_leader = self.node.leader_pose_msg.rtk_n, self.node.leader_pose_msg.rtk_e, self.node.leader_pose_msg.head

        if self.drone_id % 2 == 0:
            layer = self.drone_id // 2
            theta = self.angle_radian
        else:
            layer = (self.drone_id - 1) // 2
            theta = -self.angle_radian

        distance = self.distance * layer
        target_radian = yaw_leader + np.pi + theta

        target_x = (x_leader + distance * np.cos(target_radian)) - self.node.monitoring_msg.rtk_n
        target_y = (y_leader + distance * np.sin(target_radian)) - self.node.monitoring_msg.rtk_e

        return [target_x, target_y, self.node.leader_pose_msg.rtk_d]

    def calculate_yaw(self):
        if self.drone_id % 2 == 0:
            return self.node.leader_pose_msg.head + self.angle_radian
        else:
            return self.node.leader_pose_msg.head - self.angle_radian
