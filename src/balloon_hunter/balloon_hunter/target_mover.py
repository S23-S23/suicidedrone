import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose
import random
import math

class TargetMover(Node):
    def __init__(self):
        super().__init__('target_mover')
        
        # 파라미터 선언
        self.declare_parameter('moving_target', False)
        self.declare_parameter('target_name', 'target_balloon')
        self.declare_parameter('move_speed', 0.5)      # m/s
        self.declare_parameter('move_interval', 2.0)   # 방향 전환 간격 (초)

        self.moving = self.get_parameter('moving_target').value
        self.target_name = self.get_parameter('target_name').value
        self.speed = self.get_parameter('move_speed').value
        self.interval = self.get_parameter('move_interval').value

        if not self.moving:
            self.get_logger().info('Moving target is disabled. Node exiting...')
            return

        # Gazebo 서비스 클라이언트 (Gazebo Classic 기준)
        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')

        # 초기 위치 설정 (world 파일 기준: 3, 10, 2)
        self.current_pos = [3.0, 10.0, 2.0]
        self.velocity = [0.0, 0.0, 0.0]

        # 타이머 설정
        self.create_timer(self.interval, self.update_direction)
        self.create_timer(0.05, self.update_position) # 20Hz로 위치 업데이트

    def update_direction(self):
        # 상하좌우(X, Y, Z) 랜덤 방향 벡터 생성
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        
        self.velocity[0] = self.speed * math.sin(phi) * math.cos(theta)
        self.velocity[1] = self.speed * math.sin(phi) * math.sin(theta)
        self.velocity[2] = self.speed * math.cos(phi) * 0.5 # 수직 이동은 조금 작게

    def update_position(self):
        # 위치 계산 (v * dt)
        self.current_pos[0] += self.velocity[0] * 0.05
        self.current_pos[1] += self.velocity[1] * 0.05
        self.current_pos[2] += self.velocity[2] * 0.05
        
        # Z축 최소 높이 제한 (바닥 뚫기 방지)
        self.current_pos[2] = max(0.5, self.current_pos[2])

        # Gazebo에 적용
        request = SetEntityState.Request()
        request.state.name = self.target_name
        request.state.pose.position.x = self.current_pos[0]
        request.state.pose.position.y = self.current_pos[1]
        request.state.pose.position.z = self.current_pos[2]
        
        self.client.call_async(request)

def main(args=None):
    rclpy.init(args=args)
    node = TargetMover()
    if node.moving:
        rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()