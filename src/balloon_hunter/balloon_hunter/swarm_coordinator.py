#!/usr/bin/env python3
"""
Swarm Coordinator — publishes /swarm/start when ALL expected drones are ready.

Each drone_manager publishes its system_id to /swarm/ready once it is
ARMED + OFFBOARD while in IDLE state.  This node collects those signals
and fires /swarm/start only when the full set has reported in, guaranteeing
all drones depart at the same moment regardless of PX4 init timing.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SwarmCoordinator(Node):
    def __init__(self):
        super().__init__('swarm_coordinator')

        self.declare_parameter('expected_drone_ids', [1, 2, 3])
        self.declare_parameter('start_publish_count', 10)

        ids = self.get_parameter('expected_drone_ids').value
        self._expected = set(int(i) for i in ids)
        self._start_count = self.get_parameter('start_publish_count').value
        self._ready = set()
        self._started = False

        self._start_pub = self.create_publisher(String, '/swarm/start', 10)
        self.create_subscription(String, '/swarm/ready', self._ready_cb, 10)

        self.get_logger().info(
            f'SwarmCoordinator waiting for drones: {sorted(self._expected)}'
        )

    def _ready_cb(self, msg: String):
        try:
            drone_id = int(msg.data)
        except ValueError:
            return

        if drone_id not in self._expected:
            return

        if drone_id not in self._ready:
            self._ready.add(drone_id)
            self.get_logger().info(
                f'Drone {drone_id} ready  '
                f'({len(self._ready)}/{len(self._expected)})'
            )

        if not self._started and self._ready >= self._expected:
            self._started = True
            self.get_logger().info(
                'All drones ready — firing /swarm/start!'
            )
            # Publish several times to ensure all subscribers receive it.
            timer = self.create_timer(0.1, self._publish_start)
            self._publish_timer = timer
            self._published = 0

    def _publish_start(self):
        msg = String()
        msg.data = 'go'
        self._start_pub.publish(msg)
        self._published += 1
        if self._published >= self._start_count:
            self._publish_timer.cancel()
            self.get_logger().info('/swarm/start sent — coordinator done.')


def main(args=None):
    rclpy.init(args=args)
    node = SwarmCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
