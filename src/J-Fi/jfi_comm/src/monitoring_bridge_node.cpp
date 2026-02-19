#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include "jfi_comm/msg/swarm_comm.hpp"
#include "jfi_comm/msg/pos_yaw.hpp"
#include "px4_msgs/msg/monitoring.hpp"
#include "px4_msgs/msg/vehicle_odometry.hpp"

class MonitoringBridgeNode : public rclcpp::Node
{
public:
  MonitoringBridgeNode() : Node("monitoring_bridge_node"), seq_(0)
  {
    // Subscribe
    monitoring_sub_ = this->create_subscription<px4_msgs::msg::Monitoring>(
      "/drone1/fmu/out/monitoring",
      rclcpp::SensorDataQoS(),
      std::bind(&MonitoringBridgeNode::monitoring_callback, this, std::placeholders::_1)
    );
    odometry_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
      "/drone1/fmu/out/vehicle_odometry",
      rclcpp::SensorDataQoS(),
      std::bind(&MonitoringBridgeNode::odometry_callback, this, std::placeholders::_1)
    );

    // Publish Timer
    auto jfi_period = std::chrono::milliseconds(50);  // 20Hz
    jfi_timer_ = this->create_wall_timer(
      jfi_period, std::bind(&MonitoringBridgeNode::jfi_timer_callback, this)
    );

    // Publisher
    jfi_pub_ = this->create_publisher<jfi_comm::msg::SwarmComm>(
      "jfi_comm/in/packet", 10
    );

    velocity_.resize(3, 0.0f);
    RCLCPP_INFO(this->get_logger(), "Monitoring Bridge Node started");
  }

private:
  void odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr odometry_msg)
  {
    velocity_[0] = odometry_msg->velocity[0];
    velocity_[1] = odometry_msg->velocity[1];
    velocity_[2] = odometry_msg->velocity[2];
  }
  void monitoring_callback(const px4_msgs::msg::Monitoring::SharedPtr monitoring_msg)
  {
    if (velocity_.size() != 3) {
      RCLCPP_WARN(this->get_logger(), "Velocity data not yet received. Skipping monitoring message.");
      return;
    }
    pos_yaw_msg_.head = monitoring_msg->head;
    pos_yaw_msg_.rtk_n = monitoring_msg->rtk_n;
    pos_yaw_msg_.rtk_e = monitoring_msg->rtk_e;
    pos_yaw_msg_.pos_z = monitoring_msg->pos_z;
    pos_yaw_msg_.velocity[0] = velocity_[0];
    pos_yaw_msg_.velocity[1] = velocity_[1];
    pos_yaw_msg_.velocity[2] = velocity_[2];
  }

  void jfi_timer_callback() {
    // Topic 직렬화
    RCLCPP_INFO_ONCE(this->get_logger(), "Publishing RTK data to jfi_comm/in/packet");
    rclcpp::Serialization<jfi_comm::msg::PosYaw> serializer;
    rclcpp::SerializedMessage serialized_msg;
    serializer.serialize_message(&pos_yaw_msg_, &serialized_msg);

    std::vector<uint8_t> payload(
      serialized_msg.get_rcl_serialized_message().buffer,
      serialized_msg.get_rcl_serialized_message().buffer +
      serialized_msg.get_rcl_serialized_message().buffer_length
    );

    // JFiComm 메시지 생성 (Tid: 10)
    auto packet = std::make_unique<jfi_comm::msg::SwarmComm>();
    packet->header.stamp = this->get_clock()->now();
    packet->src_sysid = 1;
    packet->seq = seq_++;
    packet->tid = 10;
    packet->payload = payload;

    // jfi_comm/in/packet
    jfi_pub_->publish(std::move(packet));

    RCLCPP_DEBUG(this->get_logger(),
      "Sent RTK data: head=%.2f, N=%.2f, E=%.2f, D=%.2f",
      pos_yaw_msg_.head, pos_yaw_msg_.rtk_n,
      pos_yaw_msg_.rtk_e, pos_yaw_msg_.pos_z);
  }

  rclcpp::Subscription<px4_msgs::msg::Monitoring>::SharedPtr monitoring_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odometry_sub_;
  rclcpp::Publisher<jfi_comm::msg::SwarmComm>::SharedPtr jfi_pub_;
  rclcpp::TimerBase::SharedPtr jfi_timer_;
  jfi_comm::msg::PosYaw pos_yaw_msg_;
  uint32_t seq_;
  std::vector<float> velocity_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MonitoringBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
