#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include "jfi_comm/msg/swarm_comm.hpp"
#include "jfi_comm/msg/pos_yaw.hpp"
#include "std_msgs/msg/u_int32.hpp"

class MonitoringBridgeNode : public rclcpp::Node
{
public:
  MonitoringBridgeNode() : Node("monitoring_bridge_node"), seq_(0)
  {
    // Subscribe
    trigger_sub_ = this->create_subscription<std_msgs::msg::UInt32>(
      "/tracking_trigger",
      rclcpp::SensorDataQoS(),
      std::bind(&MonitoringBridgeNode::trigger_callback, this, std::placeholders::_1)
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

    RCLCPP_INFO(this->get_logger(), "Monitoring Bridge Node started");
  }

private:
  void trigger_callback(const std_msgs::msg::UInt32::SharedPtr msg) {
    // Trigger 메시지 수신 시, PosYaw 메시지 업데이트
    RCLCPP_DEBUG(this->get_logger(), "Received tracking trigger: %u", msg->data);
    trigger_msg_.data = msg->data;
  }

  void jfi_timer_callback() {
    // Topic 직렬화
    RCLCPP_INFO_ONCE(this->get_logger(), "Publishing RTK data to jfi_comm/in/packet");
    rclcpp::Serialization<jfi_comm::msg::PosYaw> serializer;
    rclcpp::SerializedMessage serialized_msg;
    serializer.serialize_message(&trigger_msg_, &serialized_msg);

    std::vector<uint8_t> payload(
      serialized_msg.get_rcl_serialized_message().buffer,
      serialized_msg.get_rcl_serialized_message().buffer +
      serialized_msg.get_rcl_serialized_message().buffer_length
    );

    // JFiComm 메시지 생성 (Tid: 11)
    auto packet = std::make_unique<jfi_comm::msg::SwarmComm>();
    packet->header.stamp = this->get_clock()->now();
    packet->src_sysid = 1;
    packet->seq = seq_++;
    packet->tid = 11;
    packet->payload = payload;

    // jfi_comm/in/packet
    jfi_pub_->publish(std::move(packet));

    RCLCPP_DEBUG(this->get_logger(), "trigger_msg_ data: %u", trigger_msg_.data);
  }

  rclcpp::Subscription<std_msgs::msg::UInt32>::SharedPtr trigger_sub_;
  rclcpp::Publisher<jfi_comm::msg::SwarmComm>::SharedPtr jfi_pub_;
  rclcpp::TimerBase::SharedPtr jfi_timer_;
  std_msgs::msg::UInt32 trigger_msg_;
  uint32_t seq_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MonitoringBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
