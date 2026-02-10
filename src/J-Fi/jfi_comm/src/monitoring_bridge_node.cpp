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
    // 1. PX4 Monitoring 토픽 구독
    monitoring_sub_ = this->create_subscription<px4_msgs::msg::Monitoring>(
      "/drone1/fmu/out/monitoring",  // PX4 토픽 이름 (실제 이름 확인 필요)
      10,
      std::bind(&MonitoringBridgeNode::monitoring_callback, this, std::placeholders::_1)
    );
    odometry_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
      "/drone1/fmu/out/vehicle_odometry",  // PX4 토픽 이름 (실제 이름 확인 필요)
      10,
      std::bind(&MonitoringBridgeNode::odometry_callback, this, std::placeholders::_1)
    );

    // 2. JFI 전송용 Publisher (jfi_comm/in/packet → 시리얼로 전송됨)
    jfi_pub_ = this->create_publisher<jfi_comm::msg::SwarmComm>(
      "jfi_comm/in/packet", 10
    );

    RCLCPP_INFO(this->get_logger(), "Monitoring Bridge Node started");
  }

private:
  // PX4 Monitoring 메시지가 올 때마다 호출되는 콜백
  void odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr odometry_msg)
  {
    velocity_[0] = odometry_msg->velocity[0];
    velocity_[1] = odometry_msg->velocity[1];
    velocity_[2] = odometry_msg->velocity[2];
  }
  void monitoring_callback(const px4_msgs::msg::Monitoring::SharedPtr monitoring_msg)
  {
    // Step 1: PX4 Monitoring에서 필요한 데이터 추출하여 PosYaw 생성
    if (velocity_.size() != 3) {
      RCLCPP_WARN(this->get_logger(), "Velocity data not yet received. Skipping monitoring message.");
      return;
    }
    auto pos_yaw_msg = std::make_shared<jfi_comm::msg::PosYaw>();
    pos_yaw_msg->head = monitoring_msg->head;
    pos_yaw_msg->rtk_n = monitoring_msg->rtk_n;
    pos_yaw_msg->rtk_e = monitoring_msg->rtk_e;
    pos_yaw_msg->pos_z = monitoring_msg->pos_z;
    pos_yaw_msg->velocity[0] = velocity_[0];
    pos_yaw_msg->velocity[1] = velocity_[1];
    pos_yaw_msg->velocity[2] = velocity_[2];

    // Step 2: PosYaw를 바이트 배열로 직렬화
    rclcpp::Serialization<jfi_comm::msg::PosYaw> serializer;
    rclcpp::SerializedMessage serialized_msg;
    serializer.serialize_message(pos_yaw_msg.get(), &serialized_msg);

    std::vector<uint8_t> payload(
      serialized_msg.get_rcl_serialized_message().buffer,
      serialized_msg.get_rcl_serialized_message().buffer +
      serialized_msg.get_rcl_serialized_message().buffer_length
    );

    // Step 3: SwarmComm 메시지 생성 (전송 컨테이너)
    auto packet = std::make_unique<jfi_comm::msg::SwarmComm>();
    packet->header.stamp = this->get_clock()->now();
    packet->src_sysid = 1;  // 내 시스템 ID (파라미터로 받을 수도 있음)
    packet->seq = seq_++;
    packet->tid = 10;       // TID 10 = RTK 데이터 (원하는 번호 사용)
    packet->payload = payload;

    // Step 4: JFI로 전송 (jfi_comm/in/packet)
    jfi_pub_->publish(std::move(packet));

    RCLCPP_DEBUG(this->get_logger(),
      "Sent RTK data: head=%.2f, N=%.2f, E=%.2f, D=%.2f",
      pos_yaw_msg->head, pos_yaw_msg->rtk_n,
      pos_yaw_msg->rtk_e, pos_yaw_msg->pos_z);
  }

  rclcpp::Subscription<px4_msgs::msg::Monitoring>::SharedPtr monitoring_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odometry_sub_;
  rclcpp::Publisher<jfi_comm::msg::SwarmComm>::SharedPtr jfi_pub_;
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
