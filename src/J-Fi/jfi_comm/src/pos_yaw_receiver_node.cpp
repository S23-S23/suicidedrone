#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include "rclcpp/qos.hpp"
#include "jfi_comm/msg/swarm_comm.hpp"
#include "jfi_comm/msg/pos_yaw.hpp"

class PosYawReceiverNode : public rclcpp::Node
{
public:
  PosYawReceiverNode() : Node("pos_yaw_receiver_node")
  {
    // 파라미터 설정 (선택사항)
    declare_parameter<int>("drone_id", 2);
    drone_id_ = static_cast<uint8_t>(get_parameter("drone_id").as_int());

    // 1. jfi_comm/out/packet 구독 (시리얼에서 받은 데이터)
    packet_sub_ = this->create_subscription<jfi_comm::msg::SwarmComm>(
      "jfi_comm/out/packet", 10,
      std::bind(&PosYawReceiverNode::packet_callback, this, std::placeholders::_1)
    );

    // 2. 역직렬화된 PosYaw 재발행 (선택사항 - 다른 노드에서 사용하려면)
    pos_yaw_pub_ = this->create_publisher<jfi_comm::msg::PosYaw>(
      "/drone1/jfi/out/pos_yaw", rclcpp::SensorDataQoS()
    );

    RCLCPP_INFO(this->get_logger(), "PosYaw Receiver Node started (System ID: %u)", drone_id_);
  }

private:
  void packet_callback(const jfi_comm::msg::SwarmComm::SharedPtr packet)
  {
    // TID 10 = RTK 데이터 확인
    if (packet->tid == 10) {

      RCLCPP_DEBUG(this->get_logger(),
        "Received packet with TID 10 from system %u, seq %u, payload size: %zu",
        packet->src_sysid, packet->seq, packet->payload.size());

      try {
        // Step 1: payload에서 PosYaw 역직렬화
        rclcpp::Serialization<jfi_comm::msg::PosYaw> serializer;
        rclcpp::SerializedMessage serialized_msg;

        serialized_msg.reserve(packet->payload.size());
        serialized_msg.get_rcl_serialized_message().buffer_length = packet->payload.size();
        std::memcpy(
          serialized_msg.get_rcl_serialized_message().buffer,
          packet->payload.data(),
          packet->payload.size()
        );

        jfi_comm::msg::PosYaw pos_yaw;
        serializer.deserialize_message(&serialized_msg, &pos_yaw);

        // Step 2: 수신된 데이터 출력
        RCLCPP_INFO_ONCE(this->get_logger(),
          "Received RTK from Drone %u: "
          "Heading=%.2f°, North=%.2f m, East=%.2f m, Down=%.2f m",
          packet->src_sysid,
          pos_yaw.head,
          pos_yaw.rtk_n,
          pos_yaw.rtk_e,
          pos_yaw.pos_z
        );

        // Step 3: 다른 노드에서 사용할 수 있도록 재발행 (선택사항)
        pos_yaw_pub_->publish(pos_yaw);

      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(),
          "Failed to deserialize PosYaw from system %u: %s",
          packet->src_sysid, e.what());
      }
    }
    // 다른 TID는 무시 (필요시 추가 처리)
  }

  rclcpp::Subscription<jfi_comm::msg::SwarmComm>::SharedPtr packet_sub_;
  rclcpp::Publisher<jfi_comm::msg::PosYaw>::SharedPtr pos_yaw_pub_;
  uint8_t drone_id_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PosYawReceiverNode>());
  rclcpp::shutdown();
  return 0;
}
