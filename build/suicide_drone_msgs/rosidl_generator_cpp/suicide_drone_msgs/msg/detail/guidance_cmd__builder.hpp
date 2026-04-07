// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from suicide_drone_msgs:msg/GuidanceCmd.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__BUILDER_HPP_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "suicide_drone_msgs/msg/detail/guidance_cmd__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace suicide_drone_msgs
{

namespace msg
{

namespace builder
{

class Init_GuidanceCmd_yaw_rate
{
public:
  explicit Init_GuidanceCmd_yaw_rate(::suicide_drone_msgs::msg::GuidanceCmd & msg)
  : msg_(msg)
  {}
  ::suicide_drone_msgs::msg::GuidanceCmd yaw_rate(::suicide_drone_msgs::msg::GuidanceCmd::_yaw_rate_type arg)
  {
    msg_.yaw_rate = std::move(arg);
    return std::move(msg_);
  }

private:
  ::suicide_drone_msgs::msg::GuidanceCmd msg_;
};

class Init_GuidanceCmd_vel_d
{
public:
  explicit Init_GuidanceCmd_vel_d(::suicide_drone_msgs::msg::GuidanceCmd & msg)
  : msg_(msg)
  {}
  Init_GuidanceCmd_yaw_rate vel_d(::suicide_drone_msgs::msg::GuidanceCmd::_vel_d_type arg)
  {
    msg_.vel_d = std::move(arg);
    return Init_GuidanceCmd_yaw_rate(msg_);
  }

private:
  ::suicide_drone_msgs::msg::GuidanceCmd msg_;
};

class Init_GuidanceCmd_vel_e
{
public:
  explicit Init_GuidanceCmd_vel_e(::suicide_drone_msgs::msg::GuidanceCmd & msg)
  : msg_(msg)
  {}
  Init_GuidanceCmd_vel_d vel_e(::suicide_drone_msgs::msg::GuidanceCmd::_vel_e_type arg)
  {
    msg_.vel_e = std::move(arg);
    return Init_GuidanceCmd_vel_d(msg_);
  }

private:
  ::suicide_drone_msgs::msg::GuidanceCmd msg_;
};

class Init_GuidanceCmd_vel_n
{
public:
  explicit Init_GuidanceCmd_vel_n(::suicide_drone_msgs::msg::GuidanceCmd & msg)
  : msg_(msg)
  {}
  Init_GuidanceCmd_vel_e vel_n(::suicide_drone_msgs::msg::GuidanceCmd::_vel_n_type arg)
  {
    msg_.vel_n = std::move(arg);
    return Init_GuidanceCmd_vel_e(msg_);
  }

private:
  ::suicide_drone_msgs::msg::GuidanceCmd msg_;
};

class Init_GuidanceCmd_target_detected
{
public:
  explicit Init_GuidanceCmd_target_detected(::suicide_drone_msgs::msg::GuidanceCmd & msg)
  : msg_(msg)
  {}
  Init_GuidanceCmd_vel_n target_detected(::suicide_drone_msgs::msg::GuidanceCmd::_target_detected_type arg)
  {
    msg_.target_detected = std::move(arg);
    return Init_GuidanceCmd_vel_n(msg_);
  }

private:
  ::suicide_drone_msgs::msg::GuidanceCmd msg_;
};

class Init_GuidanceCmd_header
{
public:
  Init_GuidanceCmd_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GuidanceCmd_target_detected header(::suicide_drone_msgs::msg::GuidanceCmd::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_GuidanceCmd_target_detected(msg_);
  }

private:
  ::suicide_drone_msgs::msg::GuidanceCmd msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::suicide_drone_msgs::msg::GuidanceCmd>()
{
  return suicide_drone_msgs::msg::builder::Init_GuidanceCmd_header();
}

}  // namespace suicide_drone_msgs

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__BUILDER_HPP_
