// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from suicide_drone_msgs:msg/IBVSOutput.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__BUILDER_HPP_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "suicide_drone_msgs/msg/detail/ibvs_output__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace suicide_drone_msgs
{

namespace msg
{

namespace builder
{

class Init_IBVSOutput_fov_vel_z
{
public:
  explicit Init_IBVSOutput_fov_vel_z(::suicide_drone_msgs::msg::IBVSOutput & msg)
  : msg_(msg)
  {}
  ::suicide_drone_msgs::msg::IBVSOutput fov_vel_z(::suicide_drone_msgs::msg::IBVSOutput::_fov_vel_z_type arg)
  {
    msg_.fov_vel_z = std::move(arg);
    return std::move(msg_);
  }

private:
  ::suicide_drone_msgs::msg::IBVSOutput msg_;
};

class Init_IBVSOutput_fov_yaw_rate
{
public:
  explicit Init_IBVSOutput_fov_yaw_rate(::suicide_drone_msgs::msg::IBVSOutput & msg)
  : msg_(msg)
  {}
  Init_IBVSOutput_fov_vel_z fov_yaw_rate(::suicide_drone_msgs::msg::IBVSOutput::_fov_yaw_rate_type arg)
  {
    msg_.fov_yaw_rate = std::move(arg);
    return Init_IBVSOutput_fov_vel_z(msg_);
  }

private:
  ::suicide_drone_msgs::msg::IBVSOutput msg_;
};

class Init_IBVSOutput_q_z
{
public:
  explicit Init_IBVSOutput_q_z(::suicide_drone_msgs::msg::IBVSOutput & msg)
  : msg_(msg)
  {}
  Init_IBVSOutput_fov_yaw_rate q_z(::suicide_drone_msgs::msg::IBVSOutput::_q_z_type arg)
  {
    msg_.q_z = std::move(arg);
    return Init_IBVSOutput_fov_yaw_rate(msg_);
  }

private:
  ::suicide_drone_msgs::msg::IBVSOutput msg_;
};

class Init_IBVSOutput_q_y
{
public:
  explicit Init_IBVSOutput_q_y(::suicide_drone_msgs::msg::IBVSOutput & msg)
  : msg_(msg)
  {}
  Init_IBVSOutput_q_z q_y(::suicide_drone_msgs::msg::IBVSOutput::_q_y_type arg)
  {
    msg_.q_y = std::move(arg);
    return Init_IBVSOutput_q_z(msg_);
  }

private:
  ::suicide_drone_msgs::msg::IBVSOutput msg_;
};

class Init_IBVSOutput_detected
{
public:
  explicit Init_IBVSOutput_detected(::suicide_drone_msgs::msg::IBVSOutput & msg)
  : msg_(msg)
  {}
  Init_IBVSOutput_q_y detected(::suicide_drone_msgs::msg::IBVSOutput::_detected_type arg)
  {
    msg_.detected = std::move(arg);
    return Init_IBVSOutput_q_y(msg_);
  }

private:
  ::suicide_drone_msgs::msg::IBVSOutput msg_;
};

class Init_IBVSOutput_header
{
public:
  Init_IBVSOutput_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_IBVSOutput_detected header(::suicide_drone_msgs::msg::IBVSOutput::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_IBVSOutput_detected(msg_);
  }

private:
  ::suicide_drone_msgs::msg::IBVSOutput msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::suicide_drone_msgs::msg::IBVSOutput>()
{
  return suicide_drone_msgs::msg::builder::Init_IBVSOutput_header();
}

}  // namespace suicide_drone_msgs

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__BUILDER_HPP_
