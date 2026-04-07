// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from px4_msgs:msg/SuvMonitoring.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SUV_MONITORING__BUILDER_HPP_
#define PX4_MSGS__MSG__DETAIL__SUV_MONITORING__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "px4_msgs/msg/detail/suv_monitoring__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace px4_msgs
{

namespace msg
{

namespace builder
{

class Init_SuvMonitoring_mode
{
public:
  explicit Init_SuvMonitoring_mode(::px4_msgs::msg::SuvMonitoring & msg)
  : msg_(msg)
  {}
  ::px4_msgs::msg::SuvMonitoring mode(::px4_msgs::msg::SuvMonitoring::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::msg::SuvMonitoring msg_;
};

class Init_SuvMonitoring_monitoring
{
public:
  Init_SuvMonitoring_monitoring()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SuvMonitoring_mode monitoring(::px4_msgs::msg::SuvMonitoring::_monitoring_type arg)
  {
    msg_.monitoring = std::move(arg);
    return Init_SuvMonitoring_mode(msg_);
  }

private:
  ::px4_msgs::msg::SuvMonitoring msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::msg::SuvMonitoring>()
{
  return px4_msgs::msg::builder::Init_SuvMonitoring_monitoring();
}

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__SUV_MONITORING__BUILDER_HPP_
