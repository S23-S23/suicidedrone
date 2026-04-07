// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from px4_msgs:msg/OffboardScenario.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__BUILDER_HPP_
#define PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "px4_msgs/msg/detail/offboard_scenario__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace px4_msgs
{

namespace msg
{

namespace builder
{

class Init_OffboardScenario_ready_sc_file
{
public:
  explicit Init_OffboardScenario_ready_sc_file(::px4_msgs::msg::OffboardScenario & msg)
  : msg_(msg)
  {}
  ::px4_msgs::msg::OffboardScenario ready_sc_file(::px4_msgs::msg::OffboardScenario::_ready_sc_file_type arg)
  {
    msg_.ready_sc_file = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::msg::OffboardScenario msg_;
};

class Init_OffboardScenario_offset_y
{
public:
  explicit Init_OffboardScenario_offset_y(::px4_msgs::msg::OffboardScenario & msg)
  : msg_(msg)
  {}
  Init_OffboardScenario_ready_sc_file offset_y(::px4_msgs::msg::OffboardScenario::_offset_y_type arg)
  {
    msg_.offset_y = std::move(arg);
    return Init_OffboardScenario_ready_sc_file(msg_);
  }

private:
  ::px4_msgs::msg::OffboardScenario msg_;
};

class Init_OffboardScenario_offset_x
{
public:
  explicit Init_OffboardScenario_offset_x(::px4_msgs::msg::OffboardScenario & msg)
  : msg_(msg)
  {}
  Init_OffboardScenario_offset_y offset_x(::px4_msgs::msg::OffboardScenario::_offset_x_type arg)
  {
    msg_.offset_x = std::move(arg);
    return Init_OffboardScenario_offset_y(msg_);
  }

private:
  ::px4_msgs::msg::OffboardScenario msg_;
};

class Init_OffboardScenario_seq
{
public:
  explicit Init_OffboardScenario_seq(::px4_msgs::msg::OffboardScenario & msg)
  : msg_(msg)
  {}
  Init_OffboardScenario_offset_x seq(::px4_msgs::msg::OffboardScenario::_seq_type arg)
  {
    msg_.seq = std::move(arg);
    return Init_OffboardScenario_offset_x(msg_);
  }

private:
  ::px4_msgs::msg::OffboardScenario msg_;
};

class Init_OffboardScenario_start_time
{
public:
  explicit Init_OffboardScenario_start_time(::px4_msgs::msg::OffboardScenario & msg)
  : msg_(msg)
  {}
  Init_OffboardScenario_seq start_time(::px4_msgs::msg::OffboardScenario::_start_time_type arg)
  {
    msg_.start_time = std::move(arg);
    return Init_OffboardScenario_seq(msg_);
  }

private:
  ::px4_msgs::msg::OffboardScenario msg_;
};

class Init_OffboardScenario_current_time
{
public:
  explicit Init_OffboardScenario_current_time(::px4_msgs::msg::OffboardScenario & msg)
  : msg_(msg)
  {}
  Init_OffboardScenario_start_time current_time(::px4_msgs::msg::OffboardScenario::_current_time_type arg)
  {
    msg_.current_time = std::move(arg);
    return Init_OffboardScenario_start_time(msg_);
  }

private:
  ::px4_msgs::msg::OffboardScenario msg_;
};

class Init_OffboardScenario_timestamp
{
public:
  Init_OffboardScenario_timestamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_OffboardScenario_current_time timestamp(::px4_msgs::msg::OffboardScenario::_timestamp_type arg)
  {
    msg_.timestamp = std::move(arg);
    return Init_OffboardScenario_current_time(msg_);
  }

private:
  ::px4_msgs::msg::OffboardScenario msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::msg::OffboardScenario>()
{
  return px4_msgs::msg::builder::Init_OffboardScenario_timestamp();
}

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__BUILDER_HPP_
