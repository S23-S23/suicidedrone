// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from px4_msgs:msg/ScenarioEvent.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SCENARIO_EVENT__BUILDER_HPP_
#define PX4_MSGS__MSG__DETAIL__SCENARIO_EVENT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "px4_msgs/msg/detail/scenario_event__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace px4_msgs
{

namespace msg
{

namespace builder
{

class Init_ScenarioEvent_is_scenario_active
{
public:
  explicit Init_ScenarioEvent_is_scenario_active(::px4_msgs::msg::ScenarioEvent & msg)
  : msg_(msg)
  {}
  ::px4_msgs::msg::ScenarioEvent is_scenario_active(::px4_msgs::msg::ScenarioEvent::_is_scenario_active_type arg)
  {
    msg_.is_scenario_active = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioEvent msg_;
};

class Init_ScenarioEvent_led_b
{
public:
  explicit Init_ScenarioEvent_led_b(::px4_msgs::msg::ScenarioEvent & msg)
  : msg_(msg)
  {}
  Init_ScenarioEvent_is_scenario_active led_b(::px4_msgs::msg::ScenarioEvent::_led_b_type arg)
  {
    msg_.led_b = std::move(arg);
    return Init_ScenarioEvent_is_scenario_active(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioEvent msg_;
};

class Init_ScenarioEvent_led_g
{
public:
  explicit Init_ScenarioEvent_led_g(::px4_msgs::msg::ScenarioEvent & msg)
  : msg_(msg)
  {}
  Init_ScenarioEvent_led_b led_g(::px4_msgs::msg::ScenarioEvent::_led_g_type arg)
  {
    msg_.led_g = std::move(arg);
    return Init_ScenarioEvent_led_b(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioEvent msg_;
};

class Init_ScenarioEvent_led_r
{
public:
  explicit Init_ScenarioEvent_led_r(::px4_msgs::msg::ScenarioEvent & msg)
  : msg_(msg)
  {}
  Init_ScenarioEvent_led_g led_r(::px4_msgs::msg::ScenarioEvent::_led_r_type arg)
  {
    msg_.led_r = std::move(arg);
    return Init_ScenarioEvent_led_g(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioEvent msg_;
};

class Init_ScenarioEvent_z
{
public:
  explicit Init_ScenarioEvent_z(::px4_msgs::msg::ScenarioEvent & msg)
  : msg_(msg)
  {}
  Init_ScenarioEvent_led_r z(::px4_msgs::msg::ScenarioEvent::_z_type arg)
  {
    msg_.z = std::move(arg);
    return Init_ScenarioEvent_led_r(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioEvent msg_;
};

class Init_ScenarioEvent_y
{
public:
  explicit Init_ScenarioEvent_y(::px4_msgs::msg::ScenarioEvent & msg)
  : msg_(msg)
  {}
  Init_ScenarioEvent_z y(::px4_msgs::msg::ScenarioEvent::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_ScenarioEvent_z(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioEvent msg_;
};

class Init_ScenarioEvent_x
{
public:
  explicit Init_ScenarioEvent_x(::px4_msgs::msg::ScenarioEvent & msg)
  : msg_(msg)
  {}
  Init_ScenarioEvent_y x(::px4_msgs::msg::ScenarioEvent::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_ScenarioEvent_y(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioEvent msg_;
};

class Init_ScenarioEvent_cmd_type
{
public:
  explicit Init_ScenarioEvent_cmd_type(::px4_msgs::msg::ScenarioEvent & msg)
  : msg_(msg)
  {}
  Init_ScenarioEvent_x cmd_type(::px4_msgs::msg::ScenarioEvent::_cmd_type_type arg)
  {
    msg_.cmd_type = std::move(arg);
    return Init_ScenarioEvent_x(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioEvent msg_;
};

class Init_ScenarioEvent_event_type
{
public:
  explicit Init_ScenarioEvent_event_type(::px4_msgs::msg::ScenarioEvent & msg)
  : msg_(msg)
  {}
  Init_ScenarioEvent_cmd_type event_type(::px4_msgs::msg::ScenarioEvent::_event_type_type arg)
  {
    msg_.event_type = std::move(arg);
    return Init_ScenarioEvent_cmd_type(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioEvent msg_;
};

class Init_ScenarioEvent_event_time
{
public:
  explicit Init_ScenarioEvent_event_time(::px4_msgs::msg::ScenarioEvent & msg)
  : msg_(msg)
  {}
  Init_ScenarioEvent_event_type event_time(::px4_msgs::msg::ScenarioEvent::_event_time_type arg)
  {
    msg_.event_time = std::move(arg);
    return Init_ScenarioEvent_event_type(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioEvent msg_;
};

class Init_ScenarioEvent_timestamp
{
public:
  Init_ScenarioEvent_timestamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ScenarioEvent_event_time timestamp(::px4_msgs::msg::ScenarioEvent::_timestamp_type arg)
  {
    msg_.timestamp = std::move(arg);
    return Init_ScenarioEvent_event_time(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioEvent msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::msg::ScenarioEvent>()
{
  return px4_msgs::msg::builder::Init_ScenarioEvent_timestamp();
}

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__SCENARIO_EVENT__BUILDER_HPP_
