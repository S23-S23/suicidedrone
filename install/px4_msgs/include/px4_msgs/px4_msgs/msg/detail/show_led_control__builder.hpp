// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from px4_msgs:msg/ShowLedControl.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SHOW_LED_CONTROL__BUILDER_HPP_
#define PX4_MSGS__MSG__DETAIL__SHOW_LED_CONTROL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "px4_msgs/msg/detail/show_led_control__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace px4_msgs
{

namespace msg
{

namespace builder
{

class Init_ShowLedControl_speed
{
public:
  explicit Init_ShowLedControl_speed(::px4_msgs::msg::ShowLedControl & msg)
  : msg_(msg)
  {}
  ::px4_msgs::msg::ShowLedControl speed(::px4_msgs::msg::ShowLedControl::_speed_type arg)
  {
    msg_.speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::msg::ShowLedControl msg_;
};

class Init_ShowLedControl_brightness
{
public:
  explicit Init_ShowLedControl_brightness(::px4_msgs::msg::ShowLedControl & msg)
  : msg_(msg)
  {}
  Init_ShowLedControl_speed brightness(::px4_msgs::msg::ShowLedControl::_brightness_type arg)
  {
    msg_.brightness = std::move(arg);
    return Init_ShowLedControl_speed(msg_);
  }

private:
  ::px4_msgs::msg::ShowLedControl msg_;
};

class Init_ShowLedControl_b
{
public:
  explicit Init_ShowLedControl_b(::px4_msgs::msg::ShowLedControl & msg)
  : msg_(msg)
  {}
  Init_ShowLedControl_brightness b(::px4_msgs::msg::ShowLedControl::_b_type arg)
  {
    msg_.b = std::move(arg);
    return Init_ShowLedControl_brightness(msg_);
  }

private:
  ::px4_msgs::msg::ShowLedControl msg_;
};

class Init_ShowLedControl_g
{
public:
  explicit Init_ShowLedControl_g(::px4_msgs::msg::ShowLedControl & msg)
  : msg_(msg)
  {}
  Init_ShowLedControl_b g(::px4_msgs::msg::ShowLedControl::_g_type arg)
  {
    msg_.g = std::move(arg);
    return Init_ShowLedControl_b(msg_);
  }

private:
  ::px4_msgs::msg::ShowLedControl msg_;
};

class Init_ShowLedControl_r
{
public:
  explicit Init_ShowLedControl_r(::px4_msgs::msg::ShowLedControl & msg)
  : msg_(msg)
  {}
  Init_ShowLedControl_g r(::px4_msgs::msg::ShowLedControl::_r_type arg)
  {
    msg_.r = std::move(arg);
    return Init_ShowLedControl_g(msg_);
  }

private:
  ::px4_msgs::msg::ShowLedControl msg_;
};

class Init_ShowLedControl_type
{
public:
  explicit Init_ShowLedControl_type(::px4_msgs::msg::ShowLedControl & msg)
  : msg_(msg)
  {}
  Init_ShowLedControl_r type(::px4_msgs::msg::ShowLedControl::_type_type arg)
  {
    msg_.type = std::move(arg);
    return Init_ShowLedControl_r(msg_);
  }

private:
  ::px4_msgs::msg::ShowLedControl msg_;
};

class Init_ShowLedControl_timestamp
{
public:
  Init_ShowLedControl_timestamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ShowLedControl_type timestamp(::px4_msgs::msg::ShowLedControl::_timestamp_type arg)
  {
    msg_.timestamp = std::move(arg);
    return Init_ShowLedControl_type(msg_);
  }

private:
  ::px4_msgs::msg::ShowLedControl msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::msg::ShowLedControl>()
{
  return px4_msgs::msg::builder::Init_ShowLedControl_timestamp();
}

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__SHOW_LED_CONTROL__BUILDER_HPP_
