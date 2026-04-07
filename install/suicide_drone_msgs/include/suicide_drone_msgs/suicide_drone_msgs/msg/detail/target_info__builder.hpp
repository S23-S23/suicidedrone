// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from suicide_drone_msgs:msg/TargetInfo.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__TARGET_INFO__BUILDER_HPP_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__TARGET_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "suicide_drone_msgs/msg/detail/target_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace suicide_drone_msgs
{

namespace msg
{

namespace builder
{

class Init_TargetInfo_right
{
public:
  explicit Init_TargetInfo_right(::suicide_drone_msgs::msg::TargetInfo & msg)
  : msg_(msg)
  {}
  ::suicide_drone_msgs::msg::TargetInfo right(::suicide_drone_msgs::msg::TargetInfo::_right_type arg)
  {
    msg_.right = std::move(arg);
    return std::move(msg_);
  }

private:
  ::suicide_drone_msgs::msg::TargetInfo msg_;
};

class Init_TargetInfo_bottom
{
public:
  explicit Init_TargetInfo_bottom(::suicide_drone_msgs::msg::TargetInfo & msg)
  : msg_(msg)
  {}
  Init_TargetInfo_right bottom(::suicide_drone_msgs::msg::TargetInfo::_bottom_type arg)
  {
    msg_.bottom = std::move(arg);
    return Init_TargetInfo_right(msg_);
  }

private:
  ::suicide_drone_msgs::msg::TargetInfo msg_;
};

class Init_TargetInfo_left
{
public:
  explicit Init_TargetInfo_left(::suicide_drone_msgs::msg::TargetInfo & msg)
  : msg_(msg)
  {}
  Init_TargetInfo_bottom left(::suicide_drone_msgs::msg::TargetInfo::_left_type arg)
  {
    msg_.left = std::move(arg);
    return Init_TargetInfo_bottom(msg_);
  }

private:
  ::suicide_drone_msgs::msg::TargetInfo msg_;
};

class Init_TargetInfo_top
{
public:
  explicit Init_TargetInfo_top(::suicide_drone_msgs::msg::TargetInfo & msg)
  : msg_(msg)
  {}
  Init_TargetInfo_left top(::suicide_drone_msgs::msg::TargetInfo::_top_type arg)
  {
    msg_.top = std::move(arg);
    return Init_TargetInfo_left(msg_);
  }

private:
  ::suicide_drone_msgs::msg::TargetInfo msg_;
};

class Init_TargetInfo_class_name
{
public:
  explicit Init_TargetInfo_class_name(::suicide_drone_msgs::msg::TargetInfo & msg)
  : msg_(msg)
  {}
  Init_TargetInfo_top class_name(::suicide_drone_msgs::msg::TargetInfo::_class_name_type arg)
  {
    msg_.class_name = std::move(arg);
    return Init_TargetInfo_top(msg_);
  }

private:
  ::suicide_drone_msgs::msg::TargetInfo msg_;
};

class Init_TargetInfo_header
{
public:
  Init_TargetInfo_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_TargetInfo_class_name header(::suicide_drone_msgs::msg::TargetInfo::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_TargetInfo_class_name(msg_);
  }

private:
  ::suicide_drone_msgs::msg::TargetInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::suicide_drone_msgs::msg::TargetInfo>()
{
  return suicide_drone_msgs::msg::builder::Init_TargetInfo_header();
}

}  // namespace suicide_drone_msgs

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__TARGET_INFO__BUILDER_HPP_
