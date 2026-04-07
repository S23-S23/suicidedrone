// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from px4_msgs:srv/ModeChange.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__SRV__DETAIL__MODE_CHANGE__BUILDER_HPP_
#define PX4_MSGS__SRV__DETAIL__MODE_CHANGE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "px4_msgs/srv/detail/mode_change__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace px4_msgs
{

namespace srv
{

namespace builder
{

class Init_ModeChange_Request_suv_mode
{
public:
  Init_ModeChange_Request_suv_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::px4_msgs::srv::ModeChange_Request suv_mode(::px4_msgs::srv::ModeChange_Request::_suv_mode_type arg)
  {
    msg_.suv_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::srv::ModeChange_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::srv::ModeChange_Request>()
{
  return px4_msgs::srv::builder::Init_ModeChange_Request_suv_mode();
}

}  // namespace px4_msgs


namespace px4_msgs
{

namespace srv
{

namespace builder
{

class Init_ModeChange_Response_reply
{
public:
  Init_ModeChange_Response_reply()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::px4_msgs::srv::ModeChange_Response reply(::px4_msgs::srv::ModeChange_Response::_reply_type arg)
  {
    msg_.reply = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::srv::ModeChange_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::srv::ModeChange_Response>()
{
  return px4_msgs::srv::builder::Init_ModeChange_Response_reply();
}

}  // namespace px4_msgs

#endif  // PX4_MSGS__SRV__DETAIL__MODE_CHANGE__BUILDER_HPP_
