// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from px4_msgs:srv/GlobalPath.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__BUILDER_HPP_
#define PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "px4_msgs/srv/detail/global_path__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace px4_msgs
{

namespace srv
{

namespace builder
{

class Init_GlobalPath_Request_waypoints
{
public:
  Init_GlobalPath_Request_waypoints()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::px4_msgs::srv::GlobalPath_Request waypoints(::px4_msgs::srv::GlobalPath_Request::_waypoints_type arg)
  {
    msg_.waypoints = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::srv::GlobalPath_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::srv::GlobalPath_Request>()
{
  return px4_msgs::srv::builder::Init_GlobalPath_Request_waypoints();
}

}  // namespace px4_msgs


namespace px4_msgs
{

namespace srv
{

namespace builder
{

class Init_GlobalPath_Response_reply
{
public:
  Init_GlobalPath_Response_reply()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::px4_msgs::srv::GlobalPath_Response reply(::px4_msgs::srv::GlobalPath_Response::_reply_type arg)
  {
    msg_.reply = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::srv::GlobalPath_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::srv::GlobalPath_Response>()
{
  return px4_msgs::srv::builder::Init_GlobalPath_Response_reply();
}

}  // namespace px4_msgs

#endif  // PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__BUILDER_HPP_
