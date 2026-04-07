// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from px4_msgs:srv/TrajectorySetpoint.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__SRV__DETAIL__TRAJECTORY_SETPOINT__BUILDER_HPP_
#define PX4_MSGS__SRV__DETAIL__TRAJECTORY_SETPOINT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "px4_msgs/srv/detail/trajectory_setpoint__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace px4_msgs
{

namespace srv
{

namespace builder
{

class Init_TrajectorySetpoint_Request_request
{
public:
  Init_TrajectorySetpoint_Request_request()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::px4_msgs::srv::TrajectorySetpoint_Request request(::px4_msgs::srv::TrajectorySetpoint_Request::_request_type arg)
  {
    msg_.request = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::srv::TrajectorySetpoint_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::srv::TrajectorySetpoint_Request>()
{
  return px4_msgs::srv::builder::Init_TrajectorySetpoint_Request_request();
}

}  // namespace px4_msgs


namespace px4_msgs
{

namespace srv
{

namespace builder
{

class Init_TrajectorySetpoint_Response_reply
{
public:
  Init_TrajectorySetpoint_Response_reply()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::px4_msgs::srv::TrajectorySetpoint_Response reply(::px4_msgs::srv::TrajectorySetpoint_Response::_reply_type arg)
  {
    msg_.reply = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::srv::TrajectorySetpoint_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::srv::TrajectorySetpoint_Response>()
{
  return px4_msgs::srv::builder::Init_TrajectorySetpoint_Response_reply();
}

}  // namespace px4_msgs

#endif  // PX4_MSGS__SRV__DETAIL__TRAJECTORY_SETPOINT__BUILDER_HPP_
