// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from px4_msgs:msg/ScenarioCommand.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__BUILDER_HPP_
#define PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "px4_msgs/msg/detail/scenario_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace px4_msgs
{

namespace msg
{

namespace builder
{

class Init_ScenarioCommand_param5
{
public:
  explicit Init_ScenarioCommand_param5(::px4_msgs::msg::ScenarioCommand & msg)
  : msg_(msg)
  {}
  ::px4_msgs::msg::ScenarioCommand param5(::px4_msgs::msg::ScenarioCommand::_param5_type arg)
  {
    msg_.param5 = std::move(arg);
    return std::move(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioCommand msg_;
};

class Init_ScenarioCommand_param4
{
public:
  explicit Init_ScenarioCommand_param4(::px4_msgs::msg::ScenarioCommand & msg)
  : msg_(msg)
  {}
  Init_ScenarioCommand_param5 param4(::px4_msgs::msg::ScenarioCommand::_param4_type arg)
  {
    msg_.param4 = std::move(arg);
    return Init_ScenarioCommand_param5(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioCommand msg_;
};

class Init_ScenarioCommand_param3
{
public:
  explicit Init_ScenarioCommand_param3(::px4_msgs::msg::ScenarioCommand & msg)
  : msg_(msg)
  {}
  Init_ScenarioCommand_param4 param3(::px4_msgs::msg::ScenarioCommand::_param3_type arg)
  {
    msg_.param3 = std::move(arg);
    return Init_ScenarioCommand_param4(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioCommand msg_;
};

class Init_ScenarioCommand_param2
{
public:
  explicit Init_ScenarioCommand_param2(::px4_msgs::msg::ScenarioCommand & msg)
  : msg_(msg)
  {}
  Init_ScenarioCommand_param3 param2(::px4_msgs::msg::ScenarioCommand::_param2_type arg)
  {
    msg_.param2 = std::move(arg);
    return Init_ScenarioCommand_param3(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioCommand msg_;
};

class Init_ScenarioCommand_param1
{
public:
  explicit Init_ScenarioCommand_param1(::px4_msgs::msg::ScenarioCommand & msg)
  : msg_(msg)
  {}
  Init_ScenarioCommand_param2 param1(::px4_msgs::msg::ScenarioCommand::_param1_type arg)
  {
    msg_.param1 = std::move(arg);
    return Init_ScenarioCommand_param2(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioCommand msg_;
};

class Init_ScenarioCommand_cmd
{
public:
  explicit Init_ScenarioCommand_cmd(::px4_msgs::msg::ScenarioCommand & msg)
  : msg_(msg)
  {}
  Init_ScenarioCommand_param1 cmd(::px4_msgs::msg::ScenarioCommand::_cmd_type arg)
  {
    msg_.cmd = std::move(arg);
    return Init_ScenarioCommand_param1(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioCommand msg_;
};

class Init_ScenarioCommand_timestamp
{
public:
  Init_ScenarioCommand_timestamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ScenarioCommand_cmd timestamp(::px4_msgs::msg::ScenarioCommand::_timestamp_type arg)
  {
    msg_.timestamp = std::move(arg);
    return Init_ScenarioCommand_cmd(msg_);
  }

private:
  ::px4_msgs::msg::ScenarioCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::px4_msgs::msg::ScenarioCommand>()
{
  return px4_msgs::msg::builder::Init_ScenarioCommand_timestamp();
}

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__BUILDER_HPP_
