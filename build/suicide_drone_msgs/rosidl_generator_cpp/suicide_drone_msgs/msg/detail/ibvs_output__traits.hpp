// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from suicide_drone_msgs:msg/IBVSOutput.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__TRAITS_HPP_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "suicide_drone_msgs/msg/detail/ibvs_output__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace suicide_drone_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const IBVSOutput & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: detected
  {
    out << "detected: ";
    rosidl_generator_traits::value_to_yaml(msg.detected, out);
    out << ", ";
  }

  // member: q_y
  {
    out << "q_y: ";
    rosidl_generator_traits::value_to_yaml(msg.q_y, out);
    out << ", ";
  }

  // member: q_z
  {
    out << "q_z: ";
    rosidl_generator_traits::value_to_yaml(msg.q_z, out);
    out << ", ";
  }

  // member: fov_yaw_rate
  {
    out << "fov_yaw_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.fov_yaw_rate, out);
    out << ", ";
  }

  // member: fov_vel_z
  {
    out << "fov_vel_z: ";
    rosidl_generator_traits::value_to_yaml(msg.fov_vel_z, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const IBVSOutput & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: detected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "detected: ";
    rosidl_generator_traits::value_to_yaml(msg.detected, out);
    out << "\n";
  }

  // member: q_y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "q_y: ";
    rosidl_generator_traits::value_to_yaml(msg.q_y, out);
    out << "\n";
  }

  // member: q_z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "q_z: ";
    rosidl_generator_traits::value_to_yaml(msg.q_z, out);
    out << "\n";
  }

  // member: fov_yaw_rate
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "fov_yaw_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.fov_yaw_rate, out);
    out << "\n";
  }

  // member: fov_vel_z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "fov_vel_z: ";
    rosidl_generator_traits::value_to_yaml(msg.fov_vel_z, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const IBVSOutput & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace suicide_drone_msgs

namespace rosidl_generator_traits
{

[[deprecated("use suicide_drone_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const suicide_drone_msgs::msg::IBVSOutput & msg,
  std::ostream & out, size_t indentation = 0)
{
  suicide_drone_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use suicide_drone_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const suicide_drone_msgs::msg::IBVSOutput & msg)
{
  return suicide_drone_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<suicide_drone_msgs::msg::IBVSOutput>()
{
  return "suicide_drone_msgs::msg::IBVSOutput";
}

template<>
inline const char * name<suicide_drone_msgs::msg::IBVSOutput>()
{
  return "suicide_drone_msgs/msg/IBVSOutput";
}

template<>
struct has_fixed_size<suicide_drone_msgs::msg::IBVSOutput>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<suicide_drone_msgs::msg::IBVSOutput>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<suicide_drone_msgs::msg::IBVSOutput>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__TRAITS_HPP_
