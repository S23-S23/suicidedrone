// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from suicide_drone_msgs:msg/GuidanceCmd.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__TRAITS_HPP_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "suicide_drone_msgs/msg/detail/guidance_cmd__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace suicide_drone_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const GuidanceCmd & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: target_detected
  {
    out << "target_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.target_detected, out);
    out << ", ";
  }

  // member: vel_n
  {
    out << "vel_n: ";
    rosidl_generator_traits::value_to_yaml(msg.vel_n, out);
    out << ", ";
  }

  // member: vel_e
  {
    out << "vel_e: ";
    rosidl_generator_traits::value_to_yaml(msg.vel_e, out);
    out << ", ";
  }

  // member: vel_d
  {
    out << "vel_d: ";
    rosidl_generator_traits::value_to_yaml(msg.vel_d, out);
    out << ", ";
  }

  // member: yaw_rate
  {
    out << "yaw_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw_rate, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GuidanceCmd & msg,
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

  // member: target_detected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_detected: ";
    rosidl_generator_traits::value_to_yaml(msg.target_detected, out);
    out << "\n";
  }

  // member: vel_n
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vel_n: ";
    rosidl_generator_traits::value_to_yaml(msg.vel_n, out);
    out << "\n";
  }

  // member: vel_e
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vel_e: ";
    rosidl_generator_traits::value_to_yaml(msg.vel_e, out);
    out << "\n";
  }

  // member: vel_d
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vel_d: ";
    rosidl_generator_traits::value_to_yaml(msg.vel_d, out);
    out << "\n";
  }

  // member: yaw_rate
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw_rate, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GuidanceCmd & msg, bool use_flow_style = false)
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
  const suicide_drone_msgs::msg::GuidanceCmd & msg,
  std::ostream & out, size_t indentation = 0)
{
  suicide_drone_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use suicide_drone_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const suicide_drone_msgs::msg::GuidanceCmd & msg)
{
  return suicide_drone_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<suicide_drone_msgs::msg::GuidanceCmd>()
{
  return "suicide_drone_msgs::msg::GuidanceCmd";
}

template<>
inline const char * name<suicide_drone_msgs::msg::GuidanceCmd>()
{
  return "suicide_drone_msgs/msg/GuidanceCmd";
}

template<>
struct has_fixed_size<suicide_drone_msgs::msg::GuidanceCmd>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<suicide_drone_msgs::msg::GuidanceCmd>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<suicide_drone_msgs::msg::GuidanceCmd>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__TRAITS_HPP_
