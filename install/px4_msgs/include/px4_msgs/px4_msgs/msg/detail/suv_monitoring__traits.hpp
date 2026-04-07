// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from px4_msgs:msg/SuvMonitoring.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SUV_MONITORING__TRAITS_HPP_
#define PX4_MSGS__MSG__DETAIL__SUV_MONITORING__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "px4_msgs/msg/detail/suv_monitoring__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'monitoring'
#include "px4_msgs/msg/detail/monitoring__traits.hpp"

namespace px4_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const SuvMonitoring & msg,
  std::ostream & out)
{
  out << "{";
  // member: monitoring
  {
    out << "monitoring: ";
    to_flow_style_yaml(msg.monitoring, out);
    out << ", ";
  }

  // member: mode
  {
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SuvMonitoring & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: monitoring
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "monitoring:\n";
    to_block_style_yaml(msg.monitoring, out, indentation + 2);
  }

  // member: mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SuvMonitoring & msg, bool use_flow_style = false)
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

}  // namespace px4_msgs

namespace rosidl_generator_traits
{

[[deprecated("use px4_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const px4_msgs::msg::SuvMonitoring & msg,
  std::ostream & out, size_t indentation = 0)
{
  px4_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use px4_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const px4_msgs::msg::SuvMonitoring & msg)
{
  return px4_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<px4_msgs::msg::SuvMonitoring>()
{
  return "px4_msgs::msg::SuvMonitoring";
}

template<>
inline const char * name<px4_msgs::msg::SuvMonitoring>()
{
  return "px4_msgs/msg/SuvMonitoring";
}

template<>
struct has_fixed_size<px4_msgs::msg::SuvMonitoring>
  : std::integral_constant<bool, has_fixed_size<px4_msgs::msg::Monitoring>::value> {};

template<>
struct has_bounded_size<px4_msgs::msg::SuvMonitoring>
  : std::integral_constant<bool, has_bounded_size<px4_msgs::msg::Monitoring>::value> {};

template<>
struct is_message<px4_msgs::msg::SuvMonitoring>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PX4_MSGS__MSG__DETAIL__SUV_MONITORING__TRAITS_HPP_
