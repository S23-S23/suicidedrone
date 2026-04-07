// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from px4_msgs:msg/OffboardScenario.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__TRAITS_HPP_
#define PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "px4_msgs/msg/detail/offboard_scenario__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace px4_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const OffboardScenario & msg,
  std::ostream & out)
{
  out << "{";
  // member: timestamp
  {
    out << "timestamp: ";
    rosidl_generator_traits::value_to_yaml(msg.timestamp, out);
    out << ", ";
  }

  // member: current_time
  {
    out << "current_time: ";
    rosidl_generator_traits::value_to_yaml(msg.current_time, out);
    out << ", ";
  }

  // member: start_time
  {
    out << "start_time: ";
    rosidl_generator_traits::value_to_yaml(msg.start_time, out);
    out << ", ";
  }

  // member: seq
  {
    out << "seq: ";
    rosidl_generator_traits::value_to_yaml(msg.seq, out);
    out << ", ";
  }

  // member: offset_x
  {
    out << "offset_x: ";
    rosidl_generator_traits::value_to_yaml(msg.offset_x, out);
    out << ", ";
  }

  // member: offset_y
  {
    out << "offset_y: ";
    rosidl_generator_traits::value_to_yaml(msg.offset_y, out);
    out << ", ";
  }

  // member: ready_sc_file
  {
    out << "ready_sc_file: ";
    rosidl_generator_traits::value_to_yaml(msg.ready_sc_file, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const OffboardScenario & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: timestamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "timestamp: ";
    rosidl_generator_traits::value_to_yaml(msg.timestamp, out);
    out << "\n";
  }

  // member: current_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_time: ";
    rosidl_generator_traits::value_to_yaml(msg.current_time, out);
    out << "\n";
  }

  // member: start_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "start_time: ";
    rosidl_generator_traits::value_to_yaml(msg.start_time, out);
    out << "\n";
  }

  // member: seq
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "seq: ";
    rosidl_generator_traits::value_to_yaml(msg.seq, out);
    out << "\n";
  }

  // member: offset_x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "offset_x: ";
    rosidl_generator_traits::value_to_yaml(msg.offset_x, out);
    out << "\n";
  }

  // member: offset_y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "offset_y: ";
    rosidl_generator_traits::value_to_yaml(msg.offset_y, out);
    out << "\n";
  }

  // member: ready_sc_file
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ready_sc_file: ";
    rosidl_generator_traits::value_to_yaml(msg.ready_sc_file, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const OffboardScenario & msg, bool use_flow_style = false)
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
  const px4_msgs::msg::OffboardScenario & msg,
  std::ostream & out, size_t indentation = 0)
{
  px4_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use px4_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const px4_msgs::msg::OffboardScenario & msg)
{
  return px4_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<px4_msgs::msg::OffboardScenario>()
{
  return "px4_msgs::msg::OffboardScenario";
}

template<>
inline const char * name<px4_msgs::msg::OffboardScenario>()
{
  return "px4_msgs/msg/OffboardScenario";
}

template<>
struct has_fixed_size<px4_msgs::msg::OffboardScenario>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<px4_msgs::msg::OffboardScenario>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<px4_msgs::msg::OffboardScenario>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__TRAITS_HPP_
