// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from px4_msgs:msg/ScenarioEvent.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SCENARIO_EVENT__TRAITS_HPP_
#define PX4_MSGS__MSG__DETAIL__SCENARIO_EVENT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "px4_msgs/msg/detail/scenario_event__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace px4_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ScenarioEvent & msg,
  std::ostream & out)
{
  out << "{";
  // member: timestamp
  {
    out << "timestamp: ";
    rosidl_generator_traits::value_to_yaml(msg.timestamp, out);
    out << ", ";
  }

  // member: event_time
  {
    out << "event_time: ";
    rosidl_generator_traits::value_to_yaml(msg.event_time, out);
    out << ", ";
  }

  // member: event_type
  {
    out << "event_type: ";
    rosidl_generator_traits::value_to_yaml(msg.event_type, out);
    out << ", ";
  }

  // member: cmd_type
  {
    out << "cmd_type: ";
    rosidl_generator_traits::value_to_yaml(msg.cmd_type, out);
    out << ", ";
  }

  // member: x
  {
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << ", ";
  }

  // member: y
  {
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << ", ";
  }

  // member: z
  {
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << ", ";
  }

  // member: led_r
  {
    out << "led_r: ";
    rosidl_generator_traits::value_to_yaml(msg.led_r, out);
    out << ", ";
  }

  // member: led_g
  {
    out << "led_g: ";
    rosidl_generator_traits::value_to_yaml(msg.led_g, out);
    out << ", ";
  }

  // member: led_b
  {
    out << "led_b: ";
    rosidl_generator_traits::value_to_yaml(msg.led_b, out);
    out << ", ";
  }

  // member: is_scenario_active
  {
    out << "is_scenario_active: ";
    rosidl_generator_traits::value_to_yaml(msg.is_scenario_active, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ScenarioEvent & msg,
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

  // member: event_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "event_time: ";
    rosidl_generator_traits::value_to_yaml(msg.event_time, out);
    out << "\n";
  }

  // member: event_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "event_type: ";
    rosidl_generator_traits::value_to_yaml(msg.event_type, out);
    out << "\n";
  }

  // member: cmd_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cmd_type: ";
    rosidl_generator_traits::value_to_yaml(msg.cmd_type, out);
    out << "\n";
  }

  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << "\n";
  }

  // member: led_r
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "led_r: ";
    rosidl_generator_traits::value_to_yaml(msg.led_r, out);
    out << "\n";
  }

  // member: led_g
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "led_g: ";
    rosidl_generator_traits::value_to_yaml(msg.led_g, out);
    out << "\n";
  }

  // member: led_b
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "led_b: ";
    rosidl_generator_traits::value_to_yaml(msg.led_b, out);
    out << "\n";
  }

  // member: is_scenario_active
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_scenario_active: ";
    rosidl_generator_traits::value_to_yaml(msg.is_scenario_active, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ScenarioEvent & msg, bool use_flow_style = false)
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
  const px4_msgs::msg::ScenarioEvent & msg,
  std::ostream & out, size_t indentation = 0)
{
  px4_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use px4_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const px4_msgs::msg::ScenarioEvent & msg)
{
  return px4_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<px4_msgs::msg::ScenarioEvent>()
{
  return "px4_msgs::msg::ScenarioEvent";
}

template<>
inline const char * name<px4_msgs::msg::ScenarioEvent>()
{
  return "px4_msgs/msg/ScenarioEvent";
}

template<>
struct has_fixed_size<px4_msgs::msg::ScenarioEvent>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<px4_msgs::msg::ScenarioEvent>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<px4_msgs::msg::ScenarioEvent>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PX4_MSGS__MSG__DETAIL__SCENARIO_EVENT__TRAITS_HPP_
