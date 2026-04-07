// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from px4_msgs:msg/ScenarioCommand.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__TRAITS_HPP_
#define PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "px4_msgs/msg/detail/scenario_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace px4_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ScenarioCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: timestamp
  {
    out << "timestamp: ";
    rosidl_generator_traits::value_to_yaml(msg.timestamp, out);
    out << ", ";
  }

  // member: cmd
  {
    out << "cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.cmd, out);
    out << ", ";
  }

  // member: param1
  {
    out << "param1: ";
    rosidl_generator_traits::value_to_yaml(msg.param1, out);
    out << ", ";
  }

  // member: param2
  {
    out << "param2: ";
    rosidl_generator_traits::value_to_yaml(msg.param2, out);
    out << ", ";
  }

  // member: param3
  {
    out << "param3: ";
    rosidl_generator_traits::value_to_yaml(msg.param3, out);
    out << ", ";
  }

  // member: param4
  {
    out << "param4: ";
    rosidl_generator_traits::value_to_yaml(msg.param4, out);
    out << ", ";
  }

  // member: param5
  {
    if (msg.param5.size() == 0) {
      out << "param5: []";
    } else {
      out << "param5: [";
      size_t pending_items = msg.param5.size();
      for (auto item : msg.param5) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ScenarioCommand & msg,
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

  // member: cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.cmd, out);
    out << "\n";
  }

  // member: param1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "param1: ";
    rosidl_generator_traits::value_to_yaml(msg.param1, out);
    out << "\n";
  }

  // member: param2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "param2: ";
    rosidl_generator_traits::value_to_yaml(msg.param2, out);
    out << "\n";
  }

  // member: param3
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "param3: ";
    rosidl_generator_traits::value_to_yaml(msg.param3, out);
    out << "\n";
  }

  // member: param4
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "param4: ";
    rosidl_generator_traits::value_to_yaml(msg.param4, out);
    out << "\n";
  }

  // member: param5
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.param5.size() == 0) {
      out << "param5: []\n";
    } else {
      out << "param5:\n";
      for (auto item : msg.param5) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ScenarioCommand & msg, bool use_flow_style = false)
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
  const px4_msgs::msg::ScenarioCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  px4_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use px4_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const px4_msgs::msg::ScenarioCommand & msg)
{
  return px4_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<px4_msgs::msg::ScenarioCommand>()
{
  return "px4_msgs::msg::ScenarioCommand";
}

template<>
inline const char * name<px4_msgs::msg::ScenarioCommand>()
{
  return "px4_msgs/msg/ScenarioCommand";
}

template<>
struct has_fixed_size<px4_msgs::msg::ScenarioCommand>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<px4_msgs::msg::ScenarioCommand>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<px4_msgs::msg::ScenarioCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__TRAITS_HPP_
