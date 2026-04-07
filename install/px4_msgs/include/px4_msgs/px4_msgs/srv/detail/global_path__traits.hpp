// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from px4_msgs:srv/GlobalPath.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__TRAITS_HPP_
#define PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "px4_msgs/srv/detail/global_path__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'waypoints'
#include "px4_msgs/msg/detail/trajectory_setpoint__traits.hpp"

namespace px4_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const GlobalPath_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: waypoints
  {
    if (msg.waypoints.size() == 0) {
      out << "waypoints: []";
    } else {
      out << "waypoints: [";
      size_t pending_items = msg.waypoints.size();
      for (auto item : msg.waypoints) {
        to_flow_style_yaml(item, out);
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
  const GlobalPath_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: waypoints
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.waypoints.size() == 0) {
      out << "waypoints: []\n";
    } else {
      out << "waypoints:\n";
      for (auto item : msg.waypoints) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GlobalPath_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace px4_msgs

namespace rosidl_generator_traits
{

[[deprecated("use px4_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const px4_msgs::srv::GlobalPath_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  px4_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use px4_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const px4_msgs::srv::GlobalPath_Request & msg)
{
  return px4_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<px4_msgs::srv::GlobalPath_Request>()
{
  return "px4_msgs::srv::GlobalPath_Request";
}

template<>
inline const char * name<px4_msgs::srv::GlobalPath_Request>()
{
  return "px4_msgs/srv/GlobalPath_Request";
}

template<>
struct has_fixed_size<px4_msgs::srv::GlobalPath_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<px4_msgs::srv::GlobalPath_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<px4_msgs::srv::GlobalPath_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace px4_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const GlobalPath_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: reply
  {
    out << "reply: ";
    rosidl_generator_traits::value_to_yaml(msg.reply, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GlobalPath_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: reply
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reply: ";
    rosidl_generator_traits::value_to_yaml(msg.reply, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GlobalPath_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace px4_msgs

namespace rosidl_generator_traits
{

[[deprecated("use px4_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const px4_msgs::srv::GlobalPath_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  px4_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use px4_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const px4_msgs::srv::GlobalPath_Response & msg)
{
  return px4_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<px4_msgs::srv::GlobalPath_Response>()
{
  return "px4_msgs::srv::GlobalPath_Response";
}

template<>
inline const char * name<px4_msgs::srv::GlobalPath_Response>()
{
  return "px4_msgs/srv/GlobalPath_Response";
}

template<>
struct has_fixed_size<px4_msgs::srv::GlobalPath_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<px4_msgs::srv::GlobalPath_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<px4_msgs::srv::GlobalPath_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<px4_msgs::srv::GlobalPath>()
{
  return "px4_msgs::srv::GlobalPath";
}

template<>
inline const char * name<px4_msgs::srv::GlobalPath>()
{
  return "px4_msgs/srv/GlobalPath";
}

template<>
struct has_fixed_size<px4_msgs::srv::GlobalPath>
  : std::integral_constant<
    bool,
    has_fixed_size<px4_msgs::srv::GlobalPath_Request>::value &&
    has_fixed_size<px4_msgs::srv::GlobalPath_Response>::value
  >
{
};

template<>
struct has_bounded_size<px4_msgs::srv::GlobalPath>
  : std::integral_constant<
    bool,
    has_bounded_size<px4_msgs::srv::GlobalPath_Request>::value &&
    has_bounded_size<px4_msgs::srv::GlobalPath_Response>::value
  >
{
};

template<>
struct is_service<px4_msgs::srv::GlobalPath>
  : std::true_type
{
};

template<>
struct is_service_request<px4_msgs::srv::GlobalPath_Request>
  : std::true_type
{
};

template<>
struct is_service_response<px4_msgs::srv::GlobalPath_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__TRAITS_HPP_
