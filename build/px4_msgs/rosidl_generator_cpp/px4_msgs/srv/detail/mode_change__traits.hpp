// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from px4_msgs:srv/ModeChange.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__SRV__DETAIL__MODE_CHANGE__TRAITS_HPP_
#define PX4_MSGS__SRV__DETAIL__MODE_CHANGE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "px4_msgs/srv/detail/mode_change__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace px4_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const ModeChange_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: suv_mode
  {
    out << "suv_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.suv_mode, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ModeChange_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: suv_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "suv_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.suv_mode, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ModeChange_Request & msg, bool use_flow_style = false)
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
  const px4_msgs::srv::ModeChange_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  px4_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use px4_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const px4_msgs::srv::ModeChange_Request & msg)
{
  return px4_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<px4_msgs::srv::ModeChange_Request>()
{
  return "px4_msgs::srv::ModeChange_Request";
}

template<>
inline const char * name<px4_msgs::srv::ModeChange_Request>()
{
  return "px4_msgs/srv/ModeChange_Request";
}

template<>
struct has_fixed_size<px4_msgs::srv::ModeChange_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<px4_msgs::srv::ModeChange_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<px4_msgs::srv::ModeChange_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace px4_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const ModeChange_Response & msg,
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
  const ModeChange_Response & msg,
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

inline std::string to_yaml(const ModeChange_Response & msg, bool use_flow_style = false)
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
  const px4_msgs::srv::ModeChange_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  px4_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use px4_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const px4_msgs::srv::ModeChange_Response & msg)
{
  return px4_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<px4_msgs::srv::ModeChange_Response>()
{
  return "px4_msgs::srv::ModeChange_Response";
}

template<>
inline const char * name<px4_msgs::srv::ModeChange_Response>()
{
  return "px4_msgs/srv/ModeChange_Response";
}

template<>
struct has_fixed_size<px4_msgs::srv::ModeChange_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<px4_msgs::srv::ModeChange_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<px4_msgs::srv::ModeChange_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<px4_msgs::srv::ModeChange>()
{
  return "px4_msgs::srv::ModeChange";
}

template<>
inline const char * name<px4_msgs::srv::ModeChange>()
{
  return "px4_msgs/srv/ModeChange";
}

template<>
struct has_fixed_size<px4_msgs::srv::ModeChange>
  : std::integral_constant<
    bool,
    has_fixed_size<px4_msgs::srv::ModeChange_Request>::value &&
    has_fixed_size<px4_msgs::srv::ModeChange_Response>::value
  >
{
};

template<>
struct has_bounded_size<px4_msgs::srv::ModeChange>
  : std::integral_constant<
    bool,
    has_bounded_size<px4_msgs::srv::ModeChange_Request>::value &&
    has_bounded_size<px4_msgs::srv::ModeChange_Response>::value
  >
{
};

template<>
struct is_service<px4_msgs::srv::ModeChange>
  : std::true_type
{
};

template<>
struct is_service_request<px4_msgs::srv::ModeChange_Request>
  : std::true_type
{
};

template<>
struct is_service_response<px4_msgs::srv::ModeChange_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // PX4_MSGS__SRV__DETAIL__MODE_CHANGE__TRAITS_HPP_
