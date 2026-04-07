// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from px4_msgs:msg/F9pRtk.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__F9P_RTK__TRAITS_HPP_
#define PX4_MSGS__MSG__DETAIL__F9P_RTK__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "px4_msgs/msg/detail/f9p_rtk__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace px4_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const F9pRtk & msg,
  std::ostream & out)
{
  out << "{";
  // member: timestamp
  {
    out << "timestamp: ";
    rosidl_generator_traits::value_to_yaml(msg.timestamp, out);
    out << ", ";
  }

  // member: device_id
  {
    out << "device_id: ";
    rosidl_generator_traits::value_to_yaml(msg.device_id, out);
    out << ", ";
  }

  // member: tow
  {
    out << "tow: ";
    rosidl_generator_traits::value_to_yaml(msg.tow, out);
    out << ", ";
  }

  // member: age_corr
  {
    out << "age_corr: ";
    rosidl_generator_traits::value_to_yaml(msg.age_corr, out);
    out << ", ";
  }

  // member: fix_type
  {
    out << "fix_type: ";
    rosidl_generator_traits::value_to_yaml(msg.fix_type, out);
    out << ", ";
  }

  // member: satellites_used
  {
    out << "satellites_used: ";
    rosidl_generator_traits::value_to_yaml(msg.satellites_used, out);
    out << ", ";
  }

  // member: n
  {
    out << "n: ";
    rosidl_generator_traits::value_to_yaml(msg.n, out);
    out << ", ";
  }

  // member: e
  {
    out << "e: ";
    rosidl_generator_traits::value_to_yaml(msg.e, out);
    out << ", ";
  }

  // member: d
  {
    out << "d: ";
    rosidl_generator_traits::value_to_yaml(msg.d, out);
    out << ", ";
  }

  // member: v_n
  {
    out << "v_n: ";
    rosidl_generator_traits::value_to_yaml(msg.v_n, out);
    out << ", ";
  }

  // member: v_e
  {
    out << "v_e: ";
    rosidl_generator_traits::value_to_yaml(msg.v_e, out);
    out << ", ";
  }

  // member: v_d
  {
    out << "v_d: ";
    rosidl_generator_traits::value_to_yaml(msg.v_d, out);
    out << ", ";
  }

  // member: acc_n
  {
    out << "acc_n: ";
    rosidl_generator_traits::value_to_yaml(msg.acc_n, out);
    out << ", ";
  }

  // member: acc_e
  {
    out << "acc_e: ";
    rosidl_generator_traits::value_to_yaml(msg.acc_e, out);
    out << ", ";
  }

  // member: acc_d
  {
    out << "acc_d: ";
    rosidl_generator_traits::value_to_yaml(msg.acc_d, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const F9pRtk & msg,
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

  // member: device_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "device_id: ";
    rosidl_generator_traits::value_to_yaml(msg.device_id, out);
    out << "\n";
  }

  // member: tow
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tow: ";
    rosidl_generator_traits::value_to_yaml(msg.tow, out);
    out << "\n";
  }

  // member: age_corr
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "age_corr: ";
    rosidl_generator_traits::value_to_yaml(msg.age_corr, out);
    out << "\n";
  }

  // member: fix_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "fix_type: ";
    rosidl_generator_traits::value_to_yaml(msg.fix_type, out);
    out << "\n";
  }

  // member: satellites_used
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "satellites_used: ";
    rosidl_generator_traits::value_to_yaml(msg.satellites_used, out);
    out << "\n";
  }

  // member: n
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "n: ";
    rosidl_generator_traits::value_to_yaml(msg.n, out);
    out << "\n";
  }

  // member: e
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "e: ";
    rosidl_generator_traits::value_to_yaml(msg.e, out);
    out << "\n";
  }

  // member: d
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "d: ";
    rosidl_generator_traits::value_to_yaml(msg.d, out);
    out << "\n";
  }

  // member: v_n
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "v_n: ";
    rosidl_generator_traits::value_to_yaml(msg.v_n, out);
    out << "\n";
  }

  // member: v_e
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "v_e: ";
    rosidl_generator_traits::value_to_yaml(msg.v_e, out);
    out << "\n";
  }

  // member: v_d
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "v_d: ";
    rosidl_generator_traits::value_to_yaml(msg.v_d, out);
    out << "\n";
  }

  // member: acc_n
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "acc_n: ";
    rosidl_generator_traits::value_to_yaml(msg.acc_n, out);
    out << "\n";
  }

  // member: acc_e
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "acc_e: ";
    rosidl_generator_traits::value_to_yaml(msg.acc_e, out);
    out << "\n";
  }

  // member: acc_d
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "acc_d: ";
    rosidl_generator_traits::value_to_yaml(msg.acc_d, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const F9pRtk & msg, bool use_flow_style = false)
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
  const px4_msgs::msg::F9pRtk & msg,
  std::ostream & out, size_t indentation = 0)
{
  px4_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use px4_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const px4_msgs::msg::F9pRtk & msg)
{
  return px4_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<px4_msgs::msg::F9pRtk>()
{
  return "px4_msgs::msg::F9pRtk";
}

template<>
inline const char * name<px4_msgs::msg::F9pRtk>()
{
  return "px4_msgs/msg/F9pRtk";
}

template<>
struct has_fixed_size<px4_msgs::msg::F9pRtk>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<px4_msgs::msg::F9pRtk>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<px4_msgs::msg::F9pRtk>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PX4_MSGS__MSG__DETAIL__F9P_RTK__TRAITS_HPP_
