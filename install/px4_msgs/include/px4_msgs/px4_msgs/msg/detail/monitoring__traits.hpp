// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from px4_msgs:msg/Monitoring.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__MONITORING__TRAITS_HPP_
#define PX4_MSGS__MSG__DETAIL__MONITORING__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "px4_msgs/msg/detail/monitoring__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace px4_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Monitoring & msg,
  std::ostream & out)
{
  out << "{";
  // member: timestamp
  {
    out << "timestamp: ";
    rosidl_generator_traits::value_to_yaml(msg.timestamp, out);
    out << ", ";
  }

  // member: tow
  {
    out << "tow: ";
    rosidl_generator_traits::value_to_yaml(msg.tow, out);
    out << ", ";
  }

  // member: pos_x
  {
    out << "pos_x: ";
    rosidl_generator_traits::value_to_yaml(msg.pos_x, out);
    out << ", ";
  }

  // member: pos_y
  {
    out << "pos_y: ";
    rosidl_generator_traits::value_to_yaml(msg.pos_y, out);
    out << ", ";
  }

  // member: pos_z
  {
    out << "pos_z: ";
    rosidl_generator_traits::value_to_yaml(msg.pos_z, out);
    out << ", ";
  }

  // member: lat
  {
    out << "lat: ";
    rosidl_generator_traits::value_to_yaml(msg.lat, out);
    out << ", ";
  }

  // member: lon
  {
    out << "lon: ";
    rosidl_generator_traits::value_to_yaml(msg.lon, out);
    out << ", ";
  }

  // member: alt
  {
    out << "alt: ";
    rosidl_generator_traits::value_to_yaml(msg.alt, out);
    out << ", ";
  }

  // member: ref_lat
  {
    out << "ref_lat: ";
    rosidl_generator_traits::value_to_yaml(msg.ref_lat, out);
    out << ", ";
  }

  // member: ref_lon
  {
    out << "ref_lon: ";
    rosidl_generator_traits::value_to_yaml(msg.ref_lon, out);
    out << ", ";
  }

  // member: ref_alt
  {
    out << "ref_alt: ";
    rosidl_generator_traits::value_to_yaml(msg.ref_alt, out);
    out << ", ";
  }

  // member: head
  {
    out << "head: ";
    rosidl_generator_traits::value_to_yaml(msg.head, out);
    out << ", ";
  }

  // member: roll
  {
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << ", ";
  }

  // member: pitch
  {
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << ", ";
  }

  // member: status1
  {
    out << "status1: ";
    rosidl_generator_traits::value_to_yaml(msg.status1, out);
    out << ", ";
  }

  // member: status2
  {
    out << "status2: ";
    rosidl_generator_traits::value_to_yaml(msg.status2, out);
    out << ", ";
  }

  // member: rtk_nbase
  {
    out << "rtk_nbase: ";
    rosidl_generator_traits::value_to_yaml(msg.rtk_nbase, out);
    out << ", ";
  }

  // member: rtk_nrover
  {
    out << "rtk_nrover: ";
    rosidl_generator_traits::value_to_yaml(msg.rtk_nrover, out);
    out << ", ";
  }

  // member: battery
  {
    out << "battery: ";
    rosidl_generator_traits::value_to_yaml(msg.battery, out);
    out << ", ";
  }

  // member: r
  {
    out << "r: ";
    rosidl_generator_traits::value_to_yaml(msg.r, out);
    out << ", ";
  }

  // member: g
  {
    out << "g: ";
    rosidl_generator_traits::value_to_yaml(msg.g, out);
    out << ", ";
  }

  // member: b
  {
    out << "b: ";
    rosidl_generator_traits::value_to_yaml(msg.b, out);
    out << ", ";
  }

  // member: rtk_n
  {
    out << "rtk_n: ";
    rosidl_generator_traits::value_to_yaml(msg.rtk_n, out);
    out << ", ";
  }

  // member: rtk_e
  {
    out << "rtk_e: ";
    rosidl_generator_traits::value_to_yaml(msg.rtk_e, out);
    out << ", ";
  }

  // member: rtk_d
  {
    out << "rtk_d: ";
    rosidl_generator_traits::value_to_yaml(msg.rtk_d, out);
    out << ", ";
  }

  // member: nav_state
  {
    out << "nav_state: ";
    rosidl_generator_traits::value_to_yaml(msg.nav_state, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Monitoring & msg,
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

  // member: tow
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tow: ";
    rosidl_generator_traits::value_to_yaml(msg.tow, out);
    out << "\n";
  }

  // member: pos_x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pos_x: ";
    rosidl_generator_traits::value_to_yaml(msg.pos_x, out);
    out << "\n";
  }

  // member: pos_y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pos_y: ";
    rosidl_generator_traits::value_to_yaml(msg.pos_y, out);
    out << "\n";
  }

  // member: pos_z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pos_z: ";
    rosidl_generator_traits::value_to_yaml(msg.pos_z, out);
    out << "\n";
  }

  // member: lat
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "lat: ";
    rosidl_generator_traits::value_to_yaml(msg.lat, out);
    out << "\n";
  }

  // member: lon
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "lon: ";
    rosidl_generator_traits::value_to_yaml(msg.lon, out);
    out << "\n";
  }

  // member: alt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "alt: ";
    rosidl_generator_traits::value_to_yaml(msg.alt, out);
    out << "\n";
  }

  // member: ref_lat
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ref_lat: ";
    rosidl_generator_traits::value_to_yaml(msg.ref_lat, out);
    out << "\n";
  }

  // member: ref_lon
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ref_lon: ";
    rosidl_generator_traits::value_to_yaml(msg.ref_lon, out);
    out << "\n";
  }

  // member: ref_alt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ref_alt: ";
    rosidl_generator_traits::value_to_yaml(msg.ref_alt, out);
    out << "\n";
  }

  // member: head
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "head: ";
    rosidl_generator_traits::value_to_yaml(msg.head, out);
    out << "\n";
  }

  // member: roll
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << "\n";
  }

  // member: pitch
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << "\n";
  }

  // member: status1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status1: ";
    rosidl_generator_traits::value_to_yaml(msg.status1, out);
    out << "\n";
  }

  // member: status2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status2: ";
    rosidl_generator_traits::value_to_yaml(msg.status2, out);
    out << "\n";
  }

  // member: rtk_nbase
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rtk_nbase: ";
    rosidl_generator_traits::value_to_yaml(msg.rtk_nbase, out);
    out << "\n";
  }

  // member: rtk_nrover
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rtk_nrover: ";
    rosidl_generator_traits::value_to_yaml(msg.rtk_nrover, out);
    out << "\n";
  }

  // member: battery
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "battery: ";
    rosidl_generator_traits::value_to_yaml(msg.battery, out);
    out << "\n";
  }

  // member: r
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "r: ";
    rosidl_generator_traits::value_to_yaml(msg.r, out);
    out << "\n";
  }

  // member: g
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "g: ";
    rosidl_generator_traits::value_to_yaml(msg.g, out);
    out << "\n";
  }

  // member: b
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "b: ";
    rosidl_generator_traits::value_to_yaml(msg.b, out);
    out << "\n";
  }

  // member: rtk_n
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rtk_n: ";
    rosidl_generator_traits::value_to_yaml(msg.rtk_n, out);
    out << "\n";
  }

  // member: rtk_e
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rtk_e: ";
    rosidl_generator_traits::value_to_yaml(msg.rtk_e, out);
    out << "\n";
  }

  // member: rtk_d
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rtk_d: ";
    rosidl_generator_traits::value_to_yaml(msg.rtk_d, out);
    out << "\n";
  }

  // member: nav_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "nav_state: ";
    rosidl_generator_traits::value_to_yaml(msg.nav_state, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Monitoring & msg, bool use_flow_style = false)
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
  const px4_msgs::msg::Monitoring & msg,
  std::ostream & out, size_t indentation = 0)
{
  px4_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use px4_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const px4_msgs::msg::Monitoring & msg)
{
  return px4_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<px4_msgs::msg::Monitoring>()
{
  return "px4_msgs::msg::Monitoring";
}

template<>
inline const char * name<px4_msgs::msg::Monitoring>()
{
  return "px4_msgs/msg/Monitoring";
}

template<>
struct has_fixed_size<px4_msgs::msg::Monitoring>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<px4_msgs::msg::Monitoring>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<px4_msgs::msg::Monitoring>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PX4_MSGS__MSG__DETAIL__MONITORING__TRAITS_HPP_
