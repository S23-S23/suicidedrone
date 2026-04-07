// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from px4_msgs:msg/Monitoring.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__MONITORING__STRUCT_HPP_
#define PX4_MSGS__MSG__DETAIL__MONITORING__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__px4_msgs__msg__Monitoring __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__msg__Monitoring __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Monitoring_
{
  using Type = Monitoring_<ContainerAllocator>;

  explicit Monitoring_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->tow = 0ul;
      this->pos_x = 0.0f;
      this->pos_y = 0.0f;
      this->pos_z = 0.0f;
      this->lat = 0.0;
      this->lon = 0.0;
      this->alt = 0.0f;
      this->ref_lat = 0.0;
      this->ref_lon = 0.0;
      this->ref_alt = 0.0f;
      this->head = 0.0f;
      this->roll = 0.0f;
      this->pitch = 0.0f;
      this->status1 = 0ul;
      this->status2 = 0ul;
      this->rtk_nbase = 0;
      this->rtk_nrover = 0;
      this->battery = 0;
      this->r = 0;
      this->g = 0;
      this->b = 0;
      this->rtk_n = 0.0f;
      this->rtk_e = 0.0f;
      this->rtk_d = 0.0f;
      this->nav_state = 0;
    }
  }

  explicit Monitoring_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->tow = 0ul;
      this->pos_x = 0.0f;
      this->pos_y = 0.0f;
      this->pos_z = 0.0f;
      this->lat = 0.0;
      this->lon = 0.0;
      this->alt = 0.0f;
      this->ref_lat = 0.0;
      this->ref_lon = 0.0;
      this->ref_alt = 0.0f;
      this->head = 0.0f;
      this->roll = 0.0f;
      this->pitch = 0.0f;
      this->status1 = 0ul;
      this->status2 = 0ul;
      this->rtk_nbase = 0;
      this->rtk_nrover = 0;
      this->battery = 0;
      this->r = 0;
      this->g = 0;
      this->b = 0;
      this->rtk_n = 0.0f;
      this->rtk_e = 0.0f;
      this->rtk_d = 0.0f;
      this->nav_state = 0;
    }
  }

  // field types and members
  using _timestamp_type =
    uint64_t;
  _timestamp_type timestamp;
  using _tow_type =
    uint32_t;
  _tow_type tow;
  using _pos_x_type =
    float;
  _pos_x_type pos_x;
  using _pos_y_type =
    float;
  _pos_y_type pos_y;
  using _pos_z_type =
    float;
  _pos_z_type pos_z;
  using _lat_type =
    double;
  _lat_type lat;
  using _lon_type =
    double;
  _lon_type lon;
  using _alt_type =
    float;
  _alt_type alt;
  using _ref_lat_type =
    double;
  _ref_lat_type ref_lat;
  using _ref_lon_type =
    double;
  _ref_lon_type ref_lon;
  using _ref_alt_type =
    float;
  _ref_alt_type ref_alt;
  using _head_type =
    float;
  _head_type head;
  using _roll_type =
    float;
  _roll_type roll;
  using _pitch_type =
    float;
  _pitch_type pitch;
  using _status1_type =
    uint32_t;
  _status1_type status1;
  using _status2_type =
    uint32_t;
  _status2_type status2;
  using _rtk_nbase_type =
    uint8_t;
  _rtk_nbase_type rtk_nbase;
  using _rtk_nrover_type =
    uint8_t;
  _rtk_nrover_type rtk_nrover;
  using _battery_type =
    uint8_t;
  _battery_type battery;
  using _r_type =
    uint8_t;
  _r_type r;
  using _g_type =
    uint8_t;
  _g_type g;
  using _b_type =
    uint8_t;
  _b_type b;
  using _rtk_n_type =
    float;
  _rtk_n_type rtk_n;
  using _rtk_e_type =
    float;
  _rtk_e_type rtk_e;
  using _rtk_d_type =
    float;
  _rtk_d_type rtk_d;
  using _nav_state_type =
    uint8_t;
  _nav_state_type nav_state;

  // setters for named parameter idiom
  Type & set__timestamp(
    const uint64_t & _arg)
  {
    this->timestamp = _arg;
    return *this;
  }
  Type & set__tow(
    const uint32_t & _arg)
  {
    this->tow = _arg;
    return *this;
  }
  Type & set__pos_x(
    const float & _arg)
  {
    this->pos_x = _arg;
    return *this;
  }
  Type & set__pos_y(
    const float & _arg)
  {
    this->pos_y = _arg;
    return *this;
  }
  Type & set__pos_z(
    const float & _arg)
  {
    this->pos_z = _arg;
    return *this;
  }
  Type & set__lat(
    const double & _arg)
  {
    this->lat = _arg;
    return *this;
  }
  Type & set__lon(
    const double & _arg)
  {
    this->lon = _arg;
    return *this;
  }
  Type & set__alt(
    const float & _arg)
  {
    this->alt = _arg;
    return *this;
  }
  Type & set__ref_lat(
    const double & _arg)
  {
    this->ref_lat = _arg;
    return *this;
  }
  Type & set__ref_lon(
    const double & _arg)
  {
    this->ref_lon = _arg;
    return *this;
  }
  Type & set__ref_alt(
    const float & _arg)
  {
    this->ref_alt = _arg;
    return *this;
  }
  Type & set__head(
    const float & _arg)
  {
    this->head = _arg;
    return *this;
  }
  Type & set__roll(
    const float & _arg)
  {
    this->roll = _arg;
    return *this;
  }
  Type & set__pitch(
    const float & _arg)
  {
    this->pitch = _arg;
    return *this;
  }
  Type & set__status1(
    const uint32_t & _arg)
  {
    this->status1 = _arg;
    return *this;
  }
  Type & set__status2(
    const uint32_t & _arg)
  {
    this->status2 = _arg;
    return *this;
  }
  Type & set__rtk_nbase(
    const uint8_t & _arg)
  {
    this->rtk_nbase = _arg;
    return *this;
  }
  Type & set__rtk_nrover(
    const uint8_t & _arg)
  {
    this->rtk_nrover = _arg;
    return *this;
  }
  Type & set__battery(
    const uint8_t & _arg)
  {
    this->battery = _arg;
    return *this;
  }
  Type & set__r(
    const uint8_t & _arg)
  {
    this->r = _arg;
    return *this;
  }
  Type & set__g(
    const uint8_t & _arg)
  {
    this->g = _arg;
    return *this;
  }
  Type & set__b(
    const uint8_t & _arg)
  {
    this->b = _arg;
    return *this;
  }
  Type & set__rtk_n(
    const float & _arg)
  {
    this->rtk_n = _arg;
    return *this;
  }
  Type & set__rtk_e(
    const float & _arg)
  {
    this->rtk_e = _arg;
    return *this;
  }
  Type & set__rtk_d(
    const float & _arg)
  {
    this->rtk_d = _arg;
    return *this;
  }
  Type & set__nav_state(
    const uint8_t & _arg)
  {
    this->nav_state = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t NAVIGATION_STATE_MANUAL =
    0u;
  static constexpr uint8_t NAVIGATION_STATE_ALTCTL =
    1u;
  static constexpr uint8_t NAVIGATION_STATE_POSCTL =
    2u;
  static constexpr uint8_t NAVIGATION_STATE_AUTO_MISSION =
    3u;
  static constexpr uint8_t NAVIGATION_STATE_AUTO_LOITER =
    4u;
  static constexpr uint8_t NAVIGATION_STATE_AUTO_RTL =
    5u;
  static constexpr uint8_t NAVIGATION_STATE_UNUSED3 =
    8u;
  static constexpr uint8_t NAVIGATION_STATE_UNUSED =
    9u;
  static constexpr uint8_t NAVIGATION_STATE_ACRO =
    10u;
  static constexpr uint8_t NAVIGATION_STATE_UNUSED1 =
    11u;
  static constexpr uint8_t NAVIGATION_STATE_DESCEND =
    12u;
  static constexpr uint8_t NAVIGATION_STATE_TERMINATION =
    13u;
  static constexpr uint8_t NAVIGATION_STATE_OFFBOARD =
    14u;
  static constexpr uint8_t NAVIGATION_STATE_STAB =
    15u;
  static constexpr uint8_t NAVIGATION_STATE_UNUSED2 =
    16u;
  static constexpr uint8_t NAVIGATION_STATE_AUTO_TAKEOFF =
    17u;
  static constexpr uint8_t NAVIGATION_STATE_AUTO_LAND =
    18u;
  static constexpr uint8_t NAVIGATION_STATE_AUTO_FOLLOW_TARGET =
    19u;
  static constexpr uint8_t NAVIGATION_STATE_AUTO_PRECLAND =
    20u;
  static constexpr uint8_t NAVIGATION_STATE_ORBIT =
    21u;
  static constexpr uint8_t NAVIGATION_STATE_AUTO_VTOL_TAKEOFF =
    22u;
  static constexpr uint8_t NAVIGATION_STATE_MAX =
    23u;

  // pointer types
  using RawPtr =
    px4_msgs::msg::Monitoring_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::msg::Monitoring_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::msg::Monitoring_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::msg::Monitoring_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::Monitoring_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::Monitoring_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::Monitoring_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::Monitoring_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::msg::Monitoring_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::msg::Monitoring_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__msg__Monitoring
    std::shared_ptr<px4_msgs::msg::Monitoring_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__msg__Monitoring
    std::shared_ptr<px4_msgs::msg::Monitoring_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Monitoring_ & other) const
  {
    if (this->timestamp != other.timestamp) {
      return false;
    }
    if (this->tow != other.tow) {
      return false;
    }
    if (this->pos_x != other.pos_x) {
      return false;
    }
    if (this->pos_y != other.pos_y) {
      return false;
    }
    if (this->pos_z != other.pos_z) {
      return false;
    }
    if (this->lat != other.lat) {
      return false;
    }
    if (this->lon != other.lon) {
      return false;
    }
    if (this->alt != other.alt) {
      return false;
    }
    if (this->ref_lat != other.ref_lat) {
      return false;
    }
    if (this->ref_lon != other.ref_lon) {
      return false;
    }
    if (this->ref_alt != other.ref_alt) {
      return false;
    }
    if (this->head != other.head) {
      return false;
    }
    if (this->roll != other.roll) {
      return false;
    }
    if (this->pitch != other.pitch) {
      return false;
    }
    if (this->status1 != other.status1) {
      return false;
    }
    if (this->status2 != other.status2) {
      return false;
    }
    if (this->rtk_nbase != other.rtk_nbase) {
      return false;
    }
    if (this->rtk_nrover != other.rtk_nrover) {
      return false;
    }
    if (this->battery != other.battery) {
      return false;
    }
    if (this->r != other.r) {
      return false;
    }
    if (this->g != other.g) {
      return false;
    }
    if (this->b != other.b) {
      return false;
    }
    if (this->rtk_n != other.rtk_n) {
      return false;
    }
    if (this->rtk_e != other.rtk_e) {
      return false;
    }
    if (this->rtk_d != other.rtk_d) {
      return false;
    }
    if (this->nav_state != other.nav_state) {
      return false;
    }
    return true;
  }
  bool operator!=(const Monitoring_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Monitoring_

// alias to use template instance with default allocator
using Monitoring =
  px4_msgs::msg::Monitoring_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_MANUAL;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_ALTCTL;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_POSCTL;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_AUTO_MISSION;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_AUTO_LOITER;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_AUTO_RTL;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_UNUSED3;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_UNUSED;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_ACRO;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_UNUSED1;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_DESCEND;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_TERMINATION;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_OFFBOARD;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_STAB;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_UNUSED2;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_AUTO_TAKEOFF;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_AUTO_LAND;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_AUTO_FOLLOW_TARGET;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_AUTO_PRECLAND;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_ORBIT;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_AUTO_VTOL_TAKEOFF;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Monitoring_<ContainerAllocator>::NAVIGATION_STATE_MAX;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__MONITORING__STRUCT_HPP_
