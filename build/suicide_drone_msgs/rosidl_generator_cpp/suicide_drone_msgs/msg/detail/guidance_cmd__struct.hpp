// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from suicide_drone_msgs:msg/GuidanceCmd.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__STRUCT_HPP_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__suicide_drone_msgs__msg__GuidanceCmd __attribute__((deprecated))
#else
# define DEPRECATED__suicide_drone_msgs__msg__GuidanceCmd __declspec(deprecated)
#endif

namespace suicide_drone_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct GuidanceCmd_
{
  using Type = GuidanceCmd_<ContainerAllocator>;

  explicit GuidanceCmd_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->target_detected = false;
      this->vel_n = 0.0;
      this->vel_e = 0.0;
      this->vel_d = 0.0;
      this->yaw_rate = 0.0;
    }
  }

  explicit GuidanceCmd_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->target_detected = false;
      this->vel_n = 0.0;
      this->vel_e = 0.0;
      this->vel_d = 0.0;
      this->yaw_rate = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _target_detected_type =
    bool;
  _target_detected_type target_detected;
  using _vel_n_type =
    double;
  _vel_n_type vel_n;
  using _vel_e_type =
    double;
  _vel_e_type vel_e;
  using _vel_d_type =
    double;
  _vel_d_type vel_d;
  using _yaw_rate_type =
    double;
  _yaw_rate_type yaw_rate;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__target_detected(
    const bool & _arg)
  {
    this->target_detected = _arg;
    return *this;
  }
  Type & set__vel_n(
    const double & _arg)
  {
    this->vel_n = _arg;
    return *this;
  }
  Type & set__vel_e(
    const double & _arg)
  {
    this->vel_e = _arg;
    return *this;
  }
  Type & set__vel_d(
    const double & _arg)
  {
    this->vel_d = _arg;
    return *this;
  }
  Type & set__yaw_rate(
    const double & _arg)
  {
    this->yaw_rate = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator> *;
  using ConstRawPtr =
    const suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__suicide_drone_msgs__msg__GuidanceCmd
    std::shared_ptr<suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__suicide_drone_msgs__msg__GuidanceCmd
    std::shared_ptr<suicide_drone_msgs::msg::GuidanceCmd_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GuidanceCmd_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->target_detected != other.target_detected) {
      return false;
    }
    if (this->vel_n != other.vel_n) {
      return false;
    }
    if (this->vel_e != other.vel_e) {
      return false;
    }
    if (this->vel_d != other.vel_d) {
      return false;
    }
    if (this->yaw_rate != other.yaw_rate) {
      return false;
    }
    return true;
  }
  bool operator!=(const GuidanceCmd_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GuidanceCmd_

// alias to use template instance with default allocator
using GuidanceCmd =
  suicide_drone_msgs::msg::GuidanceCmd_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace suicide_drone_msgs

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__STRUCT_HPP_
