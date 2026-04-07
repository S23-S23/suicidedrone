// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from suicide_drone_msgs:msg/IBVSOutput.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__STRUCT_HPP_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__STRUCT_HPP_

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
# define DEPRECATED__suicide_drone_msgs__msg__IBVSOutput __attribute__((deprecated))
#else
# define DEPRECATED__suicide_drone_msgs__msg__IBVSOutput __declspec(deprecated)
#endif

namespace suicide_drone_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct IBVSOutput_
{
  using Type = IBVSOutput_<ContainerAllocator>;

  explicit IBVSOutput_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->detected = false;
      this->q_y = 0.0;
      this->q_z = 0.0;
      this->fov_yaw_rate = 0.0;
      this->fov_vel_z = 0.0;
    }
  }

  explicit IBVSOutput_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->detected = false;
      this->q_y = 0.0;
      this->q_z = 0.0;
      this->fov_yaw_rate = 0.0;
      this->fov_vel_z = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _detected_type =
    bool;
  _detected_type detected;
  using _q_y_type =
    double;
  _q_y_type q_y;
  using _q_z_type =
    double;
  _q_z_type q_z;
  using _fov_yaw_rate_type =
    double;
  _fov_yaw_rate_type fov_yaw_rate;
  using _fov_vel_z_type =
    double;
  _fov_vel_z_type fov_vel_z;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__detected(
    const bool & _arg)
  {
    this->detected = _arg;
    return *this;
  }
  Type & set__q_y(
    const double & _arg)
  {
    this->q_y = _arg;
    return *this;
  }
  Type & set__q_z(
    const double & _arg)
  {
    this->q_z = _arg;
    return *this;
  }
  Type & set__fov_yaw_rate(
    const double & _arg)
  {
    this->fov_yaw_rate = _arg;
    return *this;
  }
  Type & set__fov_vel_z(
    const double & _arg)
  {
    this->fov_vel_z = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator> *;
  using ConstRawPtr =
    const suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__suicide_drone_msgs__msg__IBVSOutput
    std::shared_ptr<suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__suicide_drone_msgs__msg__IBVSOutput
    std::shared_ptr<suicide_drone_msgs::msg::IBVSOutput_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const IBVSOutput_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->detected != other.detected) {
      return false;
    }
    if (this->q_y != other.q_y) {
      return false;
    }
    if (this->q_z != other.q_z) {
      return false;
    }
    if (this->fov_yaw_rate != other.fov_yaw_rate) {
      return false;
    }
    if (this->fov_vel_z != other.fov_vel_z) {
      return false;
    }
    return true;
  }
  bool operator!=(const IBVSOutput_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct IBVSOutput_

// alias to use template instance with default allocator
using IBVSOutput =
  suicide_drone_msgs::msg::IBVSOutput_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace suicide_drone_msgs

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__STRUCT_HPP_
