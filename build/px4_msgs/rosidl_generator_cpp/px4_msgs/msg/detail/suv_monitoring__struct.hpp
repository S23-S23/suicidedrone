// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from px4_msgs:msg/SuvMonitoring.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SUV_MONITORING__STRUCT_HPP_
#define PX4_MSGS__MSG__DETAIL__SUV_MONITORING__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'monitoring'
#include "px4_msgs/msg/detail/monitoring__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__px4_msgs__msg__SuvMonitoring __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__msg__SuvMonitoring __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SuvMonitoring_
{
  using Type = SuvMonitoring_<ContainerAllocator>;

  explicit SuvMonitoring_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : monitoring(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mode = 0;
    }
  }

  explicit SuvMonitoring_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : monitoring(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mode = 0;
    }
  }

  // field types and members
  using _monitoring_type =
    px4_msgs::msg::Monitoring_<ContainerAllocator>;
  _monitoring_type monitoring;
  using _mode_type =
    uint8_t;
  _mode_type mode;

  // setters for named parameter idiom
  Type & set__monitoring(
    const px4_msgs::msg::Monitoring_<ContainerAllocator> & _arg)
  {
    this->monitoring = _arg;
    return *this;
  }
  Type & set__mode(
    const uint8_t & _arg)
  {
    this->mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    px4_msgs::msg::SuvMonitoring_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::msg::SuvMonitoring_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::msg::SuvMonitoring_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::msg::SuvMonitoring_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::SuvMonitoring_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::SuvMonitoring_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::SuvMonitoring_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::SuvMonitoring_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::msg::SuvMonitoring_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::msg::SuvMonitoring_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__msg__SuvMonitoring
    std::shared_ptr<px4_msgs::msg::SuvMonitoring_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__msg__SuvMonitoring
    std::shared_ptr<px4_msgs::msg::SuvMonitoring_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SuvMonitoring_ & other) const
  {
    if (this->monitoring != other.monitoring) {
      return false;
    }
    if (this->mode != other.mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const SuvMonitoring_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SuvMonitoring_

// alias to use template instance with default allocator
using SuvMonitoring =
  px4_msgs::msg::SuvMonitoring_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__SUV_MONITORING__STRUCT_HPP_
