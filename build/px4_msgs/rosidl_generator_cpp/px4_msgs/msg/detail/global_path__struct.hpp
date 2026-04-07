// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from px4_msgs:msg/GlobalPath.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__GLOBAL_PATH__STRUCT_HPP_
#define PX4_MSGS__MSG__DETAIL__GLOBAL_PATH__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'waypoints'
#include "px4_msgs/msg/detail/trajectory_setpoint__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__px4_msgs__msg__GlobalPath __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__msg__GlobalPath __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct GlobalPath_
{
  using Type = GlobalPath_<ContainerAllocator>;

  explicit GlobalPath_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit GlobalPath_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _waypoints_type =
    std::vector<px4_msgs::msg::TrajectorySetpoint_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<px4_msgs::msg::TrajectorySetpoint_<ContainerAllocator>>>;
  _waypoints_type waypoints;

  // setters for named parameter idiom
  Type & set__waypoints(
    const std::vector<px4_msgs::msg::TrajectorySetpoint_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<px4_msgs::msg::TrajectorySetpoint_<ContainerAllocator>>> & _arg)
  {
    this->waypoints = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    px4_msgs::msg::GlobalPath_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::msg::GlobalPath_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::msg::GlobalPath_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::msg::GlobalPath_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::GlobalPath_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::GlobalPath_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::GlobalPath_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::GlobalPath_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::msg::GlobalPath_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::msg::GlobalPath_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__msg__GlobalPath
    std::shared_ptr<px4_msgs::msg::GlobalPath_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__msg__GlobalPath
    std::shared_ptr<px4_msgs::msg::GlobalPath_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GlobalPath_ & other) const
  {
    if (this->waypoints != other.waypoints) {
      return false;
    }
    return true;
  }
  bool operator!=(const GlobalPath_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GlobalPath_

// alias to use template instance with default allocator
using GlobalPath =
  px4_msgs::msg::GlobalPath_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__GLOBAL_PATH__STRUCT_HPP_
