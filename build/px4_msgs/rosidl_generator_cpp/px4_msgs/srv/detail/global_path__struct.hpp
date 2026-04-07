// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from px4_msgs:srv/GlobalPath.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__STRUCT_HPP_
#define PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__STRUCT_HPP_

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
# define DEPRECATED__px4_msgs__srv__GlobalPath_Request __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__srv__GlobalPath_Request __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GlobalPath_Request_
{
  using Type = GlobalPath_Request_<ContainerAllocator>;

  explicit GlobalPath_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit GlobalPath_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    px4_msgs::srv::GlobalPath_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::srv::GlobalPath_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::srv::GlobalPath_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::srv::GlobalPath_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::srv::GlobalPath_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::srv::GlobalPath_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::srv::GlobalPath_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::srv::GlobalPath_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::srv::GlobalPath_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::srv::GlobalPath_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__srv__GlobalPath_Request
    std::shared_ptr<px4_msgs::srv::GlobalPath_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__srv__GlobalPath_Request
    std::shared_ptr<px4_msgs::srv::GlobalPath_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GlobalPath_Request_ & other) const
  {
    if (this->waypoints != other.waypoints) {
      return false;
    }
    return true;
  }
  bool operator!=(const GlobalPath_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GlobalPath_Request_

// alias to use template instance with default allocator
using GlobalPath_Request =
  px4_msgs::srv::GlobalPath_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace px4_msgs


#ifndef _WIN32
# define DEPRECATED__px4_msgs__srv__GlobalPath_Response __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__srv__GlobalPath_Response __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GlobalPath_Response_
{
  using Type = GlobalPath_Response_<ContainerAllocator>;

  explicit GlobalPath_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->reply = false;
    }
  }

  explicit GlobalPath_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->reply = false;
    }
  }

  // field types and members
  using _reply_type =
    bool;
  _reply_type reply;

  // setters for named parameter idiom
  Type & set__reply(
    const bool & _arg)
  {
    this->reply = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    px4_msgs::srv::GlobalPath_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::srv::GlobalPath_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::srv::GlobalPath_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::srv::GlobalPath_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::srv::GlobalPath_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::srv::GlobalPath_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::srv::GlobalPath_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::srv::GlobalPath_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::srv::GlobalPath_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::srv::GlobalPath_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__srv__GlobalPath_Response
    std::shared_ptr<px4_msgs::srv::GlobalPath_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__srv__GlobalPath_Response
    std::shared_ptr<px4_msgs::srv::GlobalPath_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GlobalPath_Response_ & other) const
  {
    if (this->reply != other.reply) {
      return false;
    }
    return true;
  }
  bool operator!=(const GlobalPath_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GlobalPath_Response_

// alias to use template instance with default allocator
using GlobalPath_Response =
  px4_msgs::srv::GlobalPath_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace px4_msgs

namespace px4_msgs
{

namespace srv
{

struct GlobalPath
{
  using Request = px4_msgs::srv::GlobalPath_Request;
  using Response = px4_msgs::srv::GlobalPath_Response;
};

}  // namespace srv

}  // namespace px4_msgs

#endif  // PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__STRUCT_HPP_
