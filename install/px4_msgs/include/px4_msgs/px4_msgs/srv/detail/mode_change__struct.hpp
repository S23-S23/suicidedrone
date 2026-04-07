// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from px4_msgs:srv/ModeChange.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__SRV__DETAIL__MODE_CHANGE__STRUCT_HPP_
#define PX4_MSGS__SRV__DETAIL__MODE_CHANGE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__px4_msgs__srv__ModeChange_Request __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__srv__ModeChange_Request __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct ModeChange_Request_
{
  using Type = ModeChange_Request_<ContainerAllocator>;

  explicit ModeChange_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->suv_mode = 0;
    }
  }

  explicit ModeChange_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->suv_mode = 0;
    }
  }

  // field types and members
  using _suv_mode_type =
    uint8_t;
  _suv_mode_type suv_mode;

  // setters for named parameter idiom
  Type & set__suv_mode(
    const uint8_t & _arg)
  {
    this->suv_mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    px4_msgs::srv::ModeChange_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::srv::ModeChange_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::srv::ModeChange_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::srv::ModeChange_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::srv::ModeChange_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::srv::ModeChange_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::srv::ModeChange_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::srv::ModeChange_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::srv::ModeChange_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::srv::ModeChange_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__srv__ModeChange_Request
    std::shared_ptr<px4_msgs::srv::ModeChange_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__srv__ModeChange_Request
    std::shared_ptr<px4_msgs::srv::ModeChange_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ModeChange_Request_ & other) const
  {
    if (this->suv_mode != other.suv_mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const ModeChange_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ModeChange_Request_

// alias to use template instance with default allocator
using ModeChange_Request =
  px4_msgs::srv::ModeChange_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace px4_msgs


#ifndef _WIN32
# define DEPRECATED__px4_msgs__srv__ModeChange_Response __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__srv__ModeChange_Response __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct ModeChange_Response_
{
  using Type = ModeChange_Response_<ContainerAllocator>;

  explicit ModeChange_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->reply = false;
    }
  }

  explicit ModeChange_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    px4_msgs::srv::ModeChange_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::srv::ModeChange_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::srv::ModeChange_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::srv::ModeChange_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::srv::ModeChange_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::srv::ModeChange_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::srv::ModeChange_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::srv::ModeChange_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::srv::ModeChange_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::srv::ModeChange_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__srv__ModeChange_Response
    std::shared_ptr<px4_msgs::srv::ModeChange_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__srv__ModeChange_Response
    std::shared_ptr<px4_msgs::srv::ModeChange_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ModeChange_Response_ & other) const
  {
    if (this->reply != other.reply) {
      return false;
    }
    return true;
  }
  bool operator!=(const ModeChange_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ModeChange_Response_

// alias to use template instance with default allocator
using ModeChange_Response =
  px4_msgs::srv::ModeChange_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace px4_msgs

namespace px4_msgs
{

namespace srv
{

struct ModeChange
{
  using Request = px4_msgs::srv::ModeChange_Request;
  using Response = px4_msgs::srv::ModeChange_Response;
};

}  // namespace srv

}  // namespace px4_msgs

#endif  // PX4_MSGS__SRV__DETAIL__MODE_CHANGE__STRUCT_HPP_
