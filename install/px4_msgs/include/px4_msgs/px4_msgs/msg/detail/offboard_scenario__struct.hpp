// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from px4_msgs:msg/OffboardScenario.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__STRUCT_HPP_
#define PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__px4_msgs__msg__OffboardScenario __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__msg__OffboardScenario __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct OffboardScenario_
{
  using Type = OffboardScenario_<ContainerAllocator>;

  explicit OffboardScenario_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->current_time = 0ull;
      this->start_time = 0ull;
      this->seq = 0ul;
      this->offset_x = 0.0f;
      this->offset_y = 0.0f;
      this->ready_sc_file = 0;
    }
  }

  explicit OffboardScenario_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->current_time = 0ull;
      this->start_time = 0ull;
      this->seq = 0ul;
      this->offset_x = 0.0f;
      this->offset_y = 0.0f;
      this->ready_sc_file = 0;
    }
  }

  // field types and members
  using _timestamp_type =
    uint64_t;
  _timestamp_type timestamp;
  using _current_time_type =
    uint64_t;
  _current_time_type current_time;
  using _start_time_type =
    uint64_t;
  _start_time_type start_time;
  using _seq_type =
    uint32_t;
  _seq_type seq;
  using _offset_x_type =
    float;
  _offset_x_type offset_x;
  using _offset_y_type =
    float;
  _offset_y_type offset_y;
  using _ready_sc_file_type =
    uint8_t;
  _ready_sc_file_type ready_sc_file;

  // setters for named parameter idiom
  Type & set__timestamp(
    const uint64_t & _arg)
  {
    this->timestamp = _arg;
    return *this;
  }
  Type & set__current_time(
    const uint64_t & _arg)
  {
    this->current_time = _arg;
    return *this;
  }
  Type & set__start_time(
    const uint64_t & _arg)
  {
    this->start_time = _arg;
    return *this;
  }
  Type & set__seq(
    const uint32_t & _arg)
  {
    this->seq = _arg;
    return *this;
  }
  Type & set__offset_x(
    const float & _arg)
  {
    this->offset_x = _arg;
    return *this;
  }
  Type & set__offset_y(
    const float & _arg)
  {
    this->offset_y = _arg;
    return *this;
  }
  Type & set__ready_sc_file(
    const uint8_t & _arg)
  {
    this->ready_sc_file = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    px4_msgs::msg::OffboardScenario_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::msg::OffboardScenario_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::msg::OffboardScenario_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::msg::OffboardScenario_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::OffboardScenario_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::OffboardScenario_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::OffboardScenario_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::OffboardScenario_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::msg::OffboardScenario_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::msg::OffboardScenario_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__msg__OffboardScenario
    std::shared_ptr<px4_msgs::msg::OffboardScenario_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__msg__OffboardScenario
    std::shared_ptr<px4_msgs::msg::OffboardScenario_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const OffboardScenario_ & other) const
  {
    if (this->timestamp != other.timestamp) {
      return false;
    }
    if (this->current_time != other.current_time) {
      return false;
    }
    if (this->start_time != other.start_time) {
      return false;
    }
    if (this->seq != other.seq) {
      return false;
    }
    if (this->offset_x != other.offset_x) {
      return false;
    }
    if (this->offset_y != other.offset_y) {
      return false;
    }
    if (this->ready_sc_file != other.ready_sc_file) {
      return false;
    }
    return true;
  }
  bool operator!=(const OffboardScenario_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct OffboardScenario_

// alias to use template instance with default allocator
using OffboardScenario =
  px4_msgs::msg::OffboardScenario_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__STRUCT_HPP_
