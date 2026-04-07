// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from px4_msgs:msg/ShowLedControl.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SHOW_LED_CONTROL__STRUCT_HPP_
#define PX4_MSGS__MSG__DETAIL__SHOW_LED_CONTROL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__px4_msgs__msg__ShowLedControl __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__msg__ShowLedControl __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ShowLedControl_
{
  using Type = ShowLedControl_<ContainerAllocator>;

  explicit ShowLedControl_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->type = 0;
      this->r = 0;
      this->g = 0;
      this->b = 0;
      this->brightness = 0;
      this->speed = 0;
    }
  }

  explicit ShowLedControl_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->type = 0;
      this->r = 0;
      this->g = 0;
      this->b = 0;
      this->brightness = 0;
      this->speed = 0;
    }
  }

  // field types and members
  using _timestamp_type =
    uint64_t;
  _timestamp_type timestamp;
  using _type_type =
    int8_t;
  _type_type type;
  using _r_type =
    int8_t;
  _r_type r;
  using _g_type =
    int8_t;
  _g_type g;
  using _b_type =
    int8_t;
  _b_type b;
  using _brightness_type =
    int8_t;
  _brightness_type brightness;
  using _speed_type =
    int8_t;
  _speed_type speed;

  // setters for named parameter idiom
  Type & set__timestamp(
    const uint64_t & _arg)
  {
    this->timestamp = _arg;
    return *this;
  }
  Type & set__type(
    const int8_t & _arg)
  {
    this->type = _arg;
    return *this;
  }
  Type & set__r(
    const int8_t & _arg)
  {
    this->r = _arg;
    return *this;
  }
  Type & set__g(
    const int8_t & _arg)
  {
    this->g = _arg;
    return *this;
  }
  Type & set__b(
    const int8_t & _arg)
  {
    this->b = _arg;
    return *this;
  }
  Type & set__brightness(
    const int8_t & _arg)
  {
    this->brightness = _arg;
    return *this;
  }
  Type & set__speed(
    const int8_t & _arg)
  {
    this->speed = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    px4_msgs::msg::ShowLedControl_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::msg::ShowLedControl_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::msg::ShowLedControl_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::msg::ShowLedControl_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::ShowLedControl_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::ShowLedControl_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::ShowLedControl_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::ShowLedControl_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::msg::ShowLedControl_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::msg::ShowLedControl_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__msg__ShowLedControl
    std::shared_ptr<px4_msgs::msg::ShowLedControl_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__msg__ShowLedControl
    std::shared_ptr<px4_msgs::msg::ShowLedControl_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ShowLedControl_ & other) const
  {
    if (this->timestamp != other.timestamp) {
      return false;
    }
    if (this->type != other.type) {
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
    if (this->brightness != other.brightness) {
      return false;
    }
    if (this->speed != other.speed) {
      return false;
    }
    return true;
  }
  bool operator!=(const ShowLedControl_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ShowLedControl_

// alias to use template instance with default allocator
using ShowLedControl =
  px4_msgs::msg::ShowLedControl_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__SHOW_LED_CONTROL__STRUCT_HPP_
