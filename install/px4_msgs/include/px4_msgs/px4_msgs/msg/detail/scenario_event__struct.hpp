// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from px4_msgs:msg/ScenarioEvent.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SCENARIO_EVENT__STRUCT_HPP_
#define PX4_MSGS__MSG__DETAIL__SCENARIO_EVENT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__px4_msgs__msg__ScenarioEvent __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__msg__ScenarioEvent __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ScenarioEvent_
{
  using Type = ScenarioEvent_<ContainerAllocator>;

  explicit ScenarioEvent_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->event_time = 0.0f;
      this->event_type = 0;
      this->cmd_type = 0;
      this->x = 0.0f;
      this->y = 0.0f;
      this->z = 0.0f;
      this->led_r = 0;
      this->led_g = 0;
      this->led_b = 0;
      this->is_scenario_active = false;
    }
  }

  explicit ScenarioEvent_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->event_time = 0.0f;
      this->event_type = 0;
      this->cmd_type = 0;
      this->x = 0.0f;
      this->y = 0.0f;
      this->z = 0.0f;
      this->led_r = 0;
      this->led_g = 0;
      this->led_b = 0;
      this->is_scenario_active = false;
    }
  }

  // field types and members
  using _timestamp_type =
    uint64_t;
  _timestamp_type timestamp;
  using _event_time_type =
    float;
  _event_time_type event_time;
  using _event_type_type =
    uint8_t;
  _event_type_type event_type;
  using _cmd_type_type =
    uint8_t;
  _cmd_type_type cmd_type;
  using _x_type =
    float;
  _x_type x;
  using _y_type =
    float;
  _y_type y;
  using _z_type =
    float;
  _z_type z;
  using _led_r_type =
    uint8_t;
  _led_r_type led_r;
  using _led_g_type =
    uint8_t;
  _led_g_type led_g;
  using _led_b_type =
    uint8_t;
  _led_b_type led_b;
  using _is_scenario_active_type =
    bool;
  _is_scenario_active_type is_scenario_active;

  // setters for named parameter idiom
  Type & set__timestamp(
    const uint64_t & _arg)
  {
    this->timestamp = _arg;
    return *this;
  }
  Type & set__event_time(
    const float & _arg)
  {
    this->event_time = _arg;
    return *this;
  }
  Type & set__event_type(
    const uint8_t & _arg)
  {
    this->event_type = _arg;
    return *this;
  }
  Type & set__cmd_type(
    const uint8_t & _arg)
  {
    this->cmd_type = _arg;
    return *this;
  }
  Type & set__x(
    const float & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const float & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__z(
    const float & _arg)
  {
    this->z = _arg;
    return *this;
  }
  Type & set__led_r(
    const uint8_t & _arg)
  {
    this->led_r = _arg;
    return *this;
  }
  Type & set__led_g(
    const uint8_t & _arg)
  {
    this->led_g = _arg;
    return *this;
  }
  Type & set__led_b(
    const uint8_t & _arg)
  {
    this->led_b = _arg;
    return *this;
  }
  Type & set__is_scenario_active(
    const bool & _arg)
  {
    this->is_scenario_active = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    px4_msgs::msg::ScenarioEvent_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::msg::ScenarioEvent_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::msg::ScenarioEvent_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::msg::ScenarioEvent_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::ScenarioEvent_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::ScenarioEvent_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::ScenarioEvent_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::ScenarioEvent_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::msg::ScenarioEvent_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::msg::ScenarioEvent_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__msg__ScenarioEvent
    std::shared_ptr<px4_msgs::msg::ScenarioEvent_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__msg__ScenarioEvent
    std::shared_ptr<px4_msgs::msg::ScenarioEvent_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ScenarioEvent_ & other) const
  {
    if (this->timestamp != other.timestamp) {
      return false;
    }
    if (this->event_time != other.event_time) {
      return false;
    }
    if (this->event_type != other.event_type) {
      return false;
    }
    if (this->cmd_type != other.cmd_type) {
      return false;
    }
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->z != other.z) {
      return false;
    }
    if (this->led_r != other.led_r) {
      return false;
    }
    if (this->led_g != other.led_g) {
      return false;
    }
    if (this->led_b != other.led_b) {
      return false;
    }
    if (this->is_scenario_active != other.is_scenario_active) {
      return false;
    }
    return true;
  }
  bool operator!=(const ScenarioEvent_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ScenarioEvent_

// alias to use template instance with default allocator
using ScenarioEvent =
  px4_msgs::msg::ScenarioEvent_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__SCENARIO_EVENT__STRUCT_HPP_
