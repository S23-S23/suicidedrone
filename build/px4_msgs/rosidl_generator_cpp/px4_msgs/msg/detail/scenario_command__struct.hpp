// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from px4_msgs:msg/ScenarioCommand.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__STRUCT_HPP_
#define PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__px4_msgs__msg__ScenarioCommand __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__msg__ScenarioCommand __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ScenarioCommand_
{
  using Type = ScenarioCommand_<ContainerAllocator>;

  explicit ScenarioCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->cmd = 0;
      this->param1 = 0.0f;
      this->param2 = 0.0f;
      this->param3 = 0.0f;
      this->param4 = 0ul;
      std::fill<typename std::array<uint8_t, 32>::iterator, uint8_t>(this->param5.begin(), this->param5.end(), 0);
    }
  }

  explicit ScenarioCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : param5(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->cmd = 0;
      this->param1 = 0.0f;
      this->param2 = 0.0f;
      this->param3 = 0.0f;
      this->param4 = 0ul;
      std::fill<typename std::array<uint8_t, 32>::iterator, uint8_t>(this->param5.begin(), this->param5.end(), 0);
    }
  }

  // field types and members
  using _timestamp_type =
    uint64_t;
  _timestamp_type timestamp;
  using _cmd_type =
    uint8_t;
  _cmd_type cmd;
  using _param1_type =
    float;
  _param1_type param1;
  using _param2_type =
    float;
  _param2_type param2;
  using _param3_type =
    float;
  _param3_type param3;
  using _param4_type =
    uint32_t;
  _param4_type param4;
  using _param5_type =
    std::array<uint8_t, 32>;
  _param5_type param5;

  // setters for named parameter idiom
  Type & set__timestamp(
    const uint64_t & _arg)
  {
    this->timestamp = _arg;
    return *this;
  }
  Type & set__cmd(
    const uint8_t & _arg)
  {
    this->cmd = _arg;
    return *this;
  }
  Type & set__param1(
    const float & _arg)
  {
    this->param1 = _arg;
    return *this;
  }
  Type & set__param2(
    const float & _arg)
  {
    this->param2 = _arg;
    return *this;
  }
  Type & set__param3(
    const float & _arg)
  {
    this->param3 = _arg;
    return *this;
  }
  Type & set__param4(
    const uint32_t & _arg)
  {
    this->param4 = _arg;
    return *this;
  }
  Type & set__param5(
    const std::array<uint8_t, 32> & _arg)
  {
    this->param5 = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t SCENARIO_CMD_SET_START_TIME =
    0u;
  static constexpr uint8_t SCENARIO_CMD_STOP_SCENARIO =
    1u;
  static constexpr uint8_t SCENARIO_CMD_EMERGENCY_LAND =
    2u;
  static constexpr uint8_t SCENARIO_CMD_SET_CONFIGS =
    3u;
  static constexpr uint8_t SCENARIO_CMD_RESET_CONFIGS =
    4u;

  // pointer types
  using RawPtr =
    px4_msgs::msg::ScenarioCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::msg::ScenarioCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::msg::ScenarioCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::msg::ScenarioCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::ScenarioCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::ScenarioCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::ScenarioCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::ScenarioCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::msg::ScenarioCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::msg::ScenarioCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__msg__ScenarioCommand
    std::shared_ptr<px4_msgs::msg::ScenarioCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__msg__ScenarioCommand
    std::shared_ptr<px4_msgs::msg::ScenarioCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ScenarioCommand_ & other) const
  {
    if (this->timestamp != other.timestamp) {
      return false;
    }
    if (this->cmd != other.cmd) {
      return false;
    }
    if (this->param1 != other.param1) {
      return false;
    }
    if (this->param2 != other.param2) {
      return false;
    }
    if (this->param3 != other.param3) {
      return false;
    }
    if (this->param4 != other.param4) {
      return false;
    }
    if (this->param5 != other.param5) {
      return false;
    }
    return true;
  }
  bool operator!=(const ScenarioCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ScenarioCommand_

// alias to use template instance with default allocator
using ScenarioCommand =
  px4_msgs::msg::ScenarioCommand_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t ScenarioCommand_<ContainerAllocator>::SCENARIO_CMD_SET_START_TIME;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t ScenarioCommand_<ContainerAllocator>::SCENARIO_CMD_STOP_SCENARIO;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t ScenarioCommand_<ContainerAllocator>::SCENARIO_CMD_EMERGENCY_LAND;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t ScenarioCommand_<ContainerAllocator>::SCENARIO_CMD_SET_CONFIGS;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t ScenarioCommand_<ContainerAllocator>::SCENARIO_CMD_RESET_CONFIGS;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__STRUCT_HPP_
