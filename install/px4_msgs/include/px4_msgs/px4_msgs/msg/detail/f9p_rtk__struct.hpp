// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from px4_msgs:msg/F9pRtk.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__F9P_RTK__STRUCT_HPP_
#define PX4_MSGS__MSG__DETAIL__F9P_RTK__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__px4_msgs__msg__F9pRtk __attribute__((deprecated))
#else
# define DEPRECATED__px4_msgs__msg__F9pRtk __declspec(deprecated)
#endif

namespace px4_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct F9pRtk_
{
  using Type = F9pRtk_<ContainerAllocator>;

  explicit F9pRtk_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->device_id = 0ul;
      this->tow = 0ul;
      this->age_corr = 0;
      this->fix_type = 0;
      this->satellites_used = 0;
      this->n = 0.0f;
      this->e = 0.0f;
      this->d = 0.0f;
      this->v_n = 0.0f;
      this->v_e = 0.0f;
      this->v_d = 0.0f;
      this->acc_n = 0.0f;
      this->acc_e = 0.0f;
      this->acc_d = 0.0f;
    }
  }

  explicit F9pRtk_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->device_id = 0ul;
      this->tow = 0ul;
      this->age_corr = 0;
      this->fix_type = 0;
      this->satellites_used = 0;
      this->n = 0.0f;
      this->e = 0.0f;
      this->d = 0.0f;
      this->v_n = 0.0f;
      this->v_e = 0.0f;
      this->v_d = 0.0f;
      this->acc_n = 0.0f;
      this->acc_e = 0.0f;
      this->acc_d = 0.0f;
    }
  }

  // field types and members
  using _timestamp_type =
    uint64_t;
  _timestamp_type timestamp;
  using _device_id_type =
    uint32_t;
  _device_id_type device_id;
  using _tow_type =
    uint32_t;
  _tow_type tow;
  using _age_corr_type =
    uint8_t;
  _age_corr_type age_corr;
  using _fix_type_type =
    uint8_t;
  _fix_type_type fix_type;
  using _satellites_used_type =
    uint8_t;
  _satellites_used_type satellites_used;
  using _n_type =
    float;
  _n_type n;
  using _e_type =
    float;
  _e_type e;
  using _d_type =
    float;
  _d_type d;
  using _v_n_type =
    float;
  _v_n_type v_n;
  using _v_e_type =
    float;
  _v_e_type v_e;
  using _v_d_type =
    float;
  _v_d_type v_d;
  using _acc_n_type =
    float;
  _acc_n_type acc_n;
  using _acc_e_type =
    float;
  _acc_e_type acc_e;
  using _acc_d_type =
    float;
  _acc_d_type acc_d;

  // setters for named parameter idiom
  Type & set__timestamp(
    const uint64_t & _arg)
  {
    this->timestamp = _arg;
    return *this;
  }
  Type & set__device_id(
    const uint32_t & _arg)
  {
    this->device_id = _arg;
    return *this;
  }
  Type & set__tow(
    const uint32_t & _arg)
  {
    this->tow = _arg;
    return *this;
  }
  Type & set__age_corr(
    const uint8_t & _arg)
  {
    this->age_corr = _arg;
    return *this;
  }
  Type & set__fix_type(
    const uint8_t & _arg)
  {
    this->fix_type = _arg;
    return *this;
  }
  Type & set__satellites_used(
    const uint8_t & _arg)
  {
    this->satellites_used = _arg;
    return *this;
  }
  Type & set__n(
    const float & _arg)
  {
    this->n = _arg;
    return *this;
  }
  Type & set__e(
    const float & _arg)
  {
    this->e = _arg;
    return *this;
  }
  Type & set__d(
    const float & _arg)
  {
    this->d = _arg;
    return *this;
  }
  Type & set__v_n(
    const float & _arg)
  {
    this->v_n = _arg;
    return *this;
  }
  Type & set__v_e(
    const float & _arg)
  {
    this->v_e = _arg;
    return *this;
  }
  Type & set__v_d(
    const float & _arg)
  {
    this->v_d = _arg;
    return *this;
  }
  Type & set__acc_n(
    const float & _arg)
  {
    this->acc_n = _arg;
    return *this;
  }
  Type & set__acc_e(
    const float & _arg)
  {
    this->acc_e = _arg;
    return *this;
  }
  Type & set__acc_d(
    const float & _arg)
  {
    this->acc_d = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t AGE_CORR_UNAVAILABLE =
    0u;
  static constexpr uint8_t AGE_CORR_0_TO_1_SEC =
    1u;
  static constexpr uint8_t AGE_CORR_1_TO_2_SEC =
    2u;
  static constexpr uint8_t AGE_CORR_2_TO_5_SEC =
    3u;
  static constexpr uint8_t AGE_CORR_5_TO_10_SEC =
    4u;
  static constexpr uint8_t AGE_CORR_10_TO_15_SEC =
    5u;
  static constexpr uint8_t AGE_CORR_15_TO_20_SEC =
    6u;
  static constexpr uint8_t AGE_CORR_20_TO_30_SEC =
    7u;
  static constexpr uint8_t AGE_CORR_30_TO_45_SEC =
    8u;
  static constexpr uint8_t AGE_CORR_45_TO_60_SEC =
    9u;

  // pointer types
  using RawPtr =
    px4_msgs::msg::F9pRtk_<ContainerAllocator> *;
  using ConstRawPtr =
    const px4_msgs::msg::F9pRtk_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<px4_msgs::msg::F9pRtk_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<px4_msgs::msg::F9pRtk_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::F9pRtk_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::F9pRtk_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      px4_msgs::msg::F9pRtk_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<px4_msgs::msg::F9pRtk_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<px4_msgs::msg::F9pRtk_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<px4_msgs::msg::F9pRtk_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__px4_msgs__msg__F9pRtk
    std::shared_ptr<px4_msgs::msg::F9pRtk_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__px4_msgs__msg__F9pRtk
    std::shared_ptr<px4_msgs::msg::F9pRtk_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const F9pRtk_ & other) const
  {
    if (this->timestamp != other.timestamp) {
      return false;
    }
    if (this->device_id != other.device_id) {
      return false;
    }
    if (this->tow != other.tow) {
      return false;
    }
    if (this->age_corr != other.age_corr) {
      return false;
    }
    if (this->fix_type != other.fix_type) {
      return false;
    }
    if (this->satellites_used != other.satellites_used) {
      return false;
    }
    if (this->n != other.n) {
      return false;
    }
    if (this->e != other.e) {
      return false;
    }
    if (this->d != other.d) {
      return false;
    }
    if (this->v_n != other.v_n) {
      return false;
    }
    if (this->v_e != other.v_e) {
      return false;
    }
    if (this->v_d != other.v_d) {
      return false;
    }
    if (this->acc_n != other.acc_n) {
      return false;
    }
    if (this->acc_e != other.acc_e) {
      return false;
    }
    if (this->acc_d != other.acc_d) {
      return false;
    }
    return true;
  }
  bool operator!=(const F9pRtk_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct F9pRtk_

// alias to use template instance with default allocator
using F9pRtk =
  px4_msgs::msg::F9pRtk_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t F9pRtk_<ContainerAllocator>::AGE_CORR_UNAVAILABLE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t F9pRtk_<ContainerAllocator>::AGE_CORR_0_TO_1_SEC;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t F9pRtk_<ContainerAllocator>::AGE_CORR_1_TO_2_SEC;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t F9pRtk_<ContainerAllocator>::AGE_CORR_2_TO_5_SEC;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t F9pRtk_<ContainerAllocator>::AGE_CORR_5_TO_10_SEC;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t F9pRtk_<ContainerAllocator>::AGE_CORR_10_TO_15_SEC;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t F9pRtk_<ContainerAllocator>::AGE_CORR_15_TO_20_SEC;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t F9pRtk_<ContainerAllocator>::AGE_CORR_20_TO_30_SEC;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t F9pRtk_<ContainerAllocator>::AGE_CORR_30_TO_45_SEC;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t F9pRtk_<ContainerAllocator>::AGE_CORR_45_TO_60_SEC;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace px4_msgs

#endif  // PX4_MSGS__MSG__DETAIL__F9P_RTK__STRUCT_HPP_
