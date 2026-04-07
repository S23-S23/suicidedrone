// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from suicide_drone_msgs:msg/TargetInfo.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__TARGET_INFO__STRUCT_H_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__TARGET_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'class_name'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/TargetInfo in the package suicide_drone_msgs.
typedef struct suicide_drone_msgs__msg__TargetInfo
{
  std_msgs__msg__Header header;
  rosidl_runtime_c__String class_name;
  int64_t top;
  int64_t left;
  int64_t bottom;
  int64_t right;
} suicide_drone_msgs__msg__TargetInfo;

// Struct for a sequence of suicide_drone_msgs__msg__TargetInfo.
typedef struct suicide_drone_msgs__msg__TargetInfo__Sequence
{
  suicide_drone_msgs__msg__TargetInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} suicide_drone_msgs__msg__TargetInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__TARGET_INFO__STRUCT_H_
