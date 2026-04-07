// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from suicide_drone_msgs:msg/GuidanceCmd.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__STRUCT_H_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__STRUCT_H_

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

/// Struct defined in msg/GuidanceCmd in the package suicide_drone_msgs.
typedef struct suicide_drone_msgs__msg__GuidanceCmd
{
  std_msgs__msg__Header header;
  bool target_detected;
  /// NED North velocity
  double vel_n;
  /// NED East  velocity
  double vel_e;
  /// NED Down  velocity
  double vel_d;
  /// yaw rate command
  double yaw_rate;
} suicide_drone_msgs__msg__GuidanceCmd;

// Struct for a sequence of suicide_drone_msgs__msg__GuidanceCmd.
typedef struct suicide_drone_msgs__msg__GuidanceCmd__Sequence
{
  suicide_drone_msgs__msg__GuidanceCmd * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} suicide_drone_msgs__msg__GuidanceCmd__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__STRUCT_H_
