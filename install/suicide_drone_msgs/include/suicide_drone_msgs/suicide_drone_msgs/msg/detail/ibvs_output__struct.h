// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from suicide_drone_msgs:msg/IBVSOutput.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__STRUCT_H_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__STRUCT_H_

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

/// Struct defined in msg/IBVSOutput in the package suicide_drone_msgs.
typedef struct suicide_drone_msgs__msg__IBVSOutput
{
  std_msgs__msg__Header header;
  bool detected;
  /// LOS elevation angle (valid when detected=true)
  double q_y;
  /// LOS azimuth angle (valid when detected=true)
  double q_z;
  /// FOV yaw rate command Eq.(13)
  double fov_yaw_rate;
  /// FOV Z velocity correction (ey-based)
  double fov_vel_z;
} suicide_drone_msgs__msg__IBVSOutput;

// Struct for a sequence of suicide_drone_msgs__msg__IBVSOutput.
typedef struct suicide_drone_msgs__msg__IBVSOutput__Sequence
{
  suicide_drone_msgs__msg__IBVSOutput * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} suicide_drone_msgs__msg__IBVSOutput__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__STRUCT_H_
