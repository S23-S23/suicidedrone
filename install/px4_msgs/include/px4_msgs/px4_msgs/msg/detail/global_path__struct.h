// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from px4_msgs:msg/GlobalPath.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__GLOBAL_PATH__STRUCT_H_
#define PX4_MSGS__MSG__DETAIL__GLOBAL_PATH__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'waypoints'
#include "px4_msgs/msg/detail/trajectory_setpoint__struct.h"

/// Struct defined in msg/GlobalPath in the package px4_msgs.
typedef struct px4_msgs__msg__GlobalPath
{
  px4_msgs__msg__TrajectorySetpoint__Sequence waypoints;
} px4_msgs__msg__GlobalPath;

// Struct for a sequence of px4_msgs__msg__GlobalPath.
typedef struct px4_msgs__msg__GlobalPath__Sequence
{
  px4_msgs__msg__GlobalPath * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__msg__GlobalPath__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PX4_MSGS__MSG__DETAIL__GLOBAL_PATH__STRUCT_H_
