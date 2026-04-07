// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from px4_msgs:msg/SuvMonitoring.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SUV_MONITORING__STRUCT_H_
#define PX4_MSGS__MSG__DETAIL__SUV_MONITORING__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'monitoring'
#include "px4_msgs/msg/detail/monitoring__struct.h"

/// Struct defined in msg/SuvMonitoring in the package px4_msgs.
typedef struct px4_msgs__msg__SuvMonitoring
{
  px4_msgs__msg__Monitoring monitoring;
  uint8_t mode;
} px4_msgs__msg__SuvMonitoring;

// Struct for a sequence of px4_msgs__msg__SuvMonitoring.
typedef struct px4_msgs__msg__SuvMonitoring__Sequence
{
  px4_msgs__msg__SuvMonitoring * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__msg__SuvMonitoring__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PX4_MSGS__MSG__DETAIL__SUV_MONITORING__STRUCT_H_
