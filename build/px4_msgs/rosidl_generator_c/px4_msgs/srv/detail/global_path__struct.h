// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from px4_msgs:srv/GlobalPath.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__STRUCT_H_
#define PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__STRUCT_H_

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

/// Struct defined in srv/GlobalPath in the package px4_msgs.
typedef struct px4_msgs__srv__GlobalPath_Request
{
  px4_msgs__msg__TrajectorySetpoint__Sequence waypoints;
} px4_msgs__srv__GlobalPath_Request;

// Struct for a sequence of px4_msgs__srv__GlobalPath_Request.
typedef struct px4_msgs__srv__GlobalPath_Request__Sequence
{
  px4_msgs__srv__GlobalPath_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__srv__GlobalPath_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/GlobalPath in the package px4_msgs.
typedef struct px4_msgs__srv__GlobalPath_Response
{
  bool reply;
} px4_msgs__srv__GlobalPath_Response;

// Struct for a sequence of px4_msgs__srv__GlobalPath_Response.
typedef struct px4_msgs__srv__GlobalPath_Response__Sequence
{
  px4_msgs__srv__GlobalPath_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__srv__GlobalPath_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PX4_MSGS__SRV__DETAIL__GLOBAL_PATH__STRUCT_H_
