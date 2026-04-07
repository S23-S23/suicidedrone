// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from px4_msgs:srv/ModeChange.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__SRV__DETAIL__MODE_CHANGE__STRUCT_H_
#define PX4_MSGS__SRV__DETAIL__MODE_CHANGE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/ModeChange in the package px4_msgs.
typedef struct px4_msgs__srv__ModeChange_Request
{
  uint8_t suv_mode;
} px4_msgs__srv__ModeChange_Request;

// Struct for a sequence of px4_msgs__srv__ModeChange_Request.
typedef struct px4_msgs__srv__ModeChange_Request__Sequence
{
  px4_msgs__srv__ModeChange_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__srv__ModeChange_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/ModeChange in the package px4_msgs.
typedef struct px4_msgs__srv__ModeChange_Response
{
  bool reply;
} px4_msgs__srv__ModeChange_Response;

// Struct for a sequence of px4_msgs__srv__ModeChange_Response.
typedef struct px4_msgs__srv__ModeChange_Response__Sequence
{
  px4_msgs__srv__ModeChange_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__srv__ModeChange_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PX4_MSGS__SRV__DETAIL__MODE_CHANGE__STRUCT_H_
