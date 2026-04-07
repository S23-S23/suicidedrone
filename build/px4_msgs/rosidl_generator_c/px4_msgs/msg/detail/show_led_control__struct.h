// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from px4_msgs:msg/ShowLedControl.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SHOW_LED_CONTROL__STRUCT_H_
#define PX4_MSGS__MSG__DETAIL__SHOW_LED_CONTROL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/ShowLedControl in the package px4_msgs.
typedef struct px4_msgs__msg__ShowLedControl
{
  /// time since system start (microseconds)
  uint64_t timestamp;
  /// the type of led shape or some animation
  int8_t type;
  /// the color of LED (RED)
  int8_t r;
  /// the color of LED (GREEN)
  int8_t g;
  /// the color of LED (BLUE)
  int8_t b;
  /// the brightness of LED
  int8_t brightness;
  /// the speed of LED blink
  int8_t speed;
} px4_msgs__msg__ShowLedControl;

// Struct for a sequence of px4_msgs__msg__ShowLedControl.
typedef struct px4_msgs__msg__ShowLedControl__Sequence
{
  px4_msgs__msg__ShowLedControl * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__msg__ShowLedControl__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PX4_MSGS__MSG__DETAIL__SHOW_LED_CONTROL__STRUCT_H_
