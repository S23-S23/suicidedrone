// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from px4_msgs:msg/OffboardScenario.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__STRUCT_H_
#define PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/OffboardScenario in the package px4_msgs.
typedef struct px4_msgs__msg__OffboardScenario
{
  /// time since system start (microseconds)
  uint64_t timestamp;
  /// current time  (ms)
  uint64_t current_time;
  /// scenario start time (ms)
  uint64_t start_time;
  /// scenario sequence
  uint32_t seq;
  /// position offset_x (m)
  float offset_x;
  /// position offset_y (m)
  float offset_y;
  /// 1: scenario file ready, 2: scenario file length err, 3: scenario file crc err, 4: scenario payload header err
  uint8_t ready_sc_file;
} px4_msgs__msg__OffboardScenario;

// Struct for a sequence of px4_msgs__msg__OffboardScenario.
typedef struct px4_msgs__msg__OffboardScenario__Sequence
{
  px4_msgs__msg__OffboardScenario * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__msg__OffboardScenario__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PX4_MSGS__MSG__DETAIL__OFFBOARD_SCENARIO__STRUCT_H_
