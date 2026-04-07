// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from px4_msgs:msg/ScenarioCommand.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__STRUCT_H_
#define PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'SCENARIO_CMD_SET_START_TIME'.
enum
{
  px4_msgs__msg__ScenarioCommand__SCENARIO_CMD_SET_START_TIME = 0
};

/// Constant 'SCENARIO_CMD_STOP_SCENARIO'.
enum
{
  px4_msgs__msg__ScenarioCommand__SCENARIO_CMD_STOP_SCENARIO = 1
};

/// Constant 'SCENARIO_CMD_EMERGENCY_LAND'.
enum
{
  px4_msgs__msg__ScenarioCommand__SCENARIO_CMD_EMERGENCY_LAND = 2
};

/// Constant 'SCENARIO_CMD_SET_CONFIGS'.
enum
{
  px4_msgs__msg__ScenarioCommand__SCENARIO_CMD_SET_CONFIGS = 3
};

/// Constant 'SCENARIO_CMD_RESET_CONFIGS'.
enum
{
  px4_msgs__msg__ScenarioCommand__SCENARIO_CMD_RESET_CONFIGS = 4
};

/// Struct defined in msg/ScenarioCommand in the package px4_msgs.
typedef struct px4_msgs__msg__ScenarioCommand
{
  /// time since system start (microseconds)
  uint64_t timestamp;
  /// embedded scenario command
  uint8_t cmd;
  /// param1
  float param1;
  /// param2
  float param2;
  /// param3
  float param3;
  /// param4
  uint32_t param4;
  /// param5
  uint8_t param5[32];
} px4_msgs__msg__ScenarioCommand;

// Struct for a sequence of px4_msgs__msg__ScenarioCommand.
typedef struct px4_msgs__msg__ScenarioCommand__Sequence
{
  px4_msgs__msg__ScenarioCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__msg__ScenarioCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PX4_MSGS__MSG__DETAIL__SCENARIO_COMMAND__STRUCT_H_
