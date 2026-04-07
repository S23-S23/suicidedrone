// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from px4_msgs:msg/GlobalPath.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "px4_msgs/msg/detail/global_path__rosidl_typesupport_introspection_c.h"
#include "px4_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "px4_msgs/msg/detail/global_path__functions.h"
#include "px4_msgs/msg/detail/global_path__struct.h"


// Include directives for member types
// Member `waypoints`
#include "px4_msgs/msg/trajectory_setpoint.h"
// Member `waypoints`
#include "px4_msgs/msg/detail/trajectory_setpoint__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  px4_msgs__msg__GlobalPath__init(message_memory);
}

void px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_fini_function(void * message_memory)
{
  px4_msgs__msg__GlobalPath__fini(message_memory);
}

size_t px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__size_function__GlobalPath__waypoints(
  const void * untyped_member)
{
  const px4_msgs__msg__TrajectorySetpoint__Sequence * member =
    (const px4_msgs__msg__TrajectorySetpoint__Sequence *)(untyped_member);
  return member->size;
}

const void * px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__get_const_function__GlobalPath__waypoints(
  const void * untyped_member, size_t index)
{
  const px4_msgs__msg__TrajectorySetpoint__Sequence * member =
    (const px4_msgs__msg__TrajectorySetpoint__Sequence *)(untyped_member);
  return &member->data[index];
}

void * px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__get_function__GlobalPath__waypoints(
  void * untyped_member, size_t index)
{
  px4_msgs__msg__TrajectorySetpoint__Sequence * member =
    (px4_msgs__msg__TrajectorySetpoint__Sequence *)(untyped_member);
  return &member->data[index];
}

void px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__fetch_function__GlobalPath__waypoints(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const px4_msgs__msg__TrajectorySetpoint * item =
    ((const px4_msgs__msg__TrajectorySetpoint *)
    px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__get_const_function__GlobalPath__waypoints(untyped_member, index));
  px4_msgs__msg__TrajectorySetpoint * value =
    (px4_msgs__msg__TrajectorySetpoint *)(untyped_value);
  *value = *item;
}

void px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__assign_function__GlobalPath__waypoints(
  void * untyped_member, size_t index, const void * untyped_value)
{
  px4_msgs__msg__TrajectorySetpoint * item =
    ((px4_msgs__msg__TrajectorySetpoint *)
    px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__get_function__GlobalPath__waypoints(untyped_member, index));
  const px4_msgs__msg__TrajectorySetpoint * value =
    (const px4_msgs__msg__TrajectorySetpoint *)(untyped_value);
  *item = *value;
}

bool px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__resize_function__GlobalPath__waypoints(
  void * untyped_member, size_t size)
{
  px4_msgs__msg__TrajectorySetpoint__Sequence * member =
    (px4_msgs__msg__TrajectorySetpoint__Sequence *)(untyped_member);
  px4_msgs__msg__TrajectorySetpoint__Sequence__fini(member);
  return px4_msgs__msg__TrajectorySetpoint__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_message_member_array[1] = {
  {
    "waypoints",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(px4_msgs__msg__GlobalPath, waypoints),  // bytes offset in struct
    NULL,  // default value
    px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__size_function__GlobalPath__waypoints,  // size() function pointer
    px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__get_const_function__GlobalPath__waypoints,  // get_const(index) function pointer
    px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__get_function__GlobalPath__waypoints,  // get(index) function pointer
    px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__fetch_function__GlobalPath__waypoints,  // fetch(index, &value) function pointer
    px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__assign_function__GlobalPath__waypoints,  // assign(index, value) function pointer
    px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__resize_function__GlobalPath__waypoints  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_message_members = {
  "px4_msgs__msg",  // message namespace
  "GlobalPath",  // message name
  1,  // number of fields
  sizeof(px4_msgs__msg__GlobalPath),
  px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_message_member_array,  // message members
  px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_init_function,  // function to initialize message memory (memory has to be allocated)
  px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_message_type_support_handle = {
  0,
  &px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_px4_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, px4_msgs, msg, GlobalPath)() {
  px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, px4_msgs, msg, TrajectorySetpoint)();
  if (!px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_message_type_support_handle.typesupport_identifier) {
    px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &px4_msgs__msg__GlobalPath__rosidl_typesupport_introspection_c__GlobalPath_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
