// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from suicide_drone_msgs:msg/GuidanceCmd.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "suicide_drone_msgs/msg/detail/guidance_cmd__rosidl_typesupport_introspection_c.h"
#include "suicide_drone_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "suicide_drone_msgs/msg/detail/guidance_cmd__functions.h"
#include "suicide_drone_msgs/msg/detail/guidance_cmd__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  suicide_drone_msgs__msg__GuidanceCmd__init(message_memory);
}

void suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_fini_function(void * message_memory)
{
  suicide_drone_msgs__msg__GuidanceCmd__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_message_member_array[6] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__GuidanceCmd, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_detected",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__GuidanceCmd, target_detected),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vel_n",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__GuidanceCmd, vel_n),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vel_e",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__GuidanceCmd, vel_e),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vel_d",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__GuidanceCmd, vel_d),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "yaw_rate",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__GuidanceCmd, yaw_rate),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_message_members = {
  "suicide_drone_msgs__msg",  // message namespace
  "GuidanceCmd",  // message name
  6,  // number of fields
  sizeof(suicide_drone_msgs__msg__GuidanceCmd),
  suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_message_member_array,  // message members
  suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_init_function,  // function to initialize message memory (memory has to be allocated)
  suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_message_type_support_handle = {
  0,
  &suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_suicide_drone_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, suicide_drone_msgs, msg, GuidanceCmd)() {
  suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_message_type_support_handle.typesupport_identifier) {
    suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &suicide_drone_msgs__msg__GuidanceCmd__rosidl_typesupport_introspection_c__GuidanceCmd_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
