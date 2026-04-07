// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from suicide_drone_msgs:msg/TargetInfo.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "suicide_drone_msgs/msg/detail/target_info__rosidl_typesupport_introspection_c.h"
#include "suicide_drone_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "suicide_drone_msgs/msg/detail/target_info__functions.h"
#include "suicide_drone_msgs/msg/detail/target_info__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `class_name`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  suicide_drone_msgs__msg__TargetInfo__init(message_memory);
}

void suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_fini_function(void * message_memory)
{
  suicide_drone_msgs__msg__TargetInfo__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_message_member_array[6] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__TargetInfo, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "class_name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__TargetInfo, class_name),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "top",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__TargetInfo, top),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "left",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__TargetInfo, left),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "bottom",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__TargetInfo, bottom),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "right",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(suicide_drone_msgs__msg__TargetInfo, right),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_message_members = {
  "suicide_drone_msgs__msg",  // message namespace
  "TargetInfo",  // message name
  6,  // number of fields
  sizeof(suicide_drone_msgs__msg__TargetInfo),
  suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_message_member_array,  // message members
  suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_init_function,  // function to initialize message memory (memory has to be allocated)
  suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_message_type_support_handle = {
  0,
  &suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_suicide_drone_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, suicide_drone_msgs, msg, TargetInfo)() {
  suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_message_type_support_handle.typesupport_identifier) {
    suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &suicide_drone_msgs__msg__TargetInfo__rosidl_typesupport_introspection_c__TargetInfo_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
