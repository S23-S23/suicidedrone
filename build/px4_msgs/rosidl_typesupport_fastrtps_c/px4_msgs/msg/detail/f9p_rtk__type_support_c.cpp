// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from px4_msgs:msg/F9pRtk.idl
// generated code does not contain a copyright notice
#include "px4_msgs/msg/detail/f9p_rtk__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "px4_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "px4_msgs/msg/detail/f9p_rtk__struct.h"
#include "px4_msgs/msg/detail/f9p_rtk__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _F9pRtk__ros_msg_type = px4_msgs__msg__F9pRtk;

static bool _F9pRtk__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _F9pRtk__ros_msg_type * ros_message = static_cast<const _F9pRtk__ros_msg_type *>(untyped_ros_message);
  // Field name: timestamp
  {
    cdr << ros_message->timestamp;
  }

  // Field name: device_id
  {
    cdr << ros_message->device_id;
  }

  // Field name: tow
  {
    cdr << ros_message->tow;
  }

  // Field name: age_corr
  {
    cdr << ros_message->age_corr;
  }

  // Field name: fix_type
  {
    cdr << ros_message->fix_type;
  }

  // Field name: satellites_used
  {
    cdr << ros_message->satellites_used;
  }

  // Field name: n
  {
    cdr << ros_message->n;
  }

  // Field name: e
  {
    cdr << ros_message->e;
  }

  // Field name: d
  {
    cdr << ros_message->d;
  }

  // Field name: v_n
  {
    cdr << ros_message->v_n;
  }

  // Field name: v_e
  {
    cdr << ros_message->v_e;
  }

  // Field name: v_d
  {
    cdr << ros_message->v_d;
  }

  // Field name: acc_n
  {
    cdr << ros_message->acc_n;
  }

  // Field name: acc_e
  {
    cdr << ros_message->acc_e;
  }

  // Field name: acc_d
  {
    cdr << ros_message->acc_d;
  }

  return true;
}

static bool _F9pRtk__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _F9pRtk__ros_msg_type * ros_message = static_cast<_F9pRtk__ros_msg_type *>(untyped_ros_message);
  // Field name: timestamp
  {
    cdr >> ros_message->timestamp;
  }

  // Field name: device_id
  {
    cdr >> ros_message->device_id;
  }

  // Field name: tow
  {
    cdr >> ros_message->tow;
  }

  // Field name: age_corr
  {
    cdr >> ros_message->age_corr;
  }

  // Field name: fix_type
  {
    cdr >> ros_message->fix_type;
  }

  // Field name: satellites_used
  {
    cdr >> ros_message->satellites_used;
  }

  // Field name: n
  {
    cdr >> ros_message->n;
  }

  // Field name: e
  {
    cdr >> ros_message->e;
  }

  // Field name: d
  {
    cdr >> ros_message->d;
  }

  // Field name: v_n
  {
    cdr >> ros_message->v_n;
  }

  // Field name: v_e
  {
    cdr >> ros_message->v_e;
  }

  // Field name: v_d
  {
    cdr >> ros_message->v_d;
  }

  // Field name: acc_n
  {
    cdr >> ros_message->acc_n;
  }

  // Field name: acc_e
  {
    cdr >> ros_message->acc_e;
  }

  // Field name: acc_d
  {
    cdr >> ros_message->acc_d;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_px4_msgs
size_t get_serialized_size_px4_msgs__msg__F9pRtk(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _F9pRtk__ros_msg_type * ros_message = static_cast<const _F9pRtk__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name timestamp
  {
    size_t item_size = sizeof(ros_message->timestamp);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name device_id
  {
    size_t item_size = sizeof(ros_message->device_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tow
  {
    size_t item_size = sizeof(ros_message->tow);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name age_corr
  {
    size_t item_size = sizeof(ros_message->age_corr);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name fix_type
  {
    size_t item_size = sizeof(ros_message->fix_type);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name satellites_used
  {
    size_t item_size = sizeof(ros_message->satellites_used);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name n
  {
    size_t item_size = sizeof(ros_message->n);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name e
  {
    size_t item_size = sizeof(ros_message->e);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name d
  {
    size_t item_size = sizeof(ros_message->d);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name v_n
  {
    size_t item_size = sizeof(ros_message->v_n);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name v_e
  {
    size_t item_size = sizeof(ros_message->v_e);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name v_d
  {
    size_t item_size = sizeof(ros_message->v_d);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name acc_n
  {
    size_t item_size = sizeof(ros_message->acc_n);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name acc_e
  {
    size_t item_size = sizeof(ros_message->acc_e);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name acc_d
  {
    size_t item_size = sizeof(ros_message->acc_d);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _F9pRtk__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_px4_msgs__msg__F9pRtk(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_px4_msgs
size_t max_serialized_size_px4_msgs__msg__F9pRtk(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: timestamp
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: device_id
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: tow
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: age_corr
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: fix_type
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: satellites_used
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: n
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: e
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: d
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: v_n
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: v_e
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: v_d
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: acc_n
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: acc_e
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: acc_d
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = px4_msgs__msg__F9pRtk;
    is_plain =
      (
      offsetof(DataType, acc_d) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _F9pRtk__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_px4_msgs__msg__F9pRtk(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_F9pRtk = {
  "px4_msgs::msg",
  "F9pRtk",
  _F9pRtk__cdr_serialize,
  _F9pRtk__cdr_deserialize,
  _F9pRtk__get_serialized_size,
  _F9pRtk__max_serialized_size
};

static rosidl_message_type_support_t _F9pRtk__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_F9pRtk,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, px4_msgs, msg, F9pRtk)() {
  return &_F9pRtk__type_support;
}

#if defined(__cplusplus)
}
#endif
