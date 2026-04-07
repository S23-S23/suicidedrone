// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from suicide_drone_msgs:msg/GuidanceCmd.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__FUNCTIONS_H_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "suicide_drone_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "suicide_drone_msgs/msg/detail/guidance_cmd__struct.h"

/// Initialize msg/GuidanceCmd message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * suicide_drone_msgs__msg__GuidanceCmd
 * )) before or use
 * suicide_drone_msgs__msg__GuidanceCmd__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
bool
suicide_drone_msgs__msg__GuidanceCmd__init(suicide_drone_msgs__msg__GuidanceCmd * msg);

/// Finalize msg/GuidanceCmd message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
void
suicide_drone_msgs__msg__GuidanceCmd__fini(suicide_drone_msgs__msg__GuidanceCmd * msg);

/// Create msg/GuidanceCmd message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * suicide_drone_msgs__msg__GuidanceCmd__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
suicide_drone_msgs__msg__GuidanceCmd *
suicide_drone_msgs__msg__GuidanceCmd__create();

/// Destroy msg/GuidanceCmd message.
/**
 * It calls
 * suicide_drone_msgs__msg__GuidanceCmd__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
void
suicide_drone_msgs__msg__GuidanceCmd__destroy(suicide_drone_msgs__msg__GuidanceCmd * msg);

/// Check for msg/GuidanceCmd message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
bool
suicide_drone_msgs__msg__GuidanceCmd__are_equal(const suicide_drone_msgs__msg__GuidanceCmd * lhs, const suicide_drone_msgs__msg__GuidanceCmd * rhs);

/// Copy a msg/GuidanceCmd message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
bool
suicide_drone_msgs__msg__GuidanceCmd__copy(
  const suicide_drone_msgs__msg__GuidanceCmd * input,
  suicide_drone_msgs__msg__GuidanceCmd * output);

/// Initialize array of msg/GuidanceCmd messages.
/**
 * It allocates the memory for the number of elements and calls
 * suicide_drone_msgs__msg__GuidanceCmd__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
bool
suicide_drone_msgs__msg__GuidanceCmd__Sequence__init(suicide_drone_msgs__msg__GuidanceCmd__Sequence * array, size_t size);

/// Finalize array of msg/GuidanceCmd messages.
/**
 * It calls
 * suicide_drone_msgs__msg__GuidanceCmd__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
void
suicide_drone_msgs__msg__GuidanceCmd__Sequence__fini(suicide_drone_msgs__msg__GuidanceCmd__Sequence * array);

/// Create array of msg/GuidanceCmd messages.
/**
 * It allocates the memory for the array and calls
 * suicide_drone_msgs__msg__GuidanceCmd__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
suicide_drone_msgs__msg__GuidanceCmd__Sequence *
suicide_drone_msgs__msg__GuidanceCmd__Sequence__create(size_t size);

/// Destroy array of msg/GuidanceCmd messages.
/**
 * It calls
 * suicide_drone_msgs__msg__GuidanceCmd__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
void
suicide_drone_msgs__msg__GuidanceCmd__Sequence__destroy(suicide_drone_msgs__msg__GuidanceCmd__Sequence * array);

/// Check for msg/GuidanceCmd message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
bool
suicide_drone_msgs__msg__GuidanceCmd__Sequence__are_equal(const suicide_drone_msgs__msg__GuidanceCmd__Sequence * lhs, const suicide_drone_msgs__msg__GuidanceCmd__Sequence * rhs);

/// Copy an array of msg/GuidanceCmd messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
bool
suicide_drone_msgs__msg__GuidanceCmd__Sequence__copy(
  const suicide_drone_msgs__msg__GuidanceCmd__Sequence * input,
  suicide_drone_msgs__msg__GuidanceCmd__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__GUIDANCE_CMD__FUNCTIONS_H_
