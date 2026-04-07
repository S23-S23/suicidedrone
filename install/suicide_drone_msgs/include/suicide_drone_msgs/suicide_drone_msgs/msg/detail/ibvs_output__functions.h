// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from suicide_drone_msgs:msg/IBVSOutput.idl
// generated code does not contain a copyright notice

#ifndef SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__FUNCTIONS_H_
#define SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "suicide_drone_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "suicide_drone_msgs/msg/detail/ibvs_output__struct.h"

/// Initialize msg/IBVSOutput message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * suicide_drone_msgs__msg__IBVSOutput
 * )) before or use
 * suicide_drone_msgs__msg__IBVSOutput__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
bool
suicide_drone_msgs__msg__IBVSOutput__init(suicide_drone_msgs__msg__IBVSOutput * msg);

/// Finalize msg/IBVSOutput message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
void
suicide_drone_msgs__msg__IBVSOutput__fini(suicide_drone_msgs__msg__IBVSOutput * msg);

/// Create msg/IBVSOutput message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * suicide_drone_msgs__msg__IBVSOutput__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
suicide_drone_msgs__msg__IBVSOutput *
suicide_drone_msgs__msg__IBVSOutput__create();

/// Destroy msg/IBVSOutput message.
/**
 * It calls
 * suicide_drone_msgs__msg__IBVSOutput__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
void
suicide_drone_msgs__msg__IBVSOutput__destroy(suicide_drone_msgs__msg__IBVSOutput * msg);

/// Check for msg/IBVSOutput message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
bool
suicide_drone_msgs__msg__IBVSOutput__are_equal(const suicide_drone_msgs__msg__IBVSOutput * lhs, const suicide_drone_msgs__msg__IBVSOutput * rhs);

/// Copy a msg/IBVSOutput message.
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
suicide_drone_msgs__msg__IBVSOutput__copy(
  const suicide_drone_msgs__msg__IBVSOutput * input,
  suicide_drone_msgs__msg__IBVSOutput * output);

/// Initialize array of msg/IBVSOutput messages.
/**
 * It allocates the memory for the number of elements and calls
 * suicide_drone_msgs__msg__IBVSOutput__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
bool
suicide_drone_msgs__msg__IBVSOutput__Sequence__init(suicide_drone_msgs__msg__IBVSOutput__Sequence * array, size_t size);

/// Finalize array of msg/IBVSOutput messages.
/**
 * It calls
 * suicide_drone_msgs__msg__IBVSOutput__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
void
suicide_drone_msgs__msg__IBVSOutput__Sequence__fini(suicide_drone_msgs__msg__IBVSOutput__Sequence * array);

/// Create array of msg/IBVSOutput messages.
/**
 * It allocates the memory for the array and calls
 * suicide_drone_msgs__msg__IBVSOutput__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
suicide_drone_msgs__msg__IBVSOutput__Sequence *
suicide_drone_msgs__msg__IBVSOutput__Sequence__create(size_t size);

/// Destroy array of msg/IBVSOutput messages.
/**
 * It calls
 * suicide_drone_msgs__msg__IBVSOutput__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
void
suicide_drone_msgs__msg__IBVSOutput__Sequence__destroy(suicide_drone_msgs__msg__IBVSOutput__Sequence * array);

/// Check for msg/IBVSOutput message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_suicide_drone_msgs
bool
suicide_drone_msgs__msg__IBVSOutput__Sequence__are_equal(const suicide_drone_msgs__msg__IBVSOutput__Sequence * lhs, const suicide_drone_msgs__msg__IBVSOutput__Sequence * rhs);

/// Copy an array of msg/IBVSOutput messages.
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
suicide_drone_msgs__msg__IBVSOutput__Sequence__copy(
  const suicide_drone_msgs__msg__IBVSOutput__Sequence * input,
  suicide_drone_msgs__msg__IBVSOutput__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // SUICIDE_DRONE_MSGS__MSG__DETAIL__IBVS_OUTPUT__FUNCTIONS_H_
