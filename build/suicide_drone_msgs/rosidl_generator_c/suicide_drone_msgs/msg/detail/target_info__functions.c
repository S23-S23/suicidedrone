// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from suicide_drone_msgs:msg/TargetInfo.idl
// generated code does not contain a copyright notice
#include "suicide_drone_msgs/msg/detail/target_info__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `class_name`
#include "rosidl_runtime_c/string_functions.h"

bool
suicide_drone_msgs__msg__TargetInfo__init(suicide_drone_msgs__msg__TargetInfo * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    suicide_drone_msgs__msg__TargetInfo__fini(msg);
    return false;
  }
  // class_name
  if (!rosidl_runtime_c__String__init(&msg->class_name)) {
    suicide_drone_msgs__msg__TargetInfo__fini(msg);
    return false;
  }
  // top
  // left
  // bottom
  // right
  return true;
}

void
suicide_drone_msgs__msg__TargetInfo__fini(suicide_drone_msgs__msg__TargetInfo * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // class_name
  rosidl_runtime_c__String__fini(&msg->class_name);
  // top
  // left
  // bottom
  // right
}

bool
suicide_drone_msgs__msg__TargetInfo__are_equal(const suicide_drone_msgs__msg__TargetInfo * lhs, const suicide_drone_msgs__msg__TargetInfo * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // class_name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->class_name), &(rhs->class_name)))
  {
    return false;
  }
  // top
  if (lhs->top != rhs->top) {
    return false;
  }
  // left
  if (lhs->left != rhs->left) {
    return false;
  }
  // bottom
  if (lhs->bottom != rhs->bottom) {
    return false;
  }
  // right
  if (lhs->right != rhs->right) {
    return false;
  }
  return true;
}

bool
suicide_drone_msgs__msg__TargetInfo__copy(
  const suicide_drone_msgs__msg__TargetInfo * input,
  suicide_drone_msgs__msg__TargetInfo * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // class_name
  if (!rosidl_runtime_c__String__copy(
      &(input->class_name), &(output->class_name)))
  {
    return false;
  }
  // top
  output->top = input->top;
  // left
  output->left = input->left;
  // bottom
  output->bottom = input->bottom;
  // right
  output->right = input->right;
  return true;
}

suicide_drone_msgs__msg__TargetInfo *
suicide_drone_msgs__msg__TargetInfo__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  suicide_drone_msgs__msg__TargetInfo * msg = (suicide_drone_msgs__msg__TargetInfo *)allocator.allocate(sizeof(suicide_drone_msgs__msg__TargetInfo), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(suicide_drone_msgs__msg__TargetInfo));
  bool success = suicide_drone_msgs__msg__TargetInfo__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
suicide_drone_msgs__msg__TargetInfo__destroy(suicide_drone_msgs__msg__TargetInfo * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    suicide_drone_msgs__msg__TargetInfo__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
suicide_drone_msgs__msg__TargetInfo__Sequence__init(suicide_drone_msgs__msg__TargetInfo__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  suicide_drone_msgs__msg__TargetInfo * data = NULL;

  if (size) {
    data = (suicide_drone_msgs__msg__TargetInfo *)allocator.zero_allocate(size, sizeof(suicide_drone_msgs__msg__TargetInfo), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = suicide_drone_msgs__msg__TargetInfo__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        suicide_drone_msgs__msg__TargetInfo__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
suicide_drone_msgs__msg__TargetInfo__Sequence__fini(suicide_drone_msgs__msg__TargetInfo__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      suicide_drone_msgs__msg__TargetInfo__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

suicide_drone_msgs__msg__TargetInfo__Sequence *
suicide_drone_msgs__msg__TargetInfo__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  suicide_drone_msgs__msg__TargetInfo__Sequence * array = (suicide_drone_msgs__msg__TargetInfo__Sequence *)allocator.allocate(sizeof(suicide_drone_msgs__msg__TargetInfo__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = suicide_drone_msgs__msg__TargetInfo__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
suicide_drone_msgs__msg__TargetInfo__Sequence__destroy(suicide_drone_msgs__msg__TargetInfo__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    suicide_drone_msgs__msg__TargetInfo__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
suicide_drone_msgs__msg__TargetInfo__Sequence__are_equal(const suicide_drone_msgs__msg__TargetInfo__Sequence * lhs, const suicide_drone_msgs__msg__TargetInfo__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!suicide_drone_msgs__msg__TargetInfo__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
suicide_drone_msgs__msg__TargetInfo__Sequence__copy(
  const suicide_drone_msgs__msg__TargetInfo__Sequence * input,
  suicide_drone_msgs__msg__TargetInfo__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(suicide_drone_msgs__msg__TargetInfo);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    suicide_drone_msgs__msg__TargetInfo * data =
      (suicide_drone_msgs__msg__TargetInfo *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!suicide_drone_msgs__msg__TargetInfo__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          suicide_drone_msgs__msg__TargetInfo__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!suicide_drone_msgs__msg__TargetInfo__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
