// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from suicide_drone_msgs:msg/GuidanceCmd.idl
// generated code does not contain a copyright notice
#include "suicide_drone_msgs/msg/detail/guidance_cmd__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
suicide_drone_msgs__msg__GuidanceCmd__init(suicide_drone_msgs__msg__GuidanceCmd * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    suicide_drone_msgs__msg__GuidanceCmd__fini(msg);
    return false;
  }
  // target_detected
  // vel_n
  // vel_e
  // vel_d
  // yaw_rate
  return true;
}

void
suicide_drone_msgs__msg__GuidanceCmd__fini(suicide_drone_msgs__msg__GuidanceCmd * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // target_detected
  // vel_n
  // vel_e
  // vel_d
  // yaw_rate
}

bool
suicide_drone_msgs__msg__GuidanceCmd__are_equal(const suicide_drone_msgs__msg__GuidanceCmd * lhs, const suicide_drone_msgs__msg__GuidanceCmd * rhs)
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
  // target_detected
  if (lhs->target_detected != rhs->target_detected) {
    return false;
  }
  // vel_n
  if (lhs->vel_n != rhs->vel_n) {
    return false;
  }
  // vel_e
  if (lhs->vel_e != rhs->vel_e) {
    return false;
  }
  // vel_d
  if (lhs->vel_d != rhs->vel_d) {
    return false;
  }
  // yaw_rate
  if (lhs->yaw_rate != rhs->yaw_rate) {
    return false;
  }
  return true;
}

bool
suicide_drone_msgs__msg__GuidanceCmd__copy(
  const suicide_drone_msgs__msg__GuidanceCmd * input,
  suicide_drone_msgs__msg__GuidanceCmd * output)
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
  // target_detected
  output->target_detected = input->target_detected;
  // vel_n
  output->vel_n = input->vel_n;
  // vel_e
  output->vel_e = input->vel_e;
  // vel_d
  output->vel_d = input->vel_d;
  // yaw_rate
  output->yaw_rate = input->yaw_rate;
  return true;
}

suicide_drone_msgs__msg__GuidanceCmd *
suicide_drone_msgs__msg__GuidanceCmd__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  suicide_drone_msgs__msg__GuidanceCmd * msg = (suicide_drone_msgs__msg__GuidanceCmd *)allocator.allocate(sizeof(suicide_drone_msgs__msg__GuidanceCmd), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(suicide_drone_msgs__msg__GuidanceCmd));
  bool success = suicide_drone_msgs__msg__GuidanceCmd__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
suicide_drone_msgs__msg__GuidanceCmd__destroy(suicide_drone_msgs__msg__GuidanceCmd * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    suicide_drone_msgs__msg__GuidanceCmd__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
suicide_drone_msgs__msg__GuidanceCmd__Sequence__init(suicide_drone_msgs__msg__GuidanceCmd__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  suicide_drone_msgs__msg__GuidanceCmd * data = NULL;

  if (size) {
    data = (suicide_drone_msgs__msg__GuidanceCmd *)allocator.zero_allocate(size, sizeof(suicide_drone_msgs__msg__GuidanceCmd), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = suicide_drone_msgs__msg__GuidanceCmd__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        suicide_drone_msgs__msg__GuidanceCmd__fini(&data[i - 1]);
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
suicide_drone_msgs__msg__GuidanceCmd__Sequence__fini(suicide_drone_msgs__msg__GuidanceCmd__Sequence * array)
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
      suicide_drone_msgs__msg__GuidanceCmd__fini(&array->data[i]);
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

suicide_drone_msgs__msg__GuidanceCmd__Sequence *
suicide_drone_msgs__msg__GuidanceCmd__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  suicide_drone_msgs__msg__GuidanceCmd__Sequence * array = (suicide_drone_msgs__msg__GuidanceCmd__Sequence *)allocator.allocate(sizeof(suicide_drone_msgs__msg__GuidanceCmd__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = suicide_drone_msgs__msg__GuidanceCmd__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
suicide_drone_msgs__msg__GuidanceCmd__Sequence__destroy(suicide_drone_msgs__msg__GuidanceCmd__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    suicide_drone_msgs__msg__GuidanceCmd__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
suicide_drone_msgs__msg__GuidanceCmd__Sequence__are_equal(const suicide_drone_msgs__msg__GuidanceCmd__Sequence * lhs, const suicide_drone_msgs__msg__GuidanceCmd__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!suicide_drone_msgs__msg__GuidanceCmd__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
suicide_drone_msgs__msg__GuidanceCmd__Sequence__copy(
  const suicide_drone_msgs__msg__GuidanceCmd__Sequence * input,
  suicide_drone_msgs__msg__GuidanceCmd__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(suicide_drone_msgs__msg__GuidanceCmd);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    suicide_drone_msgs__msg__GuidanceCmd * data =
      (suicide_drone_msgs__msg__GuidanceCmd *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!suicide_drone_msgs__msg__GuidanceCmd__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          suicide_drone_msgs__msg__GuidanceCmd__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!suicide_drone_msgs__msg__GuidanceCmd__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
