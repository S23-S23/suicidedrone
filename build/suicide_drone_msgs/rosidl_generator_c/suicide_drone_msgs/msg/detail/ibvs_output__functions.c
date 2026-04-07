// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from suicide_drone_msgs:msg/IBVSOutput.idl
// generated code does not contain a copyright notice
#include "suicide_drone_msgs/msg/detail/ibvs_output__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
suicide_drone_msgs__msg__IBVSOutput__init(suicide_drone_msgs__msg__IBVSOutput * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    suicide_drone_msgs__msg__IBVSOutput__fini(msg);
    return false;
  }
  // detected
  // q_y
  // q_z
  // fov_yaw_rate
  // fov_vel_z
  return true;
}

void
suicide_drone_msgs__msg__IBVSOutput__fini(suicide_drone_msgs__msg__IBVSOutput * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // detected
  // q_y
  // q_z
  // fov_yaw_rate
  // fov_vel_z
}

bool
suicide_drone_msgs__msg__IBVSOutput__are_equal(const suicide_drone_msgs__msg__IBVSOutput * lhs, const suicide_drone_msgs__msg__IBVSOutput * rhs)
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
  // detected
  if (lhs->detected != rhs->detected) {
    return false;
  }
  // q_y
  if (lhs->q_y != rhs->q_y) {
    return false;
  }
  // q_z
  if (lhs->q_z != rhs->q_z) {
    return false;
  }
  // fov_yaw_rate
  if (lhs->fov_yaw_rate != rhs->fov_yaw_rate) {
    return false;
  }
  // fov_vel_z
  if (lhs->fov_vel_z != rhs->fov_vel_z) {
    return false;
  }
  return true;
}

bool
suicide_drone_msgs__msg__IBVSOutput__copy(
  const suicide_drone_msgs__msg__IBVSOutput * input,
  suicide_drone_msgs__msg__IBVSOutput * output)
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
  // detected
  output->detected = input->detected;
  // q_y
  output->q_y = input->q_y;
  // q_z
  output->q_z = input->q_z;
  // fov_yaw_rate
  output->fov_yaw_rate = input->fov_yaw_rate;
  // fov_vel_z
  output->fov_vel_z = input->fov_vel_z;
  return true;
}

suicide_drone_msgs__msg__IBVSOutput *
suicide_drone_msgs__msg__IBVSOutput__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  suicide_drone_msgs__msg__IBVSOutput * msg = (suicide_drone_msgs__msg__IBVSOutput *)allocator.allocate(sizeof(suicide_drone_msgs__msg__IBVSOutput), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(suicide_drone_msgs__msg__IBVSOutput));
  bool success = suicide_drone_msgs__msg__IBVSOutput__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
suicide_drone_msgs__msg__IBVSOutput__destroy(suicide_drone_msgs__msg__IBVSOutput * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    suicide_drone_msgs__msg__IBVSOutput__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
suicide_drone_msgs__msg__IBVSOutput__Sequence__init(suicide_drone_msgs__msg__IBVSOutput__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  suicide_drone_msgs__msg__IBVSOutput * data = NULL;

  if (size) {
    data = (suicide_drone_msgs__msg__IBVSOutput *)allocator.zero_allocate(size, sizeof(suicide_drone_msgs__msg__IBVSOutput), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = suicide_drone_msgs__msg__IBVSOutput__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        suicide_drone_msgs__msg__IBVSOutput__fini(&data[i - 1]);
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
suicide_drone_msgs__msg__IBVSOutput__Sequence__fini(suicide_drone_msgs__msg__IBVSOutput__Sequence * array)
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
      suicide_drone_msgs__msg__IBVSOutput__fini(&array->data[i]);
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

suicide_drone_msgs__msg__IBVSOutput__Sequence *
suicide_drone_msgs__msg__IBVSOutput__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  suicide_drone_msgs__msg__IBVSOutput__Sequence * array = (suicide_drone_msgs__msg__IBVSOutput__Sequence *)allocator.allocate(sizeof(suicide_drone_msgs__msg__IBVSOutput__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = suicide_drone_msgs__msg__IBVSOutput__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
suicide_drone_msgs__msg__IBVSOutput__Sequence__destroy(suicide_drone_msgs__msg__IBVSOutput__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    suicide_drone_msgs__msg__IBVSOutput__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
suicide_drone_msgs__msg__IBVSOutput__Sequence__are_equal(const suicide_drone_msgs__msg__IBVSOutput__Sequence * lhs, const suicide_drone_msgs__msg__IBVSOutput__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!suicide_drone_msgs__msg__IBVSOutput__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
suicide_drone_msgs__msg__IBVSOutput__Sequence__copy(
  const suicide_drone_msgs__msg__IBVSOutput__Sequence * input,
  suicide_drone_msgs__msg__IBVSOutput__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(suicide_drone_msgs__msg__IBVSOutput);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    suicide_drone_msgs__msg__IBVSOutput * data =
      (suicide_drone_msgs__msg__IBVSOutput *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!suicide_drone_msgs__msg__IBVSOutput__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          suicide_drone_msgs__msg__IBVSOutput__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!suicide_drone_msgs__msg__IBVSOutput__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
