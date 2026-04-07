// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from px4_msgs:srv/TrajectorySetpoint.idl
// generated code does not contain a copyright notice
#include "px4_msgs/srv/detail/trajectory_setpoint__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `request`
#include "px4_msgs/msg/detail/trajectory_setpoint__functions.h"

bool
px4_msgs__srv__TrajectorySetpoint_Request__init(px4_msgs__srv__TrajectorySetpoint_Request * msg)
{
  if (!msg) {
    return false;
  }
  // request
  if (!px4_msgs__msg__TrajectorySetpoint__init(&msg->request)) {
    px4_msgs__srv__TrajectorySetpoint_Request__fini(msg);
    return false;
  }
  return true;
}

void
px4_msgs__srv__TrajectorySetpoint_Request__fini(px4_msgs__srv__TrajectorySetpoint_Request * msg)
{
  if (!msg) {
    return;
  }
  // request
  px4_msgs__msg__TrajectorySetpoint__fini(&msg->request);
}

bool
px4_msgs__srv__TrajectorySetpoint_Request__are_equal(const px4_msgs__srv__TrajectorySetpoint_Request * lhs, const px4_msgs__srv__TrajectorySetpoint_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // request
  if (!px4_msgs__msg__TrajectorySetpoint__are_equal(
      &(lhs->request), &(rhs->request)))
  {
    return false;
  }
  return true;
}

bool
px4_msgs__srv__TrajectorySetpoint_Request__copy(
  const px4_msgs__srv__TrajectorySetpoint_Request * input,
  px4_msgs__srv__TrajectorySetpoint_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // request
  if (!px4_msgs__msg__TrajectorySetpoint__copy(
      &(input->request), &(output->request)))
  {
    return false;
  }
  return true;
}

px4_msgs__srv__TrajectorySetpoint_Request *
px4_msgs__srv__TrajectorySetpoint_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__srv__TrajectorySetpoint_Request * msg = (px4_msgs__srv__TrajectorySetpoint_Request *)allocator.allocate(sizeof(px4_msgs__srv__TrajectorySetpoint_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(px4_msgs__srv__TrajectorySetpoint_Request));
  bool success = px4_msgs__srv__TrajectorySetpoint_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
px4_msgs__srv__TrajectorySetpoint_Request__destroy(px4_msgs__srv__TrajectorySetpoint_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    px4_msgs__srv__TrajectorySetpoint_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
px4_msgs__srv__TrajectorySetpoint_Request__Sequence__init(px4_msgs__srv__TrajectorySetpoint_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__srv__TrajectorySetpoint_Request * data = NULL;

  if (size) {
    data = (px4_msgs__srv__TrajectorySetpoint_Request *)allocator.zero_allocate(size, sizeof(px4_msgs__srv__TrajectorySetpoint_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = px4_msgs__srv__TrajectorySetpoint_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        px4_msgs__srv__TrajectorySetpoint_Request__fini(&data[i - 1]);
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
px4_msgs__srv__TrajectorySetpoint_Request__Sequence__fini(px4_msgs__srv__TrajectorySetpoint_Request__Sequence * array)
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
      px4_msgs__srv__TrajectorySetpoint_Request__fini(&array->data[i]);
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

px4_msgs__srv__TrajectorySetpoint_Request__Sequence *
px4_msgs__srv__TrajectorySetpoint_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__srv__TrajectorySetpoint_Request__Sequence * array = (px4_msgs__srv__TrajectorySetpoint_Request__Sequence *)allocator.allocate(sizeof(px4_msgs__srv__TrajectorySetpoint_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = px4_msgs__srv__TrajectorySetpoint_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
px4_msgs__srv__TrajectorySetpoint_Request__Sequence__destroy(px4_msgs__srv__TrajectorySetpoint_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    px4_msgs__srv__TrajectorySetpoint_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
px4_msgs__srv__TrajectorySetpoint_Request__Sequence__are_equal(const px4_msgs__srv__TrajectorySetpoint_Request__Sequence * lhs, const px4_msgs__srv__TrajectorySetpoint_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!px4_msgs__srv__TrajectorySetpoint_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
px4_msgs__srv__TrajectorySetpoint_Request__Sequence__copy(
  const px4_msgs__srv__TrajectorySetpoint_Request__Sequence * input,
  px4_msgs__srv__TrajectorySetpoint_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(px4_msgs__srv__TrajectorySetpoint_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    px4_msgs__srv__TrajectorySetpoint_Request * data =
      (px4_msgs__srv__TrajectorySetpoint_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!px4_msgs__srv__TrajectorySetpoint_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          px4_msgs__srv__TrajectorySetpoint_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!px4_msgs__srv__TrajectorySetpoint_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
px4_msgs__srv__TrajectorySetpoint_Response__init(px4_msgs__srv__TrajectorySetpoint_Response * msg)
{
  if (!msg) {
    return false;
  }
  // reply
  return true;
}

void
px4_msgs__srv__TrajectorySetpoint_Response__fini(px4_msgs__srv__TrajectorySetpoint_Response * msg)
{
  if (!msg) {
    return;
  }
  // reply
}

bool
px4_msgs__srv__TrajectorySetpoint_Response__are_equal(const px4_msgs__srv__TrajectorySetpoint_Response * lhs, const px4_msgs__srv__TrajectorySetpoint_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // reply
  if (lhs->reply != rhs->reply) {
    return false;
  }
  return true;
}

bool
px4_msgs__srv__TrajectorySetpoint_Response__copy(
  const px4_msgs__srv__TrajectorySetpoint_Response * input,
  px4_msgs__srv__TrajectorySetpoint_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // reply
  output->reply = input->reply;
  return true;
}

px4_msgs__srv__TrajectorySetpoint_Response *
px4_msgs__srv__TrajectorySetpoint_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__srv__TrajectorySetpoint_Response * msg = (px4_msgs__srv__TrajectorySetpoint_Response *)allocator.allocate(sizeof(px4_msgs__srv__TrajectorySetpoint_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(px4_msgs__srv__TrajectorySetpoint_Response));
  bool success = px4_msgs__srv__TrajectorySetpoint_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
px4_msgs__srv__TrajectorySetpoint_Response__destroy(px4_msgs__srv__TrajectorySetpoint_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    px4_msgs__srv__TrajectorySetpoint_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
px4_msgs__srv__TrajectorySetpoint_Response__Sequence__init(px4_msgs__srv__TrajectorySetpoint_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__srv__TrajectorySetpoint_Response * data = NULL;

  if (size) {
    data = (px4_msgs__srv__TrajectorySetpoint_Response *)allocator.zero_allocate(size, sizeof(px4_msgs__srv__TrajectorySetpoint_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = px4_msgs__srv__TrajectorySetpoint_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        px4_msgs__srv__TrajectorySetpoint_Response__fini(&data[i - 1]);
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
px4_msgs__srv__TrajectorySetpoint_Response__Sequence__fini(px4_msgs__srv__TrajectorySetpoint_Response__Sequence * array)
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
      px4_msgs__srv__TrajectorySetpoint_Response__fini(&array->data[i]);
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

px4_msgs__srv__TrajectorySetpoint_Response__Sequence *
px4_msgs__srv__TrajectorySetpoint_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__srv__TrajectorySetpoint_Response__Sequence * array = (px4_msgs__srv__TrajectorySetpoint_Response__Sequence *)allocator.allocate(sizeof(px4_msgs__srv__TrajectorySetpoint_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = px4_msgs__srv__TrajectorySetpoint_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
px4_msgs__srv__TrajectorySetpoint_Response__Sequence__destroy(px4_msgs__srv__TrajectorySetpoint_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    px4_msgs__srv__TrajectorySetpoint_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
px4_msgs__srv__TrajectorySetpoint_Response__Sequence__are_equal(const px4_msgs__srv__TrajectorySetpoint_Response__Sequence * lhs, const px4_msgs__srv__TrajectorySetpoint_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!px4_msgs__srv__TrajectorySetpoint_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
px4_msgs__srv__TrajectorySetpoint_Response__Sequence__copy(
  const px4_msgs__srv__TrajectorySetpoint_Response__Sequence * input,
  px4_msgs__srv__TrajectorySetpoint_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(px4_msgs__srv__TrajectorySetpoint_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    px4_msgs__srv__TrajectorySetpoint_Response * data =
      (px4_msgs__srv__TrajectorySetpoint_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!px4_msgs__srv__TrajectorySetpoint_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          px4_msgs__srv__TrajectorySetpoint_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!px4_msgs__srv__TrajectorySetpoint_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
