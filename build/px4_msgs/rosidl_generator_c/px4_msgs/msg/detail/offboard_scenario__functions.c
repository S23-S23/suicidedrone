// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from px4_msgs:msg/OffboardScenario.idl
// generated code does not contain a copyright notice
#include "px4_msgs/msg/detail/offboard_scenario__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
px4_msgs__msg__OffboardScenario__init(px4_msgs__msg__OffboardScenario * msg)
{
  if (!msg) {
    return false;
  }
  // timestamp
  // current_time
  // start_time
  // seq
  // offset_x
  // offset_y
  // ready_sc_file
  return true;
}

void
px4_msgs__msg__OffboardScenario__fini(px4_msgs__msg__OffboardScenario * msg)
{
  if (!msg) {
    return;
  }
  // timestamp
  // current_time
  // start_time
  // seq
  // offset_x
  // offset_y
  // ready_sc_file
}

bool
px4_msgs__msg__OffboardScenario__are_equal(const px4_msgs__msg__OffboardScenario * lhs, const px4_msgs__msg__OffboardScenario * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // timestamp
  if (lhs->timestamp != rhs->timestamp) {
    return false;
  }
  // current_time
  if (lhs->current_time != rhs->current_time) {
    return false;
  }
  // start_time
  if (lhs->start_time != rhs->start_time) {
    return false;
  }
  // seq
  if (lhs->seq != rhs->seq) {
    return false;
  }
  // offset_x
  if (lhs->offset_x != rhs->offset_x) {
    return false;
  }
  // offset_y
  if (lhs->offset_y != rhs->offset_y) {
    return false;
  }
  // ready_sc_file
  if (lhs->ready_sc_file != rhs->ready_sc_file) {
    return false;
  }
  return true;
}

bool
px4_msgs__msg__OffboardScenario__copy(
  const px4_msgs__msg__OffboardScenario * input,
  px4_msgs__msg__OffboardScenario * output)
{
  if (!input || !output) {
    return false;
  }
  // timestamp
  output->timestamp = input->timestamp;
  // current_time
  output->current_time = input->current_time;
  // start_time
  output->start_time = input->start_time;
  // seq
  output->seq = input->seq;
  // offset_x
  output->offset_x = input->offset_x;
  // offset_y
  output->offset_y = input->offset_y;
  // ready_sc_file
  output->ready_sc_file = input->ready_sc_file;
  return true;
}

px4_msgs__msg__OffboardScenario *
px4_msgs__msg__OffboardScenario__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__OffboardScenario * msg = (px4_msgs__msg__OffboardScenario *)allocator.allocate(sizeof(px4_msgs__msg__OffboardScenario), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(px4_msgs__msg__OffboardScenario));
  bool success = px4_msgs__msg__OffboardScenario__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
px4_msgs__msg__OffboardScenario__destroy(px4_msgs__msg__OffboardScenario * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    px4_msgs__msg__OffboardScenario__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
px4_msgs__msg__OffboardScenario__Sequence__init(px4_msgs__msg__OffboardScenario__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__OffboardScenario * data = NULL;

  if (size) {
    data = (px4_msgs__msg__OffboardScenario *)allocator.zero_allocate(size, sizeof(px4_msgs__msg__OffboardScenario), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = px4_msgs__msg__OffboardScenario__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        px4_msgs__msg__OffboardScenario__fini(&data[i - 1]);
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
px4_msgs__msg__OffboardScenario__Sequence__fini(px4_msgs__msg__OffboardScenario__Sequence * array)
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
      px4_msgs__msg__OffboardScenario__fini(&array->data[i]);
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

px4_msgs__msg__OffboardScenario__Sequence *
px4_msgs__msg__OffboardScenario__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__OffboardScenario__Sequence * array = (px4_msgs__msg__OffboardScenario__Sequence *)allocator.allocate(sizeof(px4_msgs__msg__OffboardScenario__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = px4_msgs__msg__OffboardScenario__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
px4_msgs__msg__OffboardScenario__Sequence__destroy(px4_msgs__msg__OffboardScenario__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    px4_msgs__msg__OffboardScenario__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
px4_msgs__msg__OffboardScenario__Sequence__are_equal(const px4_msgs__msg__OffboardScenario__Sequence * lhs, const px4_msgs__msg__OffboardScenario__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!px4_msgs__msg__OffboardScenario__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
px4_msgs__msg__OffboardScenario__Sequence__copy(
  const px4_msgs__msg__OffboardScenario__Sequence * input,
  px4_msgs__msg__OffboardScenario__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(px4_msgs__msg__OffboardScenario);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    px4_msgs__msg__OffboardScenario * data =
      (px4_msgs__msg__OffboardScenario *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!px4_msgs__msg__OffboardScenario__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          px4_msgs__msg__OffboardScenario__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!px4_msgs__msg__OffboardScenario__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
