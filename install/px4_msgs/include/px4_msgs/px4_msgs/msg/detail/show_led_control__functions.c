// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from px4_msgs:msg/ShowLedControl.idl
// generated code does not contain a copyright notice
#include "px4_msgs/msg/detail/show_led_control__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
px4_msgs__msg__ShowLedControl__init(px4_msgs__msg__ShowLedControl * msg)
{
  if (!msg) {
    return false;
  }
  // timestamp
  // type
  // r
  // g
  // b
  // brightness
  // speed
  return true;
}

void
px4_msgs__msg__ShowLedControl__fini(px4_msgs__msg__ShowLedControl * msg)
{
  if (!msg) {
    return;
  }
  // timestamp
  // type
  // r
  // g
  // b
  // brightness
  // speed
}

bool
px4_msgs__msg__ShowLedControl__are_equal(const px4_msgs__msg__ShowLedControl * lhs, const px4_msgs__msg__ShowLedControl * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // timestamp
  if (lhs->timestamp != rhs->timestamp) {
    return false;
  }
  // type
  if (lhs->type != rhs->type) {
    return false;
  }
  // r
  if (lhs->r != rhs->r) {
    return false;
  }
  // g
  if (lhs->g != rhs->g) {
    return false;
  }
  // b
  if (lhs->b != rhs->b) {
    return false;
  }
  // brightness
  if (lhs->brightness != rhs->brightness) {
    return false;
  }
  // speed
  if (lhs->speed != rhs->speed) {
    return false;
  }
  return true;
}

bool
px4_msgs__msg__ShowLedControl__copy(
  const px4_msgs__msg__ShowLedControl * input,
  px4_msgs__msg__ShowLedControl * output)
{
  if (!input || !output) {
    return false;
  }
  // timestamp
  output->timestamp = input->timestamp;
  // type
  output->type = input->type;
  // r
  output->r = input->r;
  // g
  output->g = input->g;
  // b
  output->b = input->b;
  // brightness
  output->brightness = input->brightness;
  // speed
  output->speed = input->speed;
  return true;
}

px4_msgs__msg__ShowLedControl *
px4_msgs__msg__ShowLedControl__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__ShowLedControl * msg = (px4_msgs__msg__ShowLedControl *)allocator.allocate(sizeof(px4_msgs__msg__ShowLedControl), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(px4_msgs__msg__ShowLedControl));
  bool success = px4_msgs__msg__ShowLedControl__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
px4_msgs__msg__ShowLedControl__destroy(px4_msgs__msg__ShowLedControl * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    px4_msgs__msg__ShowLedControl__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
px4_msgs__msg__ShowLedControl__Sequence__init(px4_msgs__msg__ShowLedControl__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__ShowLedControl * data = NULL;

  if (size) {
    data = (px4_msgs__msg__ShowLedControl *)allocator.zero_allocate(size, sizeof(px4_msgs__msg__ShowLedControl), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = px4_msgs__msg__ShowLedControl__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        px4_msgs__msg__ShowLedControl__fini(&data[i - 1]);
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
px4_msgs__msg__ShowLedControl__Sequence__fini(px4_msgs__msg__ShowLedControl__Sequence * array)
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
      px4_msgs__msg__ShowLedControl__fini(&array->data[i]);
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

px4_msgs__msg__ShowLedControl__Sequence *
px4_msgs__msg__ShowLedControl__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__ShowLedControl__Sequence * array = (px4_msgs__msg__ShowLedControl__Sequence *)allocator.allocate(sizeof(px4_msgs__msg__ShowLedControl__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = px4_msgs__msg__ShowLedControl__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
px4_msgs__msg__ShowLedControl__Sequence__destroy(px4_msgs__msg__ShowLedControl__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    px4_msgs__msg__ShowLedControl__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
px4_msgs__msg__ShowLedControl__Sequence__are_equal(const px4_msgs__msg__ShowLedControl__Sequence * lhs, const px4_msgs__msg__ShowLedControl__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!px4_msgs__msg__ShowLedControl__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
px4_msgs__msg__ShowLedControl__Sequence__copy(
  const px4_msgs__msg__ShowLedControl__Sequence * input,
  px4_msgs__msg__ShowLedControl__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(px4_msgs__msg__ShowLedControl);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    px4_msgs__msg__ShowLedControl * data =
      (px4_msgs__msg__ShowLedControl *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!px4_msgs__msg__ShowLedControl__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          px4_msgs__msg__ShowLedControl__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!px4_msgs__msg__ShowLedControl__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
