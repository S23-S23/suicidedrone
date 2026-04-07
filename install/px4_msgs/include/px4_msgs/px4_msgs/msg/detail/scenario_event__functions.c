// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from px4_msgs:msg/ScenarioEvent.idl
// generated code does not contain a copyright notice
#include "px4_msgs/msg/detail/scenario_event__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
px4_msgs__msg__ScenarioEvent__init(px4_msgs__msg__ScenarioEvent * msg)
{
  if (!msg) {
    return false;
  }
  // timestamp
  // event_time
  // event_type
  // cmd_type
  // x
  // y
  // z
  // led_r
  // led_g
  // led_b
  // is_scenario_active
  return true;
}

void
px4_msgs__msg__ScenarioEvent__fini(px4_msgs__msg__ScenarioEvent * msg)
{
  if (!msg) {
    return;
  }
  // timestamp
  // event_time
  // event_type
  // cmd_type
  // x
  // y
  // z
  // led_r
  // led_g
  // led_b
  // is_scenario_active
}

bool
px4_msgs__msg__ScenarioEvent__are_equal(const px4_msgs__msg__ScenarioEvent * lhs, const px4_msgs__msg__ScenarioEvent * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // timestamp
  if (lhs->timestamp != rhs->timestamp) {
    return false;
  }
  // event_time
  if (lhs->event_time != rhs->event_time) {
    return false;
  }
  // event_type
  if (lhs->event_type != rhs->event_type) {
    return false;
  }
  // cmd_type
  if (lhs->cmd_type != rhs->cmd_type) {
    return false;
  }
  // x
  if (lhs->x != rhs->x) {
    return false;
  }
  // y
  if (lhs->y != rhs->y) {
    return false;
  }
  // z
  if (lhs->z != rhs->z) {
    return false;
  }
  // led_r
  if (lhs->led_r != rhs->led_r) {
    return false;
  }
  // led_g
  if (lhs->led_g != rhs->led_g) {
    return false;
  }
  // led_b
  if (lhs->led_b != rhs->led_b) {
    return false;
  }
  // is_scenario_active
  if (lhs->is_scenario_active != rhs->is_scenario_active) {
    return false;
  }
  return true;
}

bool
px4_msgs__msg__ScenarioEvent__copy(
  const px4_msgs__msg__ScenarioEvent * input,
  px4_msgs__msg__ScenarioEvent * output)
{
  if (!input || !output) {
    return false;
  }
  // timestamp
  output->timestamp = input->timestamp;
  // event_time
  output->event_time = input->event_time;
  // event_type
  output->event_type = input->event_type;
  // cmd_type
  output->cmd_type = input->cmd_type;
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // z
  output->z = input->z;
  // led_r
  output->led_r = input->led_r;
  // led_g
  output->led_g = input->led_g;
  // led_b
  output->led_b = input->led_b;
  // is_scenario_active
  output->is_scenario_active = input->is_scenario_active;
  return true;
}

px4_msgs__msg__ScenarioEvent *
px4_msgs__msg__ScenarioEvent__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__ScenarioEvent * msg = (px4_msgs__msg__ScenarioEvent *)allocator.allocate(sizeof(px4_msgs__msg__ScenarioEvent), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(px4_msgs__msg__ScenarioEvent));
  bool success = px4_msgs__msg__ScenarioEvent__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
px4_msgs__msg__ScenarioEvent__destroy(px4_msgs__msg__ScenarioEvent * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    px4_msgs__msg__ScenarioEvent__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
px4_msgs__msg__ScenarioEvent__Sequence__init(px4_msgs__msg__ScenarioEvent__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__ScenarioEvent * data = NULL;

  if (size) {
    data = (px4_msgs__msg__ScenarioEvent *)allocator.zero_allocate(size, sizeof(px4_msgs__msg__ScenarioEvent), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = px4_msgs__msg__ScenarioEvent__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        px4_msgs__msg__ScenarioEvent__fini(&data[i - 1]);
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
px4_msgs__msg__ScenarioEvent__Sequence__fini(px4_msgs__msg__ScenarioEvent__Sequence * array)
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
      px4_msgs__msg__ScenarioEvent__fini(&array->data[i]);
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

px4_msgs__msg__ScenarioEvent__Sequence *
px4_msgs__msg__ScenarioEvent__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__ScenarioEvent__Sequence * array = (px4_msgs__msg__ScenarioEvent__Sequence *)allocator.allocate(sizeof(px4_msgs__msg__ScenarioEvent__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = px4_msgs__msg__ScenarioEvent__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
px4_msgs__msg__ScenarioEvent__Sequence__destroy(px4_msgs__msg__ScenarioEvent__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    px4_msgs__msg__ScenarioEvent__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
px4_msgs__msg__ScenarioEvent__Sequence__are_equal(const px4_msgs__msg__ScenarioEvent__Sequence * lhs, const px4_msgs__msg__ScenarioEvent__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!px4_msgs__msg__ScenarioEvent__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
px4_msgs__msg__ScenarioEvent__Sequence__copy(
  const px4_msgs__msg__ScenarioEvent__Sequence * input,
  px4_msgs__msg__ScenarioEvent__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(px4_msgs__msg__ScenarioEvent);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    px4_msgs__msg__ScenarioEvent * data =
      (px4_msgs__msg__ScenarioEvent *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!px4_msgs__msg__ScenarioEvent__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          px4_msgs__msg__ScenarioEvent__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!px4_msgs__msg__ScenarioEvent__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
