// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from px4_msgs:msg/ScenarioCommand.idl
// generated code does not contain a copyright notice
#include "px4_msgs/msg/detail/scenario_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
px4_msgs__msg__ScenarioCommand__init(px4_msgs__msg__ScenarioCommand * msg)
{
  if (!msg) {
    return false;
  }
  // timestamp
  // cmd
  // param1
  // param2
  // param3
  // param4
  // param5
  return true;
}

void
px4_msgs__msg__ScenarioCommand__fini(px4_msgs__msg__ScenarioCommand * msg)
{
  if (!msg) {
    return;
  }
  // timestamp
  // cmd
  // param1
  // param2
  // param3
  // param4
  // param5
}

bool
px4_msgs__msg__ScenarioCommand__are_equal(const px4_msgs__msg__ScenarioCommand * lhs, const px4_msgs__msg__ScenarioCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // timestamp
  if (lhs->timestamp != rhs->timestamp) {
    return false;
  }
  // cmd
  if (lhs->cmd != rhs->cmd) {
    return false;
  }
  // param1
  if (lhs->param1 != rhs->param1) {
    return false;
  }
  // param2
  if (lhs->param2 != rhs->param2) {
    return false;
  }
  // param3
  if (lhs->param3 != rhs->param3) {
    return false;
  }
  // param4
  if (lhs->param4 != rhs->param4) {
    return false;
  }
  // param5
  for (size_t i = 0; i < 32; ++i) {
    if (lhs->param5[i] != rhs->param5[i]) {
      return false;
    }
  }
  return true;
}

bool
px4_msgs__msg__ScenarioCommand__copy(
  const px4_msgs__msg__ScenarioCommand * input,
  px4_msgs__msg__ScenarioCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // timestamp
  output->timestamp = input->timestamp;
  // cmd
  output->cmd = input->cmd;
  // param1
  output->param1 = input->param1;
  // param2
  output->param2 = input->param2;
  // param3
  output->param3 = input->param3;
  // param4
  output->param4 = input->param4;
  // param5
  for (size_t i = 0; i < 32; ++i) {
    output->param5[i] = input->param5[i];
  }
  return true;
}

px4_msgs__msg__ScenarioCommand *
px4_msgs__msg__ScenarioCommand__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__ScenarioCommand * msg = (px4_msgs__msg__ScenarioCommand *)allocator.allocate(sizeof(px4_msgs__msg__ScenarioCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(px4_msgs__msg__ScenarioCommand));
  bool success = px4_msgs__msg__ScenarioCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
px4_msgs__msg__ScenarioCommand__destroy(px4_msgs__msg__ScenarioCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    px4_msgs__msg__ScenarioCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
px4_msgs__msg__ScenarioCommand__Sequence__init(px4_msgs__msg__ScenarioCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__ScenarioCommand * data = NULL;

  if (size) {
    data = (px4_msgs__msg__ScenarioCommand *)allocator.zero_allocate(size, sizeof(px4_msgs__msg__ScenarioCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = px4_msgs__msg__ScenarioCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        px4_msgs__msg__ScenarioCommand__fini(&data[i - 1]);
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
px4_msgs__msg__ScenarioCommand__Sequence__fini(px4_msgs__msg__ScenarioCommand__Sequence * array)
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
      px4_msgs__msg__ScenarioCommand__fini(&array->data[i]);
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

px4_msgs__msg__ScenarioCommand__Sequence *
px4_msgs__msg__ScenarioCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__ScenarioCommand__Sequence * array = (px4_msgs__msg__ScenarioCommand__Sequence *)allocator.allocate(sizeof(px4_msgs__msg__ScenarioCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = px4_msgs__msg__ScenarioCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
px4_msgs__msg__ScenarioCommand__Sequence__destroy(px4_msgs__msg__ScenarioCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    px4_msgs__msg__ScenarioCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
px4_msgs__msg__ScenarioCommand__Sequence__are_equal(const px4_msgs__msg__ScenarioCommand__Sequence * lhs, const px4_msgs__msg__ScenarioCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!px4_msgs__msg__ScenarioCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
px4_msgs__msg__ScenarioCommand__Sequence__copy(
  const px4_msgs__msg__ScenarioCommand__Sequence * input,
  px4_msgs__msg__ScenarioCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(px4_msgs__msg__ScenarioCommand);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    px4_msgs__msg__ScenarioCommand * data =
      (px4_msgs__msg__ScenarioCommand *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!px4_msgs__msg__ScenarioCommand__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          px4_msgs__msg__ScenarioCommand__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!px4_msgs__msg__ScenarioCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
