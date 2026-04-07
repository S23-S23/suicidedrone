// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from px4_msgs:msg/F9pRtk.idl
// generated code does not contain a copyright notice
#include "px4_msgs/msg/detail/f9p_rtk__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
px4_msgs__msg__F9pRtk__init(px4_msgs__msg__F9pRtk * msg)
{
  if (!msg) {
    return false;
  }
  // timestamp
  // device_id
  // tow
  // age_corr
  // fix_type
  // satellites_used
  // n
  // e
  // d
  // v_n
  // v_e
  // v_d
  // acc_n
  // acc_e
  // acc_d
  return true;
}

void
px4_msgs__msg__F9pRtk__fini(px4_msgs__msg__F9pRtk * msg)
{
  if (!msg) {
    return;
  }
  // timestamp
  // device_id
  // tow
  // age_corr
  // fix_type
  // satellites_used
  // n
  // e
  // d
  // v_n
  // v_e
  // v_d
  // acc_n
  // acc_e
  // acc_d
}

bool
px4_msgs__msg__F9pRtk__are_equal(const px4_msgs__msg__F9pRtk * lhs, const px4_msgs__msg__F9pRtk * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // timestamp
  if (lhs->timestamp != rhs->timestamp) {
    return false;
  }
  // device_id
  if (lhs->device_id != rhs->device_id) {
    return false;
  }
  // tow
  if (lhs->tow != rhs->tow) {
    return false;
  }
  // age_corr
  if (lhs->age_corr != rhs->age_corr) {
    return false;
  }
  // fix_type
  if (lhs->fix_type != rhs->fix_type) {
    return false;
  }
  // satellites_used
  if (lhs->satellites_used != rhs->satellites_used) {
    return false;
  }
  // n
  if (lhs->n != rhs->n) {
    return false;
  }
  // e
  if (lhs->e != rhs->e) {
    return false;
  }
  // d
  if (lhs->d != rhs->d) {
    return false;
  }
  // v_n
  if (lhs->v_n != rhs->v_n) {
    return false;
  }
  // v_e
  if (lhs->v_e != rhs->v_e) {
    return false;
  }
  // v_d
  if (lhs->v_d != rhs->v_d) {
    return false;
  }
  // acc_n
  if (lhs->acc_n != rhs->acc_n) {
    return false;
  }
  // acc_e
  if (lhs->acc_e != rhs->acc_e) {
    return false;
  }
  // acc_d
  if (lhs->acc_d != rhs->acc_d) {
    return false;
  }
  return true;
}

bool
px4_msgs__msg__F9pRtk__copy(
  const px4_msgs__msg__F9pRtk * input,
  px4_msgs__msg__F9pRtk * output)
{
  if (!input || !output) {
    return false;
  }
  // timestamp
  output->timestamp = input->timestamp;
  // device_id
  output->device_id = input->device_id;
  // tow
  output->tow = input->tow;
  // age_corr
  output->age_corr = input->age_corr;
  // fix_type
  output->fix_type = input->fix_type;
  // satellites_used
  output->satellites_used = input->satellites_used;
  // n
  output->n = input->n;
  // e
  output->e = input->e;
  // d
  output->d = input->d;
  // v_n
  output->v_n = input->v_n;
  // v_e
  output->v_e = input->v_e;
  // v_d
  output->v_d = input->v_d;
  // acc_n
  output->acc_n = input->acc_n;
  // acc_e
  output->acc_e = input->acc_e;
  // acc_d
  output->acc_d = input->acc_d;
  return true;
}

px4_msgs__msg__F9pRtk *
px4_msgs__msg__F9pRtk__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__F9pRtk * msg = (px4_msgs__msg__F9pRtk *)allocator.allocate(sizeof(px4_msgs__msg__F9pRtk), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(px4_msgs__msg__F9pRtk));
  bool success = px4_msgs__msg__F9pRtk__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
px4_msgs__msg__F9pRtk__destroy(px4_msgs__msg__F9pRtk * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    px4_msgs__msg__F9pRtk__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
px4_msgs__msg__F9pRtk__Sequence__init(px4_msgs__msg__F9pRtk__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__F9pRtk * data = NULL;

  if (size) {
    data = (px4_msgs__msg__F9pRtk *)allocator.zero_allocate(size, sizeof(px4_msgs__msg__F9pRtk), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = px4_msgs__msg__F9pRtk__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        px4_msgs__msg__F9pRtk__fini(&data[i - 1]);
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
px4_msgs__msg__F9pRtk__Sequence__fini(px4_msgs__msg__F9pRtk__Sequence * array)
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
      px4_msgs__msg__F9pRtk__fini(&array->data[i]);
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

px4_msgs__msg__F9pRtk__Sequence *
px4_msgs__msg__F9pRtk__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__F9pRtk__Sequence * array = (px4_msgs__msg__F9pRtk__Sequence *)allocator.allocate(sizeof(px4_msgs__msg__F9pRtk__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = px4_msgs__msg__F9pRtk__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
px4_msgs__msg__F9pRtk__Sequence__destroy(px4_msgs__msg__F9pRtk__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    px4_msgs__msg__F9pRtk__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
px4_msgs__msg__F9pRtk__Sequence__are_equal(const px4_msgs__msg__F9pRtk__Sequence * lhs, const px4_msgs__msg__F9pRtk__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!px4_msgs__msg__F9pRtk__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
px4_msgs__msg__F9pRtk__Sequence__copy(
  const px4_msgs__msg__F9pRtk__Sequence * input,
  px4_msgs__msg__F9pRtk__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(px4_msgs__msg__F9pRtk);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    px4_msgs__msg__F9pRtk * data =
      (px4_msgs__msg__F9pRtk *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!px4_msgs__msg__F9pRtk__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          px4_msgs__msg__F9pRtk__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!px4_msgs__msg__F9pRtk__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
