// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from px4_msgs:msg/Monitoring.idl
// generated code does not contain a copyright notice
#include "px4_msgs/msg/detail/monitoring__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
px4_msgs__msg__Monitoring__init(px4_msgs__msg__Monitoring * msg)
{
  if (!msg) {
    return false;
  }
  // timestamp
  // tow
  // pos_x
  // pos_y
  // pos_z
  // lat
  // lon
  // alt
  // ref_lat
  // ref_lon
  // ref_alt
  // head
  // roll
  // pitch
  // status1
  // status2
  // rtk_nbase
  // rtk_nrover
  // battery
  // r
  // g
  // b
  // rtk_n
  // rtk_e
  // rtk_d
  // nav_state
  return true;
}

void
px4_msgs__msg__Monitoring__fini(px4_msgs__msg__Monitoring * msg)
{
  if (!msg) {
    return;
  }
  // timestamp
  // tow
  // pos_x
  // pos_y
  // pos_z
  // lat
  // lon
  // alt
  // ref_lat
  // ref_lon
  // ref_alt
  // head
  // roll
  // pitch
  // status1
  // status2
  // rtk_nbase
  // rtk_nrover
  // battery
  // r
  // g
  // b
  // rtk_n
  // rtk_e
  // rtk_d
  // nav_state
}

bool
px4_msgs__msg__Monitoring__are_equal(const px4_msgs__msg__Monitoring * lhs, const px4_msgs__msg__Monitoring * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // timestamp
  if (lhs->timestamp != rhs->timestamp) {
    return false;
  }
  // tow
  if (lhs->tow != rhs->tow) {
    return false;
  }
  // pos_x
  if (lhs->pos_x != rhs->pos_x) {
    return false;
  }
  // pos_y
  if (lhs->pos_y != rhs->pos_y) {
    return false;
  }
  // pos_z
  if (lhs->pos_z != rhs->pos_z) {
    return false;
  }
  // lat
  if (lhs->lat != rhs->lat) {
    return false;
  }
  // lon
  if (lhs->lon != rhs->lon) {
    return false;
  }
  // alt
  if (lhs->alt != rhs->alt) {
    return false;
  }
  // ref_lat
  if (lhs->ref_lat != rhs->ref_lat) {
    return false;
  }
  // ref_lon
  if (lhs->ref_lon != rhs->ref_lon) {
    return false;
  }
  // ref_alt
  if (lhs->ref_alt != rhs->ref_alt) {
    return false;
  }
  // head
  if (lhs->head != rhs->head) {
    return false;
  }
  // roll
  if (lhs->roll != rhs->roll) {
    return false;
  }
  // pitch
  if (lhs->pitch != rhs->pitch) {
    return false;
  }
  // status1
  if (lhs->status1 != rhs->status1) {
    return false;
  }
  // status2
  if (lhs->status2 != rhs->status2) {
    return false;
  }
  // rtk_nbase
  if (lhs->rtk_nbase != rhs->rtk_nbase) {
    return false;
  }
  // rtk_nrover
  if (lhs->rtk_nrover != rhs->rtk_nrover) {
    return false;
  }
  // battery
  if (lhs->battery != rhs->battery) {
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
  // rtk_n
  if (lhs->rtk_n != rhs->rtk_n) {
    return false;
  }
  // rtk_e
  if (lhs->rtk_e != rhs->rtk_e) {
    return false;
  }
  // rtk_d
  if (lhs->rtk_d != rhs->rtk_d) {
    return false;
  }
  // nav_state
  if (lhs->nav_state != rhs->nav_state) {
    return false;
  }
  return true;
}

bool
px4_msgs__msg__Monitoring__copy(
  const px4_msgs__msg__Monitoring * input,
  px4_msgs__msg__Monitoring * output)
{
  if (!input || !output) {
    return false;
  }
  // timestamp
  output->timestamp = input->timestamp;
  // tow
  output->tow = input->tow;
  // pos_x
  output->pos_x = input->pos_x;
  // pos_y
  output->pos_y = input->pos_y;
  // pos_z
  output->pos_z = input->pos_z;
  // lat
  output->lat = input->lat;
  // lon
  output->lon = input->lon;
  // alt
  output->alt = input->alt;
  // ref_lat
  output->ref_lat = input->ref_lat;
  // ref_lon
  output->ref_lon = input->ref_lon;
  // ref_alt
  output->ref_alt = input->ref_alt;
  // head
  output->head = input->head;
  // roll
  output->roll = input->roll;
  // pitch
  output->pitch = input->pitch;
  // status1
  output->status1 = input->status1;
  // status2
  output->status2 = input->status2;
  // rtk_nbase
  output->rtk_nbase = input->rtk_nbase;
  // rtk_nrover
  output->rtk_nrover = input->rtk_nrover;
  // battery
  output->battery = input->battery;
  // r
  output->r = input->r;
  // g
  output->g = input->g;
  // b
  output->b = input->b;
  // rtk_n
  output->rtk_n = input->rtk_n;
  // rtk_e
  output->rtk_e = input->rtk_e;
  // rtk_d
  output->rtk_d = input->rtk_d;
  // nav_state
  output->nav_state = input->nav_state;
  return true;
}

px4_msgs__msg__Monitoring *
px4_msgs__msg__Monitoring__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__Monitoring * msg = (px4_msgs__msg__Monitoring *)allocator.allocate(sizeof(px4_msgs__msg__Monitoring), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(px4_msgs__msg__Monitoring));
  bool success = px4_msgs__msg__Monitoring__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
px4_msgs__msg__Monitoring__destroy(px4_msgs__msg__Monitoring * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    px4_msgs__msg__Monitoring__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
px4_msgs__msg__Monitoring__Sequence__init(px4_msgs__msg__Monitoring__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__Monitoring * data = NULL;

  if (size) {
    data = (px4_msgs__msg__Monitoring *)allocator.zero_allocate(size, sizeof(px4_msgs__msg__Monitoring), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = px4_msgs__msg__Monitoring__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        px4_msgs__msg__Monitoring__fini(&data[i - 1]);
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
px4_msgs__msg__Monitoring__Sequence__fini(px4_msgs__msg__Monitoring__Sequence * array)
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
      px4_msgs__msg__Monitoring__fini(&array->data[i]);
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

px4_msgs__msg__Monitoring__Sequence *
px4_msgs__msg__Monitoring__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  px4_msgs__msg__Monitoring__Sequence * array = (px4_msgs__msg__Monitoring__Sequence *)allocator.allocate(sizeof(px4_msgs__msg__Monitoring__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = px4_msgs__msg__Monitoring__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
px4_msgs__msg__Monitoring__Sequence__destroy(px4_msgs__msg__Monitoring__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    px4_msgs__msg__Monitoring__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
px4_msgs__msg__Monitoring__Sequence__are_equal(const px4_msgs__msg__Monitoring__Sequence * lhs, const px4_msgs__msg__Monitoring__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!px4_msgs__msg__Monitoring__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
px4_msgs__msg__Monitoring__Sequence__copy(
  const px4_msgs__msg__Monitoring__Sequence * input,
  px4_msgs__msg__Monitoring__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(px4_msgs__msg__Monitoring);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    px4_msgs__msg__Monitoring * data =
      (px4_msgs__msg__Monitoring *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!px4_msgs__msg__Monitoring__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          px4_msgs__msg__Monitoring__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!px4_msgs__msg__Monitoring__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
