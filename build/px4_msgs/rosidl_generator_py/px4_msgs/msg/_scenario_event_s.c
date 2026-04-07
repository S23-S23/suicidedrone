// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from px4_msgs:msg/ScenarioEvent.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "px4_msgs/msg/detail/scenario_event__struct.h"
#include "px4_msgs/msg/detail/scenario_event__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool px4_msgs__msg__scenario_event__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[43];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("px4_msgs.msg._scenario_event.ScenarioEvent", full_classname_dest, 42) == 0);
  }
  px4_msgs__msg__ScenarioEvent * ros_message = _ros_message;
  {  // timestamp
    PyObject * field = PyObject_GetAttrString(_pymsg, "timestamp");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->timestamp = PyLong_AsUnsignedLongLong(field);
    Py_DECREF(field);
  }
  {  // event_time
    PyObject * field = PyObject_GetAttrString(_pymsg, "event_time");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->event_time = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // event_type
    PyObject * field = PyObject_GetAttrString(_pymsg, "event_type");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->event_type = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // cmd_type
    PyObject * field = PyObject_GetAttrString(_pymsg, "cmd_type");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->cmd_type = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // x
    PyObject * field = PyObject_GetAttrString(_pymsg, "x");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->x = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // y
    PyObject * field = PyObject_GetAttrString(_pymsg, "y");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->y = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // z
    PyObject * field = PyObject_GetAttrString(_pymsg, "z");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->z = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // led_r
    PyObject * field = PyObject_GetAttrString(_pymsg, "led_r");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->led_r = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // led_g
    PyObject * field = PyObject_GetAttrString(_pymsg, "led_g");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->led_g = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // led_b
    PyObject * field = PyObject_GetAttrString(_pymsg, "led_b");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->led_b = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // is_scenario_active
    PyObject * field = PyObject_GetAttrString(_pymsg, "is_scenario_active");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->is_scenario_active = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * px4_msgs__msg__scenario_event__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of ScenarioEvent */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("px4_msgs.msg._scenario_event");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "ScenarioEvent");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  px4_msgs__msg__ScenarioEvent * ros_message = (px4_msgs__msg__ScenarioEvent *)raw_ros_message;
  {  // timestamp
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLongLong(ros_message->timestamp);
    {
      int rc = PyObject_SetAttrString(_pymessage, "timestamp", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // event_time
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->event_time);
    {
      int rc = PyObject_SetAttrString(_pymessage, "event_time", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // event_type
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->event_type);
    {
      int rc = PyObject_SetAttrString(_pymessage, "event_type", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cmd_type
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->cmd_type);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cmd_type", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // x
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->x);
    {
      int rc = PyObject_SetAttrString(_pymessage, "x", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // y
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->y);
    {
      int rc = PyObject_SetAttrString(_pymessage, "y", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // z
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->z);
    {
      int rc = PyObject_SetAttrString(_pymessage, "z", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // led_r
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->led_r);
    {
      int rc = PyObject_SetAttrString(_pymessage, "led_r", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // led_g
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->led_g);
    {
      int rc = PyObject_SetAttrString(_pymessage, "led_g", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // led_b
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->led_b);
    {
      int rc = PyObject_SetAttrString(_pymessage, "led_b", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // is_scenario_active
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->is_scenario_active ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "is_scenario_active", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
