// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from px4_msgs:msg/Monitoring.idl
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
#include "px4_msgs/msg/detail/monitoring__struct.h"
#include "px4_msgs/msg/detail/monitoring__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool px4_msgs__msg__monitoring__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[36];
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
    assert(strncmp("px4_msgs.msg._monitoring.Monitoring", full_classname_dest, 35) == 0);
  }
  px4_msgs__msg__Monitoring * ros_message = _ros_message;
  {  // timestamp
    PyObject * field = PyObject_GetAttrString(_pymsg, "timestamp");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->timestamp = PyLong_AsUnsignedLongLong(field);
    Py_DECREF(field);
  }
  {  // tow
    PyObject * field = PyObject_GetAttrString(_pymsg, "tow");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->tow = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // pos_x
    PyObject * field = PyObject_GetAttrString(_pymsg, "pos_x");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->pos_x = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // pos_y
    PyObject * field = PyObject_GetAttrString(_pymsg, "pos_y");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->pos_y = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // pos_z
    PyObject * field = PyObject_GetAttrString(_pymsg, "pos_z");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->pos_z = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // lat
    PyObject * field = PyObject_GetAttrString(_pymsg, "lat");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->lat = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // lon
    PyObject * field = PyObject_GetAttrString(_pymsg, "lon");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->lon = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // alt
    PyObject * field = PyObject_GetAttrString(_pymsg, "alt");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->alt = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ref_lat
    PyObject * field = PyObject_GetAttrString(_pymsg, "ref_lat");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ref_lat = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ref_lon
    PyObject * field = PyObject_GetAttrString(_pymsg, "ref_lon");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ref_lon = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ref_alt
    PyObject * field = PyObject_GetAttrString(_pymsg, "ref_alt");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ref_alt = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // head
    PyObject * field = PyObject_GetAttrString(_pymsg, "head");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->head = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // roll
    PyObject * field = PyObject_GetAttrString(_pymsg, "roll");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->roll = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // pitch
    PyObject * field = PyObject_GetAttrString(_pymsg, "pitch");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->pitch = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // status1
    PyObject * field = PyObject_GetAttrString(_pymsg, "status1");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->status1 = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // status2
    PyObject * field = PyObject_GetAttrString(_pymsg, "status2");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->status2 = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // rtk_nbase
    PyObject * field = PyObject_GetAttrString(_pymsg, "rtk_nbase");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->rtk_nbase = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // rtk_nrover
    PyObject * field = PyObject_GetAttrString(_pymsg, "rtk_nrover");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->rtk_nrover = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // battery
    PyObject * field = PyObject_GetAttrString(_pymsg, "battery");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->battery = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // r
    PyObject * field = PyObject_GetAttrString(_pymsg, "r");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->r = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // g
    PyObject * field = PyObject_GetAttrString(_pymsg, "g");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->g = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // b
    PyObject * field = PyObject_GetAttrString(_pymsg, "b");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->b = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // rtk_n
    PyObject * field = PyObject_GetAttrString(_pymsg, "rtk_n");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->rtk_n = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // rtk_e
    PyObject * field = PyObject_GetAttrString(_pymsg, "rtk_e");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->rtk_e = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // rtk_d
    PyObject * field = PyObject_GetAttrString(_pymsg, "rtk_d");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->rtk_d = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // nav_state
    PyObject * field = PyObject_GetAttrString(_pymsg, "nav_state");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->nav_state = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * px4_msgs__msg__monitoring__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of Monitoring */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("px4_msgs.msg._monitoring");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "Monitoring");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  px4_msgs__msg__Monitoring * ros_message = (px4_msgs__msg__Monitoring *)raw_ros_message;
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
  {  // tow
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->tow);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tow", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pos_x
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->pos_x);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pos_x", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pos_y
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->pos_y);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pos_y", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pos_z
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->pos_z);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pos_z", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // lat
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->lat);
    {
      int rc = PyObject_SetAttrString(_pymessage, "lat", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // lon
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->lon);
    {
      int rc = PyObject_SetAttrString(_pymessage, "lon", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // alt
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->alt);
    {
      int rc = PyObject_SetAttrString(_pymessage, "alt", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ref_lat
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ref_lat);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ref_lat", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ref_lon
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ref_lon);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ref_lon", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ref_alt
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ref_alt);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ref_alt", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // head
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->head);
    {
      int rc = PyObject_SetAttrString(_pymessage, "head", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // roll
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->roll);
    {
      int rc = PyObject_SetAttrString(_pymessage, "roll", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pitch
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->pitch);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pitch", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // status1
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->status1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "status1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // status2
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->status2);
    {
      int rc = PyObject_SetAttrString(_pymessage, "status2", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rtk_nbase
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->rtk_nbase);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rtk_nbase", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rtk_nrover
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->rtk_nrover);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rtk_nrover", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // battery
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->battery);
    {
      int rc = PyObject_SetAttrString(_pymessage, "battery", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // r
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->r);
    {
      int rc = PyObject_SetAttrString(_pymessage, "r", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // g
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->g);
    {
      int rc = PyObject_SetAttrString(_pymessage, "g", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // b
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->b);
    {
      int rc = PyObject_SetAttrString(_pymessage, "b", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rtk_n
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->rtk_n);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rtk_n", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rtk_e
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->rtk_e);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rtk_e", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rtk_d
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->rtk_d);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rtk_d", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // nav_state
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->nav_state);
    {
      int rc = PyObject_SetAttrString(_pymessage, "nav_state", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
