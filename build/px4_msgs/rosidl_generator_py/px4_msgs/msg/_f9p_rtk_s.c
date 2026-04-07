// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from px4_msgs:msg/F9pRtk.idl
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
#include "px4_msgs/msg/detail/f9p_rtk__struct.h"
#include "px4_msgs/msg/detail/f9p_rtk__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool px4_msgs__msg__f9p_rtk__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[29];
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
    assert(strncmp("px4_msgs.msg._f9p_rtk.F9pRtk", full_classname_dest, 28) == 0);
  }
  px4_msgs__msg__F9pRtk * ros_message = _ros_message;
  {  // timestamp
    PyObject * field = PyObject_GetAttrString(_pymsg, "timestamp");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->timestamp = PyLong_AsUnsignedLongLong(field);
    Py_DECREF(field);
  }
  {  // device_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "device_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->device_id = PyLong_AsUnsignedLong(field);
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
  {  // age_corr
    PyObject * field = PyObject_GetAttrString(_pymsg, "age_corr");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->age_corr = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // fix_type
    PyObject * field = PyObject_GetAttrString(_pymsg, "fix_type");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->fix_type = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // satellites_used
    PyObject * field = PyObject_GetAttrString(_pymsg, "satellites_used");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->satellites_used = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // n
    PyObject * field = PyObject_GetAttrString(_pymsg, "n");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->n = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // e
    PyObject * field = PyObject_GetAttrString(_pymsg, "e");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->e = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // d
    PyObject * field = PyObject_GetAttrString(_pymsg, "d");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->d = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // v_n
    PyObject * field = PyObject_GetAttrString(_pymsg, "v_n");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->v_n = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // v_e
    PyObject * field = PyObject_GetAttrString(_pymsg, "v_e");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->v_e = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // v_d
    PyObject * field = PyObject_GetAttrString(_pymsg, "v_d");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->v_d = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // acc_n
    PyObject * field = PyObject_GetAttrString(_pymsg, "acc_n");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->acc_n = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // acc_e
    PyObject * field = PyObject_GetAttrString(_pymsg, "acc_e");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->acc_e = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // acc_d
    PyObject * field = PyObject_GetAttrString(_pymsg, "acc_d");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->acc_d = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * px4_msgs__msg__f9p_rtk__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of F9pRtk */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("px4_msgs.msg._f9p_rtk");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "F9pRtk");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  px4_msgs__msg__F9pRtk * ros_message = (px4_msgs__msg__F9pRtk *)raw_ros_message;
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
  {  // device_id
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->device_id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "device_id", field);
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
  {  // age_corr
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->age_corr);
    {
      int rc = PyObject_SetAttrString(_pymessage, "age_corr", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // fix_type
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->fix_type);
    {
      int rc = PyObject_SetAttrString(_pymessage, "fix_type", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // satellites_used
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->satellites_used);
    {
      int rc = PyObject_SetAttrString(_pymessage, "satellites_used", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // n
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->n);
    {
      int rc = PyObject_SetAttrString(_pymessage, "n", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // e
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->e);
    {
      int rc = PyObject_SetAttrString(_pymessage, "e", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // d
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->d);
    {
      int rc = PyObject_SetAttrString(_pymessage, "d", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // v_n
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->v_n);
    {
      int rc = PyObject_SetAttrString(_pymessage, "v_n", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // v_e
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->v_e);
    {
      int rc = PyObject_SetAttrString(_pymessage, "v_e", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // v_d
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->v_d);
    {
      int rc = PyObject_SetAttrString(_pymessage, "v_d", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // acc_n
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->acc_n);
    {
      int rc = PyObject_SetAttrString(_pymessage, "acc_n", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // acc_e
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->acc_e);
    {
      int rc = PyObject_SetAttrString(_pymessage, "acc_e", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // acc_d
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->acc_d);
    {
      int rc = PyObject_SetAttrString(_pymessage, "acc_d", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
