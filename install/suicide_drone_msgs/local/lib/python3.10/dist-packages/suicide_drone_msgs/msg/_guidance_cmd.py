# generated from rosidl_generator_py/resource/_idl.py.em
# with input from suicide_drone_msgs:msg/GuidanceCmd.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_GuidanceCmd(type):
    """Metaclass of message 'GuidanceCmd'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('suicide_drone_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'suicide_drone_msgs.msg.GuidanceCmd')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__guidance_cmd
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__guidance_cmd
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__guidance_cmd
            cls._TYPE_SUPPORT = module.type_support_msg__msg__guidance_cmd
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__guidance_cmd

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class GuidanceCmd(metaclass=Metaclass_GuidanceCmd):
    """Message class 'GuidanceCmd'."""

    __slots__ = [
        '_header',
        '_target_detected',
        '_vel_n',
        '_vel_e',
        '_vel_d',
        '_yaw_rate',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'target_detected': 'boolean',
        'vel_n': 'double',
        'vel_e': 'double',
        'vel_d': 'double',
        'yaw_rate': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.target_detected = kwargs.get('target_detected', bool())
        self.vel_n = kwargs.get('vel_n', float())
        self.vel_e = kwargs.get('vel_e', float())
        self.vel_d = kwargs.get('vel_d', float())
        self.yaw_rate = kwargs.get('yaw_rate', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.target_detected != other.target_detected:
            return False
        if self.vel_n != other.vel_n:
            return False
        if self.vel_e != other.vel_e:
            return False
        if self.vel_d != other.vel_d:
            return False
        if self.yaw_rate != other.yaw_rate:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def target_detected(self):
        """Message field 'target_detected'."""
        return self._target_detected

    @target_detected.setter
    def target_detected(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'target_detected' field must be of type 'bool'"
        self._target_detected = value

    @builtins.property
    def vel_n(self):
        """Message field 'vel_n'."""
        return self._vel_n

    @vel_n.setter
    def vel_n(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'vel_n' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'vel_n' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._vel_n = value

    @builtins.property
    def vel_e(self):
        """Message field 'vel_e'."""
        return self._vel_e

    @vel_e.setter
    def vel_e(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'vel_e' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'vel_e' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._vel_e = value

    @builtins.property
    def vel_d(self):
        """Message field 'vel_d'."""
        return self._vel_d

    @vel_d.setter
    def vel_d(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'vel_d' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'vel_d' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._vel_d = value

    @builtins.property
    def yaw_rate(self):
        """Message field 'yaw_rate'."""
        return self._yaw_rate

    @yaw_rate.setter
    def yaw_rate(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'yaw_rate' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'yaw_rate' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._yaw_rate = value
