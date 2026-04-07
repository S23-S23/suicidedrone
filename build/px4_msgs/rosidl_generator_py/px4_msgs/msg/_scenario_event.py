# generated from rosidl_generator_py/resource/_idl.py.em
# with input from px4_msgs:msg/ScenarioEvent.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ScenarioEvent(type):
    """Metaclass of message 'ScenarioEvent'."""

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
            module = import_type_support('px4_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'px4_msgs.msg.ScenarioEvent')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__scenario_event
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__scenario_event
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__scenario_event
            cls._TYPE_SUPPORT = module.type_support_msg__msg__scenario_event
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__scenario_event

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class ScenarioEvent(metaclass=Metaclass_ScenarioEvent):
    """Message class 'ScenarioEvent'."""

    __slots__ = [
        '_timestamp',
        '_event_time',
        '_event_type',
        '_cmd_type',
        '_x',
        '_y',
        '_z',
        '_led_r',
        '_led_g',
        '_led_b',
        '_is_scenario_active',
    ]

    _fields_and_field_types = {
        'timestamp': 'uint64',
        'event_time': 'float',
        'event_type': 'uint8',
        'cmd_type': 'uint8',
        'x': 'float',
        'y': 'float',
        'z': 'float',
        'led_r': 'uint8',
        'led_g': 'uint8',
        'led_b': 'uint8',
        'is_scenario_active': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.timestamp = kwargs.get('timestamp', int())
        self.event_time = kwargs.get('event_time', float())
        self.event_type = kwargs.get('event_type', int())
        self.cmd_type = kwargs.get('cmd_type', int())
        self.x = kwargs.get('x', float())
        self.y = kwargs.get('y', float())
        self.z = kwargs.get('z', float())
        self.led_r = kwargs.get('led_r', int())
        self.led_g = kwargs.get('led_g', int())
        self.led_b = kwargs.get('led_b', int())
        self.is_scenario_active = kwargs.get('is_scenario_active', bool())

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
        if self.timestamp != other.timestamp:
            return False
        if self.event_time != other.event_time:
            return False
        if self.event_type != other.event_type:
            return False
        if self.cmd_type != other.cmd_type:
            return False
        if self.x != other.x:
            return False
        if self.y != other.y:
            return False
        if self.z != other.z:
            return False
        if self.led_r != other.led_r:
            return False
        if self.led_g != other.led_g:
            return False
        if self.led_b != other.led_b:
            return False
        if self.is_scenario_active != other.is_scenario_active:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def timestamp(self):
        """Message field 'timestamp'."""
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'timestamp' field must be of type 'int'"
            assert value >= 0 and value < 18446744073709551616, \
                "The 'timestamp' field must be an unsigned integer in [0, 18446744073709551615]"
        self._timestamp = value

    @builtins.property
    def event_time(self):
        """Message field 'event_time'."""
        return self._event_time

    @event_time.setter
    def event_time(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'event_time' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'event_time' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._event_time = value

    @builtins.property
    def event_type(self):
        """Message field 'event_type'."""
        return self._event_type

    @event_type.setter
    def event_type(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'event_type' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'event_type' field must be an unsigned integer in [0, 255]"
        self._event_type = value

    @builtins.property
    def cmd_type(self):
        """Message field 'cmd_type'."""
        return self._cmd_type

    @cmd_type.setter
    def cmd_type(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'cmd_type' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'cmd_type' field must be an unsigned integer in [0, 255]"
        self._cmd_type = value

    @builtins.property
    def x(self):
        """Message field 'x'."""
        return self._x

    @x.setter
    def x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'x' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'x' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._x = value

    @builtins.property
    def y(self):
        """Message field 'y'."""
        return self._y

    @y.setter
    def y(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'y' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'y' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._y = value

    @builtins.property
    def z(self):
        """Message field 'z'."""
        return self._z

    @z.setter
    def z(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'z' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'z' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._z = value

    @builtins.property
    def led_r(self):
        """Message field 'led_r'."""
        return self._led_r

    @led_r.setter
    def led_r(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'led_r' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'led_r' field must be an unsigned integer in [0, 255]"
        self._led_r = value

    @builtins.property
    def led_g(self):
        """Message field 'led_g'."""
        return self._led_g

    @led_g.setter
    def led_g(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'led_g' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'led_g' field must be an unsigned integer in [0, 255]"
        self._led_g = value

    @builtins.property
    def led_b(self):
        """Message field 'led_b'."""
        return self._led_b

    @led_b.setter
    def led_b(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'led_b' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'led_b' field must be an unsigned integer in [0, 255]"
        self._led_b = value

    @builtins.property
    def is_scenario_active(self):
        """Message field 'is_scenario_active'."""
        return self._is_scenario_active

    @is_scenario_active.setter
    def is_scenario_active(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_scenario_active' field must be of type 'bool'"
        self._is_scenario_active = value
