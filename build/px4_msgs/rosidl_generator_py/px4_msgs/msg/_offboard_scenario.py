# generated from rosidl_generator_py/resource/_idl.py.em
# with input from px4_msgs:msg/OffboardScenario.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_OffboardScenario(type):
    """Metaclass of message 'OffboardScenario'."""

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
                'px4_msgs.msg.OffboardScenario')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__offboard_scenario
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__offboard_scenario
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__offboard_scenario
            cls._TYPE_SUPPORT = module.type_support_msg__msg__offboard_scenario
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__offboard_scenario

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class OffboardScenario(metaclass=Metaclass_OffboardScenario):
    """Message class 'OffboardScenario'."""

    __slots__ = [
        '_timestamp',
        '_current_time',
        '_start_time',
        '_seq',
        '_offset_x',
        '_offset_y',
        '_ready_sc_file',
    ]

    _fields_and_field_types = {
        'timestamp': 'uint64',
        'current_time': 'uint64',
        'start_time': 'uint64',
        'seq': 'uint32',
        'offset_x': 'float',
        'offset_y': 'float',
        'ready_sc_file': 'uint8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.timestamp = kwargs.get('timestamp', int())
        self.current_time = kwargs.get('current_time', int())
        self.start_time = kwargs.get('start_time', int())
        self.seq = kwargs.get('seq', int())
        self.offset_x = kwargs.get('offset_x', float())
        self.offset_y = kwargs.get('offset_y', float())
        self.ready_sc_file = kwargs.get('ready_sc_file', int())

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
        if self.current_time != other.current_time:
            return False
        if self.start_time != other.start_time:
            return False
        if self.seq != other.seq:
            return False
        if self.offset_x != other.offset_x:
            return False
        if self.offset_y != other.offset_y:
            return False
        if self.ready_sc_file != other.ready_sc_file:
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
    def current_time(self):
        """Message field 'current_time'."""
        return self._current_time

    @current_time.setter
    def current_time(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'current_time' field must be of type 'int'"
            assert value >= 0 and value < 18446744073709551616, \
                "The 'current_time' field must be an unsigned integer in [0, 18446744073709551615]"
        self._current_time = value

    @builtins.property
    def start_time(self):
        """Message field 'start_time'."""
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'start_time' field must be of type 'int'"
            assert value >= 0 and value < 18446744073709551616, \
                "The 'start_time' field must be an unsigned integer in [0, 18446744073709551615]"
        self._start_time = value

    @builtins.property
    def seq(self):
        """Message field 'seq'."""
        return self._seq

    @seq.setter
    def seq(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'seq' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'seq' field must be an unsigned integer in [0, 4294967295]"
        self._seq = value

    @builtins.property
    def offset_x(self):
        """Message field 'offset_x'."""
        return self._offset_x

    @offset_x.setter
    def offset_x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'offset_x' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'offset_x' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._offset_x = value

    @builtins.property
    def offset_y(self):
        """Message field 'offset_y'."""
        return self._offset_y

    @offset_y.setter
    def offset_y(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'offset_y' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'offset_y' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._offset_y = value

    @builtins.property
    def ready_sc_file(self):
        """Message field 'ready_sc_file'."""
        return self._ready_sc_file

    @ready_sc_file.setter
    def ready_sc_file(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'ready_sc_file' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'ready_sc_file' field must be an unsigned integer in [0, 255]"
        self._ready_sc_file = value
