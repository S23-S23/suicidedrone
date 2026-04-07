# generated from rosidl_generator_py/resource/_idl.py.em
# with input from px4_msgs:msg/ScenarioCommand.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'param5'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ScenarioCommand(type):
    """Metaclass of message 'ScenarioCommand'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'SCENARIO_CMD_SET_START_TIME': 0,
        'SCENARIO_CMD_STOP_SCENARIO': 1,
        'SCENARIO_CMD_EMERGENCY_LAND': 2,
        'SCENARIO_CMD_SET_CONFIGS': 3,
        'SCENARIO_CMD_RESET_CONFIGS': 4,
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
                'px4_msgs.msg.ScenarioCommand')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__scenario_command
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__scenario_command
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__scenario_command
            cls._TYPE_SUPPORT = module.type_support_msg__msg__scenario_command
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__scenario_command

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'SCENARIO_CMD_SET_START_TIME': cls.__constants['SCENARIO_CMD_SET_START_TIME'],
            'SCENARIO_CMD_STOP_SCENARIO': cls.__constants['SCENARIO_CMD_STOP_SCENARIO'],
            'SCENARIO_CMD_EMERGENCY_LAND': cls.__constants['SCENARIO_CMD_EMERGENCY_LAND'],
            'SCENARIO_CMD_SET_CONFIGS': cls.__constants['SCENARIO_CMD_SET_CONFIGS'],
            'SCENARIO_CMD_RESET_CONFIGS': cls.__constants['SCENARIO_CMD_RESET_CONFIGS'],
        }

    @property
    def SCENARIO_CMD_SET_START_TIME(self):
        """Message constant 'SCENARIO_CMD_SET_START_TIME'."""
        return Metaclass_ScenarioCommand.__constants['SCENARIO_CMD_SET_START_TIME']

    @property
    def SCENARIO_CMD_STOP_SCENARIO(self):
        """Message constant 'SCENARIO_CMD_STOP_SCENARIO'."""
        return Metaclass_ScenarioCommand.__constants['SCENARIO_CMD_STOP_SCENARIO']

    @property
    def SCENARIO_CMD_EMERGENCY_LAND(self):
        """Message constant 'SCENARIO_CMD_EMERGENCY_LAND'."""
        return Metaclass_ScenarioCommand.__constants['SCENARIO_CMD_EMERGENCY_LAND']

    @property
    def SCENARIO_CMD_SET_CONFIGS(self):
        """Message constant 'SCENARIO_CMD_SET_CONFIGS'."""
        return Metaclass_ScenarioCommand.__constants['SCENARIO_CMD_SET_CONFIGS']

    @property
    def SCENARIO_CMD_RESET_CONFIGS(self):
        """Message constant 'SCENARIO_CMD_RESET_CONFIGS'."""
        return Metaclass_ScenarioCommand.__constants['SCENARIO_CMD_RESET_CONFIGS']


class ScenarioCommand(metaclass=Metaclass_ScenarioCommand):
    """
    Message class 'ScenarioCommand'.

    Constants:
      SCENARIO_CMD_SET_START_TIME
      SCENARIO_CMD_STOP_SCENARIO
      SCENARIO_CMD_EMERGENCY_LAND
      SCENARIO_CMD_SET_CONFIGS
      SCENARIO_CMD_RESET_CONFIGS
    """

    __slots__ = [
        '_timestamp',
        '_cmd',
        '_param1',
        '_param2',
        '_param3',
        '_param4',
        '_param5',
    ]

    _fields_and_field_types = {
        'timestamp': 'uint64',
        'cmd': 'uint8',
        'param1': 'float',
        'param2': 'float',
        'param3': 'float',
        'param4': 'uint32',
        'param5': 'uint8[32]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 32),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.timestamp = kwargs.get('timestamp', int())
        self.cmd = kwargs.get('cmd', int())
        self.param1 = kwargs.get('param1', float())
        self.param2 = kwargs.get('param2', float())
        self.param3 = kwargs.get('param3', float())
        self.param4 = kwargs.get('param4', int())
        if 'param5' not in kwargs:
            self.param5 = numpy.zeros(32, dtype=numpy.uint8)
        else:
            self.param5 = kwargs.get('param5')

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
        if self.cmd != other.cmd:
            return False
        if self.param1 != other.param1:
            return False
        if self.param2 != other.param2:
            return False
        if self.param3 != other.param3:
            return False
        if self.param4 != other.param4:
            return False
        if any(self.param5 != other.param5):
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
    def cmd(self):
        """Message field 'cmd'."""
        return self._cmd

    @cmd.setter
    def cmd(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'cmd' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'cmd' field must be an unsigned integer in [0, 255]"
        self._cmd = value

    @builtins.property
    def param1(self):
        """Message field 'param1'."""
        return self._param1

    @param1.setter
    def param1(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'param1' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'param1' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._param1 = value

    @builtins.property
    def param2(self):
        """Message field 'param2'."""
        return self._param2

    @param2.setter
    def param2(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'param2' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'param2' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._param2 = value

    @builtins.property
    def param3(self):
        """Message field 'param3'."""
        return self._param3

    @param3.setter
    def param3(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'param3' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'param3' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._param3 = value

    @builtins.property
    def param4(self):
        """Message field 'param4'."""
        return self._param4

    @param4.setter
    def param4(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'param4' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'param4' field must be an unsigned integer in [0, 4294967295]"
        self._param4 = value

    @builtins.property
    def param5(self):
        """Message field 'param5'."""
        return self._param5

    @param5.setter
    def param5(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'param5' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 32, \
                "The 'param5' numpy.ndarray() must have a size of 32"
            self._param5 = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 32 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'param5' field must be a set or sequence with length 32 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._param5 = numpy.array(value, dtype=numpy.uint8)
