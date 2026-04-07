# generated from rosidl_generator_py/resource/_idl.py.em
# with input from suicide_drone_msgs:msg/IBVSOutput.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_IBVSOutput(type):
    """Metaclass of message 'IBVSOutput'."""

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
                'suicide_drone_msgs.msg.IBVSOutput')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__ibvs_output
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__ibvs_output
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__ibvs_output
            cls._TYPE_SUPPORT = module.type_support_msg__msg__ibvs_output
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__ibvs_output

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


class IBVSOutput(metaclass=Metaclass_IBVSOutput):
    """Message class 'IBVSOutput'."""

    __slots__ = [
        '_header',
        '_detected',
        '_q_y',
        '_q_z',
        '_fov_yaw_rate',
        '_fov_vel_z',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'detected': 'boolean',
        'q_y': 'double',
        'q_z': 'double',
        'fov_yaw_rate': 'double',
        'fov_vel_z': 'double',
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
        self.detected = kwargs.get('detected', bool())
        self.q_y = kwargs.get('q_y', float())
        self.q_z = kwargs.get('q_z', float())
        self.fov_yaw_rate = kwargs.get('fov_yaw_rate', float())
        self.fov_vel_z = kwargs.get('fov_vel_z', float())

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
        if self.detected != other.detected:
            return False
        if self.q_y != other.q_y:
            return False
        if self.q_z != other.q_z:
            return False
        if self.fov_yaw_rate != other.fov_yaw_rate:
            return False
        if self.fov_vel_z != other.fov_vel_z:
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
    def detected(self):
        """Message field 'detected'."""
        return self._detected

    @detected.setter
    def detected(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'detected' field must be of type 'bool'"
        self._detected = value

    @builtins.property
    def q_y(self):
        """Message field 'q_y'."""
        return self._q_y

    @q_y.setter
    def q_y(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'q_y' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'q_y' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._q_y = value

    @builtins.property
    def q_z(self):
        """Message field 'q_z'."""
        return self._q_z

    @q_z.setter
    def q_z(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'q_z' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'q_z' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._q_z = value

    @builtins.property
    def fov_yaw_rate(self):
        """Message field 'fov_yaw_rate'."""
        return self._fov_yaw_rate

    @fov_yaw_rate.setter
    def fov_yaw_rate(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'fov_yaw_rate' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'fov_yaw_rate' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._fov_yaw_rate = value

    @builtins.property
    def fov_vel_z(self):
        """Message field 'fov_vel_z'."""
        return self._fov_vel_z

    @fov_vel_z.setter
    def fov_vel_z(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'fov_vel_z' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'fov_vel_z' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._fov_vel_z = value
