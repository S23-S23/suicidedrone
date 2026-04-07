# generated from rosidl_generator_py/resource/_idl.py.em
# with input from px4_msgs:msg/F9pRtk.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_F9pRtk(type):
    """Metaclass of message 'F9pRtk'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'AGE_CORR_UNAVAILABLE': 0,
        'AGE_CORR_0_TO_1_SEC': 1,
        'AGE_CORR_1_TO_2_SEC': 2,
        'AGE_CORR_2_TO_5_SEC': 3,
        'AGE_CORR_5_TO_10_SEC': 4,
        'AGE_CORR_10_TO_15_SEC': 5,
        'AGE_CORR_15_TO_20_SEC': 6,
        'AGE_CORR_20_TO_30_SEC': 7,
        'AGE_CORR_30_TO_45_SEC': 8,
        'AGE_CORR_45_TO_60_SEC': 9,
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
                'px4_msgs.msg.F9pRtk')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__f9p_rtk
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__f9p_rtk
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__f9p_rtk
            cls._TYPE_SUPPORT = module.type_support_msg__msg__f9p_rtk
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__f9p_rtk

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'AGE_CORR_UNAVAILABLE': cls.__constants['AGE_CORR_UNAVAILABLE'],
            'AGE_CORR_0_TO_1_SEC': cls.__constants['AGE_CORR_0_TO_1_SEC'],
            'AGE_CORR_1_TO_2_SEC': cls.__constants['AGE_CORR_1_TO_2_SEC'],
            'AGE_CORR_2_TO_5_SEC': cls.__constants['AGE_CORR_2_TO_5_SEC'],
            'AGE_CORR_5_TO_10_SEC': cls.__constants['AGE_CORR_5_TO_10_SEC'],
            'AGE_CORR_10_TO_15_SEC': cls.__constants['AGE_CORR_10_TO_15_SEC'],
            'AGE_CORR_15_TO_20_SEC': cls.__constants['AGE_CORR_15_TO_20_SEC'],
            'AGE_CORR_20_TO_30_SEC': cls.__constants['AGE_CORR_20_TO_30_SEC'],
            'AGE_CORR_30_TO_45_SEC': cls.__constants['AGE_CORR_30_TO_45_SEC'],
            'AGE_CORR_45_TO_60_SEC': cls.__constants['AGE_CORR_45_TO_60_SEC'],
        }

    @property
    def AGE_CORR_UNAVAILABLE(self):
        """Message constant 'AGE_CORR_UNAVAILABLE'."""
        return Metaclass_F9pRtk.__constants['AGE_CORR_UNAVAILABLE']

    @property
    def AGE_CORR_0_TO_1_SEC(self):
        """Message constant 'AGE_CORR_0_TO_1_SEC'."""
        return Metaclass_F9pRtk.__constants['AGE_CORR_0_TO_1_SEC']

    @property
    def AGE_CORR_1_TO_2_SEC(self):
        """Message constant 'AGE_CORR_1_TO_2_SEC'."""
        return Metaclass_F9pRtk.__constants['AGE_CORR_1_TO_2_SEC']

    @property
    def AGE_CORR_2_TO_5_SEC(self):
        """Message constant 'AGE_CORR_2_TO_5_SEC'."""
        return Metaclass_F9pRtk.__constants['AGE_CORR_2_TO_5_SEC']

    @property
    def AGE_CORR_5_TO_10_SEC(self):
        """Message constant 'AGE_CORR_5_TO_10_SEC'."""
        return Metaclass_F9pRtk.__constants['AGE_CORR_5_TO_10_SEC']

    @property
    def AGE_CORR_10_TO_15_SEC(self):
        """Message constant 'AGE_CORR_10_TO_15_SEC'."""
        return Metaclass_F9pRtk.__constants['AGE_CORR_10_TO_15_SEC']

    @property
    def AGE_CORR_15_TO_20_SEC(self):
        """Message constant 'AGE_CORR_15_TO_20_SEC'."""
        return Metaclass_F9pRtk.__constants['AGE_CORR_15_TO_20_SEC']

    @property
    def AGE_CORR_20_TO_30_SEC(self):
        """Message constant 'AGE_CORR_20_TO_30_SEC'."""
        return Metaclass_F9pRtk.__constants['AGE_CORR_20_TO_30_SEC']

    @property
    def AGE_CORR_30_TO_45_SEC(self):
        """Message constant 'AGE_CORR_30_TO_45_SEC'."""
        return Metaclass_F9pRtk.__constants['AGE_CORR_30_TO_45_SEC']

    @property
    def AGE_CORR_45_TO_60_SEC(self):
        """Message constant 'AGE_CORR_45_TO_60_SEC'."""
        return Metaclass_F9pRtk.__constants['AGE_CORR_45_TO_60_SEC']


class F9pRtk(metaclass=Metaclass_F9pRtk):
    """
    Message class 'F9pRtk'.

    Constants:
      AGE_CORR_UNAVAILABLE
      AGE_CORR_0_TO_1_SEC
      AGE_CORR_1_TO_2_SEC
      AGE_CORR_2_TO_5_SEC
      AGE_CORR_5_TO_10_SEC
      AGE_CORR_10_TO_15_SEC
      AGE_CORR_15_TO_20_SEC
      AGE_CORR_20_TO_30_SEC
      AGE_CORR_30_TO_45_SEC
      AGE_CORR_45_TO_60_SEC
    """

    __slots__ = [
        '_timestamp',
        '_device_id',
        '_tow',
        '_age_corr',
        '_fix_type',
        '_satellites_used',
        '_n',
        '_e',
        '_d',
        '_v_n',
        '_v_e',
        '_v_d',
        '_acc_n',
        '_acc_e',
        '_acc_d',
    ]

    _fields_and_field_types = {
        'timestamp': 'uint64',
        'device_id': 'uint32',
        'tow': 'uint32',
        'age_corr': 'uint8',
        'fix_type': 'uint8',
        'satellites_used': 'uint8',
        'n': 'float',
        'e': 'float',
        'd': 'float',
        'v_n': 'float',
        'v_e': 'float',
        'v_d': 'float',
        'acc_n': 'float',
        'acc_e': 'float',
        'acc_d': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.timestamp = kwargs.get('timestamp', int())
        self.device_id = kwargs.get('device_id', int())
        self.tow = kwargs.get('tow', int())
        self.age_corr = kwargs.get('age_corr', int())
        self.fix_type = kwargs.get('fix_type', int())
        self.satellites_used = kwargs.get('satellites_used', int())
        self.n = kwargs.get('n', float())
        self.e = kwargs.get('e', float())
        self.d = kwargs.get('d', float())
        self.v_n = kwargs.get('v_n', float())
        self.v_e = kwargs.get('v_e', float())
        self.v_d = kwargs.get('v_d', float())
        self.acc_n = kwargs.get('acc_n', float())
        self.acc_e = kwargs.get('acc_e', float())
        self.acc_d = kwargs.get('acc_d', float())

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
        if self.device_id != other.device_id:
            return False
        if self.tow != other.tow:
            return False
        if self.age_corr != other.age_corr:
            return False
        if self.fix_type != other.fix_type:
            return False
        if self.satellites_used != other.satellites_used:
            return False
        if self.n != other.n:
            return False
        if self.e != other.e:
            return False
        if self.d != other.d:
            return False
        if self.v_n != other.v_n:
            return False
        if self.v_e != other.v_e:
            return False
        if self.v_d != other.v_d:
            return False
        if self.acc_n != other.acc_n:
            return False
        if self.acc_e != other.acc_e:
            return False
        if self.acc_d != other.acc_d:
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
    def device_id(self):
        """Message field 'device_id'."""
        return self._device_id

    @device_id.setter
    def device_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'device_id' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'device_id' field must be an unsigned integer in [0, 4294967295]"
        self._device_id = value

    @builtins.property
    def tow(self):
        """Message field 'tow'."""
        return self._tow

    @tow.setter
    def tow(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'tow' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'tow' field must be an unsigned integer in [0, 4294967295]"
        self._tow = value

    @builtins.property
    def age_corr(self):
        """Message field 'age_corr'."""
        return self._age_corr

    @age_corr.setter
    def age_corr(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'age_corr' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'age_corr' field must be an unsigned integer in [0, 255]"
        self._age_corr = value

    @builtins.property
    def fix_type(self):
        """Message field 'fix_type'."""
        return self._fix_type

    @fix_type.setter
    def fix_type(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'fix_type' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'fix_type' field must be an unsigned integer in [0, 255]"
        self._fix_type = value

    @builtins.property
    def satellites_used(self):
        """Message field 'satellites_used'."""
        return self._satellites_used

    @satellites_used.setter
    def satellites_used(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'satellites_used' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'satellites_used' field must be an unsigned integer in [0, 255]"
        self._satellites_used = value

    @builtins.property
    def n(self):
        """Message field 'n'."""
        return self._n

    @n.setter
    def n(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'n' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'n' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._n = value

    @builtins.property
    def e(self):
        """Message field 'e'."""
        return self._e

    @e.setter
    def e(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'e' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'e' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._e = value

    @builtins.property
    def d(self):
        """Message field 'd'."""
        return self._d

    @d.setter
    def d(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'd' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'd' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._d = value

    @builtins.property
    def v_n(self):
        """Message field 'v_n'."""
        return self._v_n

    @v_n.setter
    def v_n(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'v_n' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'v_n' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._v_n = value

    @builtins.property
    def v_e(self):
        """Message field 'v_e'."""
        return self._v_e

    @v_e.setter
    def v_e(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'v_e' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'v_e' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._v_e = value

    @builtins.property
    def v_d(self):
        """Message field 'v_d'."""
        return self._v_d

    @v_d.setter
    def v_d(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'v_d' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'v_d' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._v_d = value

    @builtins.property
    def acc_n(self):
        """Message field 'acc_n'."""
        return self._acc_n

    @acc_n.setter
    def acc_n(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'acc_n' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'acc_n' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._acc_n = value

    @builtins.property
    def acc_e(self):
        """Message field 'acc_e'."""
        return self._acc_e

    @acc_e.setter
    def acc_e(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'acc_e' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'acc_e' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._acc_e = value

    @builtins.property
    def acc_d(self):
        """Message field 'acc_d'."""
        return self._acc_d

    @acc_d.setter
    def acc_d(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'acc_d' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'acc_d' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._acc_d = value
