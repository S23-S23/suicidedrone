# generated from rosidl_generator_py/resource/_idl.py.em
# with input from px4_msgs:msg/Monitoring.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Monitoring(type):
    """Metaclass of message 'Monitoring'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'NAVIGATION_STATE_MANUAL': 0,
        'NAVIGATION_STATE_ALTCTL': 1,
        'NAVIGATION_STATE_POSCTL': 2,
        'NAVIGATION_STATE_AUTO_MISSION': 3,
        'NAVIGATION_STATE_AUTO_LOITER': 4,
        'NAVIGATION_STATE_AUTO_RTL': 5,
        'NAVIGATION_STATE_UNUSED3': 8,
        'NAVIGATION_STATE_UNUSED': 9,
        'NAVIGATION_STATE_ACRO': 10,
        'NAVIGATION_STATE_UNUSED1': 11,
        'NAVIGATION_STATE_DESCEND': 12,
        'NAVIGATION_STATE_TERMINATION': 13,
        'NAVIGATION_STATE_OFFBOARD': 14,
        'NAVIGATION_STATE_STAB': 15,
        'NAVIGATION_STATE_UNUSED2': 16,
        'NAVIGATION_STATE_AUTO_TAKEOFF': 17,
        'NAVIGATION_STATE_AUTO_LAND': 18,
        'NAVIGATION_STATE_AUTO_FOLLOW_TARGET': 19,
        'NAVIGATION_STATE_AUTO_PRECLAND': 20,
        'NAVIGATION_STATE_ORBIT': 21,
        'NAVIGATION_STATE_AUTO_VTOL_TAKEOFF': 22,
        'NAVIGATION_STATE_MAX': 23,
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
                'px4_msgs.msg.Monitoring')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__monitoring
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__monitoring
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__monitoring
            cls._TYPE_SUPPORT = module.type_support_msg__msg__monitoring
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__monitoring

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'NAVIGATION_STATE_MANUAL': cls.__constants['NAVIGATION_STATE_MANUAL'],
            'NAVIGATION_STATE_ALTCTL': cls.__constants['NAVIGATION_STATE_ALTCTL'],
            'NAVIGATION_STATE_POSCTL': cls.__constants['NAVIGATION_STATE_POSCTL'],
            'NAVIGATION_STATE_AUTO_MISSION': cls.__constants['NAVIGATION_STATE_AUTO_MISSION'],
            'NAVIGATION_STATE_AUTO_LOITER': cls.__constants['NAVIGATION_STATE_AUTO_LOITER'],
            'NAVIGATION_STATE_AUTO_RTL': cls.__constants['NAVIGATION_STATE_AUTO_RTL'],
            'NAVIGATION_STATE_UNUSED3': cls.__constants['NAVIGATION_STATE_UNUSED3'],
            'NAVIGATION_STATE_UNUSED': cls.__constants['NAVIGATION_STATE_UNUSED'],
            'NAVIGATION_STATE_ACRO': cls.__constants['NAVIGATION_STATE_ACRO'],
            'NAVIGATION_STATE_UNUSED1': cls.__constants['NAVIGATION_STATE_UNUSED1'],
            'NAVIGATION_STATE_DESCEND': cls.__constants['NAVIGATION_STATE_DESCEND'],
            'NAVIGATION_STATE_TERMINATION': cls.__constants['NAVIGATION_STATE_TERMINATION'],
            'NAVIGATION_STATE_OFFBOARD': cls.__constants['NAVIGATION_STATE_OFFBOARD'],
            'NAVIGATION_STATE_STAB': cls.__constants['NAVIGATION_STATE_STAB'],
            'NAVIGATION_STATE_UNUSED2': cls.__constants['NAVIGATION_STATE_UNUSED2'],
            'NAVIGATION_STATE_AUTO_TAKEOFF': cls.__constants['NAVIGATION_STATE_AUTO_TAKEOFF'],
            'NAVIGATION_STATE_AUTO_LAND': cls.__constants['NAVIGATION_STATE_AUTO_LAND'],
            'NAVIGATION_STATE_AUTO_FOLLOW_TARGET': cls.__constants['NAVIGATION_STATE_AUTO_FOLLOW_TARGET'],
            'NAVIGATION_STATE_AUTO_PRECLAND': cls.__constants['NAVIGATION_STATE_AUTO_PRECLAND'],
            'NAVIGATION_STATE_ORBIT': cls.__constants['NAVIGATION_STATE_ORBIT'],
            'NAVIGATION_STATE_AUTO_VTOL_TAKEOFF': cls.__constants['NAVIGATION_STATE_AUTO_VTOL_TAKEOFF'],
            'NAVIGATION_STATE_MAX': cls.__constants['NAVIGATION_STATE_MAX'],
        }

    @property
    def NAVIGATION_STATE_MANUAL(self):
        """Message constant 'NAVIGATION_STATE_MANUAL'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_MANUAL']

    @property
    def NAVIGATION_STATE_ALTCTL(self):
        """Message constant 'NAVIGATION_STATE_ALTCTL'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_ALTCTL']

    @property
    def NAVIGATION_STATE_POSCTL(self):
        """Message constant 'NAVIGATION_STATE_POSCTL'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_POSCTL']

    @property
    def NAVIGATION_STATE_AUTO_MISSION(self):
        """Message constant 'NAVIGATION_STATE_AUTO_MISSION'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_AUTO_MISSION']

    @property
    def NAVIGATION_STATE_AUTO_LOITER(self):
        """Message constant 'NAVIGATION_STATE_AUTO_LOITER'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_AUTO_LOITER']

    @property
    def NAVIGATION_STATE_AUTO_RTL(self):
        """Message constant 'NAVIGATION_STATE_AUTO_RTL'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_AUTO_RTL']

    @property
    def NAVIGATION_STATE_UNUSED3(self):
        """Message constant 'NAVIGATION_STATE_UNUSED3'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_UNUSED3']

    @property
    def NAVIGATION_STATE_UNUSED(self):
        """Message constant 'NAVIGATION_STATE_UNUSED'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_UNUSED']

    @property
    def NAVIGATION_STATE_ACRO(self):
        """Message constant 'NAVIGATION_STATE_ACRO'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_ACRO']

    @property
    def NAVIGATION_STATE_UNUSED1(self):
        """Message constant 'NAVIGATION_STATE_UNUSED1'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_UNUSED1']

    @property
    def NAVIGATION_STATE_DESCEND(self):
        """Message constant 'NAVIGATION_STATE_DESCEND'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_DESCEND']

    @property
    def NAVIGATION_STATE_TERMINATION(self):
        """Message constant 'NAVIGATION_STATE_TERMINATION'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_TERMINATION']

    @property
    def NAVIGATION_STATE_OFFBOARD(self):
        """Message constant 'NAVIGATION_STATE_OFFBOARD'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_OFFBOARD']

    @property
    def NAVIGATION_STATE_STAB(self):
        """Message constant 'NAVIGATION_STATE_STAB'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_STAB']

    @property
    def NAVIGATION_STATE_UNUSED2(self):
        """Message constant 'NAVIGATION_STATE_UNUSED2'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_UNUSED2']

    @property
    def NAVIGATION_STATE_AUTO_TAKEOFF(self):
        """Message constant 'NAVIGATION_STATE_AUTO_TAKEOFF'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_AUTO_TAKEOFF']

    @property
    def NAVIGATION_STATE_AUTO_LAND(self):
        """Message constant 'NAVIGATION_STATE_AUTO_LAND'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_AUTO_LAND']

    @property
    def NAVIGATION_STATE_AUTO_FOLLOW_TARGET(self):
        """Message constant 'NAVIGATION_STATE_AUTO_FOLLOW_TARGET'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_AUTO_FOLLOW_TARGET']

    @property
    def NAVIGATION_STATE_AUTO_PRECLAND(self):
        """Message constant 'NAVIGATION_STATE_AUTO_PRECLAND'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_AUTO_PRECLAND']

    @property
    def NAVIGATION_STATE_ORBIT(self):
        """Message constant 'NAVIGATION_STATE_ORBIT'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_ORBIT']

    @property
    def NAVIGATION_STATE_AUTO_VTOL_TAKEOFF(self):
        """Message constant 'NAVIGATION_STATE_AUTO_VTOL_TAKEOFF'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_AUTO_VTOL_TAKEOFF']

    @property
    def NAVIGATION_STATE_MAX(self):
        """Message constant 'NAVIGATION_STATE_MAX'."""
        return Metaclass_Monitoring.__constants['NAVIGATION_STATE_MAX']


class Monitoring(metaclass=Metaclass_Monitoring):
    """
    Message class 'Monitoring'.

    Constants:
      NAVIGATION_STATE_MANUAL
      NAVIGATION_STATE_ALTCTL
      NAVIGATION_STATE_POSCTL
      NAVIGATION_STATE_AUTO_MISSION
      NAVIGATION_STATE_AUTO_LOITER
      NAVIGATION_STATE_AUTO_RTL
      NAVIGATION_STATE_UNUSED3
      NAVIGATION_STATE_UNUSED
      NAVIGATION_STATE_ACRO
      NAVIGATION_STATE_UNUSED1
      NAVIGATION_STATE_DESCEND
      NAVIGATION_STATE_TERMINATION
      NAVIGATION_STATE_OFFBOARD
      NAVIGATION_STATE_STAB
      NAVIGATION_STATE_UNUSED2
      NAVIGATION_STATE_AUTO_TAKEOFF
      NAVIGATION_STATE_AUTO_LAND
      NAVIGATION_STATE_AUTO_FOLLOW_TARGET
      NAVIGATION_STATE_AUTO_PRECLAND
      NAVIGATION_STATE_ORBIT
      NAVIGATION_STATE_AUTO_VTOL_TAKEOFF
      NAVIGATION_STATE_MAX
    """

    __slots__ = [
        '_timestamp',
        '_tow',
        '_pos_x',
        '_pos_y',
        '_pos_z',
        '_lat',
        '_lon',
        '_alt',
        '_ref_lat',
        '_ref_lon',
        '_ref_alt',
        '_head',
        '_roll',
        '_pitch',
        '_status1',
        '_status2',
        '_rtk_nbase',
        '_rtk_nrover',
        '_battery',
        '_r',
        '_g',
        '_b',
        '_rtk_n',
        '_rtk_e',
        '_rtk_d',
        '_nav_state',
    ]

    _fields_and_field_types = {
        'timestamp': 'uint64',
        'tow': 'uint32',
        'pos_x': 'float',
        'pos_y': 'float',
        'pos_z': 'float',
        'lat': 'double',
        'lon': 'double',
        'alt': 'float',
        'ref_lat': 'double',
        'ref_lon': 'double',
        'ref_alt': 'float',
        'head': 'float',
        'roll': 'float',
        'pitch': 'float',
        'status1': 'uint32',
        'status2': 'uint32',
        'rtk_nbase': 'uint8',
        'rtk_nrover': 'uint8',
        'battery': 'uint8',
        'r': 'uint8',
        'g': 'uint8',
        'b': 'uint8',
        'rtk_n': 'float',
        'rtk_e': 'float',
        'rtk_d': 'float',
        'nav_state': 'uint8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.timestamp = kwargs.get('timestamp', int())
        self.tow = kwargs.get('tow', int())
        self.pos_x = kwargs.get('pos_x', float())
        self.pos_y = kwargs.get('pos_y', float())
        self.pos_z = kwargs.get('pos_z', float())
        self.lat = kwargs.get('lat', float())
        self.lon = kwargs.get('lon', float())
        self.alt = kwargs.get('alt', float())
        self.ref_lat = kwargs.get('ref_lat', float())
        self.ref_lon = kwargs.get('ref_lon', float())
        self.ref_alt = kwargs.get('ref_alt', float())
        self.head = kwargs.get('head', float())
        self.roll = kwargs.get('roll', float())
        self.pitch = kwargs.get('pitch', float())
        self.status1 = kwargs.get('status1', int())
        self.status2 = kwargs.get('status2', int())
        self.rtk_nbase = kwargs.get('rtk_nbase', int())
        self.rtk_nrover = kwargs.get('rtk_nrover', int())
        self.battery = kwargs.get('battery', int())
        self.r = kwargs.get('r', int())
        self.g = kwargs.get('g', int())
        self.b = kwargs.get('b', int())
        self.rtk_n = kwargs.get('rtk_n', float())
        self.rtk_e = kwargs.get('rtk_e', float())
        self.rtk_d = kwargs.get('rtk_d', float())
        self.nav_state = kwargs.get('nav_state', int())

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
        if self.tow != other.tow:
            return False
        if self.pos_x != other.pos_x:
            return False
        if self.pos_y != other.pos_y:
            return False
        if self.pos_z != other.pos_z:
            return False
        if self.lat != other.lat:
            return False
        if self.lon != other.lon:
            return False
        if self.alt != other.alt:
            return False
        if self.ref_lat != other.ref_lat:
            return False
        if self.ref_lon != other.ref_lon:
            return False
        if self.ref_alt != other.ref_alt:
            return False
        if self.head != other.head:
            return False
        if self.roll != other.roll:
            return False
        if self.pitch != other.pitch:
            return False
        if self.status1 != other.status1:
            return False
        if self.status2 != other.status2:
            return False
        if self.rtk_nbase != other.rtk_nbase:
            return False
        if self.rtk_nrover != other.rtk_nrover:
            return False
        if self.battery != other.battery:
            return False
        if self.r != other.r:
            return False
        if self.g != other.g:
            return False
        if self.b != other.b:
            return False
        if self.rtk_n != other.rtk_n:
            return False
        if self.rtk_e != other.rtk_e:
            return False
        if self.rtk_d != other.rtk_d:
            return False
        if self.nav_state != other.nav_state:
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
    def pos_x(self):
        """Message field 'pos_x'."""
        return self._pos_x

    @pos_x.setter
    def pos_x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'pos_x' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'pos_x' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._pos_x = value

    @builtins.property
    def pos_y(self):
        """Message field 'pos_y'."""
        return self._pos_y

    @pos_y.setter
    def pos_y(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'pos_y' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'pos_y' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._pos_y = value

    @builtins.property
    def pos_z(self):
        """Message field 'pos_z'."""
        return self._pos_z

    @pos_z.setter
    def pos_z(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'pos_z' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'pos_z' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._pos_z = value

    @builtins.property
    def lat(self):
        """Message field 'lat'."""
        return self._lat

    @lat.setter
    def lat(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'lat' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'lat' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._lat = value

    @builtins.property
    def lon(self):
        """Message field 'lon'."""
        return self._lon

    @lon.setter
    def lon(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'lon' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'lon' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._lon = value

    @builtins.property
    def alt(self):
        """Message field 'alt'."""
        return self._alt

    @alt.setter
    def alt(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'alt' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'alt' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._alt = value

    @builtins.property
    def ref_lat(self):
        """Message field 'ref_lat'."""
        return self._ref_lat

    @ref_lat.setter
    def ref_lat(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ref_lat' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'ref_lat' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._ref_lat = value

    @builtins.property
    def ref_lon(self):
        """Message field 'ref_lon'."""
        return self._ref_lon

    @ref_lon.setter
    def ref_lon(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ref_lon' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'ref_lon' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._ref_lon = value

    @builtins.property
    def ref_alt(self):
        """Message field 'ref_alt'."""
        return self._ref_alt

    @ref_alt.setter
    def ref_alt(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ref_alt' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'ref_alt' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._ref_alt = value

    @builtins.property
    def head(self):
        """Message field 'head'."""
        return self._head

    @head.setter
    def head(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'head' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'head' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._head = value

    @builtins.property
    def roll(self):
        """Message field 'roll'."""
        return self._roll

    @roll.setter
    def roll(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'roll' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'roll' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._roll = value

    @builtins.property
    def pitch(self):
        """Message field 'pitch'."""
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'pitch' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'pitch' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._pitch = value

    @builtins.property
    def status1(self):
        """Message field 'status1'."""
        return self._status1

    @status1.setter
    def status1(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'status1' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'status1' field must be an unsigned integer in [0, 4294967295]"
        self._status1 = value

    @builtins.property
    def status2(self):
        """Message field 'status2'."""
        return self._status2

    @status2.setter
    def status2(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'status2' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'status2' field must be an unsigned integer in [0, 4294967295]"
        self._status2 = value

    @builtins.property
    def rtk_nbase(self):
        """Message field 'rtk_nbase'."""
        return self._rtk_nbase

    @rtk_nbase.setter
    def rtk_nbase(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'rtk_nbase' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'rtk_nbase' field must be an unsigned integer in [0, 255]"
        self._rtk_nbase = value

    @builtins.property
    def rtk_nrover(self):
        """Message field 'rtk_nrover'."""
        return self._rtk_nrover

    @rtk_nrover.setter
    def rtk_nrover(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'rtk_nrover' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'rtk_nrover' field must be an unsigned integer in [0, 255]"
        self._rtk_nrover = value

    @builtins.property
    def battery(self):
        """Message field 'battery'."""
        return self._battery

    @battery.setter
    def battery(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'battery' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'battery' field must be an unsigned integer in [0, 255]"
        self._battery = value

    @builtins.property
    def r(self):
        """Message field 'r'."""
        return self._r

    @r.setter
    def r(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'r' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'r' field must be an unsigned integer in [0, 255]"
        self._r = value

    @builtins.property
    def g(self):
        """Message field 'g'."""
        return self._g

    @g.setter
    def g(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'g' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'g' field must be an unsigned integer in [0, 255]"
        self._g = value

    @builtins.property
    def b(self):
        """Message field 'b'."""
        return self._b

    @b.setter
    def b(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'b' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'b' field must be an unsigned integer in [0, 255]"
        self._b = value

    @builtins.property
    def rtk_n(self):
        """Message field 'rtk_n'."""
        return self._rtk_n

    @rtk_n.setter
    def rtk_n(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rtk_n' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'rtk_n' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._rtk_n = value

    @builtins.property
    def rtk_e(self):
        """Message field 'rtk_e'."""
        return self._rtk_e

    @rtk_e.setter
    def rtk_e(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rtk_e' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'rtk_e' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._rtk_e = value

    @builtins.property
    def rtk_d(self):
        """Message field 'rtk_d'."""
        return self._rtk_d

    @rtk_d.setter
    def rtk_d(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rtk_d' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'rtk_d' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._rtk_d = value

    @builtins.property
    def nav_state(self):
        """Message field 'nav_state'."""
        return self._nav_state

    @nav_state.setter
    def nav_state(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'nav_state' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'nav_state' field must be an unsigned integer in [0, 255]"
        self._nav_state = value
