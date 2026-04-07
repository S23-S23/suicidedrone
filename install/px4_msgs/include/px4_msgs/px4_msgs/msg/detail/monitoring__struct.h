// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from px4_msgs:msg/Monitoring.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__MONITORING__STRUCT_H_
#define PX4_MSGS__MSG__DETAIL__MONITORING__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'NAVIGATION_STATE_MANUAL'.
/**
  * Manual mode
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_MANUAL = 0
};

/// Constant 'NAVIGATION_STATE_ALTCTL'.
/**
  * Altitude control mode
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_ALTCTL = 1
};

/// Constant 'NAVIGATION_STATE_POSCTL'.
/**
  * Position control mode
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_POSCTL = 2
};

/// Constant 'NAVIGATION_STATE_AUTO_MISSION'.
/**
  * Auto mission mode
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_AUTO_MISSION = 3
};

/// Constant 'NAVIGATION_STATE_AUTO_LOITER'.
/**
  * Auto loiter mode
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_AUTO_LOITER = 4
};

/// Constant 'NAVIGATION_STATE_AUTO_RTL'.
/**
  * Auto return to launch mode
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_AUTO_RTL = 5
};

/// Constant 'NAVIGATION_STATE_UNUSED3'.
/**
  * Free slot
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_UNUSED3 = 8
};

/// Constant 'NAVIGATION_STATE_UNUSED'.
/**
  * Free slot
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_UNUSED = 9
};

/// Constant 'NAVIGATION_STATE_ACRO'.
/**
  * Acro mode
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_ACRO = 10
};

/// Constant 'NAVIGATION_STATE_UNUSED1'.
/**
  * Free slot
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_UNUSED1 = 11
};

/// Constant 'NAVIGATION_STATE_DESCEND'.
/**
  * Descend mode (no position control)
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_DESCEND = 12
};

/// Constant 'NAVIGATION_STATE_TERMINATION'.
/**
  * Termination mode
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_TERMINATION = 13
};

/// Constant 'NAVIGATION_STATE_OFFBOARD'.
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_OFFBOARD = 14
};

/// Constant 'NAVIGATION_STATE_STAB'.
/**
  * Stabilized mode
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_STAB = 15
};

/// Constant 'NAVIGATION_STATE_UNUSED2'.
/**
  * Free slot
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_UNUSED2 = 16
};

/// Constant 'NAVIGATION_STATE_AUTO_TAKEOFF'.
/**
  * Takeoff
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_AUTO_TAKEOFF = 17
};

/// Constant 'NAVIGATION_STATE_AUTO_LAND'.
/**
  * Land
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_AUTO_LAND = 18
};

/// Constant 'NAVIGATION_STATE_AUTO_FOLLOW_TARGET'.
/**
  * Auto Follow
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_AUTO_FOLLOW_TARGET = 19
};

/// Constant 'NAVIGATION_STATE_AUTO_PRECLAND'.
/**
  * Precision land with landing target
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_AUTO_PRECLAND = 20
};

/// Constant 'NAVIGATION_STATE_ORBIT'.
/**
  * Orbit in a circle
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_ORBIT = 21
};

/// Constant 'NAVIGATION_STATE_AUTO_VTOL_TAKEOFF'.
/**
  * Takeoff, transition, establish loiter
 */
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_AUTO_VTOL_TAKEOFF = 22
};

/// Constant 'NAVIGATION_STATE_MAX'.
enum
{
  px4_msgs__msg__Monitoring__NAVIGATION_STATE_MAX = 23
};

/// Struct defined in msg/Monitoring in the package px4_msgs.
typedef struct px4_msgs__msg__Monitoring
{
  /// time since system start (microseconds)
  uint64_t timestamp;
  /// GPS Time of Week
  uint32_t tow;
  /// Current position X (m)
  float pos_x;
  /// Current position Y (m)
  float pos_y;
  /// Current position Z (m)
  float pos_z;
  /// Latitude, (degrees)
  double lat;
  /// Longitude, (degrees)
  double lon;
  /// Altitude AMSL, (meters)
  float alt;
  /// Reference point latitude, (degrees)
  double ref_lat;
  /// Reference point longitude, (degrees)
  double ref_lon;
  /// Reference altitude AMSL, (metres)
  float ref_alt;
  /// Heading (degree)
  float head;
  /// Roll
  float roll;
  /// Pitch
  float pitch;
  /// status #1
  uint32_t status1;
  /// status #2 (RESERVED)
  uint32_t status2;
  /// The number of GPS satellite base obeservation
  uint8_t rtk_nbase;
  /// The number of GPS satellite base obeservation
  uint8_t rtk_nrover;
  /// Battery
  uint8_t battery;
  /// the color of LED (RED)
  uint8_t r;
  /// the color of LED (GREEN)
  uint8_t g;
  /// the color of LED (BLUE)
  uint8_t b;
  /// Rtk Baseline North coordinate (m)
  float rtk_n;
  /// Rtk Baseline East coordinate (m)
  float rtk_e;
  /// Rtk Baseline Down coordinate (m)
  float rtk_d;
  /// Currently active mode
  uint8_t nav_state;
} px4_msgs__msg__Monitoring;

// Struct for a sequence of px4_msgs__msg__Monitoring.
typedef struct px4_msgs__msg__Monitoring__Sequence
{
  px4_msgs__msg__Monitoring * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__msg__Monitoring__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PX4_MSGS__MSG__DETAIL__MONITORING__STRUCT_H_
