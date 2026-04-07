// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from px4_msgs:msg/F9pRtk.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__F9P_RTK__STRUCT_H_
#define PX4_MSGS__MSG__DETAIL__F9P_RTK__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'AGE_CORR_UNAVAILABLE'.
/**
  * age_corr state
  * Not available
 */
enum
{
  px4_msgs__msg__F9pRtk__AGE_CORR_UNAVAILABLE = 0
};

/// Constant 'AGE_CORR_0_TO_1_SEC'.
/**
  * Age between 0 and 1 second
 */
enum
{
  px4_msgs__msg__F9pRtk__AGE_CORR_0_TO_1_SEC = 1
};

/// Constant 'AGE_CORR_1_TO_2_SEC'.
/**
  * Age between 1 (inclusive) and 2 seconds
 */
enum
{
  px4_msgs__msg__F9pRtk__AGE_CORR_1_TO_2_SEC = 2
};

/// Constant 'AGE_CORR_2_TO_5_SEC'.
/**
  * Age between 2 (inclusive) and 5 seconds
 */
enum
{
  px4_msgs__msg__F9pRtk__AGE_CORR_2_TO_5_SEC = 3
};

/// Constant 'AGE_CORR_5_TO_10_SEC'.
/**
  * Age between 5 (inclusive) and 10 seconds
 */
enum
{
  px4_msgs__msg__F9pRtk__AGE_CORR_5_TO_10_SEC = 4
};

/// Constant 'AGE_CORR_10_TO_15_SEC'.
/**
  * Age between 10 (inclusive) and 15 seconds
 */
enum
{
  px4_msgs__msg__F9pRtk__AGE_CORR_10_TO_15_SEC = 5
};

/// Constant 'AGE_CORR_15_TO_20_SEC'.
/**
  * Age between 15 (inclusive) and 20 seconds
 */
enum
{
  px4_msgs__msg__F9pRtk__AGE_CORR_15_TO_20_SEC = 6
};

/// Constant 'AGE_CORR_20_TO_30_SEC'.
/**
  * Age between 20 (inclusive) and 30 seconds
 */
enum
{
  px4_msgs__msg__F9pRtk__AGE_CORR_20_TO_30_SEC = 7
};

/// Constant 'AGE_CORR_30_TO_45_SEC'.
/**
  * Age between 30 (inclusive) and 45 seconds
 */
enum
{
  px4_msgs__msg__F9pRtk__AGE_CORR_30_TO_45_SEC = 8
};

/// Constant 'AGE_CORR_45_TO_60_SEC'.
/**
  * Age between 45 (inclusive) and 60 seconds
 */
enum
{
  px4_msgs__msg__F9pRtk__AGE_CORR_45_TO_60_SEC = 9
};

/// Struct defined in msg/F9pRtk in the package px4_msgs.
/**
  * f9p_rtk
 */
typedef struct px4_msgs__msg__F9pRtk
{
  /// time since system start (microseconds)
  uint64_t timestamp;
  /// unique device ID for the sensor that does not change between power cycles
  uint32_t device_id;
  /// GPS Time of Week (ms)
  uint32_t tow;
  /// Age of the most recently received differential correction
  uint8_t age_corr;
  /// flag (0-1: no fix, 2: 2D fix, 3: 3D fix, 4: DGPS, 5: Float RTK, 6: Fixed RTK)
  uint8_t fix_type;
  /// Number of satellites used in Nav Solution
  uint8_t satellites_used;
  /// Baseline North coordinate (cm)
  float n;
  /// Baseline East coordinate (cm)
  float e;
  /// Baseline Down coordinate (cm)
  float d;
  /// Velocity of Baseline North coordinate (m/s)
  float v_n;
  /// Velocity of Baseline East coordinate (m/s)
  float v_e;
  /// Velocity of Baseline Down coordinate (m/s)
  float v_d;
  /// Accuracy of relative position North component (mm)
  float acc_n;
  /// Accuracy of relative position East component (mm)
  float acc_e;
  /// Accuracy of relative position Down component (mm)
  float acc_d;
} px4_msgs__msg__F9pRtk;

// Struct for a sequence of px4_msgs__msg__F9pRtk.
typedef struct px4_msgs__msg__F9pRtk__Sequence
{
  px4_msgs__msg__F9pRtk * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__msg__F9pRtk__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PX4_MSGS__MSG__DETAIL__F9P_RTK__STRUCT_H_
