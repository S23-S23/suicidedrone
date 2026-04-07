# f9p_rtk

uint64 timestamp            # time since system start (microseconds)
uint32 device_id            # unique device ID for the sensor that does not change between power cycles

uint32 tow                  # GPS Time of Week (ms)
uint8 age_corr              # Age of the most recently received differential correction
uint8 fix_type              # flag (0-1: no fix, 2: 2D fix, 3: 3D fix, 4: DGPS, 5: Float RTK, 6: Fixed RTK)
uint8 satellites_used       # Number of satellites used in Nav Solution
float32 n                   # Baseline North coordinate (cm)
float32 e                   # Baseline East coordinate (cm)
float32 d                   # Baseline Down coordinate (cm)
float32 v_n                 # Velocity of Baseline North coordinate (m/s)
float32 v_e                 # Velocity of Baseline East coordinate (m/s)
float32 v_d                 # Velocity of Baseline Down coordinate (m/s)
float32 acc_n               # Accuracy of relative position North component (mm)
float32 acc_e               # Accuracy of relative position East component (mm)
float32 acc_d               # Accuracy of relative position Down component (mm)


# age_corr state
uint8 AGE_CORR_UNAVAILABLE      = 0     # Not available
uint8 AGE_CORR_0_TO_1_SEC       = 1     # Age between 0 and 1 second
uint8 AGE_CORR_1_TO_2_SEC       = 2     # Age between 1 (inclusive) and 2 seconds
uint8 AGE_CORR_2_TO_5_SEC       = 3     # Age between 2 (inclusive) and 5 seconds
uint8 AGE_CORR_5_TO_10_SEC      = 4     # Age between 5 (inclusive) and 10 seconds
uint8 AGE_CORR_10_TO_15_SEC     = 5     # Age between 10 (inclusive) and 15 seconds
uint8 AGE_CORR_15_TO_20_SEC     = 6     # Age between 15 (inclusive) and 20 seconds
uint8 AGE_CORR_20_TO_30_SEC     = 7     # Age between 20 (inclusive) and 30 seconds
uint8 AGE_CORR_30_TO_45_SEC     = 8     # Age between 30 (inclusive) and 45 seconds
uint8 AGE_CORR_45_TO_60_SEC     = 9     # Age between 45 (inclusive) and 60 seconds
