uint64 timestamp        # time since system start (microseconds)

uint8 SCENARIO_CMD_SET_START_TIME       = 0
uint8 SCENARIO_CMD_STOP_SCENARIO        = 1
uint8 SCENARIO_CMD_EMERGENCY_LAND       = 2
uint8 SCENARIO_CMD_SET_CONFIGS          = 3
uint8 SCENARIO_CMD_RESET_CONFIGS        = 4

uint8 cmd               # embedded scenario command

float32 param1          # param1
float32 param2          # param2
float32 param3          # param3
uint32  param4          # param4
uint8[32] param5        # param5
