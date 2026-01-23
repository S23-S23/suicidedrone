import threading
import numpy as np

from enum import Enum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from follower_drone.follower_drone import DroneManager

class Mode(Enum):
    QHAC = 0
    TAKEOFF = 1
    FORMATION = 2
    RETURN = 5
    COMPLETED = 6
    DONE = 7

class ModeHandler():
    def __init__(self, drone_manager: "DroneManager"):
        self._timer = None
        self.mode = Mode.QHAC
        self.drone_manager = drone_manager

    def change_mode(self, mode):
        if self._timer is not None:
            return -1
        if mode == self.mode:
            return
        if mode == Mode.QHAC:
            pass
        elif mode == Mode.TAKEOFF:
            pass
        elif mode == Mode.FORMATION:
            pass
        elif mode == Mode.RETURN:                       ## Return Mode is agents returning to home position
            pass
        elif mode == Mode.COMPLETED:                      ## Landing Mode is agents landing
            pass
        self.mode = mode
        return self.mode

    def is_in_mode(self, mode):
        return self.mode == mode

    def get_mode(self):
        return self.mode
