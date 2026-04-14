import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exercise_mode import ExerciseMode, State
from config import Config
import time

def test_initial_state():
    mode = ExerciseMode()
    assert mode.state == State.IDLE
    assert mode.current_finger_idx == 0

def test_start():
    mode = ExerciseMode()
    mode.start()
    assert mode.state == State.PREPARE

def test_skip_finger():
    mode = ExerciseMode()
    mode.start() # state PREPARE
    mode.skip_finger()
    assert mode.state == State.SCORING

def test_pause():
    mode = ExerciseMode()
    mode.start()
    mode.pause()
    assert mode.is_paused == True
    time.sleep(0.1)
    mode.update()
    mode.pause()
    assert mode.is_paused == False
