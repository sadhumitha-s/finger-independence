import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exercise_mode import ExerciseMode, State
from config import Config

def test_initial_state():
    mode = ExerciseMode()
    assert mode.state == State.IDLE
    assert mode.current_finger_idx == 0


def test_start_enters_calibrate_from_idle():
    mode = ExerciseMode()
    mode.start()
    assert mode.state == State.CALIBRATE


def test_start_does_not_reset_when_not_idle():
    mode = ExerciseMode()
    mode.start()
    mode.current_finger_idx = 3
    mode.start()
    assert mode.state == State.CALIBRATE
    assert mode.current_finger_idx == 3

def test_skip_finger():
    mode = ExerciseMode()
    mode.start() # state CALIBRATE
    mode.skip_finger()
    assert mode.state == State.SCORING


def test_skip_finger_ignored_in_idle():
    mode = ExerciseMode()
    mode.skip_finger()
    assert mode.state == State.IDLE


def test_pause_toggle_in_allowed_states():
    mode = ExerciseMode()
    mode.start()
    mode.pause()
    assert mode.is_paused is True
    mode.pause()
    assert mode.is_paused is False


def test_pause_ignored_in_idle():
    mode = ExerciseMode()
    mode.pause()
    assert mode.is_paused is False


def test_prepare_and_recording_auto_advance_by_time():
    mode = ExerciseMode()
    mode._change_state(State.PREPARE)
    mode.state_start_time -= (Config.PREPARE_DURATION_SEC + 0.1)
    mode.update()
    assert mode.state == State.RECORDING

    mode.state_start_time -= (Config.RECORDING_DURATION_SEC + 0.1)
    mode.update()
    assert mode.state == State.SCORING


def test_finish_scoring_advances_and_summarizes():
    mode = ExerciseMode()
    mode._change_state(State.SCORING)
    mode.current_finger_idx = 3
    mode.finish_scoring()
    assert mode.current_finger_idx == 4
    assert mode.state == State.PREPARE

    mode._change_state(State.SCORING)
    mode.finish_scoring()
    assert mode.current_finger_idx == 5
    assert mode.state == State.SUMMARY
