import time
from enum import Enum
from config import Config

class State(Enum):
    IDLE = 0
    PREPARE = 1
    RECORDING = 2
    SCORING = 3
    SUMMARY = 4

class ExerciseMode:
    def __init__(self):
        self.state = State.IDLE
        self.current_finger_idx = 0
        self.state_start_time = 0.0
        self.is_paused = False
        self._pause_time = 0.0
        
    def _change_state(self, new_state: State):
        self.state = new_state
        self.state_start_time = time.time()

    def start(self):
        if self.state == State.IDLE:
            self.current_finger_idx = 0
            self._change_state(State.PREPARE)
            self.is_paused = False

    def pause(self):
        if self.state in [State.PREPARE, State.RECORDING]:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self._pause_time = time.time()
            else:
                self.state_start_time += (time.time() - self._pause_time)
        
    def restart(self):
        self.state = State.IDLE
        self.current_finger_idx = 0
        self.is_paused = False
        
    def skip_finger(self):
        if self.state in [State.PREPARE, State.RECORDING]:
            self._change_state(State.SCORING)
            self.is_paused = False

    def update(self):
        if self.is_paused:
            return

        if self.state == State.PREPARE:
            if self.get_time_in_state() >= Config.PREPARE_DURATION_SEC:
                self._change_state(State.RECORDING)
                
        elif self.state == State.RECORDING:
            if self.get_time_in_state() >= Config.RECORDING_DURATION_SEC:
                self._change_state(State.SCORING)

    def finish_scoring(self):
        if self.state == State.SCORING:
            self.current_finger_idx += 1
            if self.current_finger_idx < 5:
                self._change_state(State.PREPARE)
            else:
                self._change_state(State.SUMMARY)

    def get_time_in_state(self) -> float:
        if self.state == State.IDLE:
            return 0.0
        if self.is_paused:
             return self._pause_time - self.state_start_time
        return time.time() - self.state_start_time

    def get_progress(self) -> float:
        """Returns progress from 0.0 to 1.0 for the current state."""
        time_in_state = self.get_time_in_state()
        if self.state == State.PREPARE:
            return min(1.0, time_in_state / Config.PREPARE_DURATION_SEC)
        elif self.state == State.RECORDING:
            return min(1.0, time_in_state / Config.RECORDING_DURATION_SEC)
        return 0.0

    def get_instruction_text(self) -> str:
        if self.state == State.IDLE:
            return "Press SPACE to start"
        
        finger_name = Config.FINGERS[self.current_finger_idx]
        if self.state == State.PREPARE:
            if self.is_paused: return "PAUSED (Press P)"
            return f"Get ready: {finger_name}"
        elif self.state == State.RECORDING:
            if self.is_paused: return "PAUSED (Press P)"
            return f"BEND ONLY {finger_name.upper()}"
        elif self.state == State.SCORING:
            return "Calculating..."
        elif self.state == State.SUMMARY:
            return "Session Complete!"
            
        return ""
