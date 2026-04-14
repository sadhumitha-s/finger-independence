import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from config import Config

class MotionTracker:
    def __init__(self):
        self.window_size = Config.MOTION_ROLLING_WINDOW_SIZE
        # queue of shape (window_size, 5) storing distances moved
        self.motion_history = deque(maxlen=self.window_size)
        self.previous_tips = None

    def update(self, landmarks: List[Tuple[float, float, float]]) -> List[float]:
        """Calculates distance moved since last frame and updates history."""
        current_tips = np.array([landmarks[i] for i in Config.FINGER_TIP_INDICES])
        
        if self.previous_tips is None:
            self.previous_tips = current_tips
            self.motion_history.append(np.zeros(5))
            return [0.0] * 5
            
        # Compute Euclidean distance delta for each finger tip
        distances = np.linalg.norm(current_tips - self.previous_tips, axis=1)
        self.motion_history.append(distances)
        self.previous_tips = current_tips
        
        return self.get_smoothed_motion()

    def get_smoothed_motion(self) -> List[float]:
        """Returns the rolling average of motion for all 5 fingers."""
        if len(self.motion_history) == 0:
            return [0.0] * 5
        return np.mean(self.motion_history, axis=0).tolist()

    def reset(self):
        self.motion_history.clear()
        self.previous_tips = None
