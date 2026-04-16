import numpy as np
from collections import deque
from typing import List, Tuple
from config import Config

class MotionTracker:
    def __init__(self):
        self.window_size = Config.ANGLE_SMOOTHING_WINDOW
        # queue of shape (window_size, 5) storing raw joint-angle signals (degrees)
        self.angle_history = deque(maxlen=self.window_size)

    def update(self, current_angles: np.ndarray, baseline_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates baseline-relative motion from smoothed angles.
        Returns (smoothed_angles, delta_angles, thresholded_motion).
        """
        self.angle_history.append(np.array(current_angles, dtype=float))
        smoothed_angles = np.mean(self.angle_history, axis=0)
        delta_angles = baseline_angles - smoothed_angles
        raw_motion = np.abs(delta_angles)
        thresholded_motion = np.maximum(0.0, raw_motion - Config.MOTION_NOISE_THRESHOLD_DEG)
        return smoothed_angles, delta_angles, thresholded_motion

    def get_smoothed_metrics(self) -> Tuple[List[float], List[float]]:
        if len(self.angle_history) == 0:
            return [0.0] * 5, [0.0] * 5
        avg_angles = np.mean(self.angle_history, axis=0).tolist()
        return avg_angles, [0.0] * 5

    def reset(self):
        self.angle_history.clear()
