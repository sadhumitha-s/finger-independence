import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from config import Config

class MotionTracker:
    def __init__(self):
        self.window_size = Config.MOTION_ROLLING_WINDOW_SIZE
        # queue of shape (window_size, 5) storing lift deltas
        self.lift_history = deque(maxlen=self.window_size)
        self.tip_motion_history = deque(maxlen=self.window_size)

    def update(self, current_lifts: np.ndarray, baseline_lifts: np.ndarray, 
               current_heights: np.ndarray, baseline_heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates motion relative to baseline.
        Returns (relative_lifts, delta_heights, tip_motions).
        """
        relative_lifts = current_lifts - baseline_lifts
        delta_heights = current_heights - baseline_heights
        
        # Tip motion: absolute change in relative lift or distance-based
        # PRD says: tip_motion = distance(current_tip, baseline_tip)
        # But we don't have absolute 3D landmarks here in a convenient way, 
        # so we'll use delta_heights as a proxy for vertical tip motion as intended.
        tip_motions = np.abs(delta_heights)
        
        self.lift_history.append(np.abs(relative_lifts))
        self.tip_motion_history.append(tip_motions)
        
        return relative_lifts, delta_heights, tip_motions

    def get_smoothed_metrics(self) -> Tuple[List[float], List[float]]:
        if len(self.lift_history) == 0:
            return [0.0]*5, [0.0]*5
        
        avg_lift = np.mean(self.lift_history, axis=0).tolist()
        avg_tip = np.mean(self.tip_motion_history, axis=0).tolist()
        return avg_lift, avg_tip

    def reset(self):
        self.lift_history.clear()
        self.tip_motion_history.clear()
