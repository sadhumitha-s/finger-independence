from typing import List, Optional
from config import Config

class ScoreEngine:
    @staticmethod
    def calculate_independence_score(target_finger_idx: int, smoothed_motions: List[float]) -> Optional[float]:
        """
        Computes Independence Score = TargetMotion / TotalMotion.
        Returns None if TotalMotion is insufficient.
        """
        if target_finger_idx < 0 or target_finger_idx > 4:
            return None
            
        target_motion = smoothed_motions[target_finger_idx]
        total_motion = sum(smoothed_motions)
        
        if total_motion < Config.MIN_TOTAL_MOTION_THRESHOLD:
            return None
            
        score = target_motion / total_motion
        # Ensure floating point bounds
        return max(0.0, min(1.0, score))
