import numpy as np
from typing import Optional
from config import Config

class ScoreEngine:
    def __init__(self):
        # Coupling matrix C[j,i] where i is target, j is other
        self.coupling_matrix = np.zeros((5, 5))

    def calculate_frame_leakage(
        self,
        target_idx: int,
        motion: np.ndarray,
    ) -> Optional[float]:
        """
        Frame-level coupling leakage:
        leakage = mean(motion[other] / motion[target])
        Returns None when the target did not move enough to be considered valid.
        """
        if target_idx < 0 or target_idx >= 5:
            return None

        target_motion = float(motion[target_idx])
        if target_motion < Config.TARGET_MOTION_MIN_DEG:
            return None

        leakage_terms = []
        for j in range(5):
            if j == target_idx:
                continue
            ratio = float(motion[j]) / max(target_motion, 1e-6)
            leakage_terms.append(ratio)
            self.coupling_matrix[j, target_idx] = max(self.coupling_matrix[j, target_idx], ratio)

        if not leakage_terms:
            return 0.0
        return float(np.mean(leakage_terms))

    @staticmethod
    def leakage_to_independence(leakage: float) -> float:
        score = 1.0 - float(leakage)
        return max(0.0, min(1.0, score))

    def calculate_independence_score(self,
                                     target_idx: int,
                                     motion: np.ndarray) -> float:
        """
        Frame-level convenience score in [0, 1].
        Returns 0 for invalid/ignored frames.
        """
        leakage = self.calculate_frame_leakage(target_idx, motion)
        if leakage is None:
            return 0.0
        return self.leakage_to_independence(leakage)

    def get_coupling_matrix(self) -> np.ndarray:
        return self.coupling_matrix
