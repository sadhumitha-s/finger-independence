import numpy as np
from config import Config

class ScoreEngine:
    def __init__(self):
        # Coupling matrix C[j,i] where i is target, j is other
        self.coupling_matrix = np.zeros((5, 5))

    def calculate_independence_score(self,
                                     target_idx: int,
                                     relative_lifts: np.ndarray,
                                     tip_motions: np.ndarray,
                                     delta_heights: np.ndarray,
                                     sideways_motions: np.ndarray) -> float:
        """
        Independence-oriented score in [0, 1].
        - Signal: how clearly the target finger moved.
        - Noise: how much non-target fingers moved, especially relative to target motion.
        Final: signal / (signal + noise)
        """
        if target_idx < 0 or target_idx >= 5:
            return 0.0

        target_lift = abs(relative_lifts[target_idx])
        target_tip = abs(tip_motions[target_idx])

        lift_signal = max(0.0, target_lift - Config.TARGET_ACTIVITY_LIFT_DEG) / Config.TARGET_LIFT_TRIGGER
        tip_signal = max(0.0, target_tip - Config.TIP_MOVEMENT_LIMIT) / max(Config.TIP_MOVEMENT_LIMIT, 1e-6)
        signal = (0.65 * lift_signal) + (0.35 * tip_signal)

        if target_idx == 0:  # Thumb can include small sideways intent.
            thumb_sideways = max(0.0, sideways_motions[target_idx] - (Config.THUMB_SIDEWAYS_TRIGGER * 0.4))
            signal = (0.85 * signal) + (0.15 * (thumb_sideways / max(Config.THUMB_SIDEWAYS_TRIGGER, 1e-6)))

        signal = max(0.0, min(1.0, signal))
        if signal <= 0.0:
            return 0.0

        total_noise = 0.0
        target_tip_motion = abs(delta_heights[target_idx])

        for j in range(5):
            if j == target_idx:
                continue

            j_lift = abs(relative_lifts[j])
            j_tip = abs(delta_heights[j])
            j_side = abs(sideways_motions[j])

            lift_noise = max(0.0, j_lift - Config.OTHER_LIFT_LIMIT) / max(Config.TARGET_LIFT_TRIGGER, 1e-6)
            tip_noise = max(0.0, j_tip - Config.TIP_MOVEMENT_LIMIT) / max(Config.TIP_MOVEMENT_LIMIT, 1e-6)
            side_noise = max(0.0, j_side - 0.10) / 0.10

            finger_noise = (0.45 * lift_noise) + (0.45 * tip_noise) + (0.10 * side_noise)

            if target_tip_motion > (Config.TIP_MOVEMENT_LIMIT * 0.75):
                c_ji = abs(delta_heights[j]) / target_tip_motion
                self.coupling_matrix[j, target_idx] = max(self.coupling_matrix[j, target_idx], c_ji)
                if c_ji > 0.55:
                    coupling_weight = 0.5 if ((target_idx == 0 and j == 1) or (target_idx == 1 and j == 0)) else 1.0
                    finger_noise += (c_ji - 0.55) * coupling_weight

            total_noise += finger_noise

        final_score = signal / (signal + total_noise + 1e-6)
        return max(0.0, min(1.0, final_score))

    def get_coupling_matrix(self) -> np.ndarray:
        return self.coupling_matrix
