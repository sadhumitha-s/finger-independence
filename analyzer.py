import numpy as np
from typing import List, Tuple
from config import Config

class HandAnalyzer:
    FINGER_MCP = [1, 5, 9, 13, 17]
    FINGER_PIP = [2, 6, 10, 14, 18]
    FINGER_TIP = [4, 8, 12, 16, 20]
    WRIST_IDX = 0

    def __init__(self):
        self.baseline_angles = np.zeros(5)
        self.baseline_dirs = [np.array([0.0, 0.0, 0.0]) for _ in range(5)]
        self._last_angles = np.array([120.0, 180.0, 180.0, 180.0, 180.0], dtype=float)
        self.is_calibrated = False

    @staticmethod
    def _safe_normalize(vec: np.ndarray) -> Tuple[np.ndarray, bool]:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            return np.zeros(3, dtype=float), False
        return vec / norm, True

    @staticmethod
    def _angle_deg(v1: np.ndarray, v2: np.ndarray, default: float = 180.0) -> float:
        n1, ok1 = HandAnalyzer._safe_normalize(v1)
        n2, ok2 = HandAnalyzer._safe_normalize(v2)
        if not (ok1 and ok2):
            return default
        dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(dot)))

    @staticmethod
    def _supplementary_angle_deg(v1: np.ndarray, v2: np.ndarray, default: float = 180.0) -> float:
        n1, ok1 = HandAnalyzer._safe_normalize(v1)
        n2, ok2 = HandAnalyzer._safe_normalize(v2)
        if not (ok1 and ok2):
            return default
        dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        return 180.0 - float(np.degrees(np.arccos(dot)))

    def compute_palm_plane(self, landmarks: List[Tuple[float, float, float]], handedness: str = "Right") -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses wrist (0), index MCP (5), and pinky MCP (17) to define the palm plane.
        Returns (palm_normal, wrist_pt).
        """
        pts = np.array(landmarks)
        wrist = pts[0]
        index_mcp = pts[5]
        pinky_mcp = pts[17]

        # Vectors originating from wrist
        v_index = index_mcp - wrist
        v_pinky = pinky_mcp - wrist
        
        palm_normal = np.cross(v_index, v_pinky)
        norm = np.linalg.norm(palm_normal)
        if norm == 0:
            return np.array([0, 0, 1]), wrist
        
        palm_normal = palm_normal / norm
        
        # For Palm facing camera (front of hand):
        # We want palm_normal to point AWAY from camera (+Z) 
        # so that fingers bending TOWARDS camera have a negative dot product.
        
        # Physical Right Hand (not mirrored): Thumb Left, Pinky Right. Cross(Index-Wrist, Pinky-Wrist) is +Z.
        # Physical Left Hand (not mirrored): Thumb Right, Pinky Left. Cross(Index-Wrist, Pinky-Wrist) is -Z.
        
        # NOTE: Mediapipe labels are flipped in mirrored view.
        # If user shows physical Right hand, it looks like a Left hand in image.
        # Our math relies on the image appearance, so we trust the labeled handedness.
        
        if handedness == "Right":
            pass # Keep +Z
        else:
            palm_normal = -palm_normal # Flip -Z to +Z
            
        return palm_normal, wrist

    def is_orientation_valid(self, palm_normal: np.ndarray) -> bool:
        """
        Accept frames where the palm is generally facing the camera.
        With our new convention, palm_normal points INTO the palm (away from camera).
        If palm faces camera, normal points AWAY (positive Z).
        """
        return palm_normal[2] > 0.15 

    def compute_metrics(self, landmarks: List[Tuple[float, float, float]], palm_normal: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Computes joint-angle signals (degrees) and finger direction vectors.
        Signals:
        - Thumb: 0.6 * opposition + 0.4 * flexion
        - Others: MCP flexion angle using (PIP-MCP) vs (MCP-Wrist)
        """
        pts = np.array(landmarks)
        angles = []
        dirs = []
        wrist = pts[self.WRIST_IDX]

        # Thumb signal: opposition + flexion
        palm_axis = pts[5] - pts[17]
        thumb_dir = pts[4] - pts[2]
        thumb_opposition = self._angle_deg(thumb_dir, palm_axis, default=float(self._last_angles[0]))
        thumb_flexion = self._supplementary_angle_deg(
            pts[3] - pts[2],
            pts[2] - pts[1],
            default=float(self._last_angles[0]),
        )
        thumb_signal = (
            (Config.THUMB_OPPOSITION_WEIGHT * thumb_opposition)
            + (Config.THUMB_FLEXION_WEIGHT * thumb_flexion)
        )

        for i in range(5):
            mcp = pts[self.FINGER_MCP[i]]
            pip = pts[self.FINGER_PIP[i]]

            # Finger direction
            finger_dir = pip - mcp
            finger_dir, _ = self._safe_normalize(finger_dir)
            dirs.append(finger_dir)

            if i == 0:
                angles.append(float(thumb_signal))
                continue

            # Use supplementary angle so straight fingers are near 180 deg and curls reduce the value.
            flexion = self._supplementary_angle_deg(pip - mcp, mcp - wrist, default=float(self._last_angles[i]))
            angles.append(flexion)

        out = np.array(angles, dtype=float)
        self._last_angles = out
        return out, dirs

    def get_sideways_motion(self, current_dirs: List[np.ndarray], palm_normal: np.ndarray) -> np.ndarray:
        """Detects sideways MCP drift relative to baseline."""
        sideways_motions = []
        for i in range(5):
            sideways_vec = np.cross(palm_normal, current_dirs[i])
            motion = np.dot(sideways_vec, self.baseline_dirs[i])
            sideways_motions.append(abs(motion))
        return np.array(sideways_motions)

    def calibrate(self, all_landmarks: List[List[Tuple[float, float, float]]], handedness: str = "Right"):
        """Calibrates baseline using average of N frames."""
        if not all_landmarks:
            return
        
        sum_angles = np.zeros(5)
        sum_dirs = np.zeros((5, 3))
        count = 0

        for lm in all_landmarks:
            palm_normal, _ = self.compute_palm_plane(lm, handedness)
            angles, dirs = self.compute_metrics(lm, palm_normal)
            sum_angles += angles
            for i in range(5):
                sum_dirs[i] += dirs[i]
            count += 1

        self.baseline_angles = sum_angles / max(count, 1)
        for i in range(5):
            self.baseline_dirs[i] = sum_dirs[i] / count
            norm = np.linalg.norm(self.baseline_dirs[i])
            if norm > 0:
                self.baseline_dirs[i] /= norm
        
        self.is_calibrated = True
