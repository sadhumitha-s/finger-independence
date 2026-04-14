import numpy as np
from typing import List, Tuple
from config import Config

class FingerAngleCalculator:
    # Landmarks for MCP, PIP, DIP joints of each finger
    # For thumb, it's CMC(1), MCP(2), IP(3)
    FINGER_JOINTS = [
        (1, 2, 3),    # Thumb
        (5, 6, 7),    # Index
        (9, 10, 11),  # Middle
        (13, 14, 15), # Ring
        (17, 18, 19)  # Pinky
    ]

    @staticmethod
    def calculate_angle(a: Tuple[float, float, float], b: Tuple[float, float, float], c: Tuple[float, float, float]) -> float:
        """Calculates the angle ABC (in degrees) given 3 3D points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 180.0
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    @staticmethod
    def get_finger_angles(landmarks: List[Tuple[float, float, float]]) -> List[float]:
        """Returns the PIP joint angles for all 5 fingers."""
        angles = []
        for joints in FingerAngleCalculator.FINGER_JOINTS:
            idx_a, idx_b, idx_c = joints
            angle = FingerAngleCalculator.calculate_angle(
                landmarks[idx_a],
                landmarks[idx_b],
                landmarks[idx_c]
            )
            angles.append(angle)
        return angles

    @staticmethod
    def is_finger_bent(angle: float) -> bool:
        """Returns True if the finger is bent past the threshold."""
        return angle < Config.FINGER_BEND_THRESHOLD_DEG
