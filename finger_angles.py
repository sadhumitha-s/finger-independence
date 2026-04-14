import numpy as np
from typing import List, Tuple
from config import Config

class FingerAngleCalculator:
    # Landmarks for MCP, PIP, DIP joints of each finger
    # For thumb, it's CMC(1), MCP(2), IP(3)
    # Joint triples for MCP-focused tracking
    FINGER_BASE_JOINTS = [
        None,         # Thumb handled specially
        (0, 5, 6),    # Index (Wrist, MCP, PIP)
        (0, 9, 10),   # Middle
        (0, 13, 14),  # Ring
        (0, 17, 18)   # Pinky
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
        """Returns the weighted MCP/CMC angles for all fingers."""
        angles = []
        
        # 1. Handle Thumb specially (80% MCP, 20% CMC)
        angle_mcp = FingerAngleCalculator.calculate_angle(landmarks[1], landmarks[2], landmarks[3])
        angle_cmc = FingerAngleCalculator.calculate_angle(landmarks[0], landmarks[1], landmarks[2])
        thumb_combined = (angle_mcp * Config.THUMB_MCP_WEIGHT) + (angle_cmc * Config.THUMB_CMC_WEIGHT)
        angles.append(thumb_combined)
        
        # 2. Handle other fingers (Index to Pinky)
        for joints in FingerAngleCalculator.FINGER_BASE_JOINTS[1:]:
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
