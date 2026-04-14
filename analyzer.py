import numpy as np
from typing import List, Tuple, Optional
from config import Config

class HandAnalyzer:
    FINGER_MCP = [1, 5, 9, 13, 17] # Thumb uses CMC(1) as base
    FINGER_PIP = [2, 6, 10, 14, 18] # Thumb uses MCP(2) as first joint
    FINGER_TIP = [4, 8, 12, 16, 20]

    def __init__(self):
        self.baseline_lifts = np.zeros(5)
        self.baseline_heights = np.zeros(5)
        self.baseline_dirs = [np.array([0, 0, 0])] * 5
        self.is_calibrated = False

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

    def compute_metrics(self, landmarks: List[Tuple[float, float, float]], palm_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Computes lifts (degrees), fingertip heights, and finger direction vectors."""
        pts = np.array(landmarks)
        lifts = []
        heights = []
        dirs = []

        for i in range(5):
            mcp = pts[self.FINGER_MCP[i]]
            pip = pts[self.FINGER_PIP[i]]
            tip = pts[self.FINGER_TIP[i]]

            # Finger direction
            finger_dir = pip - mcp
            norm_dir = np.linalg.norm(finger_dir)
            if norm_dir > 0:
                finger_dir /= norm_dir
            
            # Lift: arcsin(dot(finger_dir, palm_normal))
            dot_prod = np.clip(np.dot(finger_dir, palm_normal), -1.0, 1.0)
            lift = np.arcsin(dot_prod)
            lift_deg = np.degrees(lift)
            
            # Height: dot((TIP - MCP), palm_normal)
            height = np.dot((tip - mcp), palm_normal)

            lifts.append(lift_deg)
            heights.append(height)
            dirs.append(finger_dir)

        return np.array(lifts), np.array(heights), dirs

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
        
        sum_lifts = np.zeros(5)
        sum_heights = np.zeros(5)
        sum_dirs = np.zeros((5, 3))
        count = 0

        for lm in all_landmarks:
            palm_normal, _ = self.compute_palm_plane(lm, handedness)
            lifts, heights, dirs = self.compute_metrics(lm, palm_normal)
            sum_lifts += lifts
            sum_heights += heights
            for i in range(5):
                sum_dirs[i] += dirs[i]
            count += 1

        self.baseline_lifts = sum_lifts / count
        self.baseline_heights = sum_heights / count
        for i in range(5):
            self.baseline_dirs[i] = sum_dirs[i] / count
            norm = np.linalg.norm(self.baseline_dirs[i])
            if norm > 0:
                self.baseline_dirs[i] /= norm
        
        self.is_calibrated = True
