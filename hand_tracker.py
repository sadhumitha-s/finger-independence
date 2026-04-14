import cv2
import mediapipe as mp
from typing import Optional, List, Tuple
from config import Config

class LandmarkSmoother:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.previous_landmarks = None

    def smooth(self, current_landmarks: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        if self.previous_landmarks is None:
            self.previous_landmarks = current_landmarks
            return current_landmarks
        
        smoothed = []
        for curr, prev in zip(current_landmarks, self.previous_landmarks):
            sx = self.alpha * curr[0] + (1 - self.alpha) * prev[0]
            sy = self.alpha * curr[1] + (1 - self.alpha) * prev[1]
            sz = self.alpha * curr[2] + (1 - self.alpha) * prev[2]
            smoothed.append((sx, sy, sz))
        
        self.previous_landmarks = smoothed
        return smoothed

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )
        self.results = None
        self.smoother = LandmarkSmoother(Config.SMOOTHING_ALPHA)

    def process_frame(self, frame: cv2.Mat) -> bool:
        """Processes the frame and returns True if a hand was found."""
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            self.results = self.hands.process(image_rgb)
            return self.results.multi_hand_landmarks is not None
        except Exception:
            return False

    def draw_landmarks(self, frame: cv2.Mat):
        """Draws the landmarks onto the frame."""
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

    def get_normalized_landmarks(self) -> Tuple[Optional[List[Tuple[float, float, float]]], Optional[str]]:
        """Returns (landmarks, handedness) where landmarks is a list of 21 smoothed (x, y, z) tuples."""
        if self.results and self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]
            raw = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            
            handedness = None
            if self.results.multi_handedness:
                # classification[0].label is typically "Left" or "Right"
                handedness = self.results.multi_handedness[0].classification[0].label
                
            return self.smoother.smooth(raw), handedness
        return None, None
