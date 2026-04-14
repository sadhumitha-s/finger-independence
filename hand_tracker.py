import cv2
import mediapipe as mp
from typing import Optional, List, Tuple
from config import Config

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

    def get_normalized_landmarks(self) -> Optional[List[Tuple[float, float, float]]]:
        """Returns a list of 21 (x, y, z) tuples for the detected hand."""
        if self.results and self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        return None
