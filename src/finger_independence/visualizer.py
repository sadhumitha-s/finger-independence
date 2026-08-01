import cv2
import numpy as np
from .config import Config
from typing import Dict

class Visualizer:
    def __init__(self):
        pass

    def create_canvas(self) -> np.ndarray:
        # Create full black canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Fill right panel background
        canvas[:, self.left_w:] = Config.COLOR_BG_RIGHT_PANEL
        # Draw separator line
        cv2.line(canvas, (self.left_w, 0), (self.left_w, self.height), Config.COLOR_UI_BORDER, 2)
        return canvas

    def draw_camera_feed(self, canvas: np.ndarray, frame: np.ndarray):
        pass # Streamlit handles drawing the camera feed directly

    def draw_ui(self, canvas: np.ndarray, instruction: str, progress: float, scores: Dict[int, float], is_valid_orientation: bool = True):
        height, width = canvas.shape[:2]
        left_w = int(width * (Config.LEFT_PANEL_WIDTH / Config.WINDOW_WIDTH))
        right_w = width - left_w

        # Draw instruction text
        color = Config.COLOR_TEXT
        if "KEEP" in instruction or "INVALID" in instruction or "Get your hand" in instruction:
            color = (0, 165, 255) # Orange or Warning color
        
        cv2.putText(canvas, instruction, (left_w + 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if not is_valid_orientation:
             cv2.putText(canvas, "WRONG HAND FACING", (left_w + 20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_WARNING, 1)

        # Draw progress bar
        bar_y = 100
        bar_h = 20
        bar_max_w = right_w - 40
        cv2.rectangle(canvas, (left_w + 20, bar_y), 
                      (left_w + 20 + bar_max_w, bar_y + bar_h), 
                      Config.COLOR_BAR_BG, -1)
        
        current_w = int(bar_max_w * progress)
        if current_w > 0:
            cv2.rectangle(canvas, (left_w + 20, bar_y), 
                          (left_w + 20 + current_w, bar_y + bar_h), 
                          Config.COLOR_ACCENT, -1)

        # Draw scores bar charts
        chart_y_start = 200
        bar_spacing = min(60, int(right_w / 6))
        max_bar_h = 200

        cv2.putText(canvas, "Scores", (left_w + 20, chart_y_start - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_TEXT, 2)

        for i, name in enumerate(Config.FINGERS):
            x = left_w + 20 + (i * bar_spacing)
            score = scores.get(i, 0.0)
            
            # Draw background bar
            cv2.rectangle(canvas, (x, chart_y_start), 
                          (x + 30, chart_y_start + max_bar_h), 
                          Config.COLOR_BAR_BG, -1)
            
            # Draw actual score
            score_h = int(score * max_bar_h)
            if score_h > 0:
                cv2.rectangle(canvas, (x, chart_y_start + max_bar_h - score_h), 
                              (x + 30, chart_y_start + max_bar_h), 
                              Config.COLOR_FINGER_COLORS[i], -1)

            # Draw label
            cv2.putText(canvas, name[:3], (x, chart_y_start + max_bar_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_TEXT, 1)
                        
            # Draw text score
            cv2.putText(canvas, f"{score:.2f}", (x - 5, chart_y_start - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_TEXT, 1)
