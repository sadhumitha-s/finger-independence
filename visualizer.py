import cv2
import numpy as np
from config import Config
from typing import Dict

class Visualizer:
    def __init__(self):
        self.width = Config.WINDOW_WIDTH
        self.height = Config.WINDOW_HEIGHT
        self.left_w = Config.LEFT_PANEL_WIDTH
        self.right_w = Config.RIGHT_PANEL_WIDTH

    def create_canvas(self) -> np.ndarray:
        # Create full black canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Fill right panel background
        canvas[:, self.left_w:] = Config.COLOR_BG_RIGHT_PANEL
        # Draw separator line
        cv2.line(canvas, (self.left_w, 0), (self.left_w, self.height), Config.COLOR_UI_BORDER, 2)
        return canvas

    def draw_camera_feed(self, canvas: np.ndarray, frame: np.ndarray):
        # Resize frame to fit left panel
        h, w = frame.shape[:2]
        scale = min(self.left_w / w, self.height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Center in left panel
        y_offset = (self.height - new_h) // 2
        x_offset = (self.left_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    def draw_ui(self, canvas: np.ndarray, instruction: str, progress: float, scores: Dict[int, float]):
        # Draw instruction text
        cv2.putText(canvas, instruction, (self.left_w + 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.COLOR_TEXT, 2)

        # Draw progress bar
        bar_y = 100
        bar_h = 20
        bar_max_w = self.right_w - 40
        cv2.rectangle(canvas, (self.left_w + 20, bar_y), 
                      (self.left_w + 20 + bar_max_w, bar_y + bar_h), 
                      Config.COLOR_BAR_BG, -1)
        
        current_w = int(bar_max_w * progress)
        if current_w > 0:
            cv2.rectangle(canvas, (self.left_w + 20, bar_y), 
                          (self.left_w + 20 + current_w, bar_y + bar_h), 
                          Config.COLOR_ACCENT, -1)

        # Draw scores bar charts
        chart_y_start = 200
        bar_spacing = 60
        max_bar_h = 200

        cv2.putText(canvas, "Scores", (self.left_w + 20, chart_y_start - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_TEXT, 2)

        for i, name in enumerate(Config.FINGERS):
            x = self.left_w + 40 + (i * bar_spacing)
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

        # Draw Controls info
        controls_y = self.height - 100
        ctrl_texts = [
            "SPACE: Start/Continue",
            "P: Pause",
            "R: Restart",
            "S: Skip finger",
            "Q: Quit"
        ]
        for idx, t in enumerate(ctrl_texts):
            cv2.putText(canvas, t, (self.left_w + 20, controls_y + (idx * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
