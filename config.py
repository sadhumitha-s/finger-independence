# config.py

class Config:
    # Camera
    CAMERA_INDEX = 0
    WINDOW_NAME = "Finger Independence Analyzer"

    # Layout
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    LEFT_PANEL_WIDTH = 800
    RIGHT_PANEL_WIDTH = 400

    # MediaPipe
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7

    # Logic Thresholds
    FINGER_BEND_THRESHOLD_DEG = 160.0
    MOTION_ROLLING_WINDOW_SIZE = 15
    MIN_TOTAL_MOTION_THRESHOLD = 0.001

    # Timeings
    PREPARE_DURATION_SEC = 3.0
    RECORDING_DURATION_SEC = 5.0

    # UI Colors (BGR)
    COLOR_BG_RIGHT_PANEL = (30, 30, 30)
    COLOR_TEXT = (255, 255, 255)
    COLOR_ACCENT = (0, 255, 0)
    COLOR_WARNING = (0, 0, 255)
    COLOR_UI_BORDER = (100, 100, 100)
    COLOR_BAR_BG = (50, 50, 50)
    COLOR_FINGER_COLORS = [
        (255, 150, 0),    # Thumb
        (0, 255, 255),    # Index
        (255, 0, 0),      # Middle
        (0, 0, 255),      # Ring
        (255, 0, 255)     # Pinky
    ]

    FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    
    # Landmark indices for fingertips
    FINGER_TIP_INDICES = [4, 8, 12, 16, 20]
