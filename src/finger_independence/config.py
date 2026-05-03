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

    # Biomechanical Algorithm Constants (angle-domain)
    SMOOTHING_ALPHA = 0.25
    ANGLE_SMOOTHING_WINDOW = 5
    MOTION_NOISE_THRESHOLD_DEG = 1.5
    TARGET_MOTION_MIN_DEG = 2.0
    TARGET_MOTION_HIGHLIGHT_DEG = 4.0
    CALIBRATION_FRAMES = 45
    
    # Logic Thresholds
    FINGER_BEND_THRESHOLD_DEG = 175.0  # Increased sensitivity for base-joint movement
    MOTION_ROLLING_WINDOW_SIZE = 8  # Reduced for more responsiveness
    MIN_TOTAL_MOTION_THRESHOLD = 0.0001 # More sensitive
    
    # Thumb Tracking Weights
    THUMB_MCP_WEIGHT = 0.8
    THUMB_CMC_WEIGHT = 0.2

    # Thumb signal blend
    THUMB_OPPOSITION_WEIGHT = 0.6
    THUMB_FLEXION_WEIGHT = 0.4

    # Timeings
    PREPARE_DURATION_SEC = 3.0
    RECORDING_DURATION_SEC = 5.0
    SCORE_NOISE_REDUCTION_FACTOR = 10.0 # Legacy

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

    # Trial tracking
    MIN_ACTIVE_FRAMES_PER_CYCLE = 4
    MIN_CYCLES_FOR_VALID_SCORE = 1
