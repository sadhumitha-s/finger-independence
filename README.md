# Finger Independence Analyzer

A computer vision system that quantifies and analyzes individual finger motor control. By leveraging real-time 3D hand pose estimation, the Finger Independence Analyzer provides precise metrics on joint isolation, unintended "leakage" movement, and overall dexterity.

---

## How It Works 

The system transforms raw 2D video input into a robust 3D kinesiological analysis through several key stages:

### 1. 3D Palm Plane Reconstruction
Unlike basic gesture recognizers, this system constructs a dynamic 3D Coordinate System localized to the user's hand. 
- **Reference Points**: It uses the Wrist (0), Index MCP (5), and Pinky MCP (17) to define the Palm Plane.
- **Normal Vector Calculation**: A normal vector (pointing out from the palm) is calculated using the cross product of the palm vectors.
- **Normalization**: This allows the system to remain accurate even as the user tilts or rotates their hand in front of the camera. The system includes orientation validation to ensure data is only recorded when the palm is correctly oriented toward the sensor.

### 2. Biomechanical Metrics
For every frame, the analyzer computes:
- **Finger Lift**: The angle (in degrees) of the finger direction relative to the palm plane.
- **Fingertip Height**: The perpendicular distance from the MCP joint to the fingertip along the palm normal.
- **Sideways Drift**: Detection of unintended lateral movement during vertical exercises.
- **Specialized Thumb Logic**: Includes specific calibration for the thumb's unique range of motion involving the CMC and MCP joints.

### 3. The Independence Score
The core metric is the Independence Ratio, calculated during a target finger's exercise window:
$$Independence Score = \frac{TargetFingerMotion}{TargetFingerMotion + \sum OtherFingerMotion}$$  

A score of 1.0 indicates perfect isolation (only the target finger moved), while lower scores quantify the degree to which other fingers "followed" the movement.

---

## Key Features

- **Robust Hand Tracking**: Real-time 21-point landmark extraction and full 3D hand pose reconstruction.
- **Handedness Independence**: Universal support for both Left and Right hand orientations with automatic coordinate adjustment.
- **Guided Exercise Mode**: A structured state machine (Prepare -> Record -> Score) that facilitates standardized data capture for all five digits.
- **Live Feedback Engine**: High-performance visualization of movement intensity and real-time isolation scores.
- **Automated Analytics**: Session logging to CSV format and post-session reporting via Matplotlib.

---

## Installation

### Requirements
- **Python 3.8+**
- Webcam

> [!TIP]
> **macOS (Apple Silicon) Users**  
> Use a virtual environment and MediaPipe version `0.10.13` for best compatibility and performance.

```bash
# Clone the repository
git clone <repository_url>
cd finger-independence

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Launch the main application:
```bash
python main.py
```

### Session Controls
| Key | Action |
|:---:|:---|
| `Space` | Start Session / Advance to next finger |
| `P` | Pause/Resume exercise |
| `R` | Reset entire session |
| `S` | Skip current finger |
| `Q` | Quit application |

---

## System Architecture

The project follows modular engineering principles:
- **`hand_tracker.py`**: MediaPipe abstraction and landmark filtering.
- **`analyzer.py`**: Core biomechanical math (3D planes, vectors, and lift).
- **`score_engine.py`**: Statistical processing of motion data into independence scores.
- **`exercise_mode.py`**: Finite State Machine managing timing and user flow.
- **`visualizer.py`**: Rendering of the interface and skeletal overlays.
- **`analytics.py`**: Data serialization and reporting.

---

## Testing

The system includes a suite of tests for coordinate transforms, motion smoothing, and score edge cases.
```bash
python -m pytest tests/
```


