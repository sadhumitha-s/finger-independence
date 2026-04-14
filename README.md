# Finger Independence Analyzer

The **Finger Independence Analyzer** is a computer vision application designed to evaluate and quantify how independently each finger can move relative to the others. It utilizes real-time hand pose estimation via MediaPipe and OpenCV to calculate a "Finger Independence Score" for each finger.

## Features

- **Real-Time Tracking**: Robust hand landmark tracking using MediaPipe.
- **Finger Bend Detection**: Uses geometry-based math on the MCP, PIP, and DIP joints to precisely track movement and actively highlights bent fingers in the UI.
- **Guided Exercise Mode**: A state machine implementation that walks users through testing each finger independently (Prepare -> Record).
- **Interactive UI**: Custom OpenCV-based UI elements showing user instructions, progress, timers, and real-time score bar charts.
- **Analytics & Data Export**: Saves testing session results dynamically to timestamped CSV files and presents scores on static Matplotlib plots at the conclusion of a session.

## Installation

### Requirements

Ensure you have Python 3.8+ installed. 

> [!NOTE]
> For **macOS ARM64 (Apple Silicon)** and **Python 3.12+**, it is highly recommended to use a virtual environment and pin MediaPipe to version `0.10.13` to avoid compatibility issues with the legacy `solutions` API.

```bash
git clone <repository_url>
cd finger-independence

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

Start the main application using:

```bash
python main.py
```

### Controls During Application Run

- **Spacebar**: Start the session or continue to the next state from `IDLE`.
- **P**: Pause / Unpause the current exercise. 
- **R**: Restart the session entirely.
- **S**: Skip the current finger's recording and immediately go to scoring.
- **Q**: Quit the application.

## System Architecture

The project is structured according to modular SDLC principles to ensure maintainability:
- `main.py`: Application entry point and orchestrator.
- `hand_tracker.py`: MediaPipe bounding box and 21-point tracking logic.
- `finger_angles.py`: Angle-based computations for bend detection.
- `motion_tracker.py`: Displacement tracking via trailing windows.
- `score_engine.py`: Encapsulated score calculations protecting against missing bounds/data.
- `exercise_mode.py`: Finite state machine coordinating session timing.
- `visualizer.py`: OpenCV geometric drawing logic.
- `analytics.py`: Data persistence and offline chart plotting via Matplotlib.

## Testing

This project features comprehensive edge-case tests covering core components like vector math, motion smoothing, edge condition bounds, and finite state machine transitions. Tests are created with `pytest`.

Run the test suite via:

```bash
python -m pytest tests/
```
