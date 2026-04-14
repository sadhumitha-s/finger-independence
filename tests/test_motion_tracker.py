import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_tracker import MotionTracker
from config import Config
import numpy as np

def test_motion_tracker_initialization():
    tracker = MotionTracker()
    assert len(tracker.motion_history) == 0

def test_motion_tracker_first_frame():
    tracker = MotionTracker()
    # 21 landmarks
    landmarks = [(0.0, 0.0, 0.0)] * 21
    smoothed = tracker.update(landmarks)
    assert smoothed == [0.0] * 5

def test_motion_tracker_movement():
    tracker = MotionTracker()
    
    # Base landmarks
    landmarks1 = [(0.0, 0.0, 0.0)] * 21
    tracker.update(landmarks1)
    
    # Second frame, thumb moves by 1 unit on x axis
    landmarks2 = [(0.0, 0.0, 0.0)] * 21
    thumb_idx = Config.FINGER_TIP_INDICES[0]
    landmarks2[thumb_idx] = (1.0, 0.0, 0.0)
    
    smoothed = tracker.update(landmarks2)
    # The motion history now has [0,0,0,0,0] and [1,0,0,0,0]
    # mean should be [0.5, 0, 0, 0, 0]
    assert np.isclose(smoothed[0], 0.5)
    assert smoothed[1:] == [0.0] * 4

def test_reset():
    tracker = MotionTracker()
    landmarks = [(0.0, 0.0, 0.0)] * 21
    tracker.update(landmarks)
    tracker.reset()
    assert len(tracker.motion_history) == 0
    assert tracker.previous_tips is None
