import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_tracker import MotionTracker
import numpy as np
from config import Config

def test_motion_tracker_initialization():
    tracker = MotionTracker()
    assert len(tracker.angle_history) == 0


def test_motion_tracker_update_returns_expected_outputs():
    tracker = MotionTracker()
    current_angles = np.array([100.0, 170.0, 160.0, 150.0, 180.0], dtype=float)
    baseline_angles = np.array([110.0, 180.0, 165.0, 145.0, 175.0], dtype=float)

    smoothed, delta_angles, motions = tracker.update(
        current_angles, baseline_angles
    )

    np.testing.assert_allclose(smoothed, current_angles)
    np.testing.assert_allclose(delta_angles, np.array([10.0, 10.0, 5.0, -5.0, -5.0], dtype=float))
    np.testing.assert_allclose(
        motions,
        np.maximum(0.0, np.array([10.0, 10.0, 5.0, 5.0, 5.0]) - Config.MOTION_NOISE_THRESHOLD_DEG),
    )


def test_smoothed_metrics_empty_then_average():
    tracker = MotionTracker()
    avg_angles, zeros = tracker.get_smoothed_metrics()
    assert avg_angles == [0.0] * 5
    assert zeros == [0.0] * 5

    tracker.update(np.array([1, 2, 3, 4, 5], dtype=float), np.zeros(5))
    tracker.update(np.array([3, 2, 1, 0, -1], dtype=float), np.zeros(5))
    avg_angles, zeros = tracker.get_smoothed_metrics()
    np.testing.assert_allclose(avg_angles, [2.0, 2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(zeros, [0.0, 0.0, 0.0, 0.0, 0.0])

def test_reset():
    tracker = MotionTracker()
    tracker.update(np.ones(5), np.zeros(5))
    tracker.reset()
    assert len(tracker.angle_history) == 0
