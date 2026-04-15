import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_tracker import MotionTracker
import numpy as np

def test_motion_tracker_initialization():
    tracker = MotionTracker()
    assert len(tracker.lift_history) == 0
    assert len(tracker.tip_motion_history) == 0


def test_motion_tracker_update_returns_expected_deltas():
    tracker = MotionTracker()
    current_lifts = np.array([10.0, -3.0, 0.0, 4.0, -7.0], dtype=float)
    baseline_lifts = np.array([7.0, -1.0, 0.0, 2.0, -2.0], dtype=float)
    current_heights = np.array([0.5, -0.4, 0.0, 0.1, -0.2], dtype=float)
    baseline_heights = np.array([0.2, -0.3, 0.0, -0.1, -0.5], dtype=float)

    relative_lifts, delta_heights, tip_motions = tracker.update(
        current_lifts, baseline_lifts, current_heights, baseline_heights
    )

    np.testing.assert_allclose(relative_lifts, np.array([3.0, -2.0, 0.0, 2.0, -5.0], dtype=float))
    np.testing.assert_allclose(delta_heights, np.array([0.3, -0.1, 0.0, 0.2, 0.3], dtype=float))
    np.testing.assert_allclose(tip_motions, np.array([0.3, 0.1, 0.0, 0.2, 0.3], dtype=float))


def test_smoothed_metrics_empty_then_average():
    tracker = MotionTracker()
    avg_lift, avg_tip = tracker.get_smoothed_metrics()
    assert avg_lift == [0.0] * 5
    assert avg_tip == [0.0] * 5

    tracker.update(np.array([1, 2, 3, 4, 5], dtype=float), np.zeros(5), np.zeros(5), np.zeros(5))
    tracker.update(np.array([3, 2, 1, 0, -1], dtype=float), np.zeros(5), np.zeros(5), np.zeros(5))
    avg_lift, avg_tip = tracker.get_smoothed_metrics()
    np.testing.assert_allclose(avg_lift, [2.0, 2.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(avg_tip, [0.0, 0.0, 0.0, 0.0, 0.0])

def test_reset():
    tracker = MotionTracker()
    tracker.update(np.ones(5), np.zeros(5), np.ones(5), np.zeros(5))
    tracker.reset()
    assert len(tracker.lift_history) == 0
    assert len(tracker.tip_motion_history) == 0
