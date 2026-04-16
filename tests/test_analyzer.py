import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import HandAnalyzer

def _base_landmarks():
    """Generates a geometrically valid set of hand landmarks (Open Palm)."""
    lm = [(0.0, 0.0, 0.0) for _ in range(21)]
    # Wrist
    lm[0] = (0.0, 0.0, 0.0)
    
    # MCPs (The knuckles)
    lm[5] = (0.2, 0.8, 0.0)    # Index MCP
    lm[9] = (0.0, 1.0, 0.0)    # Middle MCP
    lm[13] = (-0.2, 0.9, 0.0)  # Ring MCP
    lm[17] = (-0.4, 0.8, 0.0)  # Pinky MCP

    # Thumb points (roughly to the right in current coordinate system)
    lm[1] = (0.3, 0.2, 0.0)
    lm[2] = (0.5, 0.3, 0.0)
    lm[3] = (0.7, 0.4, 0.0)
    lm[4] = (0.9, 0.5, 0.0)

    # PIPs (straight fingers)
    lm[6] = (0.2, 1.3, 0.0)    # Index PIP
    lm[10] = (0.0, 1.5, 0.0)   # Middle PIP
    lm[14] = (-0.2, 1.4, 0.0)  # Ring PIP
    lm[18] = (-0.4, 1.3, 0.0)  # Pinky PIP
    
    return lm

def test_mcp_flexion_angle_drops_when_finger_curls():
    analyzer = HandAnalyzer()
    lm = _base_landmarks()
    palm_normal, _ = analyzer.compute_palm_plane(lm, "Right")

    straight_angles, _ = analyzer.compute_metrics(lm, palm_normal)

    # 1. Curl index
    lm_curled = _base_landmarks()
    # Pull PIP towards camera (-Z) to simulate bending
    lm_curled[6] = (0.2, 0.8, -0.5) 
    curled_angles, _ = analyzer.compute_metrics(lm_curled, palm_normal)

    assert curled_angles[1] < straight_angles[1], "Index angle should decrease when curled"

def test_thumb_signal_remains_stable_during_index_movement():
    """
    Core Isolation Test: Moving the index finger should NOT 
    significantly affect the thumb's reported signal.
    """
    analyzer = HandAnalyzer()
    lm = _base_landmarks()
    palm_normal, _ = analyzer.compute_palm_plane(lm, "Right")

    baseline_angles, _ = analyzer.compute_metrics(lm, palm_normal)
    baseline_thumb = baseline_angles[0]

    # 1. Simulate significant Index movement (bend and slight MCP shift/jitter)
    lm[5] = (0.22, 0.82, 0.02) # Jitter
    lm[6] = (0.2, 0.5, -0.4)   # Bend
    
    moving_angles, _ = analyzer.compute_metrics(lm, palm_normal)
    moving_thumb = moving_angles[0]

    # Change should be very minimal (< 0.5 degrees)
    assert abs(moving_thumb - baseline_thumb) < 0.5, f"Thumb signal shifted by {abs(moving_thumb - baseline_thumb)} during index movement"

def test_thumb_signal_remains_stable_during_ring_movement():
    """Verify that Ring finger movement also doesn't pollute thumb signal."""
    analyzer = HandAnalyzer()
    lm = _base_landmarks()
    palm_normal, _ = analyzer.compute_palm_plane(lm, "Right")

    baseline_angles, _ = analyzer.compute_metrics(lm, palm_normal)
    baseline_thumb = baseline_angles[0]

    # Simulate Ring movement
    lm[13] = (-0.18, 0.92, 0.01) # Jitter
    lm[14] = (-0.2, 0.6, -0.4)   # Bend
    
    moving_angles, _ = analyzer.compute_metrics(lm, palm_normal)
    moving_thumb = moving_angles[0]

    assert abs(moving_thumb - baseline_thumb) < 0.5, f"Thumb signal shifted by {abs(moving_thumb - baseline_thumb)} during ring movement"

def test_orientation_validity():
    analyzer = HandAnalyzer()
    lm = _base_landmarks()
    
    # Standard valid palm normal
    palm_normal, _ = analyzer.compute_palm_plane(lm, "Right")
    assert analyzer.is_orientation_valid(palm_normal)

    # Invalidate by flipping hand (pointing back of hand to camera)
    palm_normal_flipped = -palm_normal
    assert not analyzer.is_orientation_valid(palm_normal_flipped)
