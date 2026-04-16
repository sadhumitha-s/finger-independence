import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import HandAnalyzer


def _base_landmarks():
    lm = [(0.0, 0.0, 0.0) for _ in range(21)]
    # Wrist / palm references
    lm[0] = (0.0, 0.0, 0.0)
    lm[5] = (1.0, 0.0, 0.0)   # index MCP
    lm[17] = (-1.0, 0.0, 0.0) # pinky MCP

    # Thumb references (non-degenerate)
    lm[1] = (0.2, -0.2, 0.0)
    lm[2] = (0.4, -0.2, 0.0)
    lm[3] = (0.6, -0.15, 0.0)
    lm[4] = (0.8, -0.1, 0.0)

    # Non-target fingers (non-degenerate defaults)
    lm[6] = (2.0, 0.0, 0.0)
    lm[9] = (0.8, 0.2, 0.0)
    lm[10] = (1.1, 0.3, 0.0)
    lm[13] = (0.6, -0.1, 0.0)
    lm[14] = (0.9, -0.15, 0.0)
    lm[18] = (-0.8, 0.05, 0.0)
    return lm


def test_mcp_flexion_angle_drops_when_index_curls():
    analyzer = HandAnalyzer()
    lm = _base_landmarks()
    palm_normal, _ = analyzer.compute_palm_plane(lm, "Right")

    straight_angles, _ = analyzer.compute_metrics(lm, palm_normal)

    # Curl index: keep MCP fixed, move PIP orthogonal to MCP-wrist direction.
    lm[6] = (1.0, 1.0, 0.0)
    curled_angles, _ = analyzer.compute_metrics(lm, palm_normal)

    assert straight_angles[1] > curled_angles[1]


def test_degenerate_vectors_fall_back_to_previous_angles():
    analyzer = HandAnalyzer()
    lm = _base_landmarks()
    palm_normal, _ = analyzer.compute_palm_plane(lm, "Right")

    angles_a, _ = analyzer.compute_metrics(lm, palm_normal)

    # Make index direction degenerate (PIP == MCP).
    lm[6] = lm[5]
    angles_b, _ = analyzer.compute_metrics(lm, palm_normal)

    assert angles_b[1] == angles_a[1]
