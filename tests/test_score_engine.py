import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from score_engine import ScoreEngine
from config import Config

def _z():
    return np.zeros(5, dtype=float)


def test_invalid_target_index_returns_zero():
    engine = ScoreEngine()
    score = engine.calculate_independence_score(5, _z(), _z(), _z(), _z())
    assert score == 0.0

    score = engine.calculate_independence_score(-1, _z(), _z(), _z(), _z())
    assert score == 0.0


def test_zero_signal_returns_zero():
    engine = ScoreEngine()
    score = engine.calculate_independence_score(1, _z(), _z(), _z(), _z())
    assert score == 0.0


def test_perfect_isolation_scores_near_one():
    engine = ScoreEngine()
    lifts = _z()
    tip = _z()
    heights = _z()
    side = _z()

    lifts[2] = Config.TARGET_ACTIVITY_LIFT_DEG + Config.TARGET_LIFT_TRIGGER * 1.2
    tip[2] = Config.TIP_MOVEMENT_LIMIT * 2.0
    heights[2] = Config.TIP_MOVEMENT_LIMIT * 2.0

    score = engine.calculate_independence_score(2, lifts, tip, heights, side)
    assert score > 0.99


def test_noise_from_other_fingers_reduces_score():
    engine = ScoreEngine()
    lifts = _z()
    tip = _z()
    heights = _z()
    side = _z()

    lifts[1] = Config.TARGET_ACTIVITY_LIFT_DEG + Config.TARGET_LIFT_TRIGGER
    tip[1] = Config.TIP_MOVEMENT_LIMIT * 2.0
    heights[1] = Config.TIP_MOVEMENT_LIMIT * 2.0

    noisy_lifts = lifts.copy()
    noisy_tip = tip.copy()
    noisy_heights = heights.copy()
    noisy_side = side.copy()
    noisy_lifts[3] = Config.OTHER_LIFT_LIMIT + Config.TARGET_LIFT_TRIGGER
    noisy_tip[3] = Config.TIP_MOVEMENT_LIMIT * 2.2
    noisy_heights[3] = Config.TIP_MOVEMENT_LIMIT * 2.2
    noisy_side[3] = 0.35

    clean_score = engine.calculate_independence_score(1, lifts, tip, heights, side)
    noisy_score = engine.calculate_independence_score(1, noisy_lifts, noisy_tip, noisy_heights, noisy_side)
    assert noisy_score < clean_score


def test_coupling_matrix_updates_when_non_target_moves_with_target():
    engine = ScoreEngine()
    lifts = _z()
    tip = _z()
    heights = _z()
    side = _z()

    target_idx = 4
    lifts[target_idx] = Config.TARGET_ACTIVITY_LIFT_DEG + Config.TARGET_LIFT_TRIGGER
    tip[target_idx] = Config.TIP_MOVEMENT_LIMIT * 2.0
    heights[target_idx] = Config.TIP_MOVEMENT_LIMIT * 2.0
    heights[2] = Config.TIP_MOVEMENT_LIMIT * 1.4  # strong coupled motion

    _ = engine.calculate_independence_score(target_idx, lifts, tip, heights, side)
    coupling = engine.get_coupling_matrix()
    assert coupling[2, target_idx] > 0.55
