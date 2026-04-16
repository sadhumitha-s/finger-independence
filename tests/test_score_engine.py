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
    score = engine.calculate_independence_score(5, _z())
    assert score == 0.0

    score = engine.calculate_independence_score(-1, _z())
    assert score == 0.0


def test_zero_signal_returns_zero():
    engine = ScoreEngine()
    score = engine.calculate_independence_score(1, _z())
    assert score == 0.0


def test_frame_leakage_returns_none_when_target_below_activity_gate():
    engine = ScoreEngine()
    motion = _z()
    motion[3] = Config.TARGET_MOTION_MIN_DEG - 0.01
    motion[1] = 10.0
    assert engine.calculate_frame_leakage(3, motion) is None


def test_perfect_isolation_scores_near_one():
    engine = ScoreEngine()
    motion = _z()

    motion[2] = Config.TARGET_MOTION_MIN_DEG + 5.0

    score = engine.calculate_independence_score(2, motion)
    assert score > 0.99


def test_noise_from_other_fingers_reduces_score():
    engine = ScoreEngine()
    clean_motion = _z()
    clean_motion[1] = Config.TARGET_MOTION_MIN_DEG + 6.0

    noisy_motion = clean_motion.copy()
    noisy_motion[3] = clean_motion[1] * 0.8

    clean_score = engine.calculate_independence_score(1, clean_motion)
    noisy_score = engine.calculate_independence_score(1, noisy_motion)
    assert noisy_score < clean_score


def test_coupling_matrix_updates_when_non_target_moves_with_target():
    engine = ScoreEngine()
    motion = _z()

    target_idx = 4
    motion[target_idx] = Config.TARGET_MOTION_MIN_DEG + 10.0
    motion[2] = (Config.TARGET_MOTION_MIN_DEG + 10.0) * 0.75  # strong coupled motion

    _ = engine.calculate_independence_score(target_idx, motion)
    coupling = engine.get_coupling_matrix()
    assert coupling[2, target_idx] > 0.7
