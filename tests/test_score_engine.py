import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score_engine import ScoreEngine
from config import Config

def test_independence_score_perfect():
    # Only target finger moved
    score = ScoreEngine.calculate_independence_score(0, [1.0, 0.0, 0.0, 0.0, 0.0])
    assert score == 1.0

def test_independence_score_half():
    # Target and one other moved equally
    score = ScoreEngine.calculate_independence_score(0, [1.0, 1.0, 0.0, 0.0, 0.0])
    assert score == 0.5

def test_independence_score_zero():
    # Only other fingers moved
    score = ScoreEngine.calculate_independence_score(0, [0.0, 1.0, 0.0, 0.0, 0.0])
    assert score == 0.0

def test_independence_score_below_threshold():
    # Total motion too small
    motion = (Config.MIN_TOTAL_MOTION_THRESHOLD / 2)
    score = ScoreEngine.calculate_independence_score(0, [motion, 0.0, 0.0, 0.0, 0.0])
    assert score is None

def test_independence_score_invalid_idx():
    score = ScoreEngine.calculate_independence_score(5, [1.0, 0.0, 0.0, 0.0, 0.0])
    assert score is None
    score = ScoreEngine.calculate_independence_score(-1, [1.0, 0.0, 0.0, 0.0, 0.0])
    assert score is None
