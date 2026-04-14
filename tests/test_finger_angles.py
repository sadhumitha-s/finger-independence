import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finger_angles import FingerAngleCalculator
import numpy as np
from config import Config

def test_calculate_angle_straight():
    # A, B, C collinear
    a = (0.0, 1.0, 0.0)
    b = (0.0, 0.5, 0.0)
    c = (0.0, 0.0, 0.0)
    angle = FingerAngleCalculator.calculate_angle(a, b, c)
    assert np.isclose(angle, 180.0)

def test_calculate_angle_right():
    # A, B, C form right angle
    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (0.0, 1.0, 0.0)
    angle = FingerAngleCalculator.calculate_angle(a, b, c)
    assert np.isclose(angle, 90.0)

def test_calculate_angle_zero_vector():
    # B and C same point
    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (0.0, 0.0, 0.0)
    angle = FingerAngleCalculator.calculate_angle(a, b, c)
    assert np.isclose(angle, 180.0) # Our fallback handles zero vectors by returning 180

def test_is_finger_bent():
    assert FingerAngleCalculator.is_finger_bent(150.0) == True
    assert FingerAngleCalculator.is_finger_bent(Config.FINGER_BEND_THRESHOLD_DEG - 1) == True
    assert FingerAngleCalculator.is_finger_bent(180.0) == False
