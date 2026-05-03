import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import csv
from finger_independence.analytics import Analytics
from finger_independence.config import Config


def test_finalize_finger_aggregates_multiple_trials_and_std(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    analytics = Analytics()
    finger_idx = 1
    analytics.begin_finger_recording(finger_idx)

    # Trial 1: mean leakage 0.20 -> independence 0.80
    # target=10, others=2 -> ratio 0.2
    motion1 = np.full(5, 2.0)
    motion1[finger_idx] = 10.0
    for _ in range(4):
        analytics.record_leakage(finger_idx, motion1)
    # End trial
    analytics.record_leakage(finger_idx, np.zeros(5))

    # Trial 2: mean leakage 0.40 -> independence 0.60
    # target=10, others=4 -> ratio 0.4
    motion2 = np.full(5, 4.0)
    motion2[finger_idx] = 10.0
    for _ in range(4):
        analytics.record_leakage(finger_idx, motion2)
    # End trial
    analytics.record_leakage(finger_idx, np.zeros(5))

    analytics.finalize_finger(finger_idx)

    assert abs(analytics.final_results[finger_idx] - 0.7) < 1e-6
    assert abs(analytics.trial_std_dev[finger_idx] - 0.1) < 1e-6


def test_finalize_finger_discards_short_trials(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    analytics = Analytics()
    finger_idx = 2
    analytics.begin_finger_recording(finger_idx)

    # Below MIN_ACTIVE_FRAMES_PER_CYCLE (=4): should be dropped.
    motion = np.full(5, 1.0)
    motion[finger_idx] = 10.0
    for _ in range(3):
        analytics.record_leakage(finger_idx, motion)
    analytics.record_leakage(finger_idx, np.zeros(5))
    analytics.finalize_finger(finger_idx)

    assert analytics.final_results[finger_idx] == 0.0
    assert analytics.trial_std_dev[finger_idx] == 0.0


def test_enslavement_matrix_tracking(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    analytics = Analytics()
    
    # Target Finger 1 (Index), Response Finger 2 (Middle) moves half as much
    target_idx = 1
    motion = np.zeros(5)
    motion[1] = 10.0
    motion[2] = 5.0
    
    analytics.begin_finger_recording(target_idx)
    for _ in range(5):
        analytics.record_leakage(target_idx, motion)
    analytics.finalize_finger(target_idx)
    
    # M[1, 2] should be 0.5
    assert abs(analytics.enslavement_matrix[1, 2] - 0.5) < 1e-6
    # M[1, 1] should be 1.0
    assert abs(analytics.enslavement_matrix[1, 1] - 1.0) < 1e-6
    # M[1, 0] should be 0.0
    assert abs(analytics.enslavement_matrix[1, 0] - 0.0) < 1e-6


def test_export_csv_writes_reliability_column(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    analytics = Analytics()

    analytics.final_results[0] = 0.85
    analytics.trial_std_dev[0] = 0.07
    analytics.export_csv()

    with open(analytics.filename, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["Timestamp", "Finger ID", "Finger Name", "Independence Score", "Trial Std Dev"]
    assert rows[1][1] == "0"
    assert rows[1][2] == "Thumb"
    assert rows[1][3] == "0.8500"
    assert rows[1][4] == "0.0700"
