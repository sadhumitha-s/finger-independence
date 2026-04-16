import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from analytics import Analytics


def test_finalize_finger_aggregates_multiple_trials_and_std(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    analytics = Analytics()
    finger_idx = 1
    analytics.begin_finger_recording(finger_idx)

    # Trial 1: mean leakage 0.20 -> independence 0.80
    for leakage in [0.2, 0.2, 0.2, 0.2]:
        analytics.record_leakage(finger_idx, leakage)
    analytics.record_leakage(finger_idx, None)

    # Trial 2: mean leakage 0.40 -> independence 0.60
    for leakage in [0.4, 0.4, 0.4, 0.4]:
        analytics.record_leakage(finger_idx, leakage)
    analytics.record_leakage(finger_idx, None)

    analytics.finalize_finger(finger_idx)

    assert abs(analytics.final_results[finger_idx] - 0.7) < 1e-6
    assert abs(analytics.trial_std_dev[finger_idx] - 0.1) < 1e-6


def test_finalize_finger_discards_short_trials(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    analytics = Analytics()
    finger_idx = 2
    analytics.begin_finger_recording(finger_idx)

    # Below MIN_ACTIVE_FRAMES_PER_CYCLE (=4): should be dropped.
    for leakage in [0.1, 0.1, 0.1]:
        analytics.record_leakage(finger_idx, leakage)
    analytics.record_leakage(finger_idx, None)
    analytics.finalize_finger(finger_idx)

    assert analytics.final_results[finger_idx] == 0.0
    assert analytics.trial_std_dev[finger_idx] == 0.0


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
