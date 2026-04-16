import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from config import Config

class Analytics:
    def __init__(self):
        self.output_dir = "data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.filename = os.path.join(self.output_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        # Results shown on the UI (current finger can move up/down).
        self.results: Dict[int, float] = {}
        # Finalized per-finger session scores (for Export/Plot).
        self.final_results: Dict[int, float] = {}
        # Reliability per finger = std-dev of per-trial independence scores.
        self.trial_std_dev: Dict[int, float] = {}
        # Per-finger finalized trial-level independence list.
        self._trial_scores: Dict[int, List[float]] = {}
        # Active trial accumulators.
        self._trial_leakage_sum: Dict[int, float] = {}
        self._trial_frame_count: Dict[int, int] = {}
        self._is_trial_active: Dict[int, bool] = {}
        self._initialize_finger_maps()

    def _initialize_finger_maps(self):
        self.results = {idx: 0.0 for idx in range(len(Config.FINGERS))}
        self.final_results = {idx: 0.0 for idx in range(len(Config.FINGERS))}
        self.trial_std_dev = {idx: 0.0 for idx in range(len(Config.FINGERS))}
        self._trial_scores = {idx: [] for idx in range(len(Config.FINGERS))}
        self._trial_leakage_sum = {idx: 0.0 for idx in range(len(Config.FINGERS))}
        self._trial_frame_count = {idx: 0 for idx in range(len(Config.FINGERS))}
        self._is_trial_active = {idx: False for idx in range(len(Config.FINGERS))}

    def begin_finger_recording(self, finger_idx: int):
        if finger_idx not in self._trial_scores:
            return
        self._trial_scores[finger_idx] = []
        self._trial_leakage_sum[finger_idx] = 0.0
        self._trial_frame_count[finger_idx] = 0
        self._is_trial_active[finger_idx] = False
        self.results[finger_idx] = 0.0

    def _finalize_open_trial(self, finger_idx: int):
        frame_count = self._trial_frame_count[finger_idx]
        if frame_count < Config.MIN_ACTIVE_FRAMES_PER_CYCLE:
            self._trial_leakage_sum[finger_idx] = 0.0
            self._trial_frame_count[finger_idx] = 0
            return

        mean_leakage = self._trial_leakage_sum[finger_idx] / max(frame_count, 1)
        trial_score = max(0.0, min(1.0, 1.0 - float(mean_leakage)))
        self._trial_scores[finger_idx].append(trial_score)
        self._trial_leakage_sum[finger_idx] = 0.0
        self._trial_frame_count[finger_idx] = 0

    def record_leakage(
        self,
        finger_idx: int,
        leakage: Optional[float],
    ):
        if finger_idx not in self.results:
            return

        if leakage is not None:
            bounded_leakage = max(0.0, float(leakage))
            self._trial_leakage_sum[finger_idx] += bounded_leakage
            self._trial_frame_count[finger_idx] += 1
            self._is_trial_active[finger_idx] = True
            running_mean_leakage = self._trial_leakage_sum[finger_idx] / max(self._trial_frame_count[finger_idx], 1)
            preview_score = max(0.0, min(1.0, 1.0 - running_mean_leakage))
            self.results[finger_idx] = preview_score
            return

        if self._is_trial_active[finger_idx]:
            self._finalize_open_trial(finger_idx)
        self._is_trial_active[finger_idx] = False

    def finalize_finger(self, finger_idx: int):
        if finger_idx not in self._trial_scores:
            return

        # Close any active trial if recording ends mid-movement.
        if self._is_trial_active[finger_idx]:
            self._finalize_open_trial(finger_idx)
            self._is_trial_active[finger_idx] = False

        trial_scores = self._trial_scores[finger_idx]
        finalized = float(np.mean(trial_scores)) if len(trial_scores) >= Config.MIN_CYCLES_FOR_VALID_SCORE else 0.0
        reliability = float(np.std(trial_scores)) if len(trial_scores) > 1 else 0.0
        finalized = max(0.0, min(1.0, finalized))
        self.final_results[finger_idx] = finalized
        self.results[finger_idx] = finalized
        self.trial_std_dev[finger_idx] = reliability
        self._trial_scores[finger_idx] = []
        self._trial_leakage_sum[finger_idx] = 0.0
        self._trial_frame_count[finger_idx] = 0

    def export_csv(self):
        if not self.final_results:
            return

        file_exists = os.path.isfile(self.filename)
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Finger ID", "Finger Name", "Independence Score", "Trial Std Dev"])

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for finger_idx, score in sorted(self.final_results.items()):
                std_dev = self.trial_std_dev.get(finger_idx, 0.0)
                writer.writerow([timestamp, finger_idx, Config.FINGERS[finger_idx], f"{score:.4f}", f"{std_dev:.4f}"])
        print(f"Results exported to {self.filename}")

    def plot_results(self):
        if not self.final_results:
            return

        indices = list(self.final_results.keys())
        scores = [self.final_results[i] for i in indices]
        names = [Config.FINGERS[i] for i in indices]

        plt.figure(figsize=(8, 6))
        plt.bar(names, scores, color='skyblue')
        plt.title('Finger Independence Session Scores')
        plt.xlabel('Finger')
        plt.ylabel('Score (0.0 to 1.0)')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def reset(self):
        self._initialize_finger_maps()
