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
        # Per-finger active-frame score buffers while recording.
        self._frame_scores: Dict[int, List[float]] = {}
        # Cycle-aware buffers: active segment score chunks (bend phase) collected per finger.
        self._cycle_scores: Dict[int, List[float]] = {}
        self._current_cycle_samples: Dict[int, List[float]] = {}
        self._was_active: Dict[int, bool] = {}
        self._initialize_finger_maps()

    def _initialize_finger_maps(self):
        self.results = {idx: 0.0 for idx in range(len(Config.FINGERS))}
        self.final_results = {idx: 0.0 for idx in range(len(Config.FINGERS))}
        self._frame_scores = {idx: [] for idx in range(len(Config.FINGERS))}
        self._cycle_scores = {idx: [] for idx in range(len(Config.FINGERS))}
        self._current_cycle_samples = {idx: [] for idx in range(len(Config.FINGERS))}
        self._was_active = {idx: False for idx in range(len(Config.FINGERS))}

    def begin_finger_recording(self, finger_idx: int):
        if finger_idx not in self._frame_scores:
            return
        self._frame_scores[finger_idx] = []
        self._cycle_scores[finger_idx] = []
        self._current_cycle_samples[finger_idx] = []
        self._was_active[finger_idx] = False
        self.results[finger_idx] = 0.0

    def _finalize_open_cycle(self, finger_idx: int):
        samples = self._current_cycle_samples[finger_idx]
        if len(samples) < Config.MIN_ACTIVE_FRAMES_PER_CYCLE:
            self._current_cycle_samples[finger_idx] = []
            return

        arr = np.array(samples, dtype=float)
        # Trim extreme jitter spikes before per-cycle averaging.
        q10, q90 = np.percentile(arr, [10, 90])
        trimmed = arr[(arr >= q10) & (arr <= q90)]
        cycle_score = float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(arr))
        self._cycle_scores[finger_idx].append(max(0.0, min(1.0, cycle_score)))
        self._current_cycle_samples[finger_idx] = []

    def record_score(
        self,
        finger_idx: int,
        score: float,
        is_target_active: Optional[bool] = None,
        target_lift: Optional[float] = None,
        target_tip: Optional[float] = None,
    ):
        if finger_idx not in self.results:
            return

        if is_target_active is None:
            if target_lift is None or target_tip is None:
                is_target_active = False
            elif self._was_active[finger_idx]:
                release_lift = Config.TARGET_RELEASE_LIFT_DEG
                release_tip = Config.TIP_MOVEMENT_LIMIT * Config.TARGET_RELEASE_TIP_SCALE
                is_target_active = (target_lift >= release_lift) or (target_tip >= release_tip)
            else:
                activity_lift = Config.TARGET_ACTIVITY_LIFT_DEG
                activity_tip = Config.TIP_MOVEMENT_LIMIT * Config.TARGET_ACTIVITY_TIP_SCALE
                is_target_active = (target_lift >= activity_lift) or (target_tip >= activity_tip)

        bounded_score = max(0.0, min(1.0, float(score)))
        alpha = Config.SMOOTHING_ALPHA
        smoothed = (alpha * bounded_score) + ((1.0 - alpha) * self.results[finger_idx])
        self.results[finger_idx] = smoothed

        if is_target_active:
            self._frame_scores[finger_idx].append(bounded_score)
            self._current_cycle_samples[finger_idx].append(bounded_score)
            self._was_active[finger_idx] = True
            return

        if self._was_active[finger_idx]:
            self._finalize_open_cycle(finger_idx)
        self._was_active[finger_idx] = False

    def finalize_finger(self, finger_idx: int):
        if finger_idx not in self._frame_scores:
            return

        # Close any active cycle if recording ends mid-bend.
        if self._was_active[finger_idx]:
            self._finalize_open_cycle(finger_idx)
            self._was_active[finger_idx] = False

        cycle_scores = self._cycle_scores[finger_idx]
        frame_scores = self._frame_scores[finger_idx]

        if len(cycle_scores) >= Config.MIN_CYCLES_FOR_VALID_SCORE:
            finalized = float(np.mean(cycle_scores))
        elif len(frame_scores) >= Config.MIN_ACTIVE_FRAMES_PER_CYCLE:
            # Fallback for users who hold bent finger instead of repeating cycles.
            finalized = float(np.mean(frame_scores))
        else:
            finalized = 0.0

        finalized = max(0.0, min(1.0, finalized))
        self.final_results[finger_idx] = finalized
        self.results[finger_idx] = finalized
        self._frame_scores[finger_idx] = []
        self._cycle_scores[finger_idx] = []
        self._current_cycle_samples[finger_idx] = []

    def export_csv(self):
        if not self.final_results:
            return

        file_exists = os.path.isfile(self.filename)
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Finger ID", "Finger Name", "Independence Score"])

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for finger_idx, score in sorted(self.final_results.items()):
                writer.writerow([timestamp, finger_idx, Config.FINGERS[finger_idx], f"{score:.4f}"])
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
