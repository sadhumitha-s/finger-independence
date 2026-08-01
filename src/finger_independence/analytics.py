import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional
import threading
from .config import Config
from .db_client import db

class Analytics:
    def __init__(self, session_id: str = "local-dev-session"):
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
        
        # Telemetry buffer to batch frame data before sending to DB
        self._telemetry_buffer: Dict[int, List[List[float]]] = {}
        self.session_id = session_id
        self.user_id = "guest"
        
        # Enslavement Matrix (Synergy Mapping)
        # M[i, j] = motion of finger j when finger i is the target
        self.enslavement_matrix = np.zeros((5, 5))
        self._matrix_accumulator = np.zeros((5, 5))
        self._matrix_counts = np.zeros(5)
        
        self._initialize_finger_maps()

    def _initialize_finger_maps(self):
        self.results = {idx: 0.0 for idx in range(len(Config.FINGERS))}
        self.final_results = {idx: 0.0 for idx in range(len(Config.FINGERS))}
        self.trial_std_dev = {idx: 0.0 for idx in range(len(Config.FINGERS))}
        self._trial_scores = {idx: [] for idx in range(len(Config.FINGERS))}
        self._trial_leakage_sum = {idx: 0.0 for idx in range(len(Config.FINGERS))}
        self._trial_frame_count = {idx: 0 for idx in range(len(Config.FINGERS))}
        self._is_trial_active = {idx: False for idx in range(len(Config.FINGERS))}
        self._telemetry_buffer = {idx: [] for idx in range(len(Config.FINGERS))}
        self.enslavement_matrix = np.zeros((5, 5))
        self._matrix_accumulator = np.zeros((5, 5))
        self._matrix_counts = np.zeros(5)

    def begin_finger_recording(self, finger_idx: int):
        if finger_idx not in self._trial_scores:
            return
        self._trial_scores[finger_idx] = []
        self._trial_leakage_sum[finger_idx] = 0.0
        self._trial_frame_count[finger_idx] = 0
        self._is_trial_active[finger_idx] = False
        self._telemetry_buffer[finger_idx] = []
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
        target_idx: int,
        motion_values: np.ndarray,
    ):
        if target_idx not in self.results:
            return
            
        # Store raw motion values for telemetry
        if target_idx in self._telemetry_buffer:
            self._telemetry_buffer[target_idx].append(motion_values.tolist())

        # Validate target motion
        target_motion = float(motion_values[target_idx])
        if target_motion < Config.TARGET_MOTION_MIN_DEG:
            if self._is_trial_active[target_idx]:
                self._finalize_open_trial(target_idx)
            self._is_trial_active[target_idx] = False
            return

        # Calculate frame-level leakage (mean of other/target ratios)
        ratios = []
        for j in range(5):
            if j == target_idx:
                self._matrix_accumulator[target_idx, j] += 1.0 # Self-enslavement is 1.0
                continue
            
            ratio = float(motion_values[j]) / max(target_motion, 1e-6)
            # Clip ratio to 1.0 to avoid outliers from noise
            ratio = min(1.0, max(0.0, ratio))
            ratios.append(ratio)
            
            # Accumulate for the matrix
            self._matrix_accumulator[target_idx, j] += ratio
            
        self._matrix_counts[target_idx] += 1
        
        leakage = float(np.mean(ratios)) if ratios else 0.0
        
        # Original leakage tracking logic
        self._trial_leakage_sum[target_idx] += leakage
        self._trial_frame_count[target_idx] += 1
        self._is_trial_active[target_idx] = True
        
        running_mean_leakage = self._trial_leakage_sum[target_idx] / max(self._trial_frame_count[target_idx], 1)
        preview_score = max(0.0, min(1.0, 1.0 - running_mean_leakage))
        self.results[target_idx] = preview_score

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
        
        # Finalize the enslavement matrix row for this finger
        if self._matrix_counts[finger_idx] > 0:
            self.enslavement_matrix[finger_idx] = (
                self._matrix_accumulator[finger_idx] / self._matrix_counts[finger_idx]
            )
        
        self._trial_scores[finger_idx] = []
        self._trial_leakage_sum[finger_idx] = 0.0
        self._trial_frame_count[finger_idx] = 0

        # Push telemetry to Supabase in a background thread to avoid blocking the video stream
        if self._telemetry_buffer[finger_idx]:
            telemetry_data = self._telemetry_buffer[finger_idx].copy()
            threading.Thread(
                target=db.insert_telemetry,
                args=(self.session_id, finger_idx, telemetry_data),
                daemon=True
            ).start()
            self._telemetry_buffer[finger_idx] = []

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
        
        # Calculate overall score and update DB
        overall_score = float(np.mean(list(self.final_results.values())))
        if self.session_id and self.session_id != "local-dev-session":
            db.update_session(self.session_id, overall_score, {
                "matrix": self.enslavement_matrix.tolist(),
                "scores": self.final_results
            })

    def plot_results(self):
        if not self.final_results:
            return

        import matplotlib
        matplotlib.use('Agg')
        
        # Create a figure with two subplots: Bar chart and Heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 1. Bar Chart (General Metric)
        indices = list(self.final_results.keys())
        scores = [self.final_results[i] for i in indices]
        names = [Config.FINGERS[i] for i in indices]

        ax1.bar(names, scores, color='skyblue')
        ax1.set_title('Global Finger Independence Scores', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Finger', fontsize=12)
        ax1.set_ylabel('Independence (0.0 - 1.0)', fontsize=12)
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, v in enumerate(scores):
            ax1.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

        # 2. Heatmap (Enslavement Matrix)
        df_cm = pd.DataFrame(
            self.enslavement_matrix, 
            index=Config.FINGERS,
            columns=Config.FINGERS
        )
        
        sns.heatmap(
            df_cm, 
            annot=True, 
            fmt=".2f", 
            cmap="YlOrRd", 
            ax=ax2,
            cbar_kws={'label': 'Enslavement Ratio (Slave/Target)'}
        )
        ax2.set_title('Enslavement Matrix (Synergy Mapping)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Response Finger', fontsize=12)
        ax2.set_ylabel('Target Finger', fontsize=12)

        plt.tight_layout()
        
        # Save plot for reference
        plot_path = os.path.join(self.output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path)
        
        # Close the plot to free memory
        plt.close(fig)
        
        print(f"Analytics report saved to {plot_path}")
        return fig

    def reset(self, user_id=None):
        self._initialize_finger_maps()
        if user_id:
            self.user_id = user_id
            self.session_id = db.insert_session(user_id=self.user_id)
