import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List
from config import Config

class Analytics:
    def __init__(self):
        self.filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # Results map finger_index -> score
        self.results: Dict[int, float] = {}

    def record_score(self, finger_idx: int, score: float):
        self.results[finger_idx] = score

    def export_csv(self):
        if not self.results:
            return
            
        file_exists = os.path.isfile(self.filename)
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Finger ID", "Finger Name", "Independence Score"])
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for finger_idx, score in sorted(self.results.items()):
                writer.writerow([timestamp, finger_idx, Config.FINGERS[finger_idx], f"{score:.4f}"])
        print(f"Results exported to {self.filename}")

    def plot_results(self):
        if not self.results:
            return
        
        indices = list(self.results.keys())
        scores = [self.results[i] for i in indices]
        names = [Config.FINGERS[i] for i in indices]
        
        plt.figure(figsize=(8, 6))
        plt.bar(names, scores, color='skyblue')
        plt.title('Finger Independence Scores')
        plt.xlabel('Finger')
        plt.ylabel('Score (0.0 to 1.0)')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
