import cv2
import numpy as np

from config import Config
from hand_tracker import HandTracker
from motion_tracker import MotionTracker
from score_engine import ScoreEngine
from exercise_mode import ExerciseMode, State
from visualizer import Visualizer
from analytics import Analytics
from finger_angles import FingerAngleCalculator

def main():
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {Config.CAMERA_INDEX}")
        return

    tracker = HandTracker()
    motion = MotionTracker()
    exercise = ExerciseMode()
    vis = Visualizer()
    analytics = Analytics()
    
    cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Config.WINDOW_NAME, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for selfie-view
        frame = cv2.flip(frame, 1)
        
        has_hand = tracker.process_frame(frame)
        if has_hand:
            tracker.draw_landmarks(frame)
            
        landmarks = tracker.get_normalized_landmarks()
        
        if landmarks:
            angles = FingerAngleCalculator.get_finger_angles(landmarks)
            h, w = frame.shape[:2]
            for i, angle in enumerate(angles):
                if FingerAngleCalculator.is_finger_bent(angle):
                    pip_idx = FingerAngleCalculator.FINGER_JOINTS[i][1]
                    px = int(landmarks[pip_idx][0] * w)
                    py = int(landmarks[pip_idx][1] * h)
                    # Draw a warning circle over the bent PIP joint
                    cv2.circle(frame, (px, py), 15, Config.COLOR_WARNING, 2)
                    cv2.circle(frame, (px, py), 8, Config.COLOR_WARNING, -1)
                    
        exercise.update()
        
        # Motion tracking
        smoothed_motions = [0.0] * 5
        if landmarks and exercise.state == State.RECORDING and not exercise.is_paused:
            smoothed_motions = motion.update(landmarks)
            score = ScoreEngine.calculate_independence_score(exercise.current_finger_idx, smoothed_motions)
            if score is not None:
                # Dynamically update score for feedback
                analytics.record_score(exercise.current_finger_idx, score)
                
        # State transitions based on logic
        if exercise.state == State.SCORING:
            exercise.finish_scoring()
            motion.reset()
            
        if exercise.state == State.SUMMARY:
            print("Session Complete! Exporting data...")
            analytics.export_csv()
            analytics.plot_results()
            exercise.restart() # Reset back to IDLE
            analytics.results.clear()

        canvas = vis.create_canvas()
        vis.draw_camera_feed(canvas, frame)
        vis.draw_ui(canvas, exercise.get_instruction_text(), exercise.get_progress(), analytics.results)

        cv2.imshow(Config.WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32: # Spacebar
            if exercise.state == State.IDLE:
                exercise.start()
                analytics.results.clear()
        elif key == ord('p'):
            exercise.pause()
        elif key == ord('r'):
            exercise.restart()
            analytics.results.clear()
        elif key == ord('s'):
            exercise.skip_finger()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
