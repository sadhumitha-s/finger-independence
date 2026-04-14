import cv2
import numpy as np

from config import Config
from hand_tracker import HandTracker
from motion_tracker import MotionTracker
from score_engine import ScoreEngine
from exercise_mode import ExerciseMode, State
from visualizer import Visualizer
from analytics import Analytics
from analyzer import HandAnalyzer

def main():
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {Config.CAMERA_INDEX}")
        return

    tracker = HandTracker()
    motion = MotionTracker()
    analyzer = HandAnalyzer()
    score_engine = ScoreEngine()
    exercise = ExerciseMode()
    vis = Visualizer()
    analytics = Analytics()
    
    calibration_frames = []
    
    cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Config.WINDOW_NAME, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
    previous_state = exercise.state

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for selfie-view
        frame = cv2.flip(frame, 1)
        
        has_hand = tracker.process_frame(frame)
        if has_hand:
            tracker.draw_landmarks(frame)
            
        landmarks, handedness = tracker.get_normalized_landmarks()
        
        # Orientation and Metrics
        palm_normal = None
        if landmarks and handedness:
            palm_normal, wrist_pt = analyzer.compute_palm_plane(landmarks, handedness)
            if not analyzer.is_orientation_valid(palm_normal):
                cv2.putText(frame, "INVALID ORIENTATION - FACE PALM TO CAMERA", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_WARNING, 2)
            
            # Simple visual feedback for bending (legacy)
            # We can still use FingerAngleCalculator for simple dot drawing if needed, 
            # but let's stick to the new analyzer's lift measurements.
            lifts, heights, dirs = analyzer.compute_metrics(landmarks, palm_normal)
            h, w = frame.shape[:2]
            for i, lift in enumerate(lifts):
                # With camera-facing normal, LIFTING (away from camera) is POSITIVE relative lift
                relative_lift = lift - analyzer.baseline_lifts[i]
                # visual feedback based on lifting constraint
                if relative_lift > Config.TARGET_LIFT_TRIGGER:
                    mcp_idx = HandAnalyzer.FINGER_MCP[i]
                    px = int(landmarks[mcp_idx][0] * w)
                    py = int(landmarks[mcp_idx][1] * h)
                    cv2.circle(frame, (px, py), 15, Config.COLOR_ACCENT, 2)
                    
        exercise.update()
        
        # Calibration logic
        if landmarks and exercise.state == State.CALIBRATE and not exercise.is_paused:
            if palm_normal is not None and analyzer.is_orientation_valid(palm_normal):
                calibration_frames.append(landmarks)
                if len(calibration_frames) >= Config.CALIBRATION_FRAMES:
                    analyzer.calibrate(calibration_frames, handedness)
                    calibration_frames.clear()
                    exercise._change_state(State.PREPARE)

        if previous_state != State.RECORDING and exercise.state == State.RECORDING:
            analytics.begin_finger_recording(exercise.current_finger_idx)

        # Motion tracking
        if landmarks and exercise.state == State.RECORDING and not exercise.is_paused:
            if palm_normal is not None and analyzer.is_orientation_valid(palm_normal):
                lifts, heights, dirs = analyzer.compute_metrics(landmarks, palm_normal)
                sideways_motions = analyzer.get_sideways_motion(dirs, palm_normal)
                relative_lifts, delta_heights, tip_motions = motion.update(
                    lifts, analyzer.baseline_lifts, 
                    heights, analyzer.baseline_heights
                )
                
                score = score_engine.calculate_independence_score(
                    exercise.current_finger_idx, relative_lifts, tip_motions, delta_heights, sideways_motions
                )
                
                target_idx = exercise.current_finger_idx
                target_lift = abs(relative_lifts[target_idx])
                target_tip = abs(tip_motions[target_idx])
                analytics.record_score(
                    target_idx,
                    score,
                    target_lift=target_lift,
                    target_tip=target_tip,
                )
                
        # State transitions based on logic
        if exercise.state == State.SCORING:
            analytics.finalize_finger(exercise.current_finger_idx)
            exercise.finish_scoring()
            motion.reset()
            
        if exercise.state == State.SUMMARY:
            print("Session Complete! Exporting data...")
            analytics.export_csv()
            analytics.plot_results()
            exercise.restart() # Reset back to IDLE
            analytics.reset()

        is_valid = analyzer.is_orientation_valid(palm_normal) if palm_normal is not None else True
        canvas = vis.create_canvas()
        vis.draw_camera_feed(canvas, frame)
        vis.draw_ui(
            canvas,
            exercise.get_instruction_text(),
            exercise.get_progress(),
            analytics.results,
            is_valid
        )

        cv2.imshow(Config.WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32: # Spacebar
            if exercise.state == State.IDLE:
                exercise.start()
                analytics.reset()
        elif key == ord('p'):
            exercise.pause()
        elif key == ord('r'):
            exercise.restart()
            analytics.reset()
        elif key == ord('s'):
            exercise.skip_finger()

        previous_state = exercise.state

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
