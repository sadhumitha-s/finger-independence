import cv2

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
            palm_normal, _ = analyzer.compute_palm_plane(landmarks, handedness)
            if not analyzer.is_orientation_valid(palm_normal):
                cv2.putText(frame, "INVALID ORIENTATION - FACE PALM TO CAMERA", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_WARNING, 2)
            
            # Visual feedback for active articulation (baseline-relative joint angle change).
            angles, _ = analyzer.compute_metrics(landmarks, palm_normal)
            h, w = frame.shape[:2]
            if analyzer.is_calibrated:
                target_idx = exercise.current_finger_idx
                for i, angle in enumerate(angles):
                    relative_motion = abs(analyzer.baseline_angles[i] - angle)
                    if relative_motion > Config.TARGET_MOTION_HIGHLIGHT_DEG:
                        mcp_idx = HandAnalyzer.FINGER_MCP[i]
                        px = int(landmarks[mcp_idx][0] * w)
                        py = int(landmarks[mcp_idx][1] * h)
                        
                        # GREEN for target finger, RED for leakage (other fingers)
                        color = Config.COLOR_ACCENT if i == target_idx else Config.COLOR_WARNING
                        cv2.circle(frame, (px, py), 15, color, 2)
            else:
                for i, _ in enumerate(angles):
                    mcp_idx = HandAnalyzer.FINGER_MCP[i]
                    px = int(landmarks[mcp_idx][0] * w)
                    py = int(landmarks[mcp_idx][1] * h)
                    cv2.circle(frame, (px, py), 8, Config.COLOR_ACCENT, 1)
                    
        exercise.update()

        # Always start calibration from a clean buffer to avoid stale frame carry-over.
        if previous_state != State.CALIBRATE and exercise.state == State.CALIBRATE:
            calibration_frames.clear()
        
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
                angles, _ = analyzer.compute_metrics(landmarks, palm_normal)
                _, _, motion_values = motion.update(angles, analyzer.baseline_angles)
                target_idx = exercise.current_finger_idx
                leakage = score_engine.calculate_frame_leakage(target_idx, motion_values)
                analytics.record_leakage(target_idx, leakage)
                
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
                calibration_frames.clear()
        elif key == ord('p'):
            exercise.pause()
        elif key == ord('r'):
            exercise.restart()
            analytics.reset()
            calibration_frames.clear()
        elif key == ord('s'):
            exercise.skip_finger()

        previous_state = exercise.state

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
