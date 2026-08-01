import sys
import os
import subprocess

import cv2
import os
import av
import queue
import threading
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from finger_independence.config import Config
from finger_independence.hand_tracker import HandTracker
from finger_independence.motion_tracker import MotionTracker
from finger_independence.exercise_mode import ExerciseMode, State
from finger_independence.visualizer import Visualizer
from finger_independence.analytics import Analytics
from finger_independence.analyzer import HandAnalyzer

import hashlib
from finger_independence.db_client import db

st.set_page_config(page_title="Finger Independence", layout="wide")

from finger_independence.auth import check_password

if not check_password():
    st.stop()

st.title("Finger Independence Tracker")
st.markdown("Follow the instructions on the video feed.")

import json
import urllib.request

@st.cache_data
def get_ice_servers():
    """Use Metered TURN server if credentials are set, otherwise fallback to Google STUN."""
    try:
        metered_domain = os.environ.get("METERED_DOMAIN") 
        metered_api_key = os.environ.get("METERED_API_KEY")
        if metered_domain and metered_api_key:
            url = f"https://{metered_domain}/api/v1/turn/credentials?apiKey={metered_api_key}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to get Metered ICE servers: {e}")
        
    # 2. Fallback to Google STUN
    return [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": get_ice_servers()}
)

class FingerProcessor:
    def __init__(self):
        self.tracker = HandTracker()
        self.motion = MotionTracker()
        self.analyzer = HandAnalyzer()
        self.exercise = ExerciseMode()
        self.vis = Visualizer()
        self.analytics = Analytics()
        
        self.calibration_frames = []
        self.previous_state = self.exercise.state
        
        # Thread-safe queues for commands from Streamlit UI
        self.command_queue = queue.Queue()
        # Thread-safe queue for results to Streamlit UI
        self.result_queue = queue.Queue()
        
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        try:
            # Process UI commands safely
            with self.lock:
                while not self.command_queue.empty():
                    cmd = self.command_queue.get_nowait()
                    if isinstance(cmd, tuple) and cmd[0] == "START":
                        if self.exercise.state == State.IDLE:
                            self.exercise.start()
                            self.analytics.reset(user_id=cmd[1])
                            self.calibration_frames.clear()
                    elif cmd == "PAUSE":
                        self.exercise.pause()
                    elif isinstance(cmd, tuple) and cmd[0] == "RESTART":
                        self.exercise.restart()
                        self.analytics.reset(user_id=cmd[1])
                        self.calibration_frames.clear()
                    elif cmd == "SKIP":
                        self.exercise.skip_finger()

            # Hand Tracking
            has_hand = self.tracker.process_frame(img)
            if has_hand:
                self.tracker.draw_landmarks(img)
                
            landmarks, handedness = self.tracker.get_normalized_landmarks()
            palm_normal = None
            
            with self.lock:
                if landmarks and handedness:
                    palm_normal, _ = self.analyzer.compute_palm_plane(landmarks, handedness)
                    if not self.analyzer.is_orientation_valid(palm_normal):
                        cv2.putText(img, "INVALID ORIENTATION - FACE PALM TO CAMERA", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_WARNING, 2)
                    
                    angles, _ = self.analyzer.compute_metrics(landmarks, palm_normal)
                    h, w = img.shape[:2]
                    if self.analyzer.is_calibrated:
                        target_idx = self.exercise.current_finger_idx
                        for i, angle in enumerate(angles):
                            relative_motion = abs(self.analyzer.baseline_angles[i] - angle)
                            if relative_motion > Config.TARGET_MOTION_HIGHLIGHT_DEG:
                                mcp_idx = HandAnalyzer.FINGER_MCP[i]
                                px = int(landmarks[mcp_idx][0] * w)
                                py = int(landmarks[mcp_idx][1] * h)
                                color = Config.COLOR_ACCENT if i == target_idx else Config.COLOR_WARNING
                                cv2.circle(img, (px, py), 15, color, 2)
                    else:
                        for i, _ in enumerate(angles):
                            mcp_idx = HandAnalyzer.FINGER_MCP[i]
                            px = int(landmarks[mcp_idx][0] * w)
                            py = int(landmarks[mcp_idx][1] * h)
                            cv2.circle(img, (px, py), 8, Config.COLOR_ACCENT, 1)
                            
                self.exercise.update()

                if self.previous_state != State.CALIBRATE and self.exercise.state == State.CALIBRATE:
                    self.calibration_frames.clear()
                
                if landmarks and self.exercise.state == State.CALIBRATE and not self.exercise.is_paused:
                    if palm_normal is not None and self.analyzer.is_orientation_valid(palm_normal):
                        self.calibration_frames.append(landmarks)
                        if len(self.calibration_frames) >= Config.CALIBRATION_FRAMES:
                            self.analyzer.calibrate(self.calibration_frames, handedness)
                            self.calibration_frames.clear()
                            self.exercise._change_state(State.PREPARE)

                if self.previous_state != State.RECORDING and self.exercise.state == State.RECORDING:
                    self.analytics.begin_finger_recording(self.exercise.current_finger_idx)

                if landmarks and self.exercise.state == State.RECORDING and not self.exercise.is_paused:
                    if palm_normal is not None and self.analyzer.is_orientation_valid(palm_normal):
                        angles, _ = self.analyzer.compute_metrics(landmarks, palm_normal)
                        _, _, motion_values = self.motion.update(angles, self.analyzer.baseline_angles)
                        target_idx = self.exercise.current_finger_idx
                        self.analytics.record_leakage(target_idx, motion_values)
                        
                if self.exercise.state == State.SCORING:
                    self.analytics.finalize_finger(self.exercise.current_finger_idx)
                    self.exercise.finish_scoring()
                    self.motion.reset()
                    
                if self.exercise.state == State.SUMMARY:
                    print("Session Complete! Exporting data...")
                    self.analytics.export_csv()
                    fig = self.analytics.plot_results()
                    # Send figure back to Streamlit UI thread
                    self.result_queue.put(fig)
                    self.exercise.restart()
                    self.analytics.reset(user_id=self.analytics.user_id)
                    
                    import gc
                    gc.collect()

                self.previous_state = self.exercise.state

                is_valid = self.analyzer.is_orientation_valid(palm_normal) if palm_normal is not None else True
                
                self.vis.draw_ui(
                    img,
                    self.exercise.get_instruction_text(),
                    self.exercise.get_progress(),
                    self.analytics.results,
                    is_valid
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            cv2.putText(img, "Processing Error", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, Config.COLOR_WARNING, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Dashboard
with st.expander("My Dashboard", expanded=True):
    recent_sessions = db.get_recent_sessions(st.session_state.get("current_user", "guest"), limit=5)
    if not recent_sessions:
        st.info("No previous sessions found. Start your first session!")
    else:
        st.write("Your recent sessions:")
        for sess in recent_sessions:
            score = sess.get('final_independence_score')
            score_str = f"{score:.2f}" if score is not None else "Incomplete"
            date_str = str(sess.get('created_at', ''))[:10]
            
            sess_data = sess.get('enslavement_matrix_json')
            if isinstance(sess_data, dict) and 'scores' in sess_data:
                scores = sess_data['scores']
                from finger_independence.config import Config
                scores_str = " | ".join([f"{Config.FINGERS[int(k)][:3]}: {v:.2f}" for k, v in scores.items()])
                st.write(f"- **{date_str}**: Total Score: **{score_str}**")
                st.write(f"  - {scores_str}")
            else:
                st.write(f"- **{date_str}**: Score: **{score_str}**")

webrtc_ctx = webrtc_streamer(
    key="finger-tracker",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=FingerProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280, "min": 640},
            "height": {"ideal": 720, "min": 480},
        },
        "audio": False,
    },
    async_processing=True,
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

if webrtc_ctx.video_processor:
    if col1.button("Start Session", use_container_width=True):
        webrtc_ctx.video_processor.command_queue.put(("START", st.session_state["current_user"]))
    if col2.button("Pause / Resume", use_container_width=True):
        webrtc_ctx.video_processor.command_queue.put("PAUSE")
    if col3.button("Skip Finger", use_container_width=True):
        webrtc_ctx.video_processor.command_queue.put("SKIP")
    if col4.button("Restart", use_container_width=True):
        webrtc_ctx.video_processor.command_queue.put(("RESTART", st.session_state["current_user"]))
        
    # Check if there is a summary plot
    while not webrtc_ctx.video_processor.result_queue.empty():
        fig = webrtc_ctx.video_processor.result_queue.get_nowait()
        if fig:
            st.session_state.final_figure = fig
            
if 'final_figure' in st.session_state:
    st.subheader("Session Results")
    st.pyplot(st.session_state.final_figure)
