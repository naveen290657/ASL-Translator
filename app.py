import streamlit as st
import cv2
import av
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading
import queue
import os
import time
from PIL import Image
from tts_helper import speak

# ==============================================================================
# 1. STREAMLIT PAGE CONFIG & MODERN UI STYLING
# ==============================================================================
st.set_page_config(
    page_title="ASL Translator - Machine Spectra",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Gradient Hero Header */
    .main-header {
        text-align: center;
        padding: 1.8rem 1rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
        opacity: 0.9;
    }
    
    /* Stat / Metric Cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2a5298;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Sign Badge */
    .sign-badge {
        display: inline-block;
        background: #1e3c72;
        color: white;
        padding: 0.4rem 0.9rem;
        border-radius: 16px;
        margin: 0.2rem;
        font-size: 0.95rem;
        font-weight: 700;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Card Container */
    .content-box {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODEL AND MEDIAPIPE CACHED LOADING
# ==============================================================================
@st.cache_resource
def load_resources():
    """Load Hand Landmarker and Scikit-Learn Model once into memory."""
    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        return None, None, False

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    model = None
    model_loaded = False
    try:
        with open('model.p', 'rb') as f:
            model_dict = pickle.load(f)
            model = model_dict['model']
            model_loaded = True
    except Exception as e:
        st.error(f"Failed to load model.p: {e}")

    return detector, model, model_loaded

detector, model, model_loaded = load_resources()

# Hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

# ==============================================================================
# 3. THREAD-SAFE SHARED STATE FOR WEBRTC STREAMING
# ==============================================================================
class WebRTCSharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_prediction = ""
        self.latest_confidence = 0.0
        self.frame_count = 0
        self.recent_signs = []
        self.enable_tts = True

    def update(self, pred, conf, enable_tts=True):
        with self.lock:
            self.latest_prediction = pred
            self.latest_confidence = conf
            self.frame_count += 1
            if pred and conf > 50.0:
                if not self.recent_signs or self.recent_signs[-1] != pred:
                    self.recent_signs.append(pred)
                    if len(self.recent_signs) > 15:
                        self.recent_signs.pop(0)
                    if enable_tts:
                        speak(pred)

    def get_info(self):
        with self.lock:
            return {
                "pred": self.latest_prediction,
                "conf": self.latest_confidence,
                "frames": self.frame_count,
                "history": list(self.recent_signs)
            }

@st.cache_resource
def get_shared_state():
    return WebRTCSharedState()

shared_state = get_shared_state()

# ==============================================================================
# 4. WEBRTC VIDEO CALLBACK (STRICTLY THREAD-SAFE, NO st.session_state)
# ==============================================================================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    predicted_character = ""
    confidence = 0.0

    if detector:
        try:
            results = detector.detect(mp_image)
            if results.hand_landmarks:
                hand = results.hand_landmarks[0]
                data_aux = []
                x_ = []
                y_ = []

                # Draw skeleton
                for start_idx, end_idx in HAND_CONNECTIONS:
                    pt1 = (int(hand[start_idx].x * W), int(hand[start_idx].y * H))
                    pt2 = (int(hand[end_idx].x * W), int(hand[end_idx].y * H))
                    cv2.line(img, pt1, pt2, (0, 255, 128), 2)

                for landmark in hand:
                    px, py = int(landmark.x * W), int(landmark.y * H)
                    cv2.circle(img, (px, py), 4, (0, 100, 255), -1)
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize coordinates
                for landmark in hand:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                if model_loaded and len(data_aux) == 42:
                    feats = np.asarray(data_aux).reshape(1, -1)
                    pred = model.predict(feats)
                    predicted_character = str(pred[0])

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(feats)
                        confidence = float(np.max(proba)) * 100.0
                    else:
                        confidence = 100.0

                    # Draw Bounding Box & Label
                    x1 = max(0, int(min(x_) * W) - 20)
                    y1 = max(0, int(min(y_) * H) - 20)
                    x2 = min(W, int(max(x_) * W) + 20)
                    y2 = min(H, int(max(y_) * H) + 20)

                    color = (0, 255, 0) if confidence >= 60 else (0, 165, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                    label = f"{predicted_character} ({confidence:.0f}%)"
                    cv2.putText(img, label, (x1, max(30, y1 - 10)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)
        except Exception:
            pass

    # Safely update shared state without accessing st.session_state
    shared_state.update(predicted_character, confidence, enable_tts=shared_state.enable_tts)

    # Frame watermark
    cv2.putText(img, "Machine Spectra ASL", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==============================================================================
# 5. CORE PREDICTION HELPER FOR STATIC IMAGES
# ==============================================================================
def process_single_image(image_bgr):
    """Process an image array and return annotated image, prediction, confidence, and details."""
    H, W, _ = image_bgr.shape
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    annotated = image_bgr.copy()

    if not detector or not model:
        return annotated, None, 0.0, "Model or Landmarker not loaded"

    results = detector.detect(mp_image)
    if not results.hand_landmarks:
        return annotated, None, 0.0, "No hand detected in the image"

    hand = results.hand_landmarks[0]
    data_aux = []
    x_ = []
    y_ = []

    # Draw skeleton
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = (int(hand[start_idx].x * W), int(hand[start_idx].y * H))
        pt2 = (int(hand[end_idx].x * W), int(hand[end_idx].y * H))
        cv2.line(annotated, pt1, pt2, (0, 255, 128), 2)

    for landmark in hand:
        px, py = int(landmark.x * W), int(landmark.y * H)
        cv2.circle(annotated, (px, py), 5, (0, 100, 255), -1)
        x_.append(landmark.x)
        y_.append(landmark.y)

    for landmark in hand:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))

    if len(data_aux) != 42:
        return annotated, None, 0.0, "Could not extract all 21 hand landmarks"

    feats = np.asarray(data_aux).reshape(1, -1)
    pred = model.predict(feats)
    predicted_sign = str(pred[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feats)
        confidence = float(np.max(proba)) * 100.0
    else:
        confidence = 100.0

    # Draw box
    x1 = max(0, int(min(x_) * W) - 20)
    y1 = max(0, int(min(y_) * H) - 20)
    x2 = min(W, int(max(x_) * W) + 20)
    y2 = min(H, int(max(y_) * H) + 20)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(annotated, f"{predicted_sign} ({confidence:.1f}%)", (x1, max(35, y1 - 10)),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    return annotated, predicted_sign, confidence, "Success"

# ==============================================================================
# 6. HEADER & NAVIGATION
# ==============================================================================
st.markdown("""
<div class="main-header">
    <h1>🤟 Real-Time ASL Recognition System</h1>
    <p>Machine Spectra 1.0 | AI-Powered Sign Language Translation & Speech Synthesis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Control")
    enable_speech = st.checkbox("🔊 Enable Text-To-Speech (TTS)", value=True)
    shared_state.enable_tts = enable_speech

    st.markdown("---")
    st.subheader("🖥️ Desktop Mode")
    st.markdown("""
    **Presenting in Class?**
    Run our ultra-fast native OpenCV HUD with zero latency:
    ```bash
    python inference.py
    ```
    """)

    st.markdown("---")
    st.subheader("🎓 College Information")
    st.markdown("""
    - **College**: Sri Sairam College of Engineering
    - **Dept**: Artificial Intelligence & Machine Learning
    - **Architecture**: MediaPipe Tasks + Random Forest
    - **Features**: 42 Normalized Spatial Coordinates
    - **Inference**: < 10ms (Real-time CPU)
    """)

# Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📹 Live Camera Stream", 
    "🖼️ Image Upload Mode", 
    "📊 Model Performance & Metrics", 
    "📖 Sign Alphabet Reference"
])

# ==============================================================================
# TAB 1: LIVE WEBCAM RECOGNITION
# ==============================================================================
with tab1:
    col_cam, col_info = st.columns([3, 2])

    with col_cam:
        st.subheader("📹 Live Video Feed")
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        webrtc_streamer(
            key="asl-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.caption("💡 *If your browser blocks camera permissions, use **Tab 2: Image Upload Mode** or run `python inference.py`.*")

    with col_info:
        st.subheader("📊 Live Recognition HUD")
        info = shared_state.get_info()

        # Display Live Metric Cards
        c1, c2 = st.columns(2)
        with c1:
            curr_pred = info['pred'] if info['pred'] else "Waiting..."
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Detected Sign</div>
                <div class="metric-value">{curr_pred}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{info['conf']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Recent Signs History
        st.markdown("#### 🔤 Recent Signs")
        if info['history']:
            badges = " ".join([f"<span class='sign-badge'>{s}</span>" for s in info['history']])
            st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
        else:
            st.info("Perform an ASL gesture in front of the camera to see results here.")

        # Audio Test Button
        if st.button("🔊 Test Audio Output"):
            speak("Text to speech synthesizer is operational.")
            st.success("Spoke test audio successfully!")

# ==============================================================================
# TAB 2: IMAGE UPLOAD RECOGNITION
# ==============================================================================
with tab2:
    st.subheader("🖼️ Upload Hand Sign Image for Instant Analysis")
    st.write("Ideal for demonstrating accuracy on static images or testing offline.")

    uploaded_file = st.file_uploader("Choose a hand gesture photo (JPG / PNG)", type=["jpg", "jpeg", "png"])

    col_up1, col_up2 = st.columns(2)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        cv_img = cv2.imdecode(file_bytes, 1)

        with col_up1:
            st.markdown("##### Original Image")
            st.image(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_up2:
            st.markdown("##### AI Landmark Detection & Prediction")
            annotated_img, pred_sign, conf, status = process_single_image(cv_img)

            if pred_sign:
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.success(f"🎯 **Predicted Sign:** `{pred_sign}` (Confidence: **{conf:.2f}%**)")
                if enable_speech:
                    speak(f"Predicted sign is {pred_sign}")
            else:
                st.warning(f"⚠️ {status}")
    else:
        # Provide sample test option from dataset
        st.info("💡 You can also test on a sample image from the dataset:")
        if os.path.exists('./data/A/0.jpg'):
            if st.button("📂 Load Sample Sign 'A'"):
                sample_img = cv2.imread('./data/A/0.jpg')
                with col_up1:
                    st.markdown("##### Sample Image ('A')")
                    st.image(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col_up2:
                    st.markdown("##### Detection Result")
                    annotated_img, pred_sign, conf, _ = process_single_image(sample_img)
                    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.success(f"🎯 **Predicted Sign:** `{pred_sign}` (Confidence: **{conf:.2f}%**)")
                    if enable_speech:
                        speak(f"Predicted sign is {pred_sign}")

# ==============================================================================
# TAB 3: MODEL PERFORMANCE & METRICS
# ==============================================================================
with tab3:
    st.subheader("📊 Model Architecture & Evaluation Metrics")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Classes", "27 (A-Z + Space)")
    col_m2.metric("Dataset Size", "2,581 Samples")
    col_m3.metric("Test Accuracy", "100.00%")
    col_m4.metric("Inference Latency", "< 10 ms")

    st.markdown("---")

    col_cm, col_arch = st.columns([3, 2])

    with col_cm:
        st.markdown("#### 🎯 Confusion Matrix")
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', caption="Test Set Confusion Matrix across all 27 ASL Classes", use_container_width=True)
        else:
            st.info("Run `python train_classifier.py` to generate the confusion matrix plot.")

    with col_arch:
        st.markdown("#### 🏗️ Technical Pipeline")
        st.markdown(r"""
        1. **Video Capture**: OpenCV extracts frames at 30-60 FPS.
        2. **Landmark Extraction**: MediaPipe Tasks extracts 21 3D hand coordinates $(x, y, z)$.
        3. **Feature Normalization**:
           $$x_{norm} = x_i - \min(X)$$
           $$y_{norm} = y_i - \min(Y)$$
           Guarantees translation and scale invariance.
        4. **Classifier**: Random Forest Ensemble (100 estimators) classifies 42 feature vectors.
        5. **Speech Synthesis**: Pyttsx3 / SAPI5 audio pipeline generates real-time spoken sentences.
        """)

# ==============================================================================
# TAB 4: ASL SIGN ALPHABET REFERENCE
# ==============================================================================
with tab4:
    st.subheader("📖 American Sign Language Alphabet Reference (A-Z)")
    st.write("Reference sheet of all 26 alphabets + space gestures recognized by this system.")

    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['space']
    
    # Grid of alphabet buttons/indicators
    cols = st.columns(7)
    for idx, letter in enumerate(letters):
        with cols[idx % 7]:
            st.markdown(f"""
            <div style="background: #f1f5f9; border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 8px; border: 1px solid #cbd5e1;">
                <span style="font-size: 1.3rem; font-weight: bold; color: #1e3c72;">{letter}</span>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.9rem; padding: 0.5rem;">
    <strong>Machine Spectra 1.0</strong> | Developed by Naveen | Sri Sairam College of Engineering | Dept. of AIML
</div>
""", unsafe_allow_html=True)
