import streamlit as st
import cv2
import av
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="ASL Translator", layout="wide")
st.title("Machine Spectra 1.0 - ASL Translator")
st.write("Turn on your webcam and show a sign to the camera.")

# ==========================================
# 1. UPGRADED: NEW MEDIAPIPE TASKS API
# ==========================================
@st.cache_resource
def load_models():
    # Load the modern MediaPipe Tasks model (using your hand_landmarker.task file)
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Load your Scikit-Learn model
    try:
        model_dict = pickle.load(open('model.p', 'rb'))
        model = model_dict['model']
        model_loaded = True
    except Exception:
        model = None
        model_loaded = False
        
    return detector, model, model_loaded

detector, model, model_loaded = load_models()

if not model_loaded:
    st.warning("Warning: 'model.p' not found or failed to load. Showing hand tracking only.")

# ==========================================
# 2. WEBRTC CONFIGURATION
# ==========================================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Custom hand skeleton drawing (replaces the removed solutions.drawing_utils)
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (0,17), (17,18), (18,19), (19,20)
]

# ==========================================
# 3. VIDEO PROCESSING LOOP
# ==========================================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert image for the new Tasks API
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    try:
        results = detector.detect(mp_image)
    except Exception:
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # 1. Draw the custom skeleton lines
            for connection in HAND_CONNECTIONS:
                pt1 = hand_landmarks[connection[0]]
                pt2 = hand_landmarks[connection[1]]
                x1_line, y1_line = int(pt1.x * W), int(pt1.y * H)
                x2_line, y2_line = int(pt2.x * W), int(pt2.y * H)
                cv2.line(img, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 0), 2)
            
            # 2. Draw the custom skeleton joints
            for lm in hand_landmarks:
                cx, cy = int(lm.x * W), int(lm.y * H)
                cv2.circle(img, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

            # 3. Extract coordinates
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks)):
                x_.append(hand_landmarks[i].x)
                y_.append(hand_landmarks[i].y)

            for i in range(len(hand_landmarks)):
                data_aux.append(hand_landmarks[i].x - min(x_))
                data_aux.append(hand_landmarks[i].y - min(y_))

            # 4. Make prediction
            if model_loaded:
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = str(prediction[0])
                    
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    # Draw bounding box and text
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(img, predicted_character, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                except Exception:
                    pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the WebRTC streamer
webrtc_streamer(
    key="asl-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)