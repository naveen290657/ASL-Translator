import time
import cv2
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tts_helper import speak

# ==============================================================================
# 1. LOAD MODEL & MEDIAPIPE DETECTOR
# ==============================================================================
print("Loading trained ASL model...")
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

print("Initializing MediaPipe HandLandmarker...")
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

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
# 2. INITIALIZE CAMERA
# ==============================================================================
print("Starting Camera...")
# Try DirectShow first for fast startup on Windows, fallback to default
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. Please check your camera connection.")
    exit(1)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Welcome announcement via TTS
speak("Sign Language Translator is ready.")

# ==============================================================================
# 3. STATE VARIABLES
# ==============================================================================
sentence = ""
current_word = ""
last_predicted = ""
stable_frames = 0
FRAMES_TO_CONFIRM = 10  # Hold sign steady for ~10 frames (~0.3 sec) to commit

prev_time = time.time()
fps = 0

def draw_styled_box(img, pt1, pt2, color, thickness=2, corner_len=15):
    """Draw a modern rounded-look bounding box with accented corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Base rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # Highlighted corner corners
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness * 2)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness * 2)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness * 2)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness * 2)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness * 2)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness * 2)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness * 2)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness * 2)

# ==============================================================================
# 4. MAIN INFERENCE LOOP
# ==============================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for intuitive selfie mirror view
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    # Calculate real-time FPS
    curr_time = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
    prev_time = curr_time

    # Convert to RGB for MediaPipe Tasks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect hands
    detection_result = detector.detect(mp_image)
    predicted_character = ""
    confidence = 0.0

    if detection_result.hand_landmarks:
        hand = detection_result.hand_landmarks[0]
        data_aux = []
        x_ = []
        y_ = []

        # Draw skeleton connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (int(hand[start_idx].x * W), int(hand[start_idx].y * H))
            pt2 = (int(hand[end_idx].x * W), int(hand[end_idx].y * H))
            cv2.line(frame, pt1, pt2, (0, 255, 128), 2)

        # Collect landmarks and draw joint circles
        for landmark in hand:
            px, py = int(landmark.x * W), int(landmark.y * H)
            cv2.circle(frame, (px, py), 4, (0, 100, 255), -1)
            x_.append(landmark.x)
            y_.append(landmark.y)

        # Coordinate Normalization: (coord - min(coord))
        for landmark in hand:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        # Model Inference
        if len(data_aux) == 42:
            try:
                features = np.asarray(data_aux).reshape(1, -1)
                prediction = model.predict(features)
                predicted_character = str(prediction[0])

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)
                    confidence = float(np.max(probabilities)) * 100.0
                else:
                    confidence = 100.0

                # Bounding box coordinates with padding
                x1 = max(0, int(min(x_) * W) - 25)
                y1 = max(0, int(min(y_) * H) - 25)
                x2 = min(W, int(max(x_) * W) + 25)
                y2 = min(H, int(max(y_) * H) + 25)

                # Visual badge styling based on confidence
                box_color = (0, 255, 0) if confidence >= 70 else (0, 165, 255)
                draw_styled_box(frame, (x1, y1), (x2, y2), box_color, 2)

                # Draw prediction label banner above box
                label_text = f"{predicted_character} ({confidence:.0f}%)"
                (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
                badge_y1 = max(30, y1 - 10)
                cv2.rectangle(frame, (x1, badge_y1 - label_h - 8), (x1 + label_w + 12, badge_y1 + 4), box_color, -1)
                cv2.putText(frame, label_text, (x1 + 6, badge_y1 - 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            except Exception as err:
                pass

    # ==============================================================================
    # 5. SIGN DEBOUNCING & SENTENCE ACCUMULATION
    # ==============================================================================
    if predicted_character != "" and confidence >= 50.0:
        if predicted_character == last_predicted:
            stable_frames += 1
        else:
            stable_frames = 1
            last_predicted = predicted_character

        # Progress bar toward confirmation
        progress = min(1.0, stable_frames / FRAMES_TO_CONFIRM)
        bar_w = int(200 * progress)
        cv2.rectangle(frame, (W - 220, 60), (W - 20, 75), (50, 50, 50), -1)
        cv2.rectangle(frame, (W - 220, 60), (W - 220 + bar_w, 75), (0, 255, 0), -1)
        cv2.putText(frame, f"Holding: {last_predicted}", (W - 220, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Confirm character once held long enough
        if stable_frames >= FRAMES_TO_CONFIRM:
            if predicted_character.lower() == 'space':
                if current_word:
                    speak(current_word)
                sentence += " "
                current_word = ""
            else:
                sentence += predicted_character
                current_word += predicted_character

            # Reset debounce
            stable_frames = 0
            last_predicted = ""

    # ==============================================================================
    # 6. PROFESSIONAL PRESENTATION HUD
    # ==============================================================================
    # Top HUD Bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 45), (20, 20, 25), -1)
    # Bottom HUD Bar
    cv2.rectangle(overlay, (0, H - 90), (W, H), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Top Bar Text
    cv2.putText(frame, "REAL-TIME ASL RECOGNITION SYSTEM", (20, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)}", (W - 120, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 128), 1, cv2.LINE_AA)

    # Bottom Bar: Sentence & Current Word
    display_sentence = sentence if sentence else "[Start signing or hold letters...]"
    cv2.putText(frame, f"Sentence: {display_sentence}", (20, H - 52),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Key Controls Guide
    controls_text = "[SPACE] Word/Space  |  [S] Speak All  |  [BACKSPACE] Undo  |  [C] Clear  |  [Q] Exit"
    cv2.putText(frame, controls_text, (20, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # Display Output
    cv2.imshow('Real-Time ASL Recognition (Machine Spectra)', frame)

    # ==============================================================================
    # 7. KEYBOARD CONTROLS
    # ==============================================================================
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'), ord('Q'), 27):  # Q or ESC to quit
        break
    elif key in (ord('c'), ord('C')):    # C to clear
        sentence = ""
        current_word = ""
    elif key in (ord('s'), ord('S')):    # S to speak full sentence
        if sentence.strip():
            speak(sentence.strip())
    elif key == ord(' '):                # Manual Space
        if current_word:
            speak(current_word)
        sentence += " "
        current_word = ""
    elif key in (8, 127):                # Backspace / Delete
        if len(sentence) > 0:
            sentence = sentence[:-1]
        if len(current_word) > 0:
            current_word = current_word[:-1]

# ==============================================================================
# 8. CLEANUP
# ==============================================================================
cap.release()
cv2.destroyAllWindows()
print("ASL Recognition session closed cleanly.")