import pickle
import threading
import cv2
import numpy as np
import pyttsx3
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Initialize Text-to-Speech Engine
try:
    engine = pyttsx3.init()
    def speak(text):
        engine.say(text)
        engine.runAndWait()
except Exception:
    def speak(text):
        pass

# 2. Load your pre-trained model (Safe to use your existing model.p)
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# 3. Initialize MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Manually defining connections so we don't need the legacy drawing_utils
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

cap = cv2.VideoCapture(0)

# Sentence Building Variables
sentence = ""
current_word = ""
last_predicted = ""
stable_frames = 0
FRAMES_TO_CONFIRM = 10 # Hold the sign for ~10 frames to confirm it

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    
    # Convert to RGB and MediaPipe Image format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect hands
    detection_result = detector.detect(mp_image)
    predicted_character = ""

    if detection_result.hand_landmarks:
        hand = detection_result.hand_landmarks[0]
        data_aux = []
        x_ = []
        y_ = []

        # Draw the hand connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (int(hand[start_idx].x * W), int(hand[start_idx].y * H))
            pt2 = (int(hand[end_idx].x * W), int(hand[end_idx].y * H))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw the points and extract coordinates
        for landmark in hand:
            px, py = int(landmark.x * W), int(landmark.y * H)
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
            x_.append(landmark.x)
            y_.append(landmark.y)

        # Normalize the coordinates (Exactly like your training data)
        for landmark in hand:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        # Ensure we have exactly 42 features (21 points * 2 axes)
        if len(data_aux) == 42:
            # Create bounding box
            x1 = max(0, int(min(x_) * W) - 20)
            y1 = max(0, int(min(y_) * H) - 20)
            x2 = min(W, int(max(x_) * W) + 20)
            y2 = min(H, int(max(y_) * H) + 20)

            # Predict the character
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = str(prediction[0])

            # Draw bounding box and live prediction above the hand
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # 4. Debouncing and Sentence Building Logic
    if predicted_character != "":
        if predicted_character == last_predicted:
            stable_frames += 1
        else:
            stable_frames = 0
            last_predicted = predicted_character

        # If a sign is held steady for the required frames
        if stable_frames == FRAMES_TO_CONFIRM:
            if predicted_character.lower() == 'space':
                if current_word:
                    # Speak the completed word on a background thread
                    threading.Thread(target=speak, args=(current_word,), daemon=True).start()
                sentence += " "
                current_word = ""
            else:
                sentence += predicted_character
                current_word += predicted_character
            
            # Reset the counter so it waits before repeating the same letter
            stable_frames = 0
            last_predicted = ""

    # 5. Draw the sentence UI at the bottom of the screen
    cv2.rectangle(frame, (0, H - 60), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, f"Text: {sentence}", (20, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(frame, "Press 'C' to clear text | 'Q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # Display the final frame
    cv2.imshow('ASL Translation System', frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'): # Clear the sentence manually
        sentence = ""
        current_word = ""

cap.release()
cv2.destroyAllWindows()