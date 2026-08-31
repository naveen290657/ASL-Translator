import os
import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize Tasks API Hand Landmarker
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    raise FileNotFoundError("Missing 'hand_landmarker.task'. Run the download command first.")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

DATA_DIR = './data'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue

    for img_path in os.listdir(class_dir):
        data_aux = []
        x_ = []
        y_ = []

        img_full_path = os.path.join(class_dir, img_path)
        img = cv2.imread(img_full_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            # Process only the primary hand detected
            hand = detection_result.hand_landmarks[0]

            for landmark in hand:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            # Strictly verify 42 features (21 landmarks * 2 coords)
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Features extracted! Saved {len(data)} samples across {len(set(labels))} classes to data.pickle.")