import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Full A-Z alphabet + 'space' character
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'space'
]
dataset_size = 100 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("=" * 50)
print("ASL Dataset Collector")
print("Instructions: Position your hand and press 'Q' to record each sign.")
print("=" * 50)

for label in labels:
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Ready to collect data for class: {label}')
    
    # Wait for user readiness
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f'Next Sign: {label}', (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Position hand & press "Q" to start capture', (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.imshow('ASL Data Collection', frame)
        
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == 27: # ESC to abort
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    # Capture images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f'Recording {label}: {counter + 1}/{dataset_size}', (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('ASL Data Collection', frame)
        cv2.waitKey(25)
        
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

print("\nData collection complete for all signs!")
cap.release()
cv2.destroyAllWindows()