import os
import cv2

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

cap = cv2.VideoCapture(0)

for label in labels:
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {label}')
    
    # Wait for the user to press 'Q' to start capturing
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Press "Q" to record: {label}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Rapidly capture 100 images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
print("Full A-Z data collection complete!")