# 🤟 Real-Time ASL (American Sign Language) Recognition System

An end-to-end Machine Learning pipeline capable of translating American Sign Language alphabets in real-time using an edge-optimized AI architecture. Built for accessibility, the project includes an offline Text-to-Speech (TTS) synthesizer to bridge the communication gap.

## 🚀 Architecture
This system abandons computationally heavy CNNs in favor of a lightweight landmark-based approach:
1. **MediaPipe Hands:** Extracts 21 3D spatial coordinate landmarks from a live video feed.
2. **Feature Normalization:** Coordinates are mathematically normalized to ensure the model is invariant to hand placement, distance, or screen resolution.
3. **Random Forest Classifier (Scikit-Learn):** A highly optimized decision tree ensemble trains on the structured landmark data, ensuring 95%+ accuracy with sub-10ms inference time on standard CPUs.

## 🛠️ Tech Stack
* **Python 3.x**
* **OpenCV:** Real-time webcam manipulation and bounding-box rendering.
* **MediaPipe:** Hand-tracking framework.
* **Scikit-Learn:** Model training and evaluation.
* **pyttsx3:** Offline Text-to-Speech audio feedback.

## ⚙️ How to Run Locally

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/YOUR_USERNAME/ASL-Recognition-System.git](https://github.com/YOUR_USERNAME/ASL-Recognition-System.git)
cd ASL-Recognition-System
pip install -r requirements.txt