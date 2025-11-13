import os
import cv2


def setup_directories(expressions, base_dir):
    """Create directories for dataset"""

    os.makedirs(base_dir, exist_ok=True)

    for exp in expressions:
        os.makedirs(os.path.join(base_dir, exp), exist_ok=True)


def detect_face(frame, haar_cascade, img_size):
    """Detect face and return processed image if detected"""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    faces = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    face_img = None  # Default if no face detected

    for (x, y, w, h) in faces:
        # Draw rectangle around face for visualization
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        face = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face, img_size)

    return face_img


def init_webcam_and_detector():
    """Initialize webcam and Haar Cascade detector"""
    haar_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    return haar_cascade, cap
