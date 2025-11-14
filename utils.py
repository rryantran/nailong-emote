import os
import cv2


def setup_directories(categories, base_dir):
    """
    Create directories for dataset

    :param categories: List of category names
    :param base_dir: Base directory path to create category directories in
    """

    os.makedirs(base_dir, exist_ok=True)

    for category in categories:
        os.makedirs(os.path.join(base_dir, category), exist_ok=True)


def detect_face(frame, haar_cascade, img_size):
    """
    Detect face using Haar Cascade and return processed image if detected

    :param frame: Input image frame
    :param haar_cascade: Pre-loaded Haar Cascade classifier
    :param img_size: Output image size (width, height)

    :return: Processed image or None if no face detected
    """

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
    """
    Initialize webcam and Haar Cascade classifier

    :return: Tuple of (Haar Cascade classifier, VideoCapture object)
    """

    haar_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    return haar_cascade, cap
