import os
import cv2

# Dataset parameters
EXPRESSIONS = ["neutral", "smile", "tongue_out", "mouth_open"]
IMG_SIZE = (224, 224)
NUM_IMAGES = 200
BASE_DIR = "dataset"


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


def main():
    setup_directories(EXPRESSIONS, BASE_DIR)

    # Load Haar Cascade for face detection
    haar_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Webcam initialization
    cap = cv2.VideoCapture(0)

    # Data collection loop
    for exp in EXPRESSIONS:
        print(f"\nCollecting images for: {exp}")

        # Start count from existing images
        exp_dir = os.path.join(BASE_DIR, exp)
        count = len([f for f in os.listdir(exp_dir) if f.endswith('.jpg')])

        while count < NUM_IMAGES:
            ret, frame = cap.read()

            frame = cv2.flip(frame, 1)  # Mirror the frame
            face = detect_face(frame, haar_cascade, IMG_SIZE)

            # Wait for key stroke
            key = cv2.waitKey(1)

            if key == 32:  # Space (capture image)
                save_path = os.path.join(BASE_DIR, exp, f"{count}.jpg")
                cv2.imwrite(save_path, face)
                count += 1
                print(f"Saved {count}/{NUM_IMAGES}")

            elif key == ord('q'):  # q (quit)
                cap.release()
                cv2.destroyAllWindows()
                exit()

            elif key == ord('n'):  # n (next expression)
                print(f"Skipping to next expression...")
                break

            cv2.putText(frame, f"{exp}: {count}/{NUM_IMAGES}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

            cv2.imshow("Data Collection", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
