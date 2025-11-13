import os
import cv2
from utils import setup_directories, detect_face

# Dataset parameters
EXPRESSIONS = ["mouth_open", "neutral", "smile", "tongue_out"]
IMG_SIZE = (224, 224)
NUM_IMAGES = 200
BASE_DIR = "dataset"


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
