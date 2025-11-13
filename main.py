import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils import detect_face, init_webcam_and_detector

EXPRESSIONS = ["mouth_open", "neutral"]
IMG_SIZE = (224, 224)


def main():
    haar_cascade, cap = init_webcam_and_detector()

    # Load model
    model = tf.keras.models.load_model("nailong_exp_model.keras")

    # Data collection loop
    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)  # Mirror the frame

        face = detect_face(frame, haar_cascade, IMG_SIZE)  # Detect face
        pred_text = "No face detected"

        if face is not None:
            face_array = preprocess_input(np.expand_dims(
                face, axis=0))  # Preprocess, add batch dimension
            preds = model.predict(face_array)
            pred_exp = EXPRESSIONS[np.argmax(preds)]
            pred_text = f"Expression: {pred_exp}"

        cv2.putText(frame, pred_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Expression Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
