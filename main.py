import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils import detect_face, init_webcam_and_detector

EXPRESSIONS = ["mouth_open", "neutral"]
IMG_SIZE = (224, 224)

haar_cascade, cap = init_webcam_and_detector()
model = tf.keras.models.load_model("nailong_exp_model.keras")

neutral_img = cv2.imread("images/neutral.jpg")
mouth_open_img = cv2.imread("images/mouth_open.jpg")
display_img = None

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)  # Mirror the frame (more natural for user)

    face = detect_face(frame, haar_cascade, IMG_SIZE)
    pred_text = "No face detected"

    if face is not None:
        face_array = preprocess_input(np.expand_dims(
            face, axis=0))  # Preprocess, add batch dim
        preds = model.predict(face_array, verbose=0)
        pred_exp = EXPRESSIONS[np.argmax(preds)]
        scores = " | ".join(
            [f"{exp}: {preds[0][i]:.2f}" for i, exp in enumerate(EXPRESSIONS)])
        pred_text = f"Expression: {pred_exp} || Scores: {scores}"

        if pred_exp == "mouth_open":
            display_img = mouth_open_img
        elif pred_exp == "neutral":
            display_img = neutral_img

        # Resize display image for consistency
        display_img = cv2.resize(display_img, IMG_SIZE)

    cv2.putText(frame, pred_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

    cv2.imshow("Webcam", frame)

    if display_img is not None:
        cv2.imshow("Nailong", display_img)

    # q (quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
