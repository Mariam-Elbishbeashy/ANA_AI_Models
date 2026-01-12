#!/usr/bin/env python3
# -- coding: utf-8 --

import cv2 as cv
import numpy as np
from keras.models import load_model

# ================== PATHS ==================
EMOTION_MODEL_PATH = "model_file_30epochs.h5"
PROTOTXT_PATH = "deploy.prototxt"
CAFFEMODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

FACE_BOX_COLOR = (255, 120, 60)
EMOTION_BOX_COLOR = (202, 141, 145)

# ================== LOAD MODELS ==================
print("🔍 Loading face detector and emotion model...")

emotion_model = load_model(EMOTION_MODEL_PATH)
face_net = cv.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

print("✅ Models loaded successfully")

# ================== EMOTION LABELS ==================
labels_dict = {
    0: "Angry",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

# ================== CAMERA SETUP ==================
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

print("\n🎭 REAL-TIME FACE EMOTION RECOGNITION")
print("Press 'q' or ESC to quit")

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    (h, w) = frame.shape[:2]

    # -------- FACE DETECTION --------
    blob = cv.dnn.blobFromImage(
        cv.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    current_emotion = "Neutral"

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.85:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if (x2 - x1) < 50 or (y2 - y1) < 50:
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            face = gray[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_resized = cv.resize(face, (48, 48))
            face_normalized = face_resized / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

            preds = emotion_model.predict(face_reshaped, verbose=0)
            label = int(np.argmax(preds))
            current_emotion = labels_dict.get(label, "Unknown")

            # -------- DRAW --------
            cv.rectangle(frame, (x1, y1), (x2, y2), (226, 144, 74), 2)
            cv.rectangle(frame, (x1, y1 - 30), (x2, y1), (226, 144, 74), -1)
            cv.putText(
                frame,
                current_emotion,
                (x1 + 5, y1 - 8),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )

    # -------- INFO PANEL --------
    cv.rectangle(frame, (10, 10), (300, 60), (202, 141, 145), -1)
    cv.rectangle(frame, (10, 10), (300, 60), (255, 255, 255), 1)

    cv.putText(
        frame,
        f"Emotion: {current_emotion}",
        (20, 45),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),  # white text
        2,
        cv.LINE_AA
    )


    cv.imshow("Face Emotion Recognition", frame)

    key = cv.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

# ================== CLEANUP ==================
cap.release()
cv.destroyAllWindows()
print("👋 Exited cleanly")
