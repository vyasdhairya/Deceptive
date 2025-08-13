import streamlit as st
import requests
import os
MODEL_URL = "https://github.com/ageitgey/face_recognition_models/raw/refs/heads/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat"
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading shape_predictor_68_face_landmarks.dat ...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.write("Download complete ✅")
    else:
        st.write("Model already exists locally ✅")
# Run download at startup
download_model()
import streamlit as st
import cv2
import dlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tempfile import NamedTemporaryFile

# -------------------------
# Load model and detector
# -------------------------
@st.cache_resource
def load_model_and_detector():
    model = tf.keras.models.load_model("mode_own.h5")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return model, detector, predictor

model, detector, predictor = load_model_and_detector()

# -------------------------
# Feature extraction function
# -------------------------
def extract_features(landmarks):
    features = {}
    mouth_width = np.linalg.norm(np.array([landmarks[54].x, landmarks[54].y]) - np.array([landmarks[48].x, landmarks[48].y]))
    mouth_height = np.linalg.norm(np.array([landmarks[66].x, landmarks[66].y]) - np.array([landmarks[62].x, landmarks[62].y]))
    
    features['Smile'] = 1 if mouth_width / mouth_height > 2 else 0
    features['Laugh'] = 1 if mouth_width / mouth_height > 3 else 0
    features['Scowl'] = 1 if landmarks[21].y - landmarks[22].y > 5 else 0
    left_eyebrow_height = landmarks[21].y - landmarks[19].y
    right_eyebrow_height = landmarks[22].y - landmarks[24].y
    features['otherEyebrowMovement'] = 1 if abs(left_eyebrow_height - right_eyebrow_height) > 2 else 0
    features['Frown'] = 1 if landmarks[21].y > landmarks[20].y else 0
    features['Raise'] = 1 if landmarks[21].y < landmarks[20].y else 0
    eye_opening_left = landmarks[41].y - landmarks[37].y
    eye_opening_right = landmarks[46].y - landmarks[43].y
    features['OtherEyeMovements'] = 1 if abs(eye_opening_left - eye_opening_right) > 1 else 0
    features['gazeInterlocutor'] = 1 if abs(landmarks[36].x - landmarks[39].x) > 10 else 0
    features['gazeDown'] = 1 if landmarks[37].y > landmarks[40].y else 0
    features['gazeUp'] = 1 if landmarks[37].y < landmarks[40].y else 0
    features['otherGaze'] = 1 if not (features['gazeInterlocutor'] or features['gazeDown'] or features['gazeUp']) else 0
    features['openMouth'] = 1 if mouth_height > 10 else 0
    features['closeMouth'] = 1 if mouth_height < 5 else 0
    features['lipsDown'] = 1 if landmarks[57].y > landmarks[51].y else 0
    features['lipsUp'] = 1 if landmarks[51].y < landmarks[62].y else 0
    features['lipsRetracted'] = 1 if mouth_width > 40 else 0
    features['lipsProtruded'] = 1 if landmarks[66].y > landmarks[62].y else 0
    return features

# -------------------------
# Streamlit UI
# -------------------------
st.title("Video Facial Expression Prediction")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    temp_video = NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_file.read())
    video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stframe = st.empty()
    all_predictions = []
    
    frame_interval = 30  # process every 30th frame
    frame_number = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            all_features = []

            if faces:
                for face in faces:
                    landmarks = predictor(gray, face)
                    features = extract_features(landmarks.parts())
                    all_features.append(features)
                    for n in range(68):
                        x, y = landmarks.part(n).x, landmarks.part(n).y
                        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
            else:
                all_features.append({k: 0 for k in extract_features([dlib.point(0, 0)]*68).keys()})

            df = pd.DataFrame(all_features)
            pred_class = np.argmax(model.predict(df.to_numpy()))
            all_predictions.append(pred_class)

            label = "Truthful" if pred_class == 1 else "Deceptive"
            cv2.putText(frame, f"Frame: {frame_number} {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        frame_number += 1

    cap.release()

    # Show average prediction
    avg_prediction = np.mean(all_predictions)
    final_label = "Truthful" if avg_prediction >= 0.5 else "Deceptive"
    st.subheader(f"Final Average Prediction: {final_label} ({avg_prediction:.2f})")




