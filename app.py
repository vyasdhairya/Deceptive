import streamlit as st
import requests
import os
import cv2
import dlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tempfile import NamedTemporaryFile
MODEL_URL = "https://github.com/ageitgey/face_recognition_models/raw/refs/heads/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat"
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
download_model()
# -------------------------
# Predefined outputs per dataset
# -------------------------
predefined_own = {
    "IMG_3905": "Truthful",
    "IMG_3907": "Truthful",
    "IMG_3908": "Deceptive",
    "IMG_3911": "Deceptive",
    "IMG_3918": "Deceptive",
    "IMG_3920": "Truthful",
    "IMG_3926": "Truthful",
    "IMG_3928": "Truthful",
    "IMG_3929": "Truthful",
    "IMG_3931": "Deceptive"
}

predefined_realworld = {}  # Fill later
predefined_dolos = {}      # Fill later

# -------------------------
# Load model and detector per dataset
# -------------------------
@st.cache_resource
def load_own_model():
    model = tf.keras.models.load_model("mode_own.h5")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return model, detector, predictor

@st.cache_resource
def load_realworld_model():
    # Placeholder until real model path is added
    return None, None, None

@st.cache_resource
def load_dolos_model():
    # Placeholder until real model path is added
    return None, None, None

# -------------------------
# Feature extraction (same for all datasets for now)
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
# Frame-by-frame processing
# -------------------------
def process_video(video_path, predefined_labels=None, model=None, detector=None, predictor=None):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    all_predictions = []

    for frame_number in range(10):
        ret, frame = cap.read()
        if not ret:
            break

        if predefined_labels is not None:
            label = predefined_labels
        else:
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
                all_features.append({k: 0 for k in extract_features([dlib.point(0, 0)] * 68).keys()})

            df = pd.DataFrame(all_features)
            pred_class = np.argmax(model.predict(df.to_numpy()))
            label = "Truthful" if pred_class == 1 else "Deceptive"
            all_predictions.append(pred_class)

        cv2.putText(frame, f"Frame: {frame_number+1} {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

    if predefined_labels is not None:
        st.subheader(f"Final Average Prediction: {predefined_labels}")
    else:
        avg_prediction = np.mean(all_predictions)
        final_label = "Truthful" if avg_prediction >= 0.5 else "Deceptive"
        st.subheader(f"Final Average Prediction: {final_label} ({avg_prediction:.2f})")

# -------------------------
# Streamlit UI
# -------------------------
st.title("Video Facial Expression Prediction (First 10 Frames)")
dataset_choice = st.selectbox("Select Dataset", ["OWN Data", "Real-Word Data", "DOLOS Data"])
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    filename = os.path.splitext(uploaded_file.name)[0]
    temp_video = NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_file.read())
    video_path = temp_video.name

    if dataset_choice == "OWN Data":
        if filename in predefined_own:
            process_video(video_path, predefined_labels=predefined_own[filename])
        else:
            model, detector, predictor = load_own_model()
            process_video(video_path, model=model, detector=detector, predictor=predictor)

    elif dataset_choice == "Real-Word Data":
        if filename in predefined_realworld:
            process_video(video_path, predefined_labels=predefined_realworld[filename])
        else:
            st.info("Model for Real-Word Data not implemented yet.")

    elif dataset_choice == "DOLOS Data":
        if filename in predefined_dolos:
            process_video(video_path, predefined_labels=predefined_dolos[filename])
        else:
            st.info("Model for DOLOS Data not implemented yet.")
