
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model("model_backup.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


st.set_page_config(page_title="Real-Time Emotion Detector", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .title {
            font-size: 2.5em;
            color: #4A4A4A;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🎭 Real-Time Emotion Detection</p>', unsafe_allow_html=True)
st.subheader("Detect emotions from your facial expressions using a CNN model")


st.sidebar.title("ℹ️ About")
st.sidebar.info("This app uses OpenCV + Keras to detect human emotions in real-time from webcam video.")
st.sidebar.markdown("---")
st.sidebar.write("📁 Files Used:")
st.sidebar.write("- `model_backup.h5`")
st.sidebar.write("- `haarcascade_frontalface_default.xml`")

run = st.checkbox('🎥 Start Webcam')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)


while run:
    ret, frame = camera.read()
    if not ret:
        st.error("❌ Unable to access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    status_message = "😐 No face detected"

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(face) != 0:
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face)[0]
            emotion_probability = np.max(prediction)
            emotion = emotion_labels[np.argmax(prediction)]
            status_message = f"😄 Emotion Detected: {emotion} ({emotion_probability*100:.2f}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    st.markdown(f"**Status:** {status_message}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (720, 540))
    FRAME_WINDOW.image(frame_resized)

else:
    camera.release()
    st.info("🛑 Webcam turned off.")
