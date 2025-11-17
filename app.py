import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

st.set_page_config(page_title="ASL Real-Time Recognition", layout="centered")

model = tf.keras.models.load_model("asl_model.keras")

labels = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','nothing','space'
]

IMG_SIZE = (240, 240)

st.title("🖐 ASL Real-Time Recognition")
st.write("EfficientNetB3 Model (99% Accuracy)")

start_cam = st.checkbox("Start Webcam")
frame_view = st.empty()

if start_cam:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, IMG_SIZE)
        arr = img_to_array(resized)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        pred = model.predict(arr, verbose=0)
        pred_idx = np.argmax(pred)
        label = labels[pred_idx]
        conf = pred[0][pred_idx]

        text = f"{label} ({conf*100:.1f}%)"
        cv2.putText(rgb, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        frame_view.image(rgb)

    cap.release()
