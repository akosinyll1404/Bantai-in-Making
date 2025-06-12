import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load your trained PPE detection model
model = YOLO("best (4).pt")  # Replace with your actual path

st.set_page_config(page_title="🦺 PPE Detection System", layout="centered")
st.title("🛡️ PPE Detection using YOLO")
st.markdown("Detect helmets, vests, gloves, and more with real-time or uploaded media.")

# Sidebar options
mode = st.sidebar.radio("Choose Input Mode", ["📷 Live Webcam", "🖼️ Upload Image", "🎥 Upload Video"])

# ---------------- IMAGE MODE ----------------
if mode == "🖼️ Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        results = model(img_bgr)[0].plot()
        final_img = cv2.cvtColor(results, cv2.COLOR_BGR2RGB)
        st.image(final_img, caption="Detection Result", use_container_width=True)

# ---------------- VIDEO MODE ----------------
elif mode == "🎥 Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        st.info("Playing uploaded video with detections...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0].plot()
            frame_rgb = cv2.cvtColor(results, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()

# ---------------- LIVE CAMERA MODE ----------------
elif mode == "📷 Live Webcam":
    start = st.button("Start Webcam")
    if start:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        st.warning("Webcam running... press [Stop] to end")

        stop = st.button("Stop")

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0].plot()
            frame_rgb = cv2.cvtColor(results, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
        st.success("Webcam stopped.")
