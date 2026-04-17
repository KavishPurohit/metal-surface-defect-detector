from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

st.set_page_config(page_title="Metal Defect Detection", layout="centered")
st.title("Metal Defect Detection")
st.write("Upload a metal surface image to detect defects.")

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return YOLO(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp"]
)

conf = st.slider("Confidence threshold", 0.1, 1.0, 0.4, 0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model.predict(source=np.array(image), conf=conf, verbose=False)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        names = model.names
        st.subheader("Detected defects")
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label = f"{names[cls_id]} {score:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(y1 - 18, 0)), label, fill="red")
            st.write(f"- {names[cls_id]} ({score:.2f})")

        st.image(annotated, caption="Detection Result", use_container_width=True)
    else:
        st.info("No defect detected.")
