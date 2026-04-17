import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
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

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model.predict(source=tmp.name, conf=conf)

    plotted = results[0].plot()
    st.image(plotted, caption="Detection Result", use_container_width=True)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        st.subheader("Detected defects")
        names = model.names
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            st.write(f"- {names[cls_id]} ({score:.2f})")
    else:
        st.info("No defect detected.")
