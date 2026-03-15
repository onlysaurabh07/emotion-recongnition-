import io
from typing import Tuple, Optional

import cv2
import numpy as np
import streamlit as st
from fer.fer import FER
from PIL import Image


@st.cache_resource
def load_emotion_detector() -> FER:
    return FER(mtcnn=True)


def read_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert a PIL image (RGB) to an OpenCV image (BGR)."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def analyze_emotions(image: Image.Image, detector: FER) -> Tuple[Optional[str], Optional[float], np.ndarray]:
    """
    Run emotion recognition on the given image.

    Returns:
        dominant_emotion: Name of the most likely emotion.
        score: Confidence score between 0 and 1.
        annotated: Numpy array of the annotated image (RGB).
    """
    cv2_img = pil_to_cv2(image)
    # FER works with RGB images
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    # detect_emotions returns a list of dicts with 'box' and 'emotions'
    results = detector.detect_emotions(rgb_img)

    if not results:
        return None, None, rgb_img

    # For simplicity, use the first detected face
    best = results[0]
    emotions = best.get("emotions", {})
    if not emotions:
        return None, None, rgb_img

    dominant_emotion = max(emotions, key=emotions.get)
    score = emotions[dominant_emotion]

    # Draw bounding box and label
    (x, y, w, h) = best["box"]
    annotated = rgb_img.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"{dominant_emotion} ({score:.2f})"
    cv2.putText(
        annotated,
        label,
        (x, max(y - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return dominant_emotion, score, annotated


def analyze_and_display(image: Image.Image) -> None:
    """Shared analysis + display logic for both upload and webcam modes."""
    st.subheader("Original image")
    st.image(image, use_column_width=True)

    with st.spinner("Analyzing emotions..."):
        detector = load_emotion_detector()
        emotion, score, annotated = analyze_emotions(image, detector)

    if emotion is None:
        st.warning("I couldn't confidently detect any face or emotion in this image.")
        return

    st.subheader("Detected emotion")
    st.markdown(f"**{emotion.capitalize()}** with confidence **{score:.2f}**")

    st.subheader("Annotated image")
    st.image(annotated, use_column_width=True)


def main() -> None:
    st.set_page_config(page_title="Emotion Recognition", page_icon="🙂", layout="centered")

    st.title("Emotion Recognition from Face Image")
    st.write(
        "Upload a photo or use your webcam, and this app will try to detect the **dominant emotion**."
    )

    with st.expander("How it works", expanded=False):
        st.markdown(
            "- **Model**: Uses a pre-trained deep learning model from the `fer` library.\n"
            "- **Inputs**: Single image from file upload or webcam snapshot.\n"
            "- **Output**: The emotion with highest confidence and an annotated preview."
        )

    mode = st.radio(
        "Choose input mode:",
        ["Upload image", "Use webcam"],
        horizontal=True,
    )

    if mode == "Upload image":
        uploaded = st.file_uploader(
            "Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False
        )

        if uploaded is None:
            st.info("Upload an image to get started.")
            return

        try:
            image = read_image(uploaded.read())
        except Exception:
            st.error("Could not read this image. Please try a different file.")
            return

        analyze_and_display(image)

    else:
        st.info("Click on 'Start' in the camera widget, then take a snapshot.")
        cam_image = st.camera_input("Take a photo")

        if cam_image is None:
            return

        try:
            image = read_image(cam_image.getvalue())
        except Exception:
            st.error("Could not read the camera image. Please try again.")
            return

        analyze_and_display(image)


if __name__ == "__main__":
    main()

