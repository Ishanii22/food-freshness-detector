import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = ["freshapples", "freshbanana", "freshoranges",
               "rottenapples", "rottenbanana", "rottenoranges"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/freshness_model.keras")

model = load_model()

st.title("🍎 Food Freshness Detector")
st.write("Upload a photo of an apple, banana or orange to check if it is fresh or rotten.")

file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)

    arr = np.array(img.resize((224, 224)), dtype="float32")[None, ...]
    probs = model.predict(arr)[0]
    idx = int(np.argmax(probs))
    confidence = probs[idx] * 100

    label = CLASS_NAMES[idx]
    status = "FRESH" if label.startswith("fresh") else "ROTTEN"
    fruit = label.replace("fresh", "").replace("rotten", "")

    if confidence < 70:
        st.warning(f"Not sure ({confidence:.1f}%). Please retake the photo with better lighting.")
    else:
        st.subheader(f"{fruit.capitalize()}: {status}")
        st.write(f"Confidence: {confidence:.1f}%")