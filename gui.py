import streamlit as st
import numpy as np
import os

from tensorflow.keras.models import load_model

from preprocessing import preprocess_audio
from feature_extraction import extract_feature

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Spoken Digit Recognition",
    page_icon="🎤",
    layout="centered"
)

# ------------------------------------------------
# GALAXY STYLE
# ------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #050816,
        #0b1026,
        #140b2d,
        #1f1147
    );
    background-attachment: fixed;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #d7b3ff;
    text-shadow: 0px 0px 20px #8f5cff;
}

.sub-text {
    text-align: center;
    color: #cfcfff;
    font-size: 18px;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
}

.stButton>button {
    width: 100%;
    height: 55px;
    border-radius: 18px;
    border: none;
    font-size: 20px;
    font-weight: bold;
    color: white;
    background: linear-gradient(
        90deg,
        #7f5cff,
        #c44dff,
        #00c2ff
    );
    box-shadow: 0px 0px 20px rgba(140, 82, 255, 0.7);
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 30px rgba(0, 194, 255, 0.8);
}

.result-box {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 25px;
    text-align: center;
    margin-top: 30px;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
}

.result-digit {
    font-size: 90px;
    font-weight: bold;
    color: #b266ff;
    text-shadow: 0px 0px 25px #b266ff;
}

.result-text {
    font-size: 24px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.markdown(
    "<div class='main-title'>🎤 Spoken Digit Recognition</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-text'>Upload WAV audio and detect spoken number (0 → 9)</div>",
    unsafe_allow_html=True
)

st.write("")

# ------------------------------------------------
# CREATE FOLDERS
# ------------------------------------------------

os.makedirs("recordings", exist_ok=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

MODEL_PATH = "models/digit_model.keras"

model = load_model(MODEL_PATH)

# ------------------------------------------------
# FILE UPLOADER
# ------------------------------------------------

audio_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

# ------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------

def predict(file_path):

    preprocess_audio(
        file_path,
        "recordings/clean.wav"
    )

    feature = extract_feature(
        "recordings/clean.wav"
    )

    feature = feature / np.max(feature)

    feature = np.array(feature).reshape(1, -1)

    pred = model.predict(feature, verbose=0)

    digit = np.argmax(pred)

    confidence = np.max(pred)

    return digit, confidence

# ------------------------------------------------
# BUTTON
# ------------------------------------------------

if st.button("✨ Predict Digit"):

    if audio_file is not None:

        path = "recordings/test.wav"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        # ------------------------------------------------
        # PLAY AUDIO
        # ------------------------------------------------

        st.audio(path)

        # ------------------------------------------------
        # PREDICT
        # ------------------------------------------------

        result, confidence = predict(path)

        # ------------------------------------------------
        # RESULT UI
        # ------------------------------------------------
        st.markdown(f"## 🎯 {result}")