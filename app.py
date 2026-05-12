import streamlit as st
import os

from utils.predictor import predict

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Spoken Digit Recognition",
    page_icon="🎤",
    layout="centered"
)

# ------------------------------------------------
# GALAXY THEME
# ------------------------------------------------

st.markdown("""
<style>

/* Background */

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

/* Main Title */

.main-title {

    text-align: center;

    font-size: 52px;

    font-weight: bold;

    color: #d7b3ff;

    text-shadow: 0px 0px 25px #8f5cff;

    margin-bottom: 10px;
}

/* Subtitle */

.sub-text {

    text-align: center;

    color: #d0d7ff;

    font-size: 18px;

    margin-bottom: 35px;
}

/* Upload Box */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.06);

    border: 1px solid rgba(255,255,255,0.12);

    border-radius: 18px;

    padding: 15px;

    backdrop-filter: blur(10px);
}

/* Predict Button */

.stButton>button {

    width: 100%;

    height: 58px;

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

    box-shadow: 0px 0px 25px rgba(140, 82, 255, 0.6);

    transition: 0.3s ease;
}

/* Hover Effect */

.stButton>button:hover {

    transform: scale(1.02);

    box-shadow:
        0px 0px 35px rgba(0,194,255,0.9),
        0px 0px 20px rgba(196,77,255,0.7);
}

/* Result Box */

.result-box {

    background: rgba(255,255,255,0.08);

    padding: 30px;

    border-radius: 22px;

    text-align: center;

    margin-top: 30px;

    border: 1px solid rgba(255,255,255,0.15);

    backdrop-filter: blur(12px);

    box-shadow: 0px 0px 30px rgba(127,92,255,0.3);
}

/* Predicted Digit */

.result-digit {

    font-size: 90px;

    font-weight: bold;

    color: #c084ff;

    text-shadow:
        0px 0px 20px #c084ff,
        0px 0px 40px #7f5cff;
}

/* Confidence Text */

.result-text {

    font-size: 22px;

    color: #f2f2ff;

    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.markdown(
    """
    <div class='main-title'>
        🎤 Spoken Digit Recognition
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='sub-text'>
        Upload WAV audio → Predict spoken digit (0-9)
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# FILE UPLOADER
# ------------------------------------------------

audio_file = st.file_uploader(
    "📁 Upload WAV File",
    type=["wav"]
)

# create recordings folder
os.makedirs("recordings", exist_ok=True)

# ------------------------------------------------
# PREDICT BUTTON
# ------------------------------------------------

if st.button("✨ Predict Digit"):

    if audio_file is not None:

        # save uploaded file
        path = "recordings/test.wav"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        # audio player
        st.audio(path)

        # prediction
        digit, confidence = predict(path)

        # ------------------------------------------------
        # RESULT UI
        # ------------------------------------------------
# ------------------------------------------------
# RESULT UI
# ------------------------------------------------

        st.markdown("""
        <div class='result-box'>
        """, unsafe_allow_html=True)

        st.markdown(
            f"<div class='result-digit'>{digit}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class='result-text'>
            Confidence: {confidence:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)
    else:

        st.warning("Please upload an audio file first.")