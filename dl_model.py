import numpy as np

from tensorflow.keras.models import load_model

from feature_extraction import extract_feature

MODEL_PATH = "models/digit_model.keras"

# ------------------------
# Load Trained Model
# ------------------------

model = load_model(MODEL_PATH)


def predict_digit(file_path):

    # ------------------------
    # Extract Features
    # ------------------------

    feature = extract_feature(file_path)

    # ------------------------
    # Normalize
    # ------------------------

    feature = feature / np.max(feature)

    # ------------------------
    # Reshape
    # ------------------------

    feature = np.array(feature).reshape(1, -1)

    # ------------------------
    # Prediction
    # ------------------------

    prediction = model.predict(feature, verbose=0)

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction)

    return predicted_digit, confidence