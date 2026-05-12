import numpy as np
import librosa

from tensorflow.keras.models import load_model

from utils.preprocess import preprocess
from utils.features import extract_mfcc


# settings
SR = 16000
MODEL_PATH = "models/spoken_digit_model.keras"
model = load_model(MODEL_PATH)

# ------------------------

# Predict
def predict(path):

    audio, sr = librosa.load(path, sr=SR)

    audio = preprocess(audio)

    mfcc = extract_mfcc(audio)

    # reshape for CNN
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    pred = model.predict(mfcc, verbose=0)

    digit = np.argmax(pred)
    confidence = float(np.max(pred))

    return digit, confidence