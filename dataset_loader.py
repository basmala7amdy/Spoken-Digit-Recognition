import os
import librosa
import numpy as np

from preprocessing import remove_noise, apply_gaussian, smooth_signal
from feature_extraction import extract_features


DATA_PATH = "data/recordings"


def extract_label(filename):
    """
    Example filename:
    3_jackson_10.wav -> label = 3
    """
    return int(filename.split("_")[0])


def load_audio(file_path, sr=22050):
    """
    Load WAV file using librosa
    """
    signal, sample_rate = librosa.load(file_path, sr=sr)
    return signal, sample_rate


def build_dataset():
    """
    Builds X, y for ML model
    """
    X = []
    y = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".wav"):
            file_path = os.path.join(DATA_PATH, file)

            try:
                # 1. Load audio
                signal, sr = load_audio(file_path)

                # 2. Preprocessing pipeline
                signal = remove_noise(signal)
                signal = apply_gaussian(signal)
                signal = smooth_signal(signal)

                # 3. Feature extraction (MFCC or others)
                features = extract_features(signal, sr)

                # 4. Label extraction
                label = extract_label(file)

                # 5. Append to dataset
                X.append(features)
                y.append(label)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print(f"Dataset created successfully!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y


# Optional test run
if __name__ == "__main__":
    X, y = build_dataset()
