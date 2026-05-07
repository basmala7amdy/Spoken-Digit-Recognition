import os
import librosa
import numpy as np

from preprocessing import remove_noise, apply_gaussian, smooth_signal
from feature_extraction import extract_features

DATA_PATH = "recordings"


def extract_label(filename):
    try:
        return int(filename.split("_")[0])
    except:
        return None


def load_audio(file_path, sr=22050):
    signal, sample_rate = librosa.load(file_path, sr=sr)
    return signal, sample_rate


def build_dataset():
    X = []
    y = []

    files = os.listdir(DATA_PATH)

    if len(files) == 0:
        print("No audio files found in recordings/")
        return np.array(X), np.array(y)

    for file in files:
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(DATA_PATH, file)

        if file == "test.wav":
            continue

        try:
            signal, sr = load_audio(file_path)

            signal = remove_noise(signal)
            signal = apply_gaussian(signal)
            signal = smooth_signal(signal)

            features = extract_features(signal, sr)
            label = extract_label(file)

            if label is None:
                continue

            X.append(features)
            y.append(label)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    X = np.array(X)
    y = np.array(y)

    print("Dataset Loaded Successfully")
    print(f"Total Samples: {len(X)}")

    return X, y


if __name__ == "__main__":
    X, y = build_dataset()