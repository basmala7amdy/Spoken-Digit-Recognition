from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.playback import play

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from preprocessing import (
    load_audio,
    remove_noise,
    apply_gaussian,
    smooth_signal
)


def plot_signal(signal, sr, title="Audio Signal"):

    plt.figure(figsize=(10, 4))

    librosa.display.waveshow(
        signal,
        sr=sr
    )

    plt.title(title)

    plt.xlabel("Time")

    plt.ylabel("Amplitude")

    plt.tight_layout()

    plt.show()


def remove_silence(audio_path):

    audio_file = AudioSegment.from_file(audio_path)

    play(audio_file)

    min_silence = 500
    threshold = -30

    chunks = split_on_silence(
        audio_file,
        min_silence_len=min_silence,
        silence_thresh=threshold
    )

    output_audio = sum(chunks)

    output_path = "recordings/output_audio.wav"

    output_audio.export(output_path, format="wav")

    play(output_audio)

    signal, sr = librosa.load(output_path, sr=None)

    return signal, sr


def show_before_after(original_signal, cleaned_signal, sr):

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)

    librosa.display.waveshow(original_signal, sr=sr)

    plt.title("Before Silence Removal")

    plt.subplot(2, 1, 2)

    librosa.display.waveshow(cleaned_signal, sr=sr)

    plt.title("After Silence Removal")

    plt.tight_layout()

    plt.show()