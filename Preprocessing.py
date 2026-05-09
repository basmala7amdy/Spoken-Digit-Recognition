import numpy as np
import librosa
import soundfile as sf

from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d

from pydub import AudioSegment
from pydub.silence import split_on_silence


def avg_filter(signal, ws=5):

    w = np.ones(ws) / ws

    return convolve1d(signal, w, mode='nearest')


def gaussian_filter(signal, sigma=2):

    return gaussian_filter1d(signal, sigma)


def preprocess_audio(input_path, output_path):

    signal, sr = librosa.load(input_path, sr=16000)

    signal = avg_filter(signal)

    signal = gaussian_filter(signal)

    temp_path = "recordings/temp.wav"

    sf.write(temp_path, signal, sr)

    audio = AudioSegment.from_wav(temp_path)

    chunks = split_on_silence(
        audio,
        min_silence_len=500,
        silence_thresh=-30,
        keep_silence=100
    )

    if len(chunks) > 0:
        cleaned = sum(chunks)
    else:
        cleaned = audio

    cleaned.export(output_path, format="wav")