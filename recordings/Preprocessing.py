import librosa
import numpy as np
import noisereduce as nr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve

def load_audio(file_path):
    
    signal, sr = librosa.load(file_path, sr=None)

    return signal, sr

def remove_noise(signal, sr):

    reduced_noise = nr.reduce_noise(
        y=signal,
        sr=sr
    )

    return reduced_noise

def apply_gaussian(signal, sigma=1):

    filtered_signal = gaussian_filter1d(
        signal,
        sigma=sigma
    )

    return filtered_signal

def smooth_signal(signal, kernel_size=5):

    kernel = np.ones(kernel_size) / kernel_size

    smoothed_signal = convolve(
        signal,
        kernel,
        mode='same'
    )

    return smoothed_signal

def preprocess_audio(file_path):

    # load audio
    signal, sr = load_audio(file_path)

    # remove noise
    signal = remove_noise(signal, sr)

    # gaussian filter
    signal = apply_gaussian(signal)

    # smoothing
    signal = smooth_signal(signal)

    return signal, sr