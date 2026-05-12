import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d

SR = 16000
MAX_LEN = 16000

def preprocess(audio, sr=SR, max_len=MAX_LEN):

    # gaussian smoothing
    audio = gaussian_filter1d(audio, sigma=2)

    # normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # trim silence
    audio, _ = librosa.effects.trim(audio)

    # pad or cut
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]

    return audio