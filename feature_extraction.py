import librosa
import numpy as np


def extract_feature(file_name):

    audio, sample_rate = librosa.load(
        file_name,
        res_type='kaiser_fast'
    )

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfccs_scaled = np.mean(mfccs.T, axis=0)

    return mfccs_scaled