import librosa

SR = 16000
N_MFCC = 20

def extract_mfcc(audio, sr=SR):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    return mfcc