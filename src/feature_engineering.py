import os
import librosa
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

def extract_mfcc(audio_path, n_mfcc=20):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


def extract_acoustic_features(audio_path):
    y, sr = librosa.load(audio_path)

    features = {
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "chroma_stft": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
    }
    return features


def generate_spectrogram(audio_path, output_image):
    y, sr = librosa.load(audio_path)

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_dB, sr=sr, cmap="viridis")
    plt.axis("off")
    plt.savefig(output_image, bbox_inches="tight", pad_inches=0)
    plt.close()


def build_feature_dataset(processed_folder, feature_csv):
    data = []

    for file in os.listdir(processed_folder):
        if file.endswith(".wav"):
            path = os.path.join(processed_folder, file)
            mfcc = extract_mfcc(path)
            acoustic = extract_acoustic_features(path)

            row = {
                "file": file,
                **acoustic
            }
            for i, value in enumerate(mfcc):
                row[f"mfcc_{i}"] = value

            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(feature_csv, index=False)
    print("Feature dataset created:", feature_csv)
