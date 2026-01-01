import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, sr, lowcut=20, highcut=500):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal)


def preprocess_audio(input_path, output_path, target_sr=22050):
    audio, sr = librosa.load(input_path, sr=target_sr)

    audio = bandpass_filter(audio, sr)
    audio = librosa.util.normalize(audio)

    sf.write(output_path, audio, target_sr)
    return output_path


def preprocess_folder(raw_folder, processed_folder):
    os.makedirs(processed_folder, exist_ok=True)

    for file in os.listdir(raw_folder):
        if file.endswith(".wav"):
            preprocess_audio(
                os.path.join(raw_folder, file),
                os.path.join(processed_folder, file)
            )
    print("Preprocessing completed.")
