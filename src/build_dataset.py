import os
import glob
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

def bandpass_filter(signal, sr, lowcut=20, highcut=500, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def find_audio_file(base_dir, database, record_name):
    folder = os.path.join(base_dir, database)
    if not os.path.exists(folder):
        print("Missing folder:", folder)
        return None
    
    # Try common extensions
    for ext in ('.wav', '.aiff', '.flac', '.mp3', '.dat'):
        path = os.path.join(folder, record_name + ext)
        if os.path.exists(path):
            return path

    # Try searching anywhere
    matches = glob.glob(os.path.join(folder, f"{record_name}.*"))
    return matches[0] if matches else None

def generate_mel_spectrogram_image(y, sr, out_path, n_mels=128, fmax=800):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_db, sr=sr, cmap='viridis')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def build_dataset(
    annotations_csv='data/raw/annotations/Online_Appendix_training_set.csv',
    raw_base='data/raw',
    processed_dir='data/processed',
    spectrogram_dir='data/spectrogram',
    features_csv='data/features/features.csv',
    target_sr=2000,
):
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(spectrogram_dir, exist_ok=True)
    os.makedirs(os.path.dirname(features_csv), exist_ok=True)

    print("\nLoading annotation file:", annotations_csv)
    if not os.path.exists(annotations_csv):
        print("ERROR: Annotations file not found.")
        return

    df = pd.read_csv(annotations_csv)
    rows = []

    print("Total annotation rows:", len(df))

    for _, r in df.iterrows():
        rec = str(r['Challenge record name']).strip()
        db = str(r['Database']).strip()
        class_val = int(r['Class (-1=normal 1=abnormal)'])
        label = 1 if class_val == 1 else 0

        audio_path = find_audio_file(raw_base, db, rec)
        if not audio_path:
            print(f"[Missing Audio] {rec} in DB {db}")
            continue
        
        print("Processing:", audio_path)

        try:
            y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        except Exception as e:
            print(f"[Load Error] {audio_path}: {e}")
            continue

        if len(y) < 100:
            print(f"[Too Short] Skipping {audio_path}")
            continue

        # Filter & normalize
        try:
            y = bandpass_filter(y, target_sr)
        except:
            pass

        y = librosa.util.normalize(y)

        # Save processed wav
        out_wav = os.path.join(processed_dir, f"{rec}.wav")
        sf.write(out_wav, y, target_sr)

        # Extract features
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc.T, axis=0)
        except Exception as e:
            print(f"[MFCC Error] {audio_path}: {e}")
            continue

        try:
            acoustic = {
                'zcr': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=target_sr))),
                'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=target_sr))),
                'chroma_stft': float(np.mean(librosa.feature.chroma_stft(y=y, sr=target_sr)))
            }
        except Exception as e:
            print(f"[Acoustic Error] {audio_path}: {e}")
            continue

        # Spectrogram image
        try:
            label_name = 'abnormal' if label == 1 else 'normal'
            spec_label_dir = os.path.join(spectrogram_dir, label_name)
            os.makedirs(spec_label_dir, exist_ok=True)
            spec_out = os.path.join(spec_label_dir, f"{rec}.png")
            generate_mel_spectrogram_image(y, target_sr, spec_out)
        except Exception as e:
            print(f"[Spectrogram Error] {audio_path}: {e}")

        row = {'file': f"{rec}.wav", 'label': label, **acoustic}
        for i, v in enumerate(mfcc_mean):
            row[f'mfcc_{i}'] = float(v)

        rows.append(row)

    if len(rows) == 0:
        print("\nERROR: No rows generated. Check audio directory paths.\n")
        return

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(features_csv, index=False)

    print("\nSUCCESS: Feature CSV written to:", features_csv)
    print("Total processed files:", len(rows))

if __name__ == '__main__':
    build_dataset()
