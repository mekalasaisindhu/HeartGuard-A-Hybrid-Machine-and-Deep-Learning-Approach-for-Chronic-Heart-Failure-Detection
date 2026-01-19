# scripts/sweep_fusion.py
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score

FEATURE_CSV = "data/features/features.csv"
SCALER_PATH = "models/ml/scaler.pkl"
SVM_PATH = "models/ml/svm.pkl"
SPECTRO_DIR = "data/spectrogram"
CNN_PATH = "models/cnn/cnn_model.keras"
IMG_SIZE = 128

def load_csv_clean(path):
    df = pd.read_csv(path)
    df = df[df["label"] != "label"]
    df["label"] = df["label"].astype(int)
    return df

def load_spectrogram(rec, label):
    folder = "abnormal" if label == 1 else "normal"
    img_path = os.path.join(SPECTRO_DIR, folder, rec.replace(".wav", ".png"))
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return img

def main():
    df = load_csv_clean(FEATURE_CSV)
    feature_cols = [c for c in df.columns if c not in ("file", "label")]
    X_tab = df[feature_cols].astype(float).values
    y = df["label"].values

    scaler = joblib.load(SCALER_PATH)
    svm = joblib.load(SVM_PATH)
    cnn = tf.keras.models.load_model(CNN_PATH)

    # ML scores
    X_scaled = scaler.transform(X_tab)
    ml_scores = svm.predict_proba(X_scaled)[:, 1]

    # CNN scores (skip missing images)
    cnn_inputs = []
    keep_indices = []
    for i, row in df.iterrows():
        img = load_spectrogram(row["file"], row["label"])
        if img is None:
            continue
        cnn_inputs.append(img)
        keep_indices.append(i)

    if len(keep_indices) != len(y):
        # align arrays to available images
        y = y[keep_indices]
        ml_scores = ml_scores[keep_indices]

    cnn_inputs = np.array(cnn_inputs)
    cnn_scores = cnn.predict(cnn_inputs).flatten()

    # Sweep fusion weight
    best_w, best_acc = None, 0.0
    for w in np.linspace(0, 1, 21):
        hybrid = w * cnn_scores + (1 - w) * ml_scores
        y_pred = (hybrid > 0.5).astype(int)
        acc = accuracy_score(y, y_pred)
        print(f"w={w:.2f} -> acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_w = w

    print(f"\nBest weight: {best_w:.2f} with accuracy {best_acc:.4f}")

if __name__ == "__main__":
    main()