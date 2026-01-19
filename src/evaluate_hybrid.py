import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image


# =========================
# Paths and Configuration
# =========================
FEATURE_CSV = "data/features/features.csv"

SCALER_PATH = "models/ml/scaler.pkl"
SVM_PATH = "models/ml/svm.pkl"

SPECTRO_DIR = "data/spectrogram"
CNN_PATH = "models/cnn/cnn_model.keras"

IMG_SIZE = 128


# =========================
# Utility Functions
# =========================
def load_csv_clean(path):
    """Load feature CSV and clean duplicate headers."""
    df = pd.read_csv(path)
    df = df[df["label"] != "label"]  # remove duplicate header rows
    df["label"] = df["label"].astype(int)
    return df


def load_spectrogram(rec, label):
    """Load and preprocess spectrogram image."""
    folder = "abnormal" if label == 1 else "normal"
    img_path = os.path.join(
        SPECTRO_DIR,
        folder,
        rec.replace(".wav", ".png")
    )

    if not os.path.exists(img_path):
        return None

    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0

    return img


def load_models():
    """Load scaler, SVM, and CNN models."""
    scaler = joblib.load(SCALER_PATH)
    svm = joblib.load(SVM_PATH)
    cnn = tf.keras.models.load_model(CNN_PATH)
    return scaler, svm, cnn


# =========================
# Evaluation Pipeline
# =========================
def evaluate():
    print("Loading feature CSV...")
    df = load_csv_clean(FEATURE_CSV)

    # Prepare tabular features
    feature_cols = [c for c in df.columns if c not in ("file", "label")]
    X_tab = df[feature_cols].astype(float).values
    y = df["label"].values

    # Load models
    scaler, svm, cnn = load_models()

    # -------------------------
    # ML (SVM) Prediction
    # -------------------------
    X_scaled = scaler.transform(X_tab)
    ml_scores = svm.predict_proba(X_scaled)[:, 1]

    # -------------------------
    # CNN Prediction
    # -------------------------
    cnn_inputs = []
    drop_indices = []

    for i, row in df.iterrows():
        img = load_spectrogram(row["file"], row["label"])
        if img is None:
            drop_indices.append(i)
            continue
        cnn_inputs.append(img)

    # Remove samples without spectrograms
    if drop_indices:
        X_scaled = np.delete(X_scaled, drop_indices, axis=0)
        y = np.delete(y, drop_indices, axis=0)
        ml_scores = np.delete(ml_scores, drop_indices, axis=0)

    cnn_inputs = np.array(cnn_inputs)
    cnn_scores = cnn.predict(cnn_inputs).flatten()

    # -------------------------
    # Hybrid Prediction
    # -------------------------
    hybrid_scores = (ml_scores + cnn_scores) / 2
    y_pred = (hybrid_scores > 0.6).astype(int)

    # -------------------------
    # Evaluation Metrics
    # -------------------------
    print("\nAccuracy:")
    print(accuracy_score(y, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(y, y_pred))


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    evaluate()
