import joblib
import os
import tempfile
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from keras.layers import TFSMLayer
from feature_engineering import extract_mfcc, extract_acoustic_features, generate_spectrogram

# Module-level model cache (populated by `init_models`)
_SCALER = None
_SVM = None
_CNN = None
_IS_TFSM = False  # Flag to indicate if CNN is a TFSMLayer


def _load_cnn_model(cnn_path):
    """Load CNN model, using TFSMLayer if folder, else load_model for .keras/.h5"""
    if os.path.isdir(cnn_path):
        global _IS_TFSM
        _IS_TFSM = True
        return TFSMLayer(cnn_path, call_endpoint='serving_default')
    else:
        return load_model(cnn_path)


def init_models(ml_path, cnn_path):
    """Load ML and CNN models into module globals for fast reuse."""
    global _SCALER, _SVM, _CNN
    scaler_path = os.path.join(ml_path, "scaler.pkl")
    svm_path = os.path.join(ml_path, "svm.pkl")

    _SCALER = joblib.load(scaler_path)
    _SVM = joblib.load(svm_path)
    _CNN = _load_cnn_model(cnn_path)


def _load_models_fallback(ml_path, cnn_path):
    """Return scaler, svm, cnn loaded ad-hoc (used if init_models wasn't called)."""
    scaler = joblib.load(os.path.join(ml_path, "scaler.pkl"))
    svm = joblib.load(os.path.join(ml_path, "svm.pkl"))
    cnn = _load_cnn_model(cnn_path)
    return scaler, svm, cnn


def predict(audio_path, ml_path, cnn_path, tmp_img=None):
    """Predict probability and extract features from audio."""

    # Extract features
    mfcc = extract_mfcc(audio_path)                     # array
    acoustic = extract_acoustic_features(audio_path)    # dict

    # Create combined feature vector (for ML model)
    feat = np.hstack([list(acoustic.values()), mfcc])

    # Use preloaded models when available, otherwise load on demand
    if _SCALER is not None and _SVM is not None:
        scaler = _SCALER
        svm = _SVM
    else:
        scaler, svm, _ = _load_models_fallback(ml_path, cnn_path)

    feat_scaled = scaler.transform([feat])
    ml_prob = float(svm.predict_proba(feat_scaled)[0][1])

    # CNN path
    remove_tmp = False
    if tmp_img is None:
        fd, tmp_img = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        remove_tmp = True

    try:
        generate_spectrogram(audio_path, tmp_img)
        img = cv2.imread(tmp_img)
        img = cv2.resize(img, (128, 128)) / 255.0
        img = np.expand_dims(img, axis=0)

        if _CNN is not None:
            cnn = _CNN
        else:
            cnn = _load_cnn_model(cnn_path)

        if _IS_TFSM:
            cnn_prob = float(cnn(img).numpy()[0][0])
        else:
            cnn_prob = float(cnn.predict(img)[0][0])

    finally:
        if remove_tmp and os.path.exists(tmp_img):
            os.remove(tmp_img)

    # Hybrid fusion - 50% CNN + 50% ML
    # Default weights - CNN slightly favored as it's more discriminative
    final_prob = (0.5 * cnn_prob) + (0.5 * ml_prob)

    # Build feature dictionary (for frontend display)
    feature_dict = {}
    for k, v in acoustic.items():
        feature_dict[k] = float(v)
    for i, v in enumerate(mfcc):
        feature_dict[f"mfcc_{i+1}"] = float(v)

    return final_prob, feature_dict
