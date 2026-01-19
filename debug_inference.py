"""
Debug script to test model predictions and see what's happening
"""
import os
import sys
import joblib
import numpy as np
import librosa
import cv2
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fix encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from feature_engineering import extract_mfcc, extract_acoustic_features, generate_spectrogram
from tensorflow.keras.models import load_model

def test_inference(audio_file):
    """Test inference on a single audio file"""
    print(f"\n{'='*60}")
    print(f"Testing audio file: {audio_file}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(audio_file):
        print(f"ERROR: File not found: {audio_file}")
        return
    
    # Paths
    ML_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "ml")
    CNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn", "cnn_model.keras")
    
    print(f"ML Model Dir: {ML_MODEL_DIR}")
    print(f"CNN Model Path: {CNN_MODEL_PATH}")
    print(f"ML models exist: {os.path.exists(os.path.join(ML_MODEL_DIR, 'svm.pkl'))}")
    print(f"CNN model exists: {os.path.exists(CNN_MODEL_PATH)}\n")
    
    try:
        # Load models
        print("Loading models...")
        scaler = joblib.load(os.path.join(ML_MODEL_DIR, "scaler.pkl"))
        svm = joblib.load(os.path.join(ML_MODEL_DIR, "svm.pkl"))
        cnn = load_model(CNN_MODEL_PATH)
        print("Models loaded successfully\n")
        
        # Extract ML features
        print("Extracting ML features...")
        mfcc = extract_mfcc(audio_file)
        acoustic = extract_acoustic_features(audio_file)
        
        print(f"  MFCC shape: {mfcc.shape}")
        print(f"  MFCC values (first 5): {mfcc[:5]}")
        print(f"  Acoustic features: {acoustic}\n")
        
        # Combine features
        feat = np.hstack([list(acoustic.values()), mfcc])
        print(f"Combined feature vector shape: {feat.shape}")
        print(f"Feature vector (first 5): {feat[:5]}\n")
        
        # ML Prediction
        print("Running ML (SVM) prediction...")
        feat_scaled = scaler.transform([feat])
        print(f"  Scaled feature shape: {feat_scaled.shape}")
        
        ml_prob = float(svm.predict_proba(feat_scaled)[0][1])
        print(f"  ML Probability (CHF class): {ml_prob:.6f}")
        print(f"  ML Decision: {'CHF' if ml_prob >= 0.5 else 'NORMAL'}\n")
        
        # CNN Prediction
        print("Running CNN prediction...")
        fd, tmp_img = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        
        generate_spectrogram(audio_file, tmp_img)
        img = cv2.imread(tmp_img)
        print(f"  Image shape: {img.shape if img is not None else 'None'}")
        
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0
            img = np.expand_dims(img, axis=0)
            print(f"  Resized image shape: {img.shape}")
            
            cnn_output = cnn.predict(img, verbose=0)
            cnn_prob = float(cnn_output[0][0])
            print(f"  CNN Output shape: {cnn_output.shape}")
            print(f"  CNN Raw output: {cnn_output[0]}")
            print(f"  CNN Probability: {cnn_prob:.6f}")
            print(f"  CNN Decision: {'CHF' if cnn_prob >= 0.5 else 'NORMAL'}\n")
        
        os.remove(tmp_img)
        
        # Hybrid fusion - adjusted weights
        print("Hybrid Fusion (40% CNN + 60% ML)...")
        final_prob = (0.4 * cnn_prob) + (0.6 * ml_prob)
        print(f"  Final probability: {final_prob:.6f}")
        print(f"  Final Decision: {'CHF' if final_prob >= 0.5 else 'NORMAL'}\n")
        
        print(f"{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"ML Prob:    {ml_prob:.4f} ({'CHF' if ml_prob >= 0.5 else 'NORMAL'})")
        print(f"CNN Prob:   {cnn_prob:.4f} ({'CHF' if cnn_prob >= 0.5 else 'NORMAL'})")
        print(f"Final Prob: {final_prob:.4f} ({'CHF' if final_prob >= 0.5 else 'NORMAL'})")
        print(f"{'='*60}\n")
        
        return {
            "ml_prob": ml_prob,
            "cnn_prob": cnn_prob,
            "final_prob": final_prob
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with multiple audio files
    test_files = [
        os.path.join(PROJECT_ROOT, "data", "raw", "training-a", "a0001.wav"),
        os.path.join(PROJECT_ROOT, "data", "raw", "training-a", "a0002.wav"),
        os.path.join(PROJECT_ROOT, "data", "raw", "training-b", "b0001.wav"),
    ]
    
    results = []
    for test_audio in test_files:
        if os.path.exists(test_audio):
            result = test_inference(test_audio)
            if result:
                results.append((os.path.basename(test_audio), result))
    
    if results:
        print("\n" + "="*60)
        print("SUMMARY OF ALL TESTS")
        print("="*60)
        for filename, result in results:
            print(f"{filename}:")
            print(f"  ML:    {result['ml_prob']:.4f}")
            print(f"  CNN:   {result['cnn_prob']:.4f}")
            print(f"  Final: {result['final_prob']:.4f}")
        print("="*60)
