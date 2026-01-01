import joblib
import numpy as np
from tensorflow.keras.models import load_model

def hybrid_prediction(ml_probs, cnn_prob):
    return (0.7 * cnn_prob) + (0.3 * ml_probs)


