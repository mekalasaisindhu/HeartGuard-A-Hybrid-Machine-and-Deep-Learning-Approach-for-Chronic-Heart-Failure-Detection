from ml_models import train_ml_models
import os

FEATURE_CSV = "data/features/processed/features.csv"
MODEL_DIR = "models/ml"

train_ml_models(FEATURE_CSV, MODEL_DIR)
