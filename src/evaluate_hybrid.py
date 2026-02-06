import os
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
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


def plot_training_history():
    """Load and plot model training history."""
    history_path = "models/cnn/training_history.json"
    
    if not os.path.exists(history_path):
        print(f"⚠ Training history not found at {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot accuracy on left axis
    color_train = '#B3D9FF'  # Light Blue
    color_val = '#FFD9B3'    # Light Orange
    
    ax1.set_xlabel('Epochs', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold', color='black')
    
    line_train = ax1.plot(epochs, history['accuracy'], marker='o', color=color_train, 
                          label='Train Set', linewidth=2.5, markersize=6)
    line_val = ax1.plot(epochs, history['val_accuracy'], marker='s', color=color_val, 
                       label='Val Set', linewidth=2.5, markersize=6)
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim([min(min(history['accuracy']), min(history['val_accuracy'])) - 0.01, 1.02])
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Create second y-axis for loss
    ax2 = ax1.twinx()
    color_loss = '#B3E5B3'  # Light Green
    
    ax2.set_ylabel('Loss', fontsize=13, fontweight='bold', color=color_loss)
    line_loss = ax2.plot(epochs, history['loss'], marker='^', color=color_loss, 
                         label='Loss', linewidth=2.5, markersize=6, linestyle='--', alpha=0.8)
    
    ax2.tick_params(axis='y', labelcolor=color_loss)
    
    # Combine legends
    lines = line_train + line_val + line_loss
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.95)
    
    plt.title('Hybrid Model Training History\nChronic Heart Failure Detection', 
              fontsize=15, fontweight='bold', pad=20)
    
    fig.tight_layout()
    
    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_history_hybrid.png", dpi=300, bbox_inches='tight')
    print("✓ Training history chart saved to: results/training_history_hybrid.png")
    plt.close()


def plot_model_comparison(ml_scores, cnn_scores, hybrid_scores, y):
    """Plot comparative model performance metrics."""
    # Calculate metrics for each model
    ml_pred = (ml_scores > 0.5).astype(int)
    cnn_pred = (cnn_scores > 0.5).astype(int)
    hybrid_pred = (hybrid_scores > 0.6).astype(int)
    
    models = ['SVM\n(ML Model)', 'CNN\n(Deep Learning)', 'Hybrid\n(Proposed)']
    
    # Calculate metrics
    accuracy = [
        accuracy_score(y, ml_pred),
        accuracy_score(y, cnn_pred),
        accuracy_score(y, hybrid_pred)
    ]
    
    precision = [
        precision_score(y, ml_pred, zero_division=0),
        precision_score(y, cnn_pred, zero_division=0),
        precision_score(y, hybrid_pred, zero_division=0)
    ]
    
    recall = [
        recall_score(y, ml_pred, zero_division=0),
        recall_score(y, cnn_pred, zero_division=0),
        recall_score(y, hybrid_pred, zero_division=0)
    ]
    
    f1 = [
        f1_score(y, ml_pred, zero_division=0),
        f1_score(y, cnn_pred, zero_division=0),
        f1_score(y, hybrid_pred, zero_division=0)
    ]
    
    auc_scores = [
        roc_auc_score(y, ml_scores),
        roc_auc_score(y, cnn_scores),
        roc_auc_score(y, hybrid_scores)
    ]
    
    # Create bar chart
    x = np.arange(len(models))
    width = 0.16
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#AEC6FA', '#FFD9B3', '#B3E5B3', '#FFB3B3', '#E6B3FF']
    
    bars1 = ax.bar(x - 2*width, accuracy, width, label='Accuracy', color=colors[0], edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x - width, precision, width, label='Precision', color=colors[1], edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x, recall, width, label='Recall', color=colors[2], edgecolor='black', linewidth=1.2)
    bars4 = ax.bar(x + width, f1, width, label='F1-Score', color=colors[3], edgecolor='black', linewidth=1.2)
    bars5 = ax.bar(x + 2*width, auc_scores, width, label='AUC', color=colors[4], edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Models', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Comparative Model Performance\nChronic Heart Failure Detection', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=12, framealpha=0.95, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/model_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Model comparison chart saved to: results/model_comparison.png")
    plt.close()


# =========================
# Evaluation Pipeline
# =========================
def evaluate():
    print("Loading feature CSV...")
    df = load_csv_clean(FEATURE_CSV)

    # Plot training history first
    print("\nGenerating training history chart...")
    plot_training_history()

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

    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # -------------------------
    # Model Comparison Chart
    # -------------------------
    print("\nGenerating model comparison chart...")
    plot_model_comparison(ml_scores, cnn_scores, hybrid_scores, y)

    # -------------------------
    # Hybrid Model AUC Score
    # -------------------------
    hybrid_auc = roc_auc_score(y, hybrid_scores)

    print("\n" + "="*60)
    print("HeartGuard Hybrid Model - Receiver Operating Characteristic")
    print("="*60)
    print(f"AUC Score: {hybrid_auc:.4f}")
    print("="*60)

    # -------------------------
    # ROC Curve for Hybrid Model
    # -------------------------
    fpr_hybrid, tpr_hybrid, _ = roc_curve(y, hybrid_scores)

    plt.figure(figsize=(10, 8))

    # Plot ROC curve for hybrid model
    plt.plot(fpr_hybrid, tpr_hybrid, color='#2E86AB', label=f'HeartGuard Hybrid (AUC = {hybrid_auc:.4f})', linewidth=3)

    # Diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5000)', linewidth=2, alpha=0.7)

    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title('ROC Curve - HeartGuard Hybrid Model\nChronic Heart Failure Detection', fontsize=15, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()

    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/roc_curve_hybrid.png", dpi=300, bbox_inches='tight')
    print("\n✓ ROC curve saved to: results/roc_curve_hybrid.png")
    plt.close()

    # -------------------------
    # Confusion Matrix for Hybrid Model
    # -------------------------
    plt.figure(figsize=(10, 8))

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'],
                annot_kws={'size': 16, 'fontweight': 'bold'},
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='black')

    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.title('Confusion Matrix - HeartGuard Hybrid Model\nChronic Heart Failure Detection', fontsize=15, fontweight='bold')
    plt.tight_layout()

    # Save the confusion matrix figure
    plt.savefig("results/confusion_matrix_hybrid.png", dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to: results/confusion_matrix_hybrid.png")
    plt.close()


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    evaluate()
