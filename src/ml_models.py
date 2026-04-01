import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from xgboost import XGBClassifier


def train_ml_models(feature_csv, model_dir):
    # Load data
    df = pd.read_csv(feature_csv)

    X = df.drop(["file", "label"], axis=1)
    y = df["label"]

    # Train-test split (stratified for class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    os.makedirs(model_dir, exist_ok=True)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    # =========================
    # SVM (Balanced)
    # =========================
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
    svm.fit(X_train, y_train)
    joblib.dump(svm, os.path.join(model_dir, "svm.pkl"))

    # =========================
    # Random Forest (Balanced)
    # =========================
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(model_dir, "rf.pkl"))

    # =========================
    # XGBoost (Imbalance-aware)
    # =========================
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, os.path.join(model_dir, "xgb.pkl"))

    # =========================
    # Evaluation (XGBoost)
    # =========================
    y_pred = xgb.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nModels trained and saved successfully.")
