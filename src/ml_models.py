import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib


def train_ml_models(feature_csv, model_dir):
    df = pd.read_csv(feature_csv)
    X = df.drop(["file", "label"], axis=1)
    y = df["label"]  # Your dataset should have a "label" column

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs(model_dir, exist_ok=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    # SVM
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, os.path.join(model_dir, 'svm.pkl'))

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(model_dir, 'rf.pkl'))

    # XGBoost
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, os.path.join(model_dir, 'xgb.pkl'))

    print("Models trained successfully.")
    print(classification_report(y_test, svm.predict(X_test)))
