import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# Load Dataset
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "datasets" / "Crop_recommendation.csv"
MODELS_DIR = BASE_DIR / "models"

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================
# Encode Labels
# ==========================

le = LabelEncoder()

df["label"] = le.fit_transform(df["label"])

# ==========================
# Features and Target
# ==========================

X = df.drop("label", axis=1)

y = df["label"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Train Random Forest
# ==========================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

rf.fit(X_train, y_train)

# ==========================
# Predictions
# ==========================

y_pred = rf.predict(X_test)

# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================
# Feature Importance
# ==========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# ==========================
# Save Model
# ==========================

joblib.dump(
    rf,
    MODELS_DIR / "crop_model.pkl"
)

joblib.dump(
    le,
    MODELS_DIR / "label_encoder.pkl"
)

print("\nModel Saved Successfully!")