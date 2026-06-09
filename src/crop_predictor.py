import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "crop_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

def predict_top3(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    sample = pd.DataFrame([[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]], columns=[
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ])

    probs = model.predict_proba(sample)

    top3_idx = np.argsort(
        probs[0]
    )[-3:][::-1]

    results = []

    for idx in top3_idx:

        crop = encoder.inverse_transform(
            [idx]
        )[0]

        confidence = round(
            probs[0][idx] * 100,
            2
        )

        results.append({
            "crop": crop,
            "confidence": confidence
        })

    return results