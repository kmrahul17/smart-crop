import joblib
import pandas as pd
import numpy as np

model = joblib.load("../models/crop_model.pkl")
encoder = joblib.load("../models/label_encoder.pkl")

sample = pd.DataFrame([[
    90,
    42,
    43,
    20.87,
    82.00,
    6.50,
    202.93
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

top3_idx = np.argsort(probs[0])[-3:][::-1]

print("\nTop 3 Crops:\n")

for idx in top3_idx:

    crop = encoder.inverse_transform([idx])[0]

    confidence = probs[0][idx] * 100

    print(
        crop,
        f"{confidence:.2f}%"
    )