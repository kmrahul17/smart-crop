import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =====================================
# Load Trained Model
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "disease_model.keras"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. Run train_disease_model.py first."
    )

model = load_model(
    MODEL_PATH
)

# =====================================
# Class Names
# =====================================

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


def format_crop_name(class_name):

    crop_name = class_name.split("___", 1)[0]

    return crop_name.replace("_", " ")

# =====================================
# Prediction Function
# =====================================

def predict_disease(img_path):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )

    pred_idx = np.argmax(predictions)

    confidence = np.max(predictions) * 100

    full_prediction = class_names[pred_idx]

    crop, disease = full_prediction.split("___", 1)

    crop = crop.replace("_", " ")
    disease = disease.replace("_", " ")

    return crop, disease, confidence


# =====================================
# Test
# =====================================

if __name__ == "__main__":

    image_path = BASE_DIR / "src" / "sample_leaf.jpg"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}."
        )

    crop, disease, confidence = predict_disease(
        image_path
    )

    print("\nCrop:", crop)
    print("\nDisease:", disease)
    print(
        f"Confidence: {confidence:.2f}%"
    )