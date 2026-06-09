import tensorflow as tf
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ======================================
# Dataset Paths
# ======================================

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "datasets" / "PlantVillage" / "train"
VAL_DIR = BASE_DIR / "datasets" / "PlantVillage" / "val"
MODELS_DIR = BASE_DIR / "models"

# ======================================
# Parameters
# ======================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3

# ======================================
# Data Generators
# ======================================

train_gen = ImageDataGenerator(
    rescale=1.0 / 255
)

val_gen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ======================================
# Save Class Names
# ======================================

class_names = list(train_data.class_indices.keys())

print("\nNumber of Classes:", len(class_names))

# ======================================
# MobileNetV2 Base Model
# ======================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers

base_model.trainable = False

# ======================================
# Classification Head
# ======================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

predictions = Dense(
    train_data.num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# ======================================
# Compile
# ======================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================================
# Summary
# ======================================

model.summary()

# ======================================
# Train
# ======================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ======================================
# Evaluate
# ======================================

loss, acc = model.evaluate(
    val_data
)

print(f"\nValidation Accuracy: {acc:.4f}")

# ======================================
# Save Model
# ======================================

model.save(
    MODELS_DIR / "disease_model.keras"
)

print("\nDisease Model Saved Successfully!")