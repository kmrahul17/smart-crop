from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

train_dir = BASE_DIR / "datasets" / "PlantVillage" / "train"

gen = ImageDataGenerator(rescale=1./255)

train_data = gen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32
)

print("\nNumber of Classes:", train_data.num_classes)
print("\nClass Names:\n")
print(list(train_data.class_indices.keys()))