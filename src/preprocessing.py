import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# -----------------------------
# CONFIGURABLE PARAMETERS
# -----------------------------
IMAGE_SIZE = (224, 224)  # standard size for CNN
DATA_PATH = "data/train"  # base path to train folder
TEST_SPLIT = 0.2          # fraction for test split
RANDOM_STATE = 42

# -----------------------------
# 1. Load images and labels
# -----------------------------
def load_images_and_labels(data_path=DATA_PATH, image_size=IMAGE_SIZE):
    images = []
    labels = []
    classes = os.listdir(data_path)  # ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']
    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            try:
                img = load_img(img_path, target_size=image_size)  # resize
                img_array = img_to_array(img) / 255.0             # normalize pixels [0,1]
                images.append(img_array)
                labels.append(cls)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# -----------------------------
# 2. Encode labels
# -----------------------------
def encode_labels(labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    return labels_encoded, le

# -----------------------------
# 3. Split data into train/test
# -----------------------------
def split_data(X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# -----------------------------
# 4. Data augmentation
# -----------------------------
def create_datagen():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# -----------------------------
# 5. Save preprocessed data (optional)
# -----------------------------
def save_preprocessed_data(X_train, X_test, y_train, y_test, path="preprocessed_data/"):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, "X_train.npy"), X_train)
    np.save(os.path.join(path, "X_test.npy"), X_test)
    np.save(os.path.join(path, "y_train.npy"), y_train)
    np.save(os.path.join(path, "y_test.npy"), y_test)
    print(f"Preprocessed data saved to {path}")

# -----------------------------
# Example of using all functions
# -----------------------------
if __name__ == "__main__":
    print("Loading images...")
    X, y = load_images_and_labels()
    print(f"Total images loaded: {len(X)}")

    print("Encoding labels...")
    y_encoded, le = encode_labels(y)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    print("Creating data generator for augmentation...")
    datagen = create_datagen()
    datagen.fit(X_train)

    print("Optionally saving preprocessed data...")
    save_preprocessed_data(X_train, X_test, y_train, y_test)
