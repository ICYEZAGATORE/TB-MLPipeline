# src/model.py
import os
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# CONFIG
IMAGE_SIZE = (224, 224)
NUM_CHANNELS = 3
MODELS_DIR = "models"
LATEST_MODEL_PATH = os.path.join(MODELS_DIR, "image_classifier_latest.h5")
CLASS_MAP_PATH = os.path.join(MODELS_DIR, "class_map.json")


def get_class_map_from_train_folder(train_folder="data/train"):
    """
    Build class->index mapping from the folder names under data/train.
    This ensures consistency with preprocessing ordering when LabelEncoder isn't saved.
    """
    classes = sorted([d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))])
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    # inverse map (index -> class)
    inv = {v: k for k, v in class_map.items()}
    return class_map, inv


def build_cnn_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_preprocessed(path_prefix="preprocessed_data"):
    X_train = np.load(os.path.join(path_prefix, "X_train.npy"))
    X_test = np.load(os.path.join(path_prefix, "X_test.npy"))
    y_train = np.load(os.path.join(path_prefix, "y_train.npy"))
    y_test = np.load(os.path.join(path_prefix, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def save_class_map(inv_map):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(inv_map, f)
    print(f"Class map saved -> {CLASS_MAP_PATH}")


def load_class_map():
    if not os.path.exists(CLASS_MAP_PATH):
        # try to build from folder
        _, inv = get_class_map_from_train_folder()
        save_class_map(inv)
        return inv
    with open(CLASS_MAP_PATH, "r") as f:
        inv = json.load(f)
    # keys in json are strings; convert keys to ints
    inv = {int(k): v for k, v in inv.items()}
    return inv


def save_model_version(model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = os.path.join(MODELS_DIR, f"image_classifier_{ts}.h5")
    model.save(versioned)
    model.save(LATEST_MODEL_PATH)
    print(f"Saved model version: {versioned} and updated latest at {LATEST_MODEL_PATH}")
    return versioned


def train_model(epochs=10, batch_size=32, path_prefix="preprocessed_data"):
    X_train, X_test, y_train, y_test = load_preprocessed(path_prefix)
    num_classes = len(np.unique(y_train))
    print(f"Loaded data: X_train={X_train.shape}, X_test={X_test.shape}")

    model = build_cnn_model(num_classes=num_classes)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )
    save_model_version(model)

    # save or update class map (index->class) using train folder ordering
    _, inv = get_class_map_from_train_folder()
    save_class_map(inv)

    return model, history


def evaluate_loaded_model(model_path=None, path_prefix="preprocessed_data"):
    if model_path is None:
        model_path = LATEST_MODEL_PATH
    model = tf.keras.models.load_model(model_path)
    _, X_test, _, y_test = load_preprocessed(path_prefix)
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    return {"loss": float(loss), "accuracy": float(acc)}


def load_latest_model():
    if not os.path.exists(LATEST_MODEL_PATH):
        raise FileNotFoundError("No latest model found. Train model first.")
    return tf.keras.models.load_model(LATEST_MODEL_PATH)


if __name__ == "__main__":
    # quick CLI: python src/model.py train
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model(epochs=10)
    else:
        print("model.py: available commands: 'train' -> python src/model.py train'")
