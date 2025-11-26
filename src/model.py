# src/model.py
import os
import json
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Paths & config
IMAGE_SIZE = (224, 224)
NUM_CHANNELS = 3
MODELS_DIR = "models"
PREPROCESSED_DIR = "preprocessed_data"
LATEST_MODEL_PATH = os.path.join(MODELS_DIR, "image_classifier_latest.h5")
METRICS_DIR = os.path.join(MODELS_DIR, "metrics")


# -----------------------
# Helpers for class maps
# -----------------------
def load_class_indices(path=os.path.join(PREPROCESSED_DIR, "class_indices.json")):
    """
    Returns two dicts:
      - idx2class: {int: class_name}
      - class2idx: {class_name: int}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Class indices JSON not found at {path}. Run preprocessing first.")
    with open(path, "r") as f:
        raw = json.load(f)
    # raw keys are strings - assume mapping "0": "COVID19", ...
    idx2class = {int(k): v for k, v in raw.items()}
    class2idx = {v: int(k) for k, v in raw.items()}
    return idx2class, class2idx


# -----------------------
# Model builder
# -----------------------
def build_cnn_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# -----------------------
# Load preprocessed arrays
# -----------------------
def load_preprocessed(path_prefix=PREPROCESSED_DIR):
    X_train = np.load(os.path.join(path_prefix, "X_train.npy"))
    X_test = np.load(os.path.join(path_prefix, "X_test.npy"))
    y_train = np.load(os.path.join(path_prefix, "y_train.npy"))
    y_test = np.load(os.path.join(path_prefix, "y_test.npy"))
    return X_train, X_test, y_train, y_test


# -----------------------
# Save model version + metrics
# -----------------------
def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


def save_model_version(model):
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_path = os.path.join(MODELS_DIR, f"image_classifier_{ts}.h5")
    model.save(version_path)
    # update "latest"
    model.save(LATEST_MODEL_PATH)
    print(f"Saved model: {version_path} and updated latest -> {LATEST_MODEL_PATH}")
    return version_path


def save_metrics(metrics_dict):
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(METRICS_DIR, f"metrics_{ts}.json")
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    # also write/overwrite latest_metrics.json
    latest = os.path.join(METRICS_DIR, "latest_metrics.json")
    with open(latest, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics -> {path}")
    return path


# -----------------------
# Train from scratch
# -----------------------
def train_model(epochs=10, batch_size=32, path_prefix=PREPROCESSED_DIR, use_earlystop=True):
    X_train, X_test, y_train, y_test = load_preprocessed(path_prefix)
    num_classes = len(np.unique(y_train))
    model = build_cnn_model(num_classes)

    callbacks = []
    if use_earlystop:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))
        callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate and save metrics
    metrics = evaluate_model_on_test(model, X_test, y_test)
    save_model_version(model)
    save_metrics(metrics)

    return model, history, metrics


# -----------------------
# Evaluate helper
# -----------------------
def evaluate_model_on_test(model, X_test, y_test):
    # predictions (class indices)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "confusion_matrix": cm,
        "n_test": int(len(y_test))
    }
    return metrics


# -----------------------
# Load latest model
# -----------------------
def load_latest_model():
    if not os.path.exists(LATEST_MODEL_PATH):
        raise FileNotFoundError("No latest model found. Train a model first.")
    return tf.keras.models.load_model(LATEST_MODEL_PATH)


# -----------------------
# Retrain from existing model (continued training)
# -----------------------
def retrain_model_from_upload(upload_folder, epochs=5, batch_size=32, path_prefix=PREPROCESSED_DIR,
                              continue_from_latest=True, use_earlystop=True):
    """
    upload_folder: path to extracted upload which should contain subfolders per class:
        upload_folder/<CLASS_NAME>/*.jpg
    This function:
      - loads uploaded images, maps their class names using preprocessed class_indices.json
      - loads existing X_train,y_train from preprocessed data
      - appends new data
      - loads latest model and continues training (fine-tuning)
      - saves version and metrics
    """

    # 1) Load preprocessed baseline
    X_train, X_test, y_train, y_test = load_preprocessed(path_prefix)

    # 2) Load uploaded images using the same loader as preprocessing
    from src.preprocessing import load_images_and_labels  # local function we created
    X_new, y_new_names = load_images_and_labels(data_path=upload_folder, image_size=IMAGE_SIZE)

    # 3) map uploaded class names to indices using class_indices.json
    _, class2idx = load_class_indices()  # class2idx maps class_name -> int
    y_new = []
    for cname in y_new_names:
        if cname not in class2idx:
            raise ValueError(f"Uploaded class '{cname}' not found in existing class map. Make sure class names match exactly.")
        y_new.append(class2idx[cname])
    y_new = np.array(y_new)

    # 4) Combine datasets
    X_combined = np.concatenate([X_train, X_new], axis=0)
    y_combined = np.concatenate([y_train, y_new], axis=0)

    # 5) Prepare model (continue training from latest or create new)
    if continue_from_latest and os.path.exists(LATEST_MODEL_PATH):
        model = load_latest_model()
        print("Loaded latest model for fine-tuning.")
    else:
        num_classes = len(np.unique(y_combined))
        model = build_cnn_model(num_classes)
        print("Built a fresh model for training.")

    # 6) Callbacks
    callbacks = []
    if use_earlystop:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))
        callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3))

    # 7) Retrain / Fine-tune
    history = model.fit(
        X_combined, y_combined,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 8) Evaluate and save
    metrics = evaluate_model_on_test(model, X_test, y_test)
    save_model_version(model)
    save_metrics(metrics)

    return model, history, metrics


# -----------------------
# CLI entry
# -----------------------
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "train":
        train_model(epochs=10)
    elif cmd == "retrain":
        # usage: python src/model.py retrain path/to/uploaded_folder
        if len(sys.argv) < 3:
            print("Usage: python src/model.py retrain <uploaded_folder> [epochs]")
        else:
            folder = sys.argv[2]
            e = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            retrain_model_from_upload(folder, epochs=e)
    else:
        print("Commands: train | retrain <upload_folder>")
