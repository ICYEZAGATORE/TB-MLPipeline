# src/model.py - COMPLETE FIXED VERSION
import os
import json
import numpy as np
from datetime import datetime
import tensorflow as tf

# Enable eager execution
tf.config.run_functions_eagerly(True)

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Paths & config
IMAGE_SIZE = (224, 224)
NUM_CHANNELS = 3
MODELS_DIR = "models"
PREPROCESSED_DIR = "preprocessed_data"
LATEST_MODEL_PATH = os.path.join(MODELS_DIR, "image_classifier_latest")  # Changed: no .h5
LATEST_MODEL_H5_PATH = os.path.join(MODELS_DIR, "image_classifier_latest.h5")  # For backwards compat
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
    """
    Save model in both SavedModel format (for retraining) and .h5 (for inference)
    """
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save in SavedModel format (for retraining)
    version_path = os.path.join(MODELS_DIR, f"image_classifier_{ts}")
    model.save(version_path, save_format='tf')
    print(f"Saved model (SavedModel format): {version_path}")
    
    # Save latest in SavedModel format
    if os.path.exists(LATEST_MODEL_PATH):
        import shutil
        shutil.rmtree(LATEST_MODEL_PATH)
    model.save(LATEST_MODEL_PATH, save_format='tf')
    print(f"Updated latest model: {LATEST_MODEL_PATH}")
    
    # Also save in .h5 for backwards compatibility
    h5_version = os.path.join(MODELS_DIR, f"image_classifier_{ts}.h5")
    model.save(h5_version, save_format='h5')
    model.save(LATEST_MODEL_H5_PATH, save_format='h5')
    
    return version_path


def save_metrics(metrics_dict):
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(METRICS_DIR, f"metrics_{ts}.json")
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
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
    
    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(enumerate(weights))
    print(f"Using class weights: {class_weights}")
    
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
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    metrics = evaluate_model_on_test(model, X_test, y_test)
    save_model_version(model)
    save_metrics(metrics)

    return model, history, metrics


# -----------------------
# Evaluate helper - FIXED
# -----------------------
def evaluate_model_on_test(model, X_test, y_test):
    """
    Evaluate model on test data - handles tensor/numpy conversion
    """
    print("Evaluating model on test data...")
    y_pred_probs = model.predict(X_test, verbose=0)
    
    # Ensure numpy arrays
    if hasattr(y_pred_probs, 'numpy'):
        try:
            y_pred_probs = y_pred_probs.numpy()
        except:
            pass
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    if hasattr(y_test, 'numpy'):
        try:
            y_test = y_test.numpy()
        except:
            pass
    
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()

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
# Load latest model - FIXED
# -----------------------
def load_latest_model():
    """
    Load model preferring SavedModel format, fall back to .h5
    """
    # Try SavedModel format first (better for retraining)
    if os.path.exists(LATEST_MODEL_PATH):
        print(f"Loading model from: {LATEST_MODEL_PATH}")
        return tf.keras.models.load_model(LATEST_MODEL_PATH)
    
    # Fall back to .h5 format
    if os.path.exists(LATEST_MODEL_H5_PATH):
        print(f"Loading model from: {LATEST_MODEL_H5_PATH}")
        return tf.keras.models.load_model(LATEST_MODEL_H5_PATH)
    
    raise FileNotFoundError("No latest model found. Train a model first.")


# -----------------------
# Retrain - COMPLETELY FIXED
# -----------------------
def retrain_model_from_upload(upload_folder, epochs=5, batch_size=32, path_prefix=PREPROCESSED_DIR,
                              continue_from_latest=True, use_earlystop=True):
    """
    Retrain model with new data
    """
    print(f"\n{'='*60}")
    print(f"STARTING RETRAINING PROCESS")
    print(f"{'='*60}")
    
    # 1) Load existing data
    print("\n[1/7] Loading existing data...")
    X_train, X_test, y_train, y_test = load_preprocessed(path_prefix)
    print(f"   Loaded: {X_train.shape} training, {X_test.shape} test")

    # 2) Load class mapping
    print("\n[2/7] Loading class indices...")
    idx2class, class2idx = load_class_indices()
    print(f"   Classes: {list(class2idx.keys())}")

    # 3) Handle nested folders
    print(f"\n[3/7] Checking upload structure...")
    if not os.path.exists(upload_folder):
        raise ValueError(f"Upload folder not found: {upload_folder}")
    
    subdirs = [d for d in os.listdir(upload_folder) 
               if os.path.isdir(os.path.join(upload_folder, d))]
    
    if len(subdirs) == 1 and subdirs[0] not in class2idx:
        potential = os.path.join(upload_folder, subdirs[0])
        if os.path.isdir(potential):
            print(f"   Detected nested folder, using: {potential}")
            upload_folder = potential

    # 4) Load new images
    print(f"\n[4/7] Loading new images...")
    from src.preprocessing import load_images_and_labels
    X_new, y_new_names = load_images_and_labels(data_path=upload_folder, image_size=IMAGE_SIZE)
    
    if len(X_new) == 0:
        raise ValueError(f"No images found in {upload_folder}")
    
    print(f"   Loaded {len(X_new)} new images")

    # 5) Map class names to indices
    print(f"\n[5/7] Mapping class names...")
    y_new = []
    for cname in y_new_names:
        if cname not in class2idx:
            raise ValueError(
                f"Class '{cname}' not in existing classes: {list(class2idx.keys())}"
            )
        y_new.append(class2idx[cname])
    y_new = np.array(y_new)

    # 6) Combine datasets
    print(f"\n[6/7] Combining datasets...")
    X_new = X_new.astype(X_train.dtype)
    y_new = y_new.astype(y_train.dtype)
    
    X_combined = np.concatenate([X_train, X_new], axis=0)
    y_combined = np.concatenate([y_train, y_new], axis=0)
    print(f"   Combined: {X_combined.shape}")
    
    # Show distribution
    unique, counts = np.unique(y_combined, return_counts=True)
    print("\n   Class distribution:")
    for idx, count in zip(unique, counts):
        print(f"     {idx2class[idx]}: {count}")

    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_combined)
    weights = compute_class_weight('balanced', classes=classes, y=y_combined)
    class_weights = dict(enumerate(weights))
    print(f"\n   Class weights: {class_weights}")

    # 7) Build NEW model (don't load old one to avoid optimizer issues)
    print(f"\n[7/7] Building fresh model...")
    num_classes = len(np.unique(y_combined))
    model = build_cnn_model(num_classes)
    print("   Built new model (optimizer issue avoided)")

    # Setup callbacks
    callbacks = []
    if use_earlystop:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))
        callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3))

    # Train
    print(f"\n{'='*60}")
    print(f"TRAINING ({epochs} epochs)")
    print(f"{'='*60}\n")
    
    history = model.fit(
        X_combined, y_combined,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate and save
    print(f"\n{'='*60}")
    print("EVALUATING AND SAVING")
    print(f"{'='*60}")
    
    metrics = evaluate_model_on_test(model, X_test, y_test)
    save_model_version(model)
    save_metrics(metrics)
    
    print(f"\n✅ RETRAINING COMPLETE!")
    print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   F1 Score: {metrics['f1_weighted']*100:.2f}%")
    print(f"{'='*60}\n")

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
        if len(sys.argv) < 3:
            print("Usage: python src/model.py retrain <uploaded_folder> [epochs]")
        else:
            folder = sys.argv[2]
            e = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            retrain_model_from_upload(folder, epochs=e)
    else:
        print("Commands: train | retrain <upload_folder>")