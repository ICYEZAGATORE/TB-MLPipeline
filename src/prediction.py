# src/prediction.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from src.model import load_latest_model, load_class_indices, IMAGE_SIZE

TMP_DIR = "tmp_uploads"
os.makedirs(TMP_DIR, exist_ok=True)

def preprocess_image_file(image_path, image_size=IMAGE_SIZE):
    img = load_img(image_path, target_size=image_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image_file(image_path):
    # Load model and class map
    model = load_latest_model()
    idx2class, _ = load_class_indices()  # idx2class returned as (idx2class, class2idx) in model.load_class_indices? 
    # Note: load_class_indices in model.py returns (idx2class, class2idx)
    if isinstance(idx2class, dict) and all(isinstance(k, str) for k in idx2class.keys()):
        # handle if keys were strings by mistake
        idx2class = {int(k): v for k, v in idx2class.items()}

    x = preprocess_image_file(image_path)
    preds = model.predict(x)
    class_idx = int(np.argmax(preds, axis=1)[0])
    class_name = idx2class.get(class_idx, str(class_idx))
    confidence = float(np.max(preds))
    return {"class_idx": class_idx, "class_name": class_name, "confidence": confidence}

# CLI usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/prediction.py <image_path>")
    else:
        print(predict_image_file(sys.argv[1]))
