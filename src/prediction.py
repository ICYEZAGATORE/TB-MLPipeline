# src/prediction.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from src.model import load_latest_model, load_class_map, IMAGE_SIZE

def preprocess_image_file(image_path, image_size=IMAGE_SIZE):
    img = load_img(image_path, target_size=image_size)
    arr = img_to_array(img) / 255.0
    # ensure shape: (1, h, w, c)
    return np.expand_dims(arr, axis=0)

def predict_image_file(image_path):
    model = load_latest_model()
    class_map = load_class_map()  # index -> class_name
    x = preprocess_image_file(image_path)
    preds = model.predict(x)
    class_idx = int(np.argmax(preds, axis=1)[0])
    class_name = class_map.get(class_idx, str(class_idx))
    confidence = float(np.max(preds))
    return {"class_idx": class_idx, "class_name": class_name, "confidence": confidence}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/prediction.py path/to/image.jpg")
    else:
        out = predict_image_file(sys.argv[1])
        print(out)
