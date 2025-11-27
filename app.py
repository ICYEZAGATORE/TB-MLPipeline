# app.py
import os
import time
import shutil
import zipfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

from src.prediction import predict_image_file, TMP_DIR
from src.model import (
    retrain_model_from_upload,
    load_latest_model,
    load_class_indices,
    load_preprocessed,
    save_metrics
)

app = FastAPI(title="TB-ML-Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ----------------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": time.time() - START_TIME
    }


# ----------------------------------------------------------
# METRICS REPORT
# ----------------------------------------------------------
@app.get("/metrics")
def metrics():
    uptime = time.time() - START_TIME
    metrics_path = os.path.join("models", "metrics", "latest_metrics.json")

    model_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            model_metrics = json.load(f)

    model_exists = os.path.exists("models/image_classifier_latest.h5")

    return {
        "uptime_seconds": uptime,
        "model_exists": model_exists,
        "latest_metrics": model_metrics
    }


# ----------------------------------------------------------
# PREDICT
# ----------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Ensure we are working with binary
    contents = await file.read()

    # Validate format using PIL directly
    try:
        from PIL import Image
        import io
        Image.open(io.BytesIO(contents))
    except:
        return JSONResponse({"error": "Uploaded file is not a valid image"}, status_code=400)

    # Save file correctly
    tmp_path = os.path.join(TMP_DIR, file.filename)
    with open(tmp_path, "wb") as f:
        f.write(contents)

    # Run prediction
    try:
        out = predict_image_file(tmp_path)
        return out
    except Exception as e:
        import traceback
        print("\n🔥🔥 PREDICTION ERROR🔥🔥")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


# ----------------------------------------------------------
# UPLOAD ZIP DATA FOR RETRAINING
# ----------------------------------------------------------
@app.post("/upload-data")
async def upload_data(zip_file: UploadFile = File(...)):
    """
    Accepts a ZIP file containing class folders.
    """
    uid = str(int(time.time()))
    dest_dir = os.path.join(UPLOADS_DIR, uid)
    os.makedirs(dest_dir, exist_ok=True)

    zip_path = os.path.join(dest_dir, zip_file.filename)

    with open(zip_path, "wb") as f:
        f.write(await zip_file.read())

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
    except zipfile.BadZipFile:
        return JSONResponse({"error": "Invalid ZIP file"}, status_code=400)
    finally:
        os.remove(zip_path)

    return {
        "upload_path": dest_dir,
        "message": "Uploaded & extracted successfully. Use this path with /retrain"
    }


# ----------------------------------------------------------
# RETRAIN MODEL
# ----------------------------------------------------------
@app.post("/retrain")
def retrain(upload_path: str = Form(...), epochs: int = Form(5)):
    if not os.path.exists(upload_path):
        return JSONResponse(
            {"error": f"upload_path does not exist: {upload_path}"},
            status_code=400
        )

    try:
        model, history, metrics = retrain_model_from_upload(
    upload_folder=upload_path,  # ✅ CORRECT
    epochs=epochs
)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return {
        "message": "Retraining completed successfully",
        "metrics": metrics
    }


# ----------------------------------------------------------
# RUN API
# ----------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
