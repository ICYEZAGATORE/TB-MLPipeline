# app.py
import os
import time
import shutil
import zipfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.prediction import predict_image_file, TMP_DIR
from src.model import retrain_model_from_upload, load_latest_model, load_class_indices, load_preprocessed, save_metrics

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


@app.get("/health")
def health():
    return {"status": "ok", "uptime_seconds": time.time() - START_TIME}


@app.get("/metrics")
def metrics():
    # return uptime and latest model metrics if available
    uptime = time.time() - START_TIME
    metrics_path = os.path.join("models", "metrics", "latest_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    # basic model info
    model_exists = os.path.exists("models/image_classifier_latest.h5")
    return {"uptime_seconds": uptime, "model_exists": model_exists, "latest_metrics": metrics}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # save file temporarily and predict
    fname = file.filename
    tmp_path = os.path.join(TMP_DIR, fname)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    try:
        out = predict_image_file(tmp_path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # remove tmp file
        try:
            os.remove(tmp_path)
        except:
            pass
    return out


@app.post("/upload-data")
async def upload_data(zip_file: UploadFile = File(...)):
    """
    Accepts a zip file that contains a folder with class subfolders:
      upload.zip -> extracted to uploads/<uid>/...
    Returns path to extracted folder (server-side) which can be used to call /retrain
    """
    uid = str(int(time.time()))
    dest_dir = os.path.join(UPLOADS_DIR, uid)
    os.makedirs(dest_dir, exist_ok=True)

    zip_path = os.path.join(dest_dir, zip_file.filename)
    with open(zip_path, "wb") as f:
        f.write(await zip_file.read())

    # extract
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
    except zipfile.BadZipFile:
        # maybe single folder uploaded - move file
        return JSONResponse({"error": "Uploaded file is not a valid zip"}, status_code=400)
    finally:
        os.remove(zip_path)

    return {"upload_path": dest_dir, "message": "Uploaded and extracted. Call /retrain with this path."}


@app.post("/retrain")
def retrain(upload_path: str = Form(...), epochs: int = Form(5)):
    """
    Trigger retraining using the uploaded folder path returned by /upload-data.
    The upload_path must be a path on the server (returned by /upload-data).
    """
    if not os.path.exists(upload_path):
        return JSONResponse({"error": "upload_path does not exist on server"}, status_code=400)
    try:
        model, history, metrics = retrain_model_from_upload(upload_path, epochs=int(epochs))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"message": "Retraining finished", "metrics": metrics}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
