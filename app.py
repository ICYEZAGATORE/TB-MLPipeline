# app.py (at project root)
import os
import time
import shutil
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.prediction import predict_image_file
from src.model import load_latest_model, save_model_version, evaluate_loaded_model
from src.retrain import retrain_with_new_folder

app = FastAPI(title="TB-ML-Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

@app.get("/metrics")
def metrics():
    uptime = time.time() - START_TIME
    # basic model evaluation if model exists
    try:
        evals = evaluate_loaded_model()
    except Exception as e:
        evals = {"error": str(e)}
    return {"uptime_seconds": uptime, "model_eval": evals}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # save to temp file then predict
    tmp_dir = "tmp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    out = predict_image_file(tmp_path)
    # optionally remove file
    try:
        os.remove(tmp_path)
    except:
        pass
    return JSONResponse(content=out)

@app.post("/upload-data")
async def upload_data(zip_file: UploadFile = File(...)):
    """
    Accepts a zip file containing folder structure train/<class>/*.jpg
    The endpoint will unzip to uploads/<unique> and return path for retrain.
    """
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    uid = str(int(time.time()))
    dest_dir = os.path.join(uploads_dir, uid)
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, zip_file.filename)
    with open(zip_path, "wb") as f:
        f.write(await zip_file.read())

    # unzip
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    # remove zip
    os.remove(zip_path)
    return {"upload_path": dest_dir, "message": "Uploaded and extracted. Call /retrain with this path."}

@app.post("/retrain")
def retrain(upload_path: str = Form(...), epochs: int = Form(5)):
    """
    Trigger retraining from an uploaded folder path (returned from /upload-data)
    Example form fields: upload_path=/full/path/to/uploads/12345, epochs=5
    """
    if not os.path.exists(upload_path):
        return JSONResponse({"error": "upload_path does not exist"}, status_code=400)
    # call retrain
    try:
        model, history = retrain_with_new_folder(upload_path, epochs=int(epochs))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"message": "Retraining finished", "epochs": int(epochs)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
