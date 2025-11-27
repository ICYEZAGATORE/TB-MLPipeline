# TB-MLPipeline
#  TB ML Pipeline – End-to-End Machine Learning System

An end-to-end Machine Learning pipeline that processes Chest X-ray images to identify pneumonia covid19 and tuberculosis, trains a deep learning classification model, exposes predictions through a **FastAPI backend**, and provides a user-friendly **Streamlit frontend** for inference and monitoring.

---

##  **Video Demo**
Watch the full demonstration of the project here:

 **YouTube Demo:** *(https://youtu.be/mndaUBLNvLo)*

---

##  **APP URL**
 
- **Streamlit App URL:** http://localhost:8501  

---

Project Description

This is a full machine-learning pipeline for TB detection in chest X-ray images.
The system includes:

✔️ 1. CNN Model (TensorFlow/Keras)

Image preprocessing and augmentation

Training on labeled TB vs Normal X-rays

Saving the model (model.h5) and class indices (class_indices.json)

 2. FastAPI Backend

/predict → Predict from uploaded X-ray image

/upload-data → Upload ZIP datasets

/retrain → Retrain model with new data

/health → API uptime

/metrics → Latest accuracy, loss, timestamp

 3. Streamlit Frontend

Provides a user-friendly dashboard with:

Image upload & prediction

Dataset ZIP upload

Model retraining

Metrics display

Health status

 4. Flood Request Simulation

The system was tested using rapid repeated /predict requests to confirm API stability under load.

 ## Setup Instructions (Run Everything Locally)
 1. Clone the repository
 2. Activate Virtual environment
 3. Install required libraries
 4. Start the API
 5.Run the streamlit app ui/app.py

 ## Using the Application
Predict a Condition

1. Open Predict page

2. Upload an X-ray (.jpg, .png, .jpeg)

3. Click Predict

4. Result appears instantly

Upload ZIP Dataset

1. Upload new dataset to be used for retraining

2. API returns a folder path

 Retrain the Model

1. Enter the returned upload path

2. Set epochs

3. Start retraining

Check Metrics

Shows:

Accuracy

Loss

Epochs trained

Timestamp