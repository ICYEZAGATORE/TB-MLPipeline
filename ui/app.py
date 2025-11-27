import streamlit as st
import requests
from PIL import Image
import io
import os
import json

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="TB ML Pipeline", layout="wide")
st.title("🫁 TB ML Pipeline Dashboard")

menu = ["Predict", "Upload Data", "Retrain Model", "Metrics", "API Health"]
choice = st.sidebar.selectbox("Navigate", menu)

# --------------------------------------------------------------
# 1. PREDICT
# --------------------------------------------------------------
if choice == "Predict":
    st.header("🔍 Predict TB Condition from X-ray")

    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=350)

        if st.button("Predict"):

            # FIXED: Streamlit correct file sending format
            files = {
                "file": (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type
                )
            }

            try:
                response = requests.post(f"{API_URL}/predict", files=files)
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.stop()

            if response.status_code == 200:
                st.success("Prediction result:")
                st.json(response.json())
            else:
                st.error("Prediction failed")
                try:
                    st.json(response.json())
                except:
                    st.error(response.text)


# --------------------------------------------------------------
# 2. UPLOAD ZIP DATA
# --------------------------------------------------------------
elif choice == "Upload Data":
    st.header("📤 Upload ZIP dataset for retraining")

    zip_file = st.file_uploader("Upload ZIP", type=["zip"])

    if zip_file and st.button("Upload"):
        files = {
            "zip_file": (
                zip_file.name,
                zip_file.getvalue(),
                "application/zip"
            )
        }

        response = requests.post(f"{API_URL}/upload-data", files=files)

        if response.status_code == 200:
            st.success(response.json())
        else:
            st.error("Upload failed.")
            st.json(response.json())


# --------------------------------------------------------------
# 3. RETRAIN MODEL
# --------------------------------------------------------------
elif choice == "Retrain Model":
    st.header("🔄 Retrain Model")

    upload_path = st.text_input("Enter upload_path from /upload-data")
    epochs = st.number_input("Epochs", min_value=1, max_value=20, value=5)

    if st.button("Start Retraining"):
        data = {"upload_path": upload_path, "epochs": int(epochs)}
        response = requests.post(f"{API_URL}/retrain", data=data)

        if response.status_code == 200:
            st.success("Retraining completed")
            st.json(response.json())
        else:
            st.error("Retraining failed")
            st.json(response.json())


# --------------------------------------------------------------
# 4. METRICS
# --------------------------------------------------------------
elif choice == "Metrics":
    st.header("📊 Latest Model Metrics")

    response = requests.get(f"{API_URL}/metrics")

    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error("Failed to fetch metrics")


# --------------------------------------------------------------
# 5. HEALTH
# --------------------------