import streamlit as st
import requests
import json
import time
from PIL import Image
import numpy as np

API_URL = "http://127.0.0.1:8000"   # update when deploying

st.title("TB Classification UI")

st.sidebar.header("Menu")
page = st.sidebar.selectbox("Select page", ["Dashboard", "Predict", "Retrain Model"])

# ----------------------
# DASHBOARD
# ----------------------
if page == "Dashboard":
    st.subheader("Model Uptime")

    # PLACEHOLDER — API endpoint
    # TODO: Connect to API: GET /uptime
    st.warning("PLACEHOLDER: Connect to API /uptime")

    st.write("Example uptime: 3 hours 12 minutes")

    st.subheader("Feature Visualizations (Sample)")

    st.image("viz1.png", caption="Feature visualization 1")   # Replace with real plots
    st.image("viz2.png", caption="Feature visualization 2")
    st.image("viz3.png", caption="Feature visualization 3")

# ----------------------
# PREDICT
# ----------------------
elif page == "Predict":
    st.subheader("Upload Image for Prediction")

    uploaded = st.file_uploader("Upload image", type=["jpg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width=300)

        if st.button("Predict"):
            st.write("Sending to API...")

            # PLACEHOLDER — API endpoint
            # TODO: Connect to API: POST /predict
            st.warning("PLACEHOLDER: Connect to API /predict")

            st.success("Prediction: TB-Positive (sample placeholder)")

# ----------------------
# RETRAINING
# ----------------------
elif page == "Retrain Model":
    st.subheader("Upload New Training Data")

    uploaded_files = st.file_uploader(
        "Upload multiple images", type=["jpg", "png"], accept_multiple_files=True
    )

    if st.button("Trigger Retraining"):
        st.write("Sending files to API...")

        # PLACEHOLDER — API endpoint
        # TODO: Connect to API: POST /retrain
        st.warning("PLACEHOLDER: Connect to API /retrain")

        st.success("Retraining triggered! (placeholder)")
