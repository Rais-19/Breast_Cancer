import streamlit as st
import numpy as np
import joblib
import os

# ---------------------------
# Load model and scaler
# ---------------------------

MODEL_PATH = os.path.join("models", "breast_cancer_model.joblib")
SCALER_PATH = os.path.join("models", "scaler.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error("Model file
