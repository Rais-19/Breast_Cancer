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
    st.error("Model files not found. Make sure models/ exists in GitHub.")
    st.stop()

# ---------------------------
# UI
# ---------------------------

st.title("Breast Cancer Prediction")
st.write("Enter tumor features to predict benign or malignant.")

feature_names = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
    "radius error","texture error","perimeter error","area error","smoothness error",
    "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
    "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

inputs = []

for name in feature_names:
    inputs.append(st.number_input(name, value=0.0))

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict"):
    X = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    if pred == 1:
        st.success(f"Benign (Probability = {prob:.3f})")
    else:
        st.error(f"Malignant (Probability = {1-prob:.3f})")
