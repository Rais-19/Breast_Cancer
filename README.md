# Diabetes Prediction App

Simple web application that predicts diabetes risk using an SVM model trained on the PIMA Indians Diabetes Dataset.

## Features
- Trains SVM model from raw data
- Saves trained model
- Interactive web interface (Streamlit) for making predictions

## Project Structure
diabetes-prediction/
├── train_model.py        # Training script (creates the model)
├── app.py                # Streamlit web application
├── requirements.txt      # Dependencies
├── .gitignore
└── models/               # Saved trained model (git-ignored)
└── trained_model.sav
