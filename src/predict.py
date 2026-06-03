# src/predict.py
from pathlib import Path
import joblib
import numpy as np

# Dynamically find the absolute path to the directory this file lives in
BASE_DIR = Path(__file__).resolve().parent

def predict(input_data):
    
    model_path = BASE_DIR / "iris_model.pkl" 
    
    model = joblib.load(model_path)
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    return int(prediction[0])