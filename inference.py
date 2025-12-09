import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sys

MODEL_PATH = r"H:\Term 5\ML\Students depression\models\success_model.pkl"

def load_trained_model(model_path: str = MODEL_PATH):
    """
    Load the trained model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Did you run the training pipeline first?"
        )

    pipeline = joblib.load(model_path)
    return pipeline


# -------------------------
# Load model ONCE globally
# -------------------------
pipeline = load_trained_model()



def predict(Sleep_Duration, Dietary_Habits,Degree,Family_History_of_Mental_Illness, Have_you_ever_had_suicidal_thoughts, 
            Gender,Age,Academic_Pressure, Study_Satisfaction,Work_Study_Hours,
        Financial_Stress,GPA):

    test_df = pd.DataFrame([{
        
        'Sleep Duration': Sleep_Duration,
        'Dietary Habits': Dietary_Habits,
        'Degree': Degree,
        'Family History of Mental Illness': Family_History_of_Mental_Illness,
        'Have you ever had suicidal thoughts ?': Have_you_ever_had_suicidal_thoughts,
        'Gender': Gender,
        'Age': Age,
        'Academic Pressure': Academic_Pressure,
        'Study Satisfaction': Study_Satisfaction,
        'Work/Study Hours': Work_Study_Hours,
        'Financial Stress': Financial_Stress,
        'GPA': GPA,

    }])

    predicted = pipeline.predict(test_df)[0]
    predicted_label = "Depressed" if predicted == 1 else "Not Depressed"
    return predicted
