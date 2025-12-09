import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys


# Load model & data
pipeline = joblib.load("app/pages/success_model.joblib")


def predict(Sleep_Duration, Dietary_Habits, Degree, Family_History_of_Mental_Illness,
            Have_you_ever_had_suicidal_thoughts, Gender, Age, Academic_Pressure,
            Study_Satisfaction, Work_Study_Hours, Financial_Stress, GPA):

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
        'GPA': GPA
    }])

    predicted = pipeline.predict(test_df)[0]
    predicted_label = "Depressed" if predicted == 1 else "Not Depressed"
    return predicted_label


def main():
    st.title('🧠 Depression Prediction System')

    GPA = st.number_input('GPA', min_value=0.1, max_value=4.0, value=3.0)
    Dietary_Habits = st.selectbox('Dietary Habits', ['Unhealthy', 'Moderate', "Healthy"])
    Degree = st.selectbox('Degree', ['Bachelor', 'Master', "Others", 'Business'])
    Sleep_Duration = st.selectbox('Sleep Duration', ["'5-6 hours'", "'Less than 5 hours'", "'7-8 hours'", "'More than 8 hours'"])
    Family_History_of_Mental_Illness = st.selectbox('Family History of Mental Illness', ['Yes', 'No'])
    Have_you_ever_had_suicidal_thoughts = st.selectbox('Have you ever had suicidal thoughts ?', ['Yes', 'No'])
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age = st.number_input('Age', min_value=18, max_value=43, value=20)
    Academic_Pressure = st.slider('Academic Pressure', 1.0, 5.0, 3.0, step=1.0)
    Study_Satisfaction = st.slider('Study Satisfaction', 1.0, 5.0, 3.0, step=1.0)
    Financial_Stress = st.slider('Financial Stress', 1.0, 5.0, 3.0, step=1.0)
    Work_Study_Hours = st.slider('Work/Study Hours', 0.0, 12.0, 3.0, step=1.0)

    pred = None

    if st.button('🔮 Predict'):
        pred = predict(
            Sleep_Duration, Dietary_Habits, Degree,
            Family_History_of_Mental_Illness, Have_you_ever_had_suicidal_thoughts,
            Gender, Age, Academic_Pressure, Study_Satisfaction,
            Work_Study_Hours, Financial_Stress, GPA
        )

        # ---- RESULT CARD ----
        if pred is not None:

            if pred == "Not Depressed":
                color = "#2ecc71"   # Green
                emoji = "😊"
            else:
                color = "#e74c3c"   # Red
                emoji = "⚠️"

            st.markdown(f"""
                <div style="
                    background-color: white;
                    border-left: 10px solid {color};
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
                    margin-top: 25px;
                ">
                    <h2 style="color:{color}; margin-bottom: 5px;">{emoji} Prediction Result</h2>
                    <p style="font-size:22px; font-weight:bold; color:black;">
                        The person is: <span style="color:{color};">{pred}</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)



if __name__ == '__main__':
    main()








