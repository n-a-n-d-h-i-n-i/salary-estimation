import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Salary Estimation App")
st.write("Upload your CSV file for salary estimation or enter details below:")

st.write("Enter details below to estimate salary:")

# Input fields for all features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
education_num = st.number_input("Education Number", min_value=1, max_value=20, value=10)
capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

if st.button("Estimate Salary"):
    # Prepare input for prediction
    input_features = np.array([[age, education_num, capital_gain, hours_per_week]])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
    result = '>50K' if prediction == 1 else '<=50K'
    st.write(f"Estimated Salary: {result}")
