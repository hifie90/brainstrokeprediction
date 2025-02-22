import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the ensemble model
ensemble_model = joblib.load("ensemble_model.pkl")

# Select a model
model_name = st.selectbox("Choose a Model", ["Extra Trees", "Decision Tree", "XGBoost"])
model = ensemble_model["models"][model_name]  

st.title("🩺 Stroke Prediction App")

# User Inputs
age = float(st.slider("Age", 0, 100, 50))
hypertension = st.radio("Do you have Hypertension?", ["No", "Yes"])
heart_disease = st.radio("Do you have Heart Disease?", ["No", "Yes"])
avg_glucose_level = float(st.slider("Average Glucose Level", 50.0, 300.0, 100.0))
ever_married = st.radio("Have you ever been married?", ["No", "Yes"])

# Convert Yes/No to 0/1
hypertension = float(1 if hypertension == "Yes" else 0)
heart_disease = float(1 if heart_disease == "Yes" else 0)
ever_married = float(1 if ever_married == "Yes" else 0)

# ✅ Select only the required features
input_data = np.array([[age, hypertension, heart_disease, avg_glucose_level, ever_married]])

# Apply MinMaxScaler (fit on the same range as training)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(input_data)  # Scale input

# Predict using selected model
if st.button("🔍 Predict"):
    prediction = model.predict(scaled_data)  

    if prediction[0] == 1:
        st.error("⚠️ High risk of stroke!")
    else:
        st.success("✅ Low risk of stroke.")
