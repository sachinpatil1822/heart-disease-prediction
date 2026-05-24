import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Page Title
st.title("❤️ Heart Disease Prediction App")

st.write("Fill the patient details below")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=100)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250
)

chol = st.number_input(
    "Cholesterol Level",
    min_value=100,
    max_value=600
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250
)

oldpeak = st.number_input(
    "Oldpeak Value",
    min_value=0.0,
    max_value=10.0,
    step=0.1
)

# Prediction Button
if st.button("Predict"):

    # Prepare features
    features = np.array([[
        age,
        trestbps,
        chol,
        thalach,
        oldpeak
    ]])

    # Prediction
    prediction = model.predict(features)

    # High Risk
    if prediction[0] == 1:

        st.error("⚠️ High Risk of Heart Disease")

        st.subheader("Health Tips")

        tips = [
            "Avoid smoking and alcohol",
            "Reduce oily and junk food",
            "Exercise daily",
            "Drink more water",
            "Reduce stress",
            "Maintain proper sleep",
            "Avoid sugary drinks"
        ]

        for tip in tips:
            st.write("✅", tip)

    # Low Risk
    else:

        st.success("✅ Low Risk of Heart Disease")

        st.subheader("Health Tips")

        tips = [
            "Maintain healthy lifestyle",
            "Continue regular exercise",
            "Eat healthy food",
            "Stay hydrated",
            "Do regular health checkups"
        ]

        for tip in tips:
            st.write("✅", tip)
