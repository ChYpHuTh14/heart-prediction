import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# ======================
# Load Model & Preprocessor
# ======================
@st.cache_resource
def load_artifacts():
    model = load_model("heart_model4.keras")
    preprocessor = joblib.load("preprocessor_heart.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

# ======================
# UI
# ======================
st.title("🫀 Heart Disease Prediction")
st.write("Masukkan data kesehatan untuk memprediksi kemungkinan heart disease.")

# ======================
# Input Form
# ======================
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", ["Male", "Female"])
chest = st.selectbox("Chest Pain Type",
                     ["TA (Typical Angina)",
                      "ATA (Atypical Angina)",
                      "NAP (Non-Anginal Pain)",
                      "ASY (Asymptomatic)"
                      ]
                      )
resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, value=120)
chol = st.number_input("Cholesterol (mm/dl)", min_value=0, value=220)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 (mg/dl)?", ["Yes", "No"])
resting_ecg = st.selectbox("Resting ECG Result",
                           ["Normal",
                            "ST",
                            "LVH (Left Ventricular Hypertrophy)"
                            ]
                           )
max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=202, value=100)
exercise_angina = st.selectbox("Exercise Angina?", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak", min_value=-2.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ======================
# Convert to DataFrame
# ======================
input_dict = {
    "Age": age,
    "Sex": 1 if sex=='Male' else 0,
    "ChestPainType": chest,
    "RestingBP": resting_bp,
    "Cholesterol": chol,
    "FastingBS": 1 if fasting_bs == "Yes" else 0,
    "RestingECG": resting_ecg,
    "MaxHR": max_hr,
    "ExerciseAngina": 1 if exercise_angina=='Yes' else 0,
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope
}
num_cols=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
cat_cols=['ChestPainType','RestingECG','ST_Slope']
binary_cols=['Sex','FastingBS','ExerciseAngina']

input_df = pd.DataFrame([input_dict])
input_df[num_cols]=input_df[num_cols].astype(float)
input_df[cat_cols]=input_df[cat_cols].astype(str)
input_df[binary_cols]=input_df[binary_cols].astype(int)
# ======================
# Predict Button
# ======================
if st.button("Predict"):
    try:
        # Transform input
        processed = preprocessor.transform(input_df)

        # Predict
        pred_prob = model.predict(processed)
        pred_class = np.argmax(pred_prob, axis=1)[0]

        # ======================
        # Output
        # ======================
        st.subheader("Hasil Prediksi:")
        if pred_class == 1:
            st.error(f"🚨 **High Risk of Heart Disease**\nProb: {pred_prob[0][1]*100:.3f}%")
        else:
            st.success(f"✅ **Low Risk of Heart Disease**\nProb: {pred_prob[0][0]*100:.3f}%")

        st.write("### Probability:")
        st.json({
            "No Disease (0)": float(pred_prob[0][0]),
            "Disease (1)": float(pred_prob[0][1])
        })

    except Exception as e:
        st.error(f"Error: {e}")
