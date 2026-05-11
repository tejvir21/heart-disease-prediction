import joblib
import streamlit as st
import pandas as pd

model = joblib.load("KNN_heart_disease_model.pkl")
scalar = joblib.load("scalar.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction by Tejvir Chauhan")
st.markdown(
    "This app predicts the heart disease using the KNN model."
)

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ['ATA', "NAP", 'TA', 'ASY'])
trestbps = st.number_input("Resting Blood Pressure (mm/hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mm/dL)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", ['Normal', 'ST', 'LVH'])
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ['Y', 'N'])
oldpeak = st.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ['Up', 'Flat', 'Down'])

# Extra features
# ca = st.slider("Major Vessels Colored by Flourosopy", 0, 3, 0)
# thal = st.slider("Thalium Stress Test", 0, 3, 0)

if st.button("Predict"):

    raw_data = {}

    # Initialize all columns with 0
    for col in expected_columns:
        raw_data[col] = 0

    # Update the raw data with the user inputs
    raw_data["Age"] = age
    raw_data["RestingBP"] = trestbps
    raw_data["Cholesterol"] = chol
    raw_data["FastingBS"] = fbs
    raw_data["MaxHR"] = max_hr
    raw_data["Oldpeak"] = oldpeak
    raw_data["Sex_" + sex] = 1
    raw_data["ChestPainType_" + cp] = 1
    raw_data["RestingECG_" + restecg] = 1
    raw_data["ExerciseAngina_" + exang] = 1
    raw_data["ST_Slope_" + slope] = 1

    input_df = pd.DataFrame([raw_data])
    
    scaled_input = scalar.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    if prediction == 0:
        st.success("The person is not affected by heart disease.")
    else:
        st.error("The person is affected by heart disease.")
    