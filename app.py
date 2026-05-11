import joblib
import streamlit as st
import pandas as pd

# Load all models
models = [{
    "Selected": True,
    "Name": "KNN Model",
    "Prediction": 0,
    "Model": joblib.load("KNN_heart_disease_prediction_model.pkl"),
    "Fixed": True
},
{
    "Selected": False,
    "Name": "Logistic Regression Model",
    "Prediction": 0,
    "Model": joblib.load("Logistic_Regression_heart_disease_prediction_model.pkl"),
    "Fixed": False
},
{
    "Selected": False,
    "Name": "Decision Tree Model",
    "Prediction": 0,
    "Model": joblib.load("Decision_Tree_heart_disease_prediction_model.pkl"),
        "Fixed": False

},
{
    "Selected": False,
    "Name": "Naive Bayes Model",
    "Prediction": 0,
    "Model": joblib.load("Naive_Bayes_heart_disease_prediction_model.pkl"),
        "Fixed": False

},
{
    "Selected": False,
    "Name": "Support Vector Machine Model",
    "Prediction": 0,
    "Model": joblib.load("SVM_heart_disease_prediction_model.pkl"),
        "Fixed": False

}]

scalar = joblib.load("scalar.pkl")
expected_columns = joblib.load("columns.pkl")

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction App", page_icon=":heart:", layout="wide")

st.title("Heart Disease Prediction App")
st.header("Check Your Heart Health Risk")

st.markdown("""
<meta name="description" content="AI-powered heart disease prediction app using machine learning.">
<meta name="keywords" content="heart disease prediction, healthcare AI, machine learning">
""", unsafe_allow_html=True)

st.markdown(
    "This app predicts the heart disease risk using ML models."
)

st.markdown("---")
st.markdown("## Available Models")

# Create a checkbox for each model
for model in models:
    model["Selected"] = st.checkbox(f"{model['Name']}", value=model["Selected"], disabled=model['Fixed'])

st.markdown("---")
st.markdown("## Optional Parameters")

show_suggessions = st.checkbox("Show Suggestions", value=True)

st.markdown("---")
st.markdown("## Input Parameters")

# Age
age = st.slider("Age", 18, 100, 40)

# Suggestions based on age group
if show_suggessions:
    if age < 30:
        st.success("""
        🟢 **Young Age Group**
        
        Lower risk of heart disease, but maintaining healthy habits is important.
        Suggestion: Stay active, eat healthy, and avoid smoking.
        """)

    elif 30 <= age < 50:
        st.info("""
        🔵 **Middle Age Group**
        
        Heart health should be monitored regularly as risk factors may begin to appear.
        Suggestion: Check blood pressure, cholesterol, and maintain fitness.
        """)

    elif 50 <= age < 70:
        st.warning("""
        🟠 **Higher Risk Age Group**
        
        Risk of heart-related conditions may increase with age.
        Suggestion: Regular medical checkups are recommended.
        """)

    else:
        st.error("""
        🔴 **Senior Age Group**
        
        Heart disease risk tends to be higher in older adults.
        Suggestion: Frequent health monitoring and doctor consultations are advised.
        """)

# Gender 
sex = st.selectbox("Sex", ["M", "F"])

# Suggestions / information
if show_suggessions:
    if sex == "M":
        st.info("""
        **Male (M)** 👨  
        Studies show men may have a higher risk of developing heart disease at an earlier age.  
        Suggestion: Maintain regular exercise, a healthy diet, and routine heart checkups.
        """)

    elif sex == "F":
        st.info("""
        **Female (F)** 👩  
        Women can also develop heart disease, sometimes with different symptoms than men.  
        Suggestion: Monitor symptoms carefully and maintain regular health screenings.
        """)

# Chest Pain
cp = st.selectbox("Chest Pain Type", ['ATA', "NAP", 'TA', 'ASY'])

# Suggestions based on selection
if show_suggessions:
    if cp == "ATA":
        st.info("""
        **ATA (Atypical Angina)**  
        ⚠️ Chest discomfort that doesn't match classic angina symptoms.  
        Suggestion: Monitor symptoms and consult a doctor if pain persists.
        """)

    elif cp == "NAP":
        st.success("""
        **NAP (Non-Anginal Pain)**  
        ✅ Usually not related to heart disease.  
        Suggestion: Could be caused by stress, muscle strain, or digestion issues.
        """)

    elif cp == "TA":
        st.warning("""
        **TA (Typical Angina)**  
        ❤️ Common chest pain linked to reduced blood flow to the heart.  
        Suggestion: Seek medical advice and avoid strenuous activity.
        """)

    elif cp == "ASY":
        st.error("""
        **ASY (Asymptomatic)**  
        🚨 No chest pain symptoms, but underlying heart issues may still exist.  
        Suggestion: Regular health checkups are important.
        """)

# Resting Blood Pressure
trestbps = st.number_input("Resting Blood Pressure (mm/hg)", 80, 200, 120)

# Suggestions based on BP value
if show_suggessions:
    if trestbps < 90:
        st.warning("""
        ⚠️ **Low Blood Pressure**
        
        This may indicate hypotension.
        Suggestion: Consult a doctor if you experience dizziness or weakness.
        """)

    elif 90 <= trestbps <= 120:
        st.success("""
        ✅ **Normal Blood Pressure**
        
        Your resting blood pressure is within a healthy range.
        Suggestion: Maintain a balanced diet and regular exercise.
        """)

    elif 121 <= trestbps <= 139:
        st.info("""
        🔵 **Elevated Blood Pressure**
        
        This may indicate early signs of hypertension.
        Suggestion: Reduce salt intake and monitor regularly.
        """)

    elif 140 <= trestbps <= 180:
        st.warning("""
        🟠 **High Blood Pressure**
        
        Increased risk of heart-related issues.
        Suggestion: Seek medical advice and adopt healthier habits.
        """)

    else:
        st.error("""
        🚨 **Very High Blood Pressure**
        
        This may require immediate medical attention.
        Suggestion: Consult a healthcare professional as soon as possible.
        """)

# Cholesterol    
chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

# Suggestions based on cholesterol level
if show_suggessions:
    if chol < 125:
        st.warning("""
        ⚠️ **Low Cholesterol**
        
        Cholesterol is below the normal range.
        Suggestion: Consult a healthcare professional if this is unexpected.
        """)

    elif 125 <= chol < 200:
        st.success("""
        ✅ **Healthy Cholesterol Level**
        
        Your cholesterol level is within a desirable range.
        Suggestion: Continue maintaining a healthy diet and active lifestyle.
        """)

    elif 200 <= chol < 240:
        st.info("""
        🔵 **Borderline High Cholesterol**
        
        This may increase the risk of heart disease over time.
        Suggestion: Reduce saturated fats and monitor regularly.
        """)

    else:
        st.error("""
        🚨 **High Cholesterol Level**
        
        Higher cholesterol can significantly increase heart disease risk.
        Suggestion: Consider medical consultation and dietary changes.
        """)

# Fasting Blood Sugar
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["Below 120 mg/dL", "Above 120 mg/dL"])

# Suggestions
if show_suggessions:
    if fbs == 0 or fbs == "Below 120 mg/dL":
        st.success("""
        ✅ **Normal Fasting Blood Sugar**
        
        Your fasting blood sugar is below 120 mg/dL.
        Suggestion: Maintain a balanced diet and regular exercise.
        """)

    elif fbs == 1 or fbs == "Above 120 mg/dL":
        st.warning("""
        ⚠️ **High Fasting Blood Sugar**
        
        Your fasting blood sugar is above 120 mg/dL.
        This may indicate a higher risk of diabetes and heart-related issues.
        
        Suggestion: Monitor your sugar levels and consult a doctor if needed.
        """)

# Resting Electrocardiographic
restecg = st.selectbox("Resting Electrocardiographic Results", ['Normal', 'ST', 'LVH'])

# Suggestions based on selection
if show_suggessions:
    if restecg == "Normal":
        st.success("""
        ✅ **Normal ECG**
        
        The resting electrocardiogram appears normal.
        Suggestion: Continue maintaining a healthy lifestyle.
        """)

    elif restecg == "ST":
        st.warning("""
        ⚠️ **ST-T Wave Abnormality**
        
        This may indicate irregular heart activity or possible heart-related issues.
        Suggestion: Further medical evaluation may be recommended.
        """)

    elif restecg == "LVH":
        st.error("""
        🚨 **Left Ventricular Hypertrophy (LVH)**
        
        This may suggest enlargement of the heart's left ventricle.
        Suggestion: Consult a cardiologist for proper diagnosis.
        """)

# Maximum Heart Rate
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

# Suggestions based on heart rate
if show_suggessions:
    if max_hr < 100:
        st.warning("""
        ⚠️ **Low Maximum Heart Rate**
        
        This may indicate reduced physical activity tolerance or potential heart concerns.
        Suggestion: Consult a doctor if this value seems unusual.
        """)

    elif 100 <= max_hr < 150:
        st.info("""
        🔵 **Moderate Maximum Heart Rate**
        
        This falls within an average range depending on age and fitness level.
        Suggestion: Maintain regular exercise and monitor heart health.
        """)

    elif 150 <= max_hr < 190:
        st.success("""
        ✅ **Healthy Maximum Heart Rate**
        
        This may reflect good cardiovascular performance based on age and activity.
        Suggestion: Continue healthy lifestyle habits.
        """)

    else:
        st.warning("""
        ⚠️ **Very High Maximum Heart Rate**
        
        This could be normal during intense exercise but may require attention if unusual.
        Suggestion: Seek medical advice if accompanied by discomfort.
        """)

# Exercise Induced Angina
exang = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])

# Suggestions
if show_suggessions:
    if exang == "Yes":
        st.error("""
        🚨 **Yes – Exercise Induced Angina Detected**
        
        Chest pain during physical activity may indicate reduced blood flow to the heart.
        
        Suggestion: Avoid intense exercise and consult a healthcare professional.
        """)

    elif exang == "No":
        st.success("""
        ✅ **No Exercise Induced Angina**
        
        No chest pain experienced during exercise.
        
        Suggestion: Continue regular physical activity and maintain heart health.
        """)

# ST Depression
oldpeak = st.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 6.0, 1.0)

# Suggestions based on value
if show_suggessions:
    if oldpeak == 0:
        st.success("""
        ✅ **Normal Heart Response**
        
        Your heart appears to respond normally during physical activity.
        
        Suggestion: Keep maintaining a healthy lifestyle.
        """)

    elif 0 < oldpeak <= 1.5:
        st.info("""
        🔵 **Minor Heart Changes During Exercise**
        
        Small changes were noticed during physical activity, which are usually not serious.
        
        Suggestion: Stay active and monitor your heart health regularly.
        """)

    elif 1.5 < oldpeak <= 3.0:
        st.warning("""
        ⚠️ **Moderate Heart Stress**
        
        Your heart shows noticeable changes during exercise that may need attention.
        
        Suggestion: Consider consulting a doctor for further guidance.
        """)

    else:
        st.error("""
        🚨 **High Heart Stress**
        
        Your heart shows significant changes during physical activity.
        
        Suggestion: Please consult a healthcare professional as soon as possible.
        """)

# ST Segment Slope
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ['Up', 'Flat', 'Down'])

# Suggestions based on selection
if show_suggessions:
    if slope == "Up":
        st.success("""
        ✅ **Healthy Exercise Response**
        
        Your heart shows a normal response during physical activity.
        
        Suggestion: Keep maintaining a healthy lifestyle and regular exercise.
        """)

    elif slope == "Flat":
        st.warning("""
        ⚠️ **Some Heart Changes Detected**
        
        Your heart shows slight changes during exercise that may need attention.
        
        Suggestion: Regular health checkups are recommended.
        """)

    elif slope == "Down":
        st.error("""
        🚨 **Higher Heart Risk Indicator**
        
        Your heart shows unusual changes during exercise that could indicate potential issues.
        
        Suggestion: Consider consulting a doctor for further evaluation.
        """)

# Extra features (may not be used in the model)
# ca = st.slider("Major Vessels Colored by Flourosopy", 0, 3, 0)
# thal = st.slider("Thalium Stress Test", 0, 3, 0)

if st.button("Predict", type="primary"):

    # Create a dictionary to store the raw data
    raw_data = {}

    # Initialize all columns with 0
    for col in expected_columns:
        raw_data[col] = 0

    # Update the raw data with the user inputs
    raw_data["Age"] = age
    raw_data["RestingBP"] = trestbps
    raw_data["Cholesterol"] = chol
    raw_data["FastingBS"] = 1 if fbs == "Above 120 mg/dL" else 0 
    raw_data["MaxHR"] = max_hr
    raw_data["Oldpeak"] = oldpeak
    raw_data["Sex_" + sex] = 1
    raw_data["ChestPainType_" + cp] = 1
    raw_data["RestingECG_" + restecg] = 1
    raw_data["ExerciseAngina_" + 'Y'] = 1 if exang[0] == 'Y' else 0
    raw_data["ST_Slope_" + slope] = 1

    input_df = pd.DataFrame([raw_data])
    
    scaled_input = scalar.transform(input_df)

    st.markdown("---")
    st.markdown("## Predictions")

    for model in models:
        if model["Selected"]:
            model["Prediction"] = model["Model"].predict(scaled_input)[0]

            if model["Prediction"] == 0:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#e8f5e9;
                        padding:15px;
                        border-radius:12px;
                        border-left:6px solid #2e7d32;
                        margin-bottom:10px;">
                        <h4 style="color:#2e7d32; margin:0;">
                            ✅ {model['Name']} Prediction Result
                        </h4>
                        <p style="font-size:16px; color:#1b5e20; margin-top:8px;">
                            Great news! The person is <b>not affected by heart disease</b>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#ffebee;
                        padding:15px;
                        border-radius:12px;
                        border-left:6px solid #c62828;
                        margin-bottom:10px;">
                        <h4 style="color:#c62828; margin:0;">
                            ⚠️ {model['Name']} Prediction Result
                        </h4>
                        <p style="font-size:16px; color:#b71c1c; margin-top:8px;">
                            The person <b>may be affected by heart disease</b>. 
                            Please consult a healthcare professional for further evaluation.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")
st.markdown("## Disclaimer")
st.markdown("This app is for educational and demonstration purposes only. Always consult with a qualified healthcare professional for medical advice, diagnosis, or treatment.")

st.markdown("---")
st.markdown("## About")
st.markdown("Developed by [Tejvir Chauhan](https://github.com/tejvir21/heart-disease-prediction)")
