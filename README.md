# Heart Disease Prediction App

A machine learning web application that predicts the likelihood of heart disease based on patient medical data using a K-Nearest Neighbors (KNN) classifier.

## Features

- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Medical Parameter Input**: Accepts 11 key health indicators including age, blood pressure, cholesterol, and ECG results
- **Real-time Prediction**: Instant binary classification (heart disease detected or not)
- **Data Preprocessing**: Automatic feature scaling and categorical encoding

## Input Parameters

The model considers the following medical parameters:

- Age
- Sex (Male/Female)
- Chest Pain Type (ATA, NAP, TA, ASY)
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate
- Exercise-induced Angina
- ST Depression
- Slope of Peak Exercise ST Segment

## Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
   - Install dependencies
4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your default web browser. Fill in the medical parameters and click "Predict" to get the heart disease prediction.

## Model Details

- Algorithm: K-Nearest Neighbors (KNN) Classifier
- Preprocessing: Standard scaling for numerical features, one-hot encoding for categorical variables
- Model Files:
  - `KNN_heart_disease_model.pkl`: Trained KNN model
  - `scalar.pkl`: Feature scaler
  - `columns.pkl`: Expected feature columns

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Disclaimer

This application is for educational and demonstration purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.
