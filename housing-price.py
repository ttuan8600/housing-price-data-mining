import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Housing Price Predictor", layout="centered")

st.title("🏡 Housing Price Predictor & Outlier Detector")

# File upload
uploaded_file = st.file_uploader("Upload CSV file with house features", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data:", df.head())

    if 'price' in df.columns:
        # Load your trained model
        model = joblib.load("regression_model.pkl")  # Replace with your actual model path

        # Make predictions
        features = df.drop(columns=['price'])
        predictions = model.predict(features)
        df['predicted_price'] = predictions
        df['error'] = np.abs(df['price'] - df['predicted_price'])
        threshold = df['error'].quantile(0.95)
        df['is_outlier'] = df['error'] > threshold

        st.subheader("📈 Results")
        st.dataframe(df[['price', 'predicted_price', 'error', 'is_outlier']])

        st.subheader("🔍 Outliers")
        st.dataframe(df[df['is_outlier']])
    else:
        st.warning("⚠️ The uploaded file must contain a 'price' column.")
else:
    st.info("👈 Upload a CSV to get started.")
