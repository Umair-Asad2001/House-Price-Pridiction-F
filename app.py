
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ============================================
# LOAD MODEL, SCALER, AND COLUMNS
# ============================================

with open("housing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    training_columns = pickle.load(f)

# ============================================
# STREAMLIT UI
# ============================================

st.title("🏠 House Price Prediction App")

st.write("Enter house details below:")

# ============================================
# USER INPUTS
# ============================================

area = st.number_input("Area (sq ft)", min_value=0)

bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10) # Added missing input

bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10)

stories = st.number_input("Stories", min_value=1, max_value=10)

parking = st.number_input("Parking Spaces", min_value=0, max_value=10)

mainroad = st.selectbox("Main Road", ["yes", "no"])

guestroom = st.selectbox("Guest Room", ["yes", "no"])

basement = st.selectbox("Basement", ["yes", "no"])

hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])

airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])

prefarea = st.selectbox("Preferred Area", ["yes", "no"])

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# ============================================
# PREDICTION
# ============================================

if st.button("Predict Price"):
    # Create a DataFrame from user inputs
    input_df = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'parking': [parking],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })

    # Apply one-hot encoding to categorical columns
    # Ensure drop_first=True matches training
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)

    # Reindex to ensure all columns from training are present, filling missing with 0
    # This handles cases where a category might not be present in the user input
    final_input = input_df_encoded.reindex(columns=training_columns, fill_value=0)

    # Preprocess (scale) input using the loaded scaler
    processed_data = scaler.transform(final_input)

    # Predict
    prediction = model.predict(processed_data)

    st.success(f"🏡 Predicted House Price: ${prediction[0]:,.2f}")
