import streamlit as st
import numpy as np
import joblib

# Load Model
model = joblib.load("property_model.pkl")

# Title
st.title("🏠 SMART PROPERTY PREDICTION SYSTEM")

st.write("Enter Property Details")

# Inputs
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
sqft_living = st.number_input("Sqft Living", 500, 10000)
sqft_lot = st.number_input("Sqft Lot", 500, 50000)
floors = st.number_input("Floors", 1, 5)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.slider("View", 0, 4)
condition = st.slider("Condition", 1, 5)
grade = st.slider("Grade", 1, 13)
sqft_basement = st.number_input("Sqft Basement", 0, 5000)
lat = st.number_input("Latitude", format="%.6f")
long = st.number_input("Longitude", format="%.6f")

# Predict
if st.button("Predict Property Price"):

    input_data = np.array([[bedrooms,
                            bathrooms,
                            sqft_living,
                            sqft_lot,
                            floors,
                            waterfront,
                            view,
                            condition,
                            grade,
                            sqft_basement,
                            lat,
                            long]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Property Price: ₹ {round(prediction[0], 2)}")