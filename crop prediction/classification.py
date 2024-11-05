import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Custom CSS for enhancing UI
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
        }
        .stSlider {
            background-color: #e0f7fa;
            border-radius: 5px;
        }
        .stButton {
            background-color: #2196F3;
            color: white;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton:hover {
            background-color: #1976D2;
        }
        .main-title {
            text-align: center;
            color: #673AB7;
            font-size: 36px;
            font-family: 'Helvetica', sans-serif;
            font-weight: bold;
        }
        .prediction-result {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .input-section {
            padding: 20px;
            border-radius: 10px;
            background-color: #f3f4f6;
        }
    </style>
""", unsafe_allow_html=True)

# Function to load the data
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine.target_names

df, target_names = load_data()

# Train the model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['target'])

# Main page title
st.markdown('<p class="main-title">Wine Species Prediction</p>', unsafe_allow_html=True)

# Layout with columns
col1, col2 = st.columns([2, 3])

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Wine_glass_icon.svg/1200px-Wine_glass_icon.svg.png", width=200)
with col2:
    st.markdown("""
        <h2 style="text-align: left;">Predict the wine species based on various features.</h2>
        <p>Use the sliders to input values for the features of the wine and get a prediction of its species.</p>
    """, unsafe_allow_html=True)

# Sidebar with input features
st.sidebar.title("Input Features")
feature_sliders = {}
for feature in df.columns[:-1]:  # Exclude the target column
    min_val, max_val = float(df[feature].min()), float(df[feature].max())
    feature_sliders[feature] = st.sidebar.slider(feature, min_val, max_val, key=feature)

input_data = [list(feature_sliders.values())]

# Prediction
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

# Display prediction result
st.markdown(f'<p class="prediction-result">The predicted wine species is: {predicted_species}</p>', unsafe_allow_html=True)

# Additional info section
st.markdown("""
    <div class="input-section">
        <h3 style="text-align: center;">How it works</h3>
        <p>Select the values for each feature in the sidebar. Based on these values, the Random Forest model will predict the species of wine (from 3 possible types: Class 0, Class 1, or Class 2).</p>
        <p>Each feature represents a different property of the wine, such as alcohol content, color intensity, and more.</p>
    </div>
""", unsafe_allow_html=True)

# Adding a custom button
if st.button("Re-run Prediction", key="run_prediction"):
    st.experimental_rerun()
