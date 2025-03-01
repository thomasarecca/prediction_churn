import streamlit as st
import pandas as pd

st.title("Churn Prediction Dashboard")

st.write("Upload a CSV file for prediction")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())
