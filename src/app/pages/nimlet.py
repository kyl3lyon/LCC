# --- Module Level Imports ---
from data_processing import (
    process_launch_stats_data,
    process_launch_forecast_data,
    apply_manual_corrections,
)
from dotenv import load_dotenv
import os
import yaml
import pandas as pd
import streamlit as st

# --- Load Environment Variables and Config ---
load_dotenv()  # Load environment variables from .env.
api_key = os.getenv("API_KEY")

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# --- Page Setup ---
def show():
    st.markdown("<h1 style='text-align: center;'>Nimlet-1</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Predicting LCC Violations with ML Classifiers and AI Agents</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Load default data initially
    default_launch_stats_path = f"{config['data']['processed_dir']}/{config['data']['clean_launch_stats_file']}"
    default_launch_forecast_path = f"{config['data']['processed_dir']}/{config['data']['clean_launch_forecast_file']}"
    clean_launch_stats_df = pd.read_csv(default_launch_stats_path)
    clean_launch_forecast_df = pd.read_csv(default_launch_forecast_path)

    # --- Data Upload ---
    st.markdown("<h4 style='text-align: left;'>Upload Data</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        uploaded_launch_stats = st.file_uploader("Upload LCC Data", type="csv", key="launch_stats_file")
        if uploaded_launch_stats is not None:  # If file is uploaded
            launch_stats_df = pd.read_csv(uploaded_launch_stats)
            clean_launch_stats_df = process_launch_stats_data(launch_stats_df)
            clean_launch_stats_df, clean_launch_forecast_df = apply_manual_corrections(clean_launch_stats_df, clean_launch_forecast_df)
            temp_stats_path = f"{config['data']['temp_dir']}/clean_launch_stats.csv"
            clean_launch_stats_df.to_csv(temp_stats_path, index=False)

    with col2:
        uploaded_launch_forecast = st.file_uploader("Upload Launch Data", type="csv", key="launch_forecast_file")
        if uploaded_launch_forecast is not None:  # If file is uploaded
            launch_forecast_df = pd.read_csv(uploaded_launch_forecast)
            clean_launch_forecast_df = process_launch_forecast_data(launch_forecast_df)
            clean_launch_stats_df, clean_launch_forecast_df = apply_manual_corrections(clean_launch_stats_df, clean_launch_forecast_df)
            temp_forecast_path = f"{config['data']['temp_dir']}/clean_launch_forecast.csv"
            clean_launch_forecast_df.to_csv(temp_forecast_path, index=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Preview Data ---
    st.markdown("<h4 style='text-align: left;'>Preview Data</h4>", unsafe_allow_html=True)

    # Create tabs for displaying data
    tab1, tab2 = st.tabs(["LCC Data", "Launch Data"])

    with tab1:
        st.write(clean_launch_stats_df.head())

    with tab2:
        st.write(clean_launch_forecast_df.head())