import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import duckdb
from src.data_loader import load_processed_data, load_raw_data
from dashboard.plots import plot_demographics, plot_correlation, plot_psi_trend
from src.preprocess import process_pipeline
from src.model_monitor import train_baseline_model, simulate_drift_over_time
from src.dataset_generator import generate_synthetic_data
import os

st.set_page_config(page_title="MedTrack Analytics", layout="wide")

st.title("MedTrack Analytics Dashboard")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select View", ["Overview & EDA", "Model Monitoring", "DuckDB RWE Tool"])

@st.cache_data
def get_data():
    raw_path = "data/raw/patients_raw.csv"
    if not os.path.exists(raw_path):
        st.info("Raw data missing. Generating synthetic dataset now...")
        df_raw = generate_synthetic_data(1000)
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df_raw.to_csv(raw_path, index=False)
    else:
        df_raw = pd.read_csv(raw_path)
    
    df_clean = process_pipeline(df_raw)
    return df_raw, df_clean

raw_df, clean_df = get_data()

if menu == "Overview & EDA":
    st.header("Patient Demographics & Exploratory Data Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{len(clean_df)}")
    col2.metric("Missing Values", f"{raw_df.isnull().sum().sum()}")
    diabetes_pct = (clean_df['diagnosis'] == 'Type 2 Diabetes').mean() * 100
    col3.metric("Diabetes Prevalence", f"{diabetes_pct:.1f}%")
    avg_hba1c = clean_df['hba1c'].mean()
    col4.metric("Avg HbA1c", f"{avg_hba1c:.2f}")

    st.subheader("Distributions")
    plot_demographics(clean_df)

    st.subheader("Correlations")
    plot_correlation(clean_df)

elif menu == "Model Monitoring":
    st.header("ML Model Drift Monitoring")
    st.write("We have simulated a Baseline Model and chunked our dataset by time to observe Data Drift and Accuracy decay.")
    
    with st.spinner("Training baseline model & simulating drift..."):
        model, features, (base_acc, base_f1) = train_baseline_model(clean_df)
        drift_results = simulate_drift_over_time(clean_df, model, features)
        
    col1, col2 = st.columns(2)
    col1.metric("Baseline Accuracy", f"{base_acc:.2%}")
    col2.metric("Baseline F1", f"{base_f1:.2f}")
    
    st.dataframe(drift_results, use_container_width=True)
    
    st.subheader("Accuracy and PSI Trend")
    plot_psi_trend(drift_results)

elif menu == "DuckDB RWE Tool":
    st.header("Real-World Evidence SQL Runner")
    st.write("Run local DuckDB queries on our dataset to answer feasibility metrics.")
    
    # We load standard clean_df into duckdb context automatically
    conn = duckdb.connect(database=':memory:')
    conn.register('patients', clean_df)
    
    query = st.text_area("SQL Query", value="SELECT age_group, COUNT(*) as patient_count, AVG(hba1c) as avg_hba1c\nFROM patients\nGROUP BY age_group\nORDER BY patient_count DESC;")
    
    if st.button("Run Query"):
        try:
            result = conn.execute(query).fetchdf()
            st.dataframe(result, use_container_width=True)
            st.success("Query successful.")
        except Exception as e:
            st.error(f"Error executing query: {e}")
