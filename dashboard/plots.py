import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def plot_demographics(df):
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="age", color="diagnosis", nbins=20, title="Age Distribution by Diagnosis", barmode='overlay', opacity=0.7)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.box(df, x="diagnosis", y="bmi", color="diagnosis", title="BMI Spread by Diagnosis")
        st.plotly_chart(fig2, use_container_width=True)

def plot_correlation(df):
    num_cols = ["age", "bmi", "blood_glucose", "hba1c", "device_reading"]
    corr_matrix = df[num_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=num_cols, y=num_cols, 
                    title="Correlation Heatmap of Key Biomarkers",
                    color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)

def plot_psi_trend(drift_df):
    fig = go.Figure()

    # Accuracy line
    fig.add_trace(go.Scatter(x=drift_df['window'], y=drift_df['accuracy'],
                        mode='lines+markers',
                        name='Accuracy',
                        yaxis='y1',
                        line=dict(color='blue', width=3)))

    # PSI bars
    fig.add_trace(go.Bar(x=drift_df['window'], y=drift_df['psi_blood_glucose'],
                    name='PSI (Blood Glucose)',
                    yaxis='y2',
                    marker_color='red', opacity=0.6))

    fig.update_layout(
        title="Model Accuracy vs Distribution Drift (PSI) Over Time",
        yaxis=dict(title="Accuracy", side='left', showgrid=False),
        yaxis2=dict(title="PSI Score", side='right', overlaying='y', showgrid=False),
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig, use_container_width=True)
