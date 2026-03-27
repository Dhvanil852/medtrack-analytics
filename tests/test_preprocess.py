import pandas as pd
import numpy as np
import pytest
from src.preprocess import handle_missing_values, handle_outliers, feature_engineering

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'age': [25, 150, -10, np.nan],
        'bmi': [22, 110, np.nan, 28],
        'gender': ['Male', np.nan, 'Female', 'Male'],
        'visit_date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01']
    })

def test_handle_missing_values(sample_data):
    df_clean = handle_missing_values(sample_data)
    
    assert df_clean['age'].isnull().sum() == 0
    assert df_clean['bmi'].isnull().sum() == 0
    assert df_clean['gender'].isnull().sum() == 0
    
    # Check if imputations form logic (mode for gender, median for age/bmi)
    assert df_clean.loc[1, 'gender'] == 'Male'  # Mode
    
def test_handle_outliers(sample_data):
    df_outliers = handle_outliers(sample_data)
    
    assert df_outliers['age'].min() >= 0
    assert df_outliers['age'].max() <= 120
    assert df_outliers['bmi'].max() <= 100
    assert df_outliers['bmi'].min() >= 10
    
def test_feature_engineering(sample_data):
    df_feat = feature_engineering(sample_data)
    assert 'bmi_category' in df_feat.columns
    assert 'age_group' in df_feat.columns
    assert df_feat.loc[0, 'bmi_category'] == 'Normal'
