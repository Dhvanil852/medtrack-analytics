import pandas as pd
import numpy as np
from src.utils import setup_logger

logger = setup_logger("Preprocess")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing values using median for numericals and mode for categoricals."""
    df_clean = df.copy()
    
    # Identify column types
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            logger.info(f"Imputed missing in {col} with median: {median_val:.2f}")
            
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            logger.info(f"Imputed missing in {col} with mode: {mode_val}")
            
    return df_clean

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clips impossible clinical values to reasonable bounds."""
    df_clean = df.copy()
    
    # Clip ages < 0
    if 'age' in df_clean.columns:
        df_clean.loc[df_clean['age'] < 0, 'age'] = 0
        df_clean.loc[df_clean['age'] > 120, 'age'] = 120
        logger.info("Clipped age outliers limits [0, 120]")

    # Clip impossible BMIs
    if 'bmi' in df_clean.columns:
        df_clean.loc[df_clean['bmi'] > 100, 'bmi'] = 100
        df_clean.loc[df_clean['bmi'] < 10, 'bmi'] = 10
        logger.info("Clipped BMI outliers limits [10, 100]")
        
    return df_clean

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Adds BMI category and Age Group features."""
    df_eng = df.copy()
    
    if 'bmi' in df_eng.columns:
        conditions = [
            (df_eng['bmi'] < 18.5),
            (df_eng['bmi'] >= 18.5) & (df_eng['bmi'] < 25),
            (df_eng['bmi'] >= 25) & (df_eng['bmi'] < 30),
            (df_eng['bmi'] >= 30)
        ]
        choices = ['Underweight', 'Normal', 'Overweight', 'Obese']
        df_eng['bmi_category'] = np.select(conditions, choices, default='Unknown')
        logger.info("Engineered feature: bmi_category")
        
    if 'age' in df_eng.columns:
        bins = [0, 30, 50, 70, 120]
        labels = ['<30', '30-50', '50-70', '70+']
        df_eng['age_group'] = pd.cut(df_eng['age'], bins=bins, labels=labels, right=False)
        logger.info("Engineered feature: age_group")
        
    # Ensure correct datetime conversion
    if 'visit_date' in df_eng.columns:
        df_eng['visit_date'] = pd.to_datetime(df_eng['visit_date']).dt.date
        
    return df_eng

def process_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full data cleaning pipeline."""
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = feature_engineering(df)
    return df
