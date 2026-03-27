import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from src.utils import setup_logger, load_config

logger = setup_logger("ModelMonitor")

def calculate_psi(expected, actual, bins=10):
    """Calculate the Population Stability Index (PSI) between two distributions."""
    
    def calculate_distribution(series, bins_array):
        # Calculate percentages of items in each bin
        counts, _ = np.histogram(series, bins=bins_array)
        percents = counts / len(series)
        # Handle zero division
        percents = np.where(percents == 0, 0.0001, percents)
        return percents

    df_expected = pd.Series(expected).dropna()
    df_actual = pd.Series(actual).dropna()

    if len(df_expected) == 0 or len(df_actual) == 0:
        return 0.0

    # Determine common bins
    min_val = min(df_expected.min(), df_actual.min())
    max_val = max(df_expected.max(), df_actual.max())
    bins_array = np.linspace(min_val, max_val, bins + 1)

    expected_dist = calculate_distribution(df_expected, bins_array)
    actual_dist = calculate_distribution(df_actual, bins_array)

    psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
    return psi

def train_baseline_model(df, target_col='diagnosis'):
    """Trains a quick baseline logistic regression model."""
    config = load_config()
    random_state = config['model']['random_state']
    
    # Simple feature selection, filtering numerics
    features = ['age', 'bmi', 'blood_glucose', 'hba1c']
    X = df[features].fillna(df[features].median()) 
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['model']['test_size'], random_state=random_state)
    
    # Mapping for multiclass classification logic
    # Simplified approach: LR
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    
    logger.info(f"Baseline Random Forest trained. Acc: {acc:.4f}, F1: {f1:.4f}")
    
    return model, features, (acc, f1)

def simulate_drift_over_time(df, model, features, time_col='visit_date', target_col='diagnosis'):
    """Splits data by time windows to calculate metrics and PSI to monitor drift."""
    df_sorted = df.copy()
    df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
    df_sorted = df_sorted.sort_values(by=time_col)
    
    # Split into 4 time chunks (e.g. quarters)
    chunks = np.array_split(df_sorted, 4)
    
    # Track metrics
    results = []
    
    # Base distribution (from first chunk)
    base_df = chunks[0]
    
    for i, chunk in enumerate(chunks):
        X = chunk[features].fillna(chunk[features].median())
        y_true = chunk[target_col]
        
        preds = model.predict(X)
        acc = accuracy_score(y_true, preds)
        
        # Calculate PSI for a key feature: e.g. blood_glucose
        psi = calculate_psi(base_df['blood_glucose'], chunk['blood_glucose'])
        
        start_date = chunk[time_col].min().strftime("%Y-%m-%d")
        end_date = chunk[time_col].max().strftime("%Y-%m-%d")
        
        results.append({
            'window': f"W{i+1}: {start_date} to {end_date}",
            'accuracy': acc,
            'psi_blood_glucose': psi
        })
        
        logger.info(f"Window {i+1} | Acc: {acc:.3f} | PSI (blood_glucose): {psi:.4f}")
        
    return pd.DataFrame(results)
