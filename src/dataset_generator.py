import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime, timedelta

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def generate_synthetic_data(num_patients, seed=42):
    np.random.seed(seed)
    
    # Generate dates across the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = [start_date + timedelta(days=int(np.random.randint(0, 730))) for _ in range(num_patients)]
    
    # Features
    patient_id = [f"PT_{str(i).zfill(5)}" for i in range(1, num_patients + 1)]
    age = np.random.randint(18, 90, num_patients)
    gender = np.random.choice(['Male', 'Female', 'Unknown'], num_patients, p=[0.48, 0.48, 0.04])
    
    # Simulate somewhat realistic biometrics
    bmi = np.random.normal(28, 6, num_patients)
    blood_glucose = np.random.normal(120, 40, num_patients) 
    hba1c = np.random.normal(6.5, 1.5, num_patients)
    
    # Diagnosis based on biometrics with some noise
    prob_diabetes = 1 / (1 + np.exp(-(0.05 * (bmi - 25) + 0.1 * (blood_glucose - 100) + 0.5 * (hba1c - 6.0))))
    diagnosis = np.where(prob_diabetes > 0.5, 'Type 2 Diabetes', np.where(prob_diabetes > 0.3, 'Pre-diabetes', 'Healthy'))
    
    # Introduce some missing values
    mask_hba1c = np.random.rand(num_patients) < 0.05
    hba1c[mask_hba1c] = np.nan
    mask_bg = np.random.rand(num_patients) < 0.03
    blood_glucose[mask_bg] = np.nan
    
    # Device readings (e.g. CGM average over a week)
    device_reading = blood_glucose + np.random.normal(0, 10, num_patients)
    
    df = pd.DataFrame({
        'patient_id': patient_id,
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'blood_glucose': blood_glucose,
        'hba1c': hba1c,
        'diagnosis': diagnosis,
        'visit_date': dates,
        'device_reading': device_reading
    })
    
    # Add some outliers
    outlier_idx = np.random.choice(num_patients, size=5, replace=False)
    df.loc[outlier_idx, 'bmi'] = 999.0 # Impossible value
    df.loc[outlier_idx, 'age'] = -5    # Impossible value
    
    return df

def main():
    config = load_config()
    num_patients = config['generator']['num_patients']
    seed = config['generator']['seed']
    output_path = config['data']['raw']
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Generating {num_patients} patient records...")
    df = generate_synthetic_data(num_patients, seed)
    
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()
