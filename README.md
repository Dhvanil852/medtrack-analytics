# MedTrack Analytics

MedTrack Analytics is a comprehensive end-to-end data science project demonstrating common workflows in a healthcare or life sciences company (e.g., Abbott GDSA team).

## Project Features
- **Data Generation**: Simulated synthetic patient vitals data.
- **Data Cleaning**: Handled missing values, outliers, and type conversions.
- **Exploratory Data Analysis (EDA)**: Interactive distribution and correlation plots using Plotly.
- **Machine Learning**: Predictive model for patient diagnosis, including a demonstration of data drift (PSI) monitoring across time windows.
- **Real-World Evidence (RWE)**: Ad-hoc DuckDB-based SQL queries for feasibility insights.
- **Dashboard**: A Streamlit interactive tool showcasing the above.

## Setup Instructions

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the Dataset**:
   ```bash
   python -m src.dataset_generator
   ```

3. **Run Tests**:
   ```bash
   pytest tests/
   ```

4. **Launch the Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

## Structure
- `data/` - Raw and processed datasets
- `src/` - Core Python modules for data loading, preprocessing, and model monitoring
- `dashboard/` - Streamlit app scripts
- `notebooks/` - Demonstrations of usage via Jupyter Notebooks
- `tests/` - Unit tests for data cleaning logic
