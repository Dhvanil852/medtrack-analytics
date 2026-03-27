import pandas as pd
from src.utils import setup_logger, load_config
import os

logger = setup_logger("DataLoader")

def load_raw_data(config_path="config.yaml"):
    """Loads raw dataset from the path specified in config."""
    config = load_config(config_path)
    path = config["data"]["raw"]
    if not os.path.exists(path):
        logger.error(f"Raw data not found at {path}. Please run dataset_generator.py first.")
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data with shape {df.shape}")
    return df

def save_processed_data(df, config_path="config.yaml"):
    """Saves processed dataframe to the path specified in config."""
    config = load_config(config_path)
    path = config["data"]["processed"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved processed data with shape {df.shape} to {path}")

def load_processed_data(config_path="config.yaml"):
    """Loads processed dataset from the path specified in config."""
    config = load_config(config_path)
    path = config["data"]["processed"]
    if not os.path.exists(path):
        logger.error(f"Processed data not found at {path}.")
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded processed data with shape {df.shape}")
    return df
