# pipelines/data_processing.py
import pandas as pd
import numpy as np
# import argparse # No longer needed
import os
from typing import List, Set
import warnings
from models.utils.features import BaseFeatureConfig, get_feature_config

def run_data_processing(raw_data_path: str, output_path: str, include_odds: bool):
    """
    Loads raw data, selects features, performs basic cleaning, and saves output.

    Args:
        raw_data_path: Path to the input raw parquet file.
        output_path: Path to save the processed parquet file.
        include_odds: Boolean flag whether to include odds features.
    """
    print(f"--- Starting Data Processing ---")
    print(f"Raw data path: {raw_data_path}")
    print(f"Output path: {output_path}")
    print(f"Include odds: {include_odds}")

    # --- 1. Load Configuration ---
    try:
        feature_cfg: BaseFeatureConfig = get_feature_config(include_odds=include_odds)
        print("Feature configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading feature configuration: {e}")
        raise

    # --- 2. Load Raw Data ---
    print(f"Loading raw data from: {raw_data_path}")
    try:
        raw_df = pd.read_parquet(raw_data_path, engine='pyarrow')
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        raise

    print(f"Raw data shape: {raw_df.shape}")
    assert not raw_df.empty, "Raw DataFrame is empty."
    assert isinstance(raw_df, pd.DataFrame), "Loaded object is not a pandas DataFrame."

    # --- 3. Select Required Columns ---
    # Use feature_cfg methods if they exist, otherwise adapt
    try:
        required_columns: Set[str] = feature_cfg.get_required_columns(include_odds=include_odds)
    except AttributeError:
         print("Warning: feature_cfg does not have 'get_required_columns'. Adapt logic if needed.")
         # Example fallback: define required columns manually or based on another source
         # required_columns = {'col1', 'col2', ...} # Replace with actual logic
         raise NotImplementedError("Need to define how required columns are determined.")


    actual_columns: Set[str] = set(raw_df.columns)
    print(f"Selecting {len(required_columns)} required columns...")

    missing_cols = required_columns - actual_columns
    assert not missing_cols, f"Raw data missing required config columns: {missing_cols}"

    # Keep only required columns (which should ideally include targets + IDs if defined in config)
    df = raw_df[list(required_columns)].copy()
    print(f"Shape after selecting columns: {df.shape}")

    # --- 4. Basic Cleaning & Preprocessing ---
    print("Performing basic cleaning...")

    # Convert Date column - adapt based on actual feature_cfg structure
    date_col = getattr(feature_cfg, 'date_col', None) # Safely get date_col if defined
    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            print(f"Converted '{date_col}' to datetime.")
        except Exception as e:
            print(f"Warning: Could not convert date column '{date_col}': {e}")
    elif date_col:
         warnings.warn(f"Date column '{date_col}' defined in config but not found in DataFrame.")
    else:
         warnings.warn(f"No 'date_col' defined in feature config, skipping date conversion.")


    # Handle NaNs in numerical features
    try:
        numerical_features: List[str] = feature_cfg.get_feature_columns(include_odds=include_odds)
    except AttributeError:
        print("Warning: feature_cfg does not have 'get_feature_columns'. Adapt logic if needed.")
        raise NotImplementedError("Need to define how numerical feature columns are determined.")

    # Ensure only truly numeric columns are selected (sometimes object cols might sneak in)
    numerical_features_in_df = df.columns.intersection(numerical_features)
    numerical_features_to_process = df[numerical_features_in_df].select_dtypes(include=np.number).columns.tolist()

    print(f"Checking NaNs in {len(numerical_features_to_process)} numerical features...")
    for col in numerical_features_to_process:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled NaNs in '{col}' with median ({median_val:.4f}).")

    # Assert no NaNs remain in numerical features selected for the model
    # Re-select numeric types from the final feature list present in the dataframe
    final_numeric_feature_cols = df[numerical_features_in_df].select_dtypes(include=np.number).columns.tolist()
    nan_check = df[final_numeric_feature_cols].isnull().sum()
    assert nan_check.sum() == 0, f"NaNs still present in numerical features after cleaning: \n{nan_check[nan_check > 0]}"

    # Handle NaNs in categorical features (if any defined and present)
    categorical_features = getattr(feature_cfg, 'core_categorical_features', []) # Safely get list
    for col in categorical_features:
         if col in df.columns and df[col].isnull().any():
             # Check if the column still exists after selection
             df[col] = df[col].fillna("Unknown")
             print(f"Filled NaNs in categorical feature '{col}' with 'Unknown'.")

    # --- 5. Final Assertions ---
    print("Performing final assertions...")
    # Adjust assertion if required_columns definition changed
    assert set(df.columns).issubset(required_columns), "Final columns contain columns not in the required set."
    assert required_columns.issubset(set(df.columns)), f"Final columns missing required columns: {required_columns - set(df.columns)}"
    # Add more specific checks if needed (e.g., value ranges)

    # --- 6. Save Processed Data ---
    print(f"Saving processed data to: {output_path}")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir: # Avoid error if output path is just a filename in the current dir
             os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Processed data saved successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error saving processed data to {output_path}: {e}")
        raise

    print(f"--- Data Processing Complete ---")


if __name__ == "__main__":
    # --- Define Parameters Directly ---
    # Example paths - MODIFY THESE to your actual paths
    RAW_DATA_INPUT_PATH = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/data/parquets/final_data_with_elo.parquet'
    PROCESSED_OUTPUT_PATH_NO_ODDS = 'models/data/outputs/processed_without_odds.parquet' # Example Output
    PROCESSED_OUTPUT_PATH_WITH_ODDS = 'models/data/outputs/processed_with_odds.parquet'   # Example Output

    # --- Run Processing (Example: Without Odds) ---
    print("\n>>> Running Data Processing WITHOUT Odds <<<")
    run_data_processing(
        raw_data_path=RAW_DATA_INPUT_PATH,
        output_path=PROCESSED_OUTPUT_PATH_NO_ODDS,
        include_odds=False
    )

    # --- Run Processing (Example: With Odds) ---
    print("\n>>> Running Data Processing WITH Odds <<<")
    run_data_processing(
        raw_data_path=RAW_DATA_INPUT_PATH,
        output_path=PROCESSED_OUTPUT_PATH_WITH_ODDS,
        include_odds=True
    )