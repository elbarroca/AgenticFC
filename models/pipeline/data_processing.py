# pipelines/data_processing.py
import pandas as pd
import numpy as np
import joblib # To save scaler and PCA objects
from pathlib import Path
from typing import List, Set
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Fix the import path by adding models to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.utils.features import BaseFeatureConfig, get_feature_config

def run_pca_data_processing(
    raw_data_path: Path,
    output_dir: Path,
    include_odds: bool,
    pca_variance_threshold: float = 0.95, # Retain 95% of variance
    pca_fit_split_ratio: float = 0.8 # Use first 80% of data to fit PCA/Scaler
    ):
    """
    Loads raw data, selects features, cleans, scales, applies PCA,
    and saves processed data and transformers. Adheres to assertive principles.

    Args:
        raw_data_path: Path to the input raw parquet file.
        output_dir: Directory to save processed files and transformers.
        include_odds: Boolean flag whether to include odds features.
        pca_variance_threshold: Fraction of variance PCA should retain.
        pca_fit_split_ratio: Fraction of data (chronological start) to fit Scaler/PCA on.
    """
    odds_suffix = "with_odds" if include_odds else "without_odds"
    print(f"\n--- Starting PCA Data Processing ({odds_suffix}) ---")
    print(f"Raw data path: {raw_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"PCA Variance Threshold: {pca_variance_threshold}")
    assert 0.0 < pca_variance_threshold <= 1.0, "pca_variance_threshold must be between 0 and 1"
    assert 0.0 < pca_fit_split_ratio < 1.0, "pca_fit_split_ratio must be between 0 and 1"

    # --- Ensure output directories exist ---
    output_dir.mkdir(parents=True, exist_ok=True)
    transformers_dir = output_dir / 'transformers'
    transformers_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Config & Raw Data ---
    print("Loading config and raw data...")
    assert raw_data_path.exists(), f"Raw data file not found: {raw_data_path}"
    feature_cfg: BaseFeatureConfig = get_feature_config(include_odds=include_odds)
    raw_df = pd.read_parquet(raw_data_path, engine='pyarrow')
    assert not raw_df.empty, f"Raw DataFrame is empty: {raw_data_path}"
    assert isinstance(raw_df, pd.DataFrame), "Loaded object is not a pandas DataFrame."

    # Sort by date - CRUCIAL for fitting PCA/Scaler correctly
    date_col = getattr(feature_cfg, 'date_col', 'Date') # Default to 'Date'
    assert date_col in raw_df.columns, f"Date column '{date_col}' not found for chronological splitting."
    # Convert to datetime, coercing errors - then check for NaTs
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors='coerce')
    assert not raw_df[date_col].isnull().any(), f"Null values found in Date column '{date_col}' after conversion."
    raw_df = raw_df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Data sorted by '{date_col}'. Shape: {raw_df.shape}")

    # --- 2. Select Required Original Columns & Basic Clean ---
    print("Selecting columns and basic cleaning...")
    required_original_cols: Set[str] = feature_cfg.get_required_columns(include_odds=include_odds)
    missing_cols = required_original_cols - set(raw_df.columns)
    assert not missing_cols, f"Raw data missing required columns: {missing_cols}"
    df = raw_df[list(required_original_cols)].copy()

    # Handle NaNs in numerical features BEFORE splitting/scaling/PCA
    numerical_features: List[str] = feature_cfg.get_feature_columns(include_odds=include_odds)
    numerical_features_in_df = df.columns.intersection(numerical_features)
    numerical_features_to_process = df[numerical_features_in_df].select_dtypes(include=np.number).columns.tolist()
    assert numerical_features_to_process, "No numerical features found to process."

    print(f"Imputing NaNs in {len(numerical_features_to_process)} numerical features (using median)...")
    for col in numerical_features_to_process:
        if df[col].isnull().any():
            # Ensure column is numeric before calculating median
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' is not numeric despite selection."
            median_val = df[col].median()
            assert not pd.isna(median_val), f"Median calculation failed for column '{col}' (all NaNs?)."
            df[col] = df[col].fillna(median_val)
    # Final check after imputation
    nan_check_after_impute = df[numerical_features_to_process].isnull().sum()
    assert nan_check_after_impute.sum() == 0, f"NaNs remain after imputation: \n{nan_check_after_impute[nan_check_after_impute > 0]}"

    # --- 3. Split Data for Fitting Scaler/PCA (Chronological) ---
    n_rows = len(df)
    n_fit = int(n_rows * pca_fit_split_ratio)
    assert n_fit >= 100, f"Fit split ratio results in too few samples ({n_fit}) to fit Scaler/PCA reliably."
    print(f"Splitting data: {n_fit} rows for fitting Scaler/PCA, {n_rows - n_fit} for transforming.")

    df_fit = df.iloc[:n_fit]
    df_transform = df.iloc[n_fit:]
    assert not df_fit.empty, "df_fit partition is empty."
    assert not df_transform.empty, "df_transform partition is empty."

    X_fit_orig = df_fit[numerical_features_to_process]
    X_transform_orig = df_transform[numerical_features_to_process]

    # --- 4. Scale Features ---
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit_orig) # Fit only on df_fit
    print("Transforming data with scaler...")
    X_transform_scaled = scaler.transform(X_transform_orig) # Transform the rest

    # --- 5. Apply PCA ---
    print(f"Fitting PCA to retain {pca_variance_threshold*100:.1f}% variance...")
    pca = PCA(n_components=pca_variance_threshold, random_state=42)
    # Ensure no NaNs in scaled data before fitting PCA
    assert not np.isnan(X_fit_scaled).any(), "NaNs found in scaled data before PCA fit."
    pca.fit(X_fit_scaled)
    n_components_selected = pca.n_components_
    print(f"PCA selected {n_components_selected} components.")
    assert n_components_selected > 0, "PCA selected 0 components. Check variance threshold or data."

    print("Transforming data with PCA...")
    assert not np.isnan(X_transform_scaled).any(), "NaNs found in scaled data before PCA transform."
    X_fit_pca = pca.transform(X_fit_scaled)
    X_transform_pca = pca.transform(X_transform_scaled)

    # --- 6. Create Final Processed DataFrames ---
    print("Creating final DataFrames with PCA components...")
    pca_col_names = [f'PC{i+1}' for i in range(n_components_selected)]

    df_fit_pca = pd.DataFrame(X_fit_pca, index=df_fit.index, columns=pca_col_names)
    df_transform_pca = pd.DataFrame(X_transform_pca, index=df_transform.index, columns=pca_col_names)

    # Combine PCA features with essential ID and Target columns
    id_target_date_cols = [feature_cfg.match_id_col, feature_cfg.date_col] + feature_cfg.all_target_cols
    # Ensure these columns actually exist in the original df before selecting
    id_target_date_cols = [col for col in id_target_date_cols if col in df.columns]
    assert feature_cfg.match_id_col in id_target_date_cols, "MatchID column missing."
    assert feature_cfg.date_col in id_target_date_cols, "Date column missing."
    assert all(tc in id_target_date_cols for tc in feature_cfg.all_target_cols), "Target columns missing."

    df_processed_fit = pd.concat([df_fit[id_target_date_cols].reset_index(drop=True), df_fit_pca.reset_index(drop=True)], axis=1)
    df_processed_transform = pd.concat([df_transform[id_target_date_cols].reset_index(drop=True), df_transform_pca.reset_index(drop=True)], axis=1)

    # Combine back into a single DataFrame
    df_processed = pd.concat([df_processed_fit, df_processed_transform], axis=0, ignore_index=True)
    print(f"Final processed PCA DataFrame shape: {df_processed.shape}")

    # Final check for NaNs in PCA columns
    assert not df_processed[pca_col_names].isnull().any().any(), "NaNs found in PCA components after creation!"

    # --- 7. Save Processed Data and Transformers ---
    processed_output_path = output_dir / f'processed_pca_{odds_suffix}.parquet'
    scaler_path = transformers_dir / f'scaler_pca_{odds_suffix}.joblib'
    pca_path = transformers_dir / f'pca_object_{odds_suffix}.joblib'

    print(f"Saving PCA processed data to: {processed_output_path}")
    df_processed.to_parquet(processed_output_path, engine='pyarrow', index=False)

    print(f"Saving Scaler to: {scaler_path}")
    joblib.dump(scaler, scaler_path)

    print(f"Saving PCA object to: {pca_path}")
    joblib.dump(pca, pca_path)

    print(f"--- PCA Data Processing Complete ({odds_suffix}) ---")


if __name__ == "__main__":
    # Define Parameters with Updated Paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_INPUT_PATH = BASE_DIR / 'models' / 'data' / 'parquets' / 'dbs' / 'final_data_with_elo.parquet'
    PROCESSED_OUTPUT_DIR = BASE_DIR / 'models' / 'data' / 'outputs'

    # Ensure output directory exists
    PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Run Processing (Without Odds) ---
    print("\n>>> Running PCA Data Processing WITHOUT Odds <<<")
    run_pca_data_processing(
        raw_data_path=RAW_DATA_INPUT_PATH,
        output_dir=PROCESSED_OUTPUT_DIR,
        include_odds=False,
        pca_variance_threshold=0.95
    )

    # --- Run Processing (With Odds) ---
    print("\n>>> Running PCA Data Processing WITH Odds <<<")
    run_pca_data_processing(
        raw_data_path=RAW_DATA_INPUT_PATH,
        output_dir=PROCESSED_OUTPUT_DIR,
        include_odds=True,
        pca_variance_threshold=0.95
    )