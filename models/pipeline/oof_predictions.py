# pipelines/generate_oof_predictions.py
import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib # Keep for potential future use
from pathlib import Path
import time
import traceback
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Type, Any # Added Any

# --- Add project root to sys.path if needed ---
import sys
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
assert str(PROJECT_ROOT_PATH) not in sys.path, f"Project Root already in sys.path: {PROJECT_ROOT_PATH}"
sys.path.append(str(PROJECT_ROOT_PATH))
print(f"Project Root added to sys.path: {PROJECT_ROOT_PATH}")

# --- Import model classes ---
from models.utils.features import BaseFeatureConfig, get_feature_config
from models.ml_models.poisson_model import PoissonModel
from models.ml_models.random_forest_model import RandomForestModel
from models.ml_models.gradient_boosting_model import GradientBoostingModel
from models.ml_models.monte_carlo_model import MonteCarloModel
from models.base_model import BaseModel
from models.pipeline.train_pipeline import MODELS_TO_TRAIN_CONFIG

# --- Model Registry ---
AVAILABLE_MODELS: Dict[str, Type[BaseModel]] = {
    "poisson": PoissonModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "monte_carlo": MonteCarloModel,
}

# --- Configuration (Aligned with train_pipeline.py) ---
BASE_DIR = PROJECT_ROOT_PATH
DATA_OUTPUT_DIR = BASE_DIR / 'models' / 'data' / 'outputs'
PARAMS_OUTPUT_DIR = DATA_OUTPUT_DIR / 'optimized_params'
MODELS_SAVE_DIR = DATA_OUTPUT_DIR / 'joblib' / 'V2'

# *** Load PCA processed data ***
PROCESSED_DATA_PATH_TEMPLATE = str(DATA_OUTPUT_DIR / 'processed_pca_{}.parquet')
# *** Load params based on the metric used in train_pipeline ***
LAMBDA_OPTIMIZATION_METRIC_USED = 'rmse' # <<< ENSURE THIS MATCHES train_pipeline.py
OPTIMIZED_PARAMS_PATH_TEMPLATE = str(PARAMS_OUTPUT_DIR / f'best_params_ray_lambda_{LAMBDA_OPTIMIZATION_METRIC_USED}_{{}}_{{}}.json') # {model_key}_{odds_suffix}
# *** Output filename indicates PCA was used ***
OOF_OUTPUT_PATH_TEMPLATE = str(DATA_OUTPUT_DIR / 'level0_oof_predictions_pca_{}.parquet') # {odds_suffix}

# --- Models to Generate OOF Predictions For ---
MODELS_FOR_OOF = ["poisson", "random_forest", "gradient_boosting", "monte_carlo"] # <<< Ensure this matches trained models

# --- Core Prediction Keys to Extract from Model Output Dicts ---
# This list defines the columns that will end up in the OOF file (prefixed with model name)
CORE_PREDICTION_KEYS = [
    # Lambdas (NEW) - Assuming models return these raw keys
    'expected_HG', 'expected_AG',
    # Single outcomes
    'prob_H', 'prob_D', 'prob_A',
    'prob_1X', 'prob_12', 'prob_X2',
    'prob_O05', 'prob_U05',
    'prob_O15', 'prob_U15',
    'prob_O25', 'prob_U25',
    'prob_O35', 'prob_U35',
    'prob_O45', 'prob_U45',
    'prob_BTTS_Y', 'prob_BTTS_N',
    # Multiple outcomes - Match Result + Goals
    'prob_H_and_O05', 'prob_D_and_O05', 'prob_A_and_O05',
    'prob_H_and_U05', 'prob_D_and_U05', 'prob_A_and_U05',
    'prob_H_and_O15', 'prob_D_and_O15', 'prob_A_and_O15',
    'prob_H_and_U15', 'prob_D_and_U15', 'prob_A_and_U15',
    'prob_H_and_O25', 'prob_D_and_O25', 'prob_A_and_O25',
    'prob_H_and_U25', 'prob_D_and_U25', 'prob_A_and_U25',
    'prob_H_and_O35', 'prob_D_and_O35', 'prob_A_and_O35',
    'prob_H_and_U35', 'prob_D_and_U35', 'prob_A_and_U35',
    'prob_H_and_O45', 'prob_D_and_O45', 'prob_A_and_O45',
    'prob_H_and_U45', 'prob_D_and_U45', 'prob_A_and_U45',
    # Multiple outcomes - Double Chance + Goals
    'prob_1X_and_O05', 'prob_12_and_O05', 'prob_X2_and_O05',
    'prob_1X_and_U05', 'prob_12_and_U05', 'prob_X2_and_U05',
    'prob_1X_and_O15', 'prob_12_and_O15', 'prob_X2_and_O15',
    'prob_1X_and_U15', 'prob_12_and_U15', 'prob_X2_and_U15',
    'prob_1X_and_O25', 'prob_12_and_O25', 'prob_X2_and_O25',
    'prob_1X_and_U25', 'prob_12_and_U25', 'prob_X2_and_U25',
    'prob_1X_and_O35', 'prob_12_and_O35', 'prob_X2_and_O35',
    'prob_1X_and_U35', 'prob_12_and_U35', 'prob_X2_and_U35',
    'prob_1X_and_O45', 'prob_12_and_O45', 'prob_X2_and_O45',
    'prob_1X_and_U45', 'prob_12_and_U45', 'prob_X2_and_U45',
    # Multiple outcomes - Match Result + BTTS
    'prob_H_and_BTTS_Y', 'prob_D_and_BTTS_Y', 'prob_A_and_BTTS_Y',
    'prob_H_and_BTTS_N', 'prob_D_and_BTTS_N', 'prob_A_and_BTTS_N',
    # Multiple outcomes - Double Chance + BTTS
    'prob_1X_and_BTTS_Y', 'prob_12_and_BTTS_Y', 'prob_X2_and_BTTS_Y',
    'prob_1X_and_BTTS_N', 'prob_12_and_BTTS_N', 'prob_X2_and_BTTS_N',
    # Multiple outcomes - Goals + BTTS
    'prob_O25_and_BTTS_Y', 'prob_O25_and_BTTS_N',
    'prob_U25_and_BTTS_Y', 'prob_U25_and_BTTS_N',
    'prob_O35_and_BTTS_Y', 'prob_O35_and_BTTS_N',
    'prob_U35_and_BTTS_Y', 'prob_U35_and_BTTS_N',
    'prob_O45_and_BTTS_Y', 'prob_O45_and_BTTS_N',
    'prob_U45_and_BTTS_Y', 'prob_U45_and_BTTS_N',
]
print(f"Generating OOF for {len(CORE_PREDICTION_KEYS)} prediction keys (including lambdas).")

# --- Cross-Validation Settings ---
N_SPLITS = 8 # Number of folds for TimeSeriesSplit
IMPUTE_NANS = True # Fill NaNs from initial folds with column means

# Main OOF Generation Function
def generate_oof_predictions(include_odds: bool):
    """
    Generates Out-of-Fold (OOF) predictions for Level 0 models using PCA features.
    Includes raw lambda predictions alongside derived probabilities.

    Args:
        include_odds: Boolean flag whether to use data/models trained with odds.
    """
    odds_suffix = "with_odds" if include_odds else "without_odds"
    print(f"\n===== Generating OOF Predictions (PCA Features, {odds_suffix}) =====")

    # --- 1. Load Full PCA Processed Training Data ---
    data_path = Path(PROCESSED_DATA_PATH_TEMPLATE.format(odds_suffix))
    print(f"Loading full PCA processed data from: {data_path}")
    assert data_path.exists(), f"PCA Data file not found: {data_path}"
    df_full = pd.read_parquet(data_path, engine='pyarrow')
    assert not df_full.empty, "Loaded PCA DataFrame is empty."

    date_col = 'Date'
    assert date_col in df_full.columns, f"Date column '{date_col}' not found for sorting."
    df_full[date_col] = pd.to_datetime(df_full[date_col])
    df_full = df_full.sort_values(by=date_col).reset_index(drop=True)
    print(f"Data sorted by Date. Shape: {df_full.shape}")

    # --- 2. Load Feature Config & Prepare Base Data ---
    print("Loading feature config and preparing data...")
    feature_cfg = get_feature_config(include_odds=include_odds)
    target_cols = [feature_cfg.target_home_goals, feature_cfg.target_away_goals, feature_cfg.target_result]
    feature_cols = [col for col in df_full.columns if col.startswith('PC')]
    id_col = feature_cfg.match_id_col

    assert feature_cols, f"No PCA feature columns (PC*) found in {data_path}."
    required_cols = set(target_cols) | set(feature_cols) | {id_col, date_col}
    missing_cols = required_cols - set(df_full.columns)
    assert not missing_cols, f"PCA DataFrame missing required columns: {missing_cols}"

    X_full: pd.DataFrame = df_full[feature_cols]
    y_full: pd.DataFrame = df_full[target_cols]
    match_ids: pd.Series = df_full[id_col]

    assert not X_full.isnull().any().any(), "NaNs found in PCA features."
    assert not y_full.isnull().any().any(), "NaNs found in targets."
    assert X_full.shape[0] == y_full.shape[0] == match_ids.shape[0], "Shape mismatch."
    print(f"Data prepared: {X_full.shape[0]} matches, {X_full.shape[1]} PCA features.")

    # --- 3. Initialize OOF Storage ---
    oof_pred_dfs: Dict[str, pd.DataFrame] = {} # Stores final OOF df per model

    # --- 4. Setup Cross-Validation ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"Using TimeSeriesSplit with {N_SPLITS} splits.")

    # --- 5. Loop Through Models ---
    for model_key in MODELS_FOR_OOF:
        print(f"\n--- Generating OOF for model: {model_key} ---")
        ModelClass = AVAILABLE_MODELS.get(model_key)
        assert ModelClass is not None, f"Model class for '{model_key}' not found."

        # --- Load Correct Parameters (Lambda Optimized) ---
        params_path = Path(OPTIMIZED_PARAMS_PATH_TEMPLATE.format(model_key, odds_suffix))
        # Get defaults from imported config as fallback
        model_params = MODELS_TO_TRAIN_CONFIG.get(model_key, {}).get("params", {})
        assert isinstance(model_params, dict), f"Default params for {model_key} not found or not a dict."

        if params_path.exists():
            print(f"  Loading parameters for {model_key} from: {params_path}")
            try:
                with open(params_path, 'r') as f: loaded_params = json.load(f)
                assert isinstance(loaded_params, dict), f"Invalid format in {params_path}"
                model_params = loaded_params # Use loaded params (fixed + tuned)
            except Exception as e:
                 warnings.warn(f"  Failed loading {params_path}: {e}. Using defaults for {model_key}.", UserWarning)
        else:
             warnings.warn(f"  Params file not found: {params_path}. Using defaults for {model_key}.", UserWarning)
        print(f"  Using parameters: {model_params}")

        # Reduce n_simulations for Monte Carlo specifically for OOF if needed for speed
        if model_key == "monte_carlo" and 'n_simulations' in model_params:
            oof_sims = model_params.get('n_simulations', 10000) // 2 # Example: Halve sims for OOF
            print(f"  Adjusting n_simulations for Monte Carlo OOF: {oof_sims}")
            model_params['n_simulations'] = max(oof_sims, 1000) # Ensure reasonable minimum

        model_oof_preds_list: List[pd.DataFrame] = [] # Stores prediction DFs from each fold for this model

        # --- 6. Loop Through Folds ---
        for fold_idx, (train_indices, val_indices) in enumerate(tscv.split(X_full)):
            fold_num = fold_idx + 1
            print(f"  Processing Fold {fold_num}/{N_SPLITS}...")
            if len(val_indices) == 0: print("    Skipping empty validation set."); continue

            X_train_fold, X_val_fold = X_full.iloc[train_indices], X_full.iloc[val_indices]
            y_train_fold = y_full.iloc[train_indices]
            match_ids_val = match_ids.iloc[val_indices]
            print(f"    Train size: {len(X_train_fold)}, Validation size: {len(X_val_fold)}")

            assert not X_train_fold.isnull().any().any(), f"Fold {fold_num}: NaNs in X_train_fold."
            assert not y_train_fold.isnull().any().any(), f"Fold {fold_num}: NaNs in y_train_fold."
            assert not X_val_fold.isnull().any().any(), f"Fold {fold_num}: NaNs in X_val_fold."

            # --- 7. Instantiate and Train Model on Fold Data ---
            print(f"    Fitting model {model_key}...")
            model_fold = ModelClass(model_params=model_params, feature_config=feature_cfg, apply_scaling=False)
            assert hasattr(model_fold, 'fit'), f"Model {model_key} lacks fit method."
            # Fit using only goal targets if it's a lambda-predicting model primarily
            y_fit_targets = y_train_fold[[feature_cfg.target_home_goals, feature_cfg.target_away_goals]]
            model_fold.fit(X_train_fold, y_fit_targets)
            print(f"    Model fitted.")

            # --- 8. Predict on Validation Fold ---
            print(f"    Predicting with model {model_key}...")
            assert hasattr(model_fold, 'predict_proba'), f"Model {model_key} lacks predict_proba method."
            val_pred_dict: Dict[str, np.ndarray] = model_fold.predict_proba(X_val_fold)
            print(f"    Debug: Keys from model {model_key} predict_proba: {list(val_pred_dict.keys())}") # Temporary debug line
            assert isinstance(val_pred_dict, dict), "predict_proba did not return dict."
            print(f"    Prediction complete.")

            # --- 9. Extract and Store Core Predictions ---
            # Initialize a dictionary to hold prediction arrays for the current fold
            current_fold_pred_data: Dict[str, np.ndarray] = {}
            keys_found_count = 0
            missing_keys_in_fold = []

            # Iterate through the desired raw keys (now includes lambdas)
            for raw_key in CORE_PREDICTION_KEYS:
                # Construct the key name as it's expected to appear in the model's output dict
                expected_model_output_key = f"{model_key}_{raw_key}"

                # Check if the model's output dict contains this expected (prefixed) key
                if expected_model_output_key in val_pred_dict:
                    pred_array = val_pred_dict[expected_model_output_key]
                    assert pred_array.shape == (len(match_ids_val),), \
                        f"Fold {fold_num}, Model {model_key}, Key {expected_model_output_key}: Shape mismatch " \
                        f"({pred_array.shape} vs expected ({len(match_ids_val)},))"
                    
                    # Store the array in our dictionary
                    current_fold_pred_data[expected_model_output_key] = pred_array
                    keys_found_count += 1
                else:
                    # If the prefixed key is not found, then the raw_key is indeed missing
                    missing_keys_in_fold.append(raw_key)
            
            # Create the DataFrame for the fold's predictions in one go
            fold_preds = pd.DataFrame(current_fold_pred_data, index=match_ids_val)

            if missing_keys_in_fold:
                 warnings.warn(f"Keys {missing_keys_in_fold} not found in {model_key} predictions dict for fold {fold_num}.", UserWarning)

            assert keys_found_count > 0, f"Fold {fold_num}, Model {model_key}: No prediction keys found!"
            model_oof_preds_list.append(fold_preds)
            print(f"    Stored {keys_found_count} prediction columns for validation set.")


        # --- 10. Combine Fold Predictions for this Model ---
        if not model_oof_preds_list:
            warnings.warn(f"No OOF predictions generated for model: {model_key}. Skipping.", RuntimeWarning)
            continue

        model_oof_df = pd.concat(model_oof_preds_list).sort_index() # Sort by index (MatchID)
        oof_pred_dfs[model_key] = model_oof_df
        print(f"--- Finished OOF generation for model: {model_key} ({len(model_oof_df)} rows) ---")


    # --- 11. Combine Predictions from All Models ---
    if not oof_pred_dfs:
        raise RuntimeError("CRITICAL: No OOF predictions were generated for ANY model. Cannot proceed.")

    print("\nCombining OOF predictions from all models...")
    # Start with base info, ensure index is MatchID for joining
    final_oof_df = df_full[[id_col, date_col] + target_cols].set_index(id_col)

    for model_key, model_oof_df in oof_pred_dfs.items():
        # Ensure the model_oof_df index is also MatchID before joining
        assert model_oof_df.index.name == id_col, f"Index name mismatch for {model_key}"
        final_oof_df = final_oof_df.join(model_oof_df, how='left') # Left join preserves all original matches

    print(f"Combined OOF DataFrame shape before NaN check: {final_oof_df.shape}")
    final_oof_df = final_oof_df.sort_values(by=date_col)

    # --- 12. Handle NaNs ---
    nan_cols = final_oof_df.columns[final_oof_df.isnull().any()].tolist()
    pred_nan_cols = [c for c in nan_cols if c not in target_cols + [date_col]]

    if pred_nan_cols:
        print(f"Found NaNs in {len(pred_nan_cols)} prediction columns (likely TimeSeriesSplit gap).")
        if IMPUTE_NANS:
            print("Imputing NaNs using column means...")
            impute_values = final_oof_df[pred_nan_cols].mean()
            impute_map = {}
            for col in pred_nan_cols:
                mean_val = impute_values.get(col) # Use .get for safety
                if pd.notna(mean_val):
                    impute_map[col] = mean_val
                else: # Fallback if mean is NaN (e.g., all values were NaN in a column)
                    fallback_val = 0.5 if '_prob_' in col else 0.0
                    warnings.warn(f"Mean for column {col} is NaN. Imputing with {fallback_val}.", RuntimeWarning)
                    impute_map[col] = fallback_val

            final_oof_df.fillna(value=impute_map, inplace=True)
            remaining_nans = final_oof_df[pred_nan_cols].isnull().sum().sum()
            assert remaining_nans == 0, f"NaN imputation failed! {remaining_nans} NaNs remain."
            print("NaN imputation complete. No NaNs remaining in prediction columns.")
        else:
            warnings.warn(f"NaNs found in columns: {pred_nan_cols}. IMPUTE_NANS=False, NaNs will remain.", RuntimeWarning)
    else:
        print("No NaNs found in prediction columns.")


    # --- 13. Save Final OOF DataFrame ---
    output_path = Path(OOF_OUTPUT_PATH_TEMPLATE.format(odds_suffix))
    print(f"Saving final OOF predictions DataFrame to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Reset index to save MatchID as a column
    final_oof_df.reset_index().to_parquet(output_path, index=False, engine='pyarrow')
    print("OOF predictions saved successfully as Parquet.")

    print(f"===== OOF Prediction Generation Complete (PCA Features, {odds_suffix}) =====")

# Main Execution Block
if __name__ == "__main__":
    print("Starting OOF Prediction Generation Process...")
    start_time = time.time()

    # Generate OOF predictions for both odds settings using PCA data
    generate_oof_predictions(include_odds=False)
    generate_oof_predictions(include_odds=True)

    end_time = time.time()
    print(f"\nOOF Prediction Generation Process Finished. Total time: {end_time - start_time:.2f} seconds.")