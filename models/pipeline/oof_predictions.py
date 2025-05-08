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
from typing import Dict, List, Type
# --- Add project root to sys.path if needed ---
import sys
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_PATH))
print(f"Project Root added to sys.path: {PROJECT_ROOT_PATH}")
# --- Import model classes ---
from models.utils.features import BaseFeatureConfig, get_feature_config
from models.ml_models.poisson_model import PoissonModel
from models.ml_models.random_forest_model import RandomForestModel
from models.ml_models.gradient_boosting_model import GradientBoostingModel
from models.ml_models.monte_carlo_model import MonteCarloModel
from models.base_model import BaseModel

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
MODELS_SAVE_DIR = DATA_OUTPUT_DIR / 'joblib'

# *** Load PCA processed data ***
PROCESSED_DATA_PATH_TEMPLATE = str(DATA_OUTPUT_DIR / 'processed_pca_{}.parquet')
# *** Load params based on the metric used in train_pipeline ***
# *** Ensure this matches LAMBDA_OPTIMIZATION_METRIC in train_pipeline ***
LAMBDA_OPTIMIZATION_METRIC_USED = 'rmse'
OPTIMIZED_PARAMS_PATH_TEMPLATE = str(PARAMS_OUTPUT_DIR / f'best_params_ray_lambda_{LAMBDA_OPTIMIZATION_METRIC_USED}_{{}}_{{}}.json') # model_key, odds_suffix
# *** Output filename indicates PCA was used ***
OOF_OUTPUT_PATH_TEMPLATE = str(DATA_OUTPUT_DIR / 'level0_oof_predictions_pca_{}.parquet')

# --- Models to Generate OOF Predictions For ---
# *** Ensure this list includes all models trained by train_pipeline ***
MODELS_FOR_OOF = ["poisson", "random_forest", "gradient_boosting", "monte_carlo"]

# --- Core Prediction Keys to Extract from Model Output Dicts ---
# This list defines the columns that will end up in the OOF file (prefixed with model name)
CORE_PREDICTION_KEYS = [
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

# --- Cross-Validation Settings ---
N_SPLITS = 8 # Number of folds for TimeSeriesSplit
IMPUTE_NANS = True # Set to True to fill NaNs with column means at the end

# Main OOF Generation Function
def generate_oof_predictions(include_odds: bool):
    """
    Generates Out-of-Fold (OOF) predictions for Level 0 models using PCA features.

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

    # Sort data for TimeSeriesSplit
    date_col = 'Date' # Assuming 'Date' column exists from data processing
    assert date_col in df_full.columns, f"Date column '{date_col}' not found for sorting."
    df_full[date_col] = pd.to_datetime(df_full[date_col])
    df_full = df_full.sort_values(by=date_col).reset_index(drop=True)
    print("Data sorted by Date for TimeSeriesSplit.")

    # --- 2. Load Feature Config & Prepare Base Data ---
    print("Loading feature config and preparing data...")
    feature_cfg = get_feature_config(include_odds=include_odds)
    target_cols = [feature_cfg.target_home_goals, feature_cfg.target_away_goals, feature_cfg.target_result]
    # Features are now PCA components
    feature_cols = [col for col in df_full.columns if col.startswith('PC')]
    id_col = feature_cfg.match_id_col

    assert feature_cols, f"No PCA feature columns (PC*) found in {data_path}."
    required_cols = set(target_cols) | set(feature_cols) | {id_col, date_col}
    missing_cols = required_cols - set(df_full.columns)
    assert not missing_cols, f"PCA DataFrame missing required columns: {missing_cols}"

    X_full: pd.DataFrame = df_full[feature_cols]
    y_full: pd.DataFrame = df_full[target_cols]
    match_ids: pd.Series = df_full[id_col]

    # Assert data integrity
    assert not X_full.isnull().any().any(), "NaNs found in PCA features."
    assert not y_full.isnull().any().any(), "NaNs found in targets."
    assert X_full.shape[0] == y_full.shape[0] == match_ids.shape[0], "Shape mismatch."
    print(f"Data prepared: {X_full.shape[0]} matches, {X_full.shape[1]} PCA features.")

    # --- 3. Initialize OOF Storage ---
    oof_pred_dfs: Dict[str, pd.DataFrame] = {}

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
        # Start with default params from config (handle potential import error)
        model_params = {}
        try:
            from models.pipeline.train_pipeline import MODELS_TO_TRAIN_CONFIG
            model_params = MODELS_TO_TRAIN_CONFIG.get(model_key, {}).get("params", {})
        except ImportError:
            warnings.warn(f"Could not import MODELS_TO_TRAIN_CONFIG, using empty default params for {model_key}.")
        assert isinstance(model_params, dict), f"Default params for {model_key} not found or not a dict."

        if params_path.exists():
            print(f"Loading parameters for {model_key} from: {params_path}")
            try:
                with open(params_path, 'r') as f: loaded_params = json.load(f)
                assert isinstance(loaded_params, dict), f"Invalid format in {params_path}"
                model_params = loaded_params # Use loaded params
            except (json.JSONDecodeError, AssertionError, FileNotFoundError) as e:
                 warnings.warn(f"Failed loading {params_path}: {e}. Using defaults for {model_key}.", UserWarning)
        else:
             warnings.warn(f"Params file not found: {params_path}. Using defaults for {model_key}.", UserWarning)
        print(f"Using parameters: {model_params}")

        # Reduce n_simulations for Monte Carlo specifically for OOF if needed
        if model_key == "monte_carlo":
            oof_sims = 5000 # Use fewer sims for OOF generation speed/memory
            print(f"  Overriding n_simulations for Monte Carlo for OOF: {oof_sims}")
            model_params['n_simulations'] = oof_sims # Modify the params dict for this run

        model_oof_preds_list: List[pd.DataFrame] = []

        # --- 6. Loop Through Folds ---
        for fold_idx, (train_indices, val_indices) in enumerate(tscv.split(X_full)):
            fold_num = fold_idx + 1
            print(f"  Processing Fold {fold_num}/{N_SPLITS}...")
            if len(val_indices) == 0: print("    Skipping empty validation set."); continue

            X_train_fold, X_val_fold = X_full.iloc[train_indices], X_full.iloc[val_indices]
            y_train_fold = y_full.iloc[train_indices]
            match_ids_val = match_ids.iloc[val_indices]
            print(f"    Train size: {len(X_train_fold)}, Validation size: {len(X_val_fold)}")

            # Assert fold data validity
            assert not X_train_fold.isnull().any().any(), f"Fold {fold_num}: NaNs in X_train_fold."
            assert not y_train_fold.isnull().any().any(), f"Fold {fold_num}: NaNs in y_train_fold."
            assert not X_val_fold.isnull().any().any(), f"Fold {fold_num}: NaNs in X_val_fold."

            # --- 7. Instantiate and Train Model on Fold Data ---
            print(f"    Fitting model {model_key}...")
            # *** Instantiate with apply_scaling=False ***
            model_fold = ModelClass(model_params=model_params, feature_config=feature_cfg, apply_scaling=False)
            assert hasattr(model_fold, 'fit'), f"Model {model_key} lacks fit method."
            model_fold.fit(X_train_fold, y_train_fold) # Internal scaling is skipped
            print(f"    Model fitted.")

            # --- 8. Predict on Validation Fold ---
            print(f"    Predicting with model {model_key}...")
            assert hasattr(model_fold, 'predict_proba'), f"Model {model_key} lacks predict_proba method."
            # Predict using PCA features, scaling is skipped internally
            val_pred_dict: Dict[str, np.ndarray] = model_fold.predict_proba(X_val_fold)
            assert isinstance(val_pred_dict, dict), "predict_proba did not return dict."
            print(f"    Prediction complete.")

            # --- 9. Extract and Store Core Predictions ---
            fold_preds = pd.DataFrame(index=match_ids_val)
            keys_found_count = 0
            missing_keys_in_fold = []
            # Use CORE_PREDICTION_KEYS which contains the raw suffixes/keys
            for raw_key in CORE_PREDICTION_KEYS:
                # Models should return keys prefixed with model name (e.g., poisson_prob_H)
                # Let's adjust the check to look for the prefixed key
                prefixed_key = f"{model_key}_{raw_key}"
                if raw_key in val_pred_dict: # Check if the raw key exists in the model's output dict
                    # Assert prediction array shape matches validation index
                    assert val_pred_dict[raw_key].shape == (len(match_ids_val),), \
                        f"Fold {fold_num}, Model {model_key}, Key {raw_key}: Shape mismatch " \
                        f"({val_pred_dict[raw_key].shape} vs expected ({len(match_ids_val)},))"
                    # Store with the model prefix for clarity in the final OOF file
                    fold_preds[prefixed_key] = val_pred_dict[raw_key]
                    keys_found_count += 1
                else:
                    missing_keys_in_fold.append(raw_key)

            if missing_keys_in_fold:
                 warnings.warn(f"Keys {missing_keys_in_fold} not found in {model_key} predictions dict for fold {fold_num}.", UserWarning)

            assert keys_found_count > 0, f"Fold {fold_num}, Model {model_key}: No prediction keys found!"
            model_oof_preds_list.append(fold_preds)
            print(f"    Generated and stored {keys_found_count} prediction columns for validation set.")


        # --- 10. Combine Fold Predictions for this Model ---
        if not model_oof_preds_list:
            warnings.warn(f"No OOF predictions generated for model: {model_key} (check errors in folds).", RuntimeWarning)
            continue # Skip to next model

        model_oof_df = pd.concat(model_oof_preds_list)
        # Calculate expected rows more robustly
        first_fold_indices = tscv.split(X_full).__next__()
        expected_rows = len(X_full) - len(first_fold_indices[0])
        if len(model_oof_df) != expected_rows:
             warnings.warn(f"Model {model_key}: OOF predictions cover {len(model_oof_df)} matches, expected ~{expected_rows} due to TimeSeriesSplit gap. Check for errors in folds if significantly different.", UserWarning)
        oof_pred_dfs[model_key] = model_oof_df
        print(f"--- Finished OOF generation for model: {model_key} ---")


    # --- 11. Combine Predictions from All Models ---
    if not oof_pred_dfs:
        raise RuntimeError("CRITICAL: No OOF predictions were generated for ANY model. Cannot proceed.")

    print("\nCombining OOF predictions from all models...")
    # Start with the base info (MatchID and actual targets) indexed by MatchID
    # Include Date for potential future analysis or joining
    final_oof_df = df_full[[id_col, date_col] + target_cols].set_index(id_col)

    # Join predictions from each model that successfully generated predictions
    for model_key, model_oof_df in oof_pred_dfs.items():
        final_oof_df = final_oof_df.join(model_oof_df, how='left') # Left join keeps all original matches

    print(f"Combined OOF DataFrame shape before NaN check: {final_oof_df.shape}")

    # --- 12. Handle NaNs (Resulting from TimeSeriesSplit gap & any fold errors) ---
    nan_cols = final_oof_df.columns[final_oof_df.isnull().any()].tolist()
    # Exclude target/ID/Date columns from imputation list
    pred_nan_cols = [c for c in nan_cols if c not in target_cols + [date_col]]

    if pred_nan_cols:
        print(f"Found NaNs in {len(pred_nan_cols)} prediction columns (likely due to TimeSeriesSplit gap or fold errors).")
        if IMPUTE_NANS:
            print("Imputing NaNs using column means...")
            impute_values = final_oof_df[pred_nan_cols].mean()
            impute_map = {}
            for col in pred_nan_cols:
                if col in impute_values and not pd.isna(impute_values[col]):
                    impute_map[col] = impute_values[col]
                else:
                    fallback_val = 0.5 if '_prob_' in col else 0.0 # Impute probs with 0.5, others with 0
                    warnings.warn(f"Mean for column {col} is NaN or missing. Imputing with {fallback_val}.", RuntimeWarning)
                    impute_map[col] = fallback_val

            final_oof_df.fillna(value=impute_map, inplace=True)
            print("NaN imputation complete.")

            remaining_nans = final_oof_df[pred_nan_cols].isnull().sum().sum()
            assert remaining_nans == 0, f"NaN imputation failed! {remaining_nans} NaNs remain."
            print("Verified: No NaNs remaining in prediction columns after imputation.")
        else:
            warnings.warn(f"NaNs found in columns: {pred_nan_cols}. IMPUTE_NANS=False, NaNs will remain.", RuntimeWarning)
    else:
        print("No NaNs found in prediction columns.")


    # --- 13. Save Final OOF DataFrame ---
    output_path = Path(OOF_OUTPUT_PATH_TEMPLATE.format(odds_suffix))
    print(f"Saving final OOF predictions DataFrame to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    # Reset index to save MatchID as a column, save as PARQUET
    final_oof_df.reset_index().to_parquet(output_path, index=False, engine='pyarrow')
    print("OOF predictions saved successfully as Parquet.")

    print(f"===== OOF Prediction Generation Complete (PCA Features, {odds_suffix}) =====")

# Main Execution Block
if __name__ == "__main__":
    print("Starting OOF Prediction Generation Process...")

    # Generate OOF predictions for both odds settings using PCA data
    generate_oof_predictions(include_odds=False)
    generate_oof_predictions(include_odds=True)

    print("\nOOF Prediction Generation Process Finished.")