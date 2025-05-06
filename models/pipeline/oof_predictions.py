# pipelines/generate_oof_predictions.py
import pandas as pd
import numpy as np
import os
import json
import warnings
# joblib might not be needed if we always retrain, but keep for potential future use
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler # Needed for scaling within folds
from models.pipeline.train_pipeline import MODELS_TO_TRAIN_CONFIG
from models.utils.features import BaseFeatureConfig, get_feature_config
from models.utils.poisson_model import PoissonModel
from models.ml_models.random_forest_model import RandomForestModel # Currently excluded
from models.ml_models.gradient_boosting_model import GradientBoostingModel
from models.ml_models.monte_carlo_model import MonteCarloModel
# --- Configuration (Aligned with train_pipeline.py and user's desired structure) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Input Data Paths ---
# Processed data is read from the 'parquets' subdirectory
PROCESSED_DATA_INPUT_DIR = os.path.join(BASE_DIR, 'models', 'data', 'parquets')
PROCESSED_DATA_PATH_TEMPLATE = os.path.join(PROCESSED_DATA_INPUT_DIR, 'processed_{}.parquet')

# --- Output, Parameter, and Model Paths ---
# All outputs (OOF predictions, params, potentially models) go into the 'outputs' subdirectory
MAIN_OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'data', 'outputs')

PARAMS_INPUT_DIR = os.path.join(MAIN_OUTPUT_DIR, 'optimized_params') # Parameters are read from here
# MODELS_INPUT_DIR = MAIN_OUTPUT_DIR # Pre-trained models would be loaded from here if OOF needed them

OPTIMIZED_PARAMS_PATH_TEMPLATE = os.path.join(PARAMS_INPUT_DIR, 'best_params_all_models_{}.json') # Expects the combined params file
# MODEL_PRETRAINED_PATH_TEMPLATE = os.path.join(MODELS_INPUT_DIR, '{}_{}_v1.joblib') # Path if OOF needed to load pre-trained models (not used currently)

OOF_PREDICTIONS_OUTPUT_DIR = MAIN_OUTPUT_DIR # OOF predictions saved here
OOF_OUTPUT_PATH_TEMPLATE = os.path.join(OOF_PREDICTIONS_OUTPUT_DIR, 'level0_oof_predictions_{}.csv') # Output file as CSV

# --- Model Registry ---
AVAILABLE_MODELS = {
    "poisson": PoissonModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "monte_carlo": MonteCarloModel,
}

# --- Models to Generate OOF Predictions For ---
MODELS_FOR_OOF = ["poisson", "random_forest", "gradient_boosting", "monte_carlo"]

# --- Core Prediction Keys to Extract (Consistent List) ---
CORE_PREDICTION_KEYS = [
    'prob_H', 'prob_D', 'prob_A', 'prob_O05', 'prob_U05', 'prob_O15', 'prob_U15',
    'prob_O25', 'prob_U25', 'prob_O35', 'prob_U35', 'prob_O45', 'prob_U45',
    'prob_BTTS_Y', 'prob_BTTS_N', 'prob_1X', 'prob_12', 'prob_X2',
    'prob_goals_0_1', 'prob_goals_2_3', 'prob_goals_2_4', 'prob_goals_3_plus',
    'expected_HG', 'expected_AG', 'prob_H_and_O25', 'prob_D_and_O25',
    'prob_A_and_O25', 'prob_H_and_U25', 'prob_D_and_U25', 'prob_A_and_U25',
    'prob_1X_and_O25', 'prob_12_and_O25', 'prob_X2_and_O25', 'prob_1X_and_U25',
    'prob_12_and_U25', 'prob_X2_and_U25', 'prob_H_and_BTTS_Y', 'prob_D_and_BTTS_Y',
    'prob_A_and_BTTS_Y', 'prob_H_and_BTTS_N', 'prob_D_and_BTTS_N', 'prob_A_and_BTTS_N',
    'prob_1X_and_BTTS_Y', 'prob_12_and_BTTS_Y', 'prob_X2_and_BTTS_Y',
    'prob_1X_and_BTTS_N', 'prob_12_and_BTTS_N', 'prob_X2_and_BTTS_N',
    'prob_O25_and_BTTS_Y', 'prob_O25_and_BTTS_N', 'prob_U25_and_BTTS_Y',
    'prob_U25_and_BTTS_N',
]

# --- Cross-Validation Settings ---
N_SPLITS = 8 # Number of folds for TimeSeriesSplit
IMPUTE_NANS = True # Set to True to fill NaNs with column means at the end

# Main OOF Generation Function
def generate_oof_predictions(include_odds: bool):
    """
    Generates Out-of-Fold (OOF) predictions for Level 0 models.

    Args:
        include_odds: Boolean flag whether to use data/models trained with odds.
    """
    odds_suffix = "with_odds" if include_odds else "without_odds"
    print(f"\n===== Generating OOF Predictions ({odds_suffix}) =====")

    # --- 1. Load Full Processed Training Data ---
    data_path = PROCESSED_DATA_PATH_TEMPLATE.format(odds_suffix)
    print(f"Loading full processed data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}. Cannot generate OOF predictions.")
        return
    try:
        df_full = pd.read_parquet(data_path, engine='pyarrow')
        if 'Date' in df_full.columns:
             df_full = df_full.sort_values(by='Date').reset_index(drop=True)
             print("Data sorted by Date for TimeSeriesSplit.")
        else:
             warnings.warn("Date column not found. TimeSeriesSplit might behave unexpectedly.")
        assert not df_full.empty, "Loaded DataFrame is empty."
    except Exception as e:
        print(f"ERROR loading data from {data_path}: {e}")
        return

    # --- 2. Load Feature Config & Prepare Base Data ---
    try:
        feature_cfg = get_feature_config(include_odds=include_odds)
        # Include FTR for potential use in stacking model target
        target_cols = [feature_cfg.target_home_goals, feature_cfg.target_away_goals, feature_cfg.target_result]
        feature_cols = feature_cfg.get_feature_columns(include_odds=include_odds)
        id_col = feature_cfg.match_id_col

        assert set(target_cols).issubset(df_full.columns), f"Missing target columns: {set(target_cols) - set(df_full.columns)}"
        assert set(feature_cols).issubset(df_full.columns), f"Missing feature columns: {set(feature_cols) - set(df_full.columns)}"
        assert id_col in df_full.columns, f"Missing MatchID column: {id_col}"

        X_full = df_full[feature_cols]
        y_full = df_full[target_cols]
        match_ids = df_full[id_col]

        assert not X_full.isnull().any().any(), "NaNs found in features."
        assert not y_full.isnull().any().any(), "NaNs found in targets."

    except Exception as e:
        print(f"ERROR preparing data or loading config: {e}")
        return

    # --- 3. Initialize OOF Storage ---
    oof_pred_dfs = {}

    # --- 4. Setup Cross-Validation ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"Using TimeSeriesSplit with {N_SPLITS} splits.")

    # --- 5. Loop Through Models ---
    for model_key in MODELS_FOR_OOF:
        print(f"\n--- Generating OOF for model: {model_key} ---")
        ModelClass = AVAILABLE_MODELS.get(model_key)
        if not ModelClass:
            print(f"WARNING: Model class for '{model_key}' not found. Skipping.")
            continue

        # Parameter Loading
        params_path = OPTIMIZED_PARAMS_PATH_TEMPLATE.format(odds_suffix)
        model_params = MODELS_TO_TRAIN_CONFIG[model_key]["params"].copy() # Start with default from config, use .copy()

        if os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    loaded_all_models_params = json.load(f)
                    if isinstance(loaded_all_models_params, dict):
                        specific_model_params = loaded_all_models_params.get(model_key)
                        if specific_model_params:
                            model_params.update(specific_model_params) # Update defaults with loaded specific params
                            print(f"Loaded and applied parameters for {model_key} from: {params_path}")
                        else:
                            print(f"WARNING: Parameters for {model_key} not found in {params_path}. Using defaults for {model_key}.")
                    else:
                        print(f"WARNING: Invalid format in parameters file {params_path}. Using defaults for {model_key}.")
            except Exception as e:
                print(f"WARNING: Failed to load parameters file {params_path}: {e}. Using defaults for {model_key}.")
        else:
             if MODELS_TO_TRAIN_CONFIG[model_key].get("optimize", False):
                 print(f"WARNING: Optimized parameters file not found: {params_path}. Using defaults for {model_key}.")
             else:
                 print(f"Using default parameters for {model_key} (optimization not enabled or file not found).")

        # Reduce n_simulations for Monte Carlo specifically for OOF to prevent OOM
        if model_key == "monte_carlo":
            original_sims = model_params.get('n_simulations', 'NOT_SET (using default from class)')
            # Ensure 'params' dict exists for monte_carlo in MODELS_TO_TRAIN_CONFIG if we access it like this
            # For safety, directly assign to model_params which is now a working copy
            model_params['n_simulations'] = 10000  # Reduced value for OOF
            print(f"  Overriding n_simulations for Monte Carlo for OOF: {model_params['n_simulations']} (original/default might have been: {original_sims})")

        model_oof_preds_list = []

        # --- 6. Loop Through Folds ---
        for fold_idx, (train_indices, val_indices) in enumerate(tscv.split(X_full)):
            print(f"  Processing Fold {fold_idx + 1}/{N_SPLITS}...")
            X_train_fold, X_val_fold = X_full.iloc[train_indices], X_full.iloc[val_indices]
            y_train_fold = y_full.iloc[train_indices] # y_val_fold not needed for prediction
            match_ids_val = match_ids.iloc[val_indices]
            print(f"    Train size: {len(X_train_fold)}, Validation size: {len(X_val_fold)}")

            # --- 7. Instantiate and Train Model on Fold Data ---
            try:
                model_fold = ModelClass(model_params=model_params, feature_config=feature_cfg)
                model_fold.fit(X_train_fold, y_train_fold)
            except Exception as e:
                print(f"ERROR fitting model {model_key} on fold {fold_idx + 1}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for fitting errors
                warnings.warn(f"Skipping predictions for model {model_key} on fold {fold_idx + 1} due to fitting error.")
                continue

            # --- 8. Predict on Validation Fold ---
            try:
                val_pred_dict = model_fold.predict_proba(X_val_fold)

                # --- 9. Extract and Store Core Predictions ---
                fold_preds = pd.DataFrame(index=match_ids_val)
                keys_found_count = 0
                for key in CORE_PREDICTION_KEYS:
                    if key in val_pred_dict:
                        col_name = f"{model_key}_{key}"
                        fold_preds[col_name] = val_pred_dict[key]
                        keys_found_count += 1
                    else:
                        # This warning should NOT appear now if CORE_PREDICTION_KEYS is correct
                        warnings.warn(f"Key '{key}' not found in {model_key} predictions for fold {fold_idx + 1}.")

                model_oof_preds_list.append(fold_preds)
                print(f"    Generated and stored {keys_found_count} prediction columns for validation set.")

            except Exception as e:
                print(f"ERROR predicting with model {model_key} on fold {fold_idx + 1}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for prediction errors
                warnings.warn(f"Skipping predictions for model {model_key} on fold {fold_idx + 1} due to prediction error.")
                continue

        # --- 10. Combine Fold Predictions for this Model ---
        if model_oof_preds_list:
            model_oof_df = pd.concat(model_oof_preds_list)
            expected_rows = len(X_full) - len(X_full.iloc[tscv.split(X_full).__next__()[0]]) # Rows not in first train split = rows in first val split
            if len(model_oof_df) != expected_rows:
                 # Refined warning for TimeSeriesSplit
                 warnings.warn(f"Model {model_key}: OOF predictions cover {len(model_oof_df)} matches, expected ~{expected_rows} due to TimeSeriesSplit gap. Check for errors in folds if significantly different.")
            oof_pred_dfs[model_key] = model_oof_df
            print(f"--- Finished OOF generation for model: {model_key} ---")
        else:
            print(f"--- No OOF predictions generated for model: {model_key} (check errors) ---")

    # --- 11. Combine Predictions from All Models ---
    if not oof_pred_dfs:
        print("ERROR: No OOF predictions were generated for any model. Exiting.")
        return

    print("\nCombining OOF predictions from all models...")
    # Start with the base info (MatchID and actual targets) indexed by MatchID
    final_oof_df = df_full[[id_col] + target_cols].set_index(id_col)

    # Join predictions from each model
    for model_key, model_oof_df in oof_pred_dfs.items():
        final_oof_df = final_oof_df.join(model_oof_df, how='left') # Left join keeps all original matches

    print(f"Combined OOF DataFrame shape before NaN check: {final_oof_df.shape}")

    # --- 12. Handle NaNs (Resulting from TimeSeriesSplit gap & any fold errors) ---
    nan_cols = final_oof_df.columns[final_oof_df.isnull().any()].tolist()
    # Filter out target columns which might have NaNs if prediction failed for all models on a row
    pred_nan_cols = [c for c in nan_cols if c not in target_cols]

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
                    # This case handles if a column was all NaNs (so mean is NaN) or if a pred_nan_col somehow wasn't in impute_values
                    warnings.warn(f"Mean for column {col} is NaN or not found in impute_values. Imputing with 0.0.", RuntimeWarning)
                    impute_map[col] = 0.0
            
            # Fill NaNs using the constructed map
            final_oof_df.fillna(value=impute_map, inplace=True)
            print("NaN imputation complete.")
            
            # Verify imputation
            remaining_nans = final_oof_df[pred_nan_cols].isnull().sum().sum()
            if remaining_nans > 0:
                 warnings.warn(f"NaN imputation failed to fill all NaNs. {remaining_nans} remain.")
            else:
                 print("Verified: No NaNs remaining in prediction columns after imputation.")
        else:
            warnings.warn(f"NaNs found in columns: {pred_nan_cols}. IMPUTE_NANS=False, NaNs will remain in the output file.")
    else:
        print("No NaNs found in prediction columns.")


    # --- 13. Save Final OOF DataFrame ---
    output_path = OOF_OUTPUT_PATH_TEMPLATE.format(odds_suffix)
    print(f"Saving final OOF predictions DataFrame to: {output_path}")
    try:
        # Reset index to save MatchID as a column, save as CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        final_oof_df.reset_index().to_csv(output_path, index=False)
        print("OOF predictions saved successfully as CSV.")
    except Exception as e:
        print(f"ERROR saving OOF predictions to {output_path}: {e}")

    print(f"===== OOF Prediction Generation Complete ({odds_suffix}) =====")

# Main Execution Block
if __name__ == "__main__":
    print("Starting OOF Prediction Generation Process...")

    # Generate OOF predictions for both odds settings
    generate_oof_predictions(include_odds=False)
    generate_oof_predictions(include_odds=True)

    print("\nOOF Prediction Generation Process Finished.")