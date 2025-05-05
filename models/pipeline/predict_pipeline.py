# pipelines/predict_pipeline.py
import pandas as pd
import numpy as np
import os
import warnings # To show warnings clearly

# --- Imports from our modules ---
from models.utils.poisson_model import PoissonModel # Import specific model class
from models.utils.predict import format_predictions
from models.utils.config import PredictionConfig
from models.utils.features import BaseFeatureConfig


# --- Model Loading Registry ---
# Assumes models have a 'load' classmethod defined in BaseModel
MODEL_LOADERS = {
    "poisson": PoissonModel.load,
    # "xgboost": XGBoostModel.load, # Example for future models
}


def run_prediction(model_path: str, new_data_path: str, output_path: str, model_type: str = "poisson"):
    """
    Loads a trained model, predicts on new data, formats predictions according
    to PredictionConfig, merges actual results if available, and saves the results.
    Handles duplicate MatchIDs in input data.

    Args:
        model_path: Path to the saved trained model artifact (.joblib).
        new_data_path: Path to the *processed* new data parquet file.
                       Must match the feature set (incl./excl. odds) of the trained model.
                       If it contains actual result columns (FTHG, FTAG, FTR), they will be merged.
        output_path: Path to save the final predictions CSV file (potentially with actual results).
        model_type: String identifier for the type of model being loaded (e.g., 'poisson').
                    Used to select the correct loading method.
    """
    print(f"\n--- Starting Prediction Pipeline ---")
    print(f"Model path: {model_path}")
    print(f"New data path: {new_data_path}")
    print(f"Output path: {output_path}")
    print(f"Model type: {model_type}")

    # --- 1. Load Prediction Configuration ---
    try:
        pred_cfg = PredictionConfig() # Load prediction settings
        # Get standard target column names
        feature_cfg_defaults = BaseFeatureConfig()
        target_cols = feature_cfg_defaults.all_target_cols
        match_id_col = feature_cfg_defaults.match_id_col # Use the standard MatchID column name

        print(f"Using PredictionConfig: Top N={pred_cfg.n_top_predictions}, Threshold={pred_cfg.min_probability_threshold}")
        # Check which dual categories are included based on the lists in the config
        included_dual_types = []
        if pred_cfg.include_1x2_ou25_duals: included_dual_types.append("1X2 & O/U 2.5") 
        if pred_cfg.include_dc_ou25_duals: included_dual_types.append("Double Chance & O/U 2.5")
        if pred_cfg.include_1x2_btts_duals: included_dual_types.append("1X2 & BTTS")
        if pred_cfg.include_dc_btts_duals: included_dual_types.append("Double Chance & BTTS") 
        if pred_cfg.include_ou25_btts_duals: included_dual_types.append("O/U 2.5 & BTTS")
        print(f"Including Dual Outcome Types (calculated via scoreline): {', '.join(included_dual_types) if included_dual_types else 'None'}")
    except Exception as e:
        print(f"Error loading PredictionConfig or FeatureConfig defaults: {e}")
        raise

    # --- 2. Load Trained Model ---
    print(f"Loading trained model from: {model_path}")
    load_func = MODEL_LOADERS.get(model_type)
    assert load_func is not None, f"No loader found for model type '{model_type}'"

    try:
        model = load_func(model_path)
        # --- Assertions on loaded model ---
        assert hasattr(model, 'predict_proba') and callable(model.predict_proba), f"Loaded model from {model_path} lacks a callable 'predict_proba' method."
        assert hasattr(model, 'features_in_') and isinstance(model.features_in_, list) and model.features_in_, \
            f"Loaded model from {model_path} lacks a non-empty list attribute 'features_in_'."
        print(f"Model loaded successfully. Trained on {len(model.features_in_)} features (e.g., {model.features_in_[:3]}...).")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model file not found at {model_path}")
        raise
    except Exception as e:
        print(f"CRITICAL ERROR loading model '{model_type}' from {model_path}: {e}")
        raise

    # --- 3. Load New Processed Data ---
    print(f"Loading new data from: {new_data_path}")
    try:
        new_df_raw = pd.read_parquet(new_data_path, engine='pyarrow')
    except FileNotFoundError:
        print(f"CRITICAL ERROR: New data file not found at {new_data_path}")
        raise
    except Exception as e:
        print(f"CRITICAL ERROR loading new data parquet file: {e}")
        raise

    print(f"Loaded raw new data shape: {new_df_raw.shape}")
    if new_df_raw.empty:
        print("Warning: New data DataFrame is empty. No predictions will be generated.")
        # Create empty output file with correct columns and exit gracefully
        output_dir = os.path.dirname(output_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        # Include potential target columns in the empty output for consistency
        empty_cols = pred_cfg.required_match_info_cols + ["Rank", "Outcome", "Probability"] + target_cols
        pd.DataFrame(columns=empty_cols).to_csv(output_path, index=False)
        print(f"Empty prediction file created at {output_path}")
        print(f"--- Prediction Pipeline Complete (No Data) ---")
        return # Stop processing

    # Handle Duplicate MatchIDs
    if match_id_col not in new_df_raw.columns:
         raise KeyError(f"CRITICAL ERROR: Expected MatchID column '{match_id_col}' not found in the loaded data for deduplication.")

    initial_rows = len(new_df_raw)
    new_df = new_df_raw.drop_duplicates(subset=[match_id_col], keep='first').copy()
    dropped_rows = initial_rows - len(new_df)

    if dropped_rows > 0:
        warnings.warn(f"WARNING: Found and removed {dropped_rows} duplicate rows based on '{match_id_col}'. Kept the first occurrence for each MatchID.")
        print(f"Data shape after deduplication: {new_df.shape}")
    else:
        print("No duplicate MatchIDs found in the new data.")

    # ---> ADDED: Extract Actual Results if Available <---
    actual_results_df = None
    if set(target_cols).issubset(new_df.columns):
        print(f"Found actual result columns ({target_cols}) in the input data. Extracting for merging.")
        # Add date_col to the columns extracted for the actual_results_df
        actual_results_df = new_df[[match_id_col, feature_cfg_defaults.date_col] + target_cols].copy()
    else:
        missing_targets = set(target_cols) - set(new_df.columns)
        print(f"Warning: Actual result columns ({missing_targets}) not found in the input data. Cannot merge actual results.")
    # ---> END ADDED SECTION <---

    # --- 4. Prepare Features (X) and Match Info ---
    print("Preparing features (X) and match info...")

    # Validate required features are present in the new data
    trained_features = model.features_in_
    actual_features = new_df.columns
    missing_features = set(trained_features) - set(actual_features)
    assert not missing_features, f"CRITICAL ERROR: New data is missing features the model was trained on: {missing_features}"

    # Identify extra columns, carefully excluding targets now
    known_info_cols = set(pred_cfg.required_match_info_cols) | set(target_cols)
    extra_features = set(actual_features) - set(trained_features) - known_info_cols
    if extra_features:
        print(f"Warning: New data has extra columns not used by the model or for info/targets: {extra_features}")

    # Select features in the exact order the model expects
    try:
        X_new = new_df[trained_features].copy()
    except KeyError as e:
         print(f"CRITICAL ERROR: Could not select all required features from new data. Missing: {e}")
         raise

    # Check for NaNs in features before prediction
    if X_new.isnull().any().any():
        nan_cols = X_new.columns[X_new.isnull().any()].tolist()
        warnings.warn(f"WARNING: Features (X_new) for prediction contain NaNs in columns: {nan_cols}. Predictions may be unreliable or fail. Ensure data is processed correctly.")
        # Depending on model tolerance, you might raise an error here:
        # assert not X_new.isnull().any().any(), "Features (X_new) for prediction contain NaNs."


    # Extract Match Info using columns defined in PredictionConfig
    required_info_cols = set(pred_cfg.required_match_info_cols)
    missing_info_cols = required_info_cols - set(new_df.columns)
    assert not missing_info_cols, f"CRITICAL ERROR: New data is missing required match info columns defined in PredictionConfig: {missing_info_cols}"

    match_info = new_df[pred_cfg.required_match_info_cols].copy()

    assert X_new.shape[0] == match_info.shape[0], "CRITICAL ERROR: Mismatch between feature rows and match info rows."
    print(f"Prepared X_new shape: {X_new.shape}")


    # --- 5. Generate Base Probabilities ---
    print("Generating base probabilities...")
    try:
        base_probabilities = model.predict_proba(X_new)
        assert isinstance(base_probabilities, dict), "model.predict_proba() did not return a dictionary."
        print(f"Generated {len(base_probabilities)} probability keys (e.g., {list(base_probabilities.keys())[:5]}...).")
    except Exception as e:
        print(f"CRITICAL ERROR during model prediction: {e}")
        raise

    # --- 6. Format Predictions ---
    print("Formatting final predictions using PredictionConfig...")
    try:
        final_predictions = format_predictions(base_probabilities, match_info, pred_cfg)
        assert isinstance(final_predictions, pd.DataFrame), "'format_predictions' did not return a pandas DataFrame."
    except Exception as e:
        print(f"CRITICAL ERROR during prediction formatting: {e}")
        raise

    # ---> ADDED: Merge Actual Results <---
    output_df = final_predictions
    if actual_results_df is not None:
        print(f"Merging actual results ({target_cols}) with predictions...")
        # Ensure MatchID types match before merge, if necessary (though usually string/object)
        try:
             output_df = pd.merge(
                final_predictions,
                actual_results_df,
                on=match_id_col,
                how='left' # Keep all predictions, add results where available
            )
             print("Merge successful.")
             # Basic validation after merge
             assert len(output_df) == len(final_predictions), "Merge changed the number of prediction rows."
             assert set(target_cols).issubset(output_df.columns), "Target columns are missing after merge."
        except Exception as e:
            warnings.warn(f"WARNING: Failed to merge actual results with predictions. Error: {e}. Saving predictions without results.")
            output_df = final_predictions # Fallback to predictions only
    # ---> END ADDED SECTION <---

    # --- 7. Save Final Data (Predictions + potentially Results) ---
    print(f"Saving final output data to: {output_path}")
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_df.to_csv(output_path, index=False) # Save the (potentially merged) DataFrame
        print(f"Output saved successfully. Shape: {output_df.shape}")
        if not output_df.empty:
            print("Sample Output Data:")
            print(output_df.head())
        else:
            print("No predictions generated meeting criteria.")

    except Exception as e:
        print(f"CRITICAL ERROR saving output data to {output_path}: {e}")
        raise

    print(f"--- Prediction Pipeline Complete ---")

if __name__ == "__main__":
    # --- Define Parameters Directly ---
    BASE_PATH = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888'
   # Model paths (to be created when models are trained)
    MODEL_PATH_NO_ODDS = os.path.join(BASE_PATH, 'models/data/outputs/poisson_no_odds_v2_scaled.joblib')
    MODEL_PATH_WITH_ODDS = os.path.join(BASE_PATH, 'models/data/outputs/poisson_with_odds_v2_scaled.joblib')

    # Data paths (point to the processed data used for prediction)
    # IMPORTANT: For real predictions, change these to NEW processed data.
    # For testing the script itself, using the training data is okay.
    NEW_DATA_PATH_NO_ODDS = os.path.join(BASE_PATH, 'models', 'data', 'outputs', 'processed_without_odds.parquet')
    NEW_DATA_PATH_WITH_ODDS = os.path.join(BASE_PATH, 'models', 'data', 'outputs', 'processed_with_odds.parquet')

    # Output paths for predictions
    OUTPUT_PATH_NO_ODDS = os.path.join(BASE_PATH, 'models/data/outputs/predictions/predictions_no_odds_latest.csv')
    OUTPUT_PATH_WITH_ODDS = os.path.join(BASE_PATH, 'models/data/outputs/predictions/predictions_with_odds_latest.csv')

    MODEL_TYPE_TO_LOAD = "poisson"

    # --- Run Prediction (Example: Using No-Odds Model) ---
    if os.path.exists(MODEL_PATH_NO_ODDS) and os.path.exists(NEW_DATA_PATH_NO_ODDS):
        print(f"\n>>> Running Prediction using NO-Odds Model ({os.path.basename(MODEL_PATH_NO_ODDS)}) <<<")
        try:
            run_prediction(
                model_path=MODEL_PATH_NO_ODDS,
                new_data_path=NEW_DATA_PATH_NO_ODDS,
                output_path=OUTPUT_PATH_NO_ODDS,
                model_type=MODEL_TYPE_TO_LOAD
            )
        except Exception as e:
            print(f"!!! Prediction failed for NO-Odds model: {e} !!!")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
    else:
        print(f"\n>>> Skipping NO-Odds Prediction: Files not found <<<")
        if not os.path.exists(MODEL_PATH_NO_ODDS): print(f"Model missing: {MODEL_PATH_NO_ODDS}")
        if not os.path.exists(NEW_DATA_PATH_NO_ODDS): print(f"Data missing: {NEW_DATA_PATH_NO_ODDS}")

    # --- Run Prediction (Example: Using With-Odds Model) ---
    if os.path.exists(MODEL_PATH_WITH_ODDS) and os.path.exists(NEW_DATA_PATH_WITH_ODDS):
        print(f"\n>>> Running Prediction using WITH-Odds Model ({os.path.basename(MODEL_PATH_WITH_ODDS)}) <<<")
        try:
            run_prediction(
                model_path=MODEL_PATH_WITH_ODDS,
                new_data_path=NEW_DATA_PATH_WITH_ODDS,
                output_path=OUTPUT_PATH_WITH_ODDS,
                model_type=MODEL_TYPE_TO_LOAD
            )
        except Exception as e:
            print(f"!!! Prediction failed for WITH-Odds model: {e} !!!")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
    else:
        print(f"\n>>> Skipping WITH-Odds Prediction: Files not found <<<")
        if not os.path.exists(MODEL_PATH_WITH_ODDS): print(f"Model missing: {MODEL_PATH_WITH_ODDS}")
        if not os.path.exists(NEW_DATA_PATH_WITH_ODDS): print(f"Data missing: {NEW_DATA_PATH_WITH_ODDS}")