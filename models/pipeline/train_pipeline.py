# pipelines/train_pipeline.py
import pandas as pd
import numpy as np
import os
import joblib # Keep for fallback saving if needed
import json # Keep for potential future param loading
import inspect # To check model __init__ signature
from models.utils.poisson_model import PoissonModel
from models.utils.features import BaseFeatureConfig, get_feature_config

# --- Model Registry (Simple version) ---
AVAILABLE_MODELS = {
    "poisson": PoissonModel,
    # "xgboost": XGBoostModel, # Example
}

def run_training(
    processed_data_path: str,
    model_output_path: str,
    include_odds: bool,
    model_type: str = "poisson",
    model_params: dict = None # Allow passing params directly
    ):
    """
    Loads processed data, trains a model (which now handles internal scaling),
    and saves the artifact (including the scaler).

    Args:
        processed_data_path: Path to the input processed parquet file.
        model_output_path: Path to save the trained model (.joblib).
        include_odds: Boolean flag whether odds were included in the data.
        model_type: String identifier for the model type (e.g., 'poisson').
        model_params: Dictionary of hyperparameters for the model. If None, uses defaults.
    """
    print(f"\n--- Starting Model Training ---")
    print(f"Processed data path: {processed_data_path}")
    print(f"Model output path: {model_output_path}")
    print(f"Include odds: {include_odds}")
    print(f"Model type: {model_type}")

    # --- 1. Load Configuration ---
    try:
        feature_cfg: BaseFeatureConfig = get_feature_config(include_odds=include_odds)
        print("Feature configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading feature configuration: {e}")
        raise

    # --- 2. Load Processed Data ---
    print(f"Loading processed data from: {processed_data_path}")
    try:
        df = pd.read_parquet(processed_data_path, engine='pyarrow')
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Processed data file not found at {processed_data_path}")
        raise
    except Exception as e:
        print(f"CRITICAL ERROR loading processed parquet file: {e}")
        raise

    print(f"Loaded processed data shape: {df.shape}")
    assert not df.empty, "Processed DataFrame is empty."

    # --- 3. Prepare Features (X) and Targets (y) ---
    print("Preparing features (X) and targets (y)...")
    # Get target names safely from config
    target_home = getattr(feature_cfg, 'target_home_goals', 'FTHG') # Use actual default
    target_away = getattr(feature_cfg, 'target_away_goals', 'FTAG') # Use actual default
    target_cols = [target_home, target_away]

    missing_targets = set(target_cols) - set(df.columns)
    assert not missing_targets, f"Processed data missing target columns: {missing_targets}"
    y = df[target_cols].copy() # Use copy

    # Get feature columns safely from config
    try:
        feature_cols = feature_cfg.get_feature_columns(include_odds=include_odds)
    except AttributeError as e:
        print(f"Error: feature_cfg does not have 'get_feature_columns': {e}")
        raise NotImplementedError("Need 'get_feature_columns' method in feature config.")

    missing_features = set(feature_cols) - set(df.columns)
    assert not missing_features, f"Processed data missing feature columns: {missing_features}"
    X = df[feature_cols].copy() # Use copy

    # Final checks before training
    assert X.shape[0] == y.shape[0], f"Row mismatch between X ({X.shape[0]}) and y ({y.shape[0]})."
    assert not X.isnull().any().any(), "Features (X) contain NaNs before training. Check data_processing step."
    assert not y.isnull().any().any(), "Targets (y) contain NaNs before training."
    print(f"Prepared X shape: {X.shape}, y shape: {y.shape}")

    # --- 4. Instantiate Model ---
    print(f"Instantiating model: {model_type}")
    ModelClass = AVAILABLE_MODELS.get(model_type)
    assert ModelClass is not None, f"Model type '{model_type}' not found in AVAILABLE_MODELS."

    # --- Define default parameters (with increased max_iter) ---
    if model_params is None:
        if model_type == "poisson":
            # Increased max_iter to help convergence with potentially unscaled/complex features
            model_params = {'alpha': 1e-5, 'max_iter': 500000, 'tol': 1e-4} # <-- INCREASED max_iter
            print("Using default Poisson parameters with increased max_iter=5000.")
        else:
            model_params = {} # Add defaults for other models
    print(f"Using model parameters: {model_params}")

    try:
        # Pass feature_config to model constructor as it's needed by PoissonModel's __init__
        # The inspect logic handles models that might not need it
        sig = inspect.signature(ModelClass.__init__)
        if 'feature_config' in sig.parameters:
             model = ModelClass(model_params=model_params, feature_config=feature_cfg)
        else:
             # This path likely won't be taken for PoissonModel, but good for flexibility
             print(f"Warning: {ModelClass.__name__} does not accept 'feature_config' in __init__. Ensure it sets self.feature_config if needed.")
             model = ModelClass(model_params=model_params)
             # Manually set config if needed AFTER init for save/load consistency,
             # though ideally the model sets it internally if required.
             if not hasattr(model, 'feature_config') or model.feature_config is None:
                  model.feature_config = feature_cfg

    except Exception as e:
        print(f"CRITICAL ERROR instantiating model {model_type}: {e}")
        raise

    # --- 5. Train Model ---
    print("Starting model fitting (includes internal scaling)...")
    try:
        # Call the public fit method. It handles scaling internally now.
        model.fit(X, y)
        print("Model fitting complete.")
    except Exception as e:
        print(f"CRITICAL ERROR during model fitting: {e}")
        raise

    # --- 6. Save Model ---
    print(f"Saving trained model to: {model_output_path}")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(model_output_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
        # Use the save method from BaseModel (which now saves the scaler too)
        assert hasattr(model, 'save') and callable(model.save), "Model object lacks a callable 'save' method."
        model.save(model_output_path)
        print("Trained model (including scaler and config) saved successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR saving trained model to {model_output_path}: {e}")
        raise

    print(f"--- Model Training Complete ---")


if __name__ == "__main__":
    # --- Define Parameters Directly ---
    # Base path setup (adjust if needed)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Assumes pipelines is in models/pipeline/

    # Input Data Paths
    PROCESSED_DATA_PATH_NO_ODDS = os.path.join(BASE_DIR, 'models', 'data', 'outputs', 'processed_without_odds.parquet')
    PROCESSED_DATA_PATH_WITH_ODDS = os.path.join(BASE_DIR, 'models', 'data', 'outputs', 'processed_with_odds.parquet')

    # --- Output Model Paths (Use new version names) ---
    MODEL_OUTPUT_PATH_NO_ODDS = os.path.join(BASE_DIR, 'models', 'data', 'outputs', 'poisson_no_odds_v2_scaled.joblib')
    MODEL_OUTPUT_PATH_WITH_ODDS = os.path.join(BASE_DIR, 'models', 'data', 'outputs', 'poisson_with_odds_v2_scaled.joblib')

    MODEL_TYPE_TO_TRAIN = "poisson"
    # Define specific model parameters here to override defaults if needed
    # Example: CUSTOM_MODEL_PARAMS = {'alpha': 1e-6, 'max_iter': 7000}
    CUSTOM_MODEL_PARAMS = None # Set to None to use the defaults defined in run_training

    # --- Run Training (Without Odds) ---
    print("\n>>> Running Training WITHOUT Odds (v2 - Scaled) <<<")
    if os.path.exists(PROCESSED_DATA_PATH_NO_ODDS):
        run_training(
            processed_data_path=PROCESSED_DATA_PATH_NO_ODDS,
            model_output_path=MODEL_OUTPUT_PATH_NO_ODDS,
            include_odds=False,
            model_type=MODEL_TYPE_TO_TRAIN,
            model_params=CUSTOM_MODEL_PARAMS
        )
    else:
        print(f"Skipping NO-Odds training: Input data not found at {PROCESSED_DATA_PATH_NO_ODDS}")

    # --- Run Training (With Odds) ---
    print("\n>>> Running Training WITH Odds (v2 - Scaled) <<<")
    if os.path.exists(PROCESSED_DATA_PATH_WITH_ODDS):
        run_training(
            processed_data_path=PROCESSED_DATA_PATH_WITH_ODDS,
            model_output_path=MODEL_OUTPUT_PATH_WITH_ODDS,
            include_odds=True,
            model_type=MODEL_TYPE_TO_TRAIN,
            model_params=CUSTOM_MODEL_PARAMS
        )
    else:
        print(f"Skipping WITH-Odds training: Input data not found at {PROCESSED_DATA_PATH_WITH_ODDS}")