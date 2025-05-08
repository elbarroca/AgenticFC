# pipelines/train_pipeline.py
import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib
from pathlib import Path
import time
import traceback
from typing import Dict, Any, Type

# --- Ray Tune Imports ---
# We will import tune and specific algorithms/schedulers after ray.init
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch


# --- Sklearn Imports ---
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_poisson_deviance

# --- Add project root to sys.path if needed ---
import sys
import os
# This assumes the script is run from the project root (AgenticFC888)
# or that the 'models' directory is directly runnable as a package
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
assert str(PROJECT_ROOT_PATH) not in sys.path, "Project root path already in sys.path"
sys.path.append(str(PROJECT_ROOT_PATH))
print(f"Project Root added to sys.path: {PROJECT_ROOT_PATH}")

# --- Model Classes & Config ---
from models.utils.features import BaseFeatureConfig, get_feature_config
from models.ml_models.poisson_model import PoissonModel
from models.ml_models.random_forest_model import RandomForestModel
from models.ml_models.gradient_boosting_model import GradientBoostingModel
from models.ml_models.monte_carlo_model import MonteCarloModel
from models.base_model import BaseModel # Import base for type checking and apply_scaling flag

# --- Model Registry ---
AVAILABLE_MODELS: Dict[str, Type[BaseModel]] = {
    "poisson": PoissonModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "monte_carlo": MonteCarloModel,
}

# --- Configuration ---
BASE_DIR = PROJECT_ROOT_PATH
# *** Paths adjusted for local structure shown in screenshot ***
DATA_OUTPUT_DIR = BASE_DIR / 'models' / 'data' / 'outputs'
PARAMS_OUTPUT_DIR = DATA_OUTPUT_DIR / 'optimized_params'
MODELS_SAVE_DIR = DATA_OUTPUT_DIR / 'joblib' # Save models to joblib subdir
TRANSFORMERS_DIR = DATA_OUTPUT_DIR / 'joblib' # Transformers saved here previously

# *** Load PCA processed data from outputs directory ***
PROCESSED_DATA_PATH_TEMPLATE = str(DATA_OUTPUT_DIR / 'processed_pca_{}.parquet')
# *** Params filename indicates lambda optimization metric (e.g., rmse) ***
OPTIMIZED_PARAMS_PATH_TEMPLATE = str(PARAMS_OUTPUT_DIR / 'best_params_ray_lambda_{}_{}_{}.json') # metric, model, odds
MODEL_OUTPUT_PATH_TEMPLATE = str(MODELS_SAVE_DIR / '{}_pca_{}_v1.joblib') # PCA model

# --- Models to Train & Default Params ---

MODELS_TO_TRAIN_CONFIG = {
    #"poisson": {
    #    "params": {'max_iter': 100000, 'tol': 1e-4}, # Fixed params
    #    "optimize": True,
    #    "search_space": {'alpha': tune.loguniform(1e-5, 1.0)}
    #},
    "random_forest": { # Uncommented Random Forest
        "params": {'n_jobs': -1, 'random_state': 44}, # Fixed params
        "optimize": True,
        "search_space": {
            'n_estimators': tune.randint(100, 801),
            'max_depth': tune.randint(8, 41),
            'min_samples_split': tune.randint(2, 31),
            'min_samples_leaf': tune.randint(3, 21),
            'max_features': tune.uniform(0.1, 0.9),
        }
    },
    "gradient_boosting": {
        "params": {'n_jobs': -1, 'objective': 'poisson', 'metric': 'None', 'random_state': 42}, # Fixed
        "optimize": True,
        "search_space": {
            'learning_rate': tune.loguniform(0.005, 0.2),
            'n_estimators': tune.randint(100, 2001),
            'max_depth': tune.randint(3, 16),
            'num_leaves': tune.randint(8, 129),
            'subsample': tune.uniform(0.5, 1.0),
            'colsample_bytree': tune.uniform(0.4, 1.0),
            'reg_alpha': tune.loguniform(1e-8, 10.0),
            'reg_lambda': tune.loguniform(1e-8, 10.0),
            'min_child_samples': tune.randint(3, 51),
        }
    }
    #},
    #"monte_carlo": { # Monte Carlo still doesn't have tunable params in this setup
    #    "params": {'n_simulations': 10000, 'internal_estimator_alpha': 1.0},
    #    "optimize": False,
    #    "search_space": {}
    #},
}

# --- Ray Tune Settings ---
RUN_OPTIMIZATION: bool = True
RAY_TUNE_N_SAMPLES: int = 10 # Adjust based on time/resources
RAY_TUNE_CV_SPLITS: int = 5
# *** Define the Lambda Metric to Optimize For ***
LAMBDA_OPTIMIZATION_METRIC: str = 'rmse' # Options: 'rmse', 'poisson_deviance'
assert LAMBDA_OPTIMIZATION_METRIC in ['rmse', 'poisson_deviance'], "Invalid LAMBDA_OPTIMIZATION_METRIC"

# Ray Tune Trainable Function (Optimizing Lambda Accuracy)
def trainable_lambda_objective(config_from_tune: Dict, data: Dict):
    """
    Trainable function for Ray Tune. Optimizes based on lambda prediction accuracy.
    Assumes models handle PCA features without internal scaling.
    Reports lambda_rmse or poisson_deviance metric back to Ray Tune.
    """

    # --- Extract data and config ---
    model_type = data["model_type"]
    X_all = data["X"]
    y_all = data["y"]
    feature_cfg = data["feature_cfg"]
    fixed_model_params = data["fixed_params"]
    optimization_metric = data["optimization_metric"]
    # Ensure AVAILABLE_MODELS is accessible or pass ModelClass if needed
    ModelClass = AVAILABLE_MODELS[model_type]

    # --- Combine fixed params with tuned params ---
    current_trial_params = fixed_model_params.copy()
    current_trial_params.update(config_from_tune)

    # --- Target columns ---
    target_hg_col = feature_cfg.target_home_goals
    target_ag_col = feature_cfg.target_away_goals
    assert target_hg_col in y_all.columns and target_ag_col in y_all.columns, "Target goal columns missing."

    # --- Cross-validation loop ---
    cv_losses = []
    tscv = TimeSeriesSplit(n_splits=RAY_TUNE_CV_SPLITS)
    fold_num = 0
    for train_indices, val_indices in tscv.split(X_all):
        fold_num += 1
        if len(val_indices) == 0: continue

        X_train, X_val = X_all.iloc[train_indices], X_all.iloc[val_indices]
        y_train, y_val = y_all.iloc[train_indices], y_all.iloc[val_indices]
        y_true_h_val = y_val[target_hg_col].astype(float)
        y_true_a_val = y_val[target_ag_col].astype(float)

        # Fit and Predict within Assertive Context
        try:
            # *** Instantiate model - ASSUMES NO INTERNAL SCALING ***
            model_fold = ModelClass(model_params=current_trial_params, feature_config=feature_cfg, apply_scaling=False)
            assert hasattr(model_fold, 'fit'), f"Model {model_type} lacks fit method."
            model_fold.fit(X_train, y_train)

            assert hasattr(model_fold, 'predict_proba'), f"Model {model_type} lacks predict_proba method."
            val_pred_dict = model_fold.predict_proba(X_val)
            assert isinstance(val_pred_dict, dict), "predict_proba did not return dict."

            lambda_h_pred = val_pred_dict.get(f"{model_type}_expected_HG")
            lambda_a_pred = val_pred_dict.get(f"{model_type}_expected_AG")
            assert lambda_h_pred is not None, f"'{model_type}_expected_HG' not in pred_dict."
            assert lambda_a_pred is not None, f"'{model_type}_expected_AG' not in pred_dict."

            # --- ADD LAMBDA CLIPPING & NaN HANDLING ---
            MAX_LAMBDA = 15.0  # Define a reasonable maximum for expected goals
            # Handle potential NaNs from model prediction and clip
            lambda_h_pred = np.clip(np.nan_to_num(lambda_h_pred, nan=0.0, posinf=MAX_LAMBDA, neginf=0.0), 0.0, MAX_LAMBDA)
            lambda_a_pred = np.clip(np.nan_to_num(lambda_a_pred, nan=0.0, posinf=MAX_LAMBDA, neginf=0.0), 0.0, MAX_LAMBDA)
            
            # Ensure non-negative values (mostly redundant after clip but a good safeguard)
            lambda_h_pred = np.maximum(lambda_h_pred, 1e-9)
            lambda_a_pred = np.maximum(lambda_a_pred, 1e-9)
            # --- END LAMBDA CLIPPING ---

            valid_mask = ~np.isnan(y_true_h_val) & ~np.isnan(y_true_a_val) & \
                         ~np.isnan(lambda_h_pred) & ~np.isnan(lambda_a_pred) # NaNs in preds handled by nan_to_num & clip

            if valid_mask.sum() == 0:
                cv_losses.append(float('inf')); continue

            if optimization_metric == 'rmse':
                loss_h = root_mean_squared_error(y_true_h_val[valid_mask], lambda_h_pred[valid_mask])
                loss_a = root_mean_squared_error(y_true_a_val[valid_mask], lambda_a_pred[valid_mask])
            elif optimization_metric == 'poisson_deviance':
                lambda_h_pred_clipped = np.maximum(lambda_h_pred[valid_mask], 1e-9)
                lambda_a_pred_clipped = np.maximum(lambda_a_pred[valid_mask], 1e-9)
                loss_h = mean_poisson_deviance(y_true_h_val[valid_mask], lambda_h_pred_clipped)
                loss_a = mean_poisson_deviance(y_true_a_val[valid_mask], lambda_a_pred_clipped)
            else: raise ValueError(f"Unsupported metric: {optimization_metric}")

            fold_loss = (loss_h + loss_a) / 2.0
            if not np.isfinite(fold_loss): fold_loss = float('inf')
            cv_losses.append(fold_loss)

        except Exception as e:
            print(f"CRITICAL WARN: Error objective fold {fold_num} for {model_type}: {e}")
            traceback.print_exc()
            cv_losses.append(float('inf'))

    mean_loss = np.mean([loss for loss in cv_losses if np.isfinite(loss)]) if any(np.isfinite(cv_losses)) else float('inf')
    tune.report({optimization_metric: mean_loss, "done": True})

# Main Training Function (Handles PCA features)
def run_training(
    processed_data_path: Path,
    model_output_path: Path,
    include_odds: bool,
    model_type: str,
    model_params: Dict[str, Any],
    feature_cfg: BaseFeatureConfig
    ):
    """Loads PCA data, trains model (with internal scaling disabled), saves artifact."""
    print(f"\n--- Starting Final Model Training: {model_type} (PCA Features, Odds: {include_odds}) ---")
    print(f"Using PCA data: {processed_data_path}")
    print(f"Outputting model to: {model_output_path}")
    print(f"Using parameters: {model_params}")

    # --- Load PCA Processed Data ---
    assert processed_data_path.exists(), f"Processed PCA data file not found: {processed_data_path}"
    df = pd.read_parquet(processed_data_path, engine='pyarrow')
    assert not df.empty, "Processed PCA DataFrame is empty."

    # --- Prepare Features (PCA) and Targets (Original) ---
    target_cols = [feature_cfg.target_home_goals, feature_cfg.target_away_goals, feature_cfg.target_result]
    feature_cols = [col for col in df.columns if col.startswith('PC')]
    id_col = feature_cfg.match_id_col
    date_col = feature_cfg.date_col # Keep date if needed

    required_cols = set(target_cols) | set(feature_cols) | {id_col, date_col}
    missing_cols = required_cols - set(df.columns)
    assert not missing_cols, f"PCA DataFrame missing required columns: {missing_cols}"
    assert feature_cols, "No PCA feature columns (PC*) found."

    X = df[feature_cols]
    y = df[target_cols]

    assert not X.isnull().any().any(), "NaNs found in PCA features before training."
    assert not y.isnull().any().any(), "NaNs found in targets before training."
    print(f"Prepared X_pca shape: {X.shape}, y shape: {y.shape}")

    # --- Instantiate Model ---
    ModelClass = AVAILABLE_MODELS.get(model_type)
    assert ModelClass is not None, f"Model type '{model_type}' not found."
    print(f"Instantiating {model_type} with apply_scaling=False for PCA features.")
    # *** Instantiate with apply_scaling=False ***
    model = ModelClass(
        model_params=model_params,
        feature_config=feature_cfg,
        apply_scaling=False # Explicitly disable internal scaling for PCA data
    )

    # --- Train Model ---
    print("Starting model fitting on PCA features (internal scaling disabled)...")
    assert hasattr(model, 'fit'), f"Model {model_type} instance lacks a 'fit' method."
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    print(f"Model fitting complete. Time taken: {end_time - start_time:.2f} seconds.")

    # --- Save Model ---
    print(f"Saving trained model to: {model_output_path}")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    assert hasattr(model, 'save'), f"Model {model_type} instance lacks a 'save' method."
    # BaseModel.save should now correctly save scaler=None because apply_scaling=False
    model.save(str(model_output_path))
    print("Trained model saved successfully.")

    print(f"--- Model Training Complete: {model_type} (PCA Features, Odds: {include_odds}) ---")

# Main Execution Block (Using Ray Tune for Lambda Accuracy)
if __name__ == "__main__":
    # Ensure output directories exist
    PARAMS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Initialize Ray ---
    if not ray.is_initialized():
        # Be more explicit with excluding large files
        runtime_excludes = [
            ".git", ".git/**", 
            "**/*.parquet",  # Exclude all parquet files
            "**/*.joblib",   # Exclude all joblib files
            "**/*.csv",      # Exclude large CSV files
            "models/data/outputs/**",  # Exclude all output data
            "models/data/parquets/**", 
            "data/**", 
            "notebooks/**", 
            ".ipynb_checkpoints/**", 
            "**/__pycache__", 
            "*.pyc", 
            "*.log", 
            "logs/**"
        ]
        
        try:
            num_available_cpus = os.cpu_count()
            num_ray_cpus = max(1, num_available_cpus - 1)  # Leave 1 core free
            
            # Use local_mode instead of trying to upload working_dir
            ray.init(
                local_mode=True,
                ignore_reinit_error=True,
                log_to_driver=False
            )
            print("Ray initialized in local_mode to avoid directory size limitations.")
        except Exception as e:
            print(f"CRITICAL: Ray init failed: {e}"); exit(1)

    # --- Phase 1: Hyperparameter Optimization (Ray Tune) ---
    if RUN_OPTIMIZATION:
        print(f"\n===== PHASE 1: Running Hyperparameter Optimization (Ray Tune) - Optimizing for Lambda {LAMBDA_OPTIMIZATION_METRIC.upper()} =====")
        for include_odds in [True, False]:
            odds_suffix = "with_odds" if include_odds else "without_odds"
            print(f"\n--- Optimizing for data: pca_{odds_suffix} ---")

            data_path = Path(PROCESSED_DATA_PATH_TEMPLATE.format(odds_suffix))
            if not data_path.exists():
                 warnings.warn(f"PCA Data file not found, skipping optimization: {data_path}", RuntimeWarning)
                 continue

            print(f"Loading data for optimization: {data_path}")
            df_all = pd.read_parquet(data_path)
            feature_cfg_opt = get_feature_config(include_odds=include_odds)
            target_cols_opt = [feature_cfg_opt.target_home_goals, feature_cfg_opt.target_away_goals, feature_cfg_opt.target_result]
            feature_cols_opt = [col for col in df_all.columns if col.startswith('PC')]
            assert feature_cols_opt, f"No PCA features found in {data_path}"
            X_all_opt = df_all[feature_cols_opt]
            y_all_opt = df_all[target_cols_opt]
            assert not X_all_opt.isnull().any().any(), f"NaNs found in PCA features ({odds_suffix})"
            assert not y_all_opt.isnull().any().any(), f"NaNs found in targets ({odds_suffix})"
            print(f"Data loaded for optimization. X shape: {X_all_opt.shape}")

            # Put data into Ray object store
            print("Placing data in Ray object store...")
            data_for_objective_payload = {"X": X_all_opt, "y": y_all_opt, "feature_cfg": feature_cfg_opt, "optimization_metric": LAMBDA_OPTIMIZATION_METRIC}
            data_ref = ray.put(data_for_objective_payload)
            print("Data placed in object store.")

            for model_key, config_entry in MODELS_TO_TRAIN_CONFIG.items():
                if config_entry["optimize"]:
                    print(f"\nOptimizing {model_key} (Target: {LAMBDA_OPTIMIZATION_METRIC})...")
                    search_space_for_tune = config_entry["search_space"]
                    fixed_params_for_model = config_entry["params"]

                    # *** Use MINIMIZE mode for RMSE/Poisson Deviance ***
                    metric_mode = "min"
                    scheduler = ASHAScheduler(metric=LAMBDA_OPTIMIZATION_METRIC, mode=metric_mode, grace_period=max(1, RAY_TUNE_CV_SPLITS // 2), reduction_factor=2)
                    search_alg = OptunaSearch(metric=LAMBDA_OPTIMIZATION_METRIC, mode=metric_mode)

                    trainable_with_payload = tune.with_parameters(
                        trainable_lambda_objective, # Use the lambda objective function
                        data={ "model_type": model_key, **ray.get(data_ref), "fixed_params": fixed_params_for_model }
                    )

                    # Define where Ray Tune stores its results locally
                    local_ray_results_dir = BASE_DIR / "ray_results"

                    analysis = tune.run(
                        trainable_with_payload,
                        config=search_space_for_tune,
                        num_samples=RAY_TUNE_N_SAMPLES,
                        scheduler=scheduler,
                        search_alg=search_alg,
                        resources_per_trial={"cpu": 1}, # Limit trials to 1 CPU each
                        verbose=1, # Show progress table
                        raise_on_failed_trial=False, # Continue if one trial fails
                        storage_path=str(local_ray_results_dir), # Updated: use storage_path instead of local_dir
                        name=f"tune_lambda_{model_key}_{odds_suffix}" # Experiment name
                    )

                    best_trial = analysis.get_best_trial(metric=LAMBDA_OPTIMIZATION_METRIC, mode=metric_mode, scope="all")
                    final_params_to_save = fixed_params_for_model.copy() # Start with fixed defaults
                    if best_trial and best_trial.config and best_trial.last_result and LAMBDA_OPTIMIZATION_METRIC in best_trial.last_result:
                        best_loss_score = best_trial.last_result[LAMBDA_OPTIMIZATION_METRIC]
                        if np.isfinite(best_loss_score): # Check if the best score is valid
                            best_params_tuned = best_trial.config
                            final_params_to_save.update(best_params_tuned)
                            print(f"  Best {LAMBDA_OPTIMIZATION_METRIC} for {model_key}: {best_loss_score:.4f}")
                            print(f"  Best tuned config for {model_key}: {best_params_tuned}")
                        else:
                             warnings.warn(f"Best trial for {model_key} resulted in non-finite loss ({best_loss_score}). Using defaults.", RuntimeWarning)
                    else:
                        warnings.warn(f"Ray Tune finished without a valid best trial for {model_key}. Using defaults.", RuntimeWarning)
                        # final_params_to_save remains the defaults

                    params_output_path = Path(OPTIMIZED_PARAMS_PATH_TEMPLATE.format(LAMBDA_OPTIMIZATION_METRIC, model_key, odds_suffix))
                    PARAMS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    with open(params_output_path, 'w') as f: json.dump(final_params_to_save, f, indent=4)
                    print(f"  Saved parameters for {model_key} ({odds_suffix}) to {params_output_path}")

                else: # If optimize is False
                    params_output_path = Path(OPTIMIZED_PARAMS_PATH_TEMPLATE.format(LAMBDA_OPTIMIZATION_METRIC, model_key, odds_suffix))
                    PARAMS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    # Save default params even if not optimized, include metric name for consistency
                    with open(params_output_path, 'w') as f: json.dump(config_entry["params"], f, indent=4)
                    print(f"  Saved default parameters for {model_key} ({odds_suffix}) to {params_output_path}")


            del data_ref # Clean up object store reference

        print("\n===== PHASE 1: Optimization Complete =====")
    else:
        print("\n===== Skipping PHASE 1: Optimization =====")


    # --- Phase 2: Final Model Training ---
    print("\n===== PHASE 2: Training Final Models on PCA Features =====")
    for include_odds in [True, False]:
        odds_suffix = "with_odds" if include_odds else "without_odds"
        print(f"\n--- Training models for data: pca_{odds_suffix} ---")

        data_path = Path(PROCESSED_DATA_PATH_TEMPLATE.format(odds_suffix))
        if not data_path.exists():
            warnings.warn(f"PCA Data file not found, skipping training: {data_path}", RuntimeWarning)
            continue

        feature_cfg_train = get_feature_config(include_odds=include_odds)

        for model_key, config_entry in MODELS_TO_TRAIN_CONFIG.items():
            # *** Load params based on the metric used for optimization ***
            params_path = Path(OPTIMIZED_PARAMS_PATH_TEMPLATE.format(LAMBDA_OPTIMIZATION_METRIC, model_key, odds_suffix))
            model_params_final = config_entry["params"].copy() # Default fixed params
            if params_path.exists():
                try:
                    with open(params_path, 'r') as f: loaded_params = json.load(f)
                    # Loaded params contain fixed + tuned, use them directly
                    model_params_final = loaded_params
                    print(f"  Loaded parameters for {model_key} from {params_path}")
                except Exception as e: print(f"  WARN: Failed loading {params_path}, using defaults: {e}")
            else: print(f"  WARN: Params file {params_path} not found for {model_key}, using defaults.")

            model_output_path = Path(MODEL_OUTPUT_PATH_TEMPLATE.format(model_key, f"pca_{odds_suffix}"))
            MODELS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

            run_training(
                processed_data_path=data_path, model_output_path=model_output_path,
                include_odds=include_odds, model_type=model_key,
                model_params=model_params_final, feature_cfg=feature_cfg_train
            )
    print("\n===== PHASE 2: Final Training Complete =====")

    # --- Shutdown Ray ---
    ray.shutdown()
    print("Ray shut down.")