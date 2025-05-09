# pipelines/train_stacker.py
import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib
from pathlib import Path
import time
import traceback
from typing import Dict, List, Tuple, Any

# --- Ray Tune Imports ---
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray import train
from ray.tune import Tuner, TuneConfig, RunConfig

# --- Sklearn & LGBM Imports ---
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

# --- Add project root to sys.path if needed ---
import sys
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
assert str(PROJECT_ROOT_PATH) not in sys.path, f"Project Root already in sys.path: {PROJECT_ROOT_PATH}"
sys.path.append(str(PROJECT_ROOT_PATH))
print(f"Project Root added to sys.path: {PROJECT_ROOT_PATH}")

# --- Import shared utility ---
from models.utils.features import get_feature_config

# --- Configuration ---
BASE_DIR = PROJECT_ROOT_PATH
DATA_OUTPUT_DIR = BASE_DIR / 'models' / 'data' / 'outputs' / 'predictions'
MODELS_SAVE_DIR =  BASE_DIR / 'models' / 'data' / 'outputs' / 'joblib' / 'V1'
STACKER_OUTPUT_DIR = DATA_OUTPUT_DIR / 'stacker_outputs' # Dedicated dir for stacker results

# Input OOF data
OOF_INPUT_PATH_TEMPLATE = str(DATA_OUTPUT_DIR / 'level0_oof_predictions_pca_{}.parquet') # {odds_suffix}
# Output paths
STACKER_MODEL_OUTPUT_PATH_TEMPLATE = str(MODELS_SAVE_DIR / 'stacker_lgbm_lambda_{}_v1.joblib') # {odds_suffix}
STACKER_PARAMS_OUTPUT_PATH_TEMPLATE = str(STACKER_OUTPUT_DIR / 'stacker_best_params_{}.json') # {odds_suffix}
STACKER_IMPORTANCE_OUTPUT_PATH_TEMPLATE = str(STACKER_OUTPUT_DIR / 'stacker_feature_importance_{}.csv') # {odds_suffix}
STACKER_OOF_PREDS_OUTPUT_PATH_TEMPLATE = str(STACKER_OUTPUT_DIR / 'stacker_oof_lambdas_pca_{}.parquet') # {odds_suffix}

# --- Stacker Configuration ---
LEVEL0_MODELS = ["poisson", "random_forest", "gradient_boosting", "monte_carlo"]

# Features for the stacker (start focused, expand based on analysis)
STACKER_FEATURE_SUFFIXES = [
    'expected_HG', 'expected_AG',
    'prob_H', 'prob_D', 'prob_A',
    'prob_O15', 'prob_U15',
    'prob_O25', 'prob_U25',
    'prob_O35', 'prob_U35',
    'prob_O45', 'prob_U45',
    'prob_BTTS_Y', 'prob_BTTS_N',
  
]

# --- Stacker HYPERPARAMETER TUNING Configuration ---
RUN_STACKER_OPTIMIZATION: bool = True
STACKER_RAY_TUNE_N_SAMPLES: int = 50  # INCREASED SAMPLES (adjust based on time)
STACKER_CV_SPLITS: int = 5
STACKER_EARLY_STOPPING_ROUNDS = 50 # Used inside the objective function

# Define the search space for LGBMRegressor stacker parameters
STACKER_SEARCH_SPACE = {
    # Core Tree Structure & Learning
    'learning_rate': tune.loguniform(0.01, 0.1), # Standard range
    'n_estimators': tune.randint(500, 2500), # INCREASED UPPER BOUND (relies on early stopping)
    'num_leaves': tune.randint(10, 60),      # Slightly increased upper bound
    'max_depth': tune.randint(3, 10),        # ADDED max_depth constraint

    # Regularization
    'lambda_l1': tune.loguniform(1e-3, 5.0), # Slightly narrowed range
    'lambda_l2': tune.loguniform(1e-3, 5.0), # Slightly narrowed range
    'min_child_samples': tune.randint(5, 30),# Good range
    # 'min_gain_to_split': tune.loguniform(1e-3, 0.5), # Optional: Add if needed later

    # Subsampling / Stochasticity
    'feature_fraction': tune.uniform(0.6, 0.95), # Good range
    'bagging_fraction': tune.uniform(0.6, 0.95), # Good range
    'bagging_freq': tune.choice([1, 3, 5]),      # Good range

    # Fixed parameters for this task
    'objective': 'poisson',
    'metric': 'rmse', # Note: This is just for reporting TO Ray Tune, not LGBM's internal metric
    'verbose': -1,
    'n_jobs': -1,
    'boosting_type': 'gbdt',
    'seed': tune.sample_from(lambda spec: np.random.randint(0, 10000)), # Good for trial variation
}

# Default parameters if optimization is skipped
DEFAULT_STACKER_PARAMS = {
    'objective': 'poisson', 'n_estimators': 500, 'learning_rate': 0.05,
    'num_leaves': 31, 'max_depth': 7, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
    'boosting_type': 'gbdt', 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'min_child_samples': 20,
}

# --- Helper Function to Prepare Stacker Features ---
def get_stacker_features(oof_df: pd.DataFrame, level0_models: List[str], feature_suffixes: List[str]) -> List[str]:
    """Constructs the list of feature columns for the stacker."""
    stacker_features = []
    for model_prefix in level0_models:
        for suffix in feature_suffixes:
            col_name = f"{model_prefix}_{suffix}"
            if col_name in oof_df.columns:
                stacker_features.append(col_name)
            else:
                warnings.warn(f"Feature column '{col_name}' not found in OOF DataFrame.", UserWarning)
    assert stacker_features, "No stacker features could be constructed!"
    print(f"Constructed {len(stacker_features)} features for the stacker.")
    return stacker_features

# --- Ray Tune Objective Function for Stacker ---
def tune_stacker_objective(config: Dict, data: Dict):
    """
    Objective function for Ray Tune to optimize stacker hyperparameters.
    Uses KFold CV on the OOF data. Predicts lambdas. Reports combined RMSE.
    """
    # Retrieve actual data from Ray object store references
    X_oof = ray.get(data["X_oof"])
    y_oof_hg = ray.get(data["y_oof_hg"])
    y_oof_ag = ray.get(data["y_oof_ag"])
    n_splits = data["n_splits"]
    early_stopping_rounds = data["early_stopping_rounds"]

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=config.get('seed', 42))
    
    # Now KFold.split will work correctly with the actual DataFrame
    fold_rmse_hg = []
    fold_rmse_ag = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_oof)):
        X_train_cv, X_val_cv = X_oof.iloc[train_idx], X_oof.iloc[val_idx]
        y_train_cv_hg, y_val_cv_hg = y_oof_hg.iloc[train_idx], y_oof_hg.iloc[val_idx]
        y_train_cv_ag, y_val_cv_ag = y_oof_ag.iloc[train_idx], y_oof_ag.iloc[val_idx]

        try:
            # Train HG model for this fold
            model_hg = lgb.LGBMRegressor(**config) # Use parameters from Ray Tune trial
            model_hg.fit(X_train_cv, y_train_cv_hg,
                         eval_set=[(X_val_cv, y_val_cv_hg)],
                         eval_metric='rmse',
                         callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
            preds_hg = model_hg.predict(X_val_cv)
            rmse_hg = root_mean_squared_error(y_val_cv_hg, preds_hg)
            fold_rmse_hg.append(rmse_hg)

            # Train AG model for this fold
            # Create a slightly different seed for AG model if seed is in config
            ag_config = config.copy()
            if 'seed' in ag_config: ag_config['seed'] += 1
            model_ag = lgb.LGBMRegressor(**ag_config)
            model_ag.fit(X_train_cv, y_train_cv_ag,
                         eval_set=[(X_val_cv, y_val_cv_ag)],
                         eval_metric='rmse',
                         callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
            preds_ag = model_ag.predict(X_val_cv)
            rmse_ag = root_mean_squared_error(y_val_cv_ag, preds_ag)
            fold_rmse_ag.append(rmse_ag)

        except Exception as e:
            print(f"Error in stacker CV fold {fold+1}: {e}")
            fold_rmse_hg.append(float('inf'))
            fold_rmse_ag.append(float('inf'))
            break # Stop this trial if a fold fails

    # Calculate combined average RMSE across folds
    mean_rmse_hg = np.mean([r for r in fold_rmse_hg if np.isfinite(r)]) if any(np.isfinite(fold_rmse_hg)) else float('inf')
    mean_rmse_ag = np.mean([r for r in fold_rmse_ag if np.isfinite(r)]) if any(np.isfinite(fold_rmse_ag)) else float('inf')
    combined_rmse = (mean_rmse_hg + mean_rmse_ag) / 2.0 if np.isfinite(mean_rmse_hg) and np.isfinite(mean_rmse_ag) else float('inf')

    tune.report({"rmse": combined_rmse, "rmse_hg": mean_rmse_hg, "rmse_ag": mean_rmse_ag})

# --- Main Stacker Training Function (Incorporating Tuning) ---
def train_stacker(include_odds: bool):
    """
    Loads OOF data, tunes stacker hyperparameters using Ray Tune,
    trains final stacker models (LGBMRegressor on lambdas), and saves artifacts.
    """
    odds_suffix = "with_odds" if include_odds else "without_odds"
    print(f"\n===== Training Stacker Model (Predicting Lambdas, {odds_suffix}) =====")

    # --- 1. Load Enhanced OOF Data ---
    oof_path = Path(OOF_INPUT_PATH_TEMPLATE.format(odds_suffix))
    print(f"Loading enhanced OOF data from: {oof_path}")
    assert oof_path.exists(), f"OOF data file not found: {oof_path}"
    oof_df = pd.read_parquet(oof_path, engine='pyarrow')
    assert not oof_df.empty, "Loaded OOF DataFrame is empty."
    print(f"OOF data loaded. Shape: {oof_df.shape}")

    # --- 2. Prepare Features and Targets ---
    feature_cfg = get_feature_config(include_odds=include_odds)
    target_hg = feature_cfg.target_home_goals
    target_ag = feature_cfg.target_away_goals
    assert target_hg in oof_df.columns and target_ag in oof_df.columns, "Target columns missing."

    stacker_feature_cols = get_stacker_features(oof_df, LEVEL0_MODELS, STACKER_FEATURE_SUFFIXES)

    if oof_df[stacker_feature_cols].isnull().any().any():
        warnings.warn("NaNs found in stacker features. Applying simple mean imputation.", UserWarning)
        oof_df[stacker_feature_cols] = oof_df[stacker_feature_cols].fillna(oof_df[stacker_feature_cols].mean())
        assert not oof_df[stacker_feature_cols].isnull().any().any(), "NaNs remain after imputation."

    X_stack = oof_df[stacker_feature_cols]
    y_stack_hg = oof_df[target_hg]
    y_stack_ag = oof_df[target_ag]
    print(f"Stacker features shape: {X_stack.shape}")

    # --- 3. Hyperparameter Tuning (Optional) ---
    best_params = DEFAULT_STACKER_PARAMS.copy() # Start with defaults

    if RUN_STACKER_OPTIMIZATION and ray.is_initialized():
        print("\n--- Running Hyperparameter Optimization for Stacker ---")
        
        # Before running tune.run(), put large data objects in Ray object store
        X_stack_ref = ray.put(X_stack)
        y_stack_hg_ref = ray.put(y_stack_hg)
        y_stack_ag_ref = ray.put(y_stack_ag)

        # Then use the references in your data dictionary
        stacker_tune_data = {
            "X_oof": X_stack_ref, 
            "y_oof_hg": y_stack_hg_ref, 
            "y_oof_ag": y_stack_ag_ref,
            "n_splits": STACKER_CV_SPLITS,
            "early_stopping_rounds": STACKER_EARLY_STOPPING_ROUNDS
        }
        
        # Remove metric and mode from scheduler
        scheduler = ASHAScheduler(grace_period=max(1, STACKER_RAY_TUNE_N_SAMPLES // 5), reduction_factor=2)
        # Metric and mode for OptunaSearch are for its internal optimization, not for Tune's trial reporting/scheduler
        search_alg = OptunaSearch(metric="rmse", mode="min")

        # Define the trainable with parameters
        trainable_with_params = tune.with_parameters(tune_stacker_objective, data=stacker_tune_data)
        
        # Associate resources with the trainable
        trainable_with_resources = tune.with_resources(
            trainable_with_params,
            resources={"cpu": 2, "memory": 4 * 1024 * 1024 * 1024}  # 2 CPUs, 4GB RAM per trial
        )

        tuner = Tuner(
            trainable_with_resources, # Pass the trainable with resources
            param_space=STACKER_SEARCH_SPACE,
            tune_config=TuneConfig(
                num_samples=STACKER_RAY_TUNE_N_SAMPLES,
                scheduler=scheduler, # Scheduler without metric/mode
                search_alg=search_alg,
                metric="rmse", # Metric and mode defined here
                mode="min",
            ),
            run_config=RunConfig(
                name=f"tune_stacker_{odds_suffix}",
                storage_path=str(BASE_DIR / "ray_results_stacker"),
                failure_config=tune.FailureConfig(max_failures=3),
                verbose=1
            )
        )
        results = tuner.fit()

        # Get best results
        best_result = results.get_best_result(metric="rmse", mode="min")
        if best_result:
            print(f"  Best trial found. RMSE: {best_result.metrics.get('rmse', float('inf')):.4f}")
            best_params.update(best_result.config)
            print(f"  Best hyperparameters: {best_result.config}")
        else:
            warnings.warn("Ray Tune for stacker finished without a valid best trial. Using default parameters.", RuntimeWarning)
        
        # Save the best parameters found
        params_save_path = Path(STACKER_PARAMS_OUTPUT_PATH_TEMPLATE.format(odds_suffix))
        params_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_save_path, 'w') as f: json.dump(best_params, f, indent=4)
        print(f"  Saved best stacker parameters to: {params_save_path}")

    else:
        print("\n--- Skipping Hyperparameter Optimization for Stacker ---")
        # Optionally save default params if skipping opt
        params_save_path = Path(STACKER_PARAMS_OUTPUT_PATH_TEMPLATE.format(odds_suffix))
        if not params_save_path.exists():
             params_save_path.parent.mkdir(parents=True, exist_ok=True)
             with open(params_save_path, 'w') as f: json.dump(best_params, f, indent=4)
             print(f"  Saved default stacker parameters to: {params_save_path}")


    # --- 4. Train Final Stacker Models on Full OOF Data using Best Params ---
    print(f"\nTraining final stacker models using {'TUNED' if RUN_STACKER_OPTIMIZATION else 'DEFAULT'} parameters...")
    print(f"Final parameters: {best_params}")

    # Ensure 'metric' and other non-fit params are removed if they cause issues with direct fit
    final_fit_params = best_params.copy()
    final_fit_params.pop('metric', None) # metric is for eval_metric, not direct fit param

    # Final Home Goal Model
    print("  Fitting final HG model...")
    final_model_hg = lgb.LGBMRegressor(**final_fit_params)
    final_model_hg.fit(X_stack, y_stack_hg) # No early stopping on final fit
    print("  Final HG model fitted.")

    # Final Away Goal Model (use slightly different seed if available)
    print("  Fitting final AG model...")
    ag_fit_params = final_fit_params.copy()
    if 'seed' in ag_fit_params: ag_fit_params['seed'] += 1
    final_model_ag = lgb.LGBMRegressor(**ag_fit_params)
    final_model_ag.fit(X_stack, y_stack_ag)
    print("  Final AG model fitted.")

    # --- 5. Feature Importance Analysis ---
    print("\n--- Stacker Feature Importance ---")
    importance_hg = pd.DataFrame({'feature': stacker_feature_cols, 'importance_hg': final_model_hg.feature_importances_})
    importance_ag = pd.DataFrame({'feature': stacker_feature_cols, 'importance_ag': final_model_ag.feature_importances_})
    importance_df = pd.merge(importance_hg, importance_ag, on='feature', how='outer')
    importance_df['importance_total'] = importance_df['importance_hg'].fillna(0) + importance_df['importance_ag'].fillna(0)
    importance_df = importance_df.sort_values(by='importance_total', ascending=False).reset_index(drop=True)

    print("Top 15 Features (Total Importance):")
    print(importance_df.head(15))

    # Save importance
    importance_path = Path(STACKER_IMPORTANCE_OUTPUT_PATH_TEMPLATE.format(odds_suffix))
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_path, index=False)
    print(f"  Saved feature importance to: {importance_path}")

    # --- 6. Save Final Stacker Artifact ---
    stacker_artifact = {
        'model_hg': final_model_hg,
        'model_ag': final_model_ag,
        'feature_columns': stacker_feature_cols,
        'level0_models': LEVEL0_MODELS,
        'feature_suffixes': STACKER_FEATURE_SUFFIXES,
        'training_options': {'include_odds': include_odds},
        'tuned_parameters': best_params if RUN_STACKER_OPTIMIZATION else None # Store tuned params
    }
    model_output_path = Path(STACKER_MODEL_OUTPUT_PATH_TEMPLATE.format(odds_suffix))
    print(f"\nSaving final stacker artifact (HG & AG models) to: {model_output_path}")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(stacker_artifact, model_output_path)
    print("Stacker artifact saved successfully.")

    # --- 7. Optional: Generate and Save Stacker OOF Predictions from CV ---
    # This requires running predict within the CV loop and storing results
    # Modify tune_stacker_objective if you need this for calibration/analysis
    # For now, we focus on training the final model.

    print(f"===== Stacker Training Complete ({odds_suffix}) =====")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Stacker Model Training Process...")
    start_time = time.time()

    # --- Initialize Ray (Needed for tuning) ---
    if RUN_STACKER_OPTIMIZATION and not ray.is_initialized():
        print("Initializing Ray for Stacker Tuning...")
        try:
            ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)
            print("Ray initialized successfully.")
        except Exception as e:
            print(f"CRITICAL: Ray init failed: {e}. Stacker optimization will be skipped.")
            RUN_STACKER_OPTIMIZATION = False # Disable optimization if Ray fails

    # --- Ensure output directory exists ---
    STACKER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Train stacker for both odds settings ---
    train_stacker(include_odds=False)
    train_stacker(include_odds=True)

    # --- Shutdown Ray ---
    if ray.is_initialized():
        print("\nShutting down Ray...")
        ray.shutdown()

    end_time = time.time()
    print(f"\nStacker Training Process Finished. Total time: {end_time - start_time:.2f} seconds.")