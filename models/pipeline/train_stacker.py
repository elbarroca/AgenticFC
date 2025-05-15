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
import argparse
import gc  # For memory management
from typing import Dict, List, Tuple, Any

# --- Ray Tune Imports ---
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import Tuner, TuneConfig, RunConfig

# --- Sklearn & LGBM Imports ---
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

# --- Add project root to sys.path if needed ---
import sys
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH) not in sys.path:  # Check before appending
    sys.path.append(str(PROJECT_ROOT_PATH))
    print(f"Project Root added to sys.path: {PROJECT_ROOT_PATH}")

# --- Import shared utility ---
try:
    from models.utils.features import get_feature_config
except ImportError as e:
    print(f"ERROR importing get_feature_config: {e}. Ensure models/utils/features.py is accessible.")
    sys.exit(1)

# --- Configuration ---
BASE_DIR = PROJECT_ROOT_PATH
DATA_OUTPUT_DIR = BASE_DIR / 'models' / 'data' / 'outputs'
PREDICTIONS_DIR = DATA_OUTPUT_DIR / 'predictions'  # L0 OOFs are here
MODELS_ARTIFACTS_DIR = DATA_OUTPUT_DIR / 'joblib'  # For saving final models
STACKER_SPECIFIC_OUTPUTS_DIR = DATA_OUTPUT_DIR / 'stacker_outputs' 

# Input L0 OOF data (expects distinct PCA/NonPCA prefixes if applicable)
# This template should point to the output of your generate_oof_predictions.py
# e.g., level0_oof_predictions_pca_L0_focused_vFinal_without_odds.parquet
OOF_VERSION_SUFFIX = "_L0_focused_vFinal"
L0_OOF_INPUT_PATH_TEMPLATE = str(PREDICTIONS_DIR / 'level0_oof_predictions{L0_OOF_TAG}{OOF_VERSION_SUFFIX}_{ODDS_SUFFIX}.parquet')

# Output paths for the Stacker
STACKER_VERSION = "_v2.1_focused"  # Version for this stacker run
STACKER_MODEL_OUTPUT_PATH_TEMPLATE = str(MODELS_ARTIFACTS_DIR / f'stacker_lgbm_lambda_{{ODDS_SUFFIX}}{STACKER_VERSION}.joblib')
STACKER_PARAMS_OUTPUT_PATH_TEMPLATE = str(STACKER_SPECIFIC_OUTPUTS_DIR / f'stacker_best_params_{{ODDS_SUFFIX}}{STACKER_VERSION}.json')
STACKER_IMPORTANCE_OUTPUT_PATH_TEMPLATE = str(STACKER_SPECIFIC_OUTPUTS_DIR / f'stacker_feature_importance_{{ODDS_SUFFIX}}{STACKER_VERSION}.csv')
STACKER_OOF_LAMBDAS_OUTPUT_PATH_TEMPLATE = str(STACKER_SPECIFIC_OUTPUTS_DIR / f'stacker_oof_lambdas{{L0_OOF_TAG}}_{{ODDS_SUFFIX}}{STACKER_VERSION}.parquet')

# --- Stacker Configuration ---
# These prefixes MUST EXACTLY MATCH the column prefixes in your L0 OOF file
# Example for a PCA-based L0 OOF:
LEVEL0_MODELS_CONFIG = {
    "pca_without_odds": [  # Key to identify the L0 OOF variant
        "poisson_pca_pca_without_odds",
        "gradient_boosting_pca_pca_without_odds",
        "monte_carlo_enhanced_pca_pca_without_odds"
    ],
    "nonpca_without_odds": [  # If you train a stacker on non-pca L0 features
        "poisson_nonpca_without_odds",
        "gradient_boosting_nonpca_without_odds",
        "monte_carlo_enhanced_nonpca_without_odds"
    ]
    # Add entries for "with_odds" if you run those
}

# Features for the stacker (suffixes to append to LEVEL0_MODELS prefixes)
STACKER_FEATURE_SUFFIXES = [
    'expected_HG', 'expected_AG', 'prob_H', 'prob_D', 'prob_A', 'prob_1X', 'prob_12', 'prob_X2',
    'prob_O15', 'prob_U15', 'prob_O25', 'prob_U25', 'prob_O35', 'prob_U35', 'prob_O45', 'prob_U45',
    'prob_BTTS_Y', 'prob_BTTS_N', 'prob_goals_0_1', 'prob_goals_2_3', 'prob_goals_2_4', 'prob_goals_3_plus',
    'prob_H_and_O15', 'prob_D_and_O15', 'prob_A_and_O15', 'prob_H_and_U15', 'prob_D_and_U15', 'prob_A_and_U15',
    'prob_H_and_O25', 'prob_D_and_O25', 'prob_A_and_O25', 'prob_H_and_U25', 'prob_D_and_U25', 'prob_A_and_U25',
    'prob_H_and_O35', 'prob_D_and_O35', 'prob_A_and_O35', 'prob_H_and_U35', 'prob_D_and_U35', 'prob_A_and_U35',
    'prob_H_and_O45', 'prob_D_and_O45', 'prob_A_and_O45', 'prob_H_and_U45', 'prob_D_and_U45', 'prob_A_and_U45',
    'prob_1X_and_O15', 'prob_12_and_O15', 'prob_X2_and_O15', 'prob_1X_and_U15', 'prob_12_and_U15', 'prob_X2_and_U15',
    'prob_1X_and_O25', 'prob_12_and_O25', 'prob_X2_and_O25', 'prob_1X_and_U25', 'prob_12_and_U25', 'prob_X2_and_U25',
    'prob_1X_and_O35', 'prob_12_and_O35', 'prob_X2_and_O35', 'prob_1X_and_U35', 'prob_12_and_U35', 'prob_X2_and_U35',
    'prob_1X_and_O45', 'prob_12_and_O45', 'prob_X2_and_O45', 'prob_1X_and_U45', 'prob_12_and_U45', 'prob_X2_and_U45',
    'prob_H_and_BTTS_Y', 'prob_D_and_BTTS_Y', 'prob_A_and_BTTS_Y', 'prob_H_and_BTTS_N', 'prob_D_and_BTTS_N', 'prob_A_and_BTTS_N',
    'prob_1X_and_BTTS_Y', 'prob_12_and_BTTS_Y', 'prob_X2_and_BTTS_Y', 'prob_1X_and_BTTS_N', 'prob_12_and_BTTS_N', 'prob_X2_and_BTTS_N',
    'prob_O25_and_BTTS_Y', 'prob_O25_and_BTTS_N', 'prob_O35_and_BTTS_Y', 'prob_O35_and_BTTS_N',
]

RUN_STACKER_OPTIMIZATION: bool = True
STACKER_RAY_TUNE_N_SAMPLES: int = 10
STACKER_CV_SPLITS_FOR_TUNING: int = 5  # CV splits for hyperparameter tuning
STACKER_CV_SPLITS_FOR_OOF: int = 8    # CV splits for final OOF generation (can be same or different)
STACKER_EARLY_STOPPING_ROUNDS_TUNE = 50
# Note: Early stopping is not typically used when training final models on full data or full OOF folds

STACKER_SEARCH_SPACE = {  # Same as your good search space
    'learning_rate': tune.loguniform(0.01, 0.1), 
    'n_estimators': tune.randint(500, 2500),
    'num_leaves': tune.randint(10, 60), 
    'max_depth': tune.randint(3, 10),
    'lambda_l1': tune.loguniform(1e-3, 5.0), 
    'lambda_l2': tune.loguniform(1e-3, 5.0),
    'min_child_samples': tune.randint(5, 30), 
    'feature_fraction': tune.uniform(0.6, 0.95),
    'bagging_fraction': tune.uniform(0.6, 0.95), 
    'bagging_freq': tune.choice([1, 3, 5]),
    'objective': 'poisson', 
    'metric': 'rmse', 
    'verbose': -1, 
    'n_jobs': -1,
    'boosting_type': 'gbdt', 
    'seed': tune.sample_from(lambda spec: np.random.randint(0, 10000)),
}

DEFAULT_STACKER_PARAMS = {  # Your good defaults
    'objective': 'poisson', 
    'n_estimators': 1004, 
    'learning_rate': 0.007579,
    'num_leaves': 86, 
    'max_depth': 9, 
    'verbose': -1, 
    'n_jobs': -1, 
    'random_state': 42,  # use random_state for LGBM
    'boosting_type': 'gbdt', 
    'subsample': 0.6686, 
    'colsample_bytree': 0.8927,
    'reg_alpha': 0.4498, 
    'reg_lambda': 2.9288, 
    'min_child_samples': 17,
}

# --- Helper Functions ---
def get_stacker_features(oof_df: pd.DataFrame, level0_model_prefixes: List[str], feature_suffixes: List[str]) -> List[str]:
    """Constructs the list of feature columns for the stacker from L0 OOF data."""
    stacker_features = []
    print(f"DEBUG: OOF DF columns for get_stacker_features (first 10): {oof_df.columns.tolist()[:10]}")
    for model_prefix in level0_model_prefixes:
        print(f"DEBUG: Checking prefix: {model_prefix}")
        for suffix in feature_suffixes:
            col_name = f"{model_prefix}_{suffix}"
            if col_name in oof_df.columns:
                stacker_features.append(col_name)
            else:
                warnings.warn(f"Feature column '{col_name}' not found in OOF DataFrame.", UserWarning, stacklevel=2)
    if not stacker_features:  # Raise error instead of assert for clarity
        raise ValueError("No stacker features could be constructed! Check LEVEL0_MODELS prefixes and OOF column names.")
    print(f"Constructed {len(stacker_features)} features for the stacker.")
    return stacker_features

def tune_stacker_objective(config: Dict, data: Dict) -> None:
    """
    Objective function for Ray Tune to optimize stacker hyperparameters.
    Uses KFold CV on the OOF data. Predicts lambdas. Reports combined RMSE.
    
    Args:
        config: Hyperparameters to evaluate
        data: Dictionary containing training data references
    """
    X_oof = ray.get(data["X_oof"])
    y_oof_hg = ray.get(data["y_oof_hg"])
    y_oof_ag = ray.get(data["y_oof_ag"])
    n_splits = data["n_splits"]
    early_stopping_rounds = data["early_stopping_rounds"]
    
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=config.get('seed', 42))
    fold_rmse_hg, fold_rmse_ag = [], []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_oof)):
        try:
            # Train home goals model
            model_hg = lgb.LGBMRegressor(**config)
            model_hg.fit(
                X_oof.iloc[train_idx], 
                y_oof_hg.iloc[train_idx],
                eval_set=[(X_oof.iloc[val_idx], y_oof_hg.iloc[val_idx])],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
            preds_hg = model_hg.predict(X_oof.iloc[val_idx])
            rmse_hg = root_mean_squared_error(y_oof_hg.iloc[val_idx], preds_hg)
            fold_rmse_hg.append(rmse_hg)
            
            # Train away goals model with slightly modified config
            ag_config = config.copy()
            if 'seed' in ag_config and ag_config['seed'] is not None:
                ag_config['seed'] += 1
                
            model_ag = lgb.LGBMRegressor(**ag_config)
            model_ag.fit(
                X_oof.iloc[train_idx], 
                y_oof_ag.iloc[train_idx],
                eval_set=[(X_oof.iloc[val_idx], y_oof_ag.iloc[val_idx])],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
            preds_ag = model_ag.predict(X_oof.iloc[val_idx])
            rmse_ag = root_mean_squared_error(y_oof_ag.iloc[val_idx], preds_ag)
            fold_rmse_ag.append(rmse_ag)
            
        except Exception:
            fold_rmse_hg.append(float('inf'))
            fold_rmse_ag.append(float('inf'))
            break
    
    # Calculate combined average RMSE across folds
    mean_rmse_hg = np.mean([r for r in fold_rmse_hg if np.isfinite(r)]) if any(np.isfinite(fold_rmse_hg)) else float('inf')
    mean_rmse_ag = np.mean([r for r in fold_rmse_ag if np.isfinite(r)]) if any(np.isfinite(fold_rmse_ag)) else float('inf')
    combined_rmse = (mean_rmse_hg + mean_rmse_ag) / 2.0 if np.isfinite(mean_rmse_hg) and np.isfinite(mean_rmse_ag) else float('inf')
    
    tune.report({"rmse": combined_rmse, "rmse_hg": mean_rmse_hg, "rmse_ag": mean_rmse_ag})

def train_stacker(l0_oof_variant_key: str, include_odds: bool) -> None:
    """
    Train a stacker model on Level 0 OOF predictions.
    
    Args:
        l0_oof_variant_key: Key identifying which L0 model set to use (e.g., 'pca_without_odds')
        include_odds: Whether the L0 models included odds features
    """
    odds_suffix = "with_odds" if include_odds else "without_odds"
    
    # Determine L0 OOF tag (e.g., _pca, _nonpca, or _combined_pca_nonpca) based on l0_oof_variant_key
    l0_oof_tag = ""  # Default if key doesn't specify
    if "pca" in l0_oof_variant_key.lower() and "nonpca" in l0_oof_variant_key.lower():
        l0_oof_tag = "_combined_pca_nonpca"
    elif "pca" in l0_oof_variant_key.lower():
        l0_oof_tag = "_pca"
    elif "nonpca" in l0_oof_variant_key.lower():
        l0_oof_tag = "_nonpca"

    print(f"\n===== Training Stacker (L0 OOF: {l0_oof_variant_key}, Odds: {odds_suffix}) =====")

    # --- 1. Load L0 OOF Data ---
    if l0_oof_variant_key == "pca_without_odds":
        oof_path = PREDICTIONS_DIR / f"level0_oof_predictions_pca{OOF_VERSION_SUFFIX}_{odds_suffix}.parquet"
        current_level0_models = LEVEL0_MODELS_CONFIG["pca_without_odds"]
    elif l0_oof_variant_key == "nonpca_without_odds":
        oof_path = PREDICTIONS_DIR / f"level0_oof_predictions_nonpca{OOF_VERSION_SUFFIX}_{odds_suffix}.parquet"
        current_level0_models = LEVEL0_MODELS_CONFIG["nonpca_without_odds"]
    # Add more elif for combined OOFs or other variants
    else:
        raise ValueError(f"Unknown l0_oof_variant_key: {l0_oof_variant_key}")

    print(f"Loading L0 OOF data from: {oof_path}")
    assert oof_path.exists(), f"L0 OOF data file not found: {oof_path}"
    oof_df = pd.read_parquet(oof_path, engine='pyarrow')
    print(f"L0 OOF data loaded. Shape: {oof_df.shape}")

    feature_cfg = get_feature_config(include_odds=include_odds)
    target_hg_col = feature_cfg.target_home_goals
    target_ag_col = feature_cfg.target_away_goals
    match_id_col = feature_cfg.match_id_col
    date_col = feature_cfg.date_col
    
    assert all(c in oof_df.columns for c in [target_hg_col, target_ag_col, match_id_col, date_col]), \
           "Core columns missing from L0 OOF."

    # --- 2. Prepare Stacker Features ---
    stacker_feature_cols = get_stacker_features(oof_df, current_level0_models, STACKER_FEATURE_SUFFIXES)
    X_stack = oof_df[stacker_feature_cols].copy()
    
    # Handle missing values in stacker features
    if X_stack.isnull().any().any():
        warnings.warn("NaNs found in stacker features. Applying mean imputation.")
        X_stack = X_stack.fillna(X_stack.mean())
        if X_stack.isnull().any().any():
            X_stack = X_stack.fillna(0)
            
    y_stack_hg = oof_df[target_hg_col]
    y_stack_ag = oof_df[target_ag_col]
    print(f"Stacker features shape: {X_stack.shape}")

    # --- 3. Hyperparameter Tuning ---
    best_params_from_tune = DEFAULT_STACKER_PARAMS.copy()
    if RUN_STACKER_OPTIMIZATION and ray.is_initialized():
        print("\n--- Running Hyperparameter Optimization for Stacker ---")
        
        # Put data in Ray object store to avoid serialization overhead
        X_stack_ref = ray.put(X_stack)
        y_stack_hg_ref = ray.put(y_stack_hg)
        y_stack_ag_ref = ray.put(y_stack_ag)
        
        stacker_tune_data = {
            "X_oof": X_stack_ref, 
            "y_oof_hg": y_stack_hg_ref, 
            "y_oof_ag": y_stack_ag_ref, 
            "n_splits": STACKER_CV_SPLITS_FOR_TUNING, 
            "early_stopping_rounds": STACKER_EARLY_STOPPING_ROUNDS_TUNE
        }
        
        # Setup Ray Tune search algorithm and scheduler
        scheduler = ASHAScheduler(
            grace_period=max(1, STACKER_RAY_TUNE_N_SAMPLES // 5),
            reduction_factor=2
        )
        search_alg = OptunaSearch(metric="rmse", mode="min")
        
        # Configure trainable function with data and resources
        trainable_with_params = tune.with_parameters(tune_stacker_objective, data=stacker_tune_data)
        trainable_with_resources = tune.with_resources(
            trainable_with_params,
            resources={"cpu": 6}
        )
        
        # Run hyperparameter tuning
        tuner = Tuner(
            trainable_with_resources,
            param_space=STACKER_SEARCH_SPACE,
            tune_config=TuneConfig(
                num_samples=STACKER_RAY_TUNE_N_SAMPLES,
                scheduler=scheduler,
                search_alg=search_alg,
                metric="rmse",
                mode="min"
            ),
            run_config=RunConfig(
                name=f"tune_stacker_{l0_oof_variant_key}_{odds_suffix}",
                storage_path=str(BASE_DIR / "ray_results_stacker"),
                failure_config=tune.FailureConfig(max_failures=3),
                verbose=1
            )
        )
        
        results = tuner.fit()
        best_result = results.get_best_result(metric="rmse", mode="min")
        
        # Extract best parameters from tuning
        if best_result and best_result.config:
            print(f"  Best trial found. RMSE: {best_result.metrics.get('rmse', float('inf')):.4f}")
            best_params_from_tune.update(best_result.config)
            print(f"  Best hyperparameters: {best_result.config}")
        else:
            warnings.warn("Ray Tune for stacker finished without valid best trial. Using default parameters for final model & OOF.", RuntimeWarning)
        
        # Save best parameters
        params_save_path = Path(STACKER_PARAMS_OUTPUT_PATH_TEMPLATE.format(ODDS_SUFFIX=odds_suffix))
        params_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_save_path, 'w') as f:
            json.dump(best_params_from_tune, f, indent=4)
        print(f"  Saved best stacker parameters to: {params_save_path}")
        
        # Clean up Ray references to free memory
        del X_stack_ref, y_stack_hg_ref, y_stack_ag_ref
        gc.collect()
    else:
        print("\n--- Skipping Hyperparameter Optimization for Stacker ---")
        params_save_path = Path(STACKER_PARAMS_OUTPUT_PATH_TEMPLATE.format(ODDS_SUFFIX=odds_suffix))
        if not params_save_path.exists():
            params_save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(params_save_path, 'w') as f:
                json.dump(best_params_from_tune, f, indent=4)
            print(f"  Saved default stacker parameters to: {params_save_path}")

    # --- 4. Generate True Stacker OOF Predictions ---
    print(f"\n--- Generating True Stacker OOF Lambda Predictions using {STACKER_CV_SPLITS_FOR_OOF}-Fold CV ---")
    oof_stacker_preds_hg = np.zeros(len(X_stack))
    oof_stacker_preds_ag = np.zeros(len(X_stack))
    
    final_model_fit_params = best_params_from_tune.copy()
    final_model_fit_params.pop('metric', None)  # Not an LGBM fit param
    
    # Ensure 'random_state' is used if 'seed' was tuned, or use a fixed one
    if 'seed' in final_model_fit_params:
        final_model_fit_params['random_state'] = final_model_fit_params.pop('seed')

    # Generate OOF predictions using CV
    cv_for_oof = KFold(n_splits=STACKER_CV_SPLITS_FOR_OOF, 
                      shuffle=True, 
                      random_state=final_model_fit_params.get('random_state', 42))

    for fold, (train_idx, val_idx) in enumerate(cv_for_oof.split(X_stack)):
        print(f"  Stacker OOF - Processing Fold {fold + 1}/{STACKER_CV_SPLITS_FOR_OOF}")
        X_train_cv, X_val_cv = X_stack.iloc[train_idx], X_stack.iloc[val_idx]
        y_train_cv_hg = y_stack_hg.iloc[train_idx]
        y_train_cv_ag = y_stack_ag.iloc[train_idx]

        # Train HG stacker model for this fold
        model_hg_fold = lgb.LGBMRegressor(**final_model_fit_params)
        model_hg_fold.fit(X_train_cv, y_train_cv_hg)  # Fit on this fold's train data
        oof_stacker_preds_hg[val_idx] = model_hg_fold.predict(X_val_cv)

        # Train AG stacker model for this fold
        ag_fold_params = final_model_fit_params.copy()
        if 'random_state' in ag_fold_params and ag_fold_params['random_state'] is not None:
             ag_fold_params['random_state'] += (fold + 1)  # Vary seed per fold slightly
        model_ag_fold = lgb.LGBMRegressor(**ag_fold_params)
        model_ag_fold.fit(X_train_cv, y_train_cv_ag)
        oof_stacker_preds_ag[val_idx] = model_ag_fold.predict(X_val_cv)
    
    # Clip OOF lambdas to ensure they're positive
    oof_stacker_preds_hg = np.maximum(oof_stacker_preds_hg, 1e-9)
    oof_stacker_preds_ag = np.maximum(oof_stacker_preds_ag, 1e-9)

    # Create the Stacker OOF DataFrame
    stacker_oof_df = oof_df[[match_id_col, date_col, target_hg_col, target_ag_col]].copy()
    if feature_cfg.target_result in oof_df.columns:
        stacker_oof_df[feature_cfg.target_result] = oof_df[feature_cfg.target_result]
    
    stacker_oof_df['stacker_pred_lambda_HG'] = oof_stacker_preds_hg
    stacker_oof_df['stacker_pred_lambda_AG'] = oof_stacker_preds_ag
    
    stacker_oof_output_path = Path(STACKER_OOF_LAMBDAS_OUTPUT_PATH_TEMPLATE.format(
        L0_OOF_TAG=l0_oof_tag, 
        ODDS_SUFFIX=odds_suffix
    ))
    stacker_oof_output_path.parent.mkdir(parents=True, exist_ok=True)
    stacker_oof_df.to_parquet(stacker_oof_output_path, index=False)
    print(f"  Saved Stacker OOF lambda predictions to: {stacker_oof_output_path}")

    # --- 5. Train Final Stacker Models on ALL L0 OOF Data (for deployment) ---
    print(f"\nTraining final stacker models on ALL L0 OOF data using parameters:")
    for k, v in final_model_fit_params.items()[:5]:  # Show first few params
        print(f"  {k}: {v}")
    print("  ...")  # Indicate there are more params
    
    final_model_hg = lgb.LGBMRegressor(**final_model_fit_params)
    final_model_hg.fit(X_stack, y_stack_hg)
    print("  Final HG stacker model fitted on all data.")

    ag_final_fit_params_full_data = final_model_fit_params.copy()
    if 'random_state' in ag_final_fit_params_full_data and ag_final_fit_params_full_data['random_state'] is not None:
        ag_final_fit_params_full_data['random_state'] += 1
    final_model_ag = lgb.LGBMRegressor(**ag_final_fit_params_full_data)
    final_model_ag.fit(X_stack, y_stack_ag)
    print("  Final AG stacker model fitted on all data.")

    # --- 6. Feature Importance Analysis ---
    print("\n--- Stacker Feature Importance ---")
    importance_hg = pd.DataFrame({
        'feature': stacker_feature_cols, 
        'importance_hg': final_model_hg.feature_importances_
    })
    importance_ag = pd.DataFrame({
        'feature': stacker_feature_cols, 
        'importance_ag': final_model_ag.feature_importances_
    })
    
    importance_df = pd.merge(importance_hg, importance_ag, on='feature', how='outer')
    importance_df['importance_total'] = importance_df['importance_hg'].fillna(0) + importance_df['importance_ag'].fillna(0)
    importance_df = importance_df.sort_values(by='importance_total', ascending=False).reset_index(drop=True)
    
    print("Top 15 Features (Total Importance):")
    print(importance_df.head(15))
    
    # Save feature importance
    importance_path = Path(STACKER_IMPORTANCE_OUTPUT_PATH_TEMPLATE.format(
        ODDS_SUFFIX=odds_suffix, 
        L0_OOF_TAG=l0_oof_tag
    ))
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_path, index=False)
    print(f"  Saved feature importance to: {importance_path}")

    # --- 7. Save Final Stacker Artifact ---
    stacker_artifact = {
        'model_hg': final_model_hg, 
        'model_ag': final_model_ag,
        'feature_columns': stacker_feature_cols, 
        'level0_models_used': current_level0_models,  # Store which L0 models were inputs
        'feature_suffixes_used': STACKER_FEATURE_SUFFIXES,
        'training_options': {
            'include_odds': include_odds, 
            'l0_oof_path_used': str(oof_path), 
            'l0_oof_variant_key': l0_oof_variant_key
        },
        'tuned_parameters': best_params_from_tune, 
        'stacker_version': STACKER_VERSION
    }
    
    model_output_path = Path(STACKER_MODEL_OUTPUT_PATH_TEMPLATE.format(
        ODDS_SUFFIX=odds_suffix, 
        L0_OOF_TAG=l0_oof_tag
    ))
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(stacker_artifact, model_output_path)
    print(f"\nSaved final stacker artifact to: {model_output_path}")

    print(f"===== Stacker Training & OOF Generation Complete (L0 OOF: {l0_oof_variant_key}, Odds: {odds_suffix}) =====")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stacker Model and Generate its OOF predictions.")
    parser.add_argument(
        "--l0_oof_variant", 
        type=str, 
        default="pca_without_odds",  # Default to the one you're currently focusing on
        choices=list(LEVEL0_MODELS_CONFIG.keys()),  # Use keys from your config
        help="Specify which L0 OOF variant to use for training the stacker."
    )
    args = parser.parse_args()

    print(f"Starting Stacker Model Training Process for L0 OOF Variant: {args.l0_oof_variant}")
    start_time = time.time()

    # Determine include_odds from the variant key
    current_include_odds = "with_odds" in args.l0_oof_variant

    # --- Initialize Ray (Needed for tuning) ---
    if RUN_STACKER_OPTIMIZATION and not ray.is_initialized():
        print("Initializing Ray for Stacker Tuning...")
        try:
            num_available_cpus = os.cpu_count()
            num_ray_cpus = max(1, num_available_cpus - 2 if num_available_cpus > 2 else 1)
            ray.init(
                num_cpus=num_ray_cpus, 
                local_mode=False, 
                ignore_reinit_error=True, 
                log_to_driver=True
            )
            print(f"Ray initialized. Using {num_ray_cpus} CPUs.")
        except Exception as e:
            print(f"CRITICAL: Ray init failed: {e}. Stacker optimization will be skipped.")
            RUN_STACKER_OPTIMIZATION = False

    # --- Ensure output directories exist ---
    STACKER_SPECIFIC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Train the stacker model
    train_stacker(l0_oof_variant_key=args.l0_oof_variant, include_odds=current_include_odds)

    # --- Shutdown Ray ---
    if ray.is_initialized():
        print("\nShutting down Ray...")
        ray.shutdown()

    end_time = time.time()
    print(f"\nStacker Training Process Finished. Total time: {end_time - start_time:.2f} seconds.")