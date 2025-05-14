# pipelines/generate_oof_predictions.py
import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib
from pathlib import Path
import time
import traceback
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Type, Any
import argparse

# --- Add project root ---
import sys
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH) not in sys.path:  # Only add if not already there
    sys.path.append(str(PROJECT_ROOT_PATH))
    print(f"Project Root added to sys.path: {PROJECT_ROOT_PATH}")

# --- Import model classes ---
from models.utils.features import BaseFeatureConfig, get_feature_config
from models.ml_models.poisson_model import PoissonModel
from models.ml_models.gradient_boosting_model import GradientBoostingModel
from models.ml_models.monte_carlo_model import MonteCarloModel # Will use this in enhanced mode
from models.base_model import BaseModel
from models.pipeline.train_pipeline import MODELS_TO_TRAIN_CONFIG as MODELS_TO_TRAIN_CONFIG_L0

# --- Model Registry (Focused) ---
AVAILABLE_MODELS: Dict[str, Type[BaseModel]] = {
    "poisson": PoissonModel,
    "gradient_boosting": GradientBoostingModel,
    "monte_carlo": MonteCarloModel, # Keep for instantiation, but will be fed lambdas
}

# --- Configuration ---
BASE_DIR = PROJECT_ROOT_PATH
DATA_OUTPUT_DIR = BASE_DIR / 'models' / 'data' / 'outputs'
PARAMS_OUTPUT_DIR = DATA_OUTPUT_DIR / 'optimized_params'

# --- Updated Path Configurations ---
# PCA (V2) paths
PCA_MODEL_CONFIG_PATH = BASE_DIR / 'models' / 'data' / 'outputs' / 'joblib' / 'V2' / 'best_params_all_models_without_odds.json'
PCA_DATA_PATH = BASE_DIR / 'models' / 'data' / 'parquets' / 'ml' / 'processed_pca_without_odds.parquet'

# Non-PCA (V1) paths
NONPCA_MODEL_CONFIG_PATH = BASE_DIR / 'models' / 'data' / 'outputs' / 'joblib' / 'V1' / 'best_params_all_models_without_odds.json'
NONPCA_DATA_PATH = BASE_DIR / 'models' / 'data' / 'parquets' / 'ml' / 'processed_without_odds.parquet'

# Define where to load L0 models from based on PCA/non-PCA version
L0_MODELS_BASE_DIR = DATA_OUTPUT_DIR / 'joblib'
OOF_VERSION_SUFFIX = "_L0_focused"

# Path templates updated to use the new paths
PROCESSED_DATA_PATH_TEMPLATE_PCA = str(PCA_DATA_PATH)
PROCESSED_DATA_PATH_TEMPLATE_NONPCA = str(NONPCA_DATA_PATH)

LAMBDA_OPTIMIZATION_METRIC_USED = 'rmse'

# Updated params path to use the version-specific config files
def get_params_path(model_key: str, use_pca: bool) -> str:
    if use_pca:
        return str(PCA_MODEL_CONFIG_PATH)
    return str(NONPCA_MODEL_CONFIG_PATH)

# Output OOF path
OOF_OUTPUT_PATH_TEMPLATE = str(DATA_OUTPUT_DIR / f'level0_oof_predictions{{}}{OOF_VERSION_SUFFIX}_{{}}.parquet')

# --- Models to Generate L0 OOF Predictions For (Focused) ---
# These are the keys used to find models and params
# The actual model filenames might include _pca_ or _nonpca_
MODELS_FOR_L0_OOF_BASE = ["poisson", "gradient_boosting"]
# Monte Carlo will be handled specially after these two.

# --- Core Prediction Keys (same as before) ---
CORE_PREDICTION_KEYS = [
    'expected_HG', 'expected_AG', 'prob_H', 'prob_D', 'prob_A', 'prob_1X', 'prob_12', 'prob_X2',
    'prob_O15', 'prob_U15', 'prob_O25', 'prob_U25', 'prob_O35', 'prob_U35',
    'prob_BTTS_Y', 'prob_BTTS_N', 'prob_goals_0_1', 'prob_goals_2_3', 'prob_goals_2_4', 'prob_goals_3_plus',
    # --- Match Result + Goals (ALL LINES) ---
    'prob_H_and_O15', 'prob_D_and_O15', 'prob_A_and_O15', 'prob_H_and_U15', 'prob_D_and_U15', 'prob_A_and_U15',
    'prob_H_and_O25', 'prob_D_and_O25', 'prob_A_and_O25', 'prob_H_and_U25', 'prob_D_and_U25', 'prob_A_and_U25',
    'prob_H_and_O35', 'prob_D_and_O35', 'prob_A_and_O35', 'prob_H_and_U35', 'prob_D_and_U35', 'prob_A_and_U35',
    'prob_H_and_O45', 'prob_D_and_O45', 'prob_A_and_O45', 'prob_H_and_U45', 'prob_D_and_U45', 'prob_A_and_U45',
    # --- Double Chance + Goals (ALL LINES) ---
    'prob_1X_and_O15', 'prob_12_and_O15', 'prob_X2_and_O15', 'prob_1X_and_U15', 'prob_12_and_U15', 'prob_X2_and_U15',
    'prob_1X_and_O25', 'prob_12_and_O25', 'prob_X2_and_O25', 'prob_1X_and_U25', 'prob_12_and_U25', 'prob_X2_and_U25',
    'prob_1X_and_O35', 'prob_12_and_O35', 'prob_X2_and_O35', 'prob_1X_and_U35', 'prob_12_and_U35', 'prob_X2_and_U35',
    'prob_1X_and_O45', 'prob_12_and_O45', 'prob_X2_and_O45', 'prob_1X_and_U45', 'prob_12_and_U45', 'prob_X2_and_U45',
    # --- Match Result + BTTS ---
    'prob_H_and_BTTS_Y', 'prob_D_and_BTTS_Y', 'prob_A_and_BTTS_Y', 'prob_H_and_BTTS_N', 'prob_D_and_BTTS_N', 'prob_A_and_BTTS_N',
    # --- Double Chance + BTTS ---
    'prob_1X_and_BTTS_Y', 'prob_12_and_BTTS_Y', 'prob_X2_and_BTTS_Y', 'prob_1X_and_BTTS_N', 'prob_12_and_BTTS_N', 'prob_X2_and_BTTS_N',
    # --- Goals + BTTS (ALL LINES) ---
    'prob_O25_and_BTTS_Y', 'prob_O25_and_BTTS_N', 'prob_O35_and_BTTS_Y', 'prob_O35_and_BTTS_N',
]

N_SPLITS = 17
IMPUTE_NANS = True
# Decide which model's lambdas to primarily use for the enhanced Monte Carlo
# This could be based on which model had better lambda prediction performance
# Or you could even average them if both are good.
# Let's default to 'gradient_boosting' if available, else 'poisson'.
PRIMARY_LAMBDA_SOURCE_MODEL_KEY = "gradient_boosting"

def validate_model_params(model_key: str, params: dict, use_pca: bool) -> None:
    """Validate that critical parameters are present for each model type."""
    if model_key == "gradient_boosting":
        required = {"n_estimators", "learning_rate", "max_depth", "objective"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters for {model_key} ({'PCA' if use_pca else 'Non-PCA'}): {missing}")
    elif model_key == "poisson":
        required = {"alpha", "max_iter", "tol"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters for {model_key} ({'PCA' if use_pca else 'Non-PCA'}): {missing}")
    elif model_key == "monte_carlo":
        required = {"n_simulations"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters for {model_key} ({'PCA' if use_pca else 'Non-PCA'}): {missing}")

def generate_oof_predictions(include_odds: bool, use_pca: bool):
    odds_suffix = "with_odds" if include_odds else "without_odds"
    pca_suffix = "pca" if use_pca else "nonpca"
    pca_tag = "_pca" if use_pca else "_nonpca" # For filenames

    print(f"\n===== Generating L0 OOF (Data: {pca_suffix}, Odds: {odds_suffix}) =====")

    # --- 1. Load Processed Training Data (PCA or Non-PCA) ---
    if use_pca:
        data_path = Path(PROCESSED_DATA_PATH_TEMPLATE_PCA)
        model_joblib_dir = L0_MODELS_BASE_DIR / "V2"  # V2 for PCA models
    else:
        data_path = Path(PROCESSED_DATA_PATH_TEMPLATE_NONPCA)
        model_joblib_dir = L0_MODELS_BASE_DIR / "V1"  # V1 for non-PCA models

    print(f"Loading data from: {data_path}")
    assert data_path.exists(), f"Data file not found: {data_path}"
    df_full = pd.read_parquet(data_path, engine='pyarrow')
    df_full['Date'] = pd.to_datetime(df_full['Date'])
    df_full = df_full.sort_values(by='Date').reset_index(drop=True)

    feature_cfg = get_feature_config(include_odds=include_odds)
    target_cols = [feature_cfg.target_home_goals, feature_cfg.target_away_goals, feature_cfg.target_result]
    if use_pca:
        feature_cols = [col for col in df_full.columns if col.startswith('PC')]
    else:
        # Only select numeric columns for non-PCA features
        numeric_cols = df_full.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [f for f in numeric_cols 
                       if f not in target_cols + [feature_cfg.match_id_col, feature_cfg.date_col, 'Season', 'Tier', 'GameWeek']]

    assert feature_cols, f"No feature columns found for pca={use_pca} in {data_path}."
    print(f"Selected {len(feature_cols)} features for {'PCA' if use_pca else 'Non-PCA'} model")
    id_col = feature_cfg.match_id_col
    date_col = feature_cfg.date_col

    # Add after feature selection
    # Validate features are all numeric
    X_full = df_full[feature_cols]
    assert X_full.select_dtypes(include=['int64', 'float64']).shape[1] == len(feature_cols), \
        "Non-numeric columns found in features. Check feature selection."

    y_full = df_full[target_cols]
    match_ids = df_full[id_col]
    assert not X_full.isnull().any().any(), "NaNs in features."

    oof_pred_dfs: Dict[str, pd.DataFrame] = {}
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # --- Loop Through Base L0 Models (Poisson, GB) ---
    for model_key_base in MODELS_FOR_L0_OOF_BASE:
        model_identifier_for_loading = f"{model_key_base}_{pca_suffix}_{odds_suffix}"
        print(f"\n--- Generating OOF for L0 model: {model_identifier_for_loading} ---")
        ModelClass = AVAILABLE_MODELS.get(model_key_base)
        assert ModelClass is not None

        # Load model parameters from the appropriate version's config file
        params_path = Path(get_params_path(model_key_base, use_pca))
        model_params = MODELS_TO_TRAIN_CONFIG_L0.get(model_key_base, {}).get("params", {})

        if params_path.exists():
            try:
                with open(params_path, 'r') as f:
                    all_params = json.load(f)
                    # Extract model-specific parameters from the config file
                    if model_key_base in all_params:
                        model_params = all_params[model_key_base]
                        validate_model_params(model_key_base, model_params, use_pca)
                print(f"  Loaded params for {model_identifier_for_loading} from: {params_path}")
            except Exception as e:
                warnings.warn(f"  Failed loading params {params_path}: {e}. Using defaults for {model_key_base}.", UserWarning)
        else:
            warnings.warn(f"  Params file not found: {params_path}. Using defaults for {model_key_base}.", UserWarning)
        
        model_oof_preds_list = []
        for fold_idx, (train_indices, val_indices) in enumerate(tscv.split(X_full)):
            fold_num = fold_idx + 1; print(f"  Processing Fold {fold_num}/{N_SPLITS}...")
            if len(val_indices) == 0: continue
            X_train_fold, X_val_fold = X_full.iloc[train_indices], X_full.iloc[val_indices]
            y_train_fold = y_full.iloc[train_indices]
            match_ids_val = match_ids.iloc[val_indices]

            # --- Instantiate and Train Model ---
            # When instantiating, tell model if it's using PCA features or not for scaling
            model_fold = ModelClass(model_params=model_params, feature_config=feature_cfg, apply_scaling=not use_pca)
            y_fit_targets = y_train_fold[[feature_cfg.target_home_goals, feature_cfg.target_away_goals]]
            model_fold.fit(X_train_fold, y_fit_targets)

            val_pred_dict_raw: Dict[str, np.ndarray] = model_fold.predict_proba(X_val_fold)
            
            # --- Store Predictions with Full Model Identifier ---
            current_fold_pred_data = {}
            for raw_key in CORE_PREDICTION_KEYS:
                # The model's predict_proba should return keys prefixed with model_key_base (e.g., "poisson_expected_HG")
                model_output_key = f"{model_key_base}_{raw_key}"
                if model_output_key in val_pred_dict_raw:
                    # Store with the full identifier including PCA/odds status for clarity in OOF
                    oof_col_name = f"{model_identifier_for_loading}_{raw_key}" # e.g. poisson_pca_with_odds_expected_HG
                    current_fold_pred_data[oof_col_name] = val_pred_dict_raw[model_output_key]
            
            if not current_fold_pred_data:
                 warnings.warn(f"No core predictions extracted for {model_identifier_for_loading} in fold {fold_num}")
                 continue
            model_oof_preds_list.append(pd.DataFrame(current_fold_pred_data, index=match_ids_val))

        if model_oof_preds_list:
            oof_pred_dfs[model_identifier_for_loading] = pd.concat(model_oof_preds_list).sort_index()
            print(f"--- Finished OOF for L0 model: {model_identifier_for_loading} ---")

    # --- Enhanced Monte Carlo using Best Lambdas from GB or Poisson ---
    print(f"\n--- Generating OOF for Enhanced Monte Carlo (Data: {pca_suffix}, Odds: {odds_suffix}) ---")
    
    # Determine the source of lambdas for this MC run
    primary_lambda_source_full_id = f"{PRIMARY_LAMBDA_SOURCE_MODEL_KEY}_{pca_suffix}_{odds_suffix}"
    fallback_lambda_source_full_id = f"poisson_{pca_suffix}_{odds_suffix}"
    
    lambda_source_df = None
    lambda_source_model_name_used = ""

    if primary_lambda_source_full_id in oof_pred_dfs:
        lambda_source_df = oof_pred_dfs[primary_lambda_source_full_id]
        lambda_source_model_name_used = primary_lambda_source_full_id
    elif fallback_lambda_source_full_id in oof_pred_dfs:
        lambda_source_df = oof_pred_dfs[fallback_lambda_source_full_id]
        lambda_source_model_name_used = fallback_lambda_source_full_id
    
    if lambda_source_df is not None:
        lambda_hg_col_name = f"{lambda_source_model_name_used}_expected_HG"
        lambda_ag_col_name = f"{lambda_source_model_name_used}_expected_AG"

        if lambda_hg_col_name in lambda_source_df.columns and lambda_ag_col_name in lambda_source_df.columns:
            print(f"  Using lambdas from {lambda_source_model_name_used} for Enhanced Monte Carlo.")
            
            # Load Monte Carlo parameters from the appropriate version's config file
            mc_params_path = Path(get_params_path("monte_carlo", use_pca))
            mc_default_params = {'n_simulations': 10000 if use_pca else 80000}
            
            if mc_params_path.exists():
                try:
                    with open(mc_params_path, 'r') as f:
                        all_params = json.load(f)
                        if 'monte_carlo' in all_params:
                            mc_default_params.update(all_params['monte_carlo'])
                    print(f"  Loaded MC params from: {mc_params_path}")
                except Exception as e:
                    warnings.warn(f"  Failed loading MC params from {mc_params_path}: {e}. Using defaults.", UserWarning)
            
            mc_oof_preds_list = []
            for fold_idx, (train_indices, val_indices) in enumerate(tscv.split(X_full)):
                fold_num = fold_idx + 1
                print(f"  MC Processing Fold {fold_num}/{N_SPLITS}...")
                if len(val_indices) == 0:
                    continue
                
                match_ids_val_mc = match_ids.iloc[val_indices]
                fold_lambdas_df = lambda_source_df.loc[lambda_source_df.index.isin(match_ids_val_mc)]
                
                if fold_lambdas_df.empty or lambda_hg_col_name not in fold_lambdas_df.columns:
                    warnings.warn(f"  MC Fold {fold_num}: Lambda source data missing for validation indices. Skipping MC for this fold.")
                    continue

                lambda_h_for_mc_fold = fold_lambdas_df[lambda_hg_col_name].values
                lambda_a_for_mc_fold = fold_lambdas_df[lambda_ag_col_name].values

                X_val_fold_for_mc = pd.DataFrame({
                    'external_lambda_HG': lambda_h_for_mc_fold,
                    'external_lambda_AG': lambda_a_for_mc_fold
                }, index=match_ids_val_mc)

                mc_model_fold = MonteCarloModel(
                    model_params=mc_default_params,
                    feature_config=feature_cfg,
                    apply_scaling=False
                )
                mc_model_fold.features_in_ = ['external_lambda_HG', 'external_lambda_AG']

                val_pred_dict_mc_raw: Dict[str, np.ndarray] = mc_model_fold._predict_proba_model(X_val_fold_for_mc)

                mc_identifier_for_oof = f"monte_carlo_enhanced_{pca_suffix}_{odds_suffix}"
                current_fold_mc_data = {}
                for raw_key in CORE_PREDICTION_KEYS:
                    mc_model_output_key = f"monte_carlo_enhanced_{raw_key}"
                    if mc_model_output_key in val_pred_dict_mc_raw:
                        oof_col_name = f"{mc_identifier_for_oof}_{raw_key}"
                        current_fold_mc_data[oof_col_name] = val_pred_dict_mc_raw[mc_model_output_key]
                
                if current_fold_mc_data:
                    mc_oof_preds_list.append(pd.DataFrame(current_fold_mc_data, index=match_ids_val_mc))
            
            if mc_oof_preds_list:
                oof_pred_dfs[mc_identifier_for_oof] = pd.concat(mc_oof_preds_list).sort_index()
                print(f"--- Finished OOF for Enhanced Monte Carlo: {mc_identifier_for_oof} ---")
            else:
                warnings.warn(f"No OOF predictions generated for Enhanced Monte Carlo ({pca_suffix}, {odds_suffix}).")
        else:
            warnings.warn(f"  Lambda columns not found in {lambda_source_model_name_used} OOF for Enhanced MC. Skipping MC.")
    else:
        warnings.warn(f"  No suitable lambda source model OOF found for Enhanced MC. Skipping MC.")


    # --- Combine All Predictions ---
    if not oof_pred_dfs:
        raise RuntimeError(f"CRITICAL: No L0 OOF generated for {pca_suffix}, {odds_suffix}.")

    final_oof_df = df_full[[id_col, date_col] + target_cols].set_index(id_col)
    for model_oof_identifier, model_oof_data in oof_pred_dfs.items():
        final_oof_df = final_oof_df.join(model_oof_data, how='left') # model_oof_data is already indexed
    # --- Handle NaNs ---
    nan_cols = final_oof_df.columns[final_oof_df.isnull().any()].tolist()
    pred_nan_cols = [c for c in nan_cols if c not in target_cols + [date_col]]
    if pred_nan_cols:
        if IMPUTE_NANS:
            impute_values = final_oof_df[pred_nan_cols].mean()
            impute_map = {col: impute_values.get(col, 0.5 if '_prob_' in col else 0.0) for col in pred_nan_cols}
            final_oof_df.fillna(value=impute_map, inplace=True)
            assert final_oof_df[pred_nan_cols].isnull().sum().sum() == 0, "NaN imputation failed."
        else: warnings.warn(f"NaNs remain in {pred_nan_cols}.")

    # --- Save Final OOF DataFrame ---
    output_path = Path(OOF_OUTPUT_PATH_TEMPLATE.format(pca_tag, odds_suffix))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_oof_df.reset_index().to_parquet(output_path, index=False, engine='pyarrow')
    print(f"Saved L0 OOF ({pca_suffix}, {odds_suffix}) to: {output_path}")
    print(f"===== L0 OOF Generation Complete ({pca_suffix}, {odds_suffix}) =====")

# Main Execution Block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate L0 OOF predictions for focused models (GB, Poisson, Enhanced MC) with PCA/Non-PCA options.")
    parser.add_argument("--no_pca", action="store_true", help="Run for Non-PCA features.")
    parser.add_argument("--with_pca", action="store_true", help="Run for PCA features.")
    # parser.add_argument("--odds_config", type=str, choices=['with_odds', 'without_odds', 'both'], default='without_odds', help="Odds configuration to run for.")
    # For now, let's stick to your request: only "without_odds"
    
    args = parser.parse_args()

    if not args.no_pca and not args.with_pca:
        print("Please specify at least one of --no_pca or --with_pca. Defaulting to PCA only.")
        args.with_pca = True # Default behavior if nothing specified

    print("Starting FOCUSED L0 OOF Prediction Generation Process...")
    start_time = time.time()

    # --- Run for "without_odds" data ---
    include_odds_run = False # As per your request

    if args.with_pca:
        print("\n>>>> Generating OOF for PCA features, WITHOUT odds <<<<")
        generate_oof_predictions(include_odds=include_odds_run, use_pca=True)
    
    if args.no_pca:
        print("\n>>>> Generating OOF for Non-PCA features, WITHOUT odds <<<<")
        generate_oof_predictions(include_odds=include_odds_run, use_pca=False)


    end_time = time.time()
    print(f"\nFOCUSED L0 OOF Generation Finished. Total time: {end_time - start_time:.2f} seconds.")