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
from typing import Dict, List, Type, Any, Optional
import argparse
import logging

# --- Add project root ---
import sys
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH) not in sys.path:  # Only add if not already there
    sys.path.append(str(PROJECT_ROOT_PATH))
    print(f"Project Root added to sys.path: {PROJECT_ROOT_PATH}")

# --- Import model classes ---
from models.utils.features import BaseFeatureConfig, get_feature_config
from models.ml_models.poisson_model import PoissonModel, calculate_poisson_outcome_probs # Ensure calculate_poisson_outcome_probs is directly importable if MC uses it
from models.ml_models.gradient_boosting_model import GradientBoostingModel
from models.ml_models.monte_carlo_model import MonteCarloModel
from models.base_model import BaseModel
try: # Robust import for model configs
    from models.pipeline.train_pipeline import MODELS_TO_TRAIN_CONFIG as MODELS_TO_TRAIN_CONFIG_L0
except ImportError:
    print("Warning: Could not import MODELS_TO_TRAIN_CONFIG from train_pipeline. Using empty dict.")
    MODELS_TO_TRAIN_CONFIG_L0 = {}


# --- Model Registry (Focused) ---
AVAILABLE_MODELS: Dict[str, Type[BaseModel]] = {
    "poisson": PoissonModel,
    "gradient_boosting": GradientBoostingModel,
    "monte_carlo": MonteCarloModel,
}

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s [%(funcName)s] - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = PROJECT_ROOT_PATH
DATA_OUTPUT_DIR = BASE_DIR / 'models' / 'data' / 'outputs'
PARAMS_OUTPUT_DIR = DATA_OUTPUT_DIR / 'optimized_params'
L0_MODELS_JOBClIB_BASE_DIR = DATA_OUTPUT_DIR / 'joblib' # Base for V1 (NonPCA), V2 (PCA)

OOF_VERSION_SUFFIX = "_L0_focused_vFinal" # New version for this OOF

PROCESSED_DATA_PATH_TEMPLATE_PCA = str(BASE_DIR / 'models' / 'data' / 'parquets' / 'ml' / 'processed_pca_{}.parquet')
PROCESSED_DATA_PATH_TEMPLATE_NONPCA = str(BASE_DIR / 'models' / 'data' / 'parquets' / 'ml' / 'processed_without_odds.parquet') # Assuming only without_odds for NonPCA for now

# Updated params path to use the version-specific config files from V1 (NonPCA) or V2 (PCA)
def get_params_path(model_key: str, use_pca: bool, include_odds: bool = False) -> Path:
    # This function should point to where train_pipeline.py SAVED the optimized params
    # e.g., PARAMS_OUTPUT_DIR / f'best_params_ray_lambda_{METRIC}_{MODEL}_{PCASUFFIX}_{ODDSSUFFIX}.json'
    # For now, using the hardcoded paths from your script and extracting from a combined JSON.
    odds_suffix_param_file = "with_odds" if include_odds else "without_odds" # Param files might be generic for odds
    if use_pca:
        return Path(L0_MODELS_JOBClIB_BASE_DIR / "V2" / f"best_params_all_models_{odds_suffix_param_file}.json")
    else:
        return Path(L0_MODELS_JOBClIB_BASE_DIR / "V1" / f"best_params_all_models_{odds_suffix_param_file}.json")

OOF_OUTPUT_PATH_TEMPLATE = str(DATA_OUTPUT_DIR / f'level0_oof_predictions{{}}{OOF_VERSION_SUFFIX}_{{}}.parquet') # {pca_tag}, {odds_suffix}

MODELS_FOR_L0_OOF_BASE = ["poisson", "gradient_boosting"] # Models that predict lambdas directly

CORE_PREDICTION_KEYS = [ # Keep this comprehensive
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

N_SPLITS = 17
IMPUTE_NANS = True
LAMBDA_COMBINATION_STRATEGY = "average" # Options: "average", "use_gb", "use_poisson"

def validate_model_params(model_key: str, params: dict) -> None: # Removed use_pca as params should be specific to model_key
    """Validate that critical parameters are present for each model type."""
    if model_key == "gradient_boosting":
        required = {"n_estimators", "learning_rate", "max_depth", "objective"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters for {model_key}: {missing}")
    elif model_key == "poisson":
        required = {"alpha", "max_iter", "tol"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters for {model_key}: {missing}")
    elif model_key == "monte_carlo":
        required = {"n_simulations"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters for {model_key}: {missing}")

def generate_oof_predictions(include_odds: bool, use_pca: bool):
    odds_suffix = "with_odds" if include_odds else "without_odds"
    pca_suffix_for_id = "_pca_pca" if use_pca else "_nonpca" # For constructing model identifiers like "poisson_pca_pca_without_odds"
    pca_tag_for_filename = "_pca" if use_pca else "_nonpca"     # For output filenames like "level0_oof_predictions_pca..."

    print(f"\n===== Generating L0 OOF (Data features: {pca_suffix_for_id}, Odds: {odds_suffix}) =====")

    data_path = Path(PROCESSED_DATA_PATH_TEMPLATE_PCA.format(odds_suffix)) if use_pca else Path(PROCESSED_DATA_PATH_TEMPLATE_NONPCA.format(odds_suffix))
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
        cols_to_exclude = set(target_cols + [feature_cfg.match_id_col, feature_cfg.date_col, 'Season', 'Tier', 'GameWeek', feature_cfg.home_team_col, feature_cfg.away_team_col, feature_cfg.league_id_col])
        potential_features = df_full.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [f for f in potential_features if f not in cols_to_exclude]
    
    assert feature_cols, f"No feature columns for pca={use_pca}."
    print(f"Selected {len(feature_cols)} features. First 5: {feature_cols[:5]}")
    
    id_col = feature_cfg.match_id_col; date_col = feature_cfg.date_col
    X_full = df_full[feature_cols]; y_full = df_full[target_cols]; match_ids = df_full[id_col]
    assert not X_full.isnull().values.any(), "NaNs in X_full."

    # This dictionary will store OOF predictions for each model for each fold temporarily
    # before being combined into final model-specific OOF DataFrames.
    fold_predictions_all_models: Dict[str, List[pd.DataFrame]] = {} # Key: model_identifier, Value: List of DFs from folds

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"Using TimeSeriesSplit with {N_SPLITS} splits.")

    for fold_idx, (train_indices, val_indices) in enumerate(tscv.split(X_full)):
        fold_num = fold_idx + 1
        print(f"\n--- Processing Fold {fold_num}/{N_SPLITS} ---")
        if len(val_indices) == 0: continue

        X_train_fold, X_val_fold = X_full.iloc[train_indices], X_full.iloc[val_indices]
        y_train_fold = y_full.iloc[train_indices]
        match_ids_val_fold = match_ids.iloc[val_indices]
        
        # Store lambdas from this fold's Poisson and GB predictions
        fold_lambdas_poisson_h: Optional[np.ndarray] = None
        fold_lambdas_poisson_a: Optional[np.ndarray] = None
        fold_lambdas_gb_h: Optional[np.ndarray] = None
        fold_lambdas_gb_a: Optional[np.ndarray] = None

        # --- Generate OOF for Poisson and Gradient Boosting for this fold ---
        for model_key_base in MODELS_FOR_L0_OOF_BASE: # "poisson", "gradient_boosting"
            model_identifier = f"{model_key_base}{pca_suffix_for_id}_{odds_suffix}" # Construct full name
            print(f"  Fitting & Predicting for: {model_identifier} on Fold {fold_num}")

            ModelClass = AVAILABLE_MODELS[model_key_base]
            params_path = get_params_path(model_key_base, use_pca, include_odds) # Gets V1 or V2 json path
            
            model_base_params = MODELS_TO_TRAIN_CONFIG_L0.get(model_key_base, {}).get("params", {})
            current_model_params = model_base_params.copy()
            if params_path.exists():
                try:
                    with open(params_path, 'r') as f: all_params_in_file = json.load(f)
                    if model_key_base in all_params_in_file: # Check if base key exists
                        current_model_params.update(all_params_in_file[model_key_base])
                except Exception: pass # Ignore if file format issue, use defaults
            validate_model_params(model_key_base, current_model_params) # Validate

            model_fold = ModelClass(model_params=current_model_params, feature_config=feature_cfg, apply_scaling=not use_pca)
            y_fit_targets = y_train_fold[[feature_cfg.target_home_goals, feature_cfg.target_away_goals]]
            model_fold.fit(X_train_fold, y_fit_targets)
            val_pred_dict_raw: Dict[str, np.ndarray] = model_fold.predict_proba(X_val_fold)

            current_fold_model_data = {}
            for raw_key in CORE_PREDICTION_KEYS:
                model_internal_output_key = f"{model_key_base}_{raw_key}"
                if model_internal_output_key in val_pred_dict_raw:
                    oof_col_name = f"{model_identifier}_{raw_key}" # Full name for OOF
                    current_fold_model_data[oof_col_name] = val_pred_dict_raw[model_internal_output_key]
            
            if current_fold_model_data:
                if model_identifier not in fold_predictions_all_models: fold_predictions_all_models[model_identifier] = []
                fold_predictions_all_models[model_identifier].append(pd.DataFrame(current_fold_model_data, index=match_ids_val_fold))
                
                # Store lambdas for MC
                if model_key_base == "poisson":
                    fold_lambdas_poisson_h = val_pred_dict_raw.get(f"poisson_expected_HG")
                    fold_lambdas_poisson_a = val_pred_dict_raw.get(f"poisson_expected_AG")
                elif model_key_base == "gradient_boosting":
                    fold_lambdas_gb_h = val_pred_dict_raw.get(f"gradient_boosting_expected_HG")
                    fold_lambdas_gb_a = val_pred_dict_raw.get(f"gradient_boosting_expected_AG")
            else:
                warnings.warn(f"No predictions for {model_identifier} in fold {fold_num}")

        # --- Generate OOF for Enhanced Monte Carlo for this fold ---
        print(f"  Preparing Enhanced Monte Carlo for Fold {fold_num}")
        lambda_h_for_mc: Optional[np.ndarray] = None
        lambda_a_for_mc: Optional[np.ndarray] = None

        if LAMBDA_COMBINATION_STRATEGY == "average":
            if fold_lambdas_gb_h is not None and fold_lambdas_poisson_h is not None:
                print(f"    MC Fold {fold_num}: Using AVERAGED lambdas from GB & Poisson.")
                lambda_h_for_mc = (fold_lambdas_gb_h + fold_lambdas_poisson_h) / 2.0
                lambda_a_for_mc = (fold_lambdas_gb_a + fold_lambdas_poisson_a) / 2.0
            elif fold_lambdas_gb_h is not None: # Fallback to GB if Poisson missing
                print(f"    MC Fold {fold_num}: Using GB lambdas (Poisson missing for avg).")
                lambda_h_for_mc = fold_lambdas_gb_h
                lambda_a_for_mc = fold_lambdas_gb_a
            elif fold_lambdas_poisson_h is not None: # Fallback to Poisson if GB missing
                print(f"    MC Fold {fold_num}: Using Poisson lambdas (GB missing for avg).")
                lambda_h_for_mc = fold_lambdas_poisson_h
                lambda_a_for_mc = fold_lambdas_poisson_a
        elif LAMBDA_COMBINATION_STRATEGY == "use_gb" and fold_lambdas_gb_h is not None:
            print(f"    MC Fold {fold_num}: Using GB lambdas.")
            lambda_h_for_mc = fold_lambdas_gb_h; lambda_a_for_mc = fold_lambdas_gb_a
        elif LAMBDA_COMBINATION_STRATEGY == "use_poisson" and fold_lambdas_poisson_h is not None:
            print(f"    MC Fold {fold_num}: Using Poisson lambdas.")
            lambda_h_for_mc = fold_lambdas_poisson_h; lambda_a_for_mc = fold_lambdas_poisson_a
        
        if lambda_h_for_mc is not None and lambda_a_for_mc is not None:
            # Ensure no NaNs in final lambdas for MC for this fold
            if np.isnan(lambda_h_for_mc).any() or np.isnan(lambda_a_for_mc).any():
                warnings.warn(f"    MC Fold {fold_num}: NaNs found in combined/selected lambdas. Attempting imputation with 0. Fallback might be needed.")
                lambda_h_for_mc = np.nan_to_num(lambda_h_for_mc, nan=0.0) # Impute with 0 if NaN
                lambda_a_for_mc = np.nan_to_num(lambda_a_for_mc, nan=0.0)

            X_val_for_mc_sim = pd.DataFrame({
                'external_lambda_HG': lambda_h_for_mc,
                'external_lambda_AG': lambda_a_for_mc
            }, index=match_ids_val_fold) # Ensure index is correct for this fold's validation set

            mc_params_file = get_params_path("monte_carlo", use_pca, include_odds)
            mc_final_params = {'n_simulations': 10000 if use_pca else 80000} # Default
            if mc_params_file.exists():
                try:
                    with open(mc_params_file, 'r') as f: all_mc_p = json.load(f)
                    if 'monte_carlo' in all_mc_p: mc_final_params.update(all_mc_p['monte_carlo'])
                except Exception: pass
            validate_model_params("monte_carlo", mc_final_params)

            mc_model_fold = MonteCarloModel(model_params=mc_final_params, feature_config=feature_cfg, apply_scaling=False)
            mc_model_fold.features_in_ = ['external_lambda_HG', 'external_lambda_AG']
            mc_fold_pred_dict_raw = mc_model_fold._predict_proba_model(X_val_for_mc_sim)

            mc_oof_identifier_fold = f"monte_carlo_enhanced{pca_suffix_for_id}_{odds_suffix}"
            current_fold_mc_data = {}
            for raw_key in CORE_PREDICTION_KEYS:
                mc_internal_key = f"monte_carlo_enhanced_{raw_key}" # Key from MC class
                if mc_internal_key in mc_fold_pred_dict_raw:
                    oof_col_name = f"{mc_oof_identifier_fold}_{raw_key}" # Key for final OOF
                    current_fold_mc_data[oof_col_name] = mc_fold_pred_dict_raw[mc_internal_key]
            
            # Add the input lambdas with the expected_HG and expected_AG suffixes
            # This ensures compatibility with the stacker's expected column naming
            current_fold_mc_data[f"{mc_oof_identifier_fold}_expected_HG"] = lambda_h_for_mc
            current_fold_mc_data[f"{mc_oof_identifier_fold}_expected_AG"] = lambda_a_for_mc
            
            if current_fold_mc_data:
                if mc_oof_identifier_fold not in fold_predictions_all_models: fold_predictions_all_models[mc_oof_identifier_fold] = []
                fold_predictions_all_models[mc_oof_identifier_fold].append(pd.DataFrame(current_fold_mc_data, index=match_ids_val_fold))
            else: warnings.warn(f"    MC Fold {fold_num}: No predictions extracted from MC output.")
        else:
            warnings.warn(f"    MC Fold {fold_num}: Could not derive final lambdas for MC. Skipping MC for this fold.")

    # --- Combine all fold predictions for each model ---
    final_oof_dfs_combined: Dict[str, pd.DataFrame] = {}
    for model_id_final, list_of_fold_dfs in fold_predictions_all_models.items():
        if list_of_fold_dfs:
            final_oof_dfs_combined[model_id_final] = pd.concat(list_of_fold_dfs).sort_index()
            print(f"Combined OOF for {model_id_final}: {final_oof_dfs_combined[model_id_final].shape} rows")
        else:
            warnings.warn(f"No fold data to combine for {model_id_final}")


    if not final_oof_dfs_combined:
        raise RuntimeError(f"CRITICAL: No L0 OOF generated for ANY model for {pca_suffix_for_id}, {odds_suffix}.")

    # --- Join all model OOFs to the base df_full subset ---
    final_oof_output_df = df_full[[id_col, date_col] + target_cols].set_index(id_col)
    for model_id_to_join, oof_data_to_join in final_oof_dfs_combined.items():
        if not oof_data_to_join.empty:
            final_oof_output_df = final_oof_output_df.join(oof_data_to_join, how='left')
    
    # --- Handle NaNs ---
    nan_cols = final_oof_output_df.columns[final_oof_output_df.isnull().any()].tolist()
    pred_nan_cols = [c for c in nan_cols if c not in target_cols + [date_col]] # Exclude base cols
    if pred_nan_cols:
        if IMPUTE_NANS:
            # print(f"Imputing NaNs for columns: {pred_nan_cols[:5]}...")
            impute_values = final_oof_output_df[pred_nan_cols].mean()
            impute_map = {col: impute_values.get(col, 0.5 if '_prob_' in col else (0.0 if 'expected' in col else 0.0)) for col in pred_nan_cols}
            final_oof_output_df.fillna(value=impute_map, inplace=True)
            assert final_oof_output_df[pred_nan_cols].isnull().sum().sum() == 0, "NaN imputation failed."
    
    output_path = Path(OOF_OUTPUT_PATH_TEMPLATE.format(pca_tag_for_filename, odds_suffix))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_oof_output_df.reset_index().to_parquet(output_path, index=False, engine='pyarrow')
    print(f"Saved L0 OOF ({pca_suffix_for_id}, {odds_suffix}) to: {output_path}")
    print(f"===== L0 OOF Generation Complete ({pca_suffix_for_id}, {odds_suffix}) =====")

# Main Execution Block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate L0 OOF predictions for focused models (GB, Poisson, Enhanced MC) with PCA/Non-PCA options.")
    parser.add_argument("--no_pca", action="store_true", help="Run for Non-PCA features.")
    parser.add_argument("--with_pca", action="store_true", help="Run for PCA features.")
    parser.add_argument("--lambda_combo_strategy", type=str, 
                        choices=["average", "use_gb", "use_poisson"], 
                        default="average", 
                        help="Strategy for combining lambdas for Monte Carlo.")

    args = parser.parse_args()

    if not args.no_pca and not args.with_pca:
        print("Please specify at least one of --no_pca or --with_pca. Defaulting to PCA only for this run.")
        args.with_pca = True
    
    LAMBDA_COMBINATION_STRATEGY = args.lambda_combo_strategy # Set global from arg
    logging.info(f"Running OOF generation with args: {args}")
    
    print("Starting FOCUSED L0 OOF Prediction Generation Process...")
    start_time = time.time()
    include_odds_run = False # For "without_odds" only

    if args.with_pca:
        print("\n>>>> Generating OOF for PCA features, WITHOUT odds <<<<")
        generate_oof_predictions(include_odds=include_odds_run, use_pca=True)
    
    if args.no_pca:
        print("\n>>>> Generating OOF for Non-PCA features, WITHOUT odds <<<<")
        generate_oof_predictions(include_odds=include_odds_run, use_pca=False)

    end_time = time.time()
    print(f"\nFOCUSED L0 OOF Generation Finished. Total time: {end_time - start_time:.2f} seconds.")