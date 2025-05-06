# pipelines/train_pipeline.py
import pandas as pd
import numpy as np
import os
import json # For saving/loading optimized params
import warnings
import optuna # Import Optuna
from typing import Dict, Any
from sklearn.model_selection import train_test_split # For Optuna validation split
from sklearn.preprocessing import StandardScaler
# Import necessary estimators for optimization function
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_poisson_deviance, root_mean_squared_error
import lightgbm as lgb
# --- Import Model Classes ---
from models.utils.features import BaseFeatureConfig, get_feature_config
from models.utils.poisson_model import PoissonModel
from models.ml_models.random_forest_model import RandomForestModel
from models.ml_models.gradient_boosting_model import GradientBoostingModel
from models.ml_models.monte_carlo_model import MonteCarloModel # Use the updated MC model

# --- Model Registry ---
AVAILABLE_MODELS = {
    "poisson": PoissonModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "monte_carlo": MonteCarloModel,
}

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'data', 'outputs')
PARAMS_OUTPUT_DIR = os.path.join(DATA_OUTPUT_DIR, 'optimized_params')
MODELS_OUTPUT_DIR = DATA_OUTPUT_DIR

PROCESSED_DATA_PATH_TEMPLATE = os.path.join(DATA_OUTPUT_DIR, 'processed_{}.parquet') # with_odds or without_odds
OPTIMIZED_PARAMS_PATH_TEMPLATE = os.path.join(PARAMS_OUTPUT_DIR, 'best_params_{}_{}.json') # model_type, odds_suffix
MODEL_OUTPUT_PATH_TEMPLATE = os.path.join(MODELS_OUTPUT_DIR, '{}_{}_v1.joblib') # model_type, odds_suffix


# --- Models to Train & Default Params ---
MODELS_TO_TRAIN_CONFIG = {
    "poisson": {
        # User requested aggressive params/tuning
        "params": {'alpha': 1e-4, 'max_iter': 8000000, 'tol': 1e-3},
        "optimize": True, # Tune alpha
    },
    #"random_forest": {
        # Default params before optimization
        #"params": {'n_estimators': 800, 'max_depth': 35, 'min_samples_leaf': 10, 'n_jobs': -1, 'random_state': 42},
        #"optimize": True,
    #},
    "gradient_boosting": {
        # Default params before optimization
        "params": {'n_estimators': 800, 'learning_rate': 0.1, 'max_depth': 8, 'num_leaves': 35, 'n_jobs': -1, 'random_state': 44, 'objective': 'poisson'},
        "optimize": True,
    },
    "monte_carlo": {
        # Parameters for the *new* MonteCarloModel structure
        "params": {'n_simulations': 80000, 'internal_estimator_alpha': 1.0},
        "optimize": False, # This model structure is not tuned via Optuna here
    },
}

# Optuna Settings
OPTUNA_N_TRIALS = 80 # Number of trials for optimization (adjust as needed)
RUN_OPTIMIZATION = True # Set to False to skip optimization and use defaults/saved params

# Bayesian Optimization Function (Updated for Poisson fixed params)
def run_bayesian_optimization(
    model_type: str,
    X_train_all: pd.DataFrame,
    y_train_all: pd.DataFrame,
    feature_cfg: BaseFeatureConfig,
    n_trials: int = 25,
    validation_size: float = 0.2
) -> Dict[str, Any]:
    """Performs Optuna optimization for Poisson, RF, or LGBM."""
    # --- Check if optimization is defined for this model ---
    if model_type not in ['poisson', 'random_forest', 'gradient_boosting']:
         warnings.warn(f"Optimization requested but not defined for model type: {model_type}. Returning defaults.", UserWarning)
         return MODELS_TO_TRAIN_CONFIG[model_type]["params"]

    print(f"\n--- Running Optuna Optimization for {model_type} ---")
    target_hg = feature_cfg.target_home_goals
    target_ag = feature_cfg.target_away_goals

    X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
        X_train_all, y_train_all, test_size=validation_size, random_state=42
    )
    scaler = StandardScaler()
    X_train_tune_scaled = scaler.fit_transform(X_train_tune)
    X_val_tune_scaled = scaler.transform(X_val_tune)
    X_train_tune_scaled = pd.DataFrame(X_train_tune_scaled, index=X_train_tune.index, columns=X_train_tune.columns)
    X_val_tune_scaled = pd.DataFrame(X_val_tune_scaled, index=X_val_tune.index, columns=X_val_tune.columns)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # --- Get default fixed params for the model ---
        default_params = MODELS_TO_TRAIN_CONFIG[model_type]["params"]

        if model_type == "poisson":
            params = {
                'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
                'max_iter': default_params.get('max_iter', 500000), # Use default fixed value
                'tol': default_params.get('tol', 1e-4),          # Use default fixed value
            }
            model_home = PoissonRegressor(**params)
            model_away = PoissonRegressor(**params)
            # Use Poisson Deviance for evaluation metric
            loss_metric_func = mean_poisson_deviance

        elif model_type == "random_forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'n_jobs': -1, 'random_state': 42, # Fixed during trial
            }
            model_home = RandomForestRegressor(**params)
            params_away = params.copy(); params_away['random_state'] = 43
            model_away = RandomForestRegressor(**params_away)
            # Use root_mean_squared_error for evaluation metric
            loss_metric_func = root_mean_squared_error

        elif model_type == "gradient_boosting":
            params = {
                'objective': 'poisson', 'metric': 'poisson', 'random_state': 42, 'n_jobs': -1, # Fixed
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 8, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            model_home = lgb.LGBMRegressor(**params)
            params_away = params.copy(); params_away['random_state'] = 43
            model_away = lgb.LGBMRegressor(**params_away)
            # Use mean_poisson_deviance for evaluation metric, consistent with the objective
            loss_metric_func = mean_poisson_deviance
        else:
             raise ValueError(f"Invalid model_type for optimization: {model_type}")

        # --- Train & Evaluate ---
        try:
            model_home.fit(X_train_tune_scaled, y_train_tune[target_hg])
            preds_home = model_home.predict(X_val_tune_scaled)
            preds_home = np.maximum(preds_home, 0)
            # Call the selected metric function
            loss_home = loss_metric_func(y_val_tune[target_hg], preds_home)

            model_away.fit(X_train_tune_scaled, y_train_tune[target_ag])
            preds_away = model_away.predict(X_val_tune_scaled)
            preds_away = np.maximum(preds_away, 0)
            # Call the selected metric function
            loss_away = loss_metric_func(y_val_tune[target_ag], preds_away)

        except Exception as fit_eval_error:
             print(f"Error during fit/eval in trial for {model_type} with params {params}: {fit_eval_error}")
             return float('inf')

        combined_loss = (loss_home + loss_away) / 2.0
        return combined_loss

    # --- Run Optuna Study ---
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    try:
        study.optimize(objective, n_trials=n_trials, timeout=3600) # 1hr timeout
    except Exception as e:
         print(f"An error occurred during Optuna optimization study: {e}")
         warnings.warn(f"Optimization failed for {model_type}. Returning default parameters.", UserWarning)
         return MODELS_TO_TRAIN_CONFIG[model_type]["params"]

    if not study.trials or study.best_trial is None:
         warnings.warn(f"Optuna study finished without a valid best trial for {model_type}. Returning default parameters.", UserWarning)
         return MODELS_TO_TRAIN_CONFIG[model_type]["params"]

    print(f"Optimization complete for {model_type}.")
    print(f"  Best Trial #{study.best_trial.number}")
    print(f"  Best Value (Loss): {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")

    # --- Combine best tuned params with fixed params ---
    best_params = study.best_params
    fixed_params = MODELS_TO_TRAIN_CONFIG[model_type]["params"]
    final_params = fixed_params.copy() # Start with defaults
    final_params.update(best_params) # Overwrite with tuned values

    # Ensure essential fixed keys are present if not tuned (e.g., objective for LGBM)
    if model_type == 'gradient_boosting' and 'objective' not in final_params:
        final_params['objective'] = 'poisson'
    if model_type in ['random_forest', 'gradient_boosting'] and 'n_jobs' not in final_params:
         final_params['n_jobs'] = -1
    if model_type in ['poisson', 'random_forest', 'gradient_boosting'] and 'random_state' not in final_params:
         final_params['random_state'] = 42 # Ensure base random state

    print(f"  Final combined params: {final_params}")
    return final_params

# Main Training Function (No changes needed here)
def run_training(
    processed_data_path: str,
    model_output_path: str,
    include_odds: bool,
    model_type: str,
    model_params: Dict[str, Any], # Expects final params
    feature_cfg: BaseFeatureConfig # Pass config explicitly
    ):
    """
    Loads processed data, trains a single model instance with final parameters,
    and saves the artifact.
    """
    print(f"\n--- Starting Final Model Training: {model_type} (Odds: {include_odds}) ---")
    print(f"Using data: {processed_data_path}")
    print(f"Outputting model to: {model_output_path}")
    print(f"Using parameters: {model_params}")

    # --- Load Processed Data ---
    print("Loading processed data...")
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

    # --- Prepare Features (X) and Targets (y) ---
    print("Preparing features (X) and targets (y)...")
    target_home = feature_cfg.target_home_goals
    target_away = feature_cfg.target_away_goals
    target_cols = [target_home, target_away]
    assert set(target_cols).issubset(df.columns), f"Data missing target columns: {set(target_cols) - set(df.columns)}"
    y = df[target_cols].copy()

    feature_cols = feature_cfg.get_feature_columns(include_odds=include_odds)
    missing_features = set(feature_cols) - set(df.columns)
    assert not missing_features, f"Data missing feature columns: {missing_features}"
    X = df[feature_cols].copy()

    assert X.shape[0] == y.shape[0], f"Row mismatch X({X.shape[0]}) vs y({y.shape[0]})."
    assert not X.isnull().any().any(), "Features (X) contain NaNs before training."
    assert not y.isnull().any().any(), "Targets (y) contain NaNs before training."
    print(f"Prepared X shape: {X.shape}, y shape: {y.shape}")

    # --- Instantiate Model ---
    print(f"Instantiating model: {model_type}")
    ModelClass = AVAILABLE_MODELS.get(model_type)
    assert ModelClass is not None, f"Model type '{model_type}' not found."

    try:
        # Pass feature_config explicitly, required by all our models now
        model = ModelClass(model_params=model_params, feature_config=feature_cfg)
    except Exception as e:
        print(f"CRITICAL ERROR instantiating model {model_type}: {e}")
        raise

    # --- Train Model ---
    print("Starting model fitting (includes internal scaling)...")
    try:
        model.fit(X, y) # BaseModel's fit handles scaling
        print("Model fitting complete.")
    except Exception as e:
        print(f"CRITICAL ERROR during model fitting: {e}")
        raise

    # --- Save Model ---
    print(f"Saving trained model to: {model_output_path}")
    try:
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        model.save(model_output_path) # BaseModel's save handles scaler etc.
        print("Trained model saved successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR saving trained model to {model_output_path}: {e}")
        raise

    print(f"--- Model Training Complete: {model_type} (Odds: {include_odds}) ---")

# Main Execution Block
if __name__ == "__main__":

    os.makedirs(PARAMS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

    # --- Phase 1: Bayesian Optimization (Optional) ---
    if RUN_OPTIMIZATION:
        print("\n===== PHASE 1: Running Bayesian Optimization (Optuna) =====")
        for include_odds in [True, False]:
            odds_suffix = "with_odds" if include_odds else "without_odds"
            print(f"\n--- Optimizing for data: {odds_suffix} ---")

            data_path = PROCESSED_DATA_PATH_TEMPLATE.format(odds_suffix)
            if not os.path.exists(data_path):
                print(f"WARNING: Data file not found, skipping optimization: {data_path}")
                continue

            try:
                df_all = pd.read_parquet(data_path)
                feature_cfg_opt = get_feature_config(include_odds=include_odds)
                target_cols_opt = [feature_cfg_opt.target_home_goals, feature_cfg_opt.target_away_goals]
                feature_cols_opt = feature_cfg_opt.get_feature_columns(include_odds=include_odds)

                # Basic validation
                assert not df_all.empty
                assert set(target_cols_opt).issubset(df_all.columns)
                assert set(feature_cols_opt).issubset(df_all.columns)

                X_all_opt = df_all[feature_cols_opt]
                y_all_opt = df_all[target_cols_opt]
                assert not X_all_opt.isnull().any().any(), f"NaNs found in features for optimization ({odds_suffix})"
                assert not y_all_opt.isnull().any().any(), f"NaNs found in targets for optimization ({odds_suffix})"

            except Exception as e:
                print(f"ERROR loading or preparing data for optimization ({odds_suffix}): {e}")
                continue

            optimized_params_for_setting = {}
            for model_key, config in MODELS_TO_TRAIN_CONFIG.items():
                if config["optimize"]:
                    try:
                        best_params = run_bayesian_optimization(
                            model_type=model_key,
                            X_train_all=X_all_opt,
                            y_train_all=y_all_opt,
                            feature_cfg=feature_cfg_opt,
                            n_trials=OPTUNA_N_TRIALS
                        )
                        optimized_params_for_setting[model_key] = best_params
                    except Exception as e:
                         print(f"ERROR during optimization for {model_key} ({odds_suffix}): {e}")
                         print(f"Will use default parameters for {model_key}.")
                         optimized_params_for_setting[model_key] = config["params"] # Fallback
                else:
                    # Use default params if optimization is disabled for this model
                    optimized_params_for_setting[model_key] = config["params"]

            # Save the optimized (or default) parameters for this odds setting
            params_output_path = OPTIMIZED_PARAMS_PATH_TEMPLATE.format('all_models', odds_suffix)
            try:
                with open(params_output_path, 'w') as f:
                    json.dump(optimized_params_for_setting, f, indent=4)
                print(f"Saved parameters for {odds_suffix} to {params_output_path}")
            except Exception as e:
                print(f"ERROR saving parameters file {params_output_path}: {e}")

        print("\n===== PHASE 1: Optimization Complete =====")
    else:
        print("\n===== Skipping PHASE 1: Bayesian Optimization =====")


    # --- Phase 2: Final Model Training ---
    print("\n===== PHASE 2: Training Final Models =====")
    for include_odds in [True, False]:
        odds_suffix = "with_odds" if include_odds else "without_odds"
        print(f"\n--- Training models for data: {odds_suffix} ---")

        data_path = PROCESSED_DATA_PATH_TEMPLATE.format(odds_suffix)
        params_path = OPTIMIZED_PARAMS_PATH_TEMPLATE.format('all_models', odds_suffix)

        if not os.path.exists(data_path):
            print(f"WARNING: Data file not found, skipping training for {odds_suffix}: {data_path}")
            continue

        # Load parameters (optimized or default)
        final_params_all_models = {}
        if os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    final_params_all_models = json.load(f)
                print(f"Loaded parameters from: {params_path}")
            except Exception as e:
                print(f"WARNING: Failed to load parameters file {params_path}: {e}. Using defaults.")
                final_params_all_models = {m: c["params"] for m, c in MODELS_TO_TRAIN_CONFIG.items()}
        else:
             print(f"WARNING: Parameters file not found: {params_path}. Using defaults.")
             final_params_all_models = {m: c["params"] for m, c in MODELS_TO_TRAIN_CONFIG.items()}


        # Get feature config for this setting
        try:
             feature_cfg_train = get_feature_config(include_odds=include_odds)
        except Exception as e:
             print(f"ERROR loading feature config for {odds_suffix}: {e}. Skipping training.")
             continue

        # Loop through models defined in config and train them
        for model_key, config in MODELS_TO_TRAIN_CONFIG.items():
            model_params_final = final_params_all_models.get(model_key, config["params"]) # Use loaded/optimized or fallback to default
            model_output_path = MODEL_OUTPUT_PATH_TEMPLATE.format(model_key, odds_suffix)

            try:
                run_training(
                    processed_data_path=data_path,
                    model_output_path=model_output_path,
                    include_odds=include_odds,
                    model_type=model_key,
                    model_params=model_params_final,
                    feature_cfg=feature_cfg_train # Pass the loaded config
                )
            except FileNotFoundError:
                 print(f"Skipping training for {model_key} ({odds_suffix}) due to missing data.")
            except Exception as e:
                 print(f"!!! ERROR during final training for {model_key} ({odds_suffix}) !!!")
                 print(f"Error details: {e}")
                 import traceback
                 traceback.print_exc() # Print full traceback for debugging

    print("\n===== PHASE 2: Final Training Complete =====")