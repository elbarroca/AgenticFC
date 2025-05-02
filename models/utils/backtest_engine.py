# utils/backtest_engine.py

import pandas as pd
import numpy as np
from tqdm.auto import tqdm # Progress bar
import logging
from typing import Dict, List, Optional, Any, Tuple
import time
import warnings

# Import necessary components from your project structure
try:
    from models.model_registry import get_model_class, list_available_models
    from models.base_model import BaseModel # For type hinting
    from utils import metrics
    from utils import features
    from utils import config # To get default settings if needed
    # Import specific model classes if needed for type hints or special handling
    # from models.poisson_model import PoissonModel
    # from models.monte_carlo_model import MonteCarloModel
except ImportError as e:
    logging.error(f"Error importing project components: {e}. Ensure paths are correct.")
    # Define dummy classes or raise error if imports are critical
    BaseModel = type('BaseModel', (object,), {}) # Dummy for type hint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

class BacktestEngine:
    """
    Performs time-series cross-validation (backtesting) for predictive models.

    Handles data splitting, feature generation (with leakage awareness),
    model training, prediction, evaluation, and optional betting simulation
    using a rolling or expanding window approach.
    """

    def __init__(self,
                 full_historical_data: pd.DataFrame,
                 date_col: str = 'Date',
                 target_col: str = 'FTR', # Or specify target for regression/binary
                 model_names: List[str] = ['random_forest'], # Models to test from registry
                 model_params: Optional[Dict[str, Dict]] = None, # Specific params per model
                 feature_config: Optional[Dict] = None, # Config for feature generation
                 backtest_config: Optional[Dict] = None, # Config for splitting/windowing
                 evaluation_metrics: Optional[List[str]] = None, # Metrics to calculate
                 betting_config: Optional[Dict] = None # Config for ROI simulation
                 ):
        """
        Initializes the BacktestEngine.

        Args:
            full_historical_data (pd.DataFrame): Complete dataset with dates and results.
            date_col (str): Name of the date column for time-series splitting.
            target_col (str): Name of the target variable column (e.g., 'FTR', 'Over2.5').
            model_names (List[str]): List of model names (keys in model_registry) to backtest.
            model_params (Optional[Dict[str, Dict]]): Dictionary mapping model names to their
                                                      initialization parameters. If None, uses defaults.
            feature_config (Optional[Dict]): Configuration for feature generation (e.g., window sizes).
                                             Defaults used if None.
            backtest_config (Optional[Dict]): Configuration for backtesting strategy:
                                              {'start_date', 'end_date', 'train_window_size',
                                               'test_window_size', 'step_size', 'strategy' ('rolling'/'expanding')}.
                                              Defaults used if None.
            evaluation_metrics (Optional[List[str]]): List of metric function names from utils.metrics
                                                      to compute (e.g., ['accuracy', 'multi_logloss']).
            betting_config (Optional[Dict]): Configuration for betting simulation:
                                             {'enabled': bool, 'odds_cols': list, 'threshold': float, 'stake': float}.
                                             If None or enabled=False, betting simulation is skipped.
        """
        self.full_data = full_historical_data.sort_values(by=date_col).reset_index(drop=True)
        self.date_col = date_col
        self.target_col = target_col
        self.model_names = model_names
        self.model_params = model_params if model_params else {}
        self.feature_config = feature_config if feature_config else {}
        self.backtest_config = self._get_default_backtest_config(backtest_config)
        self.evaluation_metrics = evaluation_metrics if evaluation_metrics else ['accuracy', 'multi_logloss'] # Default metrics
        self.betting_config = self._get_default_betting_config(betting_config)

        self.results: List[Dict] = [] # Stores results from each step

        logging.info("BacktestEngine initialized.")
        logging.info(f"Data range: {self.full_data[self.date_col].min()} to {self.full_data[self.date_col].max()}")
        logging.info(f"Models to test: {self.model_names}")
        logging.info(f"Backtest strategy: {self.backtest_config['strategy']}, Train window: {self.backtest_config['train_window_size']}, Test window: {self.backtest_config['test_window_size']}, Step: {self.backtest_config['step_size']}")

    def _get_default_backtest_config(self, config_in: Optional[Dict]) -> Dict:
        """Provides default backtesting parameters."""
        defaults = {
            'start_date': None, # Auto-detect if None
            'end_date': None, # Auto-detect if None
            'train_window_size': pd.Timedelta(days=365 * 2), # Example: 2 years training data
            'test_window_size': pd.Timedelta(days=30), # Example: Predict next month
            'step_size': pd.Timedelta(days=30), # Example: Retrain every month
            'strategy': 'rolling' # 'rolling' or 'expanding'
        }
        if config_in:
            # Convert string durations to Timedeltas if needed
            for key in ['train_window_size', 'test_window_size', 'step_size']:
                if key in config_in and isinstance(config_in[key], str):
                    try:
                        config_in[key] = pd.Timedelta(config_in[key])
                    except ValueError:
                         logging.warning(f"Could not parse Timedelta for {key}: '{config_in[key]}'. Using default.")
                         config_in[key] = defaults[key]
            defaults.update(config_in)
        return defaults

    def _get_default_betting_config(self, config_in: Optional[Dict]) -> Dict:
        """Provides default betting simulation parameters with available odds columns."""
        # Find available odds columns in the data instead of hardcoding 
        available_odds_cols = [col for col in self.full_data.columns if 
                              any(col.startswith(prefix) for prefix in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC'])]
        
        if not available_odds_cols:
            logging.warning("No recognized odds columns found in data")
            available_odds_cols = []  # Empty list if none found
        
        defaults = {
            'enabled': len(available_odds_cols) > 0,  # Only enable if odds columns exist
            'odds_cols': available_odds_cols[:3] if len(available_odds_cols) >= 3 else available_odds_cols,
            'threshold': 0.05,
            'stake': 1.0
        }
        
        if config_in:
            defaults.update(config_in)
        
        return defaults

    def _get_time_splits(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generates time splits based on the backtest configuration.

        Returns:
            List[Tuple]: A list of tuples, each containing:
                         (train_start_date, train_end_date, test_start_date, test_end_date)
        """
        splits = []
        start_date = pd.to_datetime(self.backtest_config['start_date'] or self.full_data[self.date_col].min())
        end_date = pd.to_datetime(self.backtest_config['end_date'] or self.full_data[self.date_col].max())
        train_window = self.backtest_config['train_window_size']
        test_window = self.backtest_config['test_window_size']
        step = self.backtest_config['step_size']
        strategy = self.backtest_config['strategy']

        current_train_start = start_date
        current_test_end = start_date + train_window + test_window

        while current_test_end <= end_date:
            train_end = current_train_start + train_window
            test_start = train_end # Test period starts immediately after train period

            # Ensure we don't exceed the overall end_date for the test period
            actual_test_end = min(test_start + test_window, end_date)

            # Check if the test period has any duration left
            if actual_test_end > test_start:
                splits.append((current_train_start, train_end, test_start, actual_test_end))

            # Move to the next window
            if strategy == 'rolling':
                current_train_start += step
            # For expanding, train_start remains fixed, only test_end moves forward by step
            # This logic assumes test_window and step align; adjust if needed
            current_test_end = test_start + test_window + step # Estimate next cycle's end

            # Safety break if step is zero or negative
            if step <= pd.Timedelta(0):
                 logging.error("Step size must be positive. Stopping split generation.")
                 break
            # Safety break if start date doesn't advance in rolling strategy
            if strategy == 'rolling' and splits and splits[-1][0] >= current_train_start:
                 logging.error("Training start date did not advance. Check step/window sizes. Stopping split generation.")
                 break


        logging.info(f"Generated {len(splits)} time splits for backtesting.")
        if splits:
             logging.info(f"First split: Train {splits[0][0]} - {splits[0][1]}, Test {splits[0][2]} - {splits[0][3]}")
             logging.info(f"Last split: Train {splits[-1][0]} - {splits[-1][1]}, Test {splits[-1][2]} - {splits[-1][3]}")
        return splits

    def _prepare_data_for_split(self, train_indices, test_indices):
        """
        Prepares features and targets for a single train/test split.
        """
        try:
            train_data = self.full_data.loc[train_indices].copy()
            test_data = self.full_data.loc[test_indices].copy()

            # Generate features for combined data
            logging.debug(f"Generating features for split (Train size: {len(train_data)}, Test size: {len(test_data)})...")
            combined_data_for_features = pd.concat([train_data, test_data])
            all_features_df = features.generate_features(combined_data_for_features)

            # Separate Train/Test Features and Targets
            X_train = all_features_df.loc[train_indices]
            X_test = all_features_df.loc[test_indices]
            
            # Special handling for Monte Carlo model which needs both goals columns
            if 'monte_carlo' in self.model_names:
                logging.info("Preparing targets for Monte Carlo model (FTHG and FTAG)")
                y_train = train_data[['FTR', 'FTHG', 'FTAG']]
                y_test = test_data[['FTR', 'FTHG', 'FTAG']]
            else:
                y_train = train_data[self.target_col]
                y_test = test_data[self.target_col]

            # Handle NaNs if any
            if X_train.isnull().values.any() or X_test.isnull().values.any():
                logging.warning("NaNs found after feature generation. Filling with 0.")
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logging.error(f"Error preparing data for split: {e}", exc_info=True)
            return None, None, None, None


    def run_backtest(self):
        """
        Executes the entire backtesting process over the defined time splits.
        """
        logging.info("Starting backtest run...")
        self.results = [] # Clear previous results
        time_splits = self._get_time_splits()

        if not time_splits:
            logging.warning("No time splits generated. Backtest cannot run.")
            return

        # --- Main Backtesting Loop ---
        for i, (train_start, train_end, test_start, test_end) in enumerate(tqdm(time_splits, desc="Backtest Progress")):
            split_info = f"Split {i+1}/{len(time_splits)}: Train [{train_start.date()}-{train_end.date()}] Test [{test_start.date()}-{test_end.date()}]"
            logging.info(split_info)

            # --- Get data indices for the current split ---
            train_indices = self.full_data[
                (self.full_data[self.date_col] >= train_start) &
                (self.full_data[self.date_col] < train_end) # Train up to (but not including) train_end
            ].index
            test_indices = self.full_data[
                (self.full_data[self.date_col] >= test_start) &
                (self.full_data[self.date_col] < test_end) # Test up to (but not including) test_end
            ].index

            if train_indices.empty or test_indices.empty:
                logging.warning(f"Skipping split {i+1} due to empty train or test set.")
                continue

            # --- Prepare Data (Features & Target) ---
            X_train, y_train, X_test, y_test = self._prepare_data_for_split(train_indices, test_indices)

            if X_train is None or X_test is None:
                 logging.error(f"Skipping split {i+1} due to data preparation error.")
                 continue

            # Store original test data details for results merging
            test_data_orig = self.full_data.loc[test_indices].copy()

            # --- Iterate through Models ---
            for model_name in self.model_names:
                model_start_time = time.time()
                logging.info(f"  Processing model: {model_name}")

                try:
                    # --- Instantiate Model ---
                    ModelClass = get_model_class(model_name)
                    if ModelClass is None:
                        logging.warning(f"    Could not find model class for '{model_name}'. Skipping.")
                        continue

                    # Get specific params or use defaults
                    params = self.model_params.get(model_name, {})
                    model_instance = ModelClass(**params)

                    # --- Train Model ---
                    # Handle models needing eval_set (like XGBoost with early stopping)
                    fit_kwargs = {}
                    if self.backtest_config.get('use_validation_set', False) and hasattr(model_instance, 'early_stopping_rounds') and model_instance.early_stopping_rounds:
                         # Need to create a validation split from the training data
                         # This adds complexity, simplified here: assumes fit handles val_split if needed
                         # Or pass a dedicated eval_set if prepared earlier
                         logging.warning("Validation set logic for early stopping not fully implemented in this blueprint.")
                         # Example: fit_kwargs['eval_set'] = [(X_val, y_val)]

                    # Handle models with different fit signatures (e.g., Poisson, Markov)
                    # This requires specific logic per model type if they don't conform to fit(X, y)
                    if isinstance(model_instance, (features.PoissonModel, features.MarkovModel)): # Assuming these classes exist
                         logging.debug(f"    Fitting {model_name} using historical data subset.")
                         model_instance.fit(self.full_data.loc[train_indices]) # Pass relevant historical slice
                    else:
                         logging.debug(f"    Fitting {model_name} with X_train, y_train.")
                         model_instance.fit(X_train, y_train, **fit_kwargs)


                    # --- Predict on Test Set ---
                     # Handle models with different predict signatures
                    if isinstance(model_instance, (features.PoissonModel, features.MarkovModel)):
                         # These might need different inputs (e.g., team names) extracted from test_data_orig
                         logging.warning(f"Prediction logic for {model_name} needs specific input handling. Placeholder used.")
                         # Placeholder: generate dummy predictions matching expected format
                         # Real implementation needs to iterate through test_data_orig and call predict appropriately
                         predictions_df = pd.DataFrame(index=X_test.index) # Dummy
                         if model_instance.target_type == 'classification': # Assuming target_type attr exists
                              predictions_df['prediction'] = y_test.iloc[0] # Dummy pred
                              for cls_ in model_instance.classes_: predictions_df[f'prob_{cls_}'] = 1.0/len(model_instance.classes_)
                         else: predictions_df['prediction'] = 0 # Dummy pred
                    else:
                         predictions_df = model_instance.predict(X_test)


                    model_end_time = time.time()
                    logging.info(f"    {model_name} processing took {model_end_time - model_start_time:.2f}s")

                    # --- Evaluate Predictions ---
                    eval_results = self._evaluate_predictions(y_test, predictions_df)

                    # --- Simulate Betting (Optional) ---
                    betting_results = {}
                    if self.betting_config['enabled']:
                        # Merge predictions with original test data containing odds
                        results_with_odds = test_data_orig.merge(
                            predictions_df, left_index=True, right_index=True
                        )
                        betting_results = self._simulate_betting(results_with_odds)

                    # --- Store Results for this model and split ---
                    self.results.append({
                        'split': i + 1,
                        'train_start': train_start,
                        'train_end': train_end,
                        'test_start': test_start,
                        'test_end': test_end,
                        'model_name': model_name,
                        'num_train_samples': len(X_train),
                        'num_test_samples': len(X_test),
                        **eval_results,
                        **betting_results,
                        # Optionally store raw predictions/targets if memory allows
                        # 'predictions': predictions_df,
                        # 'targets': y_test
                    })

                except Exception as e:
                    logging.error(f"    Error processing model {model_name} for split {i+1}: {e}", exc_info=True)
                    # Store failure information?
                    self.results.append({
                        'split': i + 1, 'model_name': model_name, 'status': 'FAILED', 'error': str(e)
                    })

        logging.info("Backtest run finished.")

   