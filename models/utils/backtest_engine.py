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
        """Provides default betting simulation parameters."""
        defaults = {
            'enabled': False,
            'odds_cols': getattr(config, 'ODDS_COLUMNS', ['B365H', 'B365D', 'B365A']), # Get from config if possible
            'threshold': getattr(config, 'ROI_BET_THRESHOLD', 0.05),
            'stake': getattr(config, 'ROI_STAKE', 1.0)
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

    def _prepare_data_for_split(self, train_indices: pd.Index, test_indices: pd.Index) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepares features and targets for a single train/test split.

        **CRITICAL NOTE:** Feature generation (esp. rolling stats) must avoid leakage.
        This implementation assumes `features.generate_features` can be called on the
        combined train+test data for the *current* split, BUT that the feature generation
        logic itself is time-aware (e.g., rolling calculations use `closed='left'` or similar).
        A truly robust implementation might require fitting feature transformers (scalers,
        imputers) ONLY on train data and applying to test, and calculating rolling features
        iteratively or within `generate_features` based on the split dates. This blueprint
        uses a simplified approach assuming `generate_features` handles some time awareness.

        Args:
            train_indices (pd.Index): Indices for the training data in self.full_data.
            test_indices (pd.Index): Indices for the test data in self.full_data.

        Returns:
            Tuple: (X_train, y_train, X_test, y_test) ready for model fitting/prediction.
                   Returns (None, None, None, None) if feature generation fails.
        """
        try:
            train_data = self.full_data.loc[train_indices].copy()
            test_data = self.full_data.loc[test_indices].copy()

            # --- Feature Generation ---
            # Ideally, fit transformers (scalers etc.) ONLY on train_data here
            # For simplicity, we call generate_features, assuming it's somewhat time-aware
            # A more robust approach might pass train/test data separately to generate_features
            # or perform scaling/imputation explicitly here.

            logging.debug(f"Generating features for split (Train size: {len(train_data)}, Test size: {len(test_data)})...")
            # Combine temporarily for feature generation that might need context,
            # but ensure generate_features respects time boundaries internally if possible.
            combined_data_for_features = pd.concat([train_data, test_data])

            # --- Elo Calculation (Example: Recalculate up to train_end for this split) ---
            # This ensures Elo reflects state *before* test period
            split_train_end_date = train_data[self.date_col].max()
            elo_calc = features.EloCalculator( # Assuming EloCalculator is in features or utils
                 k_factor=self.feature_config.get('ELO_K_FACTOR', config.ELO_K_FACTOR),
                 home_advantage=self.feature_config.get('ELO_HOME_ADVANTAGE', config.ELO_HOME_ADVANTAGE)
            )
            # Calculate Elo only on data up to the end of the training period for this split
            elo_data_for_split = elo_calc.calculate_historical_elos(
                 self.full_data[self.full_data[self.date_col] <= split_train_end_date],
                 date_col=self.date_col
            )

            # --- Generate other features using combined data ---
            # Pass the correctly calculated Elo data for this split
            # Note: generate_features needs to handle merging based on index or keys
            all_features_df = features.generate_features(
                combined_data_for_features,
                elo_df=elo_data_for_split, # Pass the time-aware Elo data
                odds_cols=self.betting_config.get('odds_cols', config.ODDS_COLUMNS),
                rolling_window=self.feature_config.get('ROLLING_WINDOW_SIZE', config.ROLLING_WINDOW_SIZE)
            )

            # --- Separate Train/Test Features and Targets ---
            X_train = all_features_df.loc[train_indices]
            y_train = train_data[self.target_col]
            X_test = all_features_df.loc[test_indices]
            y_test = test_data[self.target_col]

            # --- Handle NaNs resulting from feature generation (e.g., initial rolling windows) ---
            # Option 1: Drop rows with NaNs (might lose early test data)
            # Option 2: Impute (fit imputer ONLY on X_train, transform X_train & X_test)
            # Simple approach: Fill with 0 (as done in generate_features example) - check if appropriate
            if X_train.isnull().values.any() or X_test.isnull().values.any():
                 logging.warning("NaNs found after feature generation for split. Check imputation/feature logic.")
                 # Applying fillna(0) again if generate_features didn't handle it fully
                 X_train = X_train.fillna(0)
                 X_test = X_test.fillna(0)


            logging.debug("Data preparation for split complete.")
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

    def _evaluate_predictions(self, y_true: pd.Series, predictions_df: pd.DataFrame) -> Dict:
        """Calculates evaluation metrics based on configuration."""
        eval_results = {}
        y_pred = predictions_df.get('prediction', None)

        for metric_name in self.evaluation_metrics:
            try:
                metric_func = getattr(metrics, metric_name, None)
                if metric_func is None:
                    logging.warning(f"Metric function '{metric_name}' not found in utils.metrics. Skipping.")
                    continue

                # Pass appropriate arguments based on metric name convention
                if 'logloss' in metric_name:
                     # Assumes probability columns exist (e.g., prob_H, prob_D, prob_A or prob_1)
                     if 'multi' in metric_name:
                         eval_results[metric_name] = metric_func(y_true, predictions_df)
                     else: # binary
                         prob_col = next((c for c in predictions_df.columns if c.startswith('prob_') and c != 'prob_0'), 'prob_1') # Heuristic
                         if prob_col in predictions_df:
                              eval_results[metric_name] = metric_func(y_true, predictions_df[prob_col])
                         else: logging.warning(f"Could not find positive probability column for {metric_name}.")
                elif metric_name == 'accuracy':
                     if y_pred is not None:
                         eval_results[metric_name] = metric_func(y_true, y_pred)
                     else: logging.warning(f"Cannot calculate accuracy without 'prediction' column.")
                # Add other metric handling here (e.g., RMSE, MAE for regression)
                else:
                     # Default: assume metric takes y_true, y_pred
                     if y_pred is not None:
                          eval_results[metric_name] = metric_func(y_true, y_pred)
                     else: logging.warning(f"Cannot calculate {metric_name} without 'prediction' column.")

            except Exception as e:
                logging.error(f"Error calculating metric '{metric_name}': {e}")
                eval_results[metric_name] = None # Indicate failure

        return eval_results

    def _simulate_betting(self, results_with_odds_df: pd.DataFrame) -> Dict:
        """Simulates betting based on the configuration."""
        if not self.betting_config['enabled']:
            return {}

        try:
            # Use the ROI function from metrics module
            roi_results = metrics.calculate_roi(
                results_with_odds_df,
                # Assuming classification probabilities are present
                prob_col_h=next((c for c in results_with_odds_df.columns if c.endswith('_H')), 'prob_H'),
                prob_col_d=next((c for c in results_with_odds_df.columns if c.endswith('_D')), 'prob_D'),
                prob_col_a=next((c for c in results_with_odds_df.columns if c.endswith('_A')), 'prob_A'),
                odds_col_h=self.betting_config['odds_cols'][0],
                odds_col_d=self.betting_config['odds_cols'][1],
                odds_col_a=self.betting_config['odds_cols'][2],
                true_result_col=self.target_col, # Assumes target is FTR for ROI calc
                bet_threshold=self.betting_config['threshold'],
                stake=self.betting_config['stake']
            )
            # Rename keys slightly for clarity in results
            return {
                'bets_placed': roi_results.get('bets_placed', 0),
                'betting_stake': roi_results.get('total_staked', 0.0),
                'betting_return': roi_results.get('total_returned', 0.0),
                'betting_roi': roi_results.get('roi', 0.0)
            }
        except Exception as e:
            logging.error(f"Error during betting simulation: {e}")
            return {'bets_placed': 0, 'betting_stake': 0.0, 'betting_return': 0.0, 'betting_roi': None}


    def get_results(self) -> pd.DataFrame:
        """Returns the collected backtest results as a DataFrame."""
        if not self.results:
            logging.warning("No results collected yet. Run run_backtest() first.")
            return pd.DataFrame()
        return pd.DataFrame(self.results)

    def summary(self, group_by: str = 'model_name') -> pd.DataFrame:
        """Generates a summary of the backtest results, aggregated by model or split."""
        results_df = self.get_results()
        if results_df.empty:
            return pd.DataFrame()

        if 'status' in results_df.columns and 'FAILED' in results_df['status'].unique():
             logging.warning("Some models/splits failed during backtesting. Summary might be incomplete.")
             results_df = results_df[results_df['status'] != 'FAILED'].copy() # Exclude failed runs from summary stats

        if results_df.empty:
             logging.warning("No successful runs found to summarize.")
             return pd.DataFrame()


        numeric_cols = results_df.select_dtypes(include=np.number).columns.tolist()
        # Exclude split number from averaging if grouping by model
        cols_to_agg = [col for col in numeric_cols if col not in ['split', 'num_train_samples', 'num_test_samples']]

        if not cols_to_agg:
             logging.warning("No numeric metric columns found to aggregate.")
             return pd.DataFrame()

        # Define aggregation functions
        agg_funcs = {col: 'mean' for col in cols_to_agg}
        # Add specific aggregations if needed (e.g., total stake/return for ROI)
        if 'betting_stake' in results_df.columns: agg_funcs['betting_stake'] = 'sum'
        if 'betting_return' in results_df.columns: agg_funcs['betting_return'] = 'sum'
        if 'bets_placed' in results_df.columns: agg_funcs['bets_placed'] = 'sum'
        agg_funcs['num_test_samples'] = 'sum' # Total samples tested

        summary_df = results_df.groupby(group_by).agg(agg_funcs)

        # Recalculate overall ROI if betting was enabled
        if 'betting_stake' in summary_df.columns and 'betting_return' in summary_df.columns:
            summary_df['overall_roi'] = summary_df.apply(
                lambda row: ((row['betting_return'] - row['betting_stake']) / row['betting_stake']) * 100 if row['betting_stake'] > 0 else 0,
                axis=1
            )
            # Maybe drop the averaged ROI column if overall is calculated
            if 'betting_roi' in summary_df.columns:
                 summary_df = summary_df.drop(columns=['betting_roi'])


        return summary_df.round(4) # Round for display


# Example Usage (Illustrative - requires actual data and configured components)
if __name__ == '__main__':
    print("\n--- BacktestEngine Example ---")
    # --- 1. Load Dummy Data (replace with your actual data loading) ---
    # Create minimal dummy data for structure
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=100, freq='W'))
    dummy_data = pd.DataFrame({
        'Date': dates,
        'HomeTeam': [f'Team{i % 10}' for i in range(100)],
        'AwayTeam': [f'Team{(i+5) % 10}' for i in range(100)],
        'FTHG': np.random.randint(0, 4, 100),
        'FTAG': np.random.randint(0, 4, 100),
        'B365H': np.random.uniform(1.5, 5.0, 100),
        'B365D': np.random.uniform(3.0, 4.5, 100),
        'B365A': np.random.uniform(1.8, 6.0, 100),
        # Add other raw stats if needed by feature generation
        'HS': np.random.randint(5, 20, 100), 'AS': np.random.randint(5, 20, 100),
        'HST': np.random.randint(1, 10, 100), 'AST': np.random.randint(1, 10, 100),
        'HC': np.random.randint(1, 10, 100), 'AC': np.random.randint(1, 10, 100),
    })
    dummy_data['FTR'] = np.select([dummy_data['FTHG'] > dummy_data['FTAG'], dummy_data['FTHG'] < dummy_data['FTAG']], ['H', 'A'], default='D')
    print(f"Loaded dummy data: {len(dummy_data)} rows")

    # --- 2. Configure Backtest ---
    # Define models to test (ensure they exist in your registry)
    models_to_test = ['random_forest'] # Add 'gradient_boosting', 'poisson' etc. if available

    # Define specific model parameters (optional, otherwise defaults from model class are used)
    custom_model_params = {
        'random_forest': {'n_estimators': 50, 'max_depth': 8, 'random_state': config.RANDOM_SEED}
        # 'gradient_boosting': {'n_estimators': 75, 'learning_rate': 0.08}
    }

    # Define backtest windowing strategy
    bt_config = {
        'train_window_size': '180d', # Use strings or Timedeltas
        'test_window_size': '30d',
        'step_size': '30d',
        'strategy': 'rolling'
    }

    # Define betting simulation (optional)
    bet_config = {
        'enabled': True,
        'odds_cols': ['B365H', 'B365D', 'B365A'],
        'threshold': 0.05,
        'stake': 10
    }

    # --- 3. Initialize and Run Engine ---
    try:
        backtester = BacktestEngine(
            full_historical_data=dummy_data,
            date_col='Date',
            target_col='FTR',
            model_names=models_to_test,
            model_params=custom_model_params,
            backtest_config=bt_config,
            betting_config=bet_config,
            evaluation_metrics=['accuracy', 'multi_logloss'] # Add more from metrics.py
        )
        backtester.run_backtest()

        # --- 4. Get and Display Results ---
        results_df = backtester.get_results()
        print("\n--- Raw Backtest Results (Sample) ---")
        print(results_df.head())

        summary_df = backtester.summary()
        print("\n--- Backtest Summary (Aggregated by Model) ---")
        print(summary_df)

    except ImportError:
         print("\nNote: Could not run example because project components (models, utils) were not found.")
    except Exception as e:
         logging.error(f"An error occurred during the backtest example: {e}", exc_info=True)