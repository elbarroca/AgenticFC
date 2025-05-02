# Run backtest with Monte Carlo model
import pandas as pd
import numpy as np
import os
import sys
import importlib
import logging
from tqdm.auto import tqdm
import time

# Fix the relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_registry import list_available_models, get_model_class
from wrappers.monte_carlo import MonteCarloModel

# Add utils to path
utils_path = os.path.join(os.path.dirname(__file__), 'utils')
sys.path.append(utils_path)
from utils.backtest_engine import BacktestEngine

# Create a direct reference to MonteCarloModel
MODEL_CLASSES = {'monte_carlo': MonteCarloModel}

# Check if Monte Carlo model is available
available_models = list_available_models()
print(f"Available models: {available_models}")
if 'monte_carlo' not in available_models:
    print("WARNING: Monte Carlo model not available in registry! Using direct reference.")

# Load your data
data = pd.read_parquet('/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/parquets/final_data.parquet')
print(f"Loaded data with {len(data)} rows")

# Define our feature generation function
def generate_features(df):
    """Simple feature generator as a fallback"""
    # Create a copy to avoid modifying the original
    features_df = df.copy()
    feature_cols = [col for col in df.columns if 
                   (col.startswith('Home_') or col.startswith('Away_'))]
    print(f"Using {len(feature_cols)} existing feature columns from data")
    return features_df

# Add this function at the top of the file
def calculate_metrics(predictions_df, y_test):
    """Calculate evaluation metrics from predictions and actual values"""
    metrics = {}
    
    # Calculate accuracy (most likely outcome vs actual)
    if 'prediction' in predictions_df.columns and 'FTR' in y_test.columns:
        metrics['accuracy'] = (predictions_df['prediction'] == y_test['FTR']).mean()
    
    # Calculate probabilities metrics
    if all(col in predictions_df.columns for col in ['prob_H', 'prob_D', 'prob_A']):
        # Create actual one-hot encoding
        actual_h = (y_test['FTR'] == 'H').astype(int)
        actual_d = (y_test['FTR'] == 'D').astype(int) 
        actual_a = (y_test['FTR'] == 'A').astype(int)
        
        # Calculate Brier scores (lower is better)
        metrics['brier_h'] = ((predictions_df['prob_H'] - actual_h) ** 2).mean()
        metrics['brier_d'] = ((predictions_df['prob_D'] - actual_d) ** 2).mean() 
        metrics['brier_a'] = ((predictions_df['prob_A'] - actual_a) ** 2).mean()
        metrics['brier_score'] = (metrics['brier_h'] + metrics['brier_d'] + metrics['brier_a']) / 3
        
        # Calculate ROI if odds are present
        if all(col in y_test.index.to_frame().columns for col in ['B365H', 'B365D', 'B365A']):
            # Calculate expected value
            ev_h = predictions_df['prob_H'] * y_test['B365H'] - 1
            ev_d = predictions_df['prob_D'] * y_test['B365D'] - 1
            ev_a = predictions_df['prob_A'] * y_test['B365A'] - 1
            
            # Count positive expected value bets
            positive_ev_count = ((ev_h > 0.05) | (ev_d > 0.05) | (ev_a > 0.05)).sum()
            metrics['positive_ev_bets'] = positive_ev_count
    
    # Add other metrics here
    
    return metrics

def custom_run_backtest(self):
    """Completely rewritten run_backtest method that doesn't rely on imported functions"""
    print("Starting backtest with custom implementation...")
    self.results = []  # Clear previous results
    
    # Generate time splits
    time_splits = self._get_time_splits()
    if not time_splits:
        print("No time splits generated. Backtest cannot run.")
        return

    # Main Backtesting Loop
    for i, (train_start, train_end, test_start, test_end) in enumerate(tqdm(time_splits, desc="Backtest Progress")):
        split_info = f"Split {i+1}/{len(time_splits)}: Train [{train_start.date()}-{train_end.date()}] Test [{test_start.date()}-{test_end.date()}]"
        print(split_info)

        # Get data indices for the current split
        train_indices = self.full_data[
            (self.full_data[self.date_col] >= train_start) &
            (self.full_data[self.date_col] < train_end)
        ].index
        test_indices = self.full_data[
            (self.full_data[self.date_col] >= test_start) &
            (self.full_data[self.date_col] < test_end)
        ].index

        if train_indices.empty or test_indices.empty:
            print(f"Skipping split {i+1} due to empty train or test set.")
            continue

        # Prepare Data (Features & Target)
        try:
            train_data = self.full_data.loc[train_indices].copy()
            test_data = self.full_data.loc[test_indices].copy()

            # Generate features
            combined_data = pd.concat([train_data, test_data])
            all_features_df = generate_features(combined_data)

            # Separate Train/Test Features and Targets
            X_train = all_features_df.loc[train_indices]
            X_test = all_features_df.loc[test_indices]
            
            # Special handling for Monte Carlo model
            y_train = train_data[['FTR', 'FTHG', 'FTAG']] 
            y_test = test_data[['FTR', 'FTHG', 'FTAG']]

            # Handle NaNs by data type
            for df in [X_train, X_test]:
                # Fill boolean columns with False
                bool_cols = df.select_dtypes(include=['bool']).columns
                for col in bool_cols:
                    df[col] = df[col].fillna(False)
                
                # Fill numeric columns with 0
                num_cols = df.select_dtypes(include=['number']).columns
                for col in num_cols:
                    df[col] = df[col].fillna(0)
                
                # Fill object columns with empty string
                obj_cols = df.select_dtypes(include=['object']).columns
                for col in obj_cols:
                    df[col] = df[col].fillna('')
                    
        except Exception as e:
            print(f"Error preparing data for split {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Get original test data for results merging
        test_data_orig = self.full_data.loc[test_indices].copy()

        # Process each model
        for model_name in self.model_names:
            model_start_time = time.time()
            print(f"  Processing model: {model_name}")

            try:
                # Get model class directly from our dictionary
                if model_name in MODEL_CLASSES:
                    ModelClass = MODEL_CLASSES[model_name]
                else:
                    print(f"Model {model_name} not found in MODEL_CLASSES dictionary")
                    continue

                # Get parameters and instantiate model
                params = self.model_params.get(model_name, {})
                model_instance = ModelClass(**params)

                # Train the model
                print(f"    Training {model_name} with {len(X_train)} samples...")
                model_instance.fit(X_train, y_train)

                # Make predictions
                print(f"    Predicting with {model_name} on {len(X_test)} samples...")
                predictions_df = model_instance.predict(X_test)

                # Calculate metrics
                metrics_results = calculate_metrics(predictions_df, y_test)
                print(f"    Metrics: Accuracy={metrics_results.get('accuracy', 0):.4f}, Brier={metrics_results.get('brier_score', 0):.4f}")

                model_end_time = time.time()
                print(f"    {model_name} processing took {model_end_time - model_start_time:.2f}s")

                # Add results
                self.results.append({
                    'split': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'model_name': model_name,
                    'num_train_samples': len(X_train),
                    'num_test_samples': len(X_test),
                    # Add actual metrics instead of placeholders
                    **metrics_results
                })

            except Exception as e:
                print(f"    Error processing model {model_name} for split {i+1}: {e}")
                import traceback
                traceback.print_exc()
                # Add error record
                self.results.append({
                    'split': i + 1, 
                    'model_name': model_name, 
                    'status': 'FAILED', 
                    'error': str(e)
                })

    print("Backtest complete!")
    return self.results

# Replace the method
BacktestEngine.run_backtest = custom_run_backtest

# Configure backtest parameters
backtest_config = {
    'train_window_size': '180d',  # 6 months of training data (faster)
    'test_window_size': '30d',    # Test on next month
    'step_size': '60d',           # Move forward 2 months each time (fewer splits)
    'strategy': 'rolling'         # Use rolling window strategy
}

# Configure model parameters - use fewer simulations for testing
model_params = {
    'monte_carlo': {
        'n_simulations': 10000,    # Fewer simulations for faster testing
        'batch_size': 500,        # Process in batches of 500
        'use_float16': True,      # Use float16 to save memory
    }
}

# Configure evaluation metrics
metrics = ['accuracy', 'multi_logloss', 'brier_score']

# Configure betting simulation - only use available odds columns
odds_cols = [col for col in data.columns if 
             any(col.startswith(prefix) for prefix in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC']) and
             any(col.endswith(suffix) for suffix in ['H', 'D', 'A'])][:3]

betting_config = {
    'enabled': len(odds_cols) >= 3,
    'odds_cols': odds_cols if len(odds_cols) >= 3 else ['B365H', 'B365D', 'B365A'],
    'threshold': 0.05,
    'stake': 1.0
}

print(f"Using odds columns: {betting_config['odds_cols']}")

# Define get_results method
def get_results(self):
    """Returns the results of the backtest"""
    if hasattr(self, "results") and self.results:
        return pd.DataFrame(self.results)
    else:
        print("No results found.")
        return pd.DataFrame()

# Define summary method
def summary(self, group_by='model_name'):
    """Summarizes the results of the backtest"""
    results_df = self.get_results()
    if results_df.empty:
        return pd.DataFrame()
        
    if group_by not in results_df.columns:
        print(f"Column '{group_by}' not found in results. Available columns: {results_df.columns.tolist()}")
        return pd.DataFrame()
        
    numeric_cols = results_df.select_dtypes(include=np.number).columns
    summary_df = results_df.groupby(group_by)[numeric_cols].mean()
    
    return summary_df.round(4)

# Add methods to the class
BacktestEngine.get_results = get_results
BacktestEngine.summary = summary

# Initialize backtest engine
backtest = BacktestEngine(
    full_historical_data=data,
    date_col='Date',              # Date column for time splits
    target_col='FTR',             # Target column for evaluation (match result)
    model_names=['monte_carlo'],  # Use monte_carlo model
    model_params=model_params,    # Pass model parameters
    backtest_config=backtest_config,
    evaluation_metrics=metrics,
    betting_config=betting_config
)

# Run backtest
print("Starting backtest...")
results = backtest.run_backtest()

# Get results
print("\nBacktest Results:")
results_df = backtest.get_results()
if not results_df.empty:
    print(results_df.head())
else:
    print("No results found.")

# Print summary
try:
    summary_df = backtest.summary()
    print("\nSummary by Model:")
    print(summary_df)
except Exception as e:
    print(f"Error generating summary: {e}")