# config.py

import os

# --- Paths ---
# Use absolute paths or paths relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Assumes config.py is at project root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'historical_matches.csv') # Example
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'features.parquet') # Example
MODEL_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
ELO_SAVE_PATH = os.path.join(MODEL_DIR, 'elo_calculator.joblib')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'raw'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# --- Feature Engineering ---
ROLLING_WINDOW_SIZE = 10 # Number of games for rolling stats/form
ELO_K_FACTOR = 25
ELO_HOME_ADVANTAGE = 65
ELO_DEFAULT_RATING = 1500
ODDS_COLUMNS = ['B365H', 'B365D', 'B365A'] # Example bookmaker odds columns

# List of features to generate and use in models (can be dynamically selected later)
# Example:
FEATURE_SET = [
    'ImpliedProbH', 'ImpliedProbD', 'ImpliedProbA', 'BookmakerMargin',
    f'Home_Avg_GoalsScored_L{ROLLING_WINDOW_SIZE}', f'Home_Avg_GoalsConceded_L{ROLLING_WINDOW_SIZE}',
    f'Away_Avg_GoalsScored_L{ROLLING_WINDOW_SIZE}', f'Away_Avg_GoalsConceded_L{ROLLING_WINDOW_SIZE}',
    # Add more rolling stats features...
    # f'HomeFormPts_L{ROLLING_WINDOW_SIZE}', f'AwayFormPts_L{ROLLING_WINDOW_SIZE}', # If form calc works
    'HomeEloBefore', 'AwayEloBefore', 'EloDiff'
]


# --- Model Hyperparameters (Defaults or options for tuning) ---
# Example for RandomForest
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# Example for XGBoost
XGB_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    # Objective/eval_metric often set based on task type in the model class
}


# --- Simulation Settings ---
MC_N_SIMULATIONS = 10000 # For Monte Carlo model


# --- Evaluation / Backtesting ---
ROI_BET_THRESHOLD = 0.05 # Minimum value edge for ROI calculation
ROI_STAKE = 1.0 # Fixed stake per bet for simple ROI calculation


# --- Other Settings ---
RANDOM_SEED = 42 # Global random seed for reproducibility where applicable


# --- You can add more sections as needed ---
# e.g., API Keys, Database Credentials (use environment variables for sensitive data!)

print("Configuration loaded.")

# Example of accessing config values in another file:
# import config
# print(config.ROLLING_WINDOW_SIZE)
# rf = RandomForestModel(**config.RF_PARAMS)