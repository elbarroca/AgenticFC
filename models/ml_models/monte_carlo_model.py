import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler # Optional but can help
from scipy.stats import poisson
import warnings
import gc # Garbage collector
import json # For storing top scenarios

warnings.filterwarnings('ignore', category=UserWarning) # Suppress XGBoost warnings if needed
pd.set_option('display.max_columns', None) # Show all columns in printouts

# --- Configuration ---
N_SIMULATIONS = 10000  # Total number of Monte Carlo simulations per match
BATCH_SIZE = 1000      # Process this many simulations at once
N_SPLITS = 8          # Number of folds for TimeSeriesSplit
TOP_N_SCENARIOS = 10   # Number of top scenarios to store per match
RANDOM_SEED = 42
USE_FLOAT16 = True     # Use float16 instead of float64 to save memory

# --- Load Data ---
print("Loading data...")
try:
    df = pd.read_parquet('/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/parquets/final_data.parquet')
    # CRITICAL: Ensure data is sorted chronologically
    # Check if 'Timestamp' column exists before sorting
    if 'Timestamp' not in df.columns:
        raise ValueError("Error: 'Timestamp' column not found in the loaded DataFrame. Cannot sort.")
    df = df.sort_values('Timestamp').reset_index(drop=True)
    print(f"Data loaded and sorted: {df.shape[0]} matches")
    
    # --- Define Evaluation Outcomes ---
    print("Defining evaluation outcomes...")
    # Check for necessary columns before creating new ones
    required_outcome_cols = ['FTR', 'FTHG', 'FTAG']
    if not all(col in df.columns for col in required_outcome_cols):
        missing_cols = [col for col in required_outcome_cols if col not in df.columns]
        raise ValueError(f"Error: Missing required columns for defining outcomes: {missing_cols}")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Define evaluation outcomes
df['Actual_FTR_Numeric'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).astype('Int64')
df['Actual_TotalGoals'] = df['FTHG'] + df['FTAG']
df['Actual_Over1.5'] = (df['Actual_TotalGoals'] > 1.5).astype(int)
df['Actual_Over2.5'] = (df['Actual_TotalGoals'] > 2.5).astype(int)
df['Actual_Under2.5'] = (df['Actual_TotalGoals'] < 2.5).astype(int)
df['Actual_Under3.5'] = (df['Actual_TotalGoals'] < 3.5).astype(int)
df['Actual_BTTS_Yes'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
df['Actual_BTTS_No'] = 1 - df['Actual_BTTS_Yes']
df['Actual_Home_WinOrDraw'] = ((df['FTR'] == 'H') | (df['FTR'] == 'D')).astype(int) # 1X
df['Actual_Away_WinOrDraw'] = ((df['FTR'] == 'A') | (df['FTR'] == 'D')).astype(int) # X2
df['Actual_Home_Win'] = (df['FTR'] == 'H').astype(int)
df['Actual_Draw'] = (df['FTR'] == 'D').astype(int)
df['Actual_Away_Win'] = (df['FTR'] == 'A').astype(int)

# Add actuals for combined scenarios we plan to predict
df['Actual_H_and_O25'] = (df['Actual_Home_Win'] & df['Actual_Over2.5']).astype(int)
df['Actual_D_and_U25'] = (df['Actual_Draw'] & df['Actual_Under2.5']).astype(int)
df['Actual_A_and_BTTS_Yes'] = (df['Actual_Away_Win'] & df['Actual_BTTS_Yes']).astype(int)
df['Actual_1X_and_U35'] = (df['Actual_Home_WinOrDraw'] & df['Actual_Under3.5']).astype(int)
df['Actual_X2_and_O15'] = (df['Actual_Away_WinOrDraw'] & df['Actual_Over1.5']).astype(int)
# Add more as needed

# --- Feature Selection ---
print("Selecting features...")
time_windows = ['Last5', 'Last10'] # Consider adding 'Last15'
home_features_base = [
    'AvgGoalsScored', 'AvgGoalsConceded', 'AvgShotsFor', 'AvgShotsAgainst',
    'AvgShotsTargetFor', 'AvgShotsTargetAgainst', 'AvgPossessionFor',
    'AvgPossessionAgainst', 'AvgCornersFor', 'AvgCornersAgainst',
    'FormPoints', 'BTTS_Ratio', 'W_Count', 'D_Count', 'L_Count'
]
home_features, away_features = [], []
for base in home_features_base:
    for window in time_windows:
        h_col, a_col = f"Home_{base}_Total_{window}", f"Away_{base}_Total_{window}"
        # Check if columns exist in DataFrame before adding
        if h_col in df.columns:
            home_features.append(h_col)
        else:
            print(f"Warning: Feature column '{h_col}' not found in DataFrame.")
        if a_col in df.columns:
            away_features.append(a_col)
        else:
             print(f"Warning: Feature column '{a_col}' not found in DataFrame.")


home_attack_features = [f for f in home_features if any(k in f for k in ['Scored', 'For', 'FormPoints', 'W_Count', 'BTTS', 'PossessionFor'])]
home_defense_features = [f for f in home_features if any(k in f for k in ['Conceded', 'Against', 'L_Count', 'PossessionAgainst'])]
away_attack_features = [f for f in away_features if any(k in f for k in ['Scored', 'For', 'FormPoints', 'W_Count', 'BTTS', 'PossessionFor'])]
away_defense_features = [f for f in away_features if any(k in f for k in ['Conceded', 'Against', 'L_Count', 'PossessionAgainst'])]

features_hg = sorted(list(set(home_attack_features + away_defense_features)))
features_ag = sorted(list(set(away_attack_features + home_defense_features)))

# Ensure features actually exist in the dataframe before proceeding
features_hg = [f for f in features_hg if f in df.columns]
features_ag = [f for f in features_ag if f in df.columns]

if not features_hg or not features_ag:
    raise ValueError("Feature list is empty. Check column names and availability in the DataFrame.")


print(f"Features for HG model: {len(features_hg)}")
print(f"Features for AG model: {len(features_ag)}")

target_hg = 'FTHG'
target_ag = 'FTAG'

# --- Preprocessing & Model Pipeline ---
preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
xgb_params = {
    'objective': 'count:poisson', 'eval_metric': 'poisson-nloglik',
    'eta': 0.05, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 1, 'gamma': 0.1, 'lambda': 1, 'alpha': 0,
    'enable_categorical': False
}
model_pipeline_hg = Pipeline(steps=[('preprocess', preprocessor), ('xgb', xgb.XGBRegressor(**xgb_params, early_stopping_rounds=20, n_estimators=500, random_state=RANDOM_SEED))])
model_pipeline_ag = Pipeline(steps=[('preprocess', preprocessor), ('xgb', xgb.XGBRegressor(**xgb_params, early_stopping_rounds=20, n_estimators=500, random_state=RANDOM_SEED + 1))])

# --- Time Series Cross-Validation ---
print("Starting Time Series Cross-Validation...")
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
all_fold_results = []

for fold, (train_index, test_index) in enumerate(tscv.split(df)):
    print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
    
    # Extract features and targets for this fold
    X_train_hg, X_test_hg = df.iloc[train_index][features_hg].copy(), df.iloc[test_index][features_hg].copy()
    y_train_hg, y_test_hg = df.iloc[train_index][target_hg].copy(), df.iloc[test_index][target_hg].copy()
    X_train_ag, X_test_ag = df.iloc[train_index][features_ag].copy(), df.iloc[test_index][features_ag].copy()
    y_train_ag, y_test_ag = df.iloc[train_index][target_ag].copy(), df.iloc[test_index][target_ag].copy()

    # Verify matching dimensions
    if X_train_hg.shape[1] != X_test_hg.shape[1]:
        print(f"Skipping fold {fold+1}: Feature count mismatch in HG model")
        continue
    if X_train_ag.shape[1] != X_test_ag.shape[1]:
        print(f"Skipping fold {fold+1}: Feature count mismatch in AG model")
        continue

    # Preprocess the data with the imputer
    X_train_hg_processed = preprocessor.fit_transform(X_train_hg)
    X_test_hg_processed = preprocessor.transform(X_test_hg)
    X_train_ag_processed = preprocessor.fit_transform(X_train_ag)
    X_test_ag_processed = preprocessor.transform(X_test_ag)

    # Home Goals Model - Train without scikit-learn pipeline
    print("Fitting Home Goal model...")
    model_hg = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=20, n_estimators=500, random_state=RANDOM_SEED)
    model_hg.fit(
        X_train_hg_processed, y_train_hg,
        eval_set=[(X_test_hg_processed, y_test_hg)],
        verbose=False
    )
    print(f"Best HG iteration: {model_hg.best_iteration}")

    # Away Goals Model - Train without scikit-learn pipeline
    print("Fitting Away Goal model...")
    model_ag = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=20, n_estimators=500, random_state=RANDOM_SEED + 1)
    model_ag.fit(
        X_train_ag_processed, y_train_ag,
        eval_set=[(X_test_ag_processed, y_test_ag)],
        verbose=False
    )
    print(f"Best AG iteration: {model_ag.best_iteration}")

    # Predict using the trained models
    print("Predicting lambdas...")
    lambda_hg_pred = np.maximum(model_hg.predict(X_test_hg_processed), 0.01)
    lambda_ag_pred = np.maximum(model_ag.predict(X_test_ag_processed), 0.01)

    print(f"Running Monte Carlo simulation ({N_SIMULATIONS} iterations in batches of {BATCH_SIZE})...")
    fold_match_results = []
    match_ids_test = df.iloc[test_index]['MatchID'].values
    
    # Initialize scenario counters - one counter per match per scenario
    num_matches = len(test_index)
    
    # List of all scenario names we'll track
    scenario_names = [
        'H', 'D', 'A', 'Over 1.5', 'Over 2.5', 'Under 2.5', 'Under 3.5', 
        'BTTS Yes', 'BTTS No', '1X', 'X2', '12',
        # Combined scenarios
        'H and O1.5', 'H and O2.5', 'H and U2.5', 'H and U3.5', 
        'H and BTTS Yes', 'H and BTTS No',
        'D and O1.5', 'D and U2.5', 'D and BTTS Yes', 'D and BTTS No',
        'A and O1.5', 'A and O2.5', 'A and U2.5', 'A and U3.5',
        'A and BTTS Yes', 'A and BTTS No',
        '1X and O1.5', '1X and O2.5', '1X and U2.5', '1X and U3.5',
        '1X and BTTS Yes', '1X and BTTS No',
        'X2 and O1.5', 'X2 and O2.5', 'X2 and U2.5', 'X2 and U3.5',
        'X2 and BTTS Yes', 'X2 and BTTS No',
        '12 and O1.5', '12 and O2.5', '12 and U2.5',
        '12 and BTTS Yes', '12 and BTTS No',
        'Over 2.5 and BTTS Yes', 'Under 2.5 and BTTS No'
    ]
    
    # Initialize counters for all scenarios using int64 to avoid overflow
    scenario_counts = {scenario: np.zeros(num_matches, dtype=np.int64) 
                      for scenario in scenario_names}
    
    # Process in batches
    for batch_start in range(0, N_SIMULATIONS, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N_SIMULATIONS)
        batch_size = batch_end - batch_start
        
        # Generate simulations for this batch
        dtype = np.float16 if USE_FLOAT16 else np.float64
        sim_hg_batch = poisson.rvs(mu=lambda_hg_pred[:, np.newaxis], size=(num_matches, batch_size)).astype(dtype)
        sim_ag_batch = poisson.rvs(mu=lambda_ag_pred[:, np.newaxis], size=(num_matches, batch_size)).astype(dtype)
        sim_total_goals = sim_hg_batch + sim_ag_batch
        
        # Calculate conditions for basic outcomes
        cond_H = sim_hg_batch > sim_ag_batch
        cond_D = sim_hg_batch == sim_ag_batch
        cond_A = sim_hg_batch < sim_ag_batch
        cond_O15 = sim_total_goals > 1.5
        cond_O25 = sim_total_goals > 2.5
        cond_U25 = sim_total_goals < 2.5
        cond_U35 = sim_total_goals < 3.5
        cond_BTTS_Yes = (sim_hg_batch > 0) & (sim_ag_batch > 0)
        cond_BTTS_No = ~cond_BTTS_Yes
        cond_1X = cond_H | cond_D
        cond_X2 = cond_A | cond_D
        cond_12 = cond_H | cond_A
        
        # Create dictionary mapping scenario names to condition arrays
        scenario_conditions = {
            'H': cond_H,
            'D': cond_D,
            'A': cond_A,
            'Over 1.5': cond_O15,
            'Over 2.5': cond_O25,
            'Under 2.5': cond_U25,
            'Under 3.5': cond_U35,
            'BTTS Yes': cond_BTTS_Yes,
            'BTTS No': cond_BTTS_No,
            '1X': cond_1X,
            'X2': cond_X2,
            '12': cond_12,
            # Combined scenarios
            'H and O1.5': cond_H & cond_O15,
            'H and O2.5': cond_H & cond_O25,
            'H and U2.5': cond_H & cond_U25,
            'H and U3.5': cond_H & cond_U35,
            'H and BTTS Yes': cond_H & cond_BTTS_Yes,
            'H and BTTS No': cond_H & cond_BTTS_No,
            'D and O1.5': cond_D & cond_O15,
            'D and U2.5': cond_D & cond_U25,
            'D and BTTS Yes': cond_D & cond_BTTS_Yes,
            'D and BTTS No': cond_D & cond_BTTS_No,
            'A and O1.5': cond_A & cond_O15,
            'A and O2.5': cond_A & cond_O25,
            'A and U2.5': cond_A & cond_U25,
            'A and U3.5': cond_A & cond_U35,
            'A and BTTS Yes': cond_A & cond_BTTS_Yes,
            'A and BTTS No': cond_A & cond_BTTS_No,
            '1X and O1.5': cond_1X & cond_O15,
            '1X and O2.5': cond_1X & cond_O25,
            '1X and U2.5': cond_1X & cond_U25,
            '1X and U3.5': cond_1X & cond_U35,
            '1X and BTTS Yes': cond_1X & cond_BTTS_Yes,
            '1X and BTTS No': cond_1X & cond_BTTS_No,
            'X2 and O1.5': cond_X2 & cond_O15,
            'X2 and O2.5': cond_X2 & cond_O25,
            'X2 and U2.5': cond_X2 & cond_U25,
            'X2 and U3.5': cond_X2 & cond_U35,
            'X2 and BTTS Yes': cond_X2 & cond_BTTS_Yes,
            'X2 and BTTS No': cond_X2 & cond_BTTS_No,
            '12 and O1.5': cond_12 & cond_O15,
            '12 and O2.5': cond_12 & cond_O25,
            '12 and U2.5': cond_12 & cond_U25,
            '12 and BTTS Yes': cond_12 & cond_BTTS_Yes,
            '12 and BTTS No': cond_12 & cond_BTTS_No,
            'Over 2.5 and BTTS Yes': cond_O25 & cond_BTTS_Yes,
            'Under 2.5 and BTTS No': cond_U25 & cond_BTTS_No,
        }
        
        # Update counters for each scenario
        for scenario, condition in scenario_conditions.items():
            # Sum across batch dimension (axis=1) to count occurrences for each match
            scenario_counts[scenario] += np.sum(condition, axis=1)
        
        # Clean up batch variables to free memory
        del sim_hg_batch, sim_ag_batch, sim_total_goals
        del cond_H, cond_D, cond_A, cond_O15, cond_O25, cond_U25, cond_U35
        del cond_BTTS_Yes, cond_BTTS_No, cond_1X, cond_X2, cond_12
        del scenario_conditions
        gc.collect()
        
        # Optional progress indicator
        if (batch_start + BATCH_SIZE) % (N_SIMULATIONS // 5) == 0 or batch_end == N_SIMULATIONS:
            print(f"  Processed {batch_end}/{N_SIMULATIONS} simulations ({batch_end/N_SIMULATIONS:.1%})")
    
    # Calculate final probabilities
    scenario_probs = {scenario: counts / N_SIMULATIONS for scenario, counts in scenario_counts.items()}
    
    # --- Assemble results for each match ---
    for i in range(num_matches):
        match_results = {
            'MatchID': match_ids_test[i],
            'Fold': fold + 1,
            'lambda_hg': lambda_hg_pred[i],
            'lambda_ag': lambda_ag_pred[i],
        }

        # Add individual probabilities
        match_results['P_H'] = scenario_probs['H'][i]
        match_results['P_D'] = scenario_probs['D'][i]
        match_results['P_A'] = scenario_probs['A'][i]
        match_results['P_Over1.5'] = scenario_probs['Over 1.5'][i]
        match_results['P_Over2.5'] = scenario_probs['Over 2.5'][i]
        match_results['P_Under2.5'] = scenario_probs['Under 2.5'][i]
        match_results['P_Under3.5'] = scenario_probs['Under 3.5'][i]
        match_results['P_BTTS_Yes'] = scenario_probs['BTTS Yes'][i]
        match_results['P_BTTS_No'] = scenario_probs['BTTS No'][i]
        match_results['P_1X'] = scenario_probs['1X'][i]
        match_results['P_X2'] = scenario_probs['X2'][i]
        match_results['P_12'] = scenario_probs['12'][i]

        # Add combined scenario probabilities
        for label in scenario_probs:
            if label not in ['H', 'D', 'A', 'Over 1.5', 'Over 2.5', 'Under 2.5', 'Under 3.5', 'BTTS Yes', 'BTTS No', '1X', 'X2', '12']:
                match_results[f"P_{label.replace(' ', '_').replace('.', '')}"] = scenario_probs[label][i]

        # --- Find Top N Scenarios ---
        # Get probabilities for combined scenarios for this specific match
        current_match_combined_probs = {
            label: prob[i] for label, prob in scenario_probs.items()
        }

        # Sort scenarios by probability (descending)
        sorted_scenarios = sorted(current_match_combined_probs.items(), key=lambda item: item[1], reverse=True)

        # Store Top N as JSON string
        top_scenarios_list = [(label, round(prob, 4)) for label, prob in sorted_scenarios[:TOP_N_SCENARIOS]]
        match_results['Top_Scenarios'] = json.dumps(top_scenarios_list)

        fold_match_results.append(match_results)

    all_fold_results.extend(fold_match_results)
    print(f"Fold {fold + 1} completed.")
    del X_train_hg, X_test_hg, y_train_hg, y_test_hg, X_train_ag, X_test_ag, y_train_ag, y_test_ag
    del lambda_hg_pred, lambda_ag_pred, scenario_counts, scenario_probs
    gc.collect()


# --- Combine Results & Evaluate ---
print("\nCombining results and evaluating...")
results_df = pd.DataFrame(all_fold_results)

# Define columns to merge from original df
actual_cols_to_merge = [
    'MatchID', 'Actual_FTR_Numeric', 'Actual_Over1.5', 'Actual_Over2.5',
    'Actual_Under2.5', 'Actual_Under3.5', 'Actual_BTTS_Yes', 'Actual_BTTS_No',
    'Actual_Home_WinOrDraw', 'Actual_Away_WinOrDraw', 'Actual_Home_Win', 'Actual_Draw', 'Actual_Away_Win',
    # Add combined actuals if you want to evaluate them directly
    'Actual_H_and_O25', 'Actual_D_and_U25', 'Actual_A_and_BTTS_Yes', 'Actual_1X_and_U35', 'Actual_X2_and_O15'
]
actuals_to_merge = df[actual_cols_to_merge].copy()

# Merge results with actuals
final_eval_df = pd.merge(results_df, actuals_to_merge, on='MatchID', how='left')

# Drop rows where essential actual outcomes might be missing
essential_actuals = ['Actual_FTR_Numeric', 'Actual_Over2.5', 'Actual_BTTS_Yes', 'Actual_Over1.5', 'Actual_Under3.5']
final_eval_df = final_eval_df.dropna(subset=essential_actuals)
final_eval_df['Actual_FTR_Numeric'] = final_eval_df['Actual_FTR_Numeric'].astype(int)

if final_eval_df.empty:
     print("Error: No matching predictions and actuals found after merge. Check MatchIDs or data integrity.")
else:
    print(f"Evaluating on {len(final_eval_df)} matches.")

    # --- Calculate Average Metrics ---
    avg_pred_cols = [col for col in final_eval_df.columns if col.startswith('P_') or col.startswith('lambda_')]
    avg_metrics = final_eval_df[avg_pred_cols].mean()
    print("\n--- Average Predicted Metrics ---")
    print(avg_metrics)

    # Calculate actual frequencies for comparison
    actual_freq_cols = [col for col in final_eval_df.columns if col.startswith('Actual_')]
    actual_freq = final_eval_df[actual_freq_cols].mean().drop('Actual_FTR_Numeric') # Drop numeric FTR mean
    actual_freq['Actual_H'] = (final_eval_df['Actual_FTR_Numeric'] == 0).mean()
    actual_freq['Actual_D'] = (final_eval_df['Actual_FTR_Numeric'] == 1).mean()
    actual_freq['Actual_A'] = (final_eval_df['Actual_FTR_Numeric'] == 2).mean()
    print("\n--- Actual Frequencies in Test Sets ---")
    print(actual_freq.sort_index())

    # --- Evaluation Metrics ---
    print("\n--- Performance Evaluation ---")
    # Brier Score (Lower is better)
    brier_over15 = brier_score_loss(final_eval_df['Actual_Over1.5'], final_eval_df['P_Over1.5'])
    brier_over25 = brier_score_loss(final_eval_df['Actual_Over2.5'], final_eval_df['P_Over2.5'])
    brier_under25 = brier_score_loss(final_eval_df['Actual_Under2.5'], 1.0 - final_eval_df['P_Over2.5']) # Or use P_Under2.5 directly
    brier_under35 = brier_score_loss(final_eval_df['Actual_Under3.5'], final_eval_df['P_Under3.5'])
    brier_btts_yes = brier_score_loss(final_eval_df['Actual_BTTS_Yes'], final_eval_df['P_BTTS_Yes'])
    brier_1X = brier_score_loss(final_eval_df['Actual_Home_WinOrDraw'], final_eval_df['P_1X'])
    brier_X2 = brier_score_loss(final_eval_df['Actual_Away_WinOrDraw'], final_eval_df['P_X2'])


    print(f"Brier Score (Over 1.5):  {brier_over15:.4f}")
    print(f"Brier Score (Over 2.5):  {brier_over25:.4f}")
    print(f"Brier Score (Under 2.5): {brier_under25:.4f}")
    print(f"Brier Score (Under 3.5): {brier_under35:.4f}")
    print(f"Brier Score (BTTS Yes):  {brier_btts_yes:.4f}")
    print(f"Brier Score (1X):        {brier_1X:.4f}")
    print(f"Brier Score (X2):        {brier_X2:.4f}")


    # Log Loss for HDA (Lower is better)
    hda_probs = final_eval_df[['P_H', 'P_D', 'P_A']].values
    hda_actuals = final_eval_df['Actual_FTR_Numeric'].values
    hda_probs = np.clip(hda_probs, 1e-15, 1 - 1e-15) # Clipping for numerical stability
    if hda_probs.sum(axis=1).min() > 0: # Avoid division by zero if a row sums to 0
        hda_probs /= hda_probs.sum(axis=1)[:, np.newaxis]
        logloss_hda = log_loss(hda_actuals, hda_probs, labels=[0, 1, 2])
        print(f"Log Loss (HDA): {logloss_hda:.4f}")
    else:
        print("Warning: Could not calculate Log Loss due to probability sum being zero for some rows.")


    # --- Display Top Scenarios for a Sample Match ---
    print("\n--- Example Top Scenarios (Last Match in Eval Set) ---")
    example_match = final_eval_df.iloc[-1]
    print(f"MatchID: {example_match['MatchID']}")
    print(f"Predicted Lambda HG: {example_match['lambda_hg']:.2f}, AG: {example_match['lambda_ag']:.2f}")
    top_scenarios = json.loads(example_match['Top_Scenarios'])
    for scenario, prob in top_scenarios:
        print(f"- {scenario}: {prob:.4f}")


    print("\n--- Simulation Complete ---")
    # Save results if needed
    # final_eval_df.to_csv('monte_carlo_results_expanded.csv', index=False)
    # print("Results saved to monte_carlo_results_expanded.csv")