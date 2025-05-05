import sys
import pandas as pd
import numpy as np
import lightgbm as lgb # Using LightGBM
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler # Optional
import warnings
import gc # Garbage collector

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

# --- Configuration ---
N_SPLITS = 8           # Number of folds for TimeSeriesSplit
RANDOM_SEED = 42
# Define which outcomes to model directly
# Add more 'Actual_...' columns here if defined earlier and desired
TARGET_COLUMNS = {
    # Basic match outcomes
    'HDA': 'Actual_FTR_Numeric', # Multiclass (Home/Draw/Away)
    'Home_Win': 'Actual_Home_Win', # Home win only
    'Draw': 'Actual_Draw', # Draw only  
    'Away_Win': 'Actual_Away_Win', # Away win only
    'Home_WinOrDraw': 'Actual_Home_WinOrDraw', # 1X
    'Away_WinOrDraw': 'Actual_Away_WinOrDraw', # X2
    
    # Goals totals
    'Over0.5': 'Actual_Over0.5',
    'Over1.5': 'Actual_Over1.5',
    'Over2.5': 'Actual_Over2.5', 
    'Over3.5': 'Actual_Over3.5',
    'Over4.5': 'Actual_Over4.5',
    'Under0.5': 'Actual_Under0.5',
    'Under1.5': 'Actual_Under1.5',
    'Under2.5': 'Actual_Under2.5',
    'Under3.5': 'Actual_Under3.5',
    'Under4.5': 'Actual_Under4.5',
    
    # Both teams to score
    'BTTS_Yes': 'Actual_BTTS_Yes',
    'BTTS_No': 'Actual_BTTS_No',
    
    # Combined outcomes
    'H_and_O15': 'Actual_H_and_O15', # Home & Over 1.5
    'H_and_O25': 'Actual_H_and_O25', # Home & Over 2.5
    'H_and_O35': 'Actual_H_and_O35', # Home & Over 3.5
    'H_and_U25': 'Actual_H_and_U25', # Home & Under 2.5
    'H_and_U35': 'Actual_H_and_U35', # Home & Under 3.5
    'H_and_BTTS': 'Actual_H_and_BTTS', # Home & BTTS
    
    'D_and_O15': 'Actual_D_and_O15', # Draw & Over 1.5
    'D_and_O25': 'Actual_D_and_O25', # Draw & Over 2.5
    'D_and_U25': 'Actual_D_and_U25', # Draw & Under 2.5
    'D_and_BTTS': 'Actual_D_and_BTTS', # Draw & BTTS
    
    'A_and_O15': 'Actual_A_and_O15', # Away & Over 1.5
    'A_and_O25': 'Actual_A_and_O25', # Away & Over 2.5
    'A_and_O35': 'Actual_A_and_O35', # Away & Over 3.5
    'A_and_U25': 'Actual_A_and_U25', # Away & Under 2.5
    'A_and_U35': 'Actual_A_and_U35', # Away & Under 3.5
    'A_and_BTTS': 'Actual_A_and_BTTS', # Away & BTTS
    
    # Double chance combinations
    '1X_and_O15': 'Actual_1X_and_O15', # Home/Draw & Over 1.5
    '1X_and_O25': 'Actual_1X_and_O25', # Home/Draw & Over 2.5
    '1X_and_U25': 'Actual_1X_and_U25', # Home/Draw & Under 2.5
    '1X_and_U35': 'Actual_1X_and_U35', # Home/Draw & Under 3.5
    '1X_and_BTTS': 'Actual_1X_and_BTTS', # Home/Draw & BTTS
    
    'X2_and_O15': 'Actual_X2_and_O15', # Draw/Away & Over 1.5
    'X2_and_O25': 'Actual_X2_and_O25', # Draw/Away & Over 2.5
    'X2_and_U25': 'Actual_X2_and_U25', # Draw/Away & Under 2.5
    'X2_and_U35': 'Actual_X2_and_U35', # Draw/Away & Under 3.5
    'X2_and_BTTS': 'Actual_X2_and_BTTS', # Draw/Away & BTTS
}

# --- Load Data ---
print("Loading data...")
try:
    # *** Make sure this path is correct ***
    file_path = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/parquets/final_data.parquet'
    df = pd.read_parquet(file_path)
    print(f"Successfully loaded data from: {file_path}")

    if 'Timestamp' not in df.columns:
        raise ValueError("Error: 'Timestamp' column not found.")
    df = df.sort_values('Timestamp').reset_index(drop=True)
    print(f"Data loaded and sorted: {df.shape[0]} matches")

    required_outcome_cols = ['FTR', 'FTHG', 'FTAG']
    if not all(col in df.columns for col in required_outcome_cols):
        missing_cols = [col for col in required_outcome_cols if col not in df.columns]
        raise ValueError(f"Error: Missing required columns for defining base outcomes: {missing_cols}")

except Exception as e:
    print(f"Error during data loading or initial checks: {e}")
    sys.exit(1)

# --- Define Evaluation Outcomes (Targets) ---
print("Defining evaluation outcomes...")

# Basic match outcomes
df['Actual_FTR_Numeric'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).astype('Int64')
df['Actual_Home_Win'] = (df['FTR'] == 'H').astype(int)
df['Actual_Draw'] = (df['FTR'] == 'D').astype(int)
df['Actual_Away_Win'] = (df['FTR'] == 'A').astype(int)
df['Actual_Home_WinOrDraw'] = ((df['FTR'] == 'H') | (df['FTR'] == 'D')).astype(int)
df['Actual_Away_WinOrDraw'] = ((df['FTR'] == 'A') | (df['FTR'] == 'D')).astype(int)

# Goals-related outcomes
df['Actual_TotalGoals'] = df['FTHG'] + df['FTAG']
# Over/Under goals
df['Actual_Over0.5'] = (df['Actual_TotalGoals'] > 0.5).astype(int)
df['Actual_Over1.5'] = (df['Actual_TotalGoals'] > 1.5).astype(int)
df['Actual_Over2.5'] = (df['Actual_TotalGoals'] > 2.5).astype(int)
df['Actual_Over3.5'] = (df['Actual_TotalGoals'] > 3.5).astype(int)
df['Actual_Over4.5'] = (df['Actual_TotalGoals'] > 4.5).astype(int)
df['Actual_Under0.5'] = (df['Actual_TotalGoals'] < 0.5).astype(int)
df['Actual_Under1.5'] = (df['Actual_TotalGoals'] < 1.5).astype(int)
df['Actual_Under2.5'] = (df['Actual_TotalGoals'] < 2.5).astype(int)
df['Actual_Under3.5'] = (df['Actual_TotalGoals'] < 3.5).astype(int)
df['Actual_Under4.5'] = (df['Actual_TotalGoals'] < 4.5).astype(int)

# BTTS (Both Teams To Score)
df['Actual_BTTS_Yes'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
df['Actual_BTTS_No'] = 1 - df['Actual_BTTS_Yes']

# Combined outcomes - Home Win combinations
df['Actual_H_and_O15'] = (df['Actual_Home_Win'] & df['Actual_Over1.5']).astype(int)
df['Actual_H_and_O25'] = (df['Actual_Home_Win'] & df['Actual_Over2.5']).astype(int)
df['Actual_H_and_O35'] = (df['Actual_Home_Win'] & df['Actual_Over3.5']).astype(int)
df['Actual_H_and_U25'] = (df['Actual_Home_Win'] & df['Actual_Under2.5']).astype(int)
df['Actual_H_and_U35'] = (df['Actual_Home_Win'] & df['Actual_Under3.5']).astype(int)
df['Actual_H_and_BTTS'] = (df['Actual_Home_Win'] & df['Actual_BTTS_Yes']).astype(int)

# Combined outcomes - Draw combinations
df['Actual_D_and_O15'] = (df['Actual_Draw'] & df['Actual_Over1.5']).astype(int)
df['Actual_D_and_O25'] = (df['Actual_Draw'] & df['Actual_Over2.5']).astype(int)
df['Actual_D_and_U25'] = (df['Actual_Draw'] & df['Actual_Under2.5']).astype(int)
df['Actual_D_and_BTTS'] = (df['Actual_Draw'] & df['Actual_BTTS_Yes']).astype(int)

# Combined outcomes - Away Win combinations
df['Actual_A_and_O15'] = (df['Actual_Away_Win'] & df['Actual_Over1.5']).astype(int)
df['Actual_A_and_O25'] = (df['Actual_Away_Win'] & df['Actual_Over2.5']).astype(int)
df['Actual_A_and_O35'] = (df['Actual_Away_Win'] & df['Actual_Over3.5']).astype(int)
df['Actual_A_and_U25'] = (df['Actual_Away_Win'] & df['Actual_Under2.5']).astype(int)
df['Actual_A_and_U35'] = (df['Actual_Away_Win'] & df['Actual_Under3.5']).astype(int)
df['Actual_A_and_BTTS'] = (df['Actual_Away_Win'] & df['Actual_BTTS_Yes']).astype(int)

# Double chance combinations
df['Actual_1X_and_O15'] = (df['Actual_Home_WinOrDraw'] & df['Actual_Over1.5']).astype(int)
df['Actual_1X_and_O25'] = (df['Actual_Home_WinOrDraw'] & df['Actual_Over2.5']).astype(int)
df['Actual_1X_and_U25'] = (df['Actual_Home_WinOrDraw'] & df['Actual_Under2.5']).astype(int)
df['Actual_1X_and_U35'] = (df['Actual_Home_WinOrDraw'] & df['Actual_Under3.5']).astype(int)
df['Actual_1X_and_BTTS'] = (df['Actual_Home_WinOrDraw'] & df['Actual_BTTS_Yes']).astype(int)

df['Actual_X2_and_O15'] = (df['Actual_Away_WinOrDraw'] & df['Actual_Over1.5']).astype(int)
df['Actual_X2_and_O25'] = (df['Actual_Away_WinOrDraw'] & df['Actual_Over2.5']).astype(int)
df['Actual_X2_and_U25'] = (df['Actual_Away_WinOrDraw'] & df['Actual_Under2.5']).astype(int)
df['Actual_X2_and_U35'] = (df['Actual_Away_WinOrDraw'] & df['Actual_Under3.5']).astype(int)
df['Actual_X2_and_BTTS'] = (df['Actual_Away_WinOrDraw'] & df['Actual_BTTS_Yes']).astype(int)

# Check if all target columns exist
missing_targets = [tgt for tgt in TARGET_COLUMNS.values() if tgt not in df.columns]
if missing_targets:
    print(f"Error: The following target columns defined in TARGET_COLUMNS are missing from the DataFrame: {missing_targets}")
    sys.exit(1)

# --- Feature Selection ---
print("Selecting features...")
time_windows = ['Last5', 'Last10']
home_features_base = [
    'AvgGoalsScored', 'AvgGoalsConceded', 'AvgShotsFor', 'AvgShotsAgainst',
    'AvgShotsTargetFor', 'AvgShotsTargetAgainst', 'AvgPossessionFor',
    'AvgPossessionAgainst', 'AvgCornersFor', 'AvgCornersAgainst',
    'FormPoints', 'BTTS_Ratio', 'W_Count', 'D_Count', 'L_Count'
]

# First, let's see what columns we actually have
print("\nChecking available features in DataFrame:")
print("Total columns in DataFrame:", len(df.columns))
print("Available columns:", sorted(df.columns.tolist()))

# Initialize lists for valid features
valid_features = []

# Check each potential feature and only include if it exists
for base in home_features_base:
    for window in time_windows:
        for team in ['Home', 'Away']:
            feature = f"{team}_{base}_Total_{window}"
            if feature in df.columns:
                valid_features.append(feature)
            else:
                print(f"Warning: Expected feature '{feature}' not found in DataFrame")

# Use only valid features
all_features = sorted(valid_features)

if not all_features:
    raise ValueError("No valid features found in DataFrame. Check feature naming convention.")

print(f"\nValid features found: {len(all_features)}")
print("Features to be used:")
for f in all_features:
    print(f"- {f}")

# --- Preprocessing & Model Definition ---
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    # ('scaler', StandardScaler()) # Optional for LightGBM
])

# LightGBM parameters (tune these for better performance)
lgbm_params_binary = {
    'objective': 'binary',
    'metric': 'binary_logloss', # Logloss is good for probability calibration
    'n_estimators': 1000, # High number, use early stopping
    'learning_rate': 0.05,
    'feature_fraction': 0.8, # alias: colsample_bytree
    'bagging_fraction': 0.8, # alias: subsample
    'bagging_freq': 1,
    'lambda_l1': 0.1, # L1 reg
    'lambda_l2': 0.1, # L2 reg
    'num_leaves': 31, # Default
    'verbose': -1, # Suppress verbose output
    'n_jobs': -1, # Use all cores
    'seed': RANDOM_SEED,
    'boosting_type': 'gbdt',
}

lgbm_params_multi = lgbm_params_binary.copy()
lgbm_params_multi['objective'] = 'multiclass'
lgbm_params_multi['metric'] = 'multi_logloss'
lgbm_params_multi['num_class'] = 3 # For H, D, A

# --- Time Series Cross-Validation ---
print("Starting Time Series Cross-Validation...")
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
all_fold_results = []
trained_models = {} # To store models from the last fold if needed

for fold, (train_index, test_index) in enumerate(tscv.split(df)):
    print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
    print(f"Train size: {len(train_index)}, Test size: {len(test_index)}")

    # Prepare feature data for this fold
    # Fit preprocessor on training data ONLY
    X_train = preprocessor.fit_transform(df.iloc[train_index][all_features])
    X_test = preprocessor.transform(df.iloc[test_index][all_features])

    fold_predictions = {'MatchID': df.iloc[test_index]['MatchID'].values}
    fold_models = {} # Store models trained in this fold

    # Train a model for each target
    for model_name, target_col in TARGET_COLUMNS.items():
        print(f"  Training model for: {model_name} (Target: {target_col})...")
        y_train = df.iloc[train_index][target_col]
        y_test = df.iloc[test_index][target_col] # Needed for eval_set

        # Select parameters based on target type
        is_multiclass = (model_name == 'HDA')
        params = lgbm_params_multi if is_multiclass else lgbm_params_binary
        model = lgb.LGBMClassifier(**params)

        # Define callbacks for LightGBM early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        # Fit the model
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=params['metric'], # Use the metric defined in params
            callbacks=callbacks
            # Add categorical_feature='auto' or list of names if using categoricals directly
        )

        # Store the trained model for this fold
        fold_models[model_name] = model

        # Predict probabilities on the test set
        print(f"    Predicting probabilities for {model_name}...")
        probs = model.predict_proba(X_test)

        # Store probabilities
        if is_multiclass:
            fold_predictions['P_H'] = probs[:, 0]
            fold_predictions['P_D'] = probs[:, 1]
            fold_predictions['P_A'] = probs[:, 2]
        else:
            # For binary, predict_proba returns [P(0), P(1)]
            # We store the probability of the positive class (usually 1)
            fold_predictions[f'P_{model_name}'] = probs[:, 1]

        print(f"    Best iteration for {model_name}: {model.best_iteration_}")
        gc.collect() # Clean memory after training each model

    # Combine predictions for this fold into a DataFrame
    fold_results_df = pd.DataFrame(fold_predictions)
    all_fold_results.append(fold_results_df)
    trained_models = fold_models # Keep models from the last fold

    print(f"Fold {fold + 1} completed.")
    del X_train, X_test, y_train, y_test, fold_models, fold_predictions, fold_results_df
    gc.collect()

# --- Combine Results & Evaluate ---
print("\nCombining results from all folds...")
results_df = pd.concat(all_fold_results).reset_index(drop=True)

# Define actual outcome columns needed for evaluation based on TARGET_COLUMNS
actual_cols_to_merge = ['MatchID'] + list(TARGET_COLUMNS.values())
# Add any other actual columns needed for metrics (like FTR for mapping HDA)
if 'Actual_FTR_Numeric' not in actual_cols_to_merge:
     actual_cols_to_merge.append('Actual_FTR_Numeric')

actuals_to_merge = df[actual_cols_to_merge].copy()

# Merge results with actuals
print("Merging predictions with actual outcomes...")
final_eval_df = pd.merge(results_df, actuals_to_merge, on='MatchID', how='left')

# Drop rows where essential actual outcomes might be missing
essential_actuals = list(TARGET_COLUMNS.values())
final_eval_df = final_eval_df.dropna(subset=essential_actuals)
if 'Actual_FTR_Numeric' in final_eval_df.columns:
    final_eval_df['Actual_FTR_Numeric'] = final_eval_df['Actual_FTR_Numeric'].astype(int)

if final_eval_df.empty:
     print("Error: No matching predictions and actuals found after merge. Check MatchIDs or data integrity.")
else:
    print(f"Evaluating on {len(final_eval_df)} matches.")

    # --- Calculate Average Predicted Probabilities ---
    avg_pred_cols = [col for col in final_eval_df.columns if col.startswith('P_')]
    avg_metrics = final_eval_df[avg_pred_cols].mean()
    print("\n--- Average Predicted Probabilities ---")
    print(avg_metrics.sort_index())

    # --- Calculate Actual Frequencies ---
    print("\n--- Actual Frequencies in Test Sets ---")
    actual_freq = {}
    for model_name, target_col in TARGET_COLUMNS.items():
         if model_name != 'HDA': # Handle binary targets
             actual_freq[f'Actual_{model_name}'] = final_eval_df[target_col].mean()
    # Handle HDA separately
    if 'Actual_FTR_Numeric' in final_eval_df.columns:
         actual_freq['Actual_H'] = (final_eval_df['Actual_FTR_Numeric'] == 0).mean()
         actual_freq['Actual_D'] = (final_eval_df['Actual_FTR_Numeric'] == 1).mean()
         actual_freq['Actual_A'] = (final_eval_df['Actual_FTR_Numeric'] == 2).mean()
    print(pd.Series(actual_freq).sort_index())


    # --- Evaluation Metrics ---
    print("\n--- Performance Evaluation ---")
    evaluation_scores = {}

    # Brier Scores for Binary Classifiers
    for model_name, target_col in TARGET_COLUMNS.items():
         if model_name != 'HDA':
             pred_col = f'P_{model_name}'
             if pred_col in final_eval_df.columns and target_col in final_eval_df.columns:
                 score = brier_score_loss(final_eval_df[target_col], final_eval_df[pred_col])
                 print(f"Brier Score ({model_name}): {score:.4f}")
                 evaluation_scores[f'Brier_{model_name}'] = score
             else:
                 print(f"Warning: Cannot calculate Brier score for {model_name}. Columns missing.")

    # Log Loss for HDA (Multiclass)
    if 'P_H' in final_eval_df.columns and 'Actual_FTR_Numeric' in final_eval_df.columns:
         hda_probs = final_eval_df[['P_H', 'P_D', 'P_A']].values
         hda_actuals = final_eval_df['Actual_FTR_Numeric'].values
         hda_probs = np.clip(hda_probs, 1e-15, 1 - 1e-15) # Clipping
         if hda_probs.sum(axis=1).min() > 0:
             hda_probs /= hda_probs.sum(axis=1)[:, np.newaxis] # Normalize row sums to 1
             logloss_hda = log_loss(hda_actuals, hda_probs, labels=[0, 1, 2])
             print(f"Log Loss (HDA): {logloss_hda:.4f}")
             evaluation_scores['LogLoss_HDA'] = logloss_hda
         else:
             print("Warning: Could not calculate Log Loss HDA due to probability sum being zero.")
    else:
        print("Warning: Cannot calculate Log Loss HDA. Columns missing.")

    print("\n--- Direct Probability Prediction Complete ---")
    # Save results if needed
    # final_eval_df.to_csv('gradient_boosting_results.csv', index=False)
    # print("Results saved to gradient_boosting_results.csv")