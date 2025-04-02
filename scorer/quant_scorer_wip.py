"""
Advanced Soccer Match Prediction Script
---------------------------------------

Leverages LightGBM and Poisson models to predict multiple match outcomes
based on extensive input parameters, providing confidence and predictability scores.

Disclaimer:It requires historical data
for training, extensive feature engineering tailored to the specific dataset,
hyperparameter tuning, and rigorous backtesting for real-world application.
Model file paths are placeholders.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib  # For loading/saving models
import warnings
import re  # For parsing form strings
import os  # For file handling
import json  # For reading JSON files
import argparse  # For command line arguments
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning)  # Suppress minor warnings

# --- Configuration ---
MODEL_DIR = './soccer_models/'  # Directory to save/load trained models
OUTPUT_DIR = './predictions/'  # Directory for output predictions
DAILY_GAMES_DIR = './daily_games/'  # Directory with daily game files

MODEL_FILES = {
    '1x2': f'{MODEL_DIR}model_1x2.joblib',
    'xg_home': f'{MODEL_DIR}model_xg_home.joblib',
    'xg_away': f'{MODEL_DIR}model_xg_away.joblib'
    # We derive BTTS and O/U from xG models, so no separate models needed here
}
CLASS_ORDER_1X2 = ['Away Win', 'Draw', 'Home Win']  # CRUCIAL: Must match training
EPSILON = 1e-9  # Small constant to avoid log(0) errors

# --- Feature Engineering Function (NEEDS CUSTOMIZATION) ---

def parse_form(form_str, num_matches=5):
    """Parses form string (e.g., 'WDLWL') into points."""
    points_map = {'W': 3, 'D': 1, 'L': 0}
    form_str = re.sub(r'[^WDL]', '', str(form_str))  # Clean form string
    form_points = [points_map.get(res, 0) for res in form_str[:num_matches]]
    # Pad with average points (1.0) if form string is shorter than num_matches
    while len(form_points) < num_matches:
        form_points.append(1.0)
    return form_points

def engineer_features(raw_data: dict) -> pd.DataFrame:
    """
    Transforms raw match data (dict based on parameter_summary) into a
    feature vector suitable for the models.

    THIS IS THE MOST CRITICAL PART AND REQUIRES SIGNIFICANT DEVELOPMENT.
    Placeholder examples are provided.

    Args:
        raw_data (dict): Dictionary containing data for one match, structured
                         similarly to the parameter_summary outline.

    Returns:
        pd.DataFrame: A single row DataFrame with engineered features.
    """
    features = {}

    # Example Feature Extraction & Engineering:
    # Use .get() with default values to handle potentially missing keys gracefully

    # 1. Standings-based features
    features['home_rank'] = raw_data.get('home_standing', {}).get('rank', 10)  # Use mid-rank default
    features['away_rank'] = raw_data.get('away_standing', {}).get('rank', 10)
    features['rank_diff'] = features['home_rank'] - features['away_rank']
    features['home_points_per_game'] = raw_data.get('home_standing', {}).get('points', 0) / max(1, raw_data.get('home_standing', {}).get('all', {}).get('played', 1))
    features['away_points_per_game'] = raw_data.get('away_standing', {}).get('points', 0) / max(1, raw_data.get('away_standing', {}).get('all', {}).get('played', 1))
    features['points_per_game_diff'] = features['home_points_per_game'] - features['away_points_per_game']
    features['home_goal_diff'] = raw_data.get('home_standing', {}).get('goalsDiff', 0)
    features['away_goal_diff'] = raw_data.get('away_standing', {}).get('goalsDiff', 0)
    features['goal_diff_diff'] = features['home_goal_diff'] - features['away_goal_diff']

    # 2. Form features
    home_form_str = raw_data.get('home_standing', {}).get('form', 'DDDDD')
    away_form_str = raw_data.get('away_standing', {}).get('form', 'DDDDD')
    home_form_points_list = parse_form(home_form_str, 5)
    away_form_points_list = parse_form(away_form_str, 5)
    features['home_form_points_last5'] = sum(home_form_points_list)
    features['away_form_points_last5'] = sum(away_form_points_list)
    features['form_points_diff_last5'] = features['home_form_points_last5'] - features['away_form_points_last5']
    # Store list for variance calculation later
    features['home_form_points_list'] = home_form_points_list
    features['away_form_points_list'] = away_form_points_list

    # 3. Goal averages (MongoDB stats) - Use specific home/away averages
    features['home_avg_goals_for_home'] = raw_data.get('home_mongodb_stats', {}).get('goals', {}).get('for', {}).get('average', {}).get('home', 1.0)
    features['home_avg_goals_against_home'] = raw_data.get('home_mongodb_stats', {}).get('goals', {}).get('against', {}).get('average', {}).get('home', 1.0)
    features['away_avg_goals_for_away'] = raw_data.get('away_mongodb_stats', {}).get('goals', {}).get('for', {}).get('average', {}).get('away', 1.0)
    features['away_avg_goals_against_away'] = raw_data.get('away_mongodb_stats', {}).get('goals', {}).get('against', {}).get('average', {}).get('away', 1.0)

    # Expected Goals (xG) related features (Conceptual - relies on averages here)
    features['home_xg_proxy'] = features['home_avg_goals_for_home']
    features['away_xg_proxy'] = features['away_avg_goals_for_away']
    features['home_xga_proxy'] = features['home_avg_goals_against_home']
    features['away_xga_proxy'] = features['away_avg_goals_against_away']
    features['xg_diff_proxy'] = features['home_xg_proxy'] - features['away_xg_proxy']
    features['xga_diff_proxy'] = features['home_xga_proxy'] - features['away_xga_proxy']
    # Interaction terms (Quant approach often involves basis functions or interactions)
    features['home_attack_vs_away_defense'] = features['home_xg_proxy'] * features['away_xga_proxy']
    features['away_attack_vs_home_defense'] = features['away_xg_proxy'] * features['home_xga_proxy']

    # 4. Streaks and other performance metrics
    features['home_win_streak'] = raw_data.get('home_mongodb_stats', {}).get('biggest', {}).get('streak', {}).get('wins', 0) if 'W' in home_form_str else 0  # Simplistic streak logic
    features['away_lose_streak'] = raw_data.get('away_mongodb_stats', {}).get('biggest', {}).get('streak', {}).get('loses', 0) if 'L' in away_form_str else 0
    features['home_failed_to_score_home_pct'] = raw_data.get('home_mongodb_stats', {}).get('performance', {}).get('failed_to_score', {}).get('home', 0) / max(1, raw_data.get('home_mongodb_stats', {}).get('fixtures', {}).get('played', {}).get('home', 1))
    features['away_failed_to_score_away_pct'] = raw_data.get('away_mongodb_stats', {}).get('performance', {}).get('failed_to_score', {}).get('away', 0) / max(1, raw_data.get('away_mongodb_stats', {}).get('fixtures', {}).get('played', {}).get('away', 1))
    features['home_clean_sheet_home_pct'] = raw_data.get('home_mongodb_stats', {}).get('performance', {}).get('clean_sheet', {}).get('home', 0) / max(1, raw_data.get('home_mongodb_stats', {}).get('fixtures', {}).get('played', {}).get('home', 1))
    features['away_clean_sheet_away_pct'] = raw_data.get('away_mongodb_stats', {}).get('performance', {}).get('clean_sheet', {}).get('away', 0) / max(1, raw_data.get('away_mongodb_stats', {}).get('fixtures', {}).get('played', {}).get('away', 1))

    # 5. Statarea Analysis (chance to score/concede) 
    # First, check for home_statarea_analysis explicitly from the daily_games structure
    home_statarea_data = raw_data.get('home_statarea_analysis', {})
    away_statarea_data = raw_data.get('away_statarea_analysis', {})
    
    # If not found, try to extract from the daily_games structure where statarea data is nested differently
    if not home_statarea_data and 'home_statarea_data' in raw_data:
        home_statarea_data = raw_data.get('home_statarea_data', {}).get('stats', {}).get('host_10', {})
    if not away_statarea_data and 'away_statarea_data' in raw_data:
        away_statarea_data = raw_data.get('away_statarea_data', {}).get('stats', {}).get('guest_10', {})
    
    # Extract chance to score/concede - try different formats based on the data structure
    features['home_chance_score_10g'] = 0.5  # Default 50%
    features['home_chance_concede_10g'] = 0.5
    features['away_chance_score_10g'] = 0.5
    features['away_chance_concede_10g'] = 0.5
    
    # Try to parse percentage values from the StatArea data
    if "Chance to score goal next match" in home_statarea_data:
        home_chance_score_str = home_statarea_data.get("Chance to score goal next match", "50%")
        home_chance_concede_str = home_statarea_data.get("Chance to conceded goal next match", "50%")
        features['home_chance_score_10g'] = float(home_chance_score_str.strip("%")) / 100 if "%" in home_chance_score_str else 0.5
        features['home_chance_concede_10g'] = float(home_chance_concede_str.strip("%")) / 100 if "%" in home_chance_concede_str else 0.5
    
    if "Chance to score goal next match" in away_statarea_data:
        away_chance_score_str = away_statarea_data.get("Chance to score goal next match", "50%")
        away_chance_concede_str = away_statarea_data.get("Chance to conceded goal next match", "50%")
        features['away_chance_score_10g'] = float(away_chance_score_str.strip("%")) / 100 if "%" in away_chance_score_str else 0.5
        features['away_chance_concede_10g'] = float(away_chance_concede_str.strip("%")) / 100 if "%" in away_chance_concede_str else 0.5
    
    features['chance_score_diff'] = features['home_chance_score_10g'] - features['away_chance_score_10g']
    features['chance_concede_diff'] = features['home_chance_concede_10g'] - features['away_chance_concede_10g']

    # Create DataFrame - IMPORTANT: Columns must match training data columns
    # Drop the list features used only for variance calculation
    feature_df = pd.DataFrame([features])
    feature_df = feature_df.drop(columns=['home_form_points_list', 'away_form_points_list'], errors='ignore')

    # Simple Imputation (replace NaN with median)
    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if feature_df[col].isnull().any():
            median_val = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median_val)

    return feature_df

# --- Prediction Functions ---

def predict_1X2(features: pd.DataFrame, model_path: str) -> tuple:
    """Predicts 1X2 probabilities and confidence score."""
    try:
        model = joblib.load(model_path)
        probabilities = model.predict_proba(features)[0]

        if len(probabilities) != len(CLASS_ORDER_1X2):
             raise ValueError(f"Model output size {len(probabilities)} != expected size {len(CLASS_ORDER_1X2)}")

        results = dict(zip(CLASS_ORDER_1X2, probabilities))

        # Confidence Score (Normalized Inverse Entropy)
        entropy = -np.sum([(p + EPSILON) * np.log2(p + EPSILON) for p in probabilities])
        max_entropy = np.log2(len(CLASS_ORDER_1X2))
        confidence = max(0, (max_entropy - entropy) / max_entropy) # Ensure non-negative

        return results, confidence
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None, 0.0
    except Exception as e:
        print(f"Error during 1X2 prediction: {e}")
        return None, 0.0


def predict_expected_goals(features: pd.DataFrame, model_path_home: str, model_path_away: str) -> tuple:
    """Predicts expected goals (lambda) for home and away teams."""
    try:
        model_home = joblib.load(model_path_home)
        model_away = joblib.load(model_path_away)

        lambda_home = model_home.predict(features)[0]
        lambda_away = model_away.predict(features)[0]

        # Ensure lambdas are non-negative (physical constraint)
        lambda_home = max(lambda_home, EPSILON)
        lambda_away = max(lambda_away, EPSILON)

        return lambda_home, lambda_away
    except FileNotFoundError as fnf:
        print(f"Error: Model file not found. {fnf}")
        return EPSILON, EPSILON # Return minimal non-zero value
    except Exception as e:
        print(f"Error during xG prediction: {e}")
        return EPSILON, EPSILON

def calculate_goal_probabilities(lambda_home: float, lambda_away: float, over_under_line: float = 2.5, max_goals_sim: int = 10) -> dict:
    """
    Calculates O/U and BTTS probabilities from Poisson lambdas.
    Also calculates confidence scores for these derived probabilities.
    """
    results = {}
    try:
        # Calculate P(Score = h, a) matrix
        prob_matrix = np.zeros((max_goals_sim + 1, max_goals_sim + 1))
        for hg in range(max_goals_sim + 1):
            for ag in range(max_goals_sim + 1):
                prob_matrix[hg, ag] = poisson.pmf(hg, lambda_home) * poisson.pmf(ag, lambda_away)

        # Normalize probabilities (ensures they sum ~1 after truncating max_goals_sim)
        prob_matrix /= np.sum(prob_matrix)

        # --- Over/Under Calculation ---
        prob_under = 0
        for hg in range(max_goals_sim + 1):
            for ag in range(max_goals_sim + 1):
                if hg + ag <= over_under_line:
                    prob_under += prob_matrix[hg, ag]

        prob_over = 1.0 - prob_under
        results[f'P(Over {over_under_line})'] = prob_over
        results[f'P(Under {over_under_line})'] = prob_under
        # Confidence for O/U (distance from 0.5)
        results[f'Confidence O/U {over_under_line}'] = abs(prob_over - 0.5) * 2

        # --- BTTS Calculation ---
        prob_home_scores = 1.0 - np.sum(prob_matrix[0, :]) # P(H > 0)
        prob_away_scores = 1.0 - np.sum(prob_matrix[:, 0]) # P(A > 0)
        # More accurate BTTS calc from matrix: Sum all cells where hg>0 AND ag>0
        prob_btts_yes = np.sum(prob_matrix[1:, 1:])

        # Ensure consistency (small numerical errors possible)
        prob_btts_no = 1.0 - prob_btts_yes
        results['P(BTTS Yes)'] = prob_btts_yes
        results['P(BTTS No)'] = prob_btts_no
        # Confidence for BTTS (distance from 0.5)
        results['Confidence BTTS'] = abs(prob_btts_yes - 0.5) * 2

    except Exception as e:
        print(f"Error calculating goal probabilities: {e}")
        # Return defaults in case of error
        results[f'P(Over {over_under_line})'] = 0.5
        results[f'P(Under {over_under_line})'] = 0.5
        results[f'Confidence O/U {over_under_line}'] = 0.0
        results['P(BTTS Yes)'] = 0.5
        results['P(BTTS No)'] = 0.5
        results['Confidence BTTS'] = 0.0

    return results


# --- Overall Predictability Score Function ---

def calculate_overall_predictability(features_used: pd.DataFrame, confidences: dict) -> float:
    """
    Calculates an overall predictability score (0-100).
    Combines average model confidence with data stability metrics.
    Needs calibration based on backtesting.

    Args:
        features_used (pd.DataFrame): The engineered features used for prediction.
        confidences (dict): Dictionary containing confidence scores for 1X2, O/U, BTTS.

    Returns:
        float: Predictability score (0-100).
    """
    # 1. Average Model Confidence
    avg_model_confidence = np.mean(list(confidences.values()))

    # 2. Data Stability (Example: Inverse Form Variance)
    try:
        # Retrieve original form points lists (requires them to be passed/stored)
        # This is why feature engineering function design is crucial.
        # Simulating retrieval from features_used if they were kept:
        # For demo, we'll just use the calculated sum variance as a proxy
        home_form_points_list = parse_form(features_used.get('home_form_points_last5', 1*5)) # Re-parsing needed if list not stored
        away_form_points_list = parse_form(features_used.get('away_form_points_last5', 1*5))
        home_form_variance = np.var(home_form_points_list)
        away_form_variance = np.var(away_form_points_list)

        # Normalize variance (using heuristic max)
        max_reasonable_variance = 4.0 # Max variance if points are 0/3
        # Low variance is good stability = higher score
        combined_variance = (home_form_variance + away_form_variance) / 2.0
        stability_metric = max(0.0, 1.0 - (combined_variance / max_reasonable_variance))
    except Exception as e:
        print(f"Warning: Could not calculate stability metric: {e}")
        stability_metric = 0.5 # Default stability if calculation fails

    # 3. Combine (Weighted Average - Weights need tuning based on backtesting)
    weight_confidence = 0.6
    weight_stability = 0.4
    overall_predictability = (weight_confidence * avg_model_confidence +
                            weight_stability * stability_metric)

    return max(0.0, min(100.0, overall_predictability * 100)) # Scale to 0-100

# --- Main Execution Function ---

def run_match_prediction(raw_match_data: dict) -> dict:
    """
    Runs the full prediction pipeline for a single match.

    Args:
        raw_match_data (dict): Raw data for the match.

    Returns:
        dict: Comprehensive prediction results.
    """
    print("1. Engineering features...")
    features = engineer_features(raw_match_data)
    # print("Engineered Features:\n", features.iloc[0].to_dict()) # DEBUG: View features

    if features is None or features.empty:
        print("Error: Feature engineering failed.")
        return {"error": "Feature engineering failed."}

    # Ensure features DataFrame has expected columns if models are sensitive to order/names
    # (This depends on how models were trained/saved, e.g., using sklearn Pipeline)

    print("2. Predicting 1X2 outcome...")
    results_1x2, conf_1x2 = predict_1X2(features, MODEL_FILES['1x2'])

    print("3. Predicting Expected Goals...")
    lambda_h, lambda_a = predict_expected_goals(features, MODEL_FILES['xg_home'], MODEL_FILES['xg_away'])

    print("4. Calculating Goal Probabilities (O/U, BTTS)...")
    goal_probs = calculate_goal_probabilities(lambda_h, lambda_a)

    # Consolidate confidences
    confidences = {
        '1X2': conf_1x2,
        'OU25': goal_probs.get('Confidence O/U 2.5', 0.0),
        'BTTS': goal_probs.get('Confidence BTTS', 0.0)
    }

    print("5. Calculating Overall Predictability...")
    # Pass original feature sums used for variance calculation, or re-calculate if needed
    predictability_score = calculate_overall_predictability(raw_match_data, confidences) # Pass raw for stability calc


    print("--- Prediction Complete ---")

    # Format match identifiers from the raw data
    home_team = raw_match_data.get('home_standing', {}).get('team', {}).get('name', 'Unknown Home')
    away_team = raw_match_data.get('away_standing', {}).get('team', {}).get('name', 'Unknown Away')
    fixture_id = raw_match_data.get('fixture_info', {}).get('id', raw_match_data.get('fixture', {}).get('id', 'N/A'))
    fixture_date = raw_match_data.get('fixture_info', {}).get('date', raw_match_data.get('fixture', {}).get('date', 'N/A'))

    # --- Format Output ---
    final_output = {
        "Match Info": {
            "Home Team": home_team,
            "Away Team": away_team,
            "Fixture ID": fixture_id,
            "Date": fixture_date
        },
        "Predictions": {
            "1X2": results_1x2 if results_1x2 else "Error",
            "Expected Goals": {
                "Home": f"{lambda_h:.3f}",
                "Away": f"{lambda_a:.3f}",
                "Total": f"{lambda_h + lambda_a:.3f}"
            },
            "Over/Under 2.5": {
                "P(Over)": f"{goal_probs.get('P(Over 2.5)', 0.0):.3f}",
                "P(Under)": f"{goal_probs.get('P(Under 2.5)', 0.0):.3f}"
            },
            "BTTS": {
                "P(Yes)": f"{goal_probs.get('P(BTTS Yes)', 0.0):.3f}",
                "P(No)": f"{goal_probs.get('P(BTTS No)', 0.0):.3f}"
            }
        },
        "Confidence Scores (0-1)": {
            "1X2 Confidence": f"{conf_1x2:.3f}",
            "O/U 2.5 Confidence": f"{goal_probs.get('Confidence O/U 2.5', 0.0):.3f}",
            "BTTS Confidence": f"{goal_probs.get('Confidence BTTS', 0.0):.3f}",
        },
        "Overall Predictability Score (0-100)": f"{predictability_score:.1f}"
    }

    return final_output

# --- Functions to Process Daily Games Files ---

def process_daily_games_file(file_path: str) -> dict:
    """
    Process a single daily games file and extract needed data for prediction.
    
    Args:
        file_path: Path to the JSON file with match data
        
    Returns:
        Dictionary with processed match data for prediction
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        
        # Prepare the raw match data structure that's expected by run_match_prediction
        raw_match_data = {
            'fixture_info': game_data.get('fixture_info', {}),
            'fixture': game_data.get('fixture_info', {}),  # Duplicating for compatibility with both formats
            'league': game_data.get('league', {}),
        }
        
        # Extract standings for home and away teams
        league_standings = game_data.get('league', {}).get('standings', [[]])[0]
        home_team_id = game_data.get('teams', {}).get('home', {}).get('id') 
        away_team_id = game_data.get('teams', {}).get('away', {}).get('id')
        
        # Find the standings entries for the home and away teams
        home_standing = next((item for item in league_standings if item.get('team', {}).get('id') == home_team_id), None)
        away_standing = next((item for item in league_standings if item.get('team', {}).get('id') == away_team_id), None)
        
        # If team IDs not found directly, try to match by name
        if not home_standing or not away_standing:
            home_team_name = game_data.get('teams', {}).get('home', {}).get('name')
            away_team_name = game_data.get('teams', {}).get('away', {}).get('name')
            home_standing = next((item for item in league_standings if item.get('team', {}).get('name') == home_team_name), None)
            away_standing = next((item for item in league_standings if item.get('team', {}).get('name') == away_team_name), None)
        
        # Add standings to raw match data
        raw_match_data['home_standing'] = home_standing or {}
        raw_match_data['away_standing'] = away_standing or {}
        
        # Add MongoDB stats from teams section
        home_mongodb_stats = game_data.get('home_mongodb_stats', {})
        away_mongodb_stats = game_data.get('away_mongodb_stats', {})
        
        if not home_mongodb_stats and 'teams' in game_data:
            home_mongodb_stats = game_data.get('teams', {}).get('home', {}).get('statistics', {})
        if not away_mongodb_stats and 'teams' in game_data:
            away_mongodb_stats = game_data.get('teams', {}).get('away', {}).get('statistics', {})
            
        raw_match_data['home_mongodb_stats'] = home_mongodb_stats
        raw_match_data['away_mongodb_stats'] = away_mongodb_stats
        
        # Add StatArea analysis data
        raw_match_data['home_statarea_data'] = game_data.get('home_statarea_data', {})
        raw_match_data['away_statarea_data'] = game_data.get('away_statarea_data', {})
        
        raw_match_data['home_statarea_analysis'] = game_data.get('home_statarea_analysis', {})
        raw_match_data['away_statarea_analysis'] = game_data.get('away_statarea_analysis', {})
        
        return raw_match_data
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {}

def process_all_daily_games(directory: str = DAILY_GAMES_DIR, date_filter: str = None) -> dict:
    """
    Process all daily games files in the directory and generate predictions.
    
    Args:
        directory: Path to directory with daily games files
        date_filter: Optional date string to filter files by date (YYYY-MM-DD format)
        
    Returns:
        Dictionary with all predictions keyed by match identifier
    """
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return {}
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    # Filter by date if specified
    if date_filter:
        json_files = [f for f in json_files if date_filter in f]
    
    if not json_files:
        print(f"No JSON files found in {directory}" + 
              (f" for date {date_filter}" if date_filter else ""))
        return {}
    
    # Process each file
    all_predictions = {}
    for json_file in json_files:
        print(f"\nProcessing {json_file}...")
        file_path = os.path.join(directory, json_file)
        
        # Extract match data from file
        raw_match_data = process_daily_games_file(file_path)
        if not raw_match_data:
            print(f"Skipping {json_file} due to processing error")
            continue
        
        # Run prediction for this match
        prediction = run_match_prediction(raw_match_data)
        
        # Generate a match identifier
        home_team = prediction.get('Match Info', {}).get('Home Team', 'Unknown')
        away_team = prediction.get('Match Info', {}).get('Away Team', 'Unknown')
        fixture_id = prediction.get('Match Info', {}).get('Fixture ID', 'unknown_id')
        match_id = f"{fixture_id}_{home_team}_vs_{away_team}"
        
        # Store prediction
        all_predictions[match_id] = prediction
        
        # Save individual prediction file
        output_filename = os.path.join(
            OUTPUT_DIR, 
            f"prediction_{os.path.splitext(json_file)[0]}.json"
        )
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(prediction, f, indent=2)
        print(f"Saved prediction to {output_filename}")
    
    # Save combined predictions file
    if all_predictions:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = os.path.join(
            OUTPUT_DIR,
            f"all_predictions_{timestamp}.json"
        )
        with open(combined_filename, 'w', encoding='utf-8') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"\nSaved all predictions to {combined_filename}")
    
    return all_predictions

# --- Example Usage ---

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Soccer Match Prediction Script')
    parser.add_argument('--date', type=str, help='Date filter in YYYY-MM-DD format')
    parser.add_argument('--file', type=str, help='Process a specific file instead of all files')
    parser.add_argument('--dir', type=str, default=DAILY_GAMES_DIR, 
                        help=f'Directory with daily games files (default: {DAILY_GAMES_DIR})')
    args = parser.parse_args()

    # Ensure model directory exists and create dummy models if needed
    import os
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created model directory: {MODEL_DIR}")
        print("NOTE: This script expects PRE-TRAINED models in this directory.")
        print("Creating dummy models for demonstration purposes...")