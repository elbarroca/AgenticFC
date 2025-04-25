# predict_fixture.py
import logging
import json
import pandas as pd
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from joblib import load as joblib_load
from typing import Optional, Tuple, Any, Dict, List
from sklearn.preprocessing import LabelEncoder # Keep if label_encoder.joblib is sklearn's
import re # For parsing form string
import sys
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from scipy.stats import poisson
import glob # Import glob for file matching
from plotting_utils import create_combined_fixture_plot

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
PODOS_MODEL_REPO = "Bettensor/podos_soccer_model"
PODOS_LABEL_ENCODER_FILENAME = "label_encoder.joblib"  # Keep original name, handled in code
PODOS_MODEL_FILENAME = "model.safetensors"
MONTE_CARLO_SIMULATIONS = 80000 # Increased number of simulations
TOP_N_SCENARIOS = 10 # Number of top scenarios to display
UNIFIED_DATA_DIR = "data/unified_data" # Directory containing the JSON files to process
OUTPUT_DIR = "data/output" # Directory for output results
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots") # Directory for saving plots
MC_MAX_SCORE_PLOT = 5 # Max goals for score matrix plot

# Podos expected features (order matters)
PODOS_EXPECTED_FEATURES = [
    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HO', 'AO', 'HY', 'AY', 'HR', 'AR',
    'oddsH', 'oddsD', 'oddsA',
    'home_encoded', 'away_encoded',
    'WinStreakHome', 'LossStreakHome', 'WinStreakAway', 'LossStreakAway',
    'HomeTeamForm', 'AwayTeamForm'
]

# --- Placeholder PodosTransformer Class (MUST BE REPLACED) ---
class PodosTransformer:
    """
    Placeholder for the PodosTransformer class.
    You MUST replace this with the actual class definition from the model source.
    """
    def __init__(self, *args, **kwargs):
        logger.warning("Using **PLACEHOLDER** PodosTransformer class. Replace with actual definition.")
        self._is_fitted = False # Track if weights are loaded

    @classmethod
    def from_pretrained(cls, repo_id: str, filename: str, **kwargs):
        logger.info(f"**PLACEHOLDER**: Simulating load of PodosTransformer from {repo_id}/{filename}")
        try:
            model_path = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
            logger.info(f"Simulated download path: {model_path}")
            instance = cls()
            instance._is_fitted = True
            logger.info("**PLACEHOLDER**: PodosTransformer model weights loaded (simulated).")
            return instance
        except Exception as e:
            logger.error(f"**PLACEHOLDER**: Failed to simulate download/load from {repo_id}/{filename}: {e}", exc_info=True)
            raise

    def predict_proba(self, X_input: torch.Tensor) -> np.ndarray:
        if not getattr(self, '_is_fitted', False):
            raise RuntimeError("PodosTransformer model weights not loaded/simulated.")
        logger.warning("**PLACEHOLDER**: Generating **RANDOM** probabilities from PodosTransformer.")
        num_samples = X_input.shape[0]
        raw_probs = np.random.rand(num_samples, 3)
        probabilities = raw_probs / raw_probs.sum(axis=1, keepdims=True)
        return probabilities

# --- Helper Functions ---
def safe_get(data: Dict, keys: List[str], default: Any = None) -> Any:
    """Safely traverse nested dictionary keys."""
    if not isinstance(data, dict) or not isinstance(keys, list): return default
    current = data
    for key in keys:
        try:
            if isinstance(current, dict): current = current.get(key)
            elif isinstance(current, (list, tuple)) and isinstance(key, int):
                if 0 <= key < len(current): current = current[key]
                else: return default
            else: return default
            if current is None: return default
        except (TypeError, KeyError, IndexError): return default
    return current if current is not None else default

def parse_form_string(form_str: Optional[str], num_matches: int = 5) -> Tuple[int, int, float]:
    """Parses form string for streaks and form PPG."""
    if not form_str or len(form_str) == 0: return 0, 0, 0.0
    relevant_form = form_str[-num_matches:]
    win_streak = len(form_str) - len(form_str.rstrip('W')) if form_str.endswith('W') else 0
    loss_streak = len(form_str) - len(form_str.rstrip('L')) if form_str.endswith('L') else 0
    points = 0
    matches_counted = 0
    for char in relevant_form:
        if char == 'W': points += 3; matches_counted += 1
        elif char == 'D': points += 1; matches_counted += 1
        elif char == 'L': matches_counted += 1
    form_ppg = (points / matches_counted) if matches_counted > 0 else 0.0
    return win_streak, loss_streak, form_ppg

# --- Podos Loading ---
def load_podos_components() -> Optional[Tuple[PodosTransformer, Any]]:
    """Loads Podos model (placeholder) and label encoder (which is optional)."""
    logger.info(f"Attempting to load Podos components from repo: {PODOS_MODEL_REPO}")
    
    # First try to load the model, which is required
    try:
        model = PodosTransformer.from_pretrained(
            repo_id=PODOS_MODEL_REPO, filename=PODOS_MODEL_FILENAME
        )
        logger.info(f"Successfully loaded Podos model")
    except Exception as e:
        logger.error(f"Error loading Podos model: {e}", exc_info=True)
        return None, None
    
    # Try to load the label encoder but make it optional
    label_encoder = None
    try:
        # Try the original filename first
        encoder_path = hf_hub_download(
            repo_id=PODOS_MODEL_REPO, filename=PODOS_LABEL_ENCODER_FILENAME
        )
        label_encoder = joblib_load(encoder_path)
        logger.info(f"Loaded label encoder type: {type(label_encoder).__name__}")
    except Exception as e:
        # Try alternative filenames
        alternative_filenames = ["encoder.pkl", "team_encoder.joblib", "teams.pkl"]
        
        for alt_filename in alternative_filenames:
            try:
                logger.info(f"Trying alternative encoder filename: {alt_filename}")
                encoder_path = hf_hub_download(
                    repo_id=PODOS_MODEL_REPO, filename=alt_filename
                )
                label_encoder = joblib_load(encoder_path)
                logger.info(f"Loaded label encoder from alternative file {alt_filename}")
                break
            except Exception:
                continue
                
        if label_encoder is None:
            # If all attempts failed, create a dummy encoder that just returns 0
            logger.warning(f"Could not load label encoder from repository. Creating dummy encoder.")
            dummy_encoder = LabelEncoder()
            dummy_encoder.classes_ = np.array(["UNKNOWN"])  # Just a placeholder
            label_encoder = dummy_encoder
            logger.warning("Using dummy label encoder. All team IDs will be encoded as 0.")
            
    return model, label_encoder

# --- Podos Data Preparation ---
def prepare_podos_input_from_json(fixture_data: Dict, label_encoder: Any) -> Optional[torch.Tensor]:
    """Prepares the 23-feature tensor for Podos from the JSON data."""
    logger.info("Preparing data for Podos model input from JSON (best effort mapping)...")
    features = {}
    raw_home_snap = safe_get(fixture_data, ['raw_data', 'home', 'match_processor_snapshot'], {})
    raw_away_snap = safe_get(fixture_data, ['raw_data', 'away', 'match_processor_snapshot'], {})
    pred_snap_home = safe_get(raw_home_snap, ['predictions_snapshot'], {})
    pred_snap_away = safe_get(raw_away_snap, ['predictions_snapshot'], {})
    eng_feat_home = safe_get(fixture_data, ['engineered_features', 'home'], {})
    eng_feat_away = safe_get(fixture_data, ['engineered_features', 'away'], {})

    home_played = safe_get(raw_home_snap, ['fixtures', 'played', 'total'], 0)
    away_played = safe_get(raw_away_snap, ['fixtures', 'played', 'total'], 0)

    # Proxies/Defaults (same logic as predict_with_podos_from_json.py)
    home_avg_goals_for = safe_get(raw_home_snap, ['goals', 'for', 'average', 'total'], 0.0)
    away_avg_goals_for = safe_get(raw_away_snap, ['goals', 'for', 'average', 'total'], 0.0)
    features['HS'] = home_avg_goals_for * 8.0
    features['AS'] = away_avg_goals_for * 8.0
    features['HST'] = features['HS'] * 0.4
    features['AST'] = features['AS'] * 0.4
    features['HC'] = 5.0 # Default - Absent Data
    features['AC'] = 5.0 # Default - Absent Data
    features['HO'] = 2.0 # Default - Absent Data
    features['AO'] = 2.0 # Default - Absent Data
    home_yellows_total = sum(safe_get(raw_home_snap, ['cards', 'yellow', k, 'total'], 0) for k in raw_home_snap.get('cards', {}).get('yellow', {}).keys())
    away_yellows_total = sum(safe_get(raw_away_snap, ['cards', 'yellow', k, 'total'], 0) for k in raw_away_snap.get('cards', {}).get('yellow', {}).keys())
    features['HY'] = (home_yellows_total / home_played) if home_played > 0 else 0.0
    features['AY'] = (away_yellows_total / away_played) if away_played > 0 else 0.0
    home_reds_total = sum(safe_get(raw_home_snap, ['cards', 'red', k, 'total'], 0) for k in raw_home_snap.get('cards', {}).get('red', {}).keys())
    away_reds_total = sum(safe_get(raw_away_snap, ['cards', 'red', k, 'total'], 0) for k in raw_away_snap.get('cards', {}).get('red', {}).keys())
    features['HR'] = (home_reds_total / home_played) if home_played > 0 else 0.0
    features['AR'] = (away_reds_total / away_played) if away_played > 0 else 0.0
    features['oddsH'] = 2.50 # *** DEFAULT VALUE - Absent Data ***
    features['oddsD'] = 3.50 # *** DEFAULT VALUE - Absent Data ***
    features['oddsA'] = 2.80 # *** DEFAULT VALUE - Absent Data ***

    home_id = safe_get(eng_feat_home, ['team_id'])
    away_id = safe_get(eng_feat_away, ['team_id'])
    unknown_team_value = 0
    try:
        known_classes = set(label_encoder.classes_) if hasattr(label_encoder, 'classes_') else None
        if known_classes:
            features['home_encoded'] = label_encoder.transform([home_id])[0] if home_id in known_classes else unknown_team_value
            features['away_encoded'] = label_encoder.transform([away_id])[0] if away_id in known_classes else unknown_team_value
        else:
            features['home_encoded'] = unknown_team_value; features['away_encoded'] = unknown_team_value
        if features['home_encoded'] == unknown_team_value: logger.warning(f"Home team ID {home_id} not found in encoder.")
        if features['away_encoded'] == unknown_team_value: logger.warning(f"Away team ID {away_id} not found in encoder.")
    except Exception as e:
         logger.error(f"Error during team ID transformation: {e}.", exc_info=True)
         features['home_encoded'] = unknown_team_value; features['away_encoded'] = unknown_team_value

    home_form_str = safe_get(raw_home_snap, ['form_string'])
    away_form_str = safe_get(raw_away_snap, ['form_string'])
    wsH, lsH, formH_ppg = parse_form_string(home_form_str)
    wsA, lsA, formA_ppg = parse_form_string(away_form_str)
    features['WinStreakHome'] = wsH; features['LossStreakHome'] = lsH
    features['WinStreakAway'] = wsA; features['LossStreakAway'] = lsA
    home_form_snapshot = safe_get(pred_snap_home, ['last_5', 'form_string'])
    away_form_snapshot = safe_get(pred_snap_away, ['last_5', 'form_string'])
    try: features['HomeTeamForm'] = float(home_form_snapshot) if home_form_snapshot is not None else formH_ppg
    except (ValueError, TypeError): features['HomeTeamForm'] = formH_ppg
    try: features['AwayTeamForm'] = float(away_form_snapshot) if away_form_snapshot is not None else formA_ppg
    except (ValueError, TypeError): features['AwayTeamForm'] = formA_ppg

    # Validation and Formatting
    try:
        df_prepared = pd.DataFrame([features])
        df_final = df_prepared[PODOS_EXPECTED_FEATURES] # Ensure order and selection
    except KeyError as e:
         logger.error(f"Critical Error: Feature missing during DataFrame creation: {e}", exc_info=True)
         return None

    df_numeric = df_final.apply(pd.to_numeric, errors='coerce')
    if df_numeric.isnull().sum().sum() > 0:
        logger.warning("NaNs detected after feature preparation. Imputing with 0.")
        df_imputed = df_numeric.fillna(0)
    else:
        df_imputed = df_numeric

    # Convert to Tensor
    try:
        podos_input_tensor = torch.tensor(df_imputed.values, dtype=torch.float32)
        if podos_input_tensor.shape == (1, len(PODOS_EXPECTED_FEATURES)):
            logger.info("Podos input tensor prepared successfully.")
            return podos_input_tensor
        else:
            logger.error(f"Tensor shape mismatch: {podos_input_tensor.shape}")
            return None
    except Exception as e:
        logger.error(f"Error converting final DataFrame to Tensor: {e}", exc_info=True)
        return None

# --- Podos Prediction ---
def predict_podos_probabilities(model: PodosTransformer, X_tensor: torch.Tensor) -> Optional[pd.DataFrame]:
    """Generates H/D/A probabilities using the Podos model."""
    logger.info("Generating Podos 1X2 predictions...")
    try:
        probabilities = model.predict_proba(X_tensor)
        if probabilities is None or probabilities.shape != (1, 3):
            logger.error(f"Podos predict_proba returned None or incorrect shape.")
            return None
        prob_df = pd.DataFrame(probabilities, columns=['prob_H', 'prob_D', 'prob_A'])
        logger.info("Podos prediction successful.")
        return prob_df
    except Exception as e:
        logger.error(f"Error during Podos prediction: {e}", exc_info=True)
        return None

# --- Strength-Adjusted Lambda Calculation ---
def calculate_strength_adjusted_lambdas(fixture_data: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates attack/defense strengths based on home/away splits vs overall average
    and returns adjusted lambda values for Monte Carlo.
    """
    logger.info("Calculating strength-adjusted lambdas...")

    home_snap = safe_get(fixture_data, ['raw_data', 'home', 'match_processor_snapshot'], {})
    away_snap = safe_get(fixture_data, ['raw_data', 'away', 'match_processor_snapshot'], {})

    # --- Get Average Goals Scored and Conceded (Home/Away/Total) ---
    # Home Team
    h_avg_for_home = safe_get(home_snap, ['goals', 'for', 'average', 'home'])
    h_avg_conceded_home = safe_get(home_snap, ['goals', 'against', 'average', 'home'])
    h_avg_for_total = safe_get(home_snap, ['goals', 'for', 'average', 'total'])
    h_avg_conceded_total = safe_get(home_snap, ['goals', 'against', 'average', 'total'])

    # Away Team
    a_avg_for_away = safe_get(away_snap, ['goals', 'for', 'average', 'away'])
    a_avg_conceded_away = safe_get(away_snap, ['goals', 'against', 'average', 'away'])
    a_avg_for_total = safe_get(away_snap, ['goals', 'for', 'average', 'total'])
    a_avg_conceded_total = safe_get(away_snap, ['goals', 'against', 'average', 'total'])

    # --- Validate data ---
    required_vals = [
        h_avg_for_home, h_avg_conceded_home, h_avg_for_total, h_avg_conceded_total,
        a_avg_for_away, a_avg_conceded_away, a_avg_for_total, a_avg_conceded_total
    ]
    if None in required_vals:
        logger.error("Missing required average goal data in JSON snapshot for strength calculation. Cannot calculate adjusted lambdas.")
        return None, None

    try:
        # Convert to float, handling potential non-numeric types gracefully
        h_avg_for_home = float(h_avg_for_home)
        h_avg_conceded_home = float(h_avg_conceded_home)
        h_avg_for_total = float(h_avg_for_total) if h_avg_for_total > 0 else 1.0 # Avoid division by zero
        h_avg_conceded_total = float(h_avg_conceded_total) if h_avg_conceded_total > 0 else 1.0 # Avoid division by zero

        a_avg_for_away = float(a_avg_for_away)
        a_avg_conceded_away = float(a_avg_conceded_away)
        a_avg_for_total = float(a_avg_for_total) if a_avg_for_total > 0 else 1.0 # Avoid division by zero
        a_avg_conceded_total = float(a_avg_conceded_total) if a_avg_conceded_total > 0 else 1.0 # Avoid division by zero

    except (ValueError, TypeError) as e:
         logger.error(f"Error converting goal average data to float: {e}. Cannot calculate adjusted lambdas.")
         return None, None


    # --- Calculate Strength Ratios ---
    # Attack Strength = Avg Goals For (Home/Away) / Avg Goals For (Total)
    home_attack_strength = h_avg_for_home / h_avg_for_total
    away_attack_strength = a_avg_for_away / a_avg_for_total

    # Defense Strength = Avg Goals Conceded (Home/Away) / Avg Goals Conceded (Total)
    home_defense_strength = h_avg_conceded_home / h_avg_conceded_total
    away_defense_strength = a_avg_conceded_away / a_avg_conceded_total

    # --- Calculate Expected Goals (Lambdas) ---
    # Lambda Home = Home Attack Strength * Away Defense Strength * Home Avg Goals For (Total)
    lambda_home = home_attack_strength * away_defense_strength * h_avg_for_total

    # Lambda Away = Away Attack Strength * Home Defense Strength * Away Avg Goals For (Total)
    lambda_away = away_attack_strength * home_defense_strength * a_avg_for_total

    # Ensure non-negative
    lambda_home = max(0.0, lambda_home)
    lambda_away = max(0.0, lambda_away)

    logger.info(f"Strengths: H Att={home_attack_strength:.2f}, H Def={home_defense_strength:.2f}, A Att={away_attack_strength:.2f}, A Def={away_defense_strength:.2f}")
    logger.info(f"Calculated Strength-Adjusted Lambdas: Home={lambda_home:.3f}, Away={lambda_away:.3f}")

    return lambda_home, lambda_away


# --- Monte Carlo Simulation ---
def evaluate_simulation_scenarios(hg: int, ag: int) -> Tuple[List[str], str]: # Return score string too
    """
    Evaluates a single simulation outcome (hg, ag) for various simple
    and compound scenarios. Includes more combinations like Result/DC + U/O 3.5.
    """
    scenarios_met = set()
    tg = hg + ag # Total goals
    score_string = f"{hg}-{ag}" # Create score string

    # --- Simple Scenarios ---
    # 1X2 Result
    if hg > ag: scenarios_met.add("H")
    elif hg == ag: scenarios_met.add("D")
    else: scenarios_met.add("A")

    # Double Chance
    if hg >= ag: scenarios_met.add("1X") # Home or Draw
    if hg <= ag: scenarios_met.add("X2") # Away or Draw
    if hg != ag: scenarios_met.add("12") # Home or Away

    # BTTS (Both Teams To Score)
    if hg > 0 and ag > 0: scenarios_met.add("BTTS Yes")
    else: scenarios_met.add("BTTS No")

    # Over/Under Goals (Common thresholds)
    if tg > 0.5: scenarios_met.add("O0.5"); scenarios_met.add("U0.5 No") # Explicit No
    else: scenarios_met.add("U0.5"); scenarios_met.add("O0.5 No")

    if tg > 1.5: scenarios_met.add("O1.5"); scenarios_met.add("U1.5 No")
    else: scenarios_met.add("U1.5"); scenarios_met.add("O1.5 No")

    if tg > 2.5: scenarios_met.add("O2.5"); scenarios_met.add("U2.5 No")
    else: scenarios_met.add("U2.5"); scenarios_met.add("O2.5 No")

    if tg > 3.5: scenarios_met.add("O3.5"); scenarios_met.add("U3.5 No")
    else: scenarios_met.add("U3.5"); scenarios_met.add("O3.5 No")

    if tg > 4.5: scenarios_met.add("O4.5"); scenarios_met.add("U4.5 No")
    else: scenarios_met.add("U4.5"); scenarios_met.add("O4.5 No")


    # --- Compound Scenarios (Expanded) ---

    # Result + O/U 2.5
    if "H" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("H and O2.5")
    if "H" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("H and U2.5")
    if "D" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("D and O2.5")
    if "D" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("D and U2.5")
    if "A" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("A and O2.5")
    if "A" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("A and U2.5")

    # Result + O/U 3.5 (Added)
    if "H" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("H and O3.5")
    if "H" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("H and U3.5")
    if "D" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("D and O3.5")
    if "D" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("D and U3.5")
    if "A" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("A and O3.5")
    if "A" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("A and U3.5")

    # Double Chance + BTTS
    if "1X" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("1X and BTTS Yes")
    if "1X" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("1X and BTTS No")
    if "X2" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("X2 and BTTS Yes")
    if "X2" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("X2 and BTTS No")
    if "12" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("12 and BTTS Yes")
    if "12" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("12 and BTTS No") # Added 12 + BTTS No

    # Double Chance + O/U 2.5 (Added)
    if "1X" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("1X and O2.5")
    if "1X" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("1X and U2.5")
    if "X2" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("X2 and O2.5")
    if "X2" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("X2 and U2.5")
    if "12" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("12 and O2.5")
    if "12" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("12 and U2.5")

    # Double Chance + O/U 3.5 (Added)
    if "1X" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("1X and O3.5")
    if "1X" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("1X and U3.5")
    if "X2" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("X2 and O3.5")
    if "X2" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("X2 and U3.5")
    if "12" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("12 and O3.5")
    if "12" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("12 and U3.5")


    # BTTS + O/U
    if "BTTS Yes" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("BTTS Yes and O2.5")
    if "BTTS Yes" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("BTTS Yes and U2.5") # Added
    if "BTTS No" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("BTTS No and O2.5")   # Added
    if "BTTS No" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("BTTS No and U2.5")
    if "BTTS Yes" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("BTTS Yes and O3.5")
    if "BTTS Yes" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("BTTS Yes and U3.5") # Added
    if "BTTS No" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("BTTS No and O3.5")   # Added
    if "BTTS No" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("BTTS No and U3.5")   # Added


    # Result + BTTS
    if "H" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("H and BTTS Yes")
    if "H" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("H and BTTS No")
    if "A" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("A and BTTS Yes")
    if "A" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("A and BTTS No")
    if "D" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("D and BTTS Yes")
    if "D" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("D and BTTS No")

    return list(scenarios_met), score_string

def run_monte_carlo_simulation(
    lambda_home: float,
    lambda_away: float,
    num_simulations: int = MONTE_CARLO_SIMULATIONS,
    random_seed: Optional[int] = 42,
    max_score_plot: int = 5 # Max goals per side for score matrix
) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]: # Return two dicts
    """
    Runs MC simulation for a single fixture based on lambda values.
    Returns:
        - Dictionary of scenario probabilities (prob_X).
        - Dictionary of scoreline probabilities (score_X-Y).
    """
    logger.info(f"Starting Monte Carlo simulation with {num_simulations} runs...")
    if lambda_home < 0 or lambda_away < 0:
        logger.error("Negative lambda values provided for Monte Carlo.")
        return None, None

    if random_seed is not None:
        np.random.seed(random_seed)

    # Simulate goals for the single match
    sim_home_goals = poisson.rvs(lambda_home, size=num_simulations)
    sim_away_goals = poisson.rvs(lambda_away, size=num_simulations)

    # Count scenarios and scorelines across all simulations
    scenario_counts = Counter()
    scoreline_counts = Counter()
    for i in range(num_simulations):
        hg = sim_home_goals[i]
        ag = sim_away_goals[i]
        scenarios_this_run, score_string = evaluate_simulation_scenarios(hg, ag)
        scenario_counts.update(scenarios_this_run)
        # Only count scorelines within the desired plot range
        if hg <= max_score_plot and ag <= max_score_plot:
            scoreline_counts[score_string] += 1 # Use the score string as key

    if not scenario_counts:
        logger.warning("Monte Carlo simulation did not produce any scenarios.")
        return {}, {} # Return empty dicts

    # Calculate probabilities
    total_sims_float = float(num_simulations)
    scenario_probabilities = {
        f"prob_{scenario}": count / total_sims_float
        for scenario, count in scenario_counts.items()
    }
    scoreline_probabilities = {
         f"score_{score}": count / total_sims_float
         for score, count in scoreline_counts.items()
    }

    logger.info(f"Monte Carlo simulation finished. Found {len(scenario_probabilities)} unique scenarios and {len(scoreline_probabilities)} scorelines (up to {max_score_plot}-{max_score_plot}).")
    return scenario_probabilities, scoreline_probabilities

# --- Processing Function for a Single File (Modify to call new plotting function) ---
def process_fixture_json(json_file_path: str, podos_model: PodosTransformer, label_encoder: Any) -> Optional[Dict[str, Any]]:
    """Loads, processes, predicts, ranks, and plots for a single fixture JSON file."""
    logger.info(f"--- Processing Fixture File: {os.path.basename(json_file_path)} ---")
    try:
        with open(json_file_path, 'r') as f:
            fixture_data = json.load(f)
        fixture_id = fixture_data.get("fixture_id", "N/A")
        home_team_name = safe_get(fixture_data, ['raw_data', 'home', 'basic_info', 'name'], 'Home')
        away_team_name = safe_get(fixture_data, ['raw_data', 'away', 'basic_info', 'name'], 'Away')
        logger.info(f"Loaded Fixture ID: {fixture_id} ({home_team_name} vs {away_team_name})")
    except (json.JSONDecodeError, IOError, KeyError) as e:
        logger.error(f"Error reading/parsing JSON file {json_file_path}: {e}")
        return None

    results = {
        "fixture_id": fixture_id,
        "home_team": home_team_name,
        "away_team": away_team_name,
        "file_path": json_file_path,
        "podos_probs": None,
        "mc_probs": None,
        "mc_score_probs": None, # Add field for scoreline probabilities
        "lambdas": (None, None), # Will store the *calculated* lambdas
        "top_n_mc_selections": None # Changed key name slightly for clarity
    }

    # Podos Prediction
    podos_input_tensor = prepare_podos_input_from_json(fixture_data, label_encoder)
    podos_results_df = None
    if podos_input_tensor is not None:
        podos_results_df = predict_podos_probabilities(podos_model, podos_input_tensor)
        if podos_results_df is not None:
             results["podos_probs"] = podos_results_df.iloc[0].to_dict()
        else:
            logger.error(f"Failed Podos prediction for {fixture_id}")
    else:
        logger.error(f"Failed to prepare Podos input for {fixture_id}")

    # Monte Carlo Simulation (using NEW lambda calculation)
    lambdas = calculate_strength_adjusted_lambdas(fixture_data) # Use the new function
    results["lambdas"] = lambdas # Store the calculated lambdas

    mc_scenario_results_dict = None
    mc_score_results_dict = None # New variable for score probs
    if lambdas[0] is not None and lambdas[1] is not None:
         mc_scenario_results_dict, mc_score_results_dict = run_monte_carlo_simulation(
             lambdas[0], lambdas[1], max_score_plot=MC_MAX_SCORE_PLOT
         ) # Capture both outputs
         if mc_scenario_results_dict is not None:
              results["mc_probs"] = mc_scenario_results_dict
              # --- Calculate Top N MC Selections ---
              try:
                  # Sort by probability (value), descending
                  sorted_mc = sorted(mc_scenario_results_dict.items(), key=lambda item: item[1], reverse=True)
                  # Format top N using config
                  results["top_n_mc_selections"] = [ # Use updated key name
                      {"selection": key.replace("prob_", ""), "probability": value}
                      for key, value in sorted_mc[:TOP_N_SCENARIOS] # Use config value (now 10)
                  ]
                  logger.info(f"Calculated Top {TOP_N_SCENARIOS} MC selections for {fixture_id}")
              except Exception as sort_err:
                   logger.error(f"Error sorting MC results for top selections: {sort_err}", exc_info=True)
         else:
              logger.error(f"Failed Monte Carlo scenario simulation for {fixture_id}")

         if mc_score_results_dict is not None:
              results["mc_score_probs"] = mc_score_results_dict # Store score probabilities
         else:
              logger.error(f"Failed Monte Carlo scoreline simulation for {fixture_id}")

    else:
         logger.error(f"Failed to calculate strength-adjusted lambdas for MC simulation for {fixture_id}") # Updated error message

    # --- Plotting ---
    # Create plots directory if it doesn't exist (moved here for safety)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if results: # Only plot if processing was somewhat successful
        # Call the combined plotting function from the utils file
        create_combined_fixture_plot(results, PLOTS_DIR, max_goals_matrix=MC_MAX_SCORE_PLOT)

    logger.info(f"--- Finished Processing: {os.path.basename(json_file_path)} ---")
    return results

# --- Main Execution Logic ---
if __name__ == '__main__':
    logger.info("--- Starting Batch Fixture Processing ---")

    # Create the input directory if it doesn't exist
    if not os.path.isdir(UNIFIED_DATA_DIR):
        logger.info(f"Creating input directory: {UNIFIED_DATA_DIR}")
        try:
            os.makedirs(UNIFIED_DATA_DIR, exist_ok=True)
            logger.info(f"Successfully created input directory: {UNIFIED_DATA_DIR}")
        except Exception as e:
            logger.error(f"Failed to create input directory {UNIFIED_DATA_DIR}: {e}")
            sys.exit(1) # Exit if we can't create input dir

    # Create the output directory if it doesn't exist
    if not os.path.isdir(OUTPUT_DIR):
        logger.info(f"Creating output directory: {OUTPUT_DIR}")
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            logger.info(f"Successfully created output directory: {OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"Failed to create output directory {OUTPUT_DIR}: {e}")
            # sys.exit(1) # Decide if you want to exit here too

    # --- Load Models Once ---
    logger.info("--- Loading Predictive Components ---")
    podos_model, label_encoder = load_podos_components()
    if not podos_model or not label_encoder:
        logger.error("Failed to load Podos components. Exiting.")
        sys.exit(1)

    # --- Find JSON Files ---
    json_files = glob.glob(os.path.join(UNIFIED_DATA_DIR, "*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in directory: {UNIFIED_DATA_DIR}")
        sys.exit(0)

    logger.info(f"Found {len(json_files)} JSON files to process.")

    # --- Process Each File ---
    all_results = []
    for json_file_path in json_files:
        try:
            fixture_results = process_fixture_json(json_file_path, podos_model, label_encoder)
            if fixture_results:
                all_results.append(fixture_results)
        except Exception as e:
            logger.error(f"Unexpected error processing file {json_file_path}: {e}", exc_info=True)

    # --- Display Aggregated Results Summary (Updated) ---
    logger.info("--- Batch Processing Summary ---")
    logger.info(f"Successfully processed {len(all_results)} out of {len(json_files)} files.")

    print("\n" + "="*70)
    print("                     BATCH PREDICTION RESULTS SUMMARY")
    print("="*70 + "\n")

    for result in all_results:
        print(f"--- Fixture: {result['fixture_id']} ({result['home_team']} vs {result['away_team']}) ---")
        print(f"File: {os.path.basename(result['file_path'])}")

        # Podos Output
        if result['podos_probs']:
            probs = result['podos_probs']
            print(" Podos (H/D/A):  "
                  f"{probs.get('prob_H', 0):.3f} / "
                  f"{probs.get('prob_D', 0):.3f} / "
                  f"{probs.get('prob_A', 0):.3f}")
            if "PLACEHOLDER" in str(type(podos_model)):
                 print("  ** WARNING: Podos results use a PLACEHOLDER model! **")

        else:
            print(" Podos: Failed")

        # MC Output & Top N
        if result['mc_probs']:
             lambda_h, lambda_a = result['lambdas']
             lambda_h_str = f"{lambda_h:.3f}" if lambda_h is not None else "N/A"
             lambda_a_str = f"{lambda_a:.3f}" if lambda_a is not None else "N/A"
             print(f" MC ({MONTE_CARLO_SIMULATIONS} sims, Lambdas H/A: {lambda_h_str}/{lambda_a_str}):") # Updated label
             # Display a few key MC results
             o25_prob = result['mc_probs'].get('prob_O2.5', 0.0)
             btts_prob = result['mc_probs'].get('prob_BTTS Yes', 0.0)
             print(f"  P(O2.5)  : {o25_prob:.3f}")
             print(f"  P(BTTS=Y): {btts_prob:.3f}")

             # --- Display Top N Selections ---
             if result['top_n_mc_selections']: # Use updated key name
                 print(f"  Top {TOP_N_SCENARIOS} MC Selections:") # Use config value (now 10)
                 for i, item in enumerate(result['top_n_mc_selections']):
                      # Adjust padding if necessary for longer scenario names
                      print(f"   {i+1: >2}. {item['selection']:<30} | P = {item['probability']:.4f}")
             else:
                 print(f"  Top {TOP_N_SCENARIOS} MC Selections: Calculation failed or no MC results.")
        else:
             print(" MC: Failed (or lambdas could not be calculated)") # Updated reason

        print("-" * 70)

    # --- Save Results ---
    output_filename = os.path.join(OUTPUT_DIR, "batch_prediction_results.json")
    try:
        # Use a custom encoder to handle potential numpy types if necessary
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, tuple) and any(x is None for x in obj):
                    return [float('nan') if x is None else x for x in obj]
                return super(NpEncoder, self).default(obj)

        with open(output_filename, 'w') as f:
            json.dump(all_results, f, indent=4, cls=NpEncoder) # Use encoder
        logger.info(f"Saved detailed results (incl. top {TOP_N_SCENARIOS} & score probs) to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save results to file: {e}")

    logger.info("--- Script Finished ---")