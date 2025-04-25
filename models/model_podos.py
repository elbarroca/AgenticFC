# models/predict_with_podos_from_json.py # Renamed for clarity
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PODOS_MODEL_REPO = "Bettensor/podos_soccer_model"
PODOS_LABEL_ENCODER_FILENAME = "label_encoder.joblib"
PODOS_MODEL_FILENAME = "model.safetensors" # Or "pytorch_model.bin" if that exists

# Define the exact 23 features the Podos model expects IN ORDER
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
            # In real implementation: load state_dict here
            instance = cls()
            instance._is_fitted = True # Simulate successful load
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
        logger.info(f"Generated placeholder probabilities shape: {probabilities.shape}")
        return probabilities

# --- Loading Function ---
def load_podos_components() -> Optional[Tuple[PodosTransformer, Any]]:
    # (Same as before - downloads real encoder, placeholder model)
    logger.info(f"Attempting to load Podos components from repo: {PODOS_MODEL_REPO}")
    
    # First try to load the model, which is required
    try:
        model = PodosTransformer.from_pretrained(
            repo_id=PODOS_MODEL_REPO,
            filename=PODOS_MODEL_FILENAME
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
            repo_id=PODOS_MODEL_REPO,
            filename=PODOS_LABEL_ENCODER_FILENAME
        )
        label_encoder = joblib_load(encoder_path)
        encoder_type = type(label_encoder).__name__
        logger.info(f"Successfully loaded label encoder type: {encoder_type}")
        if hasattr(label_encoder, 'classes_'):
            num_classes = len(label_encoder.classes_)
            logger.info(f"Label encoder contains {num_classes} classes/teams.")
        else:
             logger.warning("Loaded encoder does not have 'classes_' attribute. Team mapping might behave unexpectedly.")
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
                encoder_type = type(label_encoder).__name__
                logger.info(f"Successfully loaded label encoder type: {encoder_type}")
                if hasattr(label_encoder, 'classes_'):
                    num_classes = len(label_encoder.classes_)
                    logger.info(f"Label encoder contains {num_classes} classes/teams.")
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

# --- Helper function for safe dictionary access ---
def safe_get(data: Dict, keys: List[str], default: Any = None) -> Any:
    """Safely traverse nested dictionary keys."""
    if not isinstance(data, dict) or not isinstance(keys, list):
        return default
    current = data
    for key in keys:
        try:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, (list, tuple)) and isinstance(key, int):
                if 0 <= key < len(current):
                    current = current[key]
                else:
                    return default # Index out of bounds
            else:
                 return default # Cannot traverse further
            if current is None: # Stop if any intermediate key is None
                 return default
        except (TypeError, KeyError, IndexError):
            return default
    return current if current is not None else default # Final check

# --- Helper function to parse form string ---
def parse_form_string(form_str: Optional[str], num_matches: int = 5) -> Tuple[int, int, float]:
    # (Same as before)
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

# --- Data Preparation Function for Single JSON (Revised) ---
def prepare_podos_input_from_json(fixture_data: Dict, label_encoder: Any) -> Optional[torch.Tensor]:
    """
    Extracts, maps, and transforms features from the fixture JSON dictionary
    to the format required by the PodosTransformer model, using defaults ONLY
    when data is truly absent in the JSON structure.
    """
    logger.info("Preparing data for Podos model input from JSON (best effort mapping)...")
    features = {}

    # Access paths within the JSON
    raw_home_snap = safe_get(fixture_data, ['raw_data', 'home', 'match_processor_snapshot'], {})
    raw_away_snap = safe_get(fixture_data, ['raw_data', 'away', 'match_processor_snapshot'], {})
    pred_snap_home = safe_get(raw_home_snap, ['predictions_snapshot'], {})
    pred_snap_away = safe_get(raw_away_snap, ['predictions_snapshot'], {})
    eng_feat_home = safe_get(fixture_data, ['engineered_features', 'home'], {})
    eng_feat_away = safe_get(fixture_data, ['engineered_features', 'away'], {})

    # --- Feature Extraction / Calculation / Defaults ---

    # Played Games (for calculating averages)
    home_played = safe_get(raw_home_snap, ['fixtures', 'played', 'total'], 0)
    away_played = safe_get(raw_away_snap, ['fixtures', 'played', 'total'], 0)
    logger.info(f"Games Played: Home={home_played}, Away={away_played}")

    # Shots (HS, AS, HST, AST) - USING AVERAGE GOALS AS PROXY
    # This remains a significant simplification. Podos expects actual shots.
    home_avg_goals_for = safe_get(raw_home_snap, ['goals', 'for', 'average', 'total'], 0.0)
    away_avg_goals_for = safe_get(raw_away_snap, ['goals', 'for', 'average', 'total'], 0.0)
    features['HS'] = home_avg_goals_for * 8.0 # Crude ratio: Guess 8 shots per goal scored avg
    features['AS'] = away_avg_goals_for * 8.0 # Crude ratio: Guess 8 shots per goal scored avg
    features['HST'] = features['HS'] * 0.4    # Crude ratio: Guess 40% SOT
    features['AST'] = features['AS'] * 0.4    # Crude ratio: Guess 40% SOT
    logger.warning("Using calculated AVERAGE GOALS FOR * ratio as PROXY for HS, AS, HST, AST.")

    # Corners (HC, AC) - Data ABSENT in JSON structure provided
    # We MUST default this as it cannot be derived.
    features['HC'] = 5.0 # Using a generic average default
    features['AC'] = 5.0 # Using a generic average default
    logger.warning("Corner data (HC, AC) ABSENT in JSON. Using default value: 5.0")

    # Offsides (HO, AO) - Data ABSENT in JSON structure provided
    features['HO'] = 2.0 # Using a generic average default
    features['AO'] = 2.0 # Using a generic average default
    logger.warning("Offside data (HO, AO) ABSENT in JSON. Using default value: 2.0")

    # Yellow Cards (HY, AY) - Calculated from totals / played
    home_yellows_total = sum(safe_get(raw_home_snap, ['cards', 'yellow', k, 'total'], 0) for k in raw_home_snap.get('cards', {}).get('yellow', {}).keys())
    away_yellows_total = sum(safe_get(raw_away_snap, ['cards', 'yellow', k, 'total'], 0) for k in raw_away_snap.get('cards', {}).get('yellow', {}).keys())
    features['HY'] = (home_yellows_total / home_played) if home_played > 0 else 0.0
    features['AY'] = (away_yellows_total / away_played) if away_played > 0 else 0.0
    logger.info(f"Calculated Avg Yellows: HY={features['HY']:.2f}, AY={features['AY']:.2f}")

    # Red Cards (HR, AR) - Calculated from totals / played
    home_reds_total = sum(safe_get(raw_home_snap, ['cards', 'red', k, 'total'], 0) for k in raw_home_snap.get('cards', {}).get('red', {}).keys())
    away_reds_total = sum(safe_get(raw_away_snap, ['cards', 'red', k, 'total'], 0) for k in raw_away_snap.get('cards', {}).get('red', {}).keys())
    features['HR'] = (home_reds_total / home_played) if home_played > 0 else 0.0
    features['AR'] = (away_reds_total / away_played) if away_played > 0 else 0.0
    logger.info(f"Calculated Avg Reds: HR={features['HR']:.2f}, AR={features['AR']:.2f}")


    # Odds (oddsH, oddsD, oddsA) - Data ABSENT in JSON structure provided
    # This is a CRITICAL input for Podos. Defaulting significantly impacts results.
    features['oddsH'] = 2.50 # *** DEFAULT VALUE ***
    features['oddsD'] = 3.50 # *** DEFAULT VALUE ***
    features['oddsA'] = 2.80 # *** DEFAULT VALUE ***
    logger.critical("Odds data (oddsH, oddsD, oddsA) ABSENT in JSON. Using default values: H=2.50, D=3.50, A=2.80. Prediction quality WILL be poor.")
    # TODO: Modify your system to include pre-match odds in the JSON if possible.

    # Team ID Encoding - Directly from 'engineered_features'
    home_id = safe_get(eng_feat_home, ['team_id'])
    away_id = safe_get(eng_feat_away, ['team_id'])
    unknown_team_value = 0 # Value if team not in encoder
    logger.info(f"Raw Team IDs: Home={home_id}, Away={away_id}")

    try:
        known_classes = set(label_encoder.classes_) if hasattr(label_encoder, 'classes_') else None
        if known_classes is not None:
            features['home_encoded'] = label_encoder.transform([home_id])[0] if home_id in known_classes else unknown_team_value
            features['away_encoded'] = label_encoder.transform([away_id])[0] if away_id in known_classes else unknown_team_value
        else:
             logger.warning("Label encoder classes not available, using default team value.")
             features['home_encoded'] = unknown_team_value
             features['away_encoded'] = unknown_team_value

        if features['home_encoded'] == unknown_team_value: logger.warning(f"Home team ID {home_id} not found in encoder.")
        if features['away_encoded'] == unknown_team_value: logger.warning(f"Away team ID {away_id} not found in encoder.")
        logger.info(f"Encoded Team IDs: Home={features['home_encoded']}, Away={features['away_encoded']}")

    except Exception as e:
         logger.error(f"Error during team ID transformation: {e}. Using default value {unknown_team_value}.", exc_info=True)
         features['home_encoded'] = unknown_team_value
         features['away_encoded'] = unknown_team_value


    # Streaks and Form (Using Form String Parsing)
    home_form_str = safe_get(raw_home_snap, ['form_string'])
    away_form_str = safe_get(raw_away_snap, ['form_string'])
    logger.info(f"Raw Form Strings: Home='{home_form_str}', Away='{away_form_str}'")

    wsH, lsH, formH_ppg = parse_form_string(home_form_str)
    wsA, lsA, formA_ppg = parse_form_string(away_form_str)

    features['WinStreakHome'] = wsH
    features['LossStreakHome'] = lsH
    features['WinStreakAway'] = wsA
    features['LossStreakAway'] = lsA

    # Use prediction snapshot form (numeric) if available, otherwise use calculated PPG from main string
    # Ensure the snapshot form value is actually numeric
    home_form_snapshot = safe_get(pred_snap_home, ['last_5', 'form_string'])
    away_form_snapshot = safe_get(pred_snap_away, ['last_5', 'form_string'])

    try:
        features['HomeTeamForm'] = float(home_form_snapshot) if home_form_snapshot is not None else formH_ppg
    except (ValueError, TypeError):
        logger.warning(f"Could not convert home snapshot form '{home_form_snapshot}' to float. Using PPG from main string: {formH_ppg:.2f}")
        features['HomeTeamForm'] = formH_ppg

    try:
        features['AwayTeamForm'] = float(away_form_snapshot) if away_form_snapshot is not None else formA_ppg
    except (ValueError, TypeError):
        logger.warning(f"Could not convert away snapshot form '{away_form_snapshot}' to float. Using PPG from main string: {formA_ppg:.2f}")
        features['AwayTeamForm'] = formA_ppg

    logger.info(f"Streaks: H(W/L)={wsH}/{lsH}, A(W/L)={wsA}/{lsA}")
    logger.info(f"Form: Home={features['HomeTeamForm']:.2f}, Away={features['AwayTeamForm']:.2f}")

    # --- Validation and Formatting ---
    # Create DataFrame with the single row of features
    try:
        df_prepared = pd.DataFrame([features])
        # Reorder columns exactly as expected and select only those
        df_final = df_prepared[PODOS_EXPECTED_FEATURES]
        logger.info("Feature dictionary created and columns ordered.")
    except KeyError as e:
         logger.error(f"Critical Error: Failed to create DataFrame. Derived feature missing: {e}", exc_info=True)
         logger.error(f"Available derived features: {list(features.keys())}")
         return None

    # Convert columns to numeric, coercing errors (should be minimal now)
    df_numeric = df_final.apply(pd.to_numeric, errors='coerce')

    # Check for NaNs and impute (using 0 as a simple strategy)
    nan_counts = df_numeric.isnull().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaNs found in final Podos input columns AFTER processing JSON:\n{nan_counts[nan_counts > 0]}")
        df_imputed = df_numeric.fillna(0)
        logger.info("NaNs imputed with 0.")
    else:
        df_imputed = df_numeric
        logger.info("No NaNs detected in final numeric feature set.")

    # Final Check: Display the prepared features before tensor conversion
    logger.debug("Final features before tensor conversion:\n%s", df_imputed.to_string())


    # Convert to PyTorch Tensor
    try:
        podos_input_tensor = torch.tensor(df_imputed.values, dtype=torch.float32)
        if podos_input_tensor.shape == (1, len(PODOS_EXPECTED_FEATURES)):
            logger.info(f"Successfully prepared Podos input tensor with shape: {podos_input_tensor.shape}")
            return podos_input_tensor
        else:
            logger.error(f"Tensor shape mismatch after preparation: Expected (1, {len(PODOS_EXPECTED_FEATURES)}), Got {podos_input_tensor.shape}")
            return None
    except Exception as e:
        logger.error(f"Error converting final DataFrame to Tensor: {e}", exc_info=True)
        return None


# --- Prediction Function ---
def predict_podos_probabilities(model: PodosTransformer, X_tensor: torch.Tensor) -> Optional[pd.DataFrame]:
    # (Same as before)
    logger.info(f"Generating 1X2 predictions using Podos model for {X_tensor.shape[0]} sample(s)...")
    if not isinstance(X_tensor, torch.Tensor) or len(X_tensor.shape) != 2 or X_tensor.shape[1] != len(PODOS_EXPECTED_FEATURES):
         logger.error(f"Input is not a valid 2D tensor or has incorrect features ({X_tensor.shape[1]} != {len(PODOS_EXPECTED_FEATURES)}).")
         return None
    try:
        probabilities = model.predict_proba(X_tensor)
        if probabilities is None or probabilities.shape[0] != X_tensor.shape[0] or probabilities.shape[1] != 3:
            logger.error(f"Podos predict_proba returned None or incorrect shape: {probabilities.shape if probabilities is not None else 'None'}")
            return None
        prob_df = pd.DataFrame(probabilities, columns=['prob_H', 'prob_D', 'prob_A'])
        logger.info("Podos prediction successful.")
        return prob_df
    except Exception as e:
        logger.error(f"Error during Podos prediction: {e}", exc_info=True)
        return None

# --- Main Execution Logic ---
if __name__ == '__main__':
    # Load the input JSON file
    import sys
    import os
    
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        if not os.path.exists(json_file_path):
            logger.error(f"JSON file not found: {json_file_path}")
            sys.exit(1)
            
        logger.info(f"Using JSON file: {json_file_path}")
        try:
            with open(json_file_path, 'r') as f:
                fixture_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading JSON file: {e}")
            sys.exit(1)
    else:
        logger.error("No JSON file provided. Usage: python model_podos.py <path_to_json_file>")
        sys.exit(1)

    # --- Execution ---
    logger.info("--- Loading Model Components ---")
    podos_model, label_encoder = load_podos_components()

    if podos_model and label_encoder:
        logger.info("--- Preparing Input Data from JSON ---")
        if fixture_data:
            input_tensor = prepare_podos_input_from_json(fixture_data, label_encoder)

            if input_tensor is not None:
                logger.info("--- Running Prediction ---")
                results_df = predict_podos_probabilities(podos_model, input_tensor)

                if results_df is not None:
                    logger.info("--- Prediction Results ---")
                    home_team = safe_get(fixture_data, ['raw_data', 'home', 'basic_info', 'name'], 'Home')
                    away_team = safe_get(fixture_data, ['raw_data', 'away', 'basic_info', 'name'], 'Away')
                    fixture_id = fixture_data.get("fixture_id", "N/A")

                    print(f"\nPodos Model Prediction Probabilities for Fixture ID: {fixture_id}")
                    print(f"{home_team} vs {away_team}")
                    print("-" * 40)
                    print(f" P(Home Win) : {results_df.loc[0, 'prob_H']:.4f} ({results_df.loc[0, 'prob_H']*100:.2f}%)")
                    print(f" P(Draw)     : {results_df.loc[0, 'prob_D']:.4f} ({results_df.loc[0, 'prob_D']*100:.2f}%)")
                    print(f" P(Away Win) : {results_df.loc[0, 'prob_A']:.4f} ({results_df.loc[0, 'prob_A']*100:.2f}%)")
                    print("-" * 40)
                    print(f" Sum Check   : {results_df.loc[0].sum():.4f}")
                    if "PLACEHOLDER" in str(type(podos_model)):
                         print("\n**NOTE:** Results are from the **PLACEHOLDER** PodosTransformer.")
                         print("Replace the placeholder class definition for real predictions.")
                    print("**NOTE:** Input features derived from JSON may use proxies or defaults")
                    print("        due to missing data (e.g., Odds, Corners, Offsides). See logs.")
                else: logger.error("Prediction failed.")
            else: logger.error("Failed to prepare input data from JSON.")
    else: logger.error("Failed to load Podos components. Exiting.")

    logger.info("--- Script Finished ---")