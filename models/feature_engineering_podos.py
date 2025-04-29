# feature_engineering_podos.py
import json
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Tuple, List

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from huggingface_hub import hf_hub_download
from pymongo import MongoClient, DESCENDING
from sklearn.preprocessing import LabelEncoder

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "agenticfc")
MATCHES_COLLECTION = "matches"
ODDS_COLLECTION = "odds"

# --- Podos Model Configuration ---
PODOS_MODEL_REPO = "Nickel5HF/podos_soccer_model"
PODOS_LABEL_ENCODER_FILENAMES = [
    "label_encoder.joblib",
    "encoder.joblib",
    "team_encoder.joblib",
    "label_encoder.pkl",
    "encoder.pkl",
    "team_encoder.pkl",
]
NUM_HISTORICAL_GAMES = 15
PODOS_EXPECTED_FEATURES = [
    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HO', 'AO', 'HY', 'AY', 'HR', 'AR',
    'oddsH', 'oddsD', 'oddsA',
    'home_encoded', 'away_encoded',
    'WinStreakHome', 'LossStreakHome', 'WinStreakAway', 'LossStreakAway',
    'HomeTeamForm', 'AwayTeamForm'
]

# --- Helper Function to Create the Specific ID->Name Map (moved here) ---
def create_mongodb_id_to_name_map(source_mapping: Dict[str, Dict]) -> Optional[Dict[str, str]]:
    """
    Processes the TEAM_ID_MAPPING dictionary into a simple map from
    mongodb_id (string) to team_name (string - the key from the source).
    """
    logger.info("Processing TEAM_ID_MAPPING dictionary to create mongodb_id -> team_name map...")
    if not isinstance(source_mapping, dict):
        logger.error("Source TEAM_ID_MAPPING is not a dictionary.")
        return None

    mongodb_id_to_name_map = {}
    missing_mongo_id_count = 0
    invalid_entry_count = 0

    for team_name, details in source_mapping.items():
        if isinstance(details, dict) and 'mongodb_id' in details:
            mongo_id_val = details['mongodb_id']
            if mongo_id_val is not None:
                mongo_id = str(mongo_id_val)
                if mongo_id in mongodb_id_to_name_map:
                    logger.warning(f"Duplicate mongodb_id '{mongo_id}' found in mapping for teams '{mongodb_id_to_name_map[mongo_id]}' and '{team_name}'. Using the latter.")
                mongodb_id_to_name_map[mongo_id] = team_name
            else:
                missing_mongo_id_count += 1
                logger.warning(f"Missing 'mongodb_id' value (it is None) for team '{team_name}' in mapping.")
        else:
            invalid_entry_count += 1
            logger.warning(f"Invalid or missing 'mongodb_id' key for team '{team_name}'. Details: {details}")

    if missing_mongo_id_count > 0:
        logger.warning(f"Found {missing_mongo_id_count} entries with missing 'mongodb_id' values.")
    if invalid_entry_count > 0:
        logger.warning(f"Found {invalid_entry_count} invalid entries (not dict or missing 'mongodb_id' key).")

    if not mongodb_id_to_name_map:
        logger.error("Created empty mongodb_id_to_name_map. Source mapping might be empty or all entries lacked valid 'mongodb_id'.")
        return None

    logger.info(f"Successfully processed team mapping. Found {len(mongodb_id_to_name_map)} valid mongodb_id -> team_name entries.")
    return mongodb_id_to_name_map

# --- Helper function for safe dictionary access ---
def safe_get(data: Optional[Dict], keys: List[str], default: Any = None) -> Any:
    if data is None:
        return default
    temp = data
    for key in keys:
        if isinstance(temp, dict) and key in temp:
            temp = temp[key]
        elif isinstance(temp, list) and isinstance(key, int) and 0 <= key < len(temp):
            temp = temp[key]
        else:
            return default
    return temp

# --- Function to Load Label Encoder ---
def load_label_encoder() -> Optional[LabelEncoder]:
    label_encoder = None
    logger.info(f"Attempting to load label encoder from {PODOS_MODEL_REPO} using potential filenames...")

    for filename in PODOS_LABEL_ENCODER_FILENAMES:
        logger.debug(f"Trying filename: {filename}")
        try:
            encoder_path = hf_hub_download(
                repo_id=PODOS_MODEL_REPO,
                filename=filename
            )
            label_encoder = joblib_load(encoder_path)
            logger.info(f"Successfully loaded label encoder '{filename}' (type: {type(label_encoder).__name__})")
            if not hasattr(label_encoder, 'classes_'):
                logger.warning(f"Loaded encoder '{filename}' lacks 'classes_' attribute.")
            if hasattr(label_encoder, 'classes_') and "UNKNOWN" not in label_encoder.classes_:
                logger.info("Adding 'UNKNOWN' class to loaded label encoder for robustness.")
                label_encoder.classes_ = np.append(label_encoder.classes_, "UNKNOWN")
            return label_encoder
        except Exception as e:
            if "404" in str(e) or "EntryNotFoundError" in str(e) or "RepositoryNotFoundError" in str(e):
                logger.debug(f"Encoder file '{filename}' not found in repo '{PODOS_MODEL_REPO}'.")
            else:
                logger.warning(f"Could not load label encoder '{filename}'. Error: {e}")

    logger.warning("Could not load any valid label encoder from the repository. Using a dummy encoder.")
    dummy_encoder = LabelEncoder()
    dummy_encoder.fit(["UNKNOWN_TEAM_1", "UNKNOWN_TEAM_2"])
    if "UNKNOWN" not in dummy_encoder.classes_:
        dummy_encoder.classes_ = np.append(dummy_encoder.classes_, "UNKNOWN")
    label_encoder = dummy_encoder
    logger.warning("Using dummy label encoder. All unknown teams will be mapped to 'UNKNOWN'.")
    return label_encoder

# --- Core Feature Generation Logic ---
def generate_podos_features(
    fixture_id: str,
    matches_coll: Any,
    odds_coll: Any,
    label_encoder: LabelEncoder,
    mongodb_id_to_name_map: Dict[str, str],
    num_historical: int = NUM_HISTORICAL_GAMES
) -> Optional[pd.DataFrame]:
    logger.info(f"--- Generating Podos features for fixture_id: {fixture_id} ---")
    features = {}

    # 1. Fetch Target Fixture Details
    logger.debug(f"Fetching target fixture data for ID: {fixture_id}")
    try:
        target_fixture = matches_coll.find_one({"_id": str(fixture_id)})
    except Exception as e:
        logger.error(f"Error querying MongoDB for fixture {fixture_id}: {e}")
        return None

    if not target_fixture:
        logger.error(f"Fixture ID '{fixture_id}' not found in matches collection.")
        return None

    home_db_id = str(safe_get(target_fixture, ['home_team_id'], ''))
    away_db_id = str(safe_get(target_fixture, ['away_team_id'], ''))
    match_date_utc = None

    date_val = safe_get(target_fixture, ['date_utc'])
    if isinstance(date_val, datetime):
        match_date_utc = date_val
    elif isinstance(date_val, str):
        try:
            match_date_utc = pd.to_datetime(date_val).to_pydatetime()
        except ValueError:
            logger.warning(f"Could not parse date string '{date_val}' for fixture {fixture_id}.")

    if match_date_utc is None:
        logger.error(f"Could not determine valid match date for fixture {fixture_id}.")
        return None
    if match_date_utc.tzinfo is None:
        match_date_utc = match_date_utc.replace(tzinfo=timezone.utc)
        logger.debug(f"Added UTC timezone to naive datetime for fixture {fixture_id}")

    if not home_db_id or home_db_id == 'None' or not away_db_id or away_db_id == 'None':
        logger.error(f"Missing or invalid home_team_id ('{home_db_id}') or away_team_id ('{away_db_id}') in fixture {fixture_id}.")
        return None

    home_team_name = mongodb_id_to_name_map.get(home_db_id)
    away_team_name = mongodb_id_to_name_map.get(away_db_id)

    if not home_team_name: logger.warning(f"Could not find team name for home_db_id '{home_db_id}' in mapping file.")
    if not away_team_name: logger.warning(f"Could not find team name for away_db_id '{away_db_id}' in mapping file.")

    home_display_name = home_team_name or f"Home_ID_{home_db_id}"
    away_display_name = away_team_name or f"Away_ID_{away_db_id}"
    logger.info(f"Target Match: {home_display_name} (DB_ID: {home_db_id}) vs {away_display_name} (DB_ID: {away_db_id}) on {match_date_utc.date()}")

    # 2. Fetch Pre-Match Odds
    logger.debug(f"Fetching odds data for fixture ID: {fixture_id}")
    try:
        odds_data = odds_coll.find_one({"_id": str(fixture_id)})
        odds_found = False
        if odds_data and 'bookmakers' in odds_data and isinstance(odds_data['bookmakers'], list):
            bookmaker_priority = ['Pinnacle', 'bet365']
            target_bookmaker_data = None

            for bm_name in bookmaker_priority:
                for bm in odds_data['bookmakers']:
                    if bm.get('name') == bm_name:
                        target_bookmaker_data = bm
                        break
                if target_bookmaker_data:
                    break

            if not target_bookmaker_data and odds_data['bookmakers']:
                target_bookmaker_data = odds_data['bookmakers'][0]
                logger.warning(f"Priority bookmakers not found for {fixture_id}. Using first available: {target_bookmaker_data.get('name', 'Unknown')}")

            if target_bookmaker_data and 'bets' in target_bookmaker_data and isinstance(target_bookmaker_data['bets'], list):
                for bet in target_bookmaker_data['bets']:
                    if bet.get('name') == 'Match Winner' and 'values' in bet and isinstance(bet['values'], list):
                        odds_values = {val.get('value'): float(val.get('odd', 0.0)) for val in bet['values'] if 'value' in val and 'odd' in val}
                        if 'Home' in odds_values and 'Draw' in odds_values and 'Away' in odds_values:
                            features['oddsH'] = odds_values['Home']
                            features['oddsD'] = odds_values['Draw']
                            features['oddsA'] = odds_values['Away']
                            odds_found = True
                            logger.info(f"Found Match Winner odds from '{target_bookmaker_data.get('name', 'Unknown')}': H={features['oddsH']}, D={features['oddsD']}, A={features['oddsA']}")
                            break

        if not odds_found:
            features['oddsH'] = 2.50; features['oddsD'] = 3.40; features['oddsA'] = 2.90
            logger.critical(f"CRITICAL: Odds not found/parsed for {fixture_id}. Structure invalid or 'Match Winner' bet missing/malformed. Using defaults: H=2.5, D=3.4, A=2.9. Results unreliable.")

    except Exception as e:
        features['oddsH'] = 2.50; features['oddsD'] = 3.40; features['oddsA'] = 2.90
        logger.error(f"Error fetching or parsing odds for {fixture_id}: {e}. Using defaults.", exc_info=True)

    # 3. Fetch and Process Historical Matches (Home Team)
    logger.info(f"Processing historical data for Home Team: {home_display_name} (ID: {home_db_id})")
    try:
        home_hist_stats, home_hist_matches = calculate_historical_stats(home_db_id, match_date_utc, matches_coll, num_historical)
        features.update({f'H{stat}': avg for stat, avg in home_hist_stats.items()})
    except Exception as e:
        logger.error(f"Error calculating historical stats for home team {home_db_id}: {e}", exc_info=True)
        return None

    # 4. Fetch and Process Historical Matches (Away Team)
    logger.info(f"Processing historical data for Away Team: {away_display_name} (ID: {away_db_id})")
    try:
        away_hist_stats, away_hist_matches = calculate_historical_stats(away_db_id, match_date_utc, matches_coll, num_historical)
        features.update({f'A{stat}': avg for stat, avg in away_hist_stats.items()})
    except Exception as e:
        logger.error(f"Error calculating historical stats for away team {away_db_id}: {e}", exc_info=True)
        return None

    # 5. Calculate Streaks and Form (Home Team)
    try:
        wsH, lsH, formH_ppg = calculate_streaks_and_form(home_db_id, home_hist_matches, num_historical)
        features['WinStreakHome'] = float(wsH); features['LossStreakHome'] = float(lsH)
        features['HomeTeamForm'] = formH_ppg
        logger.info(f"Home Team Form/Streaks (Last {min(len(home_hist_matches), num_historical)} games): WStreak={wsH}, LStreak={lsH}, FormPPG={formH_ppg:.3f}")
    except Exception as e:
        logger.error(f"Error calculating streaks/form for home team {home_db_id}: {e}", exc_info=True)
        return None

    # 6. Calculate Streaks and Form (Away Team)
    try:
        wsA, lsA, formA_ppg = calculate_streaks_and_form(away_db_id, away_hist_matches, num_historical)
        features['WinStreakAway'] = float(wsA); features['LossStreakAway'] = float(lsA)
        features['AwayTeamForm'] = formA_ppg
        logger.info(f"Away Team Form/Streaks (Last {min(len(away_hist_matches), num_historical)} games): WStreak={wsA}, LStreak={lsA}, FormPPG={formA_ppg:.3f}")
    except Exception as e:
        logger.error(f"Error calculating streaks/form for away team {away_db_id}: {e}", exc_info=True)
        return None

    # 7. Encode Team IDs
    home_identifier = home_team_name
    away_identifier = away_team_name
    logger.debug(f"Attempting to encode identifiers (using mapped NAMES): Home='{home_identifier}', Away='{away_identifier}'")

    try:
        known_classes_list = list(label_encoder.classes_)
        if "UNKNOWN" not in known_classes_list:
            logger.error("FATAL: LabelEncoder is missing the 'UNKNOWN' class. Cannot proceed reliably.")
            unknown_index = 0
            logger.warning("Using index 0 as fallback for UNKNOWN. This is not ideal.")
        else:
            unknown_index = known_classes_list.index("UNKNOWN")

        if home_identifier and home_identifier in known_classes_list:
            features['home_encoded'] = float(label_encoder.transform([home_identifier])[0])
        else:
            if home_identifier:
                reason = f"mapped name '{home_identifier}' not found in label encoder classes"
            else:
                reason = f"DB ID '{home_db_id}' not found in team mapping file"
            logger.warning(f"Home team encoding failed ({reason}). Mapping to 'UNKNOWN' (index {unknown_index}).")
            features['home_encoded'] = float(unknown_index)

        if away_identifier and away_identifier in known_classes_list:
            features['away_encoded'] = float(label_encoder.transform([away_identifier])[0])
        else:
            if away_identifier:
                reason = f"mapped name '{away_identifier}' not found in label encoder classes"
            else:
                reason = f"DB ID '{away_db_id}' not found in team mapping file"
            logger.warning(f"Away team encoding failed ({reason}). Mapping to 'UNKNOWN' (index {unknown_index}).")
            features['away_encoded'] = float(unknown_index)

        logger.info(f"Encoded Team IDs (based on names, fallback UNKNOWN): Home={features['home_encoded']}, Away={features['away_encoded']}")

    except ValueError as e:
        logger.error(f"Error during team NAME transformation using label encoder: {e}. Using fallback index {unknown_index}.", exc_info=True)
        features['home_encoded'] = float(unknown_index)
        features['away_encoded'] = float(unknown_index)
    except Exception as e:
        logger.error(f"Unexpected error during team NAME encoding: {e}. Using fallback index {unknown_index}.", exc_info=True)
        features['home_encoded'] = float(unknown_index)
        features['away_encoded'] = float(unknown_index)

    # 8. Assemble and Validate DataFrame
    logger.debug("Assembling final feature DataFrame.")
    try:
        final_feature_data = {}
        for key in PODOS_EXPECTED_FEATURES:
            if key in features:
                final_feature_data[key] = features[key]
            else:
                logger.warning(f"Feature '{key}' was missing from the features dict for fixture {fixture_id}. Defaulting to 0.0.")
                final_feature_data[key] = 0.0

        df_features = pd.DataFrame([final_feature_data], columns=PODOS_EXPECTED_FEATURES)

        nan_found = False
        for col in PODOS_EXPECTED_FEATURES:
            original_dtype = df_features[col].dtype
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
            if df_features[col].isnull().any():
                logger.warning(f"Coerced NaN found in column '{col}' (original dtype: {original_dtype}) after to_numeric. Imputing with 0.0.")
                nan_found = True
        if nan_found:
            df_features = df_features.fillna(0.0)

        expected_shape = (1, len(PODOS_EXPECTED_FEATURES))
        if df_features.shape == expected_shape:
            logger.info(f"Successfully generated feature DataFrame for fixture {fixture_id}. Shape: {df_features.shape}")
            return df_features
        else:
            logger.error(f"Generated DataFrame for {fixture_id} has unexpected shape: {df_features.shape}. Expected: {expected_shape}.")
            return None
    except Exception as e:
        logger.error(f"Error assembling final DataFrame for {fixture_id}: {e}", exc_info=True)
        return None


def get_stat_value(stats_list: Optional[List[Dict[str, Any]]], stat_name: str) -> Optional[float]:
    """
    Helper function to find a specific statistic value from a list of stat dictionaries.
    Handles cases where the value might be None or non-numeric.
    """
    if not isinstance(stats_list, list):
        return None
    for stat_item in stats_list:
        if isinstance(stat_item, dict) and stat_item.get('type') == stat_name:
            value = stat_item.get('value')
            if value is None:
                return 0.0 # Treat None as 0
            try:
                # Attempt conversion to float, handle potential strings like '50%'
                if isinstance(value, str) and '%' in value:
                     return float(value.replace('%', '')) / 100.0
                return float(value)
            except (ValueError, TypeError):
                 logger.warning(f"Could not convert stat '{stat_name}' value '{value}' to float. Returning None.")
                 return None # Indicate conversion failure
    return None # Stat name not found in the list


def calculate_historical_stats(
    team_id: str, # Expect string ID now
    target_date_utc: datetime,
    matches_coll: Any,
    num_historical: int
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Fetches historical matches for a team before a target date
    and calculates average stats (Shots, SOT, Corners, Offsides, Cards).
    Returns dict of averages and the list of historical matches found.
    Handles potential missing statistics within historical documents.
    """
    query = {
        # Ensure IDs are queried correctly (might be int or str in DB)
        "$or": [{"home_team_id": int(team_id) if team_id.isdigit() else team_id},
                {"away_team_id": int(team_id) if team_id.isdigit() else team_id}],
        "date_utc": {"$lt": target_date_utc},
        "status_short": {"$in": ["FT", "AET", "PEN"]} # Only completed matches
    }
    projection = { # Project only essential fields
        "_id": 1, "date_utc": 1, "home_team_id": 1, "away_team_id": 1,
        "home_goals": 1, "away_goals": 1, "statistics_full": 1, "score_fulltime": 1
    }

    logger.debug(f"Querying historical matches for team ID '{team_id}' before {target_date_utc.date()}...")
    try:
        historical_matches = list(
            matches_coll.find(query, projection).sort("date_utc", DESCENDING).limit(num_historical)
        )
    except Exception as e:
         logger.error(f"Error querying historical matches for team {team_id}: {e}")
         return {}, [] # Return empty results on query failure

    logger.debug(f"Found {len(historical_matches)} historical matches for team ID '{team_id}'.")

    # Define stats to calculate: Key for output dict -> Key in statistics_full list
    stat_mapping = {
        "S": "Total Shots", "ST": "Shots on Goal", "C": "Corner Kicks",
        "O": "Offsides", "Y": "Yellow Cards", "R": "Red Cards"
    }
    stats_totals = {key: 0.0 for key in stat_mapping}
    stats_counts = {key: 0 for key in stat_mapping}

    if not historical_matches:
        logger.warning(f"No historical matches found for team {team_id} before {target_date_utc.date()}. Stats averages will be 0.")
        return stats_totals, [] # Return zeros and empty list

    for i, match in enumerate(historical_matches):
        match_id = match.get("_id", f"hist_{i}")
        is_home = str(safe_get(match, ['home_team_id'])) == team_id
        stats_full = safe_get(match, ['statistics_full'])

        team_stats_list = None
        if isinstance(stats_full, list) and len(stats_full) == 2:
            # Find the stats block corresponding to the team_id, comparing IDs as strings
            home_stats_id = str(safe_get(stats_full[0], ['team', 'id']))
            away_stats_id = str(safe_get(stats_full[1], ['team', 'id']))

            if is_home and home_stats_id == team_id:
                 team_stats_list = safe_get(stats_full[0], ['statistics'])
            elif not is_home and away_stats_id == team_id:
                 team_stats_list = safe_get(stats_full[1], ['statistics'])
            # Check reverse order just in case API was inconsistent
            elif is_home and away_stats_id == team_id: # Should not happen if is_home is correct
                 logger.warning(f"ID mismatch in hist match {match_id}: Team {team_id} is home but found ID in away stats.")
                 team_stats_list = safe_get(stats_full[1], ['statistics'])
            elif not is_home and home_stats_id == team_id: # Should not happen
                 logger.warning(f"ID mismatch in hist match {match_id}: Team {team_id} is away but found ID in home stats.")
                 team_stats_list = safe_get(stats_full[0], ['statistics'])

        if not team_stats_list:
            logger.warning(f"Could not find statistics for team {team_id} in historical match {match_id}. Skipping stats for this match.")
            continue # Skip this match if stats are missing/malformed

        # Extract values for each mapped stat
        for short_key, long_name in stat_mapping.items():
            value = get_stat_value(team_stats_list, long_name)
            if value is not None:
                stats_totals[short_key] += value
                stats_counts[short_key] += 1
            # else: logger.debug(f"Stat '{long_name}' missing/invalid in match {match_id} for team {team_id}")

    # Calculate averages, handle division by zero
    avg_stats = {
        key: (stats_totals[key] / stats_counts[key]) if stats_counts[key] > 0 else 0.0
        for key in stat_mapping
    }

    # Log the calculated averages
    for key, avg_val in avg_stats.items():
         logger.debug(f"Team {team_id} Avg {stat_mapping[key]} ({key}): {avg_val:.3f} (Total: {stats_totals[key]:.1f}, Count: {stats_counts[key]})")

    return avg_stats, historical_matches


def calculate_streaks_and_form(
    team_id: str, # Expect string ID
    historical_matches: List[Dict],
    num_matches_form: int
) -> Tuple[int, int, float]:
    """
    Calculates win streak, loss streak, and form PPG based on a list of
    historical matches (assumed sorted newest to oldest).
    """
    win_streak = 0
    loss_streak = 0
    form_points = 0
    form_matches_counted = 0

    if not historical_matches: # Handle case with no history
        return 0, 0, 0.0

    # Streaks: Iterate from most recent game backwards
    streak_broken = False
    current_streak_type = None # 'W' or 'L'

    for i, match in enumerate(historical_matches): # Sorted newest to oldest
        is_home = str(safe_get(match, ['home_team_id'])) == team_id

        # Robust score extraction
        home_goals = safe_get(match, ['home_goals'])
        away_goals = safe_get(match, ['away_goals'])
        if home_goals is None or away_goals is None:
            score_ft = safe_get(match, ['score_fulltime'])
            if isinstance(score_ft, dict):
                 home_goals = safe_get(score_ft, ['home'])
                 away_goals = safe_get(score_ft, ['away'])

        if home_goals is None or away_goals is None:
            logger.warning(f"Could not determine score for hist match {match.get('_id')} for team {team_id}. Skipping for form/streak.")
            continue

        try: # Ensure goals are numeric
             home_goals = int(home_goals); away_goals = int(away_goals)
        except (ValueError, TypeError):
             logger.warning(f"Non-integer score ({home_goals}/{away_goals}) in hist match {match.get('_id')}. Skipping.")
             continue

        # Determine result and points for the team_id
        if is_home:
            if home_goals > away_goals: result, team_points = 'W', 3
            elif home_goals == away_goals: result, team_points = 'D', 1
            else: result, team_points = 'L', 0
        else: # Team played away
            if away_goals > home_goals: result, team_points = 'W', 3
            elif home_goals == away_goals: result, team_points = 'D', 1
            else: result, team_points = 'L', 0

        # --- Calculate Form PPG (using first num_matches_form games) ---
        if i < num_matches_form:
            form_points += team_points
            form_matches_counted += 1

        # --- Calculate Streaks (from most recent game until streak broken) ---
        if not streak_broken:
            if i == 0: # Most recent game
                if result == 'W': current_streak_type = 'W'; win_streak = 1
                elif result == 'L': current_streak_type = 'L'; loss_streak = 1
                else: streak_broken = True # Draw breaks streak from the start
            else: # Subsequent older games
                if result == 'W' and current_streak_type == 'W': win_streak += 1
                elif result == 'L' and current_streak_type == 'L': loss_streak += 1
                else: streak_broken = True # Different result or draw breaks streak

    # Final Form PPG calculation (handle division by zero)
    form_ppg = (form_points / form_matches_counted) if form_matches_counted > 0 else 0.0
    return win_streak, loss_streak, form_ppg


# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <fixture_id>")
        print(f"Example: python {os.path.basename(__file__)} 1035144")
        sys.exit(1)

    fixture_id_to_process = sys.argv[1]

    # --- Connect to MongoDB ---
    client = None # Initialize client to None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) # Add timeout
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        logger.info(f"Successfully connected to MongoDB server: {MONGO_URI}")
        db = client[DB_NAME]
        matches_collection = db[MATCHES_COLLECTION]
        odds_collection = db[ODDS_COLLECTION]
        logger.info(f"Using Database: '{DB_NAME}', Collections: '{matches_collection.name}', '{odds_collection.name}'")
    except Exception as e:
        logger.error(f"Error connecting to MongoDB or accessing DB/Collections: {e}", exc_info=True)
        if client: client.close()
        sys.exit(1)

    # --- Load Label Encoder ---
    label_encoder = load_label_encoder()
    if label_encoder is None:
        # This should ideally not happen due to dummy fallback, but check anyway
        logger.error("Failed to load or create a label encoder. Exiting.")
        if client: client.close()
        sys.exit(1)

    # --- Load Team ID Mapping (for direct script execution) ---
    mongodb_id_to_name_map = None
    try:
        # Add project root to path to find the mapping file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from get_data.api_football.db_ids.team_id_mappings import TEAM_ID_MAPPING
        mongodb_id_to_name_map = create_mongodb_id_to_name_map(TEAM_ID_MAPPING)
        if mongodb_id_to_name_map is None:
            raise ValueError("Failed to create mongodb_id to name map from TEAM_ID_MAPPING.")

    except ImportError as e:
        logger.error(f"CRITICAL: Could not import TEAM_ID_MAPPING from 'get_data.api_football.db_ids.team_id_mappings'. Ensure path is correct. Error: {e}")
        if client: client.close()
        sys.exit(1)
    except Exception as e:
        logger.error(f"CRITICAL: Error processing team ID mapping: {e}", exc_info=True)
        if client: client.close()
        sys.exit(1)

    # --- Generate Features ---
    podos_features_df = None
    try:
        podos_features_df = generate_podos_features(
            fixture_id_to_process,
            matches_collection,
            odds_collection,
            label_encoder,
            mongodb_id_to_name_map=mongodb_id_to_name_map, # Pass the loaded map
            num_historical=NUM_HISTORICAL_GAMES
        )
    except Exception as e:
         logger.error(f"An unexpected error occurred during feature generation: {e}", exc_info=True)
         if client: client.close()
         sys.exit(1)

    # --- Output Results ---
    if podos_features_df is not None:
        print("\n--- Generated Podos Features ---")
        print(f"Fixture ID: {fixture_id_to_process}")
        # Format output for better readability
        pd.set_option('display.float_format', '{:.4f}'.format)
        print("\n--- Generated Podos Features ---")
        print("• " + "\n• ".join(podos_features_df.to_string(index=False).split('\n')))
        print("\n--- Column Verification ---")
        print(f"• Expected Columns: {len(PODOS_EXPECTED_FEATURES)}")
        print(f"• Generated Columns: {len(podos_features_df.columns)}")
        all_cols_present = all(col in podos_features_df.columns for col in PODOS_EXPECTED_FEATURES)
        order_correct = list(podos_features_df.columns) == PODOS_EXPECTED_FEATURES
        print(f"• All Expected Columns Present: {all_cols_present}")
        print(f"• Column Order Correct: {order_correct}")

        if not all_cols_present or not order_correct:
            logger.error("Feature DataFrame column mismatch or order incorrect!")

        # Optional: Save to CSV
        # output_filename = f"podos_features_{fixture_id_to_process}.csv"
        # try:
        #     podos_features_df.to_csv(output_filename, index=False)
        #     logger.info(f"Features saved to {output_filename}")
        # except Exception as e:
        #     logger.error(f"Error saving features to CSV {output_filename}: {e}")

    else:
        logger.error(f"Failed to generate Podos features for fixture ID: {fixture_id_to_process}")
        if client: client.close()
        sys.exit(1) # Exit with error code if feature generation failed

    # --- Close DB Connection ---
    if client:
        client.close()
        logger.info("MongoDB connection closed.")

    logger.info("--- Script Finished Successfully ---")