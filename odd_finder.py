# value_bet_finder.py

import json
import os
import logging
import glob
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date
import sys # Added to modify path for import
# Ensure the db_mongo import path is correct relative to this script's location
try:
    from get_data.api_football.db_mongo import MongoDBManager, db_manager, logger
except ImportError:
    # If the first import fails, adjust sys.path based on the script's location
    script_dir_for_import = os.path.dirname(os.path.abspath(__file__))
    project_root_for_import = os.path.abspath(os.path.join(script_dir_for_import)) # Assume script is in root
    get_data_dir_for_import = os.path.join(project_root_for_import, 'get_data', 'api_football')
    if get_data_dir_for_import not in sys.path:
        sys.path.insert(0, project_root_for_import) # Add project root to path for imports like get_data.api_football...
    # Retry import
    from get_data.api_football.db_mongo import MongoDBManager, db_manager, logger

# --- Define project_root based on script location ---
# Assumes odd_finder.py is directly in the project root directory (e.g., AgenticFC888)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir # Project root is the directory containing odd_finder.py
# If odd_finder.py were in a subdirectory like 'scripts', you'd use:
# project_root = os.path.abspath(os.path.join(script_dir, '..'))

# --- Target the main processed_matches directory ---
PROCESSED_MATCHES_PARENT_DIR = os.path.join(project_root, "processed_matches")
BOOKMAKER_NAME = "Bet365" # Or specify which bookmaker's odds to use
# OUTPUT_DIR removed as we write back to original files

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Prevent adding duplicate handlers if script is re-run in interactive session
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# Optional: Propagate messages to the db_manager's logger handlers if desired
# logger.propagate = True

def get_fixture_id_from_filename(filename):
    """Extracts fixture ID from the JSON filename."""
    try:
        # Assumes filename format like '..._fixtureid.json'
        return filename.split('_')[-1].split('.')[0]
    except Exception:
        logger.error(f"Could not extract fixture ID from {filename}")
        return None

def load_processed_match_data(filepath):
    """Loads JSON data from a file, returning the raw data and date."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        date_str = data.get("match_info", {}).get("date")
        match_date_simple = None
        if date_str:
            try:
                date_obj = datetime.fromisoformat(date_str.replace('+00:00', 'Z'))
                match_date_simple = date_obj.strftime('%Y-%m-%d')
            except ValueError:
                try:
                    date_obj = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                    match_date_simple = date_obj.strftime('%Y-%m-%d')
                    logger.warning(f"Parsed only date part from '{date_str}' in {filepath}")
                except ValueError:
                    logger.error(f"Could not parse date '{date_str}' from {filepath}")
        else:
            logger.warning(f"Could not find match_info.date in {filepath}")
        return data, match_date_simple
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        return None, None
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {filepath}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error loading {filepath}: {e}")
        return None, None


def get_odds_from_db(fixture_id, match_date_simple, bookmaker_name): # Added match_date_simple
    """Fetches odds data using MongoDBManager for dynamic collection selection."""
    if not fixture_id:
        logger.error("Error: No fixture ID provided for DB lookup.")
        return None
    if not match_date_simple:
        logger.error(f"Error: No valid match date provided for fixture {fixture_id}. Cannot determine odds collection.")
        return None

    try:
        fixture_id_int = int(fixture_id)
    except ValueError:
        logger.error(f"Error: Invalid fixture ID format: {fixture_id}")
        return None

    odds_data = None
    try:
        # Use db_manager to get the specific odds document using its get_odds_data method
        # This method handles getting the correct monthly collection based on date
        odds_data = db_manager.get_odds_data(match_date_simple, str(fixture_id_int))

        if not odds_data:
            # db_manager.get_odds_data already logs errors, so just a confirmation here
            logger.debug(f"No odds data found via db_manager for fixture_id: {fixture_id_int} on date {match_date_simple}")
            return None

        # Option 1: If payload saved is the top-level response containing the nested structure
        bookmakers_top_list = odds_data.get("bookmakers", []) # Check top level first
        if isinstance(bookmakers_top_list, list):
             for bookmaker_entry in bookmakers_top_list:
                 if isinstance(bookmaker_entry, dict) and "bookmakers" in bookmaker_entry:
                      nested_bookmakers = bookmaker_entry.get("bookmakers", [])
                      if isinstance(nested_bookmakers, list):
                          for sub_bookmaker in nested_bookmakers:
                              if isinstance(sub_bookmaker, dict) and sub_bookmaker.get("name") == bookmaker_name:
                                   logger.debug(f"Found bookmaker '{bookmaker_name}' in nested structure for fixture {fixture_id_int}")
                                   return sub_bookmaker.get("bets", [])

        # Option 2: If the payload saved *is* the inner bookmaker object directly (less likely based on example)
        if isinstance(odds_data.get("bookmakers"), list): # Assuming 'bookmakers' is the key holding the list of bet markets
            for bookmaker_details in odds_data.get("bookmakers", []): # This assumes a different structure than example
                if isinstance(bookmaker_details, dict) and bookmaker_details.get("name") == bookmaker_name:
                     logger.debug(f"Found bookmaker '{bookmaker_name}' directly in odds_data for fixture {fixture_id_int}")
                     return bookmaker_details.get("bets", []) # Adapt key if necessary

        # Option 3: If the odds_payload is just the list of bookmakers [{id:8, name:"Bet365", bets:[...]}, ...]
        if isinstance(odds_data, list): # If the root document IS the list of bookmakers
             for bookmaker_entry in odds_data:
                  if isinstance(bookmaker_entry, dict) and bookmaker_entry.get("name") == bookmaker_name:
                      logger.debug(f"Found bookmaker '{bookmaker_name}' in root list for fixture {fixture_id_int}")
                      return bookmaker_entry.get("bets", [])


        logger.warning(f"Could not find odds structure for bookmaker '{bookmaker_name}' within the document for fixture_id: {fixture_id_int}.")
        return None

    except Exception as e:
        logger.error(f"Error fetching/parsing odds from MongoDB via db_manager for fixture_id {fixture_id_int}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None


def parse_probability_string(prob_string):
    """Converts prediction probability string (e.g., '76.7%') to a Decimal."""
    if not isinstance(prob_string, str):
         logger.warning(f"Invalid probability input type: {type(prob_string)}, value: {prob_string}")
         return None
    try:
        return Decimal(prob_string.strip('%')) / Decimal(100)
    except Exception:
        logger.error(f"Could not parse probability string: {prob_string}")
        return None

def calculate_implied_probability(odds_string):
    """Calculates implied probability from decimal odds string."""
    if not odds_string:
         # logger.warning("Received empty odds string for implied probability calculation.") # Reduce noise
         return None
    try:
        odds = Decimal(str(odds_string)) # Ensure it's a string first for Decimal
        if odds > 0:
            # Add a small epsilon to avoid division by zero if odds are extremely high (though unlikely for decimal)
            return Decimal(1) / (odds + Decimal('0.00000001'))
        else:
            logger.warning(f"Received non-positive odds: {odds_string}")
            return None
    except Exception as e:
         logger.error(f"Could not parse odds string '{odds_string}': {e}")
         return None

def find_matching_odds(prediction_bet, prediction_type, odds_list):
    """
    Finds the matching odds for a given prediction. Includes enhanced logging.
    """
    logger.debug(f"Attempting to find odds for prediction: '{prediction_bet}' (Type: {prediction_type})") # Log input

    if not odds_list: # Added check
        logger.warning("Odds list is empty, cannot find match.")
        return None

    market_map = {
        # Simple Outcomes
        "Over 0.5 Goals": {"market_name": "Goals Over/Under", "value_prefix": "Over ", "value_suffix": ""},
        "Over 1.5 Goals": {"market_name": "Goals Over/Under", "value_prefix": "Over ", "value_suffix": ""},
        "Over 2.5 Goals": {"market_name": "Goals Over/Under", "value_prefix": "Over ", "value_suffix": ""},
        "Over 3.5 Goals": {"market_name": "Goals Over/Under", "value_prefix": "Over ", "value_suffix": ""},
        "Over 4.5 Goals": {"market_name": "Goals Over/Under", "value_prefix": "Over ", "value_suffix": ""},
        "Under 0.5 Goals": {"market_name": "Goals Over/Under", "value_prefix": "Under ", "value_suffix": ""},
        "Under 1.5 Goals": {"market_name": "Goals Over/Under", "value_prefix": "Under ", "value_suffix": ""},
        "Under 2.5 Goals": {"market_name": "Goals Over/Under", "value_prefix": "Under ", "value_suffix": ""},
        "Under 3.5 Goals": {"market_name": "Goals Over/Under", "value_prefix": "Under ", "value_suffix": ""},
        "Under 4.5 Goals": {"market_name": "Goals Over/Under", "value_prefix": "Under ", "value_suffix": ""},
        "BTTS Yes": {"market_name": "Both Teams Score", "value_prefix": "Yes", "value_suffix": ""},
        "BTTS No": {"market_name": "Both Teams Score", "value_prefix": "No", "value_suffix": ""},
        "Home Win": {"market_name": "Match Winner", "value_prefix": "Home", "value_suffix": ""},
        "Draw": {"market_name": "Match Winner", "value_prefix": "Draw", "value_suffix": ""},
        "Away Win": {"market_name": "Match Winner", "value_prefix": "Away", "value_suffix": ""},
        "Home or Draw": {"market_name": "Double Chance", "value_prefix": "Home/Draw", "value_suffix": ""},
        "Away or Draw": {"market_name": "Double Chance", "value_prefix": "Draw/Away", "value_suffix": ""},
        "No Draw (Home or Away Win)": {"market_name": "Double Chance", "value_prefix": "Home/Away", "value_suffix": ""},
    }

    simple_bet_part = prediction_bet.split(" & ")[0].split(" + ")[0]
    norm_prediction_bet = simple_bet_part.replace(" Goals", "")
    logger.debug(f"  Normalized prediction part: '{norm_prediction_bet}'")

    mapping = market_map.get(simple_bet_part)
    if not mapping:
         parts = norm_prediction_bet.split(' ')
         if len(parts) == 2 and parts[0] in ["Over", "Under"]:
             simplified_bet_key = f"{parts[0]} {parts[1]} Goals"
             logger.debug(f"  Trying simplified key: '{simplified_bet_key}'")
             mapping = market_map.get(simplified_bet_key)

    if not mapping:
        logger.debug(f"  Trying original prediction key: '{prediction_bet}'")
        mapping = market_map.get(prediction_bet)

    if not mapping:
        logger.warning(f"No mapping found for prediction: '{prediction_bet}' (tried '{simple_bet_part}')") # Keep as warning
        return None

    target_market_name = mapping["market_name"]
    target_value = mapping["value_prefix"] # Initial value from map
    logger.debug(f"  Mapping found: Market='{target_market_name}', Initial Value='{target_value}'")


    # --- Adjust target_value based on specific prediction details ---
    if target_market_name == "Goals Over/Under":
        parts = norm_prediction_bet.split(' ') # e.g., norm_prediction_bet = "Over 2.5"
        if len(parts) == 2:
             target_value = f"{parts[0]} {parts[1]}" # Reconstruct "Over 2.5"
             logger.debug(f"  Adjusted target value for O/U: '{target_value}'")

    # Explicit overrides based on the simple part (might refine target_value again)
    if simple_bet_part == "Home Win": target_value = "Home"
    if simple_bet_part == "Away Win": target_value = "Away"
    if simple_bet_part == "Draw": target_value = "Draw" # Added Draw explicit override for clarity
    if simple_bet_part == "BTTS Yes": target_value = "Yes"
    if simple_bet_part == "BTTS No": target_value = "No"
    if simple_bet_part == "Home or Draw": target_value = "Home/Draw" # Added DC overrides
    if simple_bet_part == "Away or Draw": target_value = "Draw/Away"
    if simple_bet_part == "No Draw (Home or Away Win)": target_value = "Home/Away"

    logger.debug(f"  Final target: Market='{target_market_name}', Value='{target_value}'")


    for market in odds_list:
        if not isinstance(market, dict): continue
        market_name_from_db = market.get("name")
        # logger.debug(f"    Checking DB market: '{market_name_from_db}'") # Can be noisy, enable if needed
        if market_name_from_db == target_market_name:
            logger.debug(f"    Found matching market in DB: '{target_market_name}'")
            for value_odd_pair in market.get("values", []):
                 if not isinstance(value_odd_pair, dict): continue
                 value_from_db = value_odd_pair.get("value")
                 # logger.debug(f"      Checking DB value: '{value_from_db}'") # Can be noisy
                 if value_from_db == target_value:
                    odd_found = value_odd_pair.get("odd")
                    logger.info(f"    SUCCESS: Found matching odd for '{target_market_name}' - '{target_value}': {odd_found}") # Changed level to INFO for success
                    return odd_found
            logger.warning(f"    Market '{target_market_name}' found, but value '{target_value}' not in its values.")
            return None # Value not found within the correct market

    logger.warning(f"  Market '{target_market_name}' not found in the provided odds list for this fixture.")
    return None

def get_context_stats(processed_data, bet_name):
    """Extracts relevant context stats based on the bet name."""
    stats = {}
    try:
        home_stats = processed_data.get('teams', {}).get('home', {}).get('statarea_analysis', {}).get('home', {}).get('last_15_games', {})
        away_stats = processed_data.get('teams', {}).get('away', {}).get('statarea_analysis', {}).get('away', {}).get('last_15_games', {})
        h2h_stats = processed_data.get('head_to_head', {}).get('summary', {})

        if "Over" in bet_name or "Under" in bet_name:
            stats['home_avg_scored_h15'] = home_stats.get('avg_goals_scored')
            stats['away_avg_scored_a15'] = away_stats.get('avg_goals_scored')
            stats['home_ovr25_pct_h15'] = home_stats.get('over_2_5_pct')
            stats['away_ovr25_pct_a15'] = away_stats.get('over_2_5_pct')
            stats['h2h_avg_goals'] = h2h_stats.get('avg_total_goals')
            stats['h2h_ovr25_pct'] = h2h_stats.get('over_2_5_pct')
        elif bet_name == "Home Win":
            stats['home_win_pct_h15'] = home_stats.get('outcome_probabilities_1x2', {}).get('win')
            stats['away_loss_pct_a15'] = away_stats.get('outcome_probabilities_1x2', {}).get('loss')
            stats['h2h_home_win_pct'] = h2h_stats.get('home_team_win_pct')
        elif bet_name == "Away Win":
            stats['away_win_pct_a15'] = away_stats.get('outcome_probabilities_1x2', {}).get('win')
            stats['home_loss_pct_h15'] = home_stats.get('outcome_probabilities_1x2', {}).get('loss')
            stats['h2h_away_win_pct'] = h2h_stats.get('away_team_win_pct')
        elif bet_name == "Draw":
            stats['home_draw_pct_h15'] = home_stats.get('outcome_probabilities_1x2', {}).get('draw')
            stats['away_draw_pct_a15'] = away_stats.get('outcome_probabilities_1x2', {}).get('draw')
            stats['h2h_draw_pct'] = h2h_stats.get('draw_pct')
        elif bet_name == "BTTS Yes":
            stats['home_btts_pct_h15'] = home_stats.get('btts_pct')
            stats['away_btts_pct_a15'] = away_stats.get('btts_pct')
            stats['h2h_btts_pct'] = h2h_stats.get('btts_pct')
        elif bet_name == "Home or Draw":
            stats['home_win_draw_pct_h15'] = home_stats.get('outcome_probabilities_1x2', {}).get('win', 0) + home_stats.get('outcome_probabilities_1x2', {}).get('draw', 0)
            stats['h2h_home_win_draw_pct'] = h2h_stats.get('home_team_win_pct', 0) + h2h_stats.get('draw_pct', 0)
        elif bet_name == "Away or Draw":
            stats['away_win_draw_pct_a15'] = away_stats.get('outcome_probabilities_1x2', {}).get('win', 0) + away_stats.get('outcome_probabilities_1x2', {}).get('draw', 0)
            stats['h2h_away_win_draw_pct'] = h2h_stats.get('away_team_win_pct', 0) + h2h_stats.get('draw_pct', 0)
        elif bet_name == "No Draw (Home or Away Win)":
             stats['home_win_pct_h15'] = home_stats.get('outcome_probabilities_1x2', {}).get('win')
             stats['away_win_pct_a15'] = away_stats.get('outcome_probabilities_1x2', {}).get('win')
             stats['h2h_no_draw_pct'] = h2h_stats.get('home_team_win_pct', 0) + h2h_stats.get('away_team_win_pct', 0)

        # Remove None values
        return {k: v for k, v in stats.items() if v is not None}
    except Exception as e:
        logger.error(f"Error getting context stats for bet '{bet_name}': {e}")
        return {}

def find_matched_bets(processed_data, odds_list):
    """
    Finds bets with predicted probability > 0.61, matches odds, adds context,
    calculates a predictability-weighted score, and returns the list.
    """
    matched_bets = []
    if not processed_data:
        logger.error("Cannot find matched bets: processed_data is None.")
        return []

    match_analysis = processed_data.get("match_analysis", {})
    predictions_dict = match_analysis.get("predictions", {})
    predictability_info = match_analysis.get("predictability", {})
    predictability_score_raw = predictability_info.get("score") # Might be None, float, int
    predictability_reason = predictability_info.get("reason")

    # --- Calculate Predictability Weight ---
    DEFAULT_PREDICTABILITY_WEIGHT = Decimal('0.75') # Assume 7.5/10 if missing
    predictability_weight = DEFAULT_PREDICTABILITY_WEIGHT
    if predictability_score_raw is not None:
        try:
            # Convert raw score (potentially float/int) to Decimal and normalize (0-10 -> 0-1)
            predictability_decimal = Decimal(str(predictability_score_raw))
            # Clamp between 0 and 10 before dividing
            clamped_score = max(Decimal('0.0'), min(predictability_decimal, Decimal('10.0')))
            predictability_weight = clamped_score / Decimal('10.0')
            logger.debug(f"Using predictability score {predictability_score_raw} -> weight {predictability_weight:.3f}")
        except Exception as e:
            logger.warning(f"Could not process predictability score '{predictability_score_raw}'. Using default weight. Error: {e}")
            predictability_weight = DEFAULT_PREDICTABILITY_WEIGHT
    else:
        logger.debug(f"Predictability score missing. Using default weight {predictability_weight}")
    # --- End Predictability Weight ---


    top_bets = predictions_dict.get("top_probable_bets")
    if top_bets is None:
        basic_probs = predictions_dict.get("basic_probabilities")
        if basic_probs:
            logger.debug("Using 'basic_probabilities' as 'top_probable_bets' was not found.")
            top_bets = [{"bet": key, "probability": f"{value*100:.1f}%", "type": "Simple"}
                        for key, value in basic_probs.items()]
            remap = {
                "home_win": "Home Win", "draw": "Draw", "away_win": "Away Win", "over_0.5": "Over 0.5 Goals",
                "over_1.5": "Over 1.5 Goals", "over_2.5": "Over 2.5 Goals", "over_3.5": "Over 3.5 Goals", "over_4.5": "Over 4.5 Goals",
                "under_0.5": "Under 0.5 Goals", "under_1.5": "Under 1.5 Goals", "under_2.5": "Under 2.5 Goals", "under_3.5": "Under 3.5 Goals", "under_4.5": "Under 4.5 Goals",
                "btts_yes": "BTTS Yes", "btts_no": "BTTS No", "home_draw": "Home or Draw", "away_draw": "Away or Draw", "home_away": "No Draw (Home or Away Win)"
            }
            for bet_dict in top_bets:
                bet_dict["bet"] = remap.get(bet_dict["bet"], bet_dict["bet"])
        else:
            logger.warning(f"No prediction source found in fixture {processed_data.get('match_info',{}).get('id','N/A')}")
            return []

    if not odds_list:
        logger.warning(f"Odds list is empty for fixture {processed_data.get('match_info',{}).get('id','N/A')}. Cannot find matches.")
        return []
    if not top_bets:
        logger.warning(f"No suitable predictions found in fixture {processed_data.get('match_info',{}).get('id','N/A')}")
        return []

    quantize_final_score = Decimal('0.0001') # For final score rounding

    for prediction in top_bets:
        if not isinstance(prediction, dict): continue
        bet_name = prediction.get("bet")
        bet_type = prediction.get("type")
        prob_str = prediction.get("probability")

        if not bet_name or not prob_str:
            continue

        if bet_type is not None and bet_type != "Simple":
             logger.debug(f"Skipping non-'Simple' bet type: '{bet_type}' for bet '{bet_name}'")
             continue

        predicted_prob = parse_probability_string(prob_str)
        if predicted_prob is None:
            logger.warning(f"Could not parse prediction probability '{prob_str}' for bet '{bet_name}'. Skipping.")
            continue

        # --- Probability Filter ---
        probability_threshold = Decimal('0.61')
        if predicted_prob <= probability_threshold:
            logger.debug(f"Skipping bet '{bet_name}' because PredProb {predicted_prob:.3f} is not > {probability_threshold}")
            continue

        # --- Odds Matching ---
        odds_str = find_matching_odds(bet_name, "Simple", odds_list)
        if odds_str is None:
             continue

        implied_prob = calculate_implied_probability(odds_str)
        if implied_prob is None:
             logger.warning(f"Could not calculate implied probability from odds '{odds_str}' for bet '{bet_name}'. Skipping.")
             continue

        # --- Calculations and Context ---
        try:
            if predicted_prob <= 0 or implied_prob <= 0:
                 continue

            value_ratio = predicted_prob / implied_prob
            edge = predicted_prob - implied_prob
            quantize_edge = Decimal('0.001')
            odds_decimal = Decimal(str(odds_str)).quantize(quantize_edge, ROUND_HALF_UP)

            # --- Calculate Weighted Score ---
            base_score_component = predicted_prob + edge
            score = (base_score_component * predictability_weight).quantize(quantize_final_score, ROUND_HALF_UP)
            # --- End Weighted Score ---


            # Get Context Stats
            context_stats = get_context_stats(processed_data, bet_name)

            logger.debug(f"  Bet Check (>0.61): '{bet_name}' | Weighted Score: {score} (Base: {base_score_component:.3f}, Weight: {predictability_weight:.3f}) | Pred Prob: {predicted_prob:.3f} | Odds: {odds_decimal} | Impl Prob: {implied_prob:.3f} | Edge: {edge:.3f}")

            match_data = {
                "bet": bet_name,
                "score": score, # This is now the weighted score
                "predicted_prob": predicted_prob,
                "odds": odds_decimal,
                "implied_prob": implied_prob.quantize(quantize_edge, ROUND_HALF_UP),
                "edge": edge,
                "value_ratio": value_ratio.quantize(quantize_edge, ROUND_HALF_UP),
                "context_stats": context_stats,
                "match_predictability_score": predictability_score_raw, # Store the original score
                "match_predictability_weight": predictability_weight, # Store the calculated weight
                "match_predictability_reason": predictability_reason
            }
            match_data = {k: v for k, v in match_data.items() if v is not None}


            logger.info(f"  MATCH FOUND (>0.61 Pred): Bet='{bet_name}', Score={score:.4f}, PredProb={predicted_prob:.1%}, Odds={odds_decimal}, Edge={edge:.1%}")
            matched_bets.append(match_data)

        except Exception as e:
             logger.error(f"Error during context/score calculation for bet '{bet_name}': {e}")
             import traceback
             traceback.print_exc() # More detail on calculation errors
             continue

    return matched_bets

def convert_decimals_to_strings(data):
    """Recursively convert Decimal objects and format numbers for JSON."""
    if isinstance(data, list):
        return [convert_decimals_to_strings(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_decimals_to_strings(v) for k, v in data.items()}
    elif isinstance(data, Decimal):
         # Consistent formatting for Decimals
         return f"{data:.4f}" # Use 4 decimal places for all Decimals
    elif isinstance(data, float):
         # Format floats (like original predictability score or context stats)
         # Check if it can be represented as int with .0
         if data == int(data):
              return f"{data:.1f}" # e.g., 9.0
         else:
              return f"{data:.3f}" # e.g., 9.200 or 0.733
    elif isinstance(data, (int, str, bool)) or data is None:
         return data # Keep ints, strings, bools, None as is
    else:
        # Fallback for other types
        return str(data)


# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    import argparse
    parser = argparse.ArgumentParser(description="Finds bets with predicted probability > 0.61, matches odds, adds context/score, and appends to original JSON.")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('get_data.api_football.db_mongo').setLevel(logging.DEBUG)
        logger.info("--- Debug logging enabled ---")

    # --- Initial Setup ---
    logger.info(f"Using MongoDB instance provided by db_manager.")
    logger.info(f"Using Bookmaker: {BOOKMAKER_NAME}")
    logger.info(f"Scanning for matches in: {PROCESSED_MATCHES_PARENT_DIR}")

    if not os.path.isdir(PROCESSED_MATCHES_PARENT_DIR):
         logger.error(f"Processed matches parent directory not found: {PROCESSED_MATCHES_PARENT_DIR}")
         # Attempt to use current working directory's processed_matches as fallback
         fallback_dir = os.path.join(os.getcwd(), "processed_matches")
         logger.warning(f"Attempting fallback directory: {fallback_dir}")
         if os.path.isdir(fallback_dir):
              PROCESSED_MATCHES_PARENT_DIR = fallback_dir
              logger.info(f"Using fallback directory: {PROCESSED_MATCHES_PARENT_DIR}")
         else:
              logger.error("Fallback directory also not found. Exiting.")
              sys.exit(1) # Use sys.exit

    # --- File Processing Loop ---
    search_pattern = os.path.join(PROCESSED_MATCHES_PARENT_DIR, '**', '*.json')
    json_files = glob.glob(search_pattern, recursive=True)
    json_files = [f for f in json_files if '_vs_' in os.path.basename(f)]

    if not json_files:
        logger.warning(f"No processed match JSON files found recursively in {PROCESSED_MATCHES_PARENT_DIR}")

    processed_count = 0
    files_updated_count = 0
    total_matched_bets_count = 0
    error_count = 0

    for json_file_path in json_files:
        relative_path = os.path.relpath(json_file_path, PROCESSED_MATCHES_PARENT_DIR)
        logger.debug(f"\n--- Processing: {relative_path} ---")
        processed_count += 1
        fixture_id = None
        try:
            fixture_id = get_fixture_id_from_filename(os.path.basename(json_file_path))
            if not fixture_id:
                logger.warning(f"Skipping file (no fixture ID): {relative_path}")
                continue

            # Load data *once*
            processed_data, match_date_simple = load_processed_match_data(json_file_path)
            if not processed_data:
                logger.error(f"Failed to load processed data from {json_file_path}. Skipping.")
                continue
            if not match_date_simple:
                 logger.warning(f"Skipping fixture {fixture_id} due to missing/invalid date in {json_file_path}")
                 continue

            odds_list = get_odds_from_db(fixture_id, match_date_simple, BOOKMAKER_NAME)
            if not odds_list:
                logger.debug(f"Could not retrieve odds for fixture {fixture_id} on {match_date_simple}. Skipping analysis for this file.")
                continue

            # Find bets, including context and score
            matched_bets_for_file = find_matched_bets(processed_data, odds_list)

            if matched_bets_for_file:
                # Sort by new weighted score, then probability
                matched_bets_for_file.sort(key=lambda x: (x['score'], x['predicted_prob']), reverse=True)
                total_matched_bets_count += len(matched_bets_for_file)

                # Convert all Decimals (and format floats) for JSON compatibility
                serializable_matches = convert_decimals_to_strings(matched_bets_for_file)

                # Add/Update the sorted, serializable list in the loaded data
                processed_data["matched_odds_info"] = serializable_matches # Use the same key

                # Write the updated data back
                try:
                    with open(json_file_path, 'w') as f:
                        json.dump(processed_data, f, indent=4)
                    logger.info(f"Successfully updated {len(matched_bets_for_file)} matched bets in: {relative_path}")
                    files_updated_count += 1
                except Exception as write_error:
                    logger.error(f"Error writing updated data back to {relative_path}: {write_error}")
                    error_count += 1
            else:
                # If no bets met threshold, potentially remove old key if it exists
                if "matched_odds_info" in processed_data:
                    logger.debug(f"Removing previous 'matched_odds_info' as no bets met threshold for fixture {fixture_id} in {relative_path}")
                    del processed_data["matched_odds_info"]
                    try:
                        with open(json_file_path, 'w') as f:
                           json.dump(processed_data, f, indent=4)
                        # Optionally log this removal, maybe at DEBUG level
                    except Exception as write_error:
                       logger.error(f"Error writing updated data (removing key) back to {relative_path}: {write_error}")
                       error_count += 1
                else:
                    logger.debug(f"No bets met >0.61 threshold for fixture {fixture_id} in {relative_path}")


        except Exception as e:
             logger.error(f"Unhandled error processing file {relative_path} (Fixture: {fixture_id}): {e}")
             import traceback
             traceback.print_exc()
             error_count += 1

    # --- Final Summary ---
    logger.info("\n" + "="*50)
    logger.info("--- Processing Summary ---")
    logger.info(f"Total files scanned: {processed_count}")
    logger.info(f"Files successfully updated with matched odds info: {files_updated_count}")
    logger.info(f"Total matched bets found (>0.61 Prob) across all files: {total_matched_bets_count}")
    logger.info(f"Files skipped or failed during processing: {processed_count - files_updated_count}") # Includes skips and errors
    logger.info(f"Explicit error count during processing: {error_count}")
    logger.info("="*50)


    # --- Close DB Connection ---
    try:
        db_manager.close_connection()
        logger.info("\nMongoDB connection closed via db_manager.")
    except Exception as e:
         logger.error(f"Error closing MongoDB connection: {e}")

    logger.info("Script finished.")