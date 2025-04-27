# value_bet_finder.py

import json
import os
import logging
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date
import sys

# Ensure the db_mongo import path is correct relative to this script's location
script_dir_for_import = os.path.dirname(os.path.abspath(__file__))
project_root_for_import = os.path.abspath(os.path.join(script_dir_for_import, '..'))
get_data_dir_for_import = os.path.join(project_root_for_import, 'get_data', 'api_football')

# Add project root to sys.path if not already present
if project_root_for_import not in sys.path:
    sys.path.insert(0, project_root_for_import)

try:
    from get_data.api_football.db_mongo import db_manager, logger
except ImportError as e:
    print(f"Error importing db_manager: {e}")

# --- Define project_root based on script location ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Correct project_root: Go up one level from the script's directory (score_data)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

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

# --- Target the specific input/output file ---
INPUT_OUTPUT_FILE = os.path.join(project_root, "data", "output", "batch_prediction_results.json")
BOOKMAKER_NAME = "Bet365" # Or specify which bookmaker's odds to use

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
    # Date is needed for logging/potential internal logic in db_manager, but not passed directly based on TypeError
    if not match_date_simple:
        logger.error(f"Error: No valid match date provided for fixture {fixture_id}. Cannot determine odds collection context.")
        # We might still attempt the lookup if the manager doesn't strictly need the date passed here
        # return None # Removed this early exit, let the db_manager handle it

    try:
        fixture_id_int = int(fixture_id)
    except ValueError:
        logger.error(f"Error: Invalid fixture ID format: {fixture_id}")
        return None

    odds_data = None
    try:
        # Use db_manager to get the specific odds document using its get_odds_data method
        # This method handles getting the correct monthly collection based on date (comment implies internal handling)
        # --- MODIFIED CALL: Pass only fixture_id based on TypeError ---
        logger.debug(f"Calling db_manager.get_odds_data with fixture_id: {fixture_id_int} (Date context: {match_date_simple})")
        odds_data = db_manager.get_odds_data(str(fixture_id_int))
        # --- END MODIFICATION ---

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
    Finds the matching odds for a given prediction (simple or combined).
    Includes enhanced logging.
    Handles simple types and combined selections from 'top_n_combined_selections'.
    Recognizes H/D/A abbreviations for Match Winner.
    """
    logger.debug(f"Attempting to find odds for prediction: '{prediction_bet}' (Type: {prediction_type})") # Log input

    if not odds_list: # Added check
        logger.warning("Odds list is empty, cannot find match.")
        return None

    # --- Define Market Mappings ---
    # Simple markets (as before)
    market_map_simple = {
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
        # Abbreviations
        "O0.5": {"market_name": "Goals Over/Under", "value_prefix": "Over 0.5", "value_suffix": ""},
        "O1.5": {"market_name": "Goals Over/Under", "value_prefix": "Over 1.5", "value_suffix": ""},
        "O2.5": {"market_name": "Goals Over/Under", "value_prefix": "Over 2.5", "value_suffix": ""},
        "O3.5": {"market_name": "Goals Over/Under", "value_prefix": "Over 3.5", "value_suffix": ""},
        "O4.5": {"market_name": "Goals Over/Under", "value_prefix": "Over 4.5", "value_suffix": ""},
        "U0.5": {"market_name": "Goals Over/Under", "value_prefix": "Under 0.5", "value_suffix": ""},
        "U1.5": {"market_name": "Goals Over/Under", "value_prefix": "Under 1.5", "value_suffix": ""},
        "U2.5": {"market_name": "Goals Over/Under", "value_prefix": "Under 2.5", "value_suffix": ""},
        "U3.5": {"market_name": "Goals Over/Under", "value_prefix": "Under 3.5", "value_suffix": ""},
        "U4.5": {"market_name": "Goals Over/Under", "value_prefix": "Under 4.5", "value_suffix": ""},
        "1": {"market_name": "Match Winner", "value_prefix": "Home", "value_suffix": ""},
        "X": {"market_name": "Match Winner", "value_prefix": "Draw", "value_suffix": ""},
        "2": {"market_name": "Match Winner", "value_prefix": "Away", "value_suffix": ""},
        "1X": {"market_name": "Double Chance", "value_prefix": "Home/Draw", "value_suffix": ""},
        "X2": {"market_name": "Double Chance", "value_prefix": "Draw/Away", "value_suffix": ""},
        "12": {"market_name": "Double Chance", "value_prefix": "Home/Away", "value_suffix": ""},
        # Added H/D/A mappings
        "H": {"market_name": "Match Winner", "value_prefix": "Home", "value_suffix": ""},
        "D": {"market_name": "Match Winner", "value_prefix": "Draw", "value_suffix": ""},
        "A": {"market_name": "Match Winner", "value_prefix": "Away", "value_suffix": ""},
    }

    # Combined markets (Add more as needed based on Bet365 actual names)
    # These are guesses - **VERIFY AGAINST ACTUAL DB DATA**
    market_map_combined = {
        "Match Result and Both Teams To Score": "Results/Both Teams Score", # Updated to match DB example
        "Double Chance and Total Goals": "Double Chance / Total Goals", # Example Bet365 name
        "Match Result and Total Goals": "Result/Total Goals", # Updated to match DB example
        "Both Teams To Score and Total Goals": "Total Goals/Both Teams To Score" # Updated to match DB example
        # --- Potential Alternative Names (Add if needed based on DB data) ---
        # "Match Result and Total Goals": "Result / Total Goals",
        # "Both Teams To Score and Total Goals": "BTTS / Total Goals",
    }

    # --- Parsing Logic ---
    target_market_name = None
    target_value = None
    is_combined = " and " in prediction_bet

    if not is_combined:
        # Handle Simple Bets (including normalized O/U, 1X2, HDA)
        simple_bet_part = prediction_bet.split(" + ")[0] # Handle potential future combo markers
        mapping = market_map_simple.get(simple_bet_part)

        if not mapping:
            # Try normalizing "Over X.Y Goals" format if primary lookup fails
            norm_prediction_bet = simple_bet_part.replace(" Goals", "")
            parts = norm_prediction_bet.split(' ')
            if len(parts) == 2 and parts[0] in ["Over", "Under"]:
                simplified_bet_key = f"{parts[0]} {parts[1]} Goals"
                logger.debug(f"  Trying simplified key: '{simplified_bet_key}'")
                mapping = market_map_simple.get(simplified_bet_key)

        if not mapping:
            # Check if it's a simple bet that might appear in combined selections (like 'U4.5')
            # This is needed because process_combined_selections calls this function too.
            logger.debug(f"No simple mapping found for prediction: '{prediction_bet}', checking common simple types...")
            if simple_bet_part.startswith('O') or simple_bet_part.startswith('U'):
                 ou_market_map = { # Simplified map just for O/U lookup here
                     "O0.5": ("Goals Over/Under", "Over 0.5"), "O1.5": ("Goals Over/Under", "Over 1.5"),
                     "O2.5": ("Goals Over/Under", "Over 2.5"), "O3.5": ("Goals Over/Under", "Over 3.5"),
                     "O4.5": ("Goals Over/Under", "Over 4.5"),
                     "U0.5": ("Goals Over/Under", "Under 0.5"), "U1.5": ("Goals Over/Under", "Under 1.5"),
                     "U2.5": ("Goals Over/Under", "Under 2.5"), "U3.5": ("Goals Over/Under", "Under 3.5"),
                     "U4.5": ("Goals Over/Under", "Under 4.5"),
                 }
                 market_val = ou_market_map.get(simple_bet_part)
                 if market_val:
                     target_market_name, target_value = market_val
                     logger.debug(f"  Mapped simple bet '{simple_bet_part}' to Market='{target_market_name}', Value='{target_value}'")
                 else:
                     logger.warning(f"Could not map simple O/U prediction: '{prediction_bet}'")
                     return None
            elif simple_bet_part in ["1","X","2","1X","X2","12", "H", "D", "A"]: # Added H/D/A here
                 market_val_map = {
                     "1": ("Match Winner", "Home"), "X": ("Match Winner", "Draw"), "2": ("Match Winner", "Away"),
                     "1X": ("Double Chance", "Home/Draw"), "X2": ("Double Chance", "Draw/Away"), "12": ("Double Chance", "Home/Away"),
                     "H": ("Match Winner", "Home"), "D": ("Match Winner", "Draw"), "A": ("Match Winner", "Away"), # Added H/D/A
                 }
                 market_val = market_val_map.get(simple_bet_part)
                 if market_val:
                      target_market_name, target_value = market_val
                      logger.debug(f"  Mapped simple bet '{simple_bet_part}' to Market='{target_market_name}', Value='{target_value}'")
                 else:
                      # This case should ideally not be reached if the map is correct
                      logger.warning(f"Could not map simple 1X2/DC/HDA prediction: '{prediction_bet}'")
                      return None
            elif simple_bet_part in ["BTTS Yes", "BTTS No"]:
                 market_val_map = {
                     "BTTS Yes": ("Both Teams Score", "Yes"), "BTTS No": ("Both Teams Score", "No")
                 }
                 market_val = market_val_map.get(simple_bet_part)
                 if market_val:
                     target_market_name, target_value = market_val
                     logger.debug(f"  Mapped simple bet '{simple_bet_part}' to Market='{target_market_name}', Value='{target_value}'")
                 else:
                      logger.warning(f"Could not map simple BTTS prediction: '{prediction_bet}'")
                      return None
            else:
                 logger.warning(f"No simple mapping found for prediction: '{prediction_bet}'")
                 return None


        # If mapping was found via market_map_simple earlier
        if not target_market_name: # Only if not already set by the direct O/U, 1X2 etc. lookup above
             target_market_name = mapping["market_name"]
             target_value = mapping["value_prefix"] # Start with the prefix

             # Refine target_value based on market type
             if target_market_name == "Goals Over/Under":
                  # Ensure format like "Over 2.5" or "Under 1.5"
                  ou_parts = simple_bet_part.split(" ")
                  if len(ou_parts) >= 2: # e.g. O2.5 -> Over 2.5 or Over 2.5 Goals -> Over 2.5
                      # Rebuild from normalized key if needed e.g. O2.5
                      norm_key = ou_parts[0] if len(ou_parts[0]) > 1 else f"{ou_parts[0]}{ou_parts[1]}" # e.g. O2.5
                      if norm_key.startswith('O'):
                           target_value = f"Over {norm_key[1:]}"
                      elif norm_key.startswith('U'):
                           target_value = f"Under {norm_key[1:]}"
                      else: # Handle "Over 2.5" case
                          target_value = f"{ou_parts[0]} {ou_parts[1]}"


             elif target_market_name == "Match Winner":
                  # Covers "Home Win", "Draw", "Away Win", "1", "X", "2", "H", "D", "A" from map
                  if simple_bet_part in ["Home Win", "1", "H"]: target_value = "Home"
                  elif simple_bet_part in ["Away Win", "2", "A"]: target_value = "Away"
                  elif simple_bet_part in ["Draw", "X", "D"]: target_value = "Draw"
             elif target_market_name == "Both Teams Score":
                  if simple_bet_part == "BTTS Yes": target_value = "Yes"
                  elif simple_bet_part == "BTTS No": target_value = "No"
             elif target_market_name == "Double Chance":
                  if simple_bet_part in ["Home or Draw", "1X"]: target_value = "Home/Draw"
                  elif simple_bet_part in ["Away or Draw", "X2"]: target_value = "Draw/Away"
                  elif simple_bet_part in ["No Draw (Home or Away Win)", "12"]: target_value = "Home/Away"

        logger.debug(f"  Simple Mapping Result: Target Market='{target_market_name}', Target Value='{target_value}'")

    else:
        # Handle Combined Bets
        parts = [p.strip() for p in prediction_bet.split(" and ")]
        if len(parts) != 2:
            logger.warning(f"Cannot parse combined bet with != 2 parts: '{prediction_bet}'")
            return None

        part1, part2 = parts
        logger.debug(f"  Parsing combined bet: Part1='{part1}', Part2='{part2}'")

        # Define helper maps here to avoid scope issues if defined within cases
        # Updated result_map to include H/D/A
        result_map = {"1": "Home", "X": "Draw", "2": "Away", "H": "Home", "D": "Draw", "A": "Away", "Home Win": "Home", "Draw": "Draw", "Away Win": "Away"}
        btts_map = {"BTTS Yes": "Yes", "BTTS No": "No"}
        dc_map = {"1X": "Home/Draw", "X2": "Draw/Away", "12": "Home/Away"}
        ou_map = { # Maps normalized input like U3.5 to Bet365 value like Under 3.5
             "O0.5": "Over 0.5", "O1.5": "Over 1.5", "O2.5": "Over 2.5", "O3.5": "Over 3.5", "O4.5": "Over 4.5",
             "U0.5": "Under 0.5", "U1.5": "Under 1.5", "U2.5": "Under 2.5", "U3.5": "Under 3.5", "U4.5": "Under 4.5"
        }

        # --- Determine Combined Market and Value (Needs refinement based on actual data) ---

        # Case 1: Result & BTTS (e.g., "A and BTTS Yes", "H and BTTS No")
        if (part1 in result_map and part2 in btts_map) or (part2 in result_map and part1 in btts_map):
            # Use the DB market name directly based on example data
            target_market_name = "Results/Both Teams Score"
            res_part = result_map[part1] if part1 in result_map else result_map[part2]
            btts_part = btts_map[part1] if part1 in btts_map else btts_map[part2]
            target_value = f"{res_part}/{btts_part}" # Bet365 uses "Home/Yes" etc. ** VERIFY **
            logger.debug(f"  Combined Mapping (Result/BTTS): Market='{target_market_name}', Value='{target_value}'")

        # Case 2: Double Chance & O/U (e.g., "12 and U3.5", "X2 and O1.5")
        elif (part1 in dc_map and part2 in ou_map) or (part2 in dc_map and part1 in ou_map):
            target_market_name = market_map_combined.get("Double Chance and Total Goals") # Keep guessed name for now
            dc_part = dc_map[part1] if part1 in dc_map else dc_map[part2]
            ou_part_key = part1 if part1 in ou_map else part2
            ou_value_part = ou_map[ou_part_key] # e.g., "Under 3.5"
            # Bet365 format might be like "Home/Draw & Over 2.5" - ** VERIFY **
            target_value = f"{dc_part} / {ou_value_part}"
            logger.debug(f"  Combined Mapping (DC/O-U): Market='{target_market_name}', Value='{target_value}'")

        # Case 3: BTTS & O/U (e.g., "BTTS Yes and O2.5", "BTTS No and U3.5")
        elif (part1 in btts_map and part2 in ou_map) or (part2 in btts_map and part1 in ou_map):
            # Use the DB market name directly based on example data
            target_market_name = "Total Goals/Both Teams To Score"
            btts_part = btts_map[part1] if part1 in btts_map else btts_map[part2]
            ou_part_key = part1 if part1 in ou_map else part2
            ou_value_part = ou_map[ou_part_key] # e.g., "Over 2.5"
            # Bet365 uses "o/yes 2.5" format - ** VERIFY / ADJUST **
            # Constructing based on pattern: needs verification
            o_u_prefix = "o" if ou_part_key.startswith("O") else "u"
            btts_suffix = "yes" if btts_part == "Yes" else "no"
            ou_number = ou_part_key[1:] # e.g. "2.5"
            target_value = f"{o_u_prefix}/{btts_suffix} {ou_number}"
            logger.debug(f"  Combined Mapping (BTTS/O-U): Market='{target_market_name}', Value='{target_value}'")

        # Case 4: Result & O/U (e.g., "H and O2.5", "X and U1.5")
        elif (part1 in result_map and part2 in ou_map) or (part2 in result_map and part1 in ou_map):
            # Use the DB market name directly based on example data
            target_market_name = "Result/Total Goals"
            res_part = result_map[part1] if part1 in result_map else result_map[part2]
            ou_part_key = part1 if part1 in ou_map else part2
            ou_value_part = ou_map[ou_part_key] # e.g., "Under 1.5"
            # Bet365 uses "Home/Under 2.5" etc. ** VERIFY **
            target_value = f"{res_part}/{ou_value_part}"
            logger.debug(f"  Combined Mapping (Result/O-U): Market='{target_market_name}', Value='{target_value}'")


        if not target_market_name or not target_value:
            logger.warning(f"Could not determine combined market mapping for '{prediction_bet}'")
            # Attempt fallback: Check if the DB market name *is* the prediction string (sometimes happens)
            logger.debug(f"    Attempting direct market name match for '{prediction_bet}'")
            target_market_name = prediction_bet
            target_value = prediction_bet # Value is often same as market name for simple combined representations
            # Let the search loop below try this fallback

    # --- Search in Odds List ---
    if not target_market_name:
         logger.warning(f"Target market name could not be determined for '{prediction_bet}'")
         return None

    # --- Refined Search Logic ---
    found_odd = None
    attempted_direct_match = (target_market_name == prediction_bet and target_value == prediction_bet) # Flag if using fallback

    for market in odds_list:
        if not isinstance(market, dict): continue
        market_name_from_db = market.get("name")
        if not market_name_from_db: continue

        # Normalize names for comparison (lowercase, remove spaces, slashes)
        # Example: "Result/Total Goals" -> "resulttotalgoals"
        norm_target_market = ''.join(filter(str.isalnum, target_market_name.lower()))
        norm_db_market = ''.join(filter(str.isalnum, market_name_from_db.lower()))

        # Check if normalized market names match
        if norm_target_market == norm_db_market:
            logger.debug(f"    Found potentially matching market in DB: '{market_name_from_db}' (Target: '{target_market_name}')")
            for value_odd_pair in market.get("values", []):
                if not isinstance(value_odd_pair, dict): continue
                value_from_db = value_odd_pair.get("value")
                if not value_from_db: continue

                # Normalize values for comparison
                # Example: "Home/Over 2.5" -> "homeover25"
                # Example: "o/yes 2.5" -> "oyes25"
                norm_target_val = ''.join(filter(str.isalnum, target_value.lower().replace('.', '')))
                norm_db_val = ''.join(filter(str.isalnum, value_from_db.lower().replace('.', '')))

                # Special handling for "Total Goals/Both Teams To Score" values like "o/yes 2.5"
                if market_name_from_db == "Total Goals/Both Teams To Score":
                     # Target was constructed as "o/yes 2.5", DB is "o/yes 2.5" -> normalization works
                     pass # Normalization should handle this

                # Standard comparison
                if norm_db_val == norm_target_val:
                    odd_found = value_odd_pair.get("odd")
                    logger.info(f"    SUCCESS: Found matching odd for '{market_name_from_db}' - '{value_from_db}': {odd_found}")
                    found_odd = odd_found
                    break # Exit inner loop once value is found
            # If we found the odd, exit the outer market loop as well
            if found_odd is not None:
                break
            else:
                # If market name matched but value didn't, log warning
                logger.warning(f"    Market '{market_name_from_db}' matched target '{target_market_name}', but target value '{target_value}' (normalized: '{norm_target_val}') not found in its values.")
                # Continue searching other markets in case of duplicate market names? Unlikely but possible.
                # For now, assume market names are unique enough. If value not found here, it's likely not present.

    # If after checking all markets, we haven't found the odd
    if found_odd is None:
        if attempted_direct_match:
             logger.warning(f"  Target market '{target_market_name}' (fallback attempt) not found or value not matched in the provided odds list.")
        else:
             logger.warning(f"  Target market '{target_market_name}' not found or value '{target_value}' not matched in the provided odds list.")
        return None
    else:
         return found_odd # Return the successfully found odd

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

def process_combined_selections(processed_data, odds_list):
    """
    Processes the 'top_n_combined_selections', finds odds, calculates metrics,
    and updates the list in-place within processed_data.
    If a direct combined odd isn't found, attempts to calculate one by
    multiplying the odds of the individual components (less accurate approach).
    """
    if not processed_data:
        logger.error("Cannot process combined selections: processed_data is None.")
        return processed_data

    # Locate the 'top_n_combined_selections' list
    combined_selections = None
    match_analysis_data = processed_data.get("match_analysis")
    if isinstance(match_analysis_data, dict):
        combined_selections = match_analysis_data.get("top_n_combined_selections")
    if combined_selections is None: # Fallback to top-level
         combined_selections = processed_data.get("top_n_combined_selections")

    fixture_id_log = processed_data.get('fixture_id') or processed_data.get('match_info', {}).get('id', 'N/A')

    if not combined_selections or not isinstance(combined_selections, list):
        logger.debug(f"No 'top_n_combined_selections' list found or list is empty for fixture {fixture_id_log}.")
        return processed_data # Return unchanged (no modifications needed)

    if not odds_list:
        logger.warning(f"Odds list is empty for fixture {fixture_id_log}. Cannot add odds to combined selections.")
        # Clear existing target fields if present, ensuring only these are touched
        for selection_dict in combined_selections:
             if isinstance(selection_dict, dict):
                 selection_dict.pop("odd", None)
                 selection_dict.pop("implied_prob", None)
                 selection_dict.pop("edge", None)
                 selection_dict.pop("value_ratio", None)
                 selection_dict.pop("odd_source", None) # Remove calculation source flag
        return processed_data # Return unchanged (except potential clearing)

    logger.debug(f"Processing {len(combined_selections)} combined selections for odds in fixture {fixture_id_log}...")
    quantize_edge = Decimal('0.001')
    quantize_ratio = Decimal('0.001')
    quantize_odds = Decimal('0.01') # Standard for odds
    quantize_prob = Decimal('0.0001') # Precision for probabilities

    updated_count = 0
    for selection_dict in combined_selections: # Iterate through the list to modify dicts in place
        if not isinstance(selection_dict, dict): continue

        bet_name = selection_dict.get("selection")
        predicted_prob_raw = selection_dict.get("probability") # Should be Decimal from load

        # --- Clear previous calculation source ---
        selection_dict.pop("odd_source", None)

        if not bet_name or predicted_prob_raw is None:
            logger.warning(f"Skipping combined selection due to missing 'selection' or 'probability': {selection_dict}")
            continue

        # Ensure predicted_prob is Decimal
        predicted_prob = None
        try:
            # It should already be Decimal if loaded correctly, but handle str just in case
            if isinstance(predicted_prob_raw, str):
                if '%' in predicted_prob_raw:
                     predicted_prob = parse_probability_string(predicted_prob_raw)
                else:
                     predicted_prob = Decimal(predicted_prob_raw)
            elif isinstance(predicted_prob_raw, (Decimal, float, int)):
                 predicted_prob = Decimal(predicted_prob_raw)
            else:
                raise TypeError(f"Unexpected type for probability: {type(predicted_prob_raw)}")

            if predicted_prob is None: raise ValueError("Probability became None after conversion attempt")
            predicted_prob = predicted_prob.quantize(quantize_prob, ROUND_HALF_UP) # Ensure consistent precision

        except Exception as e:
            logger.error(f"Could not ensure predicted probability '{predicted_prob_raw}' is Decimal for selection '{bet_name}'. Skipping. Error: {e}")
            # Clear fields if conversion fails
            selection_dict.pop("odd", None); selection_dict.pop("implied_prob", None); selection_dict.pop("edge", None); selection_dict.pop("value_ratio", None); selection_dict.pop("odd_source", None)
            continue

        # --- Find Odds ---
        # Attempt 1: Find the direct combined odd
        odds_str = find_matching_odds(bet_name, "Combined", odds_list)
        odd_source = "Direct Match" if odds_str is not None else None

        # Attempt 2: If direct combined odd not found, try calculating from individual parts
        if odds_str is None and " and " in bet_name:
             logger.debug(f"Direct odd not found for '{bet_name}'. Attempting calculation from individual parts.")
             parts = [p.strip() for p in bet_name.split(" and ")]
             if len(parts) == 2:
                 part1, part2 = parts
                 odd_str1 = find_matching_odds(part1, "Simple", odds_list) # Treat parts as simple bets
                 odd_str2 = find_matching_odds(part2, "Simple", odds_list)

                 if odd_str1 is not None and odd_str2 is not None:
                     try:
                         odd1 = Decimal(str(odd_str1))
                         odd2 = Decimal(str(odd_str2))
                         calculated_odd = (odd1 * odd2) # Multiply odds
                         odds_str = str(calculated_odd) # Use the calculated odd string
                         odd_source = f"Calculated ({odd1:.2f} * {odd2:.2f})"
                         logger.info(f"  Calculated combined odd for '{bet_name}': {calculated_odd:.2f} (from {part1} @ {odd1:.2f} and {part2} @ {odd2:.2f})")
                     except Exception as calc_e:
                         logger.error(f"Error calculating combined odd for '{bet_name}' from parts '{part1}'({odd_str1}) and '{part2}'({odd_str2}): {calc_e}")
                         odds_str = None # Calculation failed
                         odd_source = "Calculation Error"
                 else:
                     logger.warning(f"Could not find individual odds for both parts of '{bet_name}' ({part1}: {odd_str1}, {part2}: {odd_str2}). Cannot calculate combined odd.")
                     odd_source = "Missing Individual Odds"
             else:
                 logger.warning(f"Cannot parse '{bet_name}' into two parts for individual calculation.")
                 odd_source = "Parsing Error"


        # --- Process if odd was found (either directly or calculated) ---
        if odds_str is None:
            logger.debug(f"No matching or calculable odds found for selection: '{bet_name}'. Status: {odd_source or 'Not Found'}")
            # Remove old values if they exist - only touch these keys
            selection_dict.pop("odd", None); selection_dict.pop("implied_prob", None); selection_dict.pop("edge", None); selection_dict.pop("value_ratio", None); selection_dict.pop("odd_source", None)
            continue

        # --- Calculate Metrics ---
        implied_prob = calculate_implied_probability(odds_str)
        if implied_prob is None:
            logger.warning(f"Could not calculate implied probability from odd '{odds_str}' (Source: {odd_source}) for bet '{bet_name}'. Skipping metrics.")
            # Store odd even if implied prob fails, but clear others
            try:
                 selection_dict["odd"] = Decimal(str(odds_str)).quantize(quantize_odds, ROUND_HALF_UP)
                 selection_dict["odd_source"] = odd_source # Store source info
            except Exception:
                 selection_dict.pop("odd", None)
                 selection_dict.pop("odd_source", None)
            selection_dict.pop("implied_prob", None); selection_dict.pop("edge", None); selection_dict.pop("value_ratio", None)
            continue

        try:
            odds_decimal = Decimal(str(odds_str)).quantize(quantize_odds, ROUND_HALF_UP)
            implied_prob_quant = implied_prob.quantize(quantize_prob, ROUND_HALF_UP)
            edge = (predicted_prob - implied_prob).quantize(quantize_edge, ROUND_HALF_UP)
            value_ratio = (predicted_prob / implied_prob).quantize(quantize_ratio, ROUND_HALF_UP) if implied_prob > 0 else Decimal('Infinity')

            # --- Update the dictionary IN PLACE ---
            selection_dict["odd"] = odds_decimal
            selection_dict["implied_prob"] = implied_prob_quant
            selection_dict["edge"] = edge
            selection_dict["value_ratio"] = value_ratio
            selection_dict["odd_source"] = odd_source # Indicate how odd was obtained
            # --- End Update ---

            updated_count += 1
            logger.debug(f"Metrics updated for '{bet_name}' using odd {odds_decimal} (Source: {odd_source}). Edge: {edge:.3f}")

        except Exception as e:
            logger.error(f"Error calculating metrics for selection '{bet_name}' with odd '{odds_str}': {e}")
            # Clear fields on calculation error
            selection_dict.pop("odd", None); selection_dict.pop("implied_prob", None); selection_dict.pop("edge", None); selection_dict.pop("value_ratio", None); selection_dict.pop("odd_source", None)
            continue

    if updated_count > 0:
         logger.info(f"Added/Updated odds/metrics for {updated_count} combined selections in fixture {fixture_id_log}.")

    # Return the SAME dictionary object that was passed in, now potentially modified
    return processed_data

def convert_decimals_to_strings(data):
    """Recursively convert Decimal objects and format numbers for JSON serialization."""
    if isinstance(data, list):
        return [convert_decimals_to_strings(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_decimals_to_strings(v) for k, v in data.items()}
    elif isinstance(data, Decimal):
        if data.is_infinite():
            return 'Infinity'
        # Use appropriate precision based on value (simple heuristic)
        if data.is_nan(): return 'NaN' # Handle NaN just in case
        abs_data = data.copy_abs()
        if abs_data == 0: return "0.0000"
        if abs_data > 100: # Large number, likely not needing high precision
            return f"{data:.2f}"
        elif abs_data >= Decimal('1.0'): # Odds or ratios > 1
             return f"{data.quantize(Decimal('0.01'), ROUND_HALF_UP):.2f}"
        elif abs_data > Decimal('0.000001'): # Probabilities, edges, small ratios
            return f"{data.quantize(Decimal('0.0001'), ROUND_HALF_UP):.4f}"
        else: # Very small numbers, maybe use scientific notation or fixed small value
             return f"{data:.4E}" # Example: Scientific notation
    elif isinstance(data, float):
         # Format floats nicely
         if data.is_integer(): return f"{data:.1f}"
         if abs(data) < 1 and abs(data) > 1e-4 : return f"{data:.4f}"
         elif abs(data) < 10: return f"{data:.3f}"
         else: return f"{data:.2f}"
    elif isinstance(data, (int, str, bool)) or data is None:
        return data
    elif isinstance(data, (datetime, date)): # Handle dates/datetimes if they appear
        return data.isoformat()
    else:
        # Fallback for other types (shouldn't happen often with JSON load)
        # logger.warning(f"Converting unexpected type {type(data)} to string: {str(data)}")
        return str(data)

# --- Function to load the single batch file ---
def load_batch_prediction_data(filepath):
    """Loads the entire batch prediction JSON data from a file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f, parse_float=Decimal) # Use Decimal for precision
        logger.info(f"Successfully loaded batch data from {filepath}")
        if isinstance(data, list):
            return data, "list"
        elif isinstance(data, dict):
             first_key = next(iter(data), None)
             if first_key and isinstance(data[first_key], dict) and \
                (data[first_key].get('fixture_id') or data[first_key].get('match_info') or data[first_key].get('fixture')): # Added 'fixture' check
                 return data, "dict"
             else:
                 logger.error(f"Input file {filepath} is a dictionary, but values don't look like valid match data.")
                 return None, None
        else:
            logger.error(f"Input file {filepath} does not contain a list or a recognized dictionary structure.")
            return None, None
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {filepath}")
        return None, None
    except json.JSONDecodeError as e:
        logger.error(f"Error: Could not decode JSON from {filepath}: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error loading {filepath}: {e}")
        return None, None

# --- Function to extract date (Further Revised Search Logic) ---
def get_match_date_simple(match_data):
    """
    Extracts and formats the match date from match data.
    Searches common keys, specific nested paths, and finally the filename in 'file_path'.
    """
    date_str = None
    date_source = "N/A" # Keep track of where the date was found
    fixture_id_log = match_data.get('fixture_id', 'N/A') # Default to N/A

    # Try to get a better fixture ID for logging if available
    potential_id_paths = [['fixture_id'], ['id'], ['match_info', 'id'], ['fixture', 'id']]
    for path in potential_id_paths:
        temp_data = match_data
        try:
            for key in path: temp_data = temp_data[key]
            if temp_data is not None:
                fixture_id_log = temp_data
                break
        except (TypeError, KeyError, IndexError): continue

    logger.debug(f"--- Finding date for fixture {fixture_id_log} ---")

    # Define potential paths to the date string
    potential_date_paths = [
        (['date'], "top-level 'date'"),
        (['match_date'], "top-level 'match_date'"),
        (['fixture_date'], "top-level 'fixture_date'"),
        (['match_info', 'date'], "'match_info.date'"),
        (['match_info', 'match_date'], "'match_info.match_date'"),
        (['fixture', 'date'], "'fixture.date'"),
        (['bookmakers', 0, 'fixture', 'date'], "'bookmakers[0].fixture.date'")
    ]

    # 1. Try finding the date string using the defined paths
    for path, source_desc in potential_date_paths:
        temp_data = match_data
        found = True
        try:
            for key in path:
                if isinstance(temp_data, dict): temp_data = temp_data.get(key)
                elif isinstance(temp_data, list) and isinstance(key, int) and len(temp_data) > key: temp_data = temp_data[key]
                else: found = False; break
            if found and temp_data is not None:
                date_str = str(temp_data) # Ensure it's a string
                date_source = source_desc
                logger.debug(f"  Found potential date '{date_str}' from source: {date_source}")
                break # Date found, exit search loop
        except (TypeError, IndexError, KeyError):
            continue # Path invalid or doesn't exist

    # 2. --- Fallback to extracting date from file_path ---
    if date_str is None:
        logger.debug("  Date not found via direct keys, attempting fallback from 'file_path'.")
        file_path = match_data.get('file_path')
        if isinstance(file_path, str):
            try:
                filename = os.path.basename(file_path)
                date_part = filename.split('_')[0]
                datetime.strptime(date_part, '%Y-%m-%d') # Validate format YYYY-MM-DD
                date_str = date_part # Use the extracted date string
                date_source = f"file_path extraction ('{filename}')"
                logger.debug(f"  SUCCESS: Found date '{date_str}' from {date_source}")
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"  FAILED: Could not extract valid date from file_path '{file_path}'. Error: {e}")
                date_str = None # Ensure it remains None if extraction fails
        else:
            logger.debug(f"  SKIPPED: 'file_path' key missing or not a string ('{type(file_path)}').")

    # 3. --- Try parsing the found date string (if any) ---
    if date_str:
        try:
            # Try parsing ISO format first
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            formatted_date = date_obj.strftime('%Y-%m-%d')
            logger.debug(f"  Successfully parsed date '{date_str}' (source: {date_source}) -> {formatted_date}")
            return formatted_date
        except ValueError:
            try:
                # Fallback to parsing just YYYY-MM-DD
                date_obj = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                formatted_date = date_obj.strftime('%Y-%m-%d')
                logger.debug(f"  Successfully parsed date '{date_str}' (source: {date_source}) -> {formatted_date} (using YYYY-MM-DD parse)")
                return formatted_date
            except ValueError:
                logger.error(f"  FAILED PARSING: Could not parse potential date string '{date_str}' (found via {date_source}) for fixture {fixture_id_log}")
                return None # Parsing failed even though we found a string
    else:
        # This log means neither direct key search nor file_path extraction yielded a date string
        logger.warning(f"Could not find a recognizable date key/value OR extract from file_path for fixture {fixture_id_log}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    import argparse
    parser = argparse.ArgumentParser(description="Processes combined selections in a batch prediction file, adds odds and value metrics.")
    parser.add_argument('--input', type=str, default=INPUT_OUTPUT_FILE,
                        help=f'Path to the input/output JSON file (default: {INPUT_OUTPUT_FILE})')
    parser.add_argument('--bookmaker', type=str, default=BOOKMAKER_NAME,
                        help=f'Name of the bookmaker to fetch odds for (default: {BOOKMAKER_NAME})')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('get_data.api_football.db_mongo').setLevel(logging.DEBUG)
        logger.info("--- Debug logging enabled ---")

    input_filepath = args.input
    bookmaker = args.bookmaker

    logger.info(f"Using MongoDB instance provided by db_manager.")
    logger.info(f"Using Bookmaker: {bookmaker}")
    logger.info(f"Processing input file: {input_filepath}")

    if not os.path.exists(input_filepath):
         logger.error(f"Input file not found: {input_filepath}")
         sys.exit(1)

    # --- Load the entire batch data ---
    batch_data, data_format = load_batch_prediction_data(input_filepath)
    if batch_data is None:
         logger.error("Failed to load or parse batch data. Exiting.")
         sys.exit(1)

    # --- Processing Loop ---
    processed_matches = 0
    matches_with_updates = 0
    error_count = 0
    if data_format == "dict": updated_data = {}

    match_iterator = None
    if data_format == "list": match_iterator = enumerate(batch_data)
    elif data_format == "dict": match_iterator = batch_data.items()

    for key_or_index, match_data in match_iterator:
        processed_matches += 1
        fixture_id = None
        fixture_id_source_key = None

        # --- Find Fixture ID (Revised Logic) ---
        potential_id_paths = [
            ['fixture_id'], ['fixtureId'], ['id'], # Common top-level keys
            ['match_info', 'id'], # Inside 'match_info'
            ['fixture', 'id'] # Inside 'fixture'
        ]
        if data_format == "list":
            for path in potential_id_paths:
                 temp_data = match_data
                 found_id = True
                 try:
                     for key in path:
                         temp_data = temp_data.get(key)
                         if temp_data is None: found_id = False; break
                     if found_id and temp_data is not None:
                         fixture_id = temp_data
                         logger.debug(f"Found fixture ID {fixture_id} using path {path} in list item {key_or_index+1}")
                         break
                 except (TypeError, KeyError):
                     continue # Path doesn't exist or is invalid type

        elif data_format == "dict":
             fixture_id = key_or_index # Key is the primary ID
             fixture_id_source_key = key_or_index
             # Verify against internal ID
             internal_id = None
             for path in potential_id_paths:
                 temp_data = match_data
                 found_id = True
                 try:
                     for key in path:
                         temp_data = temp_data.get(key)
                         if temp_data is None: found_id = False; break
                     if found_id and temp_data is not None:
                         internal_id = temp_data
                         break
                 except (TypeError, KeyError):
                     continue
             if internal_id and str(internal_id) != str(fixture_id):
                  logger.warning(f"Dict key '{fixture_id}' differs from internal ID '{internal_id}' found at path {path}. Using key '{fixture_id}'.")
        # --- End Fixture ID Finding ---

        display_id = f"fixture {fixture_id}" if fixture_id else f"entry {key_or_index+1 if isinstance(key_or_index, int) else key_or_index}"

        if not fixture_id:
            logger.warning(f"Skipping {display_id}: Failed to find fixture ID in expected locations.")
            if data_format == "dict": updated_data[fixture_id_source_key] = match_data
            error_count += 1
            continue

        logger.debug(f"\n--- Processing {display_id} ---")
        match_date_simple = get_match_date_simple(match_data) # Use the MOST refined date finder
        if not match_date_simple:
            logger.warning(f"Skipping {display_id}: Could not determine valid date for odds lookup.")
            if data_format == "dict": updated_data[fixture_id_source_key] = match_data
            error_count += 1
            continue

        # --- Try getting odds and processing ---
        try:
            odds_list = get_odds_from_db(str(fixture_id), match_date_simple, bookmaker)

            selections_before = json.dumps(
                 match_data.get("match_analysis", {}).get("top_n_combined_selections") or
                 match_data.get("top_n_combined_selections", []), default=str
            )
            processed_match_data = process_combined_selections(match_data, odds_list) # Modifies match_data in place
            selections_after = json.dumps(
                 processed_match_data.get("match_analysis", {}).get("top_n_combined_selections") or
                 processed_match_data.get("top_n_combined_selections", []), default=str
            )

            if selections_before != selections_after:
                 logger.info(f"Updates applied to {display_id}.") # Changed log level to INFO for updates
                 matches_with_updates += 1

            if data_format == "dict":
                 updated_data[fixture_id_source_key] = processed_match_data

        except Exception as e:
             logger.error(f"Unhandled error during processing loop for {display_id}: {e}")
             import traceback
             traceback.print_exc()
             if data_format == "dict": updated_data[fixture_id_source_key] = match_data
             error_count += 1
             continue # Skip to next match on error

    # --- Determine final data to write ---
    final_data_to_write = batch_data if data_format == "list" else updated_data

    # --- Serialize and Write Back ---
    logger.info(f"Preparing to write updated data back to {input_filepath}...")
    try:
        serializable_data = convert_decimals_to_strings(final_data_to_write) # Convert Decimals right before writing
        with open(input_filepath, 'w') as f:
            json.dump(serializable_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully updated data written back to: {input_filepath}")
    except Exception as write_error:
        logger.error(f"Error writing updated data back to {input_filepath}: {write_error}")
        error_count += 1

    # --- Final Summary ---
    logger.info("\n" + "="*50)
    logger.info("--- Processing Summary ---")
    logger.info(f"Total matches processed from file: {processed_matches}")
    logger.info(f"Matches with updated combined selection odds/metrics: {matches_with_updates}")
    logger.info(f"Errors encountered during processing/writing: {error_count}")
    logger.info("="*50)

    # --- Close DB Connection ---
    try:
        db_manager.close_connection()
        logger.info("\nMongoDB connection closed via db_manager.")
    except Exception as e:
         logger.error(f"Error closing MongoDB connection: {e}")

    logger.info("Script finished.")