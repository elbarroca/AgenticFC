import json
import os
import sys
import logging
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import datetime , date
import argparse
from collections import defaultdict
from typing import Any

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Prevent adding duplicate handlers if script is re-run
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Constants & Default Settings ---
DEFAULT_INPUT_FILE = os.path.join("..", "data", "output", "batch_prediction_results.json") # Relative path
DEFAULT_OUTPUT_FILE = os.path.join("..", "data", "output", "generated_papers.json")      # Relative path
DEFAULT_SELECTIONS_PER_PAPER = 3
DEFAULT_EDGE_THRESHOLD = Decimal('0.00') # Only consider bets with positive edge

# Default Risk Tiers (Based on individual selection odds)
# These can be overridden by command-line arguments
DEFAULT_RISK_TIERS = {
    "Low": {"max_odds": Decimal('1.5')},
    "Mid": {"min_odds": Decimal('1.5'), "max_odds": Decimal('2.5')},
    "High": {"min_odds": Decimal('2.5')}
}

# --- Helper Functions ---

def safe_get(data: dict, keys: list, default: Any = None) -> Any:
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

def get_nested_value(data_dict, keys, default=None):
    """Safely retrieve a value from a nested dictionary."""
    current = data_dict
    try:
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list) and isinstance(key, int) and len(current) > key:
                 current = current[key]
            else:
                return default
            if current is None:
                return default
        return current
    except (TypeError, IndexError):
        return default

def parse_decimal(value, context=""):
    """Safely parse a value to Decimal, handling strings, None, etc."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        # Remove percentage signs if present
        str_value = str(value).strip().replace('%', '')
        return Decimal(str_value)
    except (InvalidOperation, ValueError, TypeError):
        logger.warning(f"Could not parse '{value}' as Decimal {context}. Returning None.")
        return None

def find_fixture_id(match_data):
    """Finds the fixture ID from various possible keys, including nested raw_data."""
    potential_id_paths = [
        ['fixture_id'], ['fixtureId'], ['id'], # Common top-level keys
        ['match_info', 'id'], # Inside 'match_info'
        ['fixture', 'id'], # Inside 'fixture'
        # Check inside raw_data and fixture_meta which are now passed top-level
        ['raw_data', 'fixture', 'id'],
        ['raw_data', 'fixture_id'], # If raw_data itself has it
        ['fixture_meta', 'fixture', 'id'], # Inside fixture_meta
        ['fixture_meta', 'id'], # Inside fixture_meta
    ]
    fixture_id = None
    for path in potential_id_paths:
        potential_id = get_nested_value(match_data, path)
        if isinstance(potential_id, dict) and 'id' in potential_id: # Check if path leads to dict with id
             potential_id = potential_id['id']

        if potential_id is not None:
            fixture_id = str(potential_id) # Ensure string representation
            logger.debug(f"Found fixture ID {fixture_id} using path {path}")
            return fixture_id
            
    # If the match_data itself is the key in a larger dict, that key might be the ID
    # This scenario is handled during the loading/iteration phase, not solely here.
    logger.debug("Could not find fixture ID via standard paths.")
    return None

def find_match_description(match_data):
    """Creates a simple 'Home vs Away' description, checking multiple common paths."""
    # Paths relative to the match_data root - Prioritize direct keys first
    potential_paths = [
        (['home_team'], ['away_team']), # Check for direct top-level string keys FIRST
        (['teams', 'home', 'name'], ['teams', 'away', 'name']),
        (['match_info', 'home_team_name'], ['match_info', 'away_team_name']),
        (['fixture', 'teams', 'home', 'name'], ['fixture', 'teams', 'away', 'name']),
        # Removed ['home_team', 'name'] as the first path covers direct key access
        (['homeTeam'], ['awayTeam']) # Check for alternative casing/naming
    ]

    home_name, away_name = 'Home', 'Away' # Defaults
    fixture_id_log = find_fixture_id(match_data) or 'N/A' # Get ID for logging

    for home_path, away_path in potential_paths:
        h = get_nested_value(match_data, home_path)
        a = get_nested_value(match_data, away_path)
        # Check if both values were found and are non-empty strings
        if h and isinstance(h, (str, Decimal, int)) and a and isinstance(a, (str, Decimal, int)):
             # Convert potential non-strings (like IDs used as names sometimes) to string
            home_name = str(h)
            away_name = str(a)
            logger.debug(f"Found team names for fixture {fixture_id_log} using paths: {home_path} / {away_path}")
            # Ensure we don't return empty strings if found
            if home_name.strip() and away_name.strip():
                 return f"{home_name} vs {away_name}"
            else:
                 logger.warning(f"Found empty team names for fixture {fixture_id_log} using paths {home_path}/{away_path}. Continuing search.")
                 home_name, away_name = 'Home', 'Away' # Reset to default if found names were empty

    logger.warning(f"Could not find specific home/away team names for fixture {fixture_id_log}. Using defaults 'Home vs Away'.")
    return f"{home_name} vs {away_name}"

def find_match_date(match_data):
    """Finds the match date from various possible keys."""
    potential_date_paths = [
        ['date'], ['match_date'], ['fixture_date'],
        ['match_info', 'date'], ['match_info', 'match_date'],
        ['fixture', 'date'],
        # Accessing date within a potential list of bookmakers is less reliable here
        # ['bookmakers', 0, 'fixture', 'date'] # Removed this less common path
    ]
    date_str = None
    for path in potential_date_paths:
        date_val = get_nested_value(match_data, path)
        if date_val:
            date_str = str(date_val)
            logger.debug(f"Found date string '{date_str}' using path {path}")
            break

    if date_str:
        try:
            # Handle ISO format with timezone
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            try:
                # Handle YYYY-MM-DD format (or just the date part)
                date_obj = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                logger.warning(f"Could not parse date string: {date_str}")
    logger.warning(f"Could not find or parse date for fixture {find_fixture_id(match_data)}.")
    return "Unknown Date"

def convert_for_json(data):
    """Recursively convert Decimal to string and handle other types for JSON."""
    if isinstance(data, list):
        return [convert_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_for_json(v) for k, v in data.items()}
    elif isinstance(data, Decimal):
        if data.is_infinite(): return 'Infinity'
        if data.is_nan(): return 'NaN'
        # Simple formatting, adjust precision as needed
        # Use '.2f' for odds, '.4f' for probabilities/edges
        # Check magnitude to decide formatting
        abs_data = data.copy_abs()
        if abs_data >= Decimal('10.0'): # Likely large odds or high value ratio
             return f"{data.quantize(Decimal('0.01'), ROUND_HALF_UP):.2f}"
        elif abs_data >= Decimal('0.00005'): # Most other cases (probs, edges, low odds)
             return f"{data.quantize(Decimal('0.0001'), ROUND_HALF_UP):.4f}"
        elif abs_data == Decimal('0'):
            return "0.0000"
        else: # Very small number, potentially use scientific notation or more decimals
            return f"{data:.6f}" # Example: Use 6 decimal places for tiny numbers
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, (str, int, float, bool)) or data is None:
        return data
    # Handle potential numpy types if they sneak in (though predict_games should handle this)
    elif hasattr(data, 'tolist'): # Basic check for numpy array
         return data.tolist()
    elif hasattr(data, 'item'): # Basic check for numpy scalar
         return data.item()
    else:
        # Fallback for other types
        try:
             # Attempt a default JSON serialization for unknown types if simple str() fails
             # This might catch some edge cases, but could also raise errors
             json.dumps(data) # Test if serializable
             return data # If it didn't raise, maybe it's fine? Be cautious.
        except TypeError:
             # If default fails, convert to string as a last resort
             return str(data)

# --- Core Logic ---

def load_data(filepath):
    """Loads batch prediction data from JSON file."""
    logger.info(f"Loading batch prediction data from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Use Decimal for numeric values
            data = json.load(f, parse_float=Decimal, parse_int=Decimal)

        # Handle both list and dictionary input formats
        if isinstance(data, dict):
            # Assuming keys are fixture IDs and values are match data dicts
            logger.info("Input data is a dictionary; converting to list of matches.")
            processed_data = []
            for fixture_key, match_info in data.items():
                 if isinstance(match_info, dict):
                     # Attempt to inject the key as fixture_id if not present internally
                     if 'fixture_id' not in match_info and 'id' not in match_info:
                          try:
                              # Ensure key is numeric-like before assigning
                              _ = Decimal(fixture_key)
                              match_info['fixture_id'] = fixture_key # Use the dict key
                              logger.debug(f"Injected dictionary key '{fixture_key}' as fixture_id.")
                          except InvalidOperation:
                               logger.warning(f"Dictionary key '{fixture_key}' is not numeric, cannot reliably use as fixture ID.")
                     processed_data.append(match_info)
                 else:
                      logger.warning(f"Skipping item with key '{fixture_key}': Value is not a dictionary.")
            data = processed_data

        if not isinstance(data, list):
             logger.error(f"Failed to process input file {filepath} into a list of match data.")
             return None

        logger.info(f"Successfully loaded/processed {len(data)} match entries.")
        return data
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error: Could not decode JSON from {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {filepath}: {e}", exc_info=True) # Add traceback
        return None

def extract_and_enrich_selections(batch_data, edge_threshold):
    """Extracts, enriches, filters, and ranks selections from all matches."""
    all_selections = []
    processed_fixture_ids = set()
    skipped_duplicates = 0
    skipped_no_id = 0
    skipped_no_selections = 0
    skipped_parsing_errors = 0
    skipped_edge_threshold = 0

    for i, match_data in enumerate(batch_data):
        if not isinstance(match_data, dict):
            logger.warning(f"Skipping entry {i+1}: Not a dictionary.")
            continue

        fixture_id = find_fixture_id(match_data)
        if not fixture_id:
            # Attempt to find ID from potentially nested raw_data if top-level failed
            fixture_id = find_fixture_id(safe_get(match_data, ['raw_data'], {}))
            if not fixture_id:
                logger.warning(f"Skipping entry {i+1}: Could not determine fixture ID even in raw_data.")
                skipped_no_id += 1
                continue
            else:
                 logger.debug(f"Found fixture ID {fixture_id} within raw_data for entry {i+1}.")


        # Handle potential duplicate fixture IDs in the input list
        if fixture_id in processed_fixture_ids:
            logger.warning(f"Skipping entry {i+1}: Duplicate fixture ID {fixture_id} encountered in list.")
            skipped_duplicates += 1
            continue
        processed_fixture_ids.add(fixture_id)

        # --- Call find_match_description AFTER finding ID and checking for duplicates ---
        # Pass the whole match_data initially, find_match_description will check nested paths including raw_data if needed
        match_desc = find_match_description(match_data)
        match_date = find_match_date(match_data) # find_match_date can also check common paths

        # --- Extract Additional Raw Info ---
        # Use safe_get, accessing the new top-level keys 'fixture_meta' and 'raw_data'
        fixture_meta = safe_get(match_data, ['fixture_meta'], {})
        raw_data = safe_get(match_data, ['raw_data'], {})

        match_timestamp_utc = safe_get(fixture_meta, ['date_utc'], None) # Example: Get full timestamp
        home_logo = safe_get(raw_data, ['home', 'basic_info', 'logo'], None) # Example: Get home logo
        away_logo = safe_get(raw_data, ['away', 'basic_info', 'logo'], None) # Example: Get away logo
        # Add more fields as needed, e.g.:
        # league_name = safe_get(fixture_meta, ['league', 'name'], None)
        # referee = safe_get(fixture_meta, ['referee'], None)
        # h2h_data = safe_get(raw_data, ['h2h'], [])


        # Check multiple possible locations for the selections list
        selections_list = get_nested_value(match_data, ['match_analysis', 'top_n_combined_selections'])
        if selections_list is None:
            selections_list = get_nested_value(match_data, ['top_n_combined_selections']) # Check top level too

        if not selections_list or not isinstance(selections_list, list):
            logger.debug(f"No 'top_n_combined_selections' found for fixture {fixture_id}.")
            skipped_no_selections +=1
            continue

        fixture_selection_count = 0
        for sel_idx, selection_dict in enumerate(selections_list):
            if not isinstance(selection_dict, dict):
                logger.warning(f"Skipping selection {sel_idx+1} in fixture {fixture_id}: Not a dictionary.")
                continue

            # Extract and parse required fields from selection_dict
            bet_name = selection_dict.get("selection")
            # --- Get probability and odd source from the selection_dict itself ---
            prob_raw = selection_dict.get("probability") # Probability comes from the calculated selections
            odd_raw = selection_dict.get("odd")         # Odd might be added here later or be part of selection_dict
            edge_raw = selection_dict.get("edge")       # Edge might be added here later or be part of selection_dict
            odd_source = selection_dict.get("odd_source", "Unknown") # Capture how odd was derived

            # Parse numerical values
            prob = parse_decimal(prob_raw, context=f"for probability in {fixture_id}/{bet_name}")
            odd = parse_decimal(odd_raw, context=f"for odd in {fixture_id}/{bet_name}")
            edge = parse_decimal(edge_raw, context=f"for edge in {fixture_id}/{bet_name}")


            # Validate essential data (Now includes checking prob, odd, edge came from selection_dict)
            if not all([bet_name, prob is not None, odd is not None, edge is not None]):
                logger.warning(f"Skipping selection '{bet_name}' in fixture {fixture_id}: Missing or invalid essential data from selection dict (prob={prob_raw}, odd={odd_raw}, edge={edge_raw}).")
                skipped_parsing_errors += 1
                continue

            # Apply edge filter
            if edge <= edge_threshold:
                logger.debug(f"Filtering out selection '{bet_name}' in fixture {fixture_id}: Edge {edge:.4f} <= threshold {edge_threshold:.4f}.")
                skipped_edge_threshold += 1
                continue

            # Enrich the selection data
            enriched = {
                "fixture_id": fixture_id,
                "match_description": match_desc, # Uses the result from find_match_description
                "match_date": match_date,        # Date part YYYY-MM-DD
                "match_timestamp_utc": match_timestamp_utc, # Full timestamp if available
                "home_team_logo": home_logo,     # Home logo URL/path
                "away_team_logo": away_logo,     # Away logo URL/path
                # Add other extracted raw fields if needed:
                # "league_name": league_name,
                # "referee": referee,
                # Add selection specific data from selection_dict
                "selection": bet_name,
                "probability": prob,
                "odds": odd,
                "edge": edge,
                "odd_source": odd_source,
                # You might also want to include the raw probability/odd/edge values
                "raw_probability": prob_raw,
                "raw_odd": odd_raw,
                "raw_edge": edge_raw,
                # Add other fields like value_ratio if needed later
            }
            all_selections.append(enriched)
            fixture_selection_count += 1

        logger.debug(f"Processed fixture {fixture_id}: Added {fixture_selection_count} selections.")


    # Rank all valid selections by edge (highest first) - This is the pool for paper generation
    all_selections.sort(key=lambda x: x['edge'], reverse=True)

    logger.info(f"Extraction Summary: Input Matches={len(batch_data)}, Unique Fixtures Processed={len(processed_fixture_ids)}, "
                f"Valid Selections (Edge > {edge_threshold:.4f})={len(all_selections)}")
    logger.info(f"Skipped Counts: No ID={skipped_no_id}, Duplicates={skipped_duplicates}, No Selections={skipped_no_selections}, "
                f"Parsing Errors={skipped_parsing_errors}, Below Edge Threshold={skipped_edge_threshold}")
    return all_selections

def define_risk_tier(selection_odds, risk_tiers_config):
    """Determines the risk tier ('Low', 'Mid', 'High', or None) based on odds."""
    if selection_odds is None: return None

    # Iterate through tiers, checking if odds fit within defined min/max
    # Important: Assumes tiers are defined non-overlappingly or handles overlap consistently
    for tier_name, limits in risk_tiers_config.items():
        min_odds = limits.get('min_odds')
        max_odds = limits.get('max_odds')

        is_match = True
        # Note: Tier boundary checks:
        # - If min_odds is defined, selection_odds MUST BE > min_odds
        # - If max_odds is defined, selection_odds MUST BE <= max_odds
        # This means a selection with odds exactly 1.5 would fall into "Mid" with default tiers.
        if min_odds is not None and not (selection_odds > min_odds):
            is_match = False
        if max_odds is not None and not (selection_odds <= max_odds):
             is_match = False

        if is_match:
            return tier_name
            
    logger.debug(f"Odds {selection_odds} did not match any defined risk tier.")
    return None # No matching tier found


def generate_papers(ranked_selections, selections_per_paper, risk_tiers_config):
    """Generates betting papers using a greedy approach based on ranked selections."""
    papers_by_tier = defaultdict(list) # { "Low": [paper1, paper2], "Mid": [...], ... }
    used_selection_indices = set() # Track indices from the *ranked_selections* list

    paper_counters = defaultdict(int) # To generate unique IDs like low_1, low_2

    logger.info(f"Generating papers with {selections_per_paper} selections each, ensuring fixture uniqueness per paper...")

    # --- Greedy Paper Filling Algorithm ---
    # Iterate through the selections ranked by highest edge first
    for i, current_selection in enumerate(ranked_selections):
        if i in used_selection_indices:
            continue # Skip if already placed in a paper

        selection_odds = current_selection.get('odds')
        risk_tier = define_risk_tier(selection_odds, risk_tiers_config)
        if risk_tier is None:
            logger.debug(f"Skipping selection {current_selection['fixture_id']}/{current_selection['selection']} (odds {selection_odds}): Does not fit defined risk tiers.")
            continue

        # Try to add to an existing, incomplete paper of the *same risk tier*
        added_to_existing = False
        # Iterate through papers already started within this tier
        for paper in papers_by_tier[risk_tier]:
            # Check if paper needs more selections
            if len(paper["selections"]) < selections_per_paper:
                # Check if the game (fixture_id) is already in this specific paper
                paper_fixture_ids = {sel['fixture_id'] for sel in paper["selections"]}
                if current_selection["fixture_id"] not in paper_fixture_ids:
                    # If paper isn't full and game isn't present, add selection
                    paper["selections"].append(current_selection)
                    used_selection_indices.add(i) # Mark this selection index as used
                    added_to_existing = True
                    logger.debug(f"Added selection {current_selection['fixture_id']}/{current_selection['selection']} to existing paper {paper['paper_id']}")
                    break # Stop searching for a paper for this selection

        # If the selection couldn't be added to any existing paper (either all full, or all had game conflicts)
        if not added_to_existing:
            # Start a new paper if we have enough selections for a full paper eventually
             if len(ranked_selections) - len(used_selection_indices) >= selections_per_paper : # Optimistic check
                paper_counters[risk_tier] += 1
                paper_id = f"{risk_tier.lower()}_{paper_counters[risk_tier]}"
                new_paper = {
                    "paper_id": paper_id,
                    "risk_tier": risk_tier,
                    "selections": [current_selection] # Start with the current selection
                }
                papers_by_tier[risk_tier].append(new_paper)
                used_selection_indices.add(i) # Mark this selection index as used
                logger.debug(f"Started new paper {paper_id} with selection {current_selection['fixture_id']}/{current_selection['selection']}")
             else:
                  logger.debug(f"Skipping starting new paper for selection {current_selection['fixture_id']}/{current_selection['selection']}: Not enough remaining selections potentially.")


    # --- Finalize and Rank Papers ---
    final_papers = []
    incomplete_papers_count = 0

    # Flatten the list of all potentially generated papers
    all_papers_flat = [paper for tier_list in papers_by_tier.values() for paper in tier_list]

    for paper in all_papers_flat:
        # Only include papers that are fully populated
        if len(paper["selections"]) == selections_per_paper:
            total_odds = Decimal('1.0')
            sum_of_edges = Decimal('0.0')
            constituent_edges = []
            all_odds_valid = True
            for sel in paper["selections"]:
                 sel_odds = sel.get('odds')
                 sel_edge = sel.get('edge')
                 if sel_odds is None or sel_edge is None:
                      all_odds_valid = False
                      logger.error(f"Paper {paper['paper_id']} contains selection with missing odds/edge. Excluding paper.")
                      break
                 total_odds *= sel_odds
                 sum_of_edges += sel_edge
                 constituent_edges.append(sel_edge) # Keep individual edges

            if not all_odds_valid:
                 incomplete_papers_count +=1 # Treat as incomplete/invalid
                 continue

            paper["total_odds"] = total_odds
            paper["constituent_edges"] = constituent_edges
            # Calculate average edge for ranking
            paper["average_edge"] = sum_of_edges / Decimal(selections_per_paper)

            final_papers.append(paper)
        else:
            # Log incomplete papers found after the generation process
            logger.warning(f"Paper {paper['paper_id']} has {len(paper['selections'])} selections, expected {selections_per_paper}. Excluding final paper.")
            incomplete_papers_count += 1


    logger.info(f"Generated {len(final_papers)} complete papers. Excluded {incomplete_papers_count} incomplete papers.")

    # Rank final papers: Primary: Average Edge (Descending), Secondary: Total Odds (Ascending)
    # Higher average edge is better. For ties, lower total odds ("more probable") is ranked higher.
    final_papers.sort(key=lambda p: (p.get('average_edge', Decimal('-1')), -p.get('total_odds', Decimal('Infinity'))), reverse=True)

    logger.info("Final papers ranked by average edge (desc) then total odds (asc).")

    return final_papers


def parse_risk_tiers(tier_args):
    """ Parses command-line risk tier definitions. """
    if not tier_args:
        logger.info(f"Using default risk tiers: {DEFAULT_RISK_TIERS}")
        return DEFAULT_RISK_TIERS

    parsed_tiers = {}
    try:
        tier_names_order = [] # Maintain definition order if needed later
        for tier_def in tier_args:
            parts = tier_def.split(':')
            if len(parts) != 2: raise ValueError(f"Invalid format (expect Name:key=value,...): {tier_def}")
            tier_name = parts[0].strip()
            if not tier_name: raise ValueError("Tier name cannot be empty")

            limits_str = parts[1].strip().split(',')
            limits = {}
            for limit in limits_str:
                limit_parts = limit.split('=')
                if len(limit_parts) != 2: raise ValueError(f"Invalid limit format (expect key=value): {limit}")
                key = limit_parts[0].strip().lower() # Standardize key names
                value_str = limit_parts[1].strip()
                if key not in ['min_odds', 'max_odds']: raise ValueError(f"Invalid limit key: {key}")
                try:
                     value = Decimal(value_str)
                except InvalidOperation:
                     raise ValueError(f"Invalid decimal value for {key}: {value_str}")
                limits[key] = value

            if not limits: raise ValueError(f"No valid limits found for tier: {tier_name}")
            # Optional: Add validation for overlapping tiers or inconsistent limits
            if 'min_odds' in limits and 'max_odds' in limits and limits['min_odds'] >= limits['max_odds']:
                 raise ValueError(f"min_odds ({limits['min_odds']}) must be less than max_odds ({limits['max_odds']}) for tier {tier_name}")

            parsed_tiers[tier_name] = limits
            tier_names_order.append(tier_name)

        logger.info(f"Using custom risk tiers: {parsed_tiers}")
        # Add check: ensure all odds ranges are covered or warn if gaps exist? (More complex)
        return parsed_tiers
    except (ValueError, InvalidOperation) as e:
        logger.error(f"Error parsing risk tiers ('{tier_args}'): {e}. Using default tiers.")
        return DEFAULT_RISK_TIERS


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates multi-bet papers from ranked selections in batch prediction results.")
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE,
                        help=f'Path to the input batch prediction JSON file (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f'Path to save the generated papers JSON file (default: {DEFAULT_OUTPUT_FILE})')
    parser.add_argument('--selections', type=int, default=DEFAULT_SELECTIONS_PER_PAPER,
                        help=f'Number of selections per paper (default: {DEFAULT_SELECTIONS_PER_PAPER})')
    parser.add_argument('--min_edge', type=Decimal, default=DEFAULT_EDGE_THRESHOLD,
                        help=f'Minimum edge threshold for selections to be considered (default: {DEFAULT_EDGE_THRESHOLD})')
    parser.add_argument('--risk_tier', action='append',
                        help='Define a risk tier. Format: "TierName:limit1=value,limit2=value". '
                             'Example: --risk_tier "Low:max_odds=1.5" --risk_tier "Mid:min_odds=1.5,max_odds=2.5" '
                             'Overrides defaults if provided. Ensure keys are min_odds or max_odds.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers: # Ensure handlers also respect level
             handler.setLevel(logging.DEBUG)
        logger.info("--- Debug logging enabled ---")

    # Resolve paths relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_filepath = os.path.normpath(os.path.join(script_dir, args.input))
    output_filepath = os.path.normpath(os.path.join(script_dir, args.output))

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filepath)
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        sys.exit(1)


    # Parse risk tiers or use defaults
    risk_tiers_config = parse_risk_tiers(args.risk_tier)

    # 1. Load Data
    batch_prediction_data = load_data(input_filepath)
    if batch_prediction_data is None:
        sys.exit(1)

    # 2. Extract, Enrich, Filter & Rank Selections (Pool for paper generation)
    ranked_selections = extract_and_enrich_selections(batch_prediction_data, args.min_edge)
    if not ranked_selections:
        logger.warning("No suitable selections found after filtering. Cannot generate papers.")
        sys.exit(0)

    # 3. Generate Papers (Greedy algorithm ensuring uniqueness and tiering)
    #    And Rank the completed papers
    generated_papers = generate_papers(ranked_selections, args.selections, risk_tiers_config)
    if not generated_papers:
         logger.warning("Failed to generate any complete papers.")
         sys.exit(0)

    # 4. Prepare Final Output Structure
    output_data = {
        "generation_info": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "input_file": os.path.relpath(input_filepath, script_dir), # Store relative path from script location
            "output_file": os.path.relpath(output_filepath, script_dir),
            "settings": {
                "selections_per_paper": args.selections,
                "min_edge_threshold": args.min_edge, # Store as Decimal before conversion
                "risk_tiers_used": risk_tiers_config
            },
            "summary": {
                 "total_matches_input": len(batch_prediction_data), # Count before potential dict conversion
                 "total_selections_in_pool": len(ranked_selections),
                 "total_papers_generated": len(generated_papers)
            }
        },
        "papers": generated_papers # Already ranked
    }

    # Convert Decimals to strings for JSON output
    serializable_output = convert_for_json(output_data)

    # 5. Write Output
    logger.info(f"Writing {len(generated_papers)} generated papers to: {output_filepath}")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_output, f, indent=4, ensure_ascii=False)
        logger.info("Successfully wrote generated papers.")
    except IOError as e:
        logger.error(f"Error writing output file {output_filepath}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during file writing: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Script finished.")
