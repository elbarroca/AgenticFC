import json
import os
import sys
import logging
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import datetime, date
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import itertools # Add itertools for combinations

# +++ Add necessary imports +++
import numpy as np
import cvxpy as cp

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
DEFAULT_OUTPUT_FILE = os.path.join("..", "data", "output", "optimized_game_portfolios.json") # Renamed output
DEFAULT_EDGE_THRESHOLD = Decimal('0.00') # Only consider bets with positive edge
DEFAULT_KELLY_FRACTION = Decimal('0.5') # Default to 50% fractional Kelly
DEFAULT_RISK_AVERSION = Decimal('1.0')  # Default risk aversion for objective function
# +++ Add new default thresholds +++
DEFAULT_MIN_PROBABILITY = Decimal('0.71') # Default minimum probability threshold
DEFAULT_MIN_ODDS = None # Default: No minimum odds filter
DEFAULT_MAX_ODDS = None # Default: No maximum odds filter
# +++ Add paper building defaults +++
DEFAULT_PAPER_SIZES = [2, 3, 4, 5] # Default sizes for papers
DEFAULT_MAX_PAPERS_PER_SIZE = 25 # <<< REDUCED from 100 to encourage larger papers
DEFAULT_PAPER_BUILD_STRATEGY = 'highest_edge' # Strategy for selecting bet within a game for a paper
# --- Solver Change ---
OPTIMIZATION_SOLVER = cp.SCS # Changed default solver from ECOS to SCS
# ---
MIN_STAKE_THRESHOLD = Decimal('0.0001') # Minimum stake fraction to include in the output paper
# --- Risk Category Thresholds (Based on Average Odds in Paper) ---
RISK_CATEGORY_THRESHOLDS = {
    "Low": Decimal('1.80'), # Avg Odds <= 1.80
    "Mid": Decimal('3.00'),  # Avg Odds > 1.80 and <= 3.00
    # "High" is implicitly > 3.00
}
# ---

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
    potential_paths = [
        # Prioritize explicit team names if available
        (['teams', 'home', 'name'], ['teams', 'away', 'name']),
        (['raw_data', 'home', 'basic_info', 'name'], ['raw_data', 'away', 'basic_info', 'name']), # Check raw_data
        (['match_info', 'home_team_name'], ['match_info', 'away_team_name']),
        (['fixture', 'teams', 'home', 'name'], ['fixture', 'teams', 'away', 'name']),
        # Fallback to potentially just IDs if names aren't strings
        (['home_team'], ['away_team']),
        (['homeTeam'], ['awayTeam'])
    ]
    home_name, away_name = 'Home', 'Away'
    fixture_id_log = find_fixture_id(match_data) or 'N/A'
    for home_path, away_path in potential_paths:
        h = get_nested_value(match_data, home_path)
        a = get_nested_value(match_data, away_path)
        # Ensure fetched values are treated as strings and are non-empty
        h_str = str(h).strip() if h is not None else ""
        a_str = str(a).strip() if a is not None else ""

        if h_str and a_str:
            home_name = h_str
            away_name = a_str
            logger.debug(f"Found team names '{home_name}' vs '{away_name}' for {fixture_id_log} using {home_path}/{away_path}")
            return f"{home_name} vs {away_name}"

    logger.warning(f"Could not find specific non-empty home/away names for {fixture_id_log}. Using defaults.")
    return f"{home_name} vs {away_name}"

def find_match_date(match_data):
    """Finds the match date and time, prioritizing fixture_meta.date_utc."""
    potential_date_paths = [
        ['fixture_meta', 'date_utc'], # *** PRIORITIZE THIS ***
        ['fixture', 'date'],
        ['date'],
        ['match_date'],
        ['fixture_date'],
        ['match_info', 'date'],
        ['match_info', 'match_date'],
        ['raw_data', 'fixture', 'date']
    ]
    date_str = None
    fixture_id_log = find_fixture_id(match_data) or 'N/A' # Get ID for logging

    for path in potential_date_paths:
        date_val = get_nested_value(match_data, path)
        if date_val:
            date_str = str(date_val)
            logger.debug(f"Fixture {fixture_id_log}: Found date string '{date_str}' using path {path}")
            break # Use the first non-null date found

    if date_str:
        try:
            # Handle ISO format with timezone, remove fractional seconds if present
            if '.' in date_str and ('+' in date_str or 'Z' in date_str):
                 # Handle cases like '2024-05-18T14:00:00.000+00:00' or '...Z'
                 base = date_str.split('.')[0]
                 tz_part = ""
                 if '+' in date_str:
                     tz_part = date_str[date_str.rfind('+'):]
                 elif 'Z' in date_str:
                     tz_part = 'Z'
                 date_str = base + tz_part

            # Ensure Z is converted to +00:00 for fromisoformat
            iso_date_str = date_str.replace('Z', '+00:00')
            date_obj = datetime.fromisoformat(iso_date_str)
            return date_obj.strftime('%Y-%m-%d %H:%M')
        except ValueError:
            try:
                # Handle YYYY-MM-DD format (or just the date part)
                date_obj = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                return date_obj.strftime('%Y-%m-%d %H:%M')
            except ValueError:
                logger.warning(f"Fixture {fixture_id_log}: Could not parse date string: '{date_str}' after trying ISO and YYYY-MM-DD.")
        except Exception as e:
             logger.warning(f"Fixture {fixture_id_log}: Unexpected error parsing date string '{date_str}': {e}")

    logger.warning(f"Fixture {fixture_id_log}: Could not find or parse a valid date/time. Using 'Unknown DateTime'.")
    return "Unknown DateTime"

def convert_for_json(data):
    """Recursively convert Decimal/numpy types to string/list/float for JSON."""
    if isinstance(data, list):
        return [convert_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_for_json(v) for k, v in data.items()}
    elif isinstance(data, Decimal):
        if data.is_infinite(): return 'Infinity'
        if data.is_nan(): return 'NaN'
        abs_data = data.copy_abs()
        if abs_data >= Decimal('10.0'):
             return f"{data.quantize(Decimal('0.01'), ROUND_HALF_UP):.2f}"
        elif abs_data >= Decimal('0.00005') or abs_data == Decimal('0'):
             return f"{data.quantize(Decimal('0.0001'), ROUND_HALF_UP):.4f}"
        else:
            return f"{data:.6f}"
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, (str, int, float, bool)) or data is None:
        return data
    # Handle numpy types
    elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        # Convert numpy floats carefully to avoid precision issues, potentially to string
        # Or just use standard float conversion
        return float(data)
    elif isinstance(data, (np.complex_, np.complex64, np.complex128)):
        return {'real': float(data.real), 'imag': float(data.imag)}
    elif isinstance(data, (np.ndarray,)):
        return data.tolist() # Convert numpy arrays to lists
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.void):
        return None # Or decide how to handle void types
    # Fallback for other types (unchanged from original)
    elif hasattr(data, 'tolist'): return data.tolist()
    elif hasattr(data, 'item'): return data.item()
    else:
        try:
            json.dumps(data)
            return data
        except TypeError:
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

# +++ MODIFIED FUNCTION: Filter selections and gather more data +++
def filter_game_selections(match_data: Dict, fixture_id: str, min_edge: Decimal,
                           min_probability: Decimal, min_odds: Optional[Decimal],
                           max_odds: Optional[Decimal]) -> List[Dict]:
    """
    Filters selections and gathers context data (teams, date, logos, form, venue, weather).
    """
    valid_selections = []
    # --- Get Match Context Data ---
    match_date = find_match_date(match_data)
    home_name = get_nested_value(match_data, ['teams', 'home', 'name'], get_nested_value(match_data, ['raw_data', 'home', 'basic_info', 'name'], 'Home'))
    away_name = get_nested_value(match_data, ['teams', 'away', 'name'], get_nested_value(match_data, ['raw_data', 'away', 'basic_info', 'name'], 'Away'))
    home_logo = get_nested_value(match_data, ['teams', 'home', 'logo'], get_nested_value(match_data, ['raw_data', 'home', 'basic_info', 'logo']))
    away_logo = get_nested_value(match_data, ['teams', 'away', 'logo'], get_nested_value(match_data, ['raw_data', 'away', 'basic_info', 'logo']))
    home_form = get_nested_value(match_data, ['raw_data', 'home', 'match_processor_snapshot', 'form_string'])
    away_form = get_nested_value(match_data, ['raw_data', 'away', 'match_processor_snapshot', 'form_string'])
    match_desc = f"{str(home_name).strip()} vs {str(away_name).strip()}" # Ensure string names

    # ++ Add Venue and Weather ++
    venue_info = get_nested_value(match_data, ['fixture_meta', 'venue']) # Gets the whole dict (name, city)
    weather_summary = get_nested_value(match_data, ['fixture_meta', 'weather_forecast', 'summary'])
    # --

    # --- Filter Selections ---
    selections_list = get_nested_value(match_data, ['match_analysis', 'top_n_combined_selections'])
    if selections_list is None:
        selections_list = get_nested_value(match_data, ['top_n_combined_selections'])

    if not selections_list or not isinstance(selections_list, list): return []
    logger.debug(f"Fixture {fixture_id}: Filtering {len(selections_list)} potential selections...")

    for sel_data in selections_list:
        if not isinstance(sel_data, dict): continue
        edge = parse_decimal(sel_data.get("edge"), context=f"edge in {fixture_id}/{sel_data.get('selection')}")
        prob = parse_decimal(sel_data.get("probability"), context=f"prob in {fixture_id}/{sel_data.get('selection')}")
        odd = parse_decimal(sel_data.get("odd"), context=f"odd in {fixture_id}/{sel_data.get('selection')}")
        selection_name = sel_data.get("selection")

        # --- Apply Filters (logic unchanged) ---
        if edge is None or prob is None or odd is None or selection_name is None: continue
        if edge <= min_edge: continue
        if prob < min_probability: continue
        if min_odds is not None and odd < min_odds: continue
        if max_odds is not None and odd > max_odds: continue
        if odd <= Decimal('1.0') or prob <= Decimal('0.0') or prob >= Decimal('1.0'): continue

        # Passed filters - add enriched data
        valid_selections.append({
            "fixture_id": fixture_id,
            "match_description": match_desc,
            "match_date": match_date,
            "home_team_name": str(home_name).strip(), # Ensure string
            "away_team_name": str(away_name).strip(), # Ensure string
            "home_team_logo": home_logo,
            "away_team_logo": away_logo,
            "home_form_string": home_form,
            "away_form_string": away_form,
            "venue_info": venue_info, # ++ Added
            "weather_summary": weather_summary, # ++ Added
            "selection": selection_name,
            "probability": prob,
            "odds": odd,
            "edge": edge,
            "odd_source": sel_data.get("odd_source", "Unknown")
        })

    logger.debug(f"Fixture {fixture_id}: Found {len(valid_selections)} selections meeting criteria.")
    return valid_selections
# --- End of MODIFIED FUNCTION ---

# +++ MODIFIED FUNCTION: Build papers ensuring selection uniqueness +++
def build_papers(filtered_selections_by_game: Dict[str, List[Dict]],
                 paper_sizes: List[int],
                 max_papers_per_size: int,
                 strategy: str = 'highest_edge') -> List[List[Dict]]:
    """
    Builds multiple betting papers, ensuring each specific selection
    (fixture_id, selection_name) is used in at most one paper.
    """
    all_papers = []
    used_selections = set() # Stores tuples: (fixture_id, selection_name)
    available_fixture_ids = list(filtered_selections_by_game.keys())
    num_available_games = len(available_fixture_ids)
    logger.info(f"--- Building Papers (Ensuring Unique Selections) ---")
    logger.info(f"Strategy: Try '{strategy}' bet per game, skipping used selections.")
    logger.info(f"Target paper sizes: {paper_sizes}")
    logger.info(f"Max papers per size: {max_papers_per_size}")
    logger.info(f"Games available with valid selections: {num_available_games}")

    if num_available_games < min(paper_sizes):
         logger.warning(f"Not enough games ({num_available_games}) to build minimum size {min(paper_sizes)} papers.")
         return []

    total_papers_built = 0
    for size in paper_sizes:
        if size > num_available_games:
            logger.info(f"Skipping paper size {size}: Not enough games available ({num_available_games}).")
            continue

        logger.info(f"Generating combinations for paper size {size}...")
        game_combinations = itertools.combinations(available_fixture_ids, size)
        papers_built_for_size = 0

        for game_combo in game_combinations:
            if papers_built_for_size >= max_papers_per_size:
                 logger.info(f"Reached max papers ({max_papers_per_size}) for size {size}.")
                 break

            current_paper = []
            possible_to_build = True
            temp_selections_for_paper = [] # Store (key, selection_dict) pairs before adding to used set

            for fixture_id in game_combo:
                selections_for_game = filtered_selections_by_game[fixture_id]
                if not selections_for_game:
                    logger.warning(f"Internal Warning: Fixture {fixture_id} in combination but has no selections. Skipping combo.")
                    possible_to_build = False
                    break

                # Sort selections by strategy
                if strategy == 'highest_edge':
                    sorted_selections = sorted(selections_for_game, key=lambda s: s.get('edge', Decimal('-Infinity')), reverse=True)
                else:
                    logger.warning(f"Unknown build strategy '{strategy}', using 'highest_edge'.")
                    sorted_selections = sorted(selections_for_game, key=lambda s: s.get('edge', Decimal('-Infinity')), reverse=True)

                selected_bet_for_game = None
                selection_key = None
                for candidate_sel in sorted_selections:
                    key = (fixture_id, candidate_sel['selection'])
                    if key not in used_selections:
                        selected_bet_for_game = candidate_sel
                        selection_key = key
                        logger.debug(f"PaperBuilder: Found unused selection '{key[1]}' for {key[0]} in combo.")
                        break # Found the best *unused* selection for this game

                if selected_bet_for_game is None:
                    logger.debug(f"PaperBuilder: Could not find any unused selection for fixture {fixture_id} in combo {game_combo}. Skipping combo.")
                    possible_to_build = False
                    break # Cannot build this paper combination
                else:
                    current_paper.append(selected_bet_for_game)
                    temp_selections_for_paper.append((selection_key, selected_bet_for_game)) # Store key and selection

            # If paper built successfully, add keys to used_selections
            if possible_to_build and len(current_paper) == size:
                 all_papers.append(current_paper)
                 for key, sel in temp_selections_for_paper:
                     used_selections.add(key)
                     logger.debug(f"PaperBuilder: Marked selection {key} as used.")
                 papers_built_for_size += 1
                 total_papers_built += 1
                 logger.debug(f"Successfully built paper #{total_papers_built} (Size {size})")

        logger.info(f"Built {papers_built_for_size} papers of size {size}.")

    logger.info(f"Total unique-selection papers built across all sizes: {total_papers_built}")
    return all_papers
# --- End of MODIFIED FUNCTION ---

# --- Modified Function: optimize_paper_stakes ---
def optimize_paper_stakes(paper_selections: List[Dict], kelly_fraction: Decimal, risk_aversion: Decimal, paper_id: str) -> Optional[Dict]:
    """
    Optimizes stakes, calculates stats, and assigns risk category based on average odds.
    """
    if not paper_selections: return None # Simplified check
    logger.info(f"--- Optimizing/Analyzing paper: {paper_id} ({len(paper_selections)} selections) ---")
    n_bets = len(paper_selections)

    # --- Prepare data & Basic Stats Calc ---
    try:
        probabilities = np.array([float(s['probability']) for s in paper_selections])
        odds_list = [s['odds'] for s in paper_selections]
        odds = np.array([float(o) for o in odds_list])
        edges_list = [s['edge'] for s in paper_selections]
        mu = np.array([float(e) for e in edges_list])

        # Calculate Basic Paper Stats
        avg_odds = sum(odds_list) / n_bets if n_bets > 0 else Decimal('NaN') # Used for Risk Category
        avg_prob = sum(s['probability'] for s in paper_selections) / n_bets if n_bets > 0 else Decimal('NaN')
        avg_edge = sum(edges_list) / n_bets if n_bets > 0 else Decimal('NaN')
        combined_odds = Decimal('1.0')
        max_odd_in_paper = Decimal('0.0')
        try:
            if odds_list:
                 max_odd_in_paper = max(odds_list)
                 for o in odds_list: combined_odds *= o
            else:
                combined_odds = max_odd_in_paper = Decimal('NaN')
        except InvalidOperation: combined_odds = Decimal('NaN')

        # --- Determine Risk Category based on AVERAGE odds ---
        risk_category = "Unknown"
        if avg_odds.is_nan():
             risk_category = "Unknown"
        elif avg_odds <= RISK_CATEGORY_THRESHOLDS["Low"]:
             risk_category = "Low"
        elif avg_odds <= RISK_CATEGORY_THRESHOLDS["Mid"]:
             risk_category = "Mid"
        else: # Avg odds > Mid threshold
             risk_category = "High"
        logger.debug(f"Paper {paper_id}: Avg Odd={avg_odds:.2f} -> Risk Category='{risk_category}'")

        # Covariance Matrix
        variances = probabilities * (odds - 1)**2 + (1 - probabilities) * (-1)**2 - mu**2
        variances[variances < 0] = 0
        Sigma = np.diag(variances)

    except Exception as e:
        logger.error(f"Paper {paper_id}: Error preparing data/stats: {e}", exc_info=True)
        return None

    # --- Setup cvxpy Optimization ---
    f = cp.Variable(n_bets, name="stake_fractions")
    gamma = float(risk_aversion)
    objective = cp.Maximize(mu @ f - (gamma / 2) * cp.quad_form(f, Sigma))
    constraints = [ f >= 0, cp.sum(f) <= float(kelly_fraction) ]
    problem = cp.Problem(objective, constraints)
    logger.debug(f"Paper {paper_id}: Solving optimization problem...")

    # --- Solve ---
    try:
        problem.solve(solver=OPTIMIZATION_SOLVER, verbose=False)
    except cp.SolverError as e:
         logger.warning(f"Paper {paper_id}: Solver {OPTIMIZATION_SOLVER} failed: {e}. Trying SCS.")
         try:
             problem.solve(solver=cp.SCS, verbose=False) # Explicitly try SCS as fallback
         except cp.SolverError as e2:
              logger.error(f"Paper {paper_id}: Fallback solver SCS failed: {e2}. Cannot optimize.")
              return None
         except Exception as ex:
             logger.error(f"Paper {paper_id}: Unexpected error during fallback solve: {ex}", exc_info=True)
             return None
    except Exception as e:
        logger.error(f"Paper {paper_id}: Unexpected error setting up/solving optimization: {e}", exc_info=True)
        return None

    # --- Process Results ---
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        logger.warning(f"Paper {paper_id}: Optimization failed. Status: {problem.status}.")
        return None
    if problem.status == cp.OPTIMAL_INACCURATE:
         logger.warning(f"Paper {paper_id}: Optimization solution inaccurate.")
    if f.value is None:
        logger.warning(f"Paper {paper_id}: Optimization status {problem.status} but f.value is None.")
        return None

    optimal_stakes = f.value
    staked_selections_output = []
    total_stake_allocated = Decimal('0.0')
    for i, sel in enumerate(paper_selections):
        stake = Decimal(str(optimal_stakes[i]))
        is_meaningful_stake = stake > MIN_STAKE_THRESHOLD
        staked_selections_output.append({**sel, "optimal_stake_fraction": stake, "has_meaningful_stake": is_meaningful_stake})
        if is_meaningful_stake: total_stake_allocated += stake

    if total_stake_allocated <= MIN_STAKE_THRESHOLD:
        logger.info(f"Paper {paper_id}: Negligible total stake ({total_stake_allocated:.6f}). Skipping.")
        return None

    # Calculate overall paper metrics
    paper_ev = sum(opt_sel['optimal_stake_fraction'] * opt_sel['edge'] for opt_sel in staked_selections_output)
    try:
        f_val_np = optimal_stakes
        if Sigma.shape == (n_bets, n_bets):
             paper_variance_float = f_val_np.T @ Sigma @ f_val_np
             paper_variance = Decimal(str(paper_variance_float))
             paper_std_dev = paper_variance.sqrt() if paper_variance >= 0 else Decimal('NaN')
             paper_sharpe = (paper_ev / paper_std_dev) if paper_std_dev and paper_std_dev > Decimal('0') else Decimal('0')
        else: paper_variance, paper_std_dev, paper_sharpe = Decimal('NaN'), Decimal('NaN'), Decimal('NaN')
    except Exception as e:
        logger.error(f"Paper {paper_id}: Error calculating variance/Sharpe: {e}")
        paper_variance, paper_std_dev, paper_sharpe = Decimal('NaN'), Decimal('NaN'), Decimal('NaN')

    logger.info(f"Paper {paper_id}: Optimization successful. Stake={total_stake_allocated:.4f}, EV={paper_ev:.4f}, Sharpe={paper_sharpe:.4f}")

    # Construct Output
    optimized_paper_data = {
        "paper_id": paper_id,
        "risk_category": risk_category, # Now based on avg_odds
        "optimization_status": problem.status,
        "settings_used": {"kelly_fraction": kelly_fraction, "risk_aversion": risk_aversion},
        "paper_summary": {
            "total_stake_fraction": total_stake_allocated, "number_of_selections": n_bets,
            "expected_value": paper_ev, "variance": paper_variance, "standard_deviation": paper_std_dev,
            "sharpe_ratio": paper_sharpe, "combined_odds": combined_odds, "average_odds": avg_odds,
            "average_probability": avg_prob, "average_edge": avg_edge,
            "max_odds_in_paper": max_odd_in_paper # Still useful info
        },
        "staked_selections": staked_selections_output # Includes venue/weather now
    }
    return optimized_paper_data

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates optimized multi-game betting papers.")
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE, help=f'Input batch prediction JSON file (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE, help=f'Output optimized papers JSON file (default: {DEFAULT_OUTPUT_FILE})')
    parser.add_argument('--min_edge', type=Decimal, default=DEFAULT_EDGE_THRESHOLD, help=f'Min edge threshold for selections (default: {DEFAULT_EDGE_THRESHOLD})')
    parser.add_argument('--kelly_fraction', type=Decimal, default=DEFAULT_KELLY_FRACTION, help=f'Max total stake fraction per paper (default: {DEFAULT_KELLY_FRACTION})')
    parser.add_argument('--risk_aversion', type=Decimal, default=DEFAULT_RISK_AVERSION, help=f'Risk aversion (gamma) for stake optimization (default: {DEFAULT_RISK_AVERSION})')
    parser.add_argument('--min_probability', type=Decimal, default=DEFAULT_MIN_PROBABILITY, help=f'Min probability threshold for selections (default: {DEFAULT_MIN_PROBABILITY})')
    parser.add_argument('--min_odds', type=Decimal, default=DEFAULT_MIN_ODDS, help=f'Optional min odds threshold for selections (default: None)')
    parser.add_argument('--max_odds', type=Decimal, default=DEFAULT_MAX_ODDS, help=f'Optional max odds threshold for selections (default: None)')
    parser.add_argument('--paper-sizes', type=int, nargs='+', default=DEFAULT_PAPER_SIZES, help=f'List of desired paper sizes (number of selections) (default: {DEFAULT_PAPER_SIZES})')
    parser.add_argument('--max-papers-per-size', type=int, default=DEFAULT_MAX_PAPERS_PER_SIZE, help=f'Maximum number of papers to generate per size (default: {DEFAULT_MAX_PAPERS_PER_SIZE})')
    parser.add_argument('--paper-build-strategy', type=str, default=DEFAULT_PAPER_BUILD_STRATEGY, choices=['highest_edge'], help=f'Strategy for selecting bets within games for papers (default: {DEFAULT_PAPER_BUILD_STRATEGY})')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()
    args.paper_sizes = sorted(list(set(args.paper_sizes)))

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
             handler.setLevel(logging.DEBUG)
        # Also set cvxpy logger level if needed? Requires cvxpy logging setup.
        logger.info("--- Debug logging enabled ---")
        # Set higher precision for numpy printing in debug mode
        np.set_printoptions(precision=8, suppress=True)


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

    # --- Main Processing Loop ---

    # 1. Load Data
    batch_prediction_data = load_data(input_filepath)
    if batch_prediction_data is None: sys.exit(1)


    # 2. Filter Selections per Game & Gather Data
    logger.info("--- Stage 1: Filtering Selections & Gathering Data per Game ---")
    all_filtered_selections_by_game = defaultdict(list)
    processed_fixture_ids = set()
    skipped_duplicates = 0
    games_with_no_valid_selections = 0
    for i, match_data in enumerate(batch_prediction_data):
        current_fixture_id = find_fixture_id(match_data)
        if not current_fixture_id:
             raw_data_nested = safe_get(match_data, ['raw_data'], {})
             current_fixture_id = find_fixture_id(raw_data_nested)
        if not current_fixture_id: continue
        if current_fixture_id in processed_fixture_ids:
             skipped_duplicates += 1
             continue
        processed_fixture_ids.add(current_fixture_id)

        # Calls the modified function that now gathers more data
        game_selections = filter_game_selections(
             match_data, current_fixture_id, args.min_edge, args.min_probability, args.min_odds, args.max_odds
         )
        if game_selections:
             all_filtered_selections_by_game[current_fixture_id] = game_selections
             logger.debug(f"Fixture {current_fixture_id}: Kept {len(game_selections)} selections.")
        else:
             logger.debug(f"Fixture {current_fixture_id}: No selections met filtering criteria.")
             games_with_no_valid_selections += 1
    logger.info(f"--- Filtering Summary ---")
    logger.info(f"Unique Fixtures Processed: {len(processed_fixture_ids)}")
    logger.info(f"Skipped Duplicate Fixtures: {skipped_duplicates}")
    logger.info(f"Games with >=1 Valid Selections: {len(all_filtered_selections_by_game)}")
    logger.info(f"Games with No Valid Selections: {games_with_no_valid_selections}")

    if not all_filtered_selections_by_game or len(all_filtered_selections_by_game) < min(args.paper_sizes):
        logger.warning("Not enough games with valid selections for minimum paper size. Exiting.")
        sys.exit(0)

    # 3. Build Papers (using the *modified* build_papers for uniqueness)
    papers_to_optimize = build_papers(
        all_filtered_selections_by_game,
        paper_sizes=args.paper_sizes,
        max_papers_per_size=args.max_papers_per_size,
        strategy=args.paper_build_strategy
    )
    if not papers_to_optimize:
        logger.warning("Paper building stage did not produce any papers. Exiting.")
        sys.exit(0)


    # 4. Optimize Stakes for Each Paper
    logger.info("--- Stage 3: Optimizing Stakes & Analyzing Papers ---")
    optimized_papers = []
    for i, paper in enumerate(papers_to_optimize):
        paper_id = f"Paper_{i+1}"
        optimized_result = optimize_paper_stakes(
            paper_selections=paper, # paper_selections now contain the rich data
            kelly_fraction=args.kelly_fraction,
            risk_aversion=args.risk_aversion,
            paper_id=paper_id
        )
        if optimized_result:
            optimized_papers.append(optimized_result)


    # 5. Rank Papers
    logger.info("--- Stage 4: Ranking Optimized Papers ---")
    if optimized_papers:
        risk_order = {"Low": 1, "Mid": 2, "High": 3, "Unknown": 4}
        def get_sort_key(paper):
            risk_cat = paper.get('risk_category', 'Unknown')
            sharpe = paper.get('paper_summary', {}).get('sharpe_ratio', Decimal('-Infinity'))
            sharpe_sort_val = sharpe if sharpe is not None and not sharpe.is_nan() else Decimal('-Infinity')
            return (risk_order.get(risk_cat, 4), -sharpe_sort_val)
        optimized_papers.sort(key=get_sort_key)
        logger.info(f"Ranked {len(optimized_papers)} papers by Risk Category (Low->High) then Sharpe Ratio (High->Low).")
    else:
         logger.warning("No optimized papers available for ranking.")


    # 6. Prepare Final Output Structure
    logger.info("--- Stage 5: Preparing Final Output ---")
    output_data = {
         "generation_info": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "input_file": os.path.relpath(input_filepath, script_dir),
            "output_file": os.path.relpath(output_filepath, script_dir),
            "settings": { # Capture all relevant settings used
                "filter_min_edge": args.min_edge,
                "filter_min_probability": args.min_probability,
                "filter_min_odds": args.min_odds,
                "filter_max_odds": args.max_odds,
                "staking_kelly_fraction": args.kelly_fraction,
                "staking_risk_aversion": args.risk_aversion,
                "paper_build_strategy": args.paper_build_strategy,
                "paper_sizes": args.paper_sizes,
                "max_papers_per_size": args.max_papers_per_size,
                "ensure_unique_selections_across_papers": True, # Added flag
                "risk_category_logic": f"AvgOdds: Low<={RISK_CATEGORY_THRESHOLDS['Low']}, Mid<={RISK_CATEGORY_THRESHOLDS['Mid']}", # Document logic
                "optimization_solver": OPTIMIZATION_SOLVER if OPTIMIZATION_SOLVER else "None",
                "covariance_assumption": "Independence (Diagonal Sigma)" # For staking step
            },
            "summary": {
                 "total_matches_input": len(batch_prediction_data),
                 "unique_fixtures_processed": len(processed_fixture_ids),
                 "skipped_duplicate_fixtures": skipped_duplicates,
                 "fixtures_with_valid_selections": len(all_filtered_selections_by_game),
                 "papers_built_attempted": len(papers_to_optimize), # How many unique papers were generated
                 "papers_successfully_staked_ranked": len(optimized_papers) # How many have final results
            }
        },
        "optimized_papers": optimized_papers # List of final papers, ranked
    }

    # 7. Write Output
    serializable_output = convert_for_json(output_data)
    logger.info(f"Writing {len(optimized_papers)} ranked optimized papers to: {output_filepath}")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_output, f, indent=4, ensure_ascii=False)
        logger.info("Successfully wrote optimized papers.")
    except IOError as e:
        logger.error(f"Error writing output file {output_filepath}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during file writing: {e}", exc_info=True)
        sys.exit(1)


    logger.info("Script finished.")
