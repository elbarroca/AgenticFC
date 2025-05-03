
import json
import os
import sys
import logging
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation, getcontext
from datetime import datetime, date
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set, Union
import itertools
import numpy as np
import cvxpy as cp
import matplotlib # Import base library first
# --- Add plotting utility import ---
from plotting_utils import plot_paper_scatter_mpl

# Set Decimal precision (important for calculations)
getcontext().prec = 28 # Standard precision

# --- Setup Logging ---
# Configure root logger or a specific logger for your application
logger = logging.getLogger(__name__) # Use __name__ for module-level logger
logger.setLevel(logging.INFO) # Default level
# Prevent adding handlers multiple times if the script/module is reloaded
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    # Consistent formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Constants & Default Settings ---

# Input/Output Defaults (Less critical for API, but useful for CLI wrapper)
DEFAULT_INPUT_FILE = os.path.join("..", "data", "output", "batch_prediction_results.json")
DEFAULT_OUTPUT_FILE = os.path.join("..", "data", "output", "optimized_betting_papers.json") # Renamed output

# Selection Filtering Defaults
DEFAULT_EDGE_THRESHOLD = Decimal('0.00')
DEFAULT_MIN_PROBABILITY = Decimal('0.50') # Adjusted default based on common usage
DEFAULT_MIN_ODDS = Decimal('1.8') # Min valid odds for *individual selections*
DEFAULT_MAX_ODDS = None # No upper limit by default for individual selections
DEFAULT_TOP_N_PER_GAME = 5 # Max selections to consider per game after initial filtering

# Efficiency Score Weights
DEFAULT_WEIGHT_PROB = Decimal('0.5')
DEFAULT_WEIGHT_EDGE = Decimal('0.3')
DEFAULT_WEIGHT_VALUE_RATIO = Decimal('0.2')

# Paper Generation Defaults
DEFAULT_PAPER_SIZES = [3] # Default to generating 3-selection papers
DEFAULT_MAX_PAPERS_PER_SIZE_GREEDY = 100 # Max papers for greedy strategy per size
DEFAULT_MAX_PAPERS_PER_SIZE_CVXPY = 1 # CVXPY finds *one* optimal paper per size
DEFAULT_PAPER_BUILD_STRATEGY = 'efficiency' # 'efficiency', 'highest_edge', 'highest_probability'
DEFAULT_USE_CVXPY = False # Keep Greedy as default

# CVXPY Constraint Defaults (Apply during CVXPY optimization)
DEFAULT_CVXPY_MAX_COMBINED_ODDS = None # No default constraint
DEFAULT_CVXPY_MIN_COMBINED_PROB = None # No default constraint

# Final Paper Filtering Defaults (Apply after generation)
DEFAULT_FILTER_LEAGUES = None # List of league names (case-insensitive)
DEFAULT_FILTER_TEAMS = None   # List of team names (case-insensitive)
DEFAULT_FILTER_MAX_COMBINED_ODDS = None # Post-filter max combined odds
DEFAULT_FILTER_MIN_COMBINED_PROB = None # Post-filter min combined prob
DEFAULT_FILTER_MIN_AVG_EDGE = None      # Post-filter min average edge
# Added filter for minimum combined odds (common requirement)
DEFAULT_FILTER_MIN_COMBINED_ODDS = Decimal('1.8') # Example: Minimum total odds for a valid paper

# Ranking Defaults
DEFAULT_RANKING_STRATEGY = 'combined_prob_then_edge' # 'combined_prob_then_edge' or 'avg_efficiency_score'

# Plotting Defaults (Updated for Matplotlib)
DEFAULT_ENABLE_PLOTTING = False
DEFAULT_PLOT_OUTPUT_DIR = os.path.join("..", "data", "output", "plots", "papers") # Directory for all paper plots
DEFAULT_PLOTS_TO_GENERATE = [
    {
        "filename": "risk_vs_efficiency.png",
        "x": "risk", "y": "paper_efficiency_score", "color": "num_legs", "size": "average_edge",
        "title": "Paper Efficiency vs Risk (Color: Legs, Size: Avg Edge)"
    },
    {
        "filename": "odds_vs_prob.png",
        "x": "combined_odds", "y": "combined_probability", "color": "num_legs", "size": "average_edge",
        "title": "Paper Combined Probability vs Combined Odds (Log Scale)",
        "x_log": True, # Add option for log scale
        "y_log": True  # Add option for log scale
     },
     {
         "filename": "avgedge_vs_avgprob.png",
         "x": "average_probability", "y": "average_edge", "color": "num_legs", "size": "paper_efficiency_score",
         "title": "Paper Average Edge vs Average Probability"
     }
]

# --- Helper Functions ---

def safe_get(data: Dict, keys: List[str], default: Any = None) -> Any:
    """Safely get a value from a nested dictionary."""
    # (Implementation as provided in the original code)
    for key in keys:
        try:
            if isinstance(data, dict):
                data = data[key]
            elif isinstance(data, list):
                try:
                    idx = int(key)
                    if 0 <= idx < len(data):
                        data = data[idx]
                    else:
                        return default
                except (ValueError, TypeError):
                    return default
            else:
                return default
        except (KeyError, IndexError, TypeError):
            return default
    return data

def get_nested_value(data: Dict, keys: List[str], default: Any = None) -> Any:
    """Wrapper around safe_get for consistency."""
    return safe_get(data, keys, default)

def parse_decimal(value: Any, context: str = "") -> Optional[Decimal]:
    """Safely parse a value to Decimal, handling strings, None, etc."""
    if value is None: return None
    if isinstance(value, Decimal): return value
    try:
        # Handle potential percentage strings, strip whitespace
        str_value = str(value).strip().replace('%', '')
        # Handle potential european decimal format (comma as decimal separator)
        # This is heuristic, assumes ',' is decimal if '.' is not present
        if '.' not in str_value and ',' in str_value:
            str_value = str_value.replace(',', '.')

        dec_val = Decimal(str_value)
        if dec_val.is_nan() or dec_val.is_infinite():
            logger.warning(f"Parsed '{value}' resulted in NaN or Infinity {context}. Returning None.")
            return None
        return dec_val
    except (InvalidOperation, ValueError, TypeError):
        logger.debug(f"Could not parse '{value}' as Decimal {context}. Returning None.")
        return None

def find_fixture_id(match_data: Dict) -> Optional[str]:
    """Attempts to find the fixture ID from various common locations."""
    # Prioritize standard keys first
    for key in ['fixture_id', 'id']:
        fixture_id = match_data.get(key)
        if fixture_id is not None: return str(fixture_id)

    # Check nested structures
    paths_to_check = [
        ['fixture_meta', 'id'],
        ['fixture', 'id'],
        ['info', 'fixture_id'],
        ['raw_data', 'home', 'basic_info', 'fixtureId'], # Example from other structures
        ['raw_data', 'fixture', 'id']
    ]
    for path in paths_to_check:
        fixture_id = safe_get(match_data, path)
        if fixture_id is not None: return str(fixture_id)

    logger.debug(f"Could not find fixture ID in structure: {list(match_data.keys())}")
    return None

def find_match_date(match_data: Dict) -> Optional[str]:
    """Attempts to find the match date (as ISO string)."""
    # Check primary sources first
    paths_and_types = [
        (['fixture_meta', 'date_utc'], 'string'), # ISO string expected
        (['fixture', 'date'], 'string'),
        (['info', 'date'], 'string'),
        (['match_date'], 'string'),
        (['raw_data', 'fixture', 'date'], 'string'),
        (['match_info', 'date'], 'string'),
        (['weather_forecast', 'dt_txt'], 'string'), # Usually YYYY-MM-DD HH:MM:SS
        # Timestamps
        (['fixture_meta', 'timestamp_utc'], 'timestamp'),
        (['fixture', 'timestamp'], 'timestamp'),
        (['info', 'timestamp'], 'timestamp'),
        (['raw_data', 'fixture', 'timestamp'], 'timestamp'),
    ]

    for path, data_type in paths_and_types:
        value = safe_get(match_data, path)
        if value is not None:
            if data_type == 'string':
                # Basic validation or attempt parsing if format is known
                try:
                    # Attempt to parse common formats to ensure validity and standardize
                    dt_obj = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
                    return dt_obj.isoformat() + "Z"
                except ValueError:
                     # If parsing fails, return the string as is, but log a warning
                     logger.debug(f"Date string '{value}' at path {path} is not standard ISO format.")
                     return str(value) # Return original string
            elif data_type == 'timestamp':
                try:
                    # Assume UTC timestamp
                    ts = int(value)
                    dt_obj = datetime.utcfromtimestamp(ts)
                    return dt_obj.isoformat() + "Z"
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse timestamp '{value}' at path {path}.")
                    continue # Try next path
    return None


def find_team_names(match_data: Dict) -> Tuple[str, str]:
    """Finds home and away team names from various locations."""
    home_name = None
    away_name = None

    # Priority: 'teams' structure
    home_name = safe_get(match_data, ['teams', 'home', 'name'])
    away_name = safe_get(match_data, ['teams', 'away', 'name'])
    if home_name and away_name: return str(home_name).strip(), str(away_name).strip()

    # Next: 'raw_data' detailed structure
    home_name = safe_get(match_data, ['raw_data', 'home', 'basic_info', 'name'])
    away_name = safe_get(match_data, ['raw_data', 'away', 'basic_info', 'name'])
    if home_name and away_name: return str(home_name).strip(), str(away_name).strip()

    # Next: Simple top-level keys
    home_name = match_data.get('home_team')
    away_name = match_data.get('away_team')
    if home_name and away_name: return str(home_name).strip(), str(away_name).strip()

    # Fallback: Try 'fixture' structure
    home_name = safe_get(match_data, ['fixture', 'teams', 'home', 'name'])
    away_name = safe_get(match_data, ['fixture', 'teams', 'away', 'name'])
    if home_name and away_name: return str(home_name).strip(), str(away_name).strip()

    # Fallback: Generic names
    return "Home Team", "Away Team"

def find_league_details(match_data: Dict) -> Dict:
    """Extract detailed league information."""
    league_info = { "name": "Unknown League", "country": None, "season": None, "logo": None }

    # Check multiple paths, prioritizing more specific ones
    paths = [
        ['league'], # Simple top-level 'league' object
        ['fixture_meta', 'league'],
        ['raw_data', 'home', 'match_processor_snapshot', 'league'], # Very specific path
        ['raw_data', 'fixture', 'league'],
        ['info'] # Check if info contains league details directly
    ]

    for path in paths:
        league_data = safe_get(match_data, path)
        if isinstance(league_data, dict):
            name = league_data.get('name')
            country = league_data.get('country')
            season = league_data.get('season')
            logo = league_data.get('logo')
            # Update if found and not already set to something more specific
            if name and league_info["name"] == "Unknown League": league_info["name"] = str(name)
            if country and not league_info["country"]: league_info["country"] = str(country)
            if season and not league_info["season"]: league_info["season"] = str(season)
            if logo and not league_info["logo"]: league_info["logo"] = str(logo)

    # If name still unknown, try simpler keys
    if league_info["name"] == "Unknown League":
        simple_name = match_data.get('league_name') or safe_get(match_data, ['info', 'league_name'])
        if simple_name:
            league_info["name"] = str(simple_name)

    return league_info


def find_venue_info(match_data: Dict) -> Optional[str]:
    """Attempts to find venue name and city."""
    venue_name = None
    city_name = None

    paths = [
        ['fixture_meta', 'venue'],
        ['fixture', 'venue'],
        ['raw_data', 'fixture', 'venue'],
        ['match_info'] # Check if match_info has venue details
    ]

    for path in paths:
        venue_data = safe_get(match_data, path)
        if isinstance(venue_data, dict):
            if not venue_name: venue_name = venue_data.get('name')
            if not city_name: city_name = venue_data.get('city')

    # Try simple top-level keys if not found
    if not venue_name: venue_name = match_data.get('venue_name') or safe_get(match_data, ['venue', 'name'])
    if not city_name: city_name = match_data.get('venue_city') or safe_get(match_data, ['venue', 'city'])

    # Check weather forecast city as a last resort for city
    if not city_name:
        city_name = safe_get(match_data, ['weather_forecast', 'city', 'name']) \
                 or safe_get(match_data, ['fixture_meta', 'weather_forecast', 'city', 'name'])

    # Construct the string
    if venue_name and city_name and str(venue_name).lower() != str(city_name).lower():
        return f"{str(venue_name).strip()}, {str(city_name).strip()}"
    elif venue_name:
        return str(venue_name).strip()
    elif city_name:
        return f"{str(city_name).strip()} Stadium" # Fallback if only city is known
    else:
        return None


def find_weather_info(match_data: Dict) -> Tuple[Optional[str], Optional[Decimal], Optional[str]]:
    """
    Extract essential weather forecast information (summary, temp, condition).
    Returns (summary, temperature_celsius, condition)
    """
    summary, temp_c, condition = None, None, None
    weather_data = None

    paths = [
        ['weather_forecast'],
        ['fixture_meta', 'weather_forecast'],
        ['raw_data', 'fixture', 'weather_forecast']
    ]

    for path in paths:
        weather_data = safe_get(match_data, path)
        if weather_data and isinstance(weather_data, dict):
            break # Found weather data

    if not weather_data:
        return None, None, None

    # Extract specific fields
    summary = weather_data.get('summary') or weather_data.get('description')
    condition = safe_get(weather_data, ['weather', 0, 'description']) or \
                safe_get(weather_data, ['weather', 0, 'main']) or \
                weather_data.get('condition')

    # Temperature: Look for common keys, try to parse as Decimal
    temp_val = safe_get(weather_data, ['main', 'temp']) or \
               safe_get(weather_data, ['main', 'temperature']) or \
               weather_data.get('temperature') or weather_data.get('temp')

    if temp_val is not None:
        temp_c = parse_decimal(temp_val, context="weather temperature")
        # Basic check for likely Kelvin -> Celsius conversion if temp > 100
        if temp_c is not None and temp_c > Decimal('100'):
            temp_c -= Decimal('273.15')

    # If summary is missing but we have details, create one
    if not summary and (condition or temp_c is not None):
        parts = []
        if condition: parts.append(str(condition).title())
        if temp_c is not None: parts.append(f"{temp_c:.1f}°C")
        summary = ", ".join(parts)

    return summary, temp_c, condition

def find_match_details(match_data: Dict) -> Dict:
    """Extracts additional match details like referee and status."""
    details = {"referee": None, "status": None}

    paths = [
        ['fixture_meta'],
        ['fixture'],
        ['raw_data', 'fixture'],
        ['match_info']
    ]

    for path in paths:
        data_subset = safe_get(match_data, path)
        if isinstance(data_subset, dict):
            if not details["referee"]:
                details["referee"] = data_subset.get('referee')
            if not details["status"]:
                status_obj = data_subset.get('status')
                if isinstance(status_obj, dict):
                    # Prefer long status, fallback to short
                    details["status"] = status_obj.get('long', status_obj.get('short'))
                elif isinstance(status_obj, str): # Handle if status is just a string
                    details["status"] = status_obj

    # Check top-level simple keys if not found
    if not details["referee"]: details["referee"] = match_data.get('referee')
    if not details["status"]: details["status"] = match_data.get('status')

    # Clean up None values if needed
    details = {k: v for k, v in details.items() if v is not None}

    return details


def convert_for_json(data: Any) -> Any:
    """Recursively convert Decimal/numpy/datetime types for JSON serialization."""
    if isinstance(data, list):
        return [convert_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_for_json(v) for k, v in data.items()}
    elif isinstance(data, Decimal):
        if data.is_nan(): return 'NaN'
        if data.is_infinite(): return 'Infinity' if data > 0 else '-Infinity'
        # Standardize output format (e.g., 4 decimal places)
        return f"{data.quantize(Decimal('0.0001'), ROUND_HALF_UP):f}" # Use :f to avoid scientific notation
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        if np.isnan(data): return 'NaN'
        if np.isinf(data): return 'Infinity' if data > 0 else '-Infinity'
        # Format float with reasonable precision
        return float(f"{data:.6f}")
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist() # Convert numpy arrays to lists
    elif hasattr(data, 'tolist'): # Handle other potential numpy types
        try:
            return data.tolist()
        except AttributeError: pass # If tolist fails, fall through
    # Check for sets and convert to sorted lists
    elif isinstance(data, set):
        try:
            # Attempt to sort if elements are comparable, otherwise just list
            return sorted(list(data))
        except TypeError:
            return list(data)
    # Basic types and None are fine
    elif isinstance(data, (str, int, float, bool)) or data is None:
        return data
    else:
        # Fallback for unknown types: convert to string
        logger.debug(f"Converting unknown type {type(data)} to string for JSON.")
        return str(data)


# --- Core Logic ---

def load_data(filepath: str) -> Optional[List[Dict]]:
    """Loads match data from JSON file, handling dict or list root."""
    logger.info(f"Loading match data from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Use Decimal for float parsing during load
            data = json.load(f, parse_float=Decimal, parse_int=Decimal)
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error: Could not decode JSON from {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {filepath}: {e}", exc_info=True)
        return None

    # Handle if the root is a dictionary (e.g., keyed by fixture_id)
    if isinstance(data, dict):
        logger.info("Input data is a dictionary; converting to a list of matches.")
        processed_data = []
        for key, match_info in data.items():
            if isinstance(match_info, dict):
                # Attempt to inject the key as fixture_id if missing
                if 'fixture_id' not in match_info and find_fixture_id(match_info) is None:
                    try:
                        # Check if key looks like an ID
                        _ = int(key) # Test if it can be int
                        match_info['fixture_id'] = str(key)
                        logger.debug(f"Injected dictionary key '{key}' as fixture_id.")
                    except (ValueError, TypeError):
                        logger.warning(f"Dictionary key '{key}' is not numeric; cannot reliably use as fixture ID if missing.")
                processed_data.append(match_info)
            else:
                logger.warning(f"Skipping item with key '{key}': Value is not a dictionary.")
        data = processed_data

    if not isinstance(data, list):
        logger.error(f"Failed to process input file {filepath} into a list of match data.")
        return None

    logger.info(f"Successfully loaded/processed {len(data)} match entries.")
    return data


def calculate_efficiency_score(selection: Dict, weights: Dict) -> Optional[Decimal]:
    """Calculates the composite efficiency score for a single selection."""
    prob = selection.get('probability')
    edge = selection.get('edge')
    odds = selection.get('odds')

    # Validate inputs are valid Decimals
    if not all(isinstance(val, Decimal) and not val.is_nan() for val in [prob, edge, odds]):
        logger.debug(f"Skipping efficiency score for selection {selection.get('selection', 'N/A')}: Invalid numeric data (Prob: {prob}, Edge: {edge}, Odds: {odds}).")
        return None
    # Further sanity checks
    if odds <= Decimal('1.0') or prob <= Decimal('0.0') or prob > Decimal('1.0'):
         logger.debug(f"Skipping efficiency score for selection {selection.get('selection', 'N/A')}: Out of bounds (Prob: {prob}, Odds: {odds}).")
         return None

    try:
        # Value Ratio = Probability * Odds. A value > 1 indicates potential value.
        value_ratio = prob * odds
        w_prob = weights.get('weight_prob', DEFAULT_WEIGHT_PROB)
        w_edge = weights.get('weight_edge', DEFAULT_WEIGHT_EDGE)
        w_val_ratio = weights.get('weight_value_ratio', DEFAULT_WEIGHT_VALUE_RATIO)

        # Ensure weights are Decimal
        w_prob = parse_decimal(w_prob, "weight_prob") or Decimal('0.0')
        w_edge = parse_decimal(w_edge, "weight_edge") or Decimal('0.0')
        w_val_ratio = parse_decimal(w_val_ratio, "weight_value_ratio") or Decimal('0.0')

        # Normalize weights if they don't sum to 1 (optional but good practice)
        total_weight = w_prob + w_edge + w_val_ratio
        if total_weight <= 0: total_weight = Decimal('1.0') # Avoid division by zero
        w_prob /= total_weight
        w_edge /= total_weight
        w_val_ratio /= total_weight

        # Score calculation: Weighted sum of components
        # Edge is used directly. Value Ratio is compared to 1 (break-even point).
        score = (w_prob * prob +
                 w_edge * edge +
                 w_val_ratio * (value_ratio - Decimal('1.0')))

        return score
    except (TypeError, InvalidOperation) as e:
        logger.warning(f"Error calculating efficiency score for selection {selection.get('selection', 'N/A')}: {e}")
        return None


def filter_and_score_game_selections(
    match_data: Dict,
    fixture_id: str,
    min_edge: Decimal,
    min_probability: Decimal,
    min_odds: Optional[Decimal],
    max_odds: Optional[Decimal],
    top_n_per_game: int,
    efficiency_weights: Dict
) -> List[Dict]:
    """
    Filters individual selections, calculates efficiency score, adds context, sorts, and trims to top N.
    """
    valid_selections_with_score = []

    # --- Extract Match Context (Do this once per game) ---
    match_date = find_match_date(match_data)
    home_name, away_name = find_team_names(match_data)
    match_desc = f"{home_name} vs {away_name}"
    league_info = find_league_details(match_data)
    venue_info = find_venue_info(match_data)
    weather_summary, weather_temp, weather_cond = find_weather_info(match_data)
    match_details = find_match_details(match_data) # Referee, Status

    # --- Find Selections List ---
    selections_list = None
    paths_to_check = [
        ['match_analysis', 'top_n_combined_selections'],
        ['top_n_combined_selections'] # Check top level as fallback
    ]
    for path in paths_to_check:
        selections_list = safe_get(match_data, path)
        if selections_list and isinstance(selections_list, list):
            break # Found a valid list

    if not selections_list:
        logger.debug(f"Fixture {fixture_id}: No 'top_n_combined_selections' list found.")
        return []

    logger.debug(f"Fixture {fixture_id} ({match_desc}): Processing {len(selections_list)} potential selections...")

    processed_count = 0
    for sel_data in selections_list:
        if not isinstance(sel_data, dict):
            logger.debug(f"Fixture {fixture_id}: Skipping non-dict item in selections list.")
            continue
        processed_count += 1

        # --- Parse Selection Data ---
        selection_name = sel_data.get("selection")
        prob = parse_decimal(sel_data.get("probability"), context=f"prob {fixture_id}/{selection_name}")
        odd = parse_decimal(sel_data.get("odd"), context=f"odd {fixture_id}/{selection_name}")
        edge = parse_decimal(sel_data.get("edge"), context=f"edge {fixture_id}/{selection_name}")

        # --- Apply Core Filters ---
        if selection_name is None or prob is None or odd is None or edge is None:
            logger.debug(f"Fixture {fixture_id}: Skipping selection '{selection_name}': Missing core data (Prob: {prob}, Odd: {odd}, Edge: {edge}).")
            continue
        if edge < min_edge:
            logger.debug(f"Fixture {fixture_id}: Skipping '{selection_name}': Edge {edge} < {min_edge}.")
            continue
        if prob < min_probability:
            logger.debug(f"Fixture {fixture_id}: Skipping '{selection_name}': Prob {prob} < {min_probability}.")
            continue
        # Apply individual odd limits
        if min_odds is not None and odd < min_odds:
            logger.debug(f"Fixture {fixture_id}: Skipping '{selection_name}': Odd {odd} < {min_odds}.")
            continue
        if max_odds is not None and odd > max_odds:
            logger.debug(f"Fixture {fixture_id}: Skipping '{selection_name}': Odd {odd} > {max_odds}.")
            continue
        # Check bounds validity
        if odd <= Decimal('1.0') or prob <= Decimal('0.0') or prob > Decimal('1.0'):
            logger.debug(f"Fixture {fixture_id}: Skipping '{selection_name}': Invalid bounds (Odd: {odd}, Prob: {prob}).")
            continue

        # --- Passed Filters: Create Enriched Dict & Calculate Score ---
        selection_details = {
            "fixture_id": fixture_id,
            "match_description": match_desc,
            "match_date": match_date,
            "league_name": league_info.get("name"),
            "league_country": league_info.get("country"),
            "league_season": league_info.get("season"),
            "league_logo": league_info.get("logo"), # Add logo if available
            "home_team_name": home_name,
            "away_team_name": away_name,
            # Add team logos if available (requires helper function extension if needed)
            # "home_team_logo": find_team_logo(match_data, 'home'),
            # "away_team_logo": find_team_logo(match_data, 'away'),
            "venue": venue_info,
            "weather_summary": weather_summary,
            "weather_temperature_celsius": weather_temp, # Store parsed Decimal
            "weather_condition": weather_cond,
            "referee": match_details.get("referee"),
            "match_status": match_details.get("status"),
            # Core bet info
            "selection": selection_name,
            "probability": prob,
            "odds": odd,
            "edge": edge,
            "odd_source": sel_data.get("odd_source", "Unknown"),
            "efficiency_score": None # Placeholder
        }

        # Calculate efficiency score
        efficiency_score = calculate_efficiency_score(selection_details, efficiency_weights)

        if efficiency_score is not None:
             selection_details["efficiency_score"] = efficiency_score
             valid_selections_with_score.append(selection_details)
        else:
             logger.debug(f"Fixture {fixture_id}: Could not calculate efficiency score for selection '{selection_name}'. Skipping.")

    # --- Sort by Efficiency Score (Highest First) ---
    # Handle potential None scores by sorting them last
    valid_selections_with_score.sort(
        key=lambda s: s.get('efficiency_score', Decimal('-Infinity')),
        reverse=True
    )

    # --- Trim to Top-N per Game ---
    trimmed_selections = valid_selections_with_score[:top_n_per_game]

    logger.debug(f"Fixture {fixture_id}: Processed {processed_count} selections. Found {len(valid_selections_with_score)} valid. Kept top {len(trimmed_selections)} based on efficiency.")
    return trimmed_selections


def build_papers_greedy(filtered_selections_by_game: Dict[str, List[Dict]],
                        paper_sizes: List[int],
                        max_papers_per_size: int,
                        strategy: str,
                        filter_teams_normalized: Optional[Set[str]] = None # Pass normalized set
                        ) -> List[List[Dict]]:
    """
    Builds betting paper combinations using a greedy strategy.
    Picks the 'best' available selection per game based on the strategy for combinations.
    Ensures each specific selection (fixture_id, selection_name) is used at most once across all generated papers.
    Optionally ensures generated papers include at least one fixture involving a target team.
    """
    all_papers = []
    used_selections: Set[Tuple[str, str]] = set() # Tracks (fixture_id, selection_name)
    available_fixture_ids = list(filtered_selections_by_game.keys())
    num_available_games = len(available_fixture_ids)
    min_req_size = min(paper_sizes) if paper_sizes else 1

    logger.info(f"--- Building Paper Combinations (Greedy Strategy: '{strategy}') ---")
    if filter_teams_normalized: logger.info(f"Applying team filter (normalized): {filter_teams_normalized}")
    logger.info(f"Target paper sizes: {paper_sizes}")
    logger.info(f"Max papers per size: {max_papers_per_size}")
    logger.info(f"Games available with valid selections: {num_available_games}")

    if not paper_sizes or num_available_games < min_req_size:
         logger.warning(f"Not enough games ({num_available_games}) with valid selections to build minimum size {min_req_size} papers.")
         return []

    # Identify target games IF filtering by team
    target_fixture_ids = set()
    if filter_teams_normalized:
        for fixture_id, selections in filtered_selections_by_game.items():
            if not selections: continue
            # Use normalized names from the first selection (consistent per fixture)
            norm_home = str(selections[0].get('home_team_name', '')).lower()
            norm_away = str(selections[0].get('away_team_name', '')).lower()
            if norm_home in filter_teams_normalized or norm_away in filter_teams_normalized:
                target_fixture_ids.add(fixture_id)
        logger.info(f"Identified {len(target_fixture_ids)} target fixtures matching team filter.")
        if not target_fixture_ids:
             logger.warning("No available games involve the specified filter teams. No papers can be generated with this team filter.")
             return []

    total_papers_built_overall = 0
    for size in sorted(paper_sizes):
        if size <= 0: continue
        # Check if enough games are available overall
        if size > num_available_games:
            logger.info(f"Skipping paper size {size}: Requires {size} games, only {num_available_games} available.")
            continue
        # Check if enough *target* games are available if filtering
        if filter_teams_normalized and size > len(target_fixture_ids):
            logger.info(f"Skipping paper size {size}: Requires {size} games involving filter teams, only {len(target_fixture_ids)} target fixtures available.")
            continue

        logger.info(f"Generating combinations for paper size {size}...")
        # Generate combinations of fixture IDs
        game_combinations = itertools.combinations(available_fixture_ids, size)
        papers_built_for_size = 0

        # Limit the number of combinations processed if needed (heuristic for performance)
        # Note: This makes the selection less exhaustive but faster.
        # combinations_to_process = list(game_combinations) # Materialize if needed, but potentially large
        # if len(combinations_to_process) > MAX_COMBINATIONS_TO_CHECK:
        #    logger.warning(f"Limiting combinations checked for size {size} to {MAX_COMBINATIONS_TO_CHECK}")
        #    combinations_to_process = combinations_to_process[:MAX_COMBINATIONS_TO_CHECK] # Or sample

        for game_combo in game_combinations:
            # --- Team Filter Check (if applicable) ---
            if filter_teams_normalized and not any(fix_id in target_fixture_ids for fix_id in game_combo):
                logger.debug(f"Greedy: Skipping combo {game_combo} - no target fixture included.")
                continue

            current_paper_selections = []
            possible_to_build = True
            temp_selection_keys_for_paper: List[Tuple[str, str]] = []

            # --- Try to build a paper for this combination ---
            for fixture_id in game_combo:
                selections_for_game = filtered_selections_by_game.get(fixture_id)
                if not selections_for_game: # Should not happen if available_fixture_ids is correct
                    logger.error(f"Internal Error: Fixture {fixture_id} in combo but has no selections. Skipping combo.")
                    possible_to_build = False
                    break

                # Sort candidate selections *within this game* based on the chosen strategy
                if strategy == 'highest_edge':
                    sort_key = lambda s: s.get('edge', Decimal('-Infinity'))
                elif strategy == 'highest_probability':
                    sort_key = lambda s: s.get('probability', Decimal('-Infinity'))
                else: # Default to 'efficiency'
                    if strategy != 'efficiency': logger.warning(f"Unknown build strategy '{strategy}', defaulting to 'efficiency'.")
                    sort_key = lambda s: s.get('efficiency_score', Decimal('-Infinity'))

                sorted_candidates = sorted(selections_for_game, key=sort_key, reverse=True)

                # Find the first available (unused) selection from the sorted candidates
                selected_bet_for_game = None
                selection_key: Optional[Tuple[str, str]] = None
                for candidate_sel in sorted_candidates:
                    key = (fixture_id, candidate_sel['selection'])
                    if key not in used_selections:
                        selected_bet_for_game = candidate_sel
                        selection_key = key
                        break # Found the best available one for this game

                if selected_bet_for_game is None or selection_key is None:
                    # Could not find an unused selection for this game in the combo
                    logger.debug(f"Greedy: No unused selection available for fixture {fixture_id} in combo {game_combo} (strategy: {strategy}). Skipping combo.")
                    possible_to_build = False
                    break
                else:
                    # Add the chosen selection and its key temporarily
                    current_paper_selections.append(selected_bet_for_game)
                    temp_selection_keys_for_paper.append(selection_key)

            # --- Finalize Paper if Possible ---
            if possible_to_build and len(current_paper_selections) == size:
                # Successfully built a paper of the target size
                all_papers.append(current_paper_selections)
                # Mark these selections as used *globally*
                for key in temp_selection_keys_for_paper:
                    used_selections.add(key)
                papers_built_for_size += 1
                total_papers_built_overall += 1
                logger.debug(f"Greedy: Built paper #{total_papers_built_overall} (Size {size}) using combo {game_combo}")

                # Check if max papers for this size is reached
                if papers_built_for_size >= max_papers_per_size:
                    logger.info(f"Greedy: Reached max papers ({max_papers_per_size}) for size {size}. Stopping generation for this size.")
                    break # Stop processing combinations for this size

        logger.info(f"Greedy: Built {papers_built_for_size} papers of size {size}.")
        # Check overall limit if needed (though max_papers_per_size is usually the intended limit)
        # if max_papers_overall and total_papers_built_overall >= max_papers_overall:
        #    logger.info("Reached overall max paper limit. Stopping.")
        #    break

    logger.info(f"Greedy: Total papers generated across all sizes: {total_papers_built_overall}")
    return all_papers


def build_papers_cvxpy(
    all_filtered_selections: List[Dict], # Flat list of all valid, scored selections
    paper_sizes: List[int],
    max_combined_odds: Optional[Decimal],
    min_combined_prob: Optional[Decimal],
    efficiency_weights: Dict, # Needed to log objective function basis
    debug_mode: bool = False
) -> List[List[Dict]]:
    """
    Builds the 'best' paper for each size using CVXPY binary optimization.
    Maximizes the sum of efficiency scores subject to constraints.
    """
    all_optimal_papers = []
    num_selections = len(all_filtered_selections)
    min_req_size = min(paper_sizes) if paper_sizes else 1

    logger.info(f"--- Building Paper Combinations (CVXPY Optimization) ---")
    logger.info(f"Optimizing over {num_selections} potential selections.")
    logger.info(f"Target paper sizes: {paper_sizes}")
    if max_combined_odds: logger.info(f"Constraint: Max Combined Odds <= {max_combined_odds}")
    if min_combined_prob: logger.info(f"Constraint: Min Combined Probability >= {min_combined_prob}")

    if num_selections == 0 or num_selections < min_req_size:
        logger.warning(f"CVXPY: Not enough selections ({num_selections}) available to build minimum size {min_req_size} papers.")
        return []

    # --- Prepare data for CVXPY ---
    # Extract necessary data, ensuring consistency and handling potential issues
    indices_map = {} # Map original index to filtered index
    clean_selections = []
    efficiency_scores_raw = []
    odds_raw = []
    probabilities_raw = []
    fixture_ids_raw = []

    for i, sel in enumerate(all_filtered_selections):
        eff_score = sel.get('efficiency_score')
        odds = sel.get('odds')
        prob = sel.get('probability')
        fix_id = sel.get('fixture_id')

        # Validate data needed for optimization
        if fix_id is None or not all(isinstance(v, Decimal) and not v.is_nan() for v in [eff_score, odds, prob]):
            logger.warning(f"CVXPY: Skipping selection index {i} due to invalid data (Score: {eff_score}, Odds: {odds}, Prob: {prob}, FixID: {fix_id}).")
            continue
        # Add log-related checks
        if odds <= Decimal('1.0') and max_combined_odds is not None:
             logger.warning(f"CVXPY: Skipping selection index {i} (Fixture {fix_id}, Sel: {sel.get('selection')}): Odds {odds} <= 1.0, incompatible with max_odds constraint.")
             continue
        if prob <= Decimal('0.0') and min_combined_prob is not None:
             logger.warning(f"CVXPY: Skipping selection index {i} (Fixture {fix_id}, Sel: {sel.get('selection')}): Prob {prob} <= 0.0, incompatible with min_prob constraint.")
             continue


        # Passed validation, add to lists
        current_clean_index = len(clean_selections)
        indices_map[current_clean_index] = i # Map clean index back to original all_filtered_selections index
        clean_selections.append(sel) # Store the full selection dict
        efficiency_scores_raw.append(float(eff_score)) # CVXPY prefers float
        odds_raw.append(float(odds))
        probabilities_raw.append(float(prob))
        fixture_ids_raw.append(fix_id)

    num_valid_selections = len(clean_selections)
    if num_valid_selections < min_req_size:
         logger.warning(f"CVXPY: Only {num_valid_selections} selections remain after validation. Not enough for minimum paper size {min_req_size}.")
         return []

    logger.info(f"Proceeding with {num_valid_selections} valid selections for optimization.")

    # Convert to NumPy arrays
    efficiency_scores = np.array(efficiency_scores_raw)
    odds = np.array(odds_raw)
    probabilities = np.array(probabilities_raw)

    # Map fixture IDs to *clean* indices for constraints
    fixtures_map = defaultdict(list)
    for i, fix_id in enumerate(fixture_ids_raw):
        fixtures_map[fix_id].append(i)

    # --- Optimization Loop per Size ---
    for size in sorted(paper_sizes):
        if size <= 0: continue
        if size > num_valid_selections:
            logger.info(f"CVXPY: Skipping paper size {size}: Not enough valid selections ({num_valid_selections}).")
            continue

        logger.info(f"Optimizing for paper size {size}...")

        # Define optimization variable (binary: 1 if selection is chosen, 0 otherwise)
        x = cp.Variable(num_valid_selections, boolean=True)

        # Define objective function: Maximize sum of efficiency scores
        objective = cp.Maximize(x @ efficiency_scores)

        # Define constraints
        constraints = []
        # 1. Paper size constraint: Sum of chosen selections must equal target size
        constraints.append(cp.sum(x) == size)

        # 2. One selection per fixture constraint
        for fix_id, indices in fixtures_map.items():
            if len(indices) > 1: # Only add constraint if multiple selections exist for the fixture
                constraints.append(cp.sum(x[indices]) <= 1)

        # 3. Max combined odds constraint (using logarithms for linearity)
        if max_combined_odds is not None:
             try:
                 log_odds = np.log(odds) # odds are already validated > 1.0 if constraint is active
                 log_max_odds = np.log(float(max_combined_odds))
                 constraints.append(cp.sum(cp.multiply(x, log_odds)) <= log_max_odds)
             except (ValueError, TypeError, FloatingPointError) as e:
                 logger.error(f"CVXPY: Error preparing max_combined_odds constraint (Value: {max_combined_odds}): {e}. Skipping constraint.")

        # 4. Min combined probability constraint (using logarithms)
        if min_combined_prob is not None:
             try:
                 # Need to handle probabilities near zero carefully if log is used
                 # Filter out zero probabilities if constraint is active (already done above)
                 log_probabilities = np.log(probabilities) # probs already validated > 0.0
                 log_min_prob = np.log(float(min_combined_prob))
                 constraints.append(cp.sum(cp.multiply(x, log_probabilities)) >= log_min_prob)
             except (ValueError, TypeError, FloatingPointError) as e:
                 logger.error(f"CVXPY: Error preparing min_combined_prob constraint (Value: {min_combined_prob}): {e}. Skipping constraint.")


        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        try:
             # Choose a suitable MIP solver (CBC or GLPK_MI are common open-source options)
             solver_options = {}
             selected_solver = None
             if cp.CBC in cp.installed_solvers():
                 selected_solver = cp.CBC
             elif cp.GLPK_MI in cp.installed_solvers():
                 selected_solver = cp.GLPK_MI
             # Add commercial solvers if available and preferred
             # elif cp.GUROBI in cp.installed_solvers(): selected_solver = cp.GUROBI
             # elif cp.MOSEK in cp.installed_solvers(): selected_solver = cp.MOSEK
             else:
                  logger.warning("CVXPY: No preferred MIP solver (CBC, GLPK_MI) found. Performance/results may vary. Consider installing one: 'pip install cvxpy[standard]'.")
                  # Let CVXPY choose default if none are explicitly found

             # Solve the problem
             problem.solve(solver=selected_solver, verbose=debug_mode) # Pass debug flag for solver output

             # --- Process Results ---
             if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                 # Get the indices of the selected items (where x.value is close to 1)
                 selected_indices_clean = np.where(x.value > 0.9)[0] # Use a threshold close to 1
                 selected_indices_clean = selected_indices_clean.astype(int) # Ensure integer indices


                 if len(selected_indices_clean) == size:
                     # Map clean indices back to the original indices in all_filtered_selections
                     original_indices = [indices_map[idx] for idx in selected_indices_clean]
                     # Retrieve the actual selection dictionaries
                     optimal_paper = [all_filtered_selections[orig_idx] for orig_idx in original_indices]

                     all_optimal_papers.append(optimal_paper)
                     logger.info(f"CVXPY: Found optimal paper of size {size} with objective value {problem.value:.4f}")
                 else:
                      # Should not happen if OPTIMAL, but check just in case
                      logger.warning(f"CVXPY: Solver status is {problem.status} but found {len(selected_indices_clean)} selections (expected {size}). Value: {x.value}. Discarding result for size {size}.")

             elif problem.status == cp.INFEASIBLE:
                  logger.warning(f"CVXPY: Problem is INFEASIBLE for size {size}. No combination satisfies all constraints (check odds/prob limits).")
             elif problem.status == cp.UNBOUNDED:
                  logger.warning(f"CVXPY: Problem is UNBOUNDED for size {size}. This suggests an issue with the objective or constraints.")
             else: # Other statuses (solver error, etc.)
                  logger.warning(f"CVXPY: Optimization failed or did not find an optimal solution for size {size}. Status: {problem.status}")

        except cp.SolverError as e:
             logger.error(f"CVXPY: Solver error occurred for size {size}: {e}")
        except Exception as e:
             logger.error(f"CVXPY: An unexpected error occurred during optimization for size {size}: {e}", exc_info=True)


    logger.info(f"CVXPY: Finished. Generated {len(all_optimal_papers)} optimal papers across requested sizes.")
    return all_optimal_papers


def calculate_paper_metrics(paper_selections: List[Dict]) -> Dict:
    """Calculates combined metrics for a single paper."""
    metrics = {
        "num_legs": 0,
        "combined_odds": Decimal('1.0'),
        "combined_probability": Decimal('1.0'),
        "average_edge": Decimal('NaN'), # Use NaN for invalid/undefined averages
        "average_probability": Decimal('NaN'),
        "average_odds": Decimal('NaN'),
        "average_efficiency_score": Decimal('NaN'), # Avg of individual scores
        "average_value_ratio": Decimal('NaN'),
        "leagues": set(), # Use set for unique collection
        "teams": set(),   # Use set for unique collection
        "fixture_ids": set(), # Track fixtures involved
        "all_odds_valid": True, # Flag if any odd was invalid
        "all_probs_valid": True, # Flag if any prob was invalid
        "contains_invalid_selection": False # General flag for any bad data
    }
    if not paper_selections:
        metrics["contains_invalid_selection"] = True
        metrics["all_odds_valid"] = False
        metrics["all_probs_valid"] = False
        return metrics # Return defaults, mostly NaN

    n_legs = len(paper_selections)
    metrics["num_legs"] = n_legs
    sum_edge = Decimal('0.0')
    sum_prob = Decimal('0.0')
    sum_odds = Decimal('0.0')
    sum_efficiency = Decimal('0.0')
    sum_value_ratio = Decimal('0.0')
    valid_legs_count = 0 # Count legs with valid data for averaging

    for sel in paper_selections:
        odds = sel.get('odds')
        prob = sel.get('probability')
        edge = sel.get('edge')
        eff_score = sel.get('efficiency_score')
        fixture_id = sel.get('fixture_id')

        # Validate core numerical data for this leg
        is_leg_valid = True
        if not isinstance(odds, Decimal) or odds.is_nan() or odds <= Decimal('1.0'):
            metrics["all_odds_valid"] = False
            is_leg_valid = False
            logger.debug(f"Paper Metric Calc: Invalid odd {odds} in fixture {fixture_id}, sel {sel.get('selection')}")
        if not isinstance(prob, Decimal) or prob.is_nan() or prob <= Decimal('0.0') or prob > Decimal('1.0'):
            metrics["all_probs_valid"] = False
            is_leg_valid = False
            logger.debug(f"Paper Metric Calc: Invalid prob {prob} in fixture {fixture_id}, sel {sel.get('selection')}")
        if not isinstance(edge, Decimal) or edge.is_nan():
            is_leg_valid = False # Edge needed for average
            logger.debug(f"Paper Metric Calc: Invalid edge {edge} in fixture {fixture_id}, sel {sel.get('selection')}")
        if not isinstance(eff_score, Decimal) or eff_score.is_nan():
             # Efficiency score might be optional, don't mark whole leg invalid, but skip for average
            logger.debug(f"Paper Metric Calc: Missing/invalid efficiency score {eff_score} for avg calculation in fixture {fixture_id}")
        pass # Don't set is_leg_valid = False just for this


        if not is_leg_valid:
            metrics["contains_invalid_selection"] = True
            # If odds or prob are invalid, the combined values become unreliable
            if not metrics["all_odds_valid"]: metrics["combined_odds"] = Decimal('NaN')
            if not metrics["all_probs_valid"]: metrics["combined_probability"] = Decimal('NaN')
            # Don't include this leg's potentially bad data in averages
            continue

        # Accumulate valid data
        valid_legs_count += 1
        metrics["combined_odds"] *= odds
        metrics["combined_probability"] *= prob
        sum_prob += prob
        sum_edge += edge
        sum_odds += odds
        if isinstance(eff_score, Decimal): # Only sum valid scores
            sum_efficiency += eff_score
        # Calculate and sum value ratio for averaging
        sum_value_ratio += (prob * odds)

        # Collect context
        if fixture_id: metrics["fixture_ids"].add(fixture_id)
        league = sel.get('league_name')
        if league and league != "Unknown League": metrics["leagues"].add(league)
        home_team = sel.get('home_team_name')
        away_team = sel.get('away_team_name')
        if home_team and home_team != "Home Team": metrics["teams"].add(home_team) # Use default from find_team_names
        if away_team and away_team != "Away Team": metrics["teams"].add(away_team)


    # Calculate averages only if there were valid legs
    if valid_legs_count > 0:
        metrics["average_edge"] = sum_edge / valid_legs_count
        metrics["average_probability"] = sum_prob / valid_legs_count
        metrics["average_odds"] = sum_odds / valid_legs_count
        metrics["average_value_ratio"] = sum_value_ratio / valid_legs_count
        # Average efficiency score calculation needs care if some legs didn't have it
        valid_eff_score_count = sum(1 for sel in paper_selections if isinstance(sel.get('efficiency_score'), Decimal) and not sel['efficiency_score'].is_nan())
        if valid_eff_score_count > 0:
            metrics["average_efficiency_score"] = sum_efficiency / valid_eff_score_count

    # Convert sets to sorted lists for consistent output
    metrics["leagues"] = sorted(list(metrics["leagues"]))
    metrics["teams"] = sorted(list(metrics["teams"]))
    metrics["fixture_ids"] = sorted(list(metrics["fixture_ids"]))

    # Final check on combined odds/prob if calculation resulted in zero/sub-zero due to multiplication
    if isinstance(metrics["combined_odds"], Decimal) and metrics["combined_odds"] <= Decimal('1.0'):
        # If it was calculated (not NaN) but ended up <= 1.0, mark as invalid unless it's a single leg paper
        if n_legs > 1:
             logger.debug(f"Paper Metric Calc: Combined odds {metrics['combined_odds']} <= 1.0 for multi-leg paper. Marking NaN.")
             metrics["combined_odds"] = Decimal('NaN')
        elif n_legs == 1 and metrics["combined_odds"] <= Decimal('1.0'):
              # Single leg paper, odds reflect individual bet odds
              pass # Allow single leg odds <= 1.0 if that's what the data says, but it was likely filtered earlier

    if isinstance(metrics["combined_probability"], Decimal) and metrics["combined_probability"] <= Decimal('0.0'):
        logger.debug(f"Paper Metric Calc: Combined probability {metrics['combined_probability']} <= 0.0. Marking NaN.")
        metrics["combined_probability"] = Decimal('NaN')


    return metrics


def calculate_paper_efficiency_score(paper_metrics: Dict, weights: Dict) -> Optional[Decimal]:
    """Calculates a composite efficiency score for the entire paper based on AVERAGE metrics."""
    # Use average values from the paper metrics
    avg_prob = paper_metrics.get('average_probability')
    avg_edge = paper_metrics.get('average_edge')
    avg_value_ratio = paper_metrics.get('average_value_ratio')

    # Ensure metrics needed are valid Decimals and not NaN
    if not all(isinstance(val, Decimal) and not val.is_nan() for val in [avg_prob, avg_edge, avg_value_ratio]):
        logger.debug("Cannot calculate paper efficiency score due to missing/invalid average metrics.")
        return None

    try:
        # Reuse the same weight logic as individual scores, but apply to averages
        w_prob = weights.get('weight_prob', DEFAULT_WEIGHT_PROB)
        w_edge = weights.get('weight_edge', DEFAULT_WEIGHT_EDGE)
        w_val_ratio = weights.get('weight_value_ratio', DEFAULT_WEIGHT_VALUE_RATIO)

        w_prob = parse_decimal(w_prob, "weight_prob") or Decimal('0.0')
        w_edge = parse_decimal(w_edge, "weight_edge") or Decimal('0.0')
        w_val_ratio = parse_decimal(w_val_ratio, "weight_value_ratio") or Decimal('0.0')

        total_weight = w_prob + w_edge + w_val_ratio
        if total_weight <= 0: total_weight = Decimal('1.0')
        w_prob /= total_weight
        w_edge /= total_weight
        w_val_ratio /= total_weight

        # Score based on average properties
        score = (w_prob * avg_prob +
                 w_edge * avg_edge +
                 w_val_ratio * (avg_value_ratio - Decimal('1.0')))

        return score
    except (TypeError, InvalidOperation) as e:
        logger.warning(f"Error calculating paper efficiency score: {e}")
        return None


def filter_papers(papers_with_metrics: List[Dict],
                  filter_leagues_normalized: Optional[Set[str]],
                  filter_teams_normalized: Optional[Set[str]],
                  filter_min_combined_odds: Optional[Decimal],
                  filter_max_combined_odds: Optional[Decimal],
                  filter_min_combined_prob: Optional[Decimal],
                  filter_min_avg_edge: Optional[Decimal]
                  ) -> List[Dict]:
    """Filters the list of generated papers based on combined metric criteria."""
    filtered_list = []
    initial_count = len(papers_with_metrics)
    logger.info(f"--- Filtering {initial_count} Generated Papers (Post-Generation) ---")
    if filter_leagues_normalized: logger.info(f"Applying league filter: {filter_leagues_normalized}")
    if filter_teams_normalized: logger.info(f"Applying team filter: {filter_teams_normalized}")
    if filter_min_combined_odds: logger.info(f"Applying Min Combined Odds filter: >= {filter_min_combined_odds}")
    if filter_max_combined_odds: logger.info(f"Applying Max Combined Odds filter: <= {filter_max_combined_odds}")
    if filter_min_combined_prob: logger.info(f"Applying Min Combined Prob filter: >= {filter_min_combined_prob}")
    if filter_min_avg_edge: logger.info(f"Applying Min Avg Edge filter: >= {filter_min_avg_edge}")


    for paper_data in papers_with_metrics:
        metrics = paper_data.get('paper_metrics', {})
        paper_id = paper_data.get('paper_id', 'Unknown') # For logging

        # Basic check: ensure metrics are present and no invalid selections flagged
        if not metrics or metrics.get('contains_invalid_selection', True):
            logger.debug(f"Filtering out Paper {paper_id}: Missing metrics or contains invalid selection data.")
            continue

        # Filter by League(s) - Check if *any* league in the paper is *not* in the allowed set
        if filter_leagues_normalized:
            paper_leagues_lower = {l.lower() for l in metrics.get('leagues', [])}
            if not paper_leagues_lower.issubset(filter_leagues_normalized):
                 # Log the difference for clarity
                 disallowed = paper_leagues_lower - filter_leagues_normalized
                 logger.debug(f"Filtering out Paper {paper_id}: Contains disallowed leagues: {disallowed}")
                 continue

        # Filter by Team(s) - Check if *at least one* required team is present
        if filter_teams_normalized:
            paper_teams_lower = {t.lower() for t in metrics.get('teams', [])}
            if not filter_teams_normalized.intersection(paper_teams_lower):
                 logger.debug(f"Filtering out Paper {paper_id}: Does not contain any required teams: {filter_teams_normalized}")
                 continue

        # --- Filter by Numerical Metrics ---
        combined_odds = metrics.get('combined_odds') # Already handled NaN during calculation
        combined_prob = metrics.get('combined_probability')
        avg_edge = metrics.get('average_edge')

        # Filter by Min Combined Odds
        if filter_min_combined_odds is not None:
            if not isinstance(combined_odds, Decimal) or combined_odds.is_nan() or combined_odds < filter_min_combined_odds:
                logger.debug(f"Filtering out Paper {paper_id}: Combined odds {combined_odds} < {filter_min_combined_odds} or invalid.")
                continue

        # Filter by Max Combined Odds
        if filter_max_combined_odds is not None:
             if not isinstance(combined_odds, Decimal) or combined_odds.is_nan() or combined_odds > filter_max_combined_odds:
                 logger.debug(f"Filtering out Paper {paper_id}: Combined odds {combined_odds} > {filter_max_combined_odds} or invalid.")
                 continue

        # Filter by Min Combined Probability
        if filter_min_combined_prob is not None:
             if not isinstance(combined_prob, Decimal) or combined_prob.is_nan() or combined_prob < filter_min_combined_prob:
                 logger.debug(f"Filtering out Paper {paper_id}: Combined prob {combined_prob} < {filter_min_combined_prob} or invalid.")
                 continue

        # Filter by Min Average Edge
        if filter_min_avg_edge is not None:
            if not isinstance(avg_edge, Decimal) or avg_edge.is_nan() or avg_edge < filter_min_avg_edge:
                logger.debug(f"Filtering out Paper {paper_id}: Average edge {avg_edge} < {filter_min_avg_edge} or invalid.")
                continue

        # If all filters pass
        filtered_list.append(paper_data)

    final_count = len(filtered_list)
    filtered_out = initial_count - final_count
    logger.info(f"Finished filtering. Kept {final_count} papers (filtered out {filtered_out}).")
    return filtered_list

# --- Main Execution Logic (Parameter Driven API Function) ---

def generate_papers(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to generate, filter, and rank betting papers based on parameters.

    Args:
        params (Dict[str, Any]): Dictionary containing configuration parameters.
            See DEFAULT values and comments above for expected keys.

    Returns:
        Dict[str, Any]: A dictionary containing generation info, summary,
                        settings used, and the list of ranked, filtered papers.
                        Uses structure suitable for JSON output.
    """
    start_time = datetime.now()
    logger.info("--- Starting VFAPI Paper Generation Process ---")

    # --- Get and Validate Parameters ---
    # Use .get() with defaults for robustness
    input_filepath = params.get('input_file', DEFAULT_INPUT_FILE)
    output_filepath = params.get('output_file', DEFAULT_OUTPUT_FILE)
    debug_mode = params.get('debug', False)

    # Configure logging level based on debug param EARLY
    if debug_mode:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers: handler.setLevel(logging.DEBUG)
        logger.info("--- Debug logging enabled ---")
    else:
        logger.setLevel(logging.INFO)
        for handler in logger.handlers: handler.setLevel(logging.INFO)

    # Selection Filtering Params (Parse to Decimal where needed)
    min_edge = parse_decimal(params.get('min_edge', DEFAULT_EDGE_THRESHOLD), "min_edge") or DEFAULT_EDGE_THRESHOLD
    min_probability = parse_decimal(params.get('min_probability', DEFAULT_MIN_PROBABILITY), "min_probability") or DEFAULT_MIN_PROBABILITY
    min_odds = parse_decimal(params.get('min_odds', DEFAULT_MIN_ODDS), "min_odds") # Can be None
    max_odds = parse_decimal(params.get('max_odds', DEFAULT_MAX_ODDS), "max_odds") # Can be None
    top_n_per_game = int(params.get('top_n_per_game', DEFAULT_TOP_N_PER_GAME))

    # Efficiency Score Weights
    raw_weights = {
        'weight_prob': params.get('weight_prob', DEFAULT_WEIGHT_PROB),
        'weight_edge': params.get('weight_edge', DEFAULT_WEIGHT_EDGE),
        'weight_value_ratio': params.get('weight_value_ratio', DEFAULT_WEIGHT_VALUE_RATIO)
    }
    efficiency_weights = {k: parse_decimal(v, f"weight {k}") or Decimal('0.0') for k, v in raw_weights.items()}

    # Paper Generation Params
    raw_paper_sizes = params.get('paper_sizes', DEFAULT_PAPER_SIZES)
    paper_sizes = sorted(list(set(s for s in raw_paper_sizes if isinstance(s, int) and s > 0))) # Clean list
    paper_build_strategy = params.get('paper_build_strategy', DEFAULT_PAPER_BUILD_STRATEGY).lower()
    use_cvxpy = params.get('use_cvxpy', DEFAULT_USE_CVXPY)
    # Determine max papers based on strategy
    max_papers_per_size = DEFAULT_MAX_PAPERS_PER_SIZE_CVXPY if use_cvxpy else int(params.get('max_papers_per_size_greedy', DEFAULT_MAX_PAPERS_PER_SIZE_GREEDY))


    # CVXPY Constraint Params (Parse to Decimal)
    cvxpy_max_combined_odds = parse_decimal(params.get('cvxpy_max_combined_odds', DEFAULT_CVXPY_MAX_COMBINED_ODDS), "cvxpy_max_odds")
    cvxpy_min_combined_prob = parse_decimal(params.get('cvxpy_min_combined_prob', DEFAULT_CVXPY_MIN_COMBINED_PROB), "cvxpy_min_prob")

    # Final Paper Filtering Params (Parse to Decimal where needed)
    raw_filter_leagues = params.get('filter_leagues', DEFAULT_FILTER_LEAGUES)
    filter_leagues_normalized = set(l.lower() for l in raw_filter_leagues if isinstance(l, str)) if raw_filter_leagues else None

    raw_filter_teams = params.get('filter_teams', DEFAULT_FILTER_TEAMS)
    filter_teams_normalized = set(t.lower() for t in raw_filter_teams if isinstance(t, str)) if raw_filter_teams else None

    filter_min_combined_odds = parse_decimal(params.get('filter_min_combined_odds', DEFAULT_FILTER_MIN_COMBINED_ODDS), "filter_min_comb_odds")
    filter_max_combined_odds = parse_decimal(params.get('filter_max_combined_odds', DEFAULT_FILTER_MAX_COMBINED_ODDS), "filter_max_comb_odds")
    filter_min_combined_prob = parse_decimal(params.get('filter_min_combined_prob', DEFAULT_FILTER_MIN_COMBINED_PROB), "filter_min_comb_prob")
    filter_min_avg_edge = parse_decimal(params.get('filter_min_avg_edge', DEFAULT_FILTER_MIN_AVG_EDGE), "filter_min_avg_edge")

    # Ranking Param
    ranking_strategy = params.get('ranking_strategy', DEFAULT_RANKING_STRATEGY).lower()

    # Plotting Params
    enable_plotting = params.get('enable_plotting', DEFAULT_ENABLE_PLOTTING)
    plot_output_dir = params.get('plot_output_dir', DEFAULT_PLOT_OUTPUT_DIR)
    plots_to_generate = params.get('plots_to_generate', DEFAULT_PLOTS_TO_GENERATE)

    # --- Configure Matplotlib Backend (Crucial for API/non-interactive) ---
    if enable_plotting:
        try:
            matplotlib.use('Agg') # Set backend *before* importing pyplot
            import matplotlib.pyplot as plt # Import now safe
            logger.info("Matplotlib backend set to 'Agg' for non-interactive plotting.")
        except Exception as e:
             logger.error(f"Failed to set Matplotlib backend or import pyplot: {e}. Disabling plotting.")
             enable_plotting = False

    # --- Prepare Output Directories ---
    output_dir = os.path.dirname(output_filepath)
    try:
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        if enable_plotting and plot_output_dir and not os.path.exists(plot_output_dir):
             os.makedirs(plot_output_dir)
             logger.info(f"Created plot output directory: {plot_output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory(ies): {e}")
        # Return error state immediately
        error_info = {
            "generation_info": {"status": "error", "error_message": f"Failed to create directories: {e}", "settings": convert_for_json(params)},
            "ranked_filtered_papers": []
        }
        return error_info # No need to convert here, handled by caller if needed


    # --- 1. Load Data ---
    batch_prediction_data = load_data(input_filepath)
    if batch_prediction_data is None:
        error_info = {
            "generation_info": {"status": "error", "error_message": f"Failed to load input data from {input_filepath}", "settings": convert_for_json(params)},
            "ranked_filtered_papers": []
        }
        return error_info


    # --- 2. Filter Selections per Game, Calculate Score, Trim ---
    logger.info("--- Stage 1: Filtering & Scoring Individual Selections ---")
    all_filtered_selections_by_game = defaultdict(list)
    all_selections_flat_list = [] # Combined list for CVXPY or analysis
    processed_fixture_ids = set()
    skipped_duplicates = 0
    games_with_no_valid_selections = 0
    total_selections_processed = 0

    for i, match_data in enumerate(batch_prediction_data):
        current_fixture_id = find_fixture_id(match_data)
        if not current_fixture_id:
             logger.warning(f"Skipping entry {i+1}: Could not determine fixture ID.")
             continue

        # Handle potential duplicate fixture entries in input
        if current_fixture_id in processed_fixture_ids:
             logger.debug(f"Skipping duplicate fixture ID: {current_fixture_id}")
             skipped_duplicates += 1
             continue
        processed_fixture_ids.add(current_fixture_id)

        # Filter, score, enrich, and trim selections for this game
        game_selections = filter_and_score_game_selections(
             match_data, current_fixture_id, min_edge, min_probability,
             min_odds, max_odds, top_n_per_game, efficiency_weights
         )
        total_selections_processed += len(safe_get(match_data, ['top_n_combined_selections'], [])) # Count before filtering

        if game_selections:
             all_filtered_selections_by_game[current_fixture_id] = game_selections
             all_selections_flat_list.extend(game_selections) # Add to flat list
        else:
             games_with_no_valid_selections += 1

    # --- Initial Filtering Summary ---
    num_fixtures_with_selections = len(all_filtered_selections_by_game)
    logger.info(f"--- Initial Selection Filtering Summary ---")
    logger.info(f"Total Input Matches/Entries: {len(batch_prediction_data)}")
    logger.info(f"Unique Fixtures Processed: {len(processed_fixture_ids)}")
    logger.info(f"Skipped Duplicate Fixture Entries: {skipped_duplicates}")
    logger.info(f"Total Selections Processed (Before Filtering): {total_selections_processed}")
    logger.info(f"Total Selections Passing Initial Filters (Combined): {len(all_selections_flat_list)}")
    logger.info(f"Fixtures with >=1 Valid Selections (After Top-N): {num_fixtures_with_selections}")
    logger.info(f"Fixtures with No Valid Selections: {games_with_no_valid_selections}")

    # --- Early Exit if Not Enough Data for Papers ---
    min_size = min(paper_sizes) if paper_sizes else 1
    if not all_filtered_selections_by_game or num_fixtures_with_selections < min_size:
        logger.warning(f"Not enough games ({num_fixtures_with_selections}) with valid selections for minimum requested paper size ({min_size}). Cannot generate papers.")
        status = "completed_no_papers_insufficient_data"
        # Prepare summary for early exit
        summary = {
             "total_matches_input": len(batch_prediction_data),
             "unique_fixtures_processed": len(processed_fixture_ids),
             "skipped_duplicate_fixtures": skipped_duplicates,
             "fixtures_with_valid_selections": num_fixtures_with_selections,
             "total_selections_considered": len(all_selections_flat_list),
             "papers_generated_before_filtering": 0,
             "papers_remaining_after_filtering": 0,
             "plots_generated": 0
         }
        final_output = {
             "generation_info": {
                 "generated_at": datetime.utcnow().isoformat() + "Z",
                 "execution_duration_seconds": (datetime.now() - start_time).total_seconds(),
                 "status": status,
                 "settings": convert_for_json(params), # Use initial params
                 "summary": summary
             },
             "ranked_filtered_papers": []
         }
        # Write output if path specified, even on early exit
        if output_filepath:
            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(final_output, f, indent=4, ensure_ascii=False)
                logger.info(f"Wrote empty result file to: {output_filepath}")
            except IOError as e:
                logger.error(f"Error writing empty result file {output_filepath}: {e}")
                final_output["generation_info"]["status"] = "error_writing_output"
                final_output["generation_info"]["error_message"] = str(e)

        return final_output # Return the structured data


    # --- 3. Build Paper Combinations ---
    logger.info("--- Stage 2: Building Paper Combinations ---")
    generated_papers_raw = []
    if use_cvxpy:
        generated_papers_raw = build_papers_cvxpy(
            all_selections_flat_list, # Pass the flat list of valid selections
            paper_sizes=paper_sizes,
            max_combined_odds=cvxpy_max_combined_odds,
            min_combined_prob=cvxpy_min_combined_prob,
            efficiency_weights=efficiency_weights, # Pass for logging objective basis
            debug_mode=debug_mode
        )
    else:
        # Pass normalized team filter set to greedy builder if needed
        # The post-generation filter will also apply, but this can be more efficient
        greedy_team_filter = filter_teams_normalized # Pass the set directly
        generated_papers_raw = build_papers_greedy(
            all_filtered_selections_by_game, # Pass selections grouped by game
            paper_sizes=paper_sizes,
            max_papers_per_size=max_papers_per_size,
            strategy=paper_build_strategy,
            filter_teams_normalized=greedy_team_filter
        )

    if not generated_papers_raw:
        logger.warning("Paper combination stage did not produce any potential papers (either greedy or CVXPY).")
        status = "completed_no_papers_generated"
        # Prepare summary similar to above
        summary = {
             "total_matches_input": len(batch_prediction_data),
             "unique_fixtures_processed": len(processed_fixture_ids),
             "skipped_duplicate_fixtures": skipped_duplicates,
             "fixtures_with_valid_selections": num_fixtures_with_selections,
             "total_selections_considered": len(all_selections_flat_list),
             "papers_generated_before_filtering": 0,
             "papers_remaining_after_filtering": 0,
             "plots_generated": 0
         }
        final_output = {
             "generation_info": {
                 "generated_at": datetime.utcnow().isoformat() + "Z",
                 "execution_duration_seconds": (datetime.now() - start_time).total_seconds(),
                 "status": status,
                 "settings": convert_for_json(params), # Use initial params
                 "summary": summary
             },
             "ranked_filtered_papers": []
         }
        # Write output if path specified
        if output_filepath:
             try:
                 with open(output_filepath, 'w', encoding='utf-8') as f:
                     json.dump(final_output, f, indent=4, ensure_ascii=False)
                 logger.info(f"Wrote empty result file to: {output_filepath}")
             except IOError as e:
                 logger.error(f"Error writing empty result file {output_filepath}: {e}")
                 final_output["generation_info"]["status"] = "error_writing_output"
                 final_output["generation_info"]["error_message"] = str(e)

        return final_output


    # --- 4. Calculate Metrics for Each Generated Paper ---
    logger.info(f"--- Stage 3: Calculating Metrics for {len(generated_papers_raw)} Generated Papers ---")
    papers_with_metrics = []
    for i, paper_legs in enumerate(generated_papers_raw):
        metrics = calculate_paper_metrics(paper_legs)
        # Calculate paper efficiency score based on average metrics
        paper_efficiency_score = calculate_paper_efficiency_score(metrics, efficiency_weights)

        papers_with_metrics.append({
            "paper_id": f"Paper_{'Opt_' if use_cvxpy else 'Grd_'}{i+1:04d}", # Unique ID
            "paper_metrics": metrics,
            "paper_efficiency_score": paper_efficiency_score, # Overall score for the paper
            "selections": paper_legs # The list of selection dicts
        })

    # --- 5. Filter Papers Based on Post-Generation Criteria ---
    logger.info("--- Stage 4: Filtering Papers Based on Post-Generation Criteria ---")
    filtered_papers = filter_papers(
        papers_with_metrics,
        filter_leagues_normalized=filter_leagues_normalized,
        filter_teams_normalized=filter_teams_normalized, # Use normalized set
        filter_min_combined_odds=filter_min_combined_odds,
        filter_max_combined_odds=filter_max_combined_odds,
        filter_min_combined_prob=filter_min_combined_prob,
        filter_min_avg_edge=filter_min_avg_edge
    )

    # --- 6. Rank Filtered Papers ---
    num_filtered_papers = len(filtered_papers)
    logger.info(f"--- Stage 5: Ranking {num_filtered_papers} Filtered Papers ---")
    ranking_description = "N/A - No papers to rank."
    if filtered_papers:
        sort_key_func = None
        if ranking_strategy == 'avg_efficiency_score':
            # Sort by paper_efficiency_score (higher is better)
            # Handle None/NaN by placing them at the end
            sort_key_func = lambda p: p.get('paper_efficiency_score', Decimal('-Infinity'))
            ranking_description = "Highest Average Paper Efficiency Score"

        elif ranking_strategy == 'combined_prob_then_edge':
            # Sort primarily by combined_probability (higher is better), then by average_edge (higher is better)
            # Handle None/NaN carefully
            def get_prob_edge_key(paper):
                prob = paper.get('paper_metrics', {}).get('combined_probability')
                edge = paper.get('paper_metrics', {}).get('average_edge')
                # Use -Infinity for None/NaN to sort them lower
                prob_sort = prob if isinstance(prob, Decimal) and not prob.is_nan() else Decimal('-Infinity')
                edge_sort = edge if isinstance(edge, Decimal) and not edge.is_nan() else Decimal('-Infinity')
                return (prob_sort, edge_sort) # Tuple for multi-level sort (both descending)
            sort_key_func = get_prob_edge_key
            ranking_description = "Highest Combined Probability, then Highest Average Edge"

        else:
             logger.warning(f"Unknown ranking strategy: '{ranking_strategy}'. Papers remain in filtered order.")
             ranking_description = f"Unknown strategy '{ranking_strategy}' - Order preserved from filtering."

        if sort_key_func:
            try:
                 # Sort in descending order (highest score/prob first)
                 filtered_papers.sort(key=sort_key_func, reverse=True)
                 logger.info(f"Ranked papers by: {ranking_description}.")
            except Exception as e:
                 logger.error(f"Error during sorting with strategy '{ranking_strategy}': {e}. Papers may not be correctly ranked.", exc_info=True)
                 ranking_description += " (Sorting Error Occurred)"

    else:
         logger.warning("No papers remained after filtering. No ranking performed.")


    # --- 7. Plotting (Optional) ---
    generated_plot_files = []
    if enable_plotting and filtered_papers:
        logger.info(f"--- Stage 6: Generating Plots (Matplotlib) ---")
        # Ensure plot directory exists again just in case
        if plot_output_dir and not os.path.exists(plot_output_dir):
            try: os.makedirs(plot_output_dir)
            except OSError as e: logger.error(f"Failed to create plot directory {plot_output_dir}: {e}. Skipping plotting.")
            enable_plotting = False # Disable if dir fails

        if enable_plotting:
            # Add risk metric if needed for plotting
            for paper in filtered_papers:
                if 'paper_metrics' in paper:
                     prob = paper['paper_metrics'].get('combined_probability')
                     if isinstance(prob, Decimal) and not prob.is_nan() and prob > 0:
                         # Simple risk: 1 / probability (higher means riskier)
                         # Or use log-odds: -log(prob / (1-prob)) if prob < 1
                         try:
                            if prob < 1:
                                paper['paper_metrics']['risk'] = -np.log(float(prob) / (1.0 - float(prob)))
                            else: # Avoid division by zero if prob is 1
                                paper['paper_metrics']['risk'] = -np.inf # Extremely low risk (theoretically)
                         except (ValueError, FloatingPointError):
                              paper['paper_metrics']['risk'] = None # Cannot calculate
                     else:
                         paper['paper_metrics']['risk'] = None

            # Generate configured plots
            for plot_config in plots_to_generate:
                filename = plot_config.get("filename")
                if not filename:
                    logger.warning("Skipping plot config due to missing 'filename'.")
                    continue

                output_plot_path = os.path.join(plot_output_dir, filename)
                logger.debug(f"Generating plot: {output_plot_path}")

                try:
                    # Call the plotting function from plotting_utils
                    plot_paper_scatter_mpl(
                        papers_data=filtered_papers, # Pass the ranked+filtered data
                        output_file=output_plot_path,
                        x_axis_metric=plot_config.get("x"),
                        y_axis_metric=plot_config.get("y"),
                        color_metric=plot_config.get("color"),
                        size_metric=plot_config.get("size"),
                        title=plot_config.get("title"),
                        x_log=plot_config.get("x_log", False), # Get log scale options
                        y_log=plot_config.get("y_log", False)
                    )
                    generated_plot_files.append(os.path.basename(output_plot_path))
                    logger.debug(f"Successfully generated plot: {filename}")
                except Exception as e:
                     logger.error(f"Failed to generate plot '{filename}': {e}", exc_info=debug_mode)

            logger.info(f"Finished generating {len(generated_plot_files)} plots.")

    elif enable_plotting:
        logger.warning("Plotting enabled, but no filtered papers available to plot.")

    # --- 8. Prepare Final Output Structure ---
    logger.info("--- Stage 7: Preparing Final Output ---")
    end_time = datetime.now()
    duration = end_time - start_time

    # Capture the actual settings used (after parsing/defaults)
    final_settings_used = {
        "input_file": input_filepath,
        "output_file": output_filepath,
        "debug": debug_mode,
        "min_edge": min_edge,
        "min_probability": min_probability,
        "min_odds": min_odds,
        "max_odds": max_odds,
        "top_n_per_game": top_n_per_game,
        "efficiency_weights": efficiency_weights,
        "paper_sizes": paper_sizes,
        "paper_build_strategy": paper_build_strategy,
        "use_cvxpy": use_cvxpy,
        "max_papers_per_size_used": max_papers_per_size,
        "cvxpy_max_combined_odds": cvxpy_max_combined_odds,
        "cvxpy_min_combined_prob": cvxpy_min_combined_prob,
        "filter_leagues": raw_filter_leagues, # Show original filter requested
        "filter_teams": raw_filter_teams,     # Show original filter requested
        "filter_min_combined_odds": filter_min_combined_odds,
        "filter_max_combined_odds": filter_max_combined_odds,
        "filter_min_combined_prob": filter_min_combined_prob,
        "filter_min_avg_edge": filter_min_avg_edge,
        "ranking_strategy": ranking_strategy,
        "ranking_method_used": ranking_description, # Description of what happened
        "enable_plotting": enable_plotting,
        "plot_output_dir": plot_output_dir,
        "plots_to_generate_config": plots_to_generate # Show plot config used
    }

    # Determine final status
    final_status = "success"
    if not filtered_papers:
        if not papers_with_metrics: # If filtering started with zero papers
             final_status = "completed_no_papers_generated" # Already handled earlier, but double check
        else: # Papers were generated but all filtered out
             final_status = "completed_no_valid_papers_found_after_filtering"


    output_data = {
         "generation_info": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "execution_duration_seconds": duration.total_seconds(),
            "input_file_processed": os.path.basename(input_filepath),
            "output_file_generated": os.path.basename(output_filepath) if output_filepath else None,
            "plotting_enabled": enable_plotting,
            "plot_output_dir": os.path.basename(plot_output_dir) if enable_plotting and plot_output_dir else None,
            "generated_plot_files": generated_plot_files,
            "settings": convert_for_json(final_settings_used), # Convert final settings
            "summary": {
                 "total_matches_input": len(batch_prediction_data),
                 "unique_fixtures_processed": len(processed_fixture_ids),
                 "skipped_duplicate_fixtures": skipped_duplicates,
                 "fixtures_with_valid_selections": num_fixtures_with_selections,
                 "total_selections_considered": len(all_selections_flat_list),
                 "papers_generated_before_filtering": len(papers_with_metrics),
                 "papers_remaining_after_filtering": num_filtered_papers,
                 "plots_generated": len(generated_plot_files)
            },
            "status": final_status
        },
        "ranked_filtered_papers": convert_for_json(filtered_papers) # Convert final list
    }

    # --- 9. Write Output File (Optional) ---
    if output_filepath:
        logger.info(f"Writing {num_filtered_papers} ranked & filtered papers to: {output_filepath}")
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                # Dump the already converted data
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            logger.info("Successfully wrote output file.")
        except IOError as e:
            logger.error(f"Error writing output file {output_filepath}: {e}")
            output_data["generation_info"]["status"] = "error_writing_output"
            output_data["generation_info"]["error_message"] = str(e)
        except Exception as e:
            logger.error(f"Unexpected error during file writing: {e}", exc_info=True)
            output_data["generation_info"]["status"] = "error_writing_output"
            output_data["generation_info"]["error_message"] = f"Unexpected error: {e}"


    logger.info(f"--- VFAPI Paper Generation Finished in {duration.total_seconds():.2f} seconds ---")
    return output_data # Return the final data structure
