#!/usr/bin/env python3
"""
Parlay Backtester - Refactored for Efficiency and Sampling.

This script backtests parlay betting strategies using OOF predictions,
a strategy guide, and consolidated team information.

Key Features:
- Yearly date sampling for lighter processing.
- Optimized daily processing with aggressive memory management.
- Combination limiting to prevent performance bottlenecks.
- Chunk-based checkpointing for robustness.
- Strict adherence to coding principles (Minimal, Typed, Assertive, etc.).
"""

import gc
import json
import logging
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import (BaseModel, Field, ValidationError, field_validator,
                      ValidationInfo, model_validator)
from tqdm import tqdm

# --- Project Setup ---
# Add project root to sys.path for local imports
try:
    # Assumes script is in AgenticFC888/scripts/analysis/
    _project_root = Path(__file__).resolve().parent.parent.parent
    if not (_project_root / "models").exists() or not (_project_root / "scripts").exists():
        # Fallback if structure is different
        _project_root = Path(os.getenv("PROJECT_ROOT", "/Users/barroca888/Downloads/Agenticfc/AgenticFC888")) # Use ENV var or hardcoded fallback
    sys.path.insert(0, str(_project_root))
    from models.utils.config import TeamNameStrict # type: ignore # If utils is under models
except ImportError as e:
    print(f"Error importing project modules: {e}. Ensure PROJECT_ROOT is set or script is in the expected location.")
    # Define a dummy TeamNameStrict if import fails, allowing script to load but likely fail later
    class TeamNameStrict(str): pass
    # sys.exit(1) # Exit if core components can't be imported

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib') # Ignore matplotlib warnings

# Determine Base Directory more robustly
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = _project_root # Use the determined project root
logger.info(f"Using BASE_DIR: {BASE_DIR}")
assert BASE_DIR.exists(), f"Base directory does not exist: {BASE_DIR}"
assert (BASE_DIR / "models").exists(), f"Models directory not found in BASE_DIR: {BASE_DIR / 'models'}"

# Define Core Paths
DATA_OUTPUT_DIR = BASE_DIR / 'models' / 'data' / 'outputs' / 'predictions'
UTILS_DIR = BASE_DIR / 'models' / 'utils' / 'files'
PARLAY_OUTPUT_DIR = DATA_OUTPUT_DIR / 'parlay_outputs_V3_sampled' # New distinct output dir

# Default Input Paths
OOF_INPUT_PATH_DEFAULT = DATA_OUTPUT_DIR / 'combined_oof_ALL_pipelines.parquet'
STRATEGY_GUIDE_PATH_DEFAULT = UTILS_DIR / 'best_strategy_per_market.csv'
MARKET_DEFINITIONS_PATH_DEFAULT = UTILS_DIR / 'parlay_market_definitions.json'
CONSOLIDATED_TEAM_INFO_PATH_DEFAULT = UTILS_DIR / 'consolidated_team_info.json'

# Default Output Paths
PARLAY_RESULTS_PATH_DEFAULT = PARLAY_OUTPUT_DIR / 'parlay_backtest_results_sampled.parquet'
PARLAY_RESULTS_CSV_PATH_DEFAULT = PARLAY_OUTPUT_DIR / 'parlay_backtest_results_sampled.csv'
VISUALIZATIONS_DIR_DEFAULT = PARLAY_OUTPUT_DIR / 'visualizations_sampled'
CHECKPOINT_DIR_DEFAULT = PARLAY_OUTPUT_DIR / 'checkpoints_sampled'

# Column Names (Constants)
DEFAULT_DATE_COL = 'Date'
DEFAULT_MATCH_ID_COL = 'MatchID'

# Performance & Resource Management
DEFAULT_MAX_CPU_WORKERS = max(1, os.cpu_count() // 2) # Default to 50% CPU cores
MEMORY_CHECK_INTERVAL_SECONDS = 15 # Check memory periodically during long operations
MEMORY_HIGH_WM_PERCENT = 85.0 # High watermark for memory usage %
MEMORY_CRITICAL_PERCENT = 92.0 # Critical threshold
MIN_FREE_MEMORY_GB = 1.0 # Minimum free GB to maintain

# Backtesting Parameters
DEFAULT_MIN_LEGS = 2
DEFAULT_MAX_LEGS = 3
DEFAULT_YEARLY_SAMPLE_RATE = 0.1 # Default to 10% of dates per year
DAYS_PER_CHUNK = 30 # Process dates in larger chunks for checkpointing efficiency
MAX_COMBINATIONS_PER_MATCH_SET = 50_000 # Limit potential parlay explosion

# --- Pydantic Models for Strict Schema Validation ---

class LeagueInfo(BaseModel, strict=True):
    name: str
    id: Optional[str] = None

class TeamConsolidatedDetails(BaseModel, strict=True):
    canonical_name: TeamNameStrict
    country: Optional[TeamNameStrict] = None # Allow None, but must be valid type if present
    statarea_id: Optional[str] = None
    mongodb_id: Optional[str] = None
    leagues: List[LeagueInfo] = Field(default_factory=list)
    alt_names: List[TeamNameStrict] = Field(default_factory=list)

    class Config:
        validate_assignment = True # Re-validate on attribute assignment

# Use RootModel for top-level dictionary validation
ConsolidatedTeamInfoSchema = Dict[TeamNameStrict, TeamConsolidatedDetails]

# --- Team Information Resolver ---
class TeamInfoResolver:
    """
    Manages loading, validation, and resolving of consolidated team information.
    Provides methods to get canonical names, countries, and parse MatchIDs.
    Designed to be picklable for multiprocessing.
    """
    def __init__(self, consolidated_info_path: Path):
        assert consolidated_info_path.exists(), f"Consolidated team info file not found: {consolidated_info_path}"
        self._info_path_str = str(consolidated_info_path) # Store as string for pickling
        self._team_data: ConsolidatedTeamInfoSchema = self._load_and_validate_team_data()
        self._normalized_lookup: Dict[str, TeamNameStrict] = {}
        self._raw_forms_for_splitting: List[str] = []
        self._build_lookups()
        logger.info(f"TeamInfoResolver initialized with {len(self._team_data)} canonical teams.")

    def _load_and_validate_team_data(self) -> ConsolidatedTeamInfoSchema:
        try:
            with open(self._info_path_str, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            assert isinstance(raw_data, dict), "Consolidated team info must be a JSON object (dictionary)."

            validated_data: ConsolidatedTeamInfoSchema = {}
            validation_errors = 0
            for canonical_key, details_dict in raw_data.items():
                try:
                    # Ensure canonical_name field matches the key
                    details_dict['canonical_name'] = canonical_key
                    team_details = TeamConsolidatedDetails(**details_dict)
                    # Use the validated canonical name from the model as the key
                    validated_data[team_details.canonical_name] = team_details
                except ValidationError as e:
                    logger.warning(f"Validation failed for team key '{canonical_key}'. Skipping. Error: {e}")
                    validation_errors += 1
                except TypeError as e: # Catch potential TeamNameStrict errors during dict init
                    logger.warning(f"Type error during validation for team key '{canonical_key}'. Skipping. Error: {e}")
                    validation_errors += 1

            if validation_errors > 0:
                 logger.warning(f"Encountered {validation_errors} validation errors while loading team info.")
            assert validated_data, "No valid team data loaded after validation."
            return validated_data
        except (json.JSONDecodeError, OSError) as e:
            logger.critical(f"Fatal error loading team data from {self._info_path_str}: {e}", exc_info=True)
            raise # Re-raise critical errors

    @staticmethod
    def _normalize_name(name: Optional[str]) -> str:
        """Lowercase alphanumeric normalization."""
        if not name or pd.isna(name):
            return ""
        return "".join(filter(str.isalnum, str(name))).lower()

    def _build_lookups(self) -> None:
        """Builds normalized lookup and list for MatchID parsing."""
        raw_forms_set: Set[str] = set()
        for canonical_name, details in self._team_data.items():
            norm_canon = self._normalize_name(canonical_name)
            if norm_canon:
                self._normalized_lookup[norm_canon] = canonical_name
                raw_forms_set.add(str(canonical_name)) # Add original form

            for alt_name in details.alt_names:
                norm_alt = self._normalize_name(alt_name)
                if norm_alt:
                    self._normalized_lookup[norm_alt] = canonical_name
                    raw_forms_set.add(str(alt_name)) # Add original form

        # Sort by length descending for greedy matching in parsing
        self._raw_forms_for_splitting = sorted(list(raw_forms_set), key=len, reverse=True)

    def get_canonical_name(self, raw_team_name: Any) -> Optional[TeamNameStrict]:
        """Finds the canonical name for a given raw team name variant."""
        if not raw_team_name or pd.isna(raw_team_name): return None
        # Assert input type after checking for None/NaN
        assert isinstance(raw_team_name, str), f"Expected string input, got {type(raw_team_name)}"

        # 1. Direct canonical match (case-sensitive)
        if raw_team_name in self._team_data:
            return self._team_data[raw_team_name].canonical_name # Return the validated one

        # 2. Normalized lookup
        norm_name = self._normalize_name(raw_team_name)
        if norm_name in self._normalized_lookup:
            return self._normalized_lookup[norm_name]

        # 3. Spaced CamelCase check (simple version)
        # More complex regex can be slow, try simple approach first
        spaced_name = ' '.join(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+', raw_team_name))
        if spaced_name != raw_team_name: # Only if different
             norm_spaced = self._normalize_name(spaced_name)
             if norm_spaced in self._normalized_lookup:
                 return self._normalized_lookup[norm_spaced]

        logger.debug(f"Canonical name not found for raw: '{raw_team_name}'")
        return None

    def get_team_details(self, team_name_variant: Any) -> Optional[TeamConsolidatedDetails]:
        """Gets the full details model for a team variant."""
        canonical_name = self.get_canonical_name(team_name_variant)
        return self._team_data.get(canonical_name) if canonical_name else None

    def get_team_country(self, team_name_variant: Any) -> Optional[str]:
        """Gets the country for a team variant."""
        details = self.get_team_details(team_name_variant)
        # Return string representation of TeamNameStrict or None
        return str(details.country) if details and details.country else None

    def get_team_primary_league_name(self, team_name_variant: Any) -> Optional[str]:
        """Gets the primary league name (heuristic: first listed)."""
        details = self.get_team_details(team_name_variant)
        if details and details.leagues:
            return details.leagues[0].name # Assumes first league is primary
        return None

    def parse_match_id(self, match_id_str: str) -> Tuple[Optional[TeamNameStrict], Optional[TeamNameStrict]]:
        """
        Attempts to parse a MatchID string into home and away canonical team names.
        Relies on the comprehensiveness of `alt_names` and `_raw_forms_for_splitting`.
        """
        assert isinstance(match_id_str, str), f"MatchID must be a string, got {type(match_id_str)}"

        # Remove optional YYYYMMDD_ prefix
        teams_part = re.sub(r"^\d{8}_", "", match_id_str)

        # Strategy 1: Greedy split using known raw forms (longest first)
        for home_form in self._raw_forms_for_splitting:
            # Find potential split point based on the raw form
            if teams_part.startswith(home_form):
                potential_away_part = teams_part[len(home_form):]
                if potential_away_part: # Must have something left for away team
                    home_canonical = self.get_canonical_name(home_form)
                    away_canonical = self.get_canonical_name(potential_away_part)
                    # Check if both resolved and are different
                    if home_canonical and away_canonical and home_canonical != away_canonical:
                        return home_canonical, away_canonical

        # Strategy 2: Fallback to underscore splitting (if exactly one underscore)
        if '_' in teams_part and teams_part.count('_') == 1:
            home_raw, away_raw = teams_part.split('_', 1)
            home_canonical = self.get_canonical_name(home_raw)
            away_canonical = self.get_canonical_name(away_raw)
            if home_canonical and away_canonical and home_canonical != away_canonical:
                return home_canonical, away_canonical

        logger.debug(f"Could not reliably parse MatchID '{match_id_str}' into two known teams. Parsed part: '{teams_part}'")
        return None, None

# --- Memory Management ---
def check_memory_usage(
    critical_threshold: float = MEMORY_CRITICAL_PERCENT,
    high_threshold: float = MEMORY_HIGH_WM_PERCENT,
    min_free_gb: float = MIN_FREE_MEMORY_GB
) -> Tuple[bool, str]:
    """Checks memory usage, logs status, and returns if critical."""
    memory_info = psutil.virtual_memory()
    memory_percent = memory_info.percent
    free_memory_gb = memory_info.available / (1024**3)

    status = "normal"
    is_critical = False

    if memory_percent > critical_threshold or free_memory_gb < min_free_gb:
        status = "critical"
        is_critical = True
        logger.warning(f"CRITICAL memory usage: {memory_percent:.1f}% used, {free_memory_gb:.2f}GB free. Attempting GC.")
        gc.collect() # Attempt immediate garbage collection
        time.sleep(1) # Short pause for GC to potentially work
        # Recheck after GC
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        free_memory_gb = memory_info.available / (1024**3)
        if memory_percent > critical_threshold or free_memory_gb < min_free_gb:
             logger.error(f"Memory still CRITICAL after GC: {memory_percent:.1f}% used, {free_memory_gb:.2f}GB free.")
        else:
             logger.info(f"Memory recovered slightly after GC: {memory_percent:.1f}% used, {free_memory_gb:.2f}GB free.")
             is_critical = False # Recovered below critical
             status = "high" # Still potentially high

    elif memory_percent > high_threshold:
        status = "high"
        logger.info(f"High memory usage: {memory_percent:.1f}% used, {free_memory_gb:.2f}GB free.")

    # logger.debug(f"Memory status: {status} ({memory_percent:.1f}%, {free_memory_gb:.2f}GB free)")
    return is_critical, status


# --- Data Loading and Preprocessing ---

def load_data(
    combined_oof_path: Path,
    strategy_guide_path: Path,
    market_definitions_path: Path,
    team_info_path: Path
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict, TeamInfoResolver]]:
    """Loads all required data files with validation."""
    logger.info("--- Loading Input Data ---")
    try:
        # 1. Load OOF Data
        s_time = time.time()
        assert combined_oof_path.exists(), f"OOF file not found: {combined_oof_path}"
        combined_oof_df = pd.read_parquet(combined_oof_path, engine='pyarrow')
        logger.info(f"Loaded OOF data: {combined_oof_df.shape} (in {time.time()-s_time:.2f}s)")
        assert DEFAULT_DATE_COL in combined_oof_df.columns, f"Date column '{DEFAULT_DATE_COL}' missing."
        assert DEFAULT_MATCH_ID_COL in combined_oof_df.columns, f"MatchID column '{DEFAULT_MATCH_ID_COL}' missing."
        # Convert date column early
        combined_oof_df[DEFAULT_DATE_COL] = pd.to_datetime(combined_oof_df[DEFAULT_DATE_COL]).dt.date # Store as date objects
        logger.info(f"Date range in OOF: {combined_oof_df[DEFAULT_DATE_COL].min()} to {combined_oof_df[DEFAULT_DATE_COL].max()}")


        # 2. Load Strategy Guide
        assert strategy_guide_path.exists(), f"Strategy guide file not found: {strategy_guide_path}"
        strategy_guide_df = pd.read_csv(strategy_guide_path)
        logger.info(f"Loaded Strategy Guide: {strategy_guide_df.shape}")
        assert not strategy_guide_df.empty, "Strategy guide is empty."

        # 3. Load Market Definitions
        assert market_definitions_path.exists(), f"Market definitions file not found: {market_definitions_path}"
        with open(market_definitions_path, 'r', encoding='utf-8') as f:
            parlay_market_definitions = json.load(f)
        assert isinstance(parlay_market_definitions, dict), "Market definitions file should contain a JSON object."
        assert parlay_market_definitions, "Market definitions are empty."
        logger.info(f"Loaded {len(parlay_market_definitions)} market definitions.")

        # 4. Initialize Team Info Resolver
        team_info_resolver = TeamInfoResolver(team_info_path) # Validation happens inside

        return combined_oof_df, strategy_guide_df, parlay_market_definitions, team_info_resolver

    except AssertionError as ae:
        logger.critical(f"Data loading assertion failed: {ae}", exc_info=True)
    except Exception as e:
        logger.critical(f"Unexpected error during data loading: {e}", exc_info=True)
    return None # Return None on any failure

def preprocess_strategy_guide(
    raw_guide_df: pd.DataFrame,
    market_definitions: Dict[str, Dict],
    oof_df_columns_set: Set[str]
) -> pd.DataFrame:
    """Validates and processes the strategy guide rules."""
    logger.info("--- Pre-processing Strategy Guide ---")
    s_time = time.time()

    processed_rules: List[Dict[str, Any]] = []
    skipped_rules_count = 0

    # Identify the threshold column to use
    threshold_col = next((col for col in ['tradeoff_threshold', 'efficient_entry_threshold']
                         if col in raw_guide_df.columns), None)
    assert threshold_col is not None, "Missing threshold column ('tradeoff_threshold' or 'efficient_entry_threshold') in strategy guide."
    logger.info(f"Using threshold column: '{threshold_col}'")

    # Assert required columns exist in the raw guide
    required_cols = {'market', 'model_identifier', threshold_col}
    assert required_cols.issubset(raw_guide_df.columns), \
        f"Strategy guide missing required columns: {required_cols - set(raw_guide_df.columns)}"

    for _, rule in raw_guide_df.iterrows():
        market_name = str(rule['market']).strip()
        model_id = str(rule['model_identifier']).strip()
        entry_threshold = rule[threshold_col]

        # Basic validation of rule values
        if not market_name or not model_id or pd.isna(entry_threshold):
            logger.warning(f"Skipping rule with missing market, model, or threshold: {rule.to_dict()}")
            skipped_rules_count += 1
            continue

        try:
             entry_threshold = float(entry_threshold)
             assert 0.0 <= entry_threshold <= 1.0, "Entry threshold must be between 0 and 1."
        except (ValueError, TypeError, AssertionError) as e:
             logger.warning(f"Skipping rule with invalid threshold '{entry_threshold}' for market '{market_name}': {e}")
             skipped_rules_count += 1
             continue

        market_info = market_definitions.get(market_name)
        if not market_info:
            logger.warning(f"Market '{market_name}' not found in definitions. Skipping rule.")
            skipped_rules_count += 1
            continue

        # Assert expected keys exist in market_info
        expected_keys = {'prob_suffix', 'target_col'}
        assert expected_keys.issubset(market_info.keys()), \
            f"Market definition for '{market_name}' missing keys: {expected_keys - market_info.keys()}"

        prob_col = f"{model_id}_{market_info['prob_suffix']}"
        target_col = market_info['target_col']

        # Critical Check: Ensure columns exist in the OOF data
        if prob_col not in oof_df_columns_set:
            logger.warning(f"Probability column '{prob_col}' (Market: {market_name}, Model: {model_id}) not in OOF data. Skipping rule.")
            skipped_rules_count += 1
            continue
        if target_col not in oof_df_columns_set:
            logger.warning(f"Target column '{target_col}' (Market: {market_name}) not in OOF data. Skipping rule.")
            skipped_rules_count += 1
            continue

        processed_rules.append({
            'market': market_name,
            'model_identifier': model_id,
            'entry_threshold': entry_threshold,
            'prob_col': prob_col,
            'target_col': target_col,
            # Use market name as default conflict group if not specified
            'conflict_group': market_info.get('conflict_group', market_name),
        })

    assert processed_rules, "No valid strategy rules were processed. Check strategy guide format, market definitions, and OOF columns."
    result_df = pd.DataFrame(processed_rules)
    logger.info(f"Strategy guide pre-processed: {len(result_df)} valid rules (in {time.time()-s_time:.2f}s). Skipped {skipped_rules_count} rules.")
    return result_df

def get_match_details(match_id: str, resolver: TeamInfoResolver) -> Dict[str, Optional[str]]:
    """Extracts canonical teams, country, and league from a MatchID using the resolver."""
    home_canon, away_canon = resolver.parse_match_id(match_id)

    home_country, away_country = None, None
    home_league, away_league = None, None

    if home_canon:
        home_details = resolver.get_team_details(home_canon)
        if home_details:
            home_country = str(home_details.country) if home_details.country else None
            if home_details.leagues:
                home_league = home_details.leagues[0].name

    if away_canon:
        away_details = resolver.get_team_details(away_canon)
        if away_details:
            away_country = str(away_details.country) if away_details.country else None
            if away_details.leagues:
                away_league = away_details.leagues[0].name

    # Determine overall match context (prioritize home team info if available)
    match_country = home_country if home_country else away_country
    match_league = home_league if home_league else away_league

    return {
        "home_team_canonical": str(home_canon) if home_canon else None,
        "away_team_canonical": str(away_canon) if away_canon else None,
        "match_country": match_country,
        "match_league": match_league,
        # Keep individual team info too if needed later
        # "home_country": home_country,
        # "away_country": away_country,
        # "home_league": home_league,
        # "away_league": away_league,
    }


# --- Core Parlay Generation Logic (Optimized) ---

def find_eligible_legs_for_day(
    daily_games_df: pd.DataFrame,
    strategy_guide: pd.DataFrame,
    team_resolver: TeamInfoResolver,
    match_id_col: str
) -> List[Dict]:
    """
    Identifies all eligible single legs for a given day based on the strategy guide.
    Optimized to work with minimal data per match.
    """
    eligible_legs = []
    required_cols = {match_id_col} | set(strategy_guide['prob_col']) | set(strategy_guide['target_col'])
    
    # Ensure necessary columns exist, filter DataFrame to only these columns
    available_cols = [col for col in required_cols if col in daily_games_df.columns]
    daily_minimal_df = daily_games_df[available_cols].copy() # Work on a minimal copy

    # Pre-calculate match details to avoid redundant calls inside the loop
    match_details_cache = {
        match_id: get_match_details(match_id, team_resolver)
        for match_id in daily_minimal_df[match_id_col].unique()
    }

    # Iterate through strategy rules (usually fewer than matches)
    for _, rule in strategy_guide.iterrows():
        prob_col = rule['prob_col']
        target_col = rule['target_col']
        threshold = rule['entry_threshold']
        market = rule['market']
        model_id = rule['model_identifier']
        conflict_group = rule['conflict_group']

        # Check if needed columns for this rule are present
        if prob_col not in daily_minimal_df.columns or target_col not in daily_minimal_df.columns:
            continue # Skip rule if columns missing

        # Vectorized check for eligible legs for this specific rule
        eligible_matches_for_rule = daily_minimal_df[
            pd.notna(daily_minimal_df[prob_col]) & (daily_minimal_df[prob_col] >= threshold)
        ]

        # Construct leg info for matches meeting the rule criteria
        for _, match_row in eligible_matches_for_rule.iterrows():
            match_id = match_row[match_id_col]
            leg_info = {
                'match_id': match_id,
                'market': market,
                'model_used': model_id,
                'prob_at_bet': float(match_row[prob_col]),
                'threshold_used': float(threshold),
                'actual_outcome': int(match_row[target_col]) if pd.notna(match_row[target_col]) else -1, # Use -1 for missing outcome
                'conflict_group': conflict_group,
            }
            # Add pre-calculated match details
            leg_info.update(match_details_cache.get(match_id, {}))
            eligible_legs.append(leg_info)

    # del daily_minimal_df # Explicitly delete minimal df
    # gc.collect() # Optional: Collect garbage if memory is very tight

    return eligible_legs


def generate_parlays_for_day(
    eligible_legs: List[Dict],
    game_date: Union[str, pd.Timestamp],
    min_legs: int,
    max_legs: int,
    max_combinations_limit: int
) -> List[Dict]:
    """
    Generates valid parlays from a list of eligible legs for a single day.
    Includes logic to prevent combination explosion and conflicting legs.
    """
    if not eligible_legs:
        return []

    parlay_results = []
    legs_by_match: Dict[str, List[Dict]] = {}
    for leg in eligible_legs:
        match_id = leg['match_id']
        if match_id not in legs_by_match:
            legs_by_match[match_id] = []
        legs_by_match[match_id].append(leg)

    unique_match_ids = list(legs_by_match.keys())
    
    # Enhanced logging: Log number of unique matches and total eligible legs
    logger.info(f"Date {game_date}: Processing {len(unique_match_ids)} unique matches with {len(eligible_legs)} total eligible legs")
    
    # Log average legs per match to identify potential explosion sources
    avg_legs_per_match = len(eligible_legs) / len(unique_match_ids) if unique_match_ids else 0
    logger.info(f"Date {game_date}: Average eligible legs per match: {avg_legs_per_match:.2f}")

    # Ensure date is string
    game_date_str = game_date.strftime('%Y-%m-%d') if hasattr(game_date, 'strftime') else str(game_date)

    # Iterate through desired number of legs (parlay size)
    for num_legs in range(min_legs, min(max_legs + 1, len(unique_match_ids) + 1)):
        # Log the number of legs being processed
        logger.debug(f"Date {game_date_str}: Processing for {num_legs}-leg parlays")
        
        # Generate combinations of unique matches for this parlay size
        match_id_combinations = list(combinations(unique_match_ids, num_legs))
        logger.debug(f"Date {game_date_str}: Number of {num_legs}-match combinations: {len(match_id_combinations)}")

        # Process combinations for this number of legs
        combination_count_for_num_legs = 0
        skipped_combinations = 0
        
        for match_combo_tuple in match_id_combinations:
            # Get the list of eligible legs for each match in the current combination
            leg_options_for_combo: List[List[Dict]] = [legs_by_match[mid] for mid in match_combo_tuple]

            # Calculate potential number of parlays for this match combination *before* generating them
            num_potential_parlays = np.prod([len(opts) for opts in leg_options_for_combo], dtype=np.int64)

            if num_potential_parlays == 0: continue

            # Check against the combination limit for this specific set of matches
            if num_potential_parlays > max_combinations_limit:
                 logger.warning(f"Date {game_date_str}: Skipping match combo {match_combo_tuple} "
                                f"due to excessive combinations: {num_potential_parlays} > {max_combinations_limit}")
                 skipped_combinations += 1
                 continue

            combination_count_for_num_legs += num_potential_parlays
            
            # Generate all valid combinations of one leg from each selected match
            valid_parlays_for_combo = 0
            for leg_combination in product(*leg_options_for_combo):
                # Check for conflicting market groups
                conflict_groups_in_parlay = [leg['conflict_group'] for leg in leg_combination]
                if len(set(conflict_groups_in_parlay)) != len(conflict_groups_in_parlay):
                    continue

                # Process valid parlay
                valid_parlays_for_combo += 1
                # If checks pass, calculate parlay outcome and details
                parlay_won = all(leg['actual_outcome'] == 1 for leg in leg_combination)
                # Handle cases where prob might be missing (though shouldn't happen if filtered earlier)
                valid_probs = [leg['prob_at_bet'] for leg in leg_combination if pd.notna(leg.get('prob_at_bet'))]
                avg_prob = np.mean(valid_probs) if valid_probs else 0.0

                # Use match details from the first leg as representative for the parlay
                # This is an approximation, as parlays can span countries/leagues
                first_leg = leg_combination[0]
                parlay_country = first_leg.get('match_country')
                parlay_league = first_leg.get('match_league')

                # Construct the result record
                record = {
                    'parlay_date': game_date_str,
                    'num_legs': num_legs,
                    'parlay_won': int(parlay_won),
                    'avg_prob': float(avg_prob),
                    'parlay_country': parlay_country if parlay_country else 'Unknown',
                    'parlay_league': parlay_league if parlay_league else 'Unknown',
                    # Create a sorted, unique representation of markets
                    'market_combination': '+'.join(sorted({leg['market'] for leg in leg_combination}))
                }

                # Add individual leg details
                for i, leg in enumerate(leg_combination, 1):
                    record[f'leg{i}_match_id'] = leg['match_id']
                    record[f'leg{i}_market'] = leg['market']
                    record[f'leg{i}_model'] = leg['model_used']
                    record[f'leg{i}_prob'] = float(leg['prob_at_bet'])
                    record[f'leg{i}_won'] = int(leg['actual_outcome'])
                    record[f'leg{i}_country'] = leg.get('match_country', 'Unknown')
                    record[f'leg{i}_league'] = leg.get('match_league', 'Unknown')

                parlay_results.append(record)

            logger.debug(f"Date {game_date_str}: Generated {valid_parlays_for_combo} valid parlays for match combo {match_combo_tuple}")

        # Log summary for this leg size
        logger.info(f"Date {game_date_str}: For {num_legs}-leg parlays - Generated {combination_count_for_num_legs} combinations, skipped {skipped_combinations} excessive combos")

    return parlay_results


# --- Main Backtesting Orchestration (Parallel / Chunked) ---

def sample_dates_yearly(
    all_dates: pd.Series,
    sample_rate: float
) -> List[pd.Timestamp]:
    """Samples a percentage of dates from each year present in the data."""
    assert 0.0 < sample_rate <= 1.0, "Sample rate must be between 0 and 1."
    if sample_rate == 1.0:
        logger.info("Sampling rate is 1.0, using all available dates.")
        return sorted(all_dates.unique().tolist())

    all_dates_df = pd.DataFrame({'date': pd.to_datetime(all_dates).dt.date}).drop_duplicates()
    all_dates_df['year'] = pd.to_datetime(all_dates_df['date']).dt.year

    sampled_dates = []
    np.random.seed(42) # For reproducible sampling

    for year, group in all_dates_df.groupby('year'):
        n_dates_in_year = len(group)
        n_to_sample = max(1, int(np.round(n_dates_in_year * sample_rate))) # Sample at least 1 date per year if possible
        sampled_indices = np.random.choice(group.index, size=n_to_sample, replace=False)
        sampled_dates.extend(group.loc[sampled_indices, 'date'].tolist())
        logger.debug(f"Year {year}: Sampled {n_to_sample}/{n_dates_in_year} dates.")

    unique_sampled_dates = sorted(list(set(sampled_dates)))
    logger.info(f"Sampled a total of {len(unique_sampled_dates)} unique dates across all years "
                f"(Target sample rate: {sample_rate * 100:.1f}% per year).")
    return unique_sampled_dates


def run_parlay_backtest_chunked(
    combined_oof_df: pd.DataFrame,
    strategy_guide_df: pd.DataFrame,
    team_info_resolver: TeamInfoResolver,
    date_col: str,
    match_id_col: str,
    min_legs: int,
    max_legs: int,
    yearly_sample_rate: float,
    days_per_chunk: int,
    max_combinations_limit: int,
    checkpoint_dir: Path,
    output_path_parquet: Path,
    output_path_csv: Path,
    reuse_checkpoints: bool = True
) -> Optional[pd.DataFrame]:
    """
    Runs the parlay backtest using yearly sampling, chunked processing,
    and checkpointing.
    """
    logger.info("\n" + "="*30 + " Starting Chunked Parlay Backtest " + "="*30)
    overall_start_time = time.time()

    # --- Setup ---
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path_parquet.parent.mkdir(parents=True, exist_ok=True)

    # --- Date Sampling ---
    all_available_dates = combined_oof_df[date_col].drop_duplicates()
    dates_to_process = sample_dates_yearly(all_available_dates, yearly_sample_rate)
    if not dates_to_process:
        logger.warning("No dates selected after sampling. Exiting.")
        return None
    del all_available_dates # Free memory
    gc.collect()

    # --- Data Filtering (Early) ---
    # Filter the main DataFrame *once* to include only sampled dates
    logger.info(f"Filtering OOF data to {len(dates_to_process)} sampled dates...")
    oof_filtered_by_date = combined_oof_df[combined_oof_df[date_col].isin(dates_to_process)].copy()
    del combined_oof_df # Free memory of the original large DF
    gc.collect()
    logger.info(f"Filtered OOF DataFrame shape: {oof_filtered_by_date.shape}")
    if oof_filtered_by_date.empty:
        logger.warning("OOF DataFrame is empty after filtering by sampled dates. Exiting.")
        return None

    # Filter to essential columns needed for the entire process
    essential_cols = {date_col, match_id_col} | set(strategy_guide_df['prob_col']) | set(strategy_guide_df['target_col'])
    available_essential_cols = [col for col in essential_cols if col in oof_filtered_by_date.columns]
    logger.info(f"Reducing OOF DataFrame to {len(available_essential_cols)} essential columns.")
    oof_minimal = oof_filtered_by_date[available_essential_cols].copy()
    del oof_filtered_by_date # Free memory
    gc.collect()
    logger.info(f"Minimal OOF DataFrame shape for processing: {oof_minimal.shape}")


    # --- Chunk Definition ---
    date_chunks: List[Tuple[str, List[pd.Timestamp]]] = []
    for i in range(0, len(dates_to_process), days_per_chunk):
        chunk_dates = dates_to_process[i : i + days_per_chunk]
        if chunk_dates:
            start_date_str = chunk_dates[0].strftime('%Y%m%d')
            end_date_str = chunk_dates[-1].strftime('%Y%m%d')
            chunk_id = f"chunk_{start_date_str}_to_{end_date_str}"
            date_chunks.append((chunk_id, chunk_dates))
    logger.info(f"Divided {len(dates_to_process)} dates into {len(date_chunks)} chunks for processing.")

    # --- Processing Chunks ---
    all_results_dfs: List[pd.DataFrame] = []
    processed_chunk_ids: Set[str] = set()

    for chunk_index, (chunk_id, chunk_dates) in enumerate(tqdm(date_chunks, desc="Processing Chunks")):
        chunk_start_time = time.time()
        checkpoint_file = checkpoint_dir / f"{chunk_id}.parquet"

        # Check for existing checkpoint
        if reuse_checkpoints and checkpoint_file.exists():
            try:
                logger.info(f"Chunk {chunk_index+1}/{len(date_chunks)} (ID: {chunk_id}): Loading from checkpoint.")
                chunk_df = pd.read_parquet(checkpoint_file)
                all_results_dfs.append(chunk_df)
                processed_chunk_ids.add(chunk_id)
                logger.info(f"Loaded {len(chunk_df):,} parlays from checkpoint.")
                continue # Skip to next chunk
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}. Re-processing chunk.")

        # Process the chunk if not loaded from checkpoint
        logger.info(f"Chunk {chunk_index+1}/{len(date_chunks)} (ID: {chunk_id}): Processing {len(chunk_dates)} dates.")

        # Get data for the current chunk
        chunk_oof_df = oof_minimal[oof_minimal[date_col].isin(chunk_dates)].copy()
        if chunk_oof_df.empty:
             logger.info(f"Chunk {chunk_id}: No OOF data for dates in this chunk. Skipping.")
             continue

        chunk_parlay_results: List[Dict] = []
        # Process day by day *within* the chunk (safer for memory than parallel days)
        for game_date in sorted(chunk_oof_df[date_col].unique()):
            is_critical, mem_status = check_memory_usage()
            if is_critical:
                logger.error(f"CRITICAL memory ({mem_status}) before processing date {game_date}. Skipping remaining dates in chunk {chunk_id}.")
                break # Stop processing this chunk

            daily_games_df = chunk_oof_df[chunk_oof_df[date_col] == game_date] # No copy needed here if chunk_oof_df is already a copy
            if daily_games_df.empty: continue

            try:
                # 1. Find eligible legs for the day
                eligible_legs = find_eligible_legs_for_day(
                    daily_games_df=daily_games_df,
                    strategy_guide=strategy_guide_df,
                    team_resolver=team_info_resolver,
                    match_id_col=match_id_col
                )

                # 2. Limit to top 3 legs per match
                limited_eligible_legs = limit_legs_per_match(eligible_legs, max_legs_per_match=3)

                # 3. Generate parlays if legs found
                if limited_eligible_legs:
                    daily_parlays = generate_parlays_for_day(
                        eligible_legs=limited_eligible_legs,
                        game_date=game_date,
                        min_legs=min_legs,
                        max_legs=max_legs,
                        max_combinations_limit=max_combinations_limit
                    )
                    if daily_parlays:
                        chunk_parlay_results.extend(daily_parlays)
                        # logger.debug(f"Date {game_date}: Generated {len(daily_parlays)} parlays.")

            except Exception as e:
                 logger.error(f"Error processing date {game_date} in chunk {chunk_id}: {e}", exc_info=True)
                 # Continue to the next date

            # Clean up daily data explicitly
            del daily_games_df
            # gc.collect() # Optional: more aggressive GC

        # --- Save Chunk Results ---
        if chunk_parlay_results:
            chunk_df = pd.DataFrame(chunk_parlay_results)
            try:
                chunk_df.to_parquet(checkpoint_file, index=False)
                all_results_dfs.append(chunk_df)
                processed_chunk_ids.add(chunk_id)
                logger.info(f"Chunk {chunk_id}: Processed and saved {len(chunk_df):,} parlays in {time.time()-chunk_start_time:.2f}s.")
            except Exception as e:
                logger.error(f"Failed to save checkpoint {checkpoint_file}: {e}")
        else:
            logger.info(f"Chunk {chunk_id}: No parlays generated. Completed in {time.time()-chunk_start_time:.2f}s.")
            # Optionally save an empty checkpoint file to mark as done
            # checkpoint_file.touch()

        # Clean up chunk data
        del chunk_oof_df
        gc.collect()


    # --- Final Aggregation ---
    logger.info(f"Finished processing all {len(date_chunks)} chunks.")
    if not all_results_dfs:
        logger.warning("No parlay results generated across all chunks.")
        return None

    try:
        logger.info(f"Concatenating results from {len(all_results_dfs)} chunk DataFrames...")
        final_df = pd.concat(all_results_dfs, ignore_index=True)
        logger.info(f"Successfully concatenated results. Final DataFrame shape: {final_df.shape}")

        # Basic validation of final DataFrame
        assert not final_df.empty
        assert 'parlay_date' in final_df.columns
        assert 'parlay_won' in final_df.columns

        # Save final results
        logger.info(f"Saving final results ({final_df.shape[0]:,} parlays) to {output_path_parquet}")
        final_df.to_parquet(output_path_parquet, index=False)

        # Save CSV version (optional, potentially large)
        try:
            logger.info(f"Saving CSV version to {output_path_csv}")
            # Select potentially fewer columns for CSV if it's huge
            cols_to_save_csv = final_df.columns.tolist()
            if len(cols_to_save_csv) > 50: # Arbitrary limit
                 cols_to_save_csv = [c for c in cols_to_save_csv if not (c.startswith('leg') and ('country' in c or 'league' in c))]
                 logger.info(f"Saving reduced columns ({len(cols_to_save_csv)}) to CSV.")

            final_df[cols_to_save_csv].to_csv(output_path_csv, index=False)
        except Exception as e:
            logger.warning(f"Could not save CSV version: {e}")

        total_time = time.time() - overall_start_time
        logger.info(f"Parlay backtest completed successfully in {total_time:.2f} seconds ({total_time/60:.2f} minutes).")
        return final_df

    except Exception as e:
        logger.critical(f"FATAL: Error during final aggregation or saving: {e}", exc_info=True)
        return None

# --- Analysis and Plotting (Keep existing functions, ensure they handle potential None/empty DF) ---
def create_parlay_visualizations(results_df: Optional[pd.DataFrame], output_viz_dir: Path):
    if results_df is None or results_df.empty:
        logger.warning("No data available for visualization. Skipping plot generation.")
        return

    logger.info(f"Generating visualizations in: {output_viz_dir}")
    output_viz_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis") # Example palette

    # --- Plot 1: Win Rate by Number of Legs ---
    try:
        plt.figure(figsize=(10, 6))
        # Ensure 'num_legs' and 'parlay_won' exist
        if 'num_legs' in results_df.columns and 'parlay_won' in results_df.columns:
            summary = results_df.groupby('num_legs')['parlay_won'].agg(['count', 'mean']).reset_index()
            summary['mean'] *= 100 # Convert to percentage

            if not summary.empty:
                max_mean_value = summary['mean'].max()
                y_limit = max(max_mean_value * 1.15, 10) # Ensure some room, min 10% limit

                ax = sns.barplot(x='num_legs', y='mean', data=summary, hue='num_legs', palette="viridis", dodge=False, legend=False)
                # Add annotations
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)
                    # Add count below or above? Let's add near bar top
                    for i, bar in enumerate(container):
                        height = bar.get_height()
                        count = summary.loc[i, 'count']
                        ax.text(bar.get_x() + bar.get_width() / 2., height + y_limit*0.02, f'(n={count:,})',
                                ha='center', va='bottom', fontsize=9)


                plt.title('Parlay Win Rate by Number of Legs')
                plt.xlabel('Number of Legs')
                plt.ylabel('Win Rate (%)')
                plt.ylim(0, y_limit)
                plt.tight_layout()
                plt.savefig(output_viz_dir / 'win_rate_by_legs.png')
            else:
                 logger.warning("Not enough data to plot 'Win Rate by Number of Legs'.")
        else:
            logger.warning("Missing 'num_legs' or 'parlay_won' column for 'Win Rate by Legs' plot.")
        plt.close()
    except Exception as e:
        logger.error(f"Error generating 'Win Rate by Legs' plot: {e}", exc_info=True)
        plt.close() # Ensure plot is closed on error

    # --- Plot 2: Win Rate by Predicted Probability ---
    try:
        if 'avg_prob' in results_df.columns and not results_df['avg_prob'].isnull().all():
            plt.figure(figsize=(10, 6))
            # Define bins dynamically based on data range, or use fixed sensible bins
            min_prob, max_prob = results_df['avg_prob'].min(), results_df['avg_prob'].max()
            # Ensure bins cover the full range, e.g., 0.4 to 1.0 in steps of 0.1
            bins = np.arange(max(0.4, np.floor(min_prob*10)/10), min(1.01, np.ceil(max_prob*10)/10 + 0.1), 0.1)
            if len(bins) < 2: bins = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Fallback bins

            results_df['prob_bin'] = pd.cut(results_df['avg_prob'], bins=bins, right=False, include_lowest=True)
            prob_summary = results_df.groupby('prob_bin', observed=False)['parlay_won'].agg(['count', 'mean']).reset_index()
            prob_summary['mean'] *= 100

            if not prob_summary.empty:
                 ax = sns.barplot(x='prob_bin', y='mean', data=prob_summary, hue='prob_bin', palette="viridis", dodge=False, legend=False)
                 # Add annotations
                 for container in ax.containers:
                     ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)
                     for i, bar in enumerate(container):
                         height = bar.get_height()
                         count = prob_summary.loc[i, 'count']
                         ax.text(bar.get_x() + bar.get_width() / 2., height + ax.get_ylim()[1]*0.02, f'n={count:,}',
                                 ha='center', va='bottom', fontsize=9)

                 plt.title('Parlay Win Rate by Avg. Predicted Probability')
                 plt.xlabel('Avg. Predicted Probability Bin')
                 plt.ylabel('Win Rate (%)')
                 plt.xticks(rotation=45, ha="right")
                 plt.tight_layout()
                 plt.savefig(output_viz_dir / 'win_rate_by_avg_prob.png')
            else:
                 logger.warning("Not enough data after binning for 'Win Rate by Avg Prob' plot.")
        else:
            logger.warning("Missing 'avg_prob' column or all null values for 'Win Rate by Avg Prob' plot.")
        plt.close()
    except Exception as e:
        logger.error(f"Error generating 'Win Rate by Avg Prob' plot: {e}", exc_info=True)
        plt.close()


    # --- Plot 3: Win Rate by Country ---
    try:
        if 'parlay_country' in results_df.columns and not results_df['parlay_country'].isnull().all():
            plt.figure(figsize=(12, 8)) # Adjusted size
            min_parlays_country = 50 # Minimum parlays to include a country
            country_summary = results_df.groupby('parlay_country')['parlay_won'].agg(['count', 'mean']).reset_index()
            country_summary['mean'] *= 100
            # Filter, sort, and take top N
            country_summary = country_summary[
                 (country_summary['count'] >= min_parlays_country) & (country_summary['parlay_country'] != 'Unknown')
            ].sort_values('mean', ascending=False).head(20) # Top 20 countries

            if not country_summary.empty:
                ax = sns.barplot(x='mean', y='parlay_country', data=country_summary, hue='parlay_country', palette="magma", dodge=False, legend=False, orient='h')
                # Add annotations (win rate and count)
                for i, bar in enumerate(ax.patches):
                     win_rate = country_summary.iloc[i]['mean']
                     count = country_summary.iloc[i]['count']
                     ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2.,
                             f"{win_rate:.1f}% (n={count:,})",
                             ha='left', va='center', fontsize=9)

                plt.title(f'Top Parlay Win Rates by Match Country (Min {min_parlays_country} Parlays)')
                plt.xlabel('Win Rate (%)')
                plt.ylabel('Country')
                # Adjust xlim to make space for labels
                plt.xlim(0, country_summary['mean'].max() * 1.2)
                plt.tight_layout()
                plt.savefig(output_viz_dir / 'win_rate_by_country.png')
            else:
                logger.warning(f"Not enough data per country (min {min_parlays_country} parlays) for 'Win Rate by Country' plot.")
        else:
            logger.warning("Missing 'parlay_country' column for 'Win Rate by Country' plot.")
        plt.close()
    except Exception as e:
        logger.error(f"Error generating 'Win Rate by Country' plot: {e}", exc_info=True)
        plt.close()


    # --- Plot 4: Win Rate by League ---
    try:
        if 'parlay_league' in results_df.columns and not results_df['parlay_league'].isnull().all():
            plt.figure(figsize=(14, 10)) # Adjusted size
            min_parlays_league = 30 # Minimum parlays for a league
            league_summary = results_df.groupby('parlay_league')['parlay_won'].agg(['count', 'mean']).reset_index()
            league_summary['mean'] *= 100
            league_summary = league_summary[
                (league_summary['count'] >= min_parlays_league) & (league_summary['parlay_league'] != 'Unknown')
            ].sort_values('mean', ascending=False).head(25) # Top 25 leagues

            if not league_summary.empty:
                ax = sns.barplot(x='mean', y='parlay_league', data=league_summary, hue='parlay_league', palette="rocket", dodge=False, legend=False, orient='h')
                 # Add annotations
                for i, bar in enumerate(ax.patches):
                     win_rate = league_summary.iloc[i]['mean']
                     count = league_summary.iloc[i]['count']
                     ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2.,
                             f"{win_rate:.1f}% (n={count:,})",
                             ha='left', va='center', fontsize=9)

                plt.title(f'Top Parlay Win Rates by Match League (Min {min_parlays_league} Parlays)')
                plt.xlabel('Win Rate (%)')
                plt.ylabel('League')
                plt.xlim(0, league_summary['mean'].max() * 1.2) # Adjust xlim
                plt.tight_layout()
                plt.savefig(output_viz_dir / 'win_rate_by_league.png')
            else:
                 logger.warning(f"Not enough data per league (min {min_parlays_league} parlays) for 'Win Rate by League' plot.")
        else:
            logger.warning("Missing 'parlay_league' column for 'Win Rate by League' plot.")
        plt.close()
    except Exception as e:
        logger.error(f"Error generating 'Win Rate by League' plot: {e}", exc_info=True)
        plt.close()

    # --- Plot 5: Model Usage Analysis (if leg data exists) ---
    # This requires leg-specific model columns (e.g., leg1_model, leg2_model...)
    # Find max legs from column names if 'num_legs' might be unreliable
    max_legs_from_cols = 0
    leg_model_cols = [col for col in results_df.columns if col.startswith('leg') and col.endswith('_model')]
    if leg_model_cols:
        try:
             max_legs_from_cols = max([int(re.search(r'leg(\d+)_model', col).group(1)) for col in leg_model_cols])
        except: pass # Ignore errors if parsing fails

    if max_legs_from_cols > 0:
         try:
            all_models, winning_models = analyze_model_usage_in_parlays(results_df, max_legs_from_cols)

            if all_models is not None and not all_models.empty:
                plt.figure(figsize=(12, 7))
                all_models.sort_values().plot(kind='barh', title='Model Usage Frequency in All Parlay Legs', color='skyblue')
                plt.xlabel('Percentage of Legs (%)')
                plt.tight_layout()
                plt.savefig(output_viz_dir / 'model_usage_all_parlays.png')
                plt.close()

            if winning_models is not None and not winning_models.empty:
                plt.figure(figsize=(12, 7))
                winning_models.sort_values().plot(kind='barh', title='Model Usage Frequency in Winning Parlay Legs', color='lightgreen')
                plt.xlabel('Percentage of Legs (%)')
                plt.tight_layout()
                plt.savefig(output_viz_dir / 'model_usage_winning_parlays.png')
                plt.close()
         except Exception as e:
             logger.error(f"Error generating model usage plots: {e}", exc_info=True)
             plt.close() # Ensure plot closure even on error
    else:
        logger.warning("Could not determine max legs from columns, skipping model usage plots.")

    # --- Plot 6: Selection Analysis Plots ---
    try:
        # Ensure the function exists and handles potential errors
        if 'create_selection_visualizations' in globals():
             create_selection_visualizations(results_df, output_viz_dir)
        else:
             logger.warning("`create_selection_visualizations` function not found. Skipping.")
    except Exception as e:
        logger.error(f"Error calling create_selection_visualizations: {e}", exc_info=True)

    logger.info(f"Visualizations generation complete. Files saved in {output_viz_dir}")


def analyze_model_usage_in_parlays(
    parlay_results_df: pd.DataFrame,
    max_legs_in_data: int
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Analyzes model usage in all and winning parlays."""
    # This function remains largely the same as before, ensure input df is checked
    if parlay_results_df is None or parlay_results_df.empty:
        logger.warning("No parlay results data to analyze model usage.")
        return None, None
    if max_legs_in_data <= 0:
         logger.warning("Invalid max_legs_in_data provided for model usage analysis.")
         return None, None

    all_leg_models: List[str] = []
    winning_leg_models: List[str] = []
    found_model_cols = False

    for i in range(1, max_legs_in_data + 1):
        model_col = f'leg{i}_model'
        won_col = f'parlay_won' # Use overall parlay win status

        if model_col in parlay_results_df.columns and won_col in parlay_results_df.columns:
            found_model_cols = True
            # Get models from all legs at position i
            all_models_at_pos_i = parlay_results_df[model_col].dropna().astype(str).tolist()
            all_leg_models.extend(all_models_at_pos_i)

            # Get models from winning parlays at position i
            winning_models_at_pos_i = parlay_results_df.loc[parlay_results_df[won_col] == 1, model_col].dropna().astype(str).tolist()
            winning_leg_models.extend(winning_models_at_pos_i)
        # else: logger.debug(f"Column {model_col} not found for model usage analysis.") # Optional debug

    if not found_model_cols:
        logger.warning("No leg-specific model columns found (e.g., 'leg1_model'). Cannot analyze model usage.")
        return None, None

    all_usage_series = pd.Series(all_leg_models).value_counts(normalize=True).mul(100) if all_leg_models else None
    winning_usage_series = pd.Series(winning_leg_models).value_counts(normalize=True).mul(100) if winning_leg_models else None

    return all_usage_series, winning_usage_series

def create_selection_visualizations(results_df: Optional[pd.DataFrame], output_viz_dir: Path):
    """Create visualizations focused on market selections."""
    # This function remains largely the same as before, ensure input df is checked
    if results_df is None or results_df.empty:
        logger.warning("No data available for selection visualizations.")
        return

    selection_viz_dir = output_viz_dir / 'selections'
    selection_viz_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating selection visualizations in: {selection_viz_dir}")

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Extract Market Selections and Win Status ---
    all_selections = []
    max_legs_from_cols = 0
    leg_market_cols = [col for col in results_df.columns if col.startswith('leg') and col.endswith('_market')]
    leg_won_cols = [col for col in results_df.columns if col.startswith('leg') and col.endswith('_won')] # Individual leg win status

    if not leg_market_cols or not leg_won_cols:
         logger.warning("Missing leg market or leg won columns (e.g., leg1_market, leg1_won). Cannot generate detailed selection plots.")
         return

    try:
        max_legs_from_cols = max([int(re.search(r'leg(\d+)_', col).group(1)) for col in leg_market_cols])
    except:
        logger.error("Could not determine max legs from market/won columns.")
        return

    for i in range(1, max_legs_from_cols + 1):
        market_col = f'leg{i}_market'
        won_col = f'leg{i}_won'
        prob_col = f'leg{i}_prob'
        country_col = f'leg{i}_country' # Assuming leg-specific country exists

        if market_col in results_df.columns and won_col in results_df.columns:
             # Add probability and country if they exist
             cols_to_extract = [market_col, won_col]
             if prob_col in results_df.columns: cols_to_extract.append(prob_col)
             if country_col in results_df.columns: cols_to_extract.append(country_col)

             leg_data = results_df[cols_to_extract].copy()
             leg_data.rename(columns={market_col: 'market', won_col: 'won', prob_col: 'prob', country_col: 'country'}, inplace=True)
             leg_data['num_legs_in_parlay'] = results_df['num_legs'] # Add parlay size context
             all_selections.append(leg_data)

    if not all_selections:
        logger.warning("Could not extract any leg selection data.")
        return

    selections_df = pd.concat(all_selections, ignore_index=True).dropna(subset=['market'])
    logger.info(f"Extracted {len(selections_df):,} individual leg selections for analysis.")


    # --- Calculate Usage and Win Rates ---
    market_stats = selections_df.groupby('market')['won'].agg(['count', 'sum', 'mean']).reset_index()
    market_stats['mean'] *= 100 # Win rate percentage
    market_stats = market_stats.sort_values('count', ascending=False)

    # --- Plot Selection Usage (Top N) ---
    try:
        top_n_usage = 25
        plot_data_usage = market_stats.head(top_n_usage)
        if not plot_data_usage.empty:
             plt.figure(figsize=(12, 9))
             ax = sns.barplot(x='count', y='market', data=plot_data_usage, hue='market', palette="viridis", legend=False, orient='h')
             ax.bar_label(ax.containers[0], fmt='{:,.0f}', label_type='edge', padding=3)
             plt.title(f'Top {top_n_usage} Market Selections by Usage Count')
             plt.xlabel('Number of Times Selected in a Leg')
             plt.ylabel('Market Selection')
             plt.tight_layout()
             plt.savefig(selection_viz_dir / 'market_selection_usage.png')
        else: logger.warning("No data for market usage plot.")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting selection usage: {e}", exc_info=True)
        plt.close()


    # --- Plot Selection Win Rates (Top N with Min Count) ---
    try:
        top_n_winrate = 25
        min_usage_winrate = 50
        plot_data_winrate = market_stats[market_stats['count'] >= min_usage_winrate].sort_values('mean', ascending=False).head(top_n_winrate)
        if not plot_data_winrate.empty:
             plt.figure(figsize=(12, 9))
             ax = sns.barplot(x='mean', y='market', data=plot_data_winrate, hue='market', palette="magma", legend=False, orient='h')
             # Add annotations (win rate and count)
             for i, bar in enumerate(ax.patches):
                 win_rate = plot_data_winrate.iloc[i]['mean']
                 count = plot_data_winrate.iloc[i]['count']
                 ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2.,
                         f"{win_rate:.1f}% (n={count:,})",
                         ha='left', va='center', fontsize=9)

             plt.title(f'Top {top_n_winrate} Market Selections by Win Rate (Min {min_usage_winrate} Selections)')
             plt.xlabel('Win Rate (%)')
             plt.ylabel('Market Selection')
             plt.xlim(0, plot_data_winrate['mean'].max() * 1.15) # Adjust xlim
             plt.tight_layout()
             plt.savefig(selection_viz_dir / 'market_selection_win_rates.png')
        else: logger.warning(f"No data meeting criteria (min count {min_usage_winrate}) for market win rate plot.")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting selection win rates: {e}", exc_info=True)
        plt.close()

    # --- Plot Selection Win Rate vs Probability (for Top Markets) ---
    if 'prob' in selections_df.columns:
        top_markets_for_prob = market_stats.head(5)['market'].tolist() # Analyze top 5 used markets
        for market in top_markets_for_prob:
            try:
                market_prob_data = selections_df[selections_df['market'] == market].dropna(subset=['prob'])
                if len(market_prob_data) < 50: continue # Need sufficient data

                # Create bins
                bins = np.arange(0.4, 1.01, 0.1)
                bin_labels = [f"{int(x*100)}-{int(y*100)}%" for x, y in zip(bins[:-1], bins[1:])]
                market_prob_data['prob_bin'] = pd.cut(market_prob_data['prob'], bins=bins, labels=bin_labels, right=False, include_lowest=True)

                bin_stats = market_prob_data.groupby('prob_bin', observed=False)['won'].agg(['count', 'mean']).reset_index()
                bin_stats['mean'] *= 100
                bin_stats = bin_stats[bin_stats['count'] >= 20] # Min count per bin

                if not bin_stats.empty:
                    plt.figure(figsize=(10, 6))
                    ax = sns.barplot(x='prob_bin', y='mean', data=bin_stats, hue='prob_bin', palette="YlGnBu", legend=False)
                    # Annotations
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)
                        for i, bar in enumerate(container):
                           count = bin_stats.loc[i, 'count']
                           ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + ax.get_ylim()[1]*0.01, f'n={count:,}',
                                   ha='center', va='bottom', fontsize=9)

                    plt.title(f'Win Rate vs Predicted Probability for Selection: {market}')
                    plt.xlabel('Predicted Probability Range (Leg)')
                    plt.ylabel('Actual Win Rate (%)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(selection_viz_dir / f'selection_{market}_by_probability.png')
                else: logger.debug(f"Not enough data per bin for probability plot for market '{market}'.")
                plt.close()
            except Exception as e:
                logger.error(f"Error plotting win rate vs prob for market '{market}': {e}", exc_info=True)
                plt.close()
    else:
        logger.warning("Missing 'prob' column in selection data, skipping win rate vs probability plots.")


    logger.info(f"Selection-based visualizations saved to {selection_viz_dir}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refactored Parlay Backtester with Yearly Sampling and Efficiency Improvements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input Files
    parser.add_argument("--oof_path", type=str, default=str(OOF_INPUT_PATH_DEFAULT), help="Path to combined OOF predictions parquet file.")
    parser.add_argument("--strategy_path", type=str, default=str(STRATEGY_GUIDE_PATH_DEFAULT), help="Path to strategy guide CSV file.")
    parser.add_argument("--markets_def_path", type=str, default=str(MARKET_DEFINITIONS_PATH_DEFAULT), help="Path to parlay market definitions JSON file.")
    parser.add_argument("--team_info_path", type=str, default=str(CONSOLIDATED_TEAM_INFO_PATH_DEFAULT), help="Path to consolidated team info JSON file.")

    # Output Files & Dirs
    parser.add_argument("--output_dir", type=str, default=str(PARLAY_OUTPUT_DIR), help="Base directory for outputs (results, checkpoints, plots).")
    # parser.add_argument("--output_path", type=str, default=str(PARLAY_RESULTS_PATH_DEFAULT), help="Path to save final parlay results (Parquet).") # Determined from output_dir now
    # parser.add_argument("--viz_dir", type=str, default=str(VISUALIZATIONS_DIR_DEFAULT), help="Directory to save plots.") # Determined from output_dir now
    # parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR_DEFAULT), help="Directory for chunk checkpoints.") # Determined from output_dir now

    # Backtesting Parameters
    parser.add_argument("--max_legs", type=int, default=DEFAULT_MAX_LEGS, help="Maximum number of legs in a parlay.")
    parser.add_argument("--min_legs", type=int, default=DEFAULT_MIN_LEGS, help="Minimum number of legs in a parlay.")
    parser.add_argument("--yearly_sample_rate", type=float, default=DEFAULT_YEARLY_SAMPLE_RATE, help="Percentage of unique dates *per year* to sample (0.0 to 1.0).")
    parser.add_argument("--max_combinations", type=int, default=MAX_COMBINATIONS_PER_MATCH_SET, help="Maximum parlay combinations allowed per set of matches to prevent explosion.")


    # Data Columns
    parser.add_argument("--date_col", type=str, default=DEFAULT_DATE_COL, help="Name of the date column in OOF data.")
    parser.add_argument("--match_id_col", type=str, default=DEFAULT_MATCH_ID_COL, help="Name of the match ID column in OOF data.")

    # Performance & Control
    # parser.add_argument("--max_cpu", type=int, default=DEFAULT_MAX_CPU_WORKERS, help="Maximum CPU workers (cores) to use for parallel tasks (if implemented). Currently runs sequentially per date.") # Parallelism removed for memory safety
    parser.add_argument("--days_per_chunk", type=int, default=DAYS_PER_CHUNK, help="Number of days to process per checkpoint chunk.")
    parser.add_argument("--no_plots", action="store_true", help="Disable generation of plots.")
    parser.add_argument("--no_reuse_checkpoints", action="store_true", help="Force reprocessing of all chunks, ignoring existing checkpoints.")
    # parser.add_argument("--fallback_thresh", type=float, default=0.0, help="Fallback probability threshold (Not currently used in core logic).") # Removed as not used

    args = parser.parse_args()

    # --- Argument Validation ---
    try:
        # Input files
        assert Path(args.oof_path).exists(), f"OOF file not found: {args.oof_path}"
        assert Path(args.strategy_path).exists(), f"Strategy guide file not found: {args.strategy_path}"
        assert Path(args.markets_def_path).exists(), f"Market definitions file not found: {args.markets_def_path}"
        assert Path(args.team_info_path).exists(), f"Consolidated team info file not found: {args.team_info_path}"
        # Parameters
        assert args.min_legs >= 1, "min_legs must be at least 1."
        assert args.max_legs >= args.min_legs, "max_legs must be >= min_legs."
        assert 0.0 < args.yearly_sample_rate <= 1.0, "yearly_sample_rate must be > 0.0 and <= 1.0."
        assert args.max_combinations >= 100, "max_combinations should be reasonably large (e.g., >= 100)."
        assert args.days_per_chunk >= 1, "days_per_chunk must be at least 1."
        # Output dir
        output_base = Path(args.output_dir)
        output_base.mkdir(parents=True, exist_ok=True) # Create base output dir

        # Define output paths based on output_dir
        output_parquet_path = output_base / f"{output_base.name}_results.parquet"
        output_csv_path = output_base / f"{output_base.name}_results.csv"
        output_viz_path = output_base / "visualizations"
        output_checkpoint_path = output_base / "checkpoints"

    except AssertionError as e:
        logger.critical(f"Argument validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error setting up paths or validating args: {e}")
        sys.exit(1)

    # --- Execute Main Orchestrator ---
    try:
        # 1. Load base data
        load_result = load_data(
            Path(args.oof_path), Path(args.strategy_path),
            Path(args.markets_def_path), Path(args.team_info_path)
        )
        if load_result is None:
             raise RuntimeError("Failed to load initial data.") # More specific error
        oof_df, raw_strategy_df, market_defs, team_resolver = load_result

        # 2. Preprocess strategy guide (requires OOF columns)
        strategy_guide = preprocess_strategy_guide(
            raw_guide_df=raw_strategy_df,
            market_definitions=market_defs,
            oof_df_columns_set=set(oof_df.columns)
        )
        del raw_strategy_df # Free memory

        # 3. Run the backtest
        final_results_df = run_parlay_backtest_chunked(
            combined_oof_df=oof_df, # Pass the full loaded df here
            strategy_guide_df=strategy_guide,
            team_info_resolver=team_resolver,
            date_col=args.date_col,
            match_id_col=args.match_id_col,
            min_legs=args.min_legs,
            max_legs=args.max_legs,
            yearly_sample_rate=args.yearly_sample_rate,
            days_per_chunk=args.days_per_chunk,
            max_combinations_limit=args.max_combinations,
            checkpoint_dir=output_checkpoint_path,
            output_path_parquet=output_parquet_path,
            output_path_csv=output_csv_path,
            reuse_checkpoints=not args.no_reuse_checkpoints
        )

        # 4. Create visualizations if requested and results exist
        if not args.no_plots:
             if final_results_df is not None and not final_results_df.empty:
                  create_parlay_visualizations(final_results_df, output_viz_path)
             else:
                  logger.info("Skipping plot generation as no final results were produced or loaded.")

    except Exception as e:
        logger.critical(f"Unhandled exception during backtesting orchestration: {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- Parlay Backtester V3 (Sampled) Finished ---")
    sys.exit(0) # Explicit success exit

def limit_legs_per_match(eligible_legs: List[Dict], max_legs_per_match: int = 3) -> List[Dict]:
    """
    Limit the number of eligible legs per match to prevent combinatorial explosion.
    For each match, keep only the top N legs with highest probability.
    
    Args:
        eligible_legs: List of eligible leg dictionaries
        max_legs_per_match: Maximum number of legs to keep per match
        
    Returns:
        List of filtered eligible legs
    """
    if not eligible_legs or max_legs_per_match <= 0:
        return eligible_legs
        
    legs_by_match: Dict[str, List[Dict]] = {}
    for leg in eligible_legs:
        match_id = leg['match_id']
        if match_id not in legs_by_match:
            legs_by_match[match_id] = []
        legs_by_match[match_id].append(leg)
    
    # For each match, keep only top N legs with highest probability
    limited_legs = []
    for match_id, match_legs in legs_by_match.items():
        # Sort legs by probability (descending)
        sorted_legs = sorted(match_legs, key=lambda x: x.get('prob_at_bet', 0), reverse=True)
        # Keep only top N legs
        limited_legs.extend(sorted_legs[:max_legs_per_match])
    
    return limited_legs