#!/usr/bin/env python3
"""
Parlay Backtester with Enhanced Team Info Integration and Analysis.

This script backtests parlay betting strategies using OOF predictions,
a strategy guide, and consolidated team information. It focuses on
assertive coding, resource management, and detailed performance analysis
including country and league-based insights.
"""
import json
import re
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, Set
from itertools import combinations, product
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import warnings
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import pandas as pd
import numpy as np
import sys
import os
# Add the project root to the path to allow importing from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.utils.config import TeamNameStrict
from pydantic import BaseModel, Field, ValidationError, field_validator
from multiprocessing import Manager

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
# Attempt to find project root (AGENTICFC888)
if (SCRIPT_DIR.parent / "models").exists() and (SCRIPT_DIR.parent / "scripts").exists():
    BASE_DIR = SCRIPT_DIR.parent
elif (SCRIPT_DIR.parent.parent / "models").exists() and (SCRIPT_DIR.parent.parent / "scripts").exists():
    BASE_DIR = SCRIPT_DIR.parent.parent
else:
    BASE_DIR = Path("/Users/barroca888/Downloads/Agenticfc/AgenticFC888") # Fallback
    logger.warning(f"Could not auto-detect project root. Using hardcoded BASE_DIR: {BASE_DIR}")

# Define paths using BASE_DIR
DATA_OUTPUT_DIR = BASE_DIR / 'models' / 'data' / 'outputs' / 'predictions'
PARLAY_OUTPUT_DIR = DATA_OUTPUT_DIR / 'parlay_outputs_V2' # New output dir for this version

# Default Input paths
OOF_INPUT_PATH_DEFAULT = DATA_OUTPUT_DIR / 'combined_oof_ALL_pipelines.parquet'
STRATEGY_GUIDE_PATH_DEFAULT = BASE_DIR / 'models' / 'utils' / 'files' / 'best_strategy_per_market.csv'
MARKET_DEFINITIONS_PATH_DEFAULT = BASE_DIR / 'models' / 'utils' / 'files' / 'parlay_market_definitions.json'
CONSOLIDATED_TEAM_INFO_PATH = BASE_DIR / 'models' / 'utils' / 'files' / 'consolidated_team_info.json'

# Default Output paths
PARLAY_RESULTS_PATH_DEFAULT = PARLAY_OUTPUT_DIR / 'parlay_backtest_results_v2.csv'
VISUALIZATIONS_DIR_DEFAULT = PARLAY_OUTPUT_DIR / 'visualizations_v2'

DEFAULT_DATE_COL = 'Date'
DEFAULT_MATCH_ID_COL = 'MatchID'

# Resource Management
DEFAULT_CPU_WORKERS = max(1, os.cpu_count() // 2) # Target ~50% CPU usage
MAX_MEMORY_PERCENT_THRESHOLD = 75.0  # More realistic threshold
MIN_FREE_MEMORY_GB_THRESHOLD = 2.0  # Ensure at least 2GB free
CHECKPOINT_INTERVAL = 1000  # Save partial results every N parlays (parlays, not days)
CHUNK_SIZE_DAYS = 1  # Process fewer days per chunk
SAMPLE_RATE = 0.25  # Use 25% of dates for testing

# --- Pydantic Models for Team Info Validation ---

class LeagueInfo(BaseModel):
    name: str
    id: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("League name must be a string")
        return v

class TeamConsolidatedDetails(BaseModel):
    canonical_name: str
    country: Optional[str] = None
    statarea_id: Optional[str] = None
    mongodb_id: Optional[str] = None
    leagues: List[LeagueInfo] = Field(default_factory=list)
    alt_names: List[str] = Field(default_factory=list)

    @field_validator("alt_names")
    @classmethod
    def validate_alt_names(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("alt_names must be a list")
        return [TeamNameStrict(name) if isinstance(name, str) else name for name in v]

    @field_validator("canonical_name", "country")
    @classmethod
    def validate_team_name_fields(cls, v: Any) -> Any:
        if v is not None:  # Allow None for country
            return TeamNameStrict(v)
        return v

    class Config:
        validate_assignment = True

ConsolidatedTeamInfoSchema = Dict[str, TeamConsolidatedDetails]

# --- Team Information Resolver ---
class TeamInfoResolver:
    """
    Manages loading and resolving team information from consolidated_team_info.json.
    """
    def __init__(self, consolidated_info_path: Path):
        assert consolidated_info_path.exists(), f"Consolidated team info file not found: {consolidated_info_path}"
        self.consolidated_info_path = str(consolidated_info_path)  # Use string path for better pickling
        self.team_data = self._load_team_data()
        
        self._normalized_lookup = {}  # normalized_variant -> canonical_name
        self._sorted_raw_forms_for_splitting = []
        self._build_lookups()
        logger.info(f"TeamInfoResolver initialized with {len(self.team_data)} canonical teams")

    def _load_team_data(self) -> ConsolidatedTeamInfoSchema:
        try:
            with open(self.consolidated_info_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Validate with Pydantic
            validated_data: ConsolidatedTeamInfoSchema = {}
            for canonical, details_dict in raw_data.items():
                try:
                    # Pydantic expects the canonical name to be part of the model if it's not the key
                    # Or ensure the dict structure matches TeamConsolidatedDetails directly
                    # Here, details_dict should directly match TeamConsolidatedDetails fields
                    if 'canonical_name' not in details_dict: # Ensure canonical_name is in details
                        details_dict['canonical_name'] = canonical
                    
                    team_details_model = TeamConsolidatedDetails(**details_dict)
                    validated_data[team_details_model.canonical_name] = team_details_model # Use model's canonical name as key
                except ValidationError as e:
                    logger.error(f"Validation error for team '{canonical}' data {details_dict}: {e}")
                    # Optionally, skip this entry or handle error
            return validated_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.consolidated_info_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading team data from {self.consolidated_info_path}: {e}")
            raise
            
    @staticmethod
    def _normalize_name(name: str) -> str:
        if not name: return ""
        return "".join(filter(str.isalnum, name)).lower()

    def _build_lookups(self) -> None:
        raw_forms_for_splitting_set: Set[str] = set()
        for canonical_name, details in self.team_data.items():
            # Primary canonical name
            self._normalized_lookup[self._normalize_name(canonical_name)] = canonical_name
            raw_forms_for_splitting_set.add(canonical_name)

            # Alternative names
            for alt_name in details.alt_names:
                self._normalized_lookup[self._normalize_name(alt_name)] = canonical_name
                raw_forms_for_splitting_set.add(alt_name)
        
        self._sorted_raw_forms_for_splitting = sorted(list(raw_forms_for_splitting_set), key=len, reverse=True)

    def get_canonical_name(self, raw_team_name: str) -> Optional[str]:
        if not raw_team_name or (isinstance(raw_team_name, float) and pd.isna(raw_team_name)):
            return None
        assert isinstance(raw_team_name, str), f"Input must be a string, got {type(raw_team_name)}"

        # Try direct match (case-sensitive) on canonical names first
        if raw_team_name in self.team_data:
            return raw_team_name

        # Try normalized lookup
        norm_name = self._normalize_name(raw_team_name)
        if norm_name in self._normalized_lookup:
            return self._normalized_lookup[norm_name]
        
        # Try spaced version (e.g., CamelCase to Spaced Name)
        spaced_version = generate_camel_case_spaced_version(raw_team_name)
        if spaced_version != raw_team_name: # Only if different
            norm_spaced_version = self._normalize_name(spaced_version)
            if norm_spaced_version in self._normalized_lookup:
                return self._normalized_lookup[norm_spaced_version]
            # Also check if the spaced version itself is a canonical key
            if spaced_version in self.team_data:
                 return spaced_version
        
        logger.debug(f"Canonical name not found for raw: '{raw_team_name}'")
        return None

    def get_team_details(self, team_name_variant: str) -> Optional[TeamConsolidatedDetails]:
        canonical_name = self.get_canonical_name(team_name_variant)
        return self.team_data.get(canonical_name) if canonical_name else None

    def get_team_country(self, team_name_variant: str) -> Optional[str]:
        details = self.get_team_details(team_name_variant)
        return details.country if details and details.country else None
        
    def get_team_primary_league_name(self, team_name_variant: str) -> Optional[str]:
        details = self.get_team_details(team_name_variant)
        if details and details.leagues:
            # Heuristic: return the first league listed, or one marked as primary if such a field existed
            return details.leagues[0].name 
        return None

    def parse_match_id(self, match_id_str: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Attempts to parse a MatchID string into home and away canonical team names.
        Relies on the comprehensiveness of `alt_names` in consolidated_team_info.json
        and the `_sorted_raw_forms_for_splitting` list.
        """
        assert isinstance(match_id_str, str), "match_id_str must be a string."
        
        teams_part = match_id_str
        # Remove YYYYMMDD_ prefix if present
        if len(match_id_str) > 8 and match_id_str[:8].isdigit() and match_id_str[8] == '_':
            teams_part = match_id_str[9:]

        # Strategy 1: Try splitting using known team forms (longest first)
        for home_form_candidate in self._sorted_raw_forms_for_splitting:
            # Normalize both for robust prefix checking
            norm_home_form = self._normalize_name(home_form_candidate)
            norm_teams_part = self._normalize_name(teams_part)

            if norm_teams_part.startswith(norm_home_form):
                # Found a potential home team. Determine the actual split point in the original `teams_part`.
                # This is tricky because normalization removes spaces/cases.
                # We need to find the shortest prefix of `teams_part` that normalizes to `norm_home_form`.
                split_idx = -1
                for i in range(1, len(teams_part) + 1): # Iterate through possible split points
                    current_prefix_original = teams_part[:i]
                    if self._normalize_name(current_prefix_original) == norm_home_form:
                        split_idx = i
                        break # Found the shortest original prefix that matches normalized form
                
                if split_idx != -1 and split_idx < len(teams_part): # Ensure there's an away part
                    home_raw_extracted = teams_part[:split_idx]
                    away_raw_extracted = teams_part[split_idx:]

                    home_canonical = self.get_canonical_name(home_raw_extracted)
                    away_canonical = self.get_canonical_name(away_raw_extracted)

                    if home_canonical and away_canonical and home_canonical != away_canonical:
                        # Additional check: ensure the away part isn't just a suffix of a longer known name
                        # that could have been the away team if the home split was shorter.
                        # This is complex. For now, if both resolve, we accept.
                        return home_canonical, away_canonical
        
        # Strategy 2: Fallback to underscore splitting if present (common for non-concatenated IDs)
        if '_' in teams_part and teams_part.count('_') == 1:
            home_raw, away_raw = teams_part.split('_', 1)
            home_canonical = self.get_canonical_name(home_raw)
            away_canonical = self.get_canonical_name(away_raw)
            if home_canonical and away_canonical and home_canonical != away_canonical:
                return home_canonical, away_canonical

        logger.debug(f"Could not parse MatchID '{match_id_str}' into two known teams. Teams part: '{teams_part}'")
        return None, None


# --- Helper Functions ---
def generate_camel_case_spaced_version(name: str) -> str: # Keep this as it can help generate alts
    assert isinstance(name, str), "Input name must be a string."
    if not name: return ""
    s1 = re.sub(r"(\B[A-Z][a-z])", r" \1", name) 
    s2 = re.sub(r"([a-z])([A-Z])", r"\1 \2", s1) 
    s3 = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s2) 
    s4 = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s3) 
    return " ".join(s4.split()) 

def check_memory_usage(critical_threshold=MAX_MEMORY_PERCENT_THRESHOLD, 
                     critical_free_gb=MIN_FREE_MEMORY_GB_THRESHOLD) -> Tuple[bool, str]:
    memory_info = psutil.virtual_memory()
    memory_percent = memory_info.percent
    free_memory_gb = memory_info.available / (1024**3)
    
    status = "normal"
    should_pause = False

    if memory_percent > critical_threshold or free_memory_gb < critical_free_gb:
        # Force garbage collection
        gc.collect()
        
        # Wait for memory to be reclaimed
        time.sleep(1.0)
        
        # Recheck
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        free_memory_gb = memory_info.available / (1024**3)
        
        if memory_percent > critical_threshold or free_memory_gb < critical_free_gb:
            logger.warning(f"Critical memory: {memory_percent:.1f}% used, {free_memory_gb:.2f}GB free")
            status = "critical"
            should_pause = True
        else:
            status = "recovered"
    
    return should_pause, status

# --- Global TeamInfoResolver Instance ---
# To be initialized in main after paths are confirmed.
TEAM_INFO_RESOLVER: Optional[TeamInfoResolver] = None

# --- Core Parlay Logic (adapted) ---
def load_data(
    combined_oof_path: Path, 
    strategy_guide_path: Path, 
    market_definitions_path: Path,
    team_info_path: Path
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict, TeamInfoResolver]]:
    logger.info("--- Loading Data ---")
    s_time = time.time()
    try:
        assert combined_oof_path.exists(), f"OOF File not found: {combined_oof_path}"
        combined_oof_df = pd.read_parquet(combined_oof_path)
        logger.info(f"Loaded combined_oof_df: {combined_oof_df.shape} (in {time.time()-s_time:.2f}s)")

        assert DEFAULT_DATE_COL in combined_oof_df.columns, f"Date column '{DEFAULT_DATE_COL}' not found."
        assert DEFAULT_MATCH_ID_COL in combined_oof_df.columns, f"MatchID column '{DEFAULT_MATCH_ID_COL}' not found."
        
        combined_oof_df[DEFAULT_DATE_COL] = pd.to_datetime(combined_oof_df[DEFAULT_DATE_COL])

        assert strategy_guide_path.exists(), f"Strategy Guide File not found: {strategy_guide_path}"
        strategy_guide_df = pd.read_csv(strategy_guide_path)
        logger.info(f"Loaded strategy_guide_df: {strategy_guide_df.shape}")
        
        assert market_definitions_path.exists(), f"Market Definitions File not found: {market_definitions_path}"
        with open(market_definitions_path, 'r', encoding='utf-8') as f:
            parlay_market_definitions = json.load(f)
        logger.info(f"Loaded PARLAY_MARKET_DEFINITIONS: {len(parlay_market_definitions)} entries")

        global TEAM_INFO_RESOLVER # Initialize global resolver
        TEAM_INFO_RESOLVER = TeamInfoResolver(team_info_path)
        
        return combined_oof_df, strategy_guide_df, parlay_market_definitions, TEAM_INFO_RESOLVER
    except Exception as e:
        logger.critical(f"CRITICAL: Error loading data: {e}", exc_info=True)
        return None

def preprocess_strategy_guide(raw_guide_df: pd.DataFrame, 
                            market_definitions: dict, 
                            oof_df_columns_set: set) -> Optional[pd.DataFrame]:
    logger.info("--- Pre-processing Strategy Guide ---")
    s_time = time.time()
    
    # Filter out O05 market if desired (this specific filter can be made configurable)
    # raw_guide_df = raw_guide_df[raw_guide_df['market'] != 'O05'].copy() 
    # For now, let's assume all markets in the guide are intentional.

    market_groups = { # This can be loaded from config or enhanced
        'match_outcome': ['H', 'A', 'D'],
        'double_chance': ['HomeOrDraw', 'DrawOrAway', 'HomeOrAway'],
        'goals_over_under': [m for m in market_definitions if m.startswith('O') or m.startswith('U')],
        'btts': ['BTTSYes', 'BTTSNo'], # Assuming BTTSNo might exist
        'match_goals_combo': [m for m in market_definitions if m.startswith('HO') or m.startswith('AO')],
        'dc_goals_combo': [m for m in market_definitions if m.startswith('1XO') or m.startswith('1XU') or m.startswith('12O') or m.startswith('12U') or m.startswith('X2O') or m.startswith('X2U')]
    }
    
    processed_rules = []
    skipped_rules_count = 0

    threshold_col = next((col for col in ['tradeoff_threshold', 'efficient_entry_threshold'] 
                         if col in raw_guide_df.columns), None)
    assert threshold_col is not None, "No valid threshold column (tradeoff_threshold or efficient_entry_threshold) found in strategy guide."
    logger.info(f"Using threshold column from strategy guide: '{threshold_col}'")

    for _, rule in raw_guide_df.iterrows():
        market_name = str(rule['market']).strip() # Ensure market is string and stripped
        model_id = str(rule['model_identifier']).strip()
        entry_threshold = float(rule[threshold_col])
        
        market_info = market_definitions.get(market_name)
        if not market_info:
            logger.warning(f"Market '{market_name}' from strategy guide not found in market_definitions. Skipping rule.")
            skipped_rules_count += 1
            continue

        prob_col = f"{model_id}_{market_info['prob_suffix']}"
        target_col = market_info['target_col']

        assert prob_col in oof_df_columns_set, f"Probability column '{prob_col}' for market '{market_name}' (model '{model_id}') not found in OOF columns. Check prob_suffix in market definitions or model_identifier in strategy guide."
        assert target_col in oof_df_columns_set, f"Target column '{target_col}' for market '{market_name}' not found in OOF columns. Check target_col in market definitions."
            
        processed_rules.append({
            'market': market_name,
            'model_identifier': model_id,
            'entry_threshold': entry_threshold,
            'prob_col': prob_col,
            'target_col': target_col,
            'conflict_group': market_info.get('conflict_group', market_name), # Default to market name if no specific conflict group
            'market_group': next((group for group, markets in market_groups.items() if market_name in markets), 'other')
        })

    assert processed_rules, "No valid rules after pre-processing strategy guide. Check market names, model identifiers, and column existence."
    
    result_df = pd.DataFrame(processed_rules)
    logger.info(f"Pre-processing complete: {len(result_df)} valid rules (in {time.time()-s_time:.2f}s). Skipped: {skipped_rules_count}")
    return result_df

def get_match_details_from_id(match_id: str, resolver: TeamInfoResolver) -> Dict[str, Optional[str]]:
    """Extracts home team, away team, country, and primary league from MatchID."""
    home_team_canon, away_team_canon = resolver.parse_match_id(match_id)
    
    home_country, away_country = None, None
    home_league, away_league = None, None # Primary league name

    if home_team_canon:
        home_details = resolver.get_team_details(home_team_canon)
        if home_details:
            home_country = home_details.country
            if home_details.leagues:
                home_league = home_details.leagues[0].name # Assuming first league is primary

    if away_team_canon:
        away_details = resolver.get_team_details(away_team_canon)
        if away_details:
            away_country = away_details.country
            if away_details.leagues:
                away_league = away_details.leagues[0].name

    # Determine overall match country/league (e.g., if home_country is primary)
    match_country = home_country if home_country else away_country
    match_league = home_league if home_league else away_league
    
    return {
        "home_team_canonical": home_team_canon,
        "away_team_canonical": away_team_canon,
        "match_country": match_country,
        "match_league": match_league,
        "home_country": home_country,
        "away_country": away_country,
        "home_league": home_league,
        "away_league": away_league,
    }


def process_daily_parlays(args_tuple: tuple) -> List[Dict]:
    """Processes parlays for a single day's games with memory optimization."""
    daily_games_df, strategy_guide_df, game_date, min_legs, max_legs, \
    match_id_col, fallback_threshold, shared_dict = args_tuple
    
    team_resolver = shared_dict['team_resolver']
    
    # Memory check first
    should_pause, mem_status = check_memory_usage()
    if should_pause:
        logger.warning(f"High memory usage ({mem_status}) at start of process_daily_parlays for {game_date}. Skipping day.")
        return []
    
    # 1. Filter columns dramatically - only keep essential ones
    needed_cols = [match_id_col, DEFAULT_DATE_COL]
    
    # Get only required probability and target columns
    for _, rule in strategy_guide_df.iterrows():
        needed_cols.append(rule['prob_col'])
        needed_cols.append(rule['target_col'])
    
    # Remove duplicates and ensure all columns exist
    needed_cols = list(set(needed_cols))
    existing_cols = [col for col in needed_cols if col in daily_games_df.columns]
    
    # 2. Create minimal dataframe
    daily_df_minimal = daily_games_df[existing_cols].copy()
    
    # 3. Release original dataframe immediately
    del daily_games_df
    gc.collect()
    
    # Process with minimal dataframe
    match_ids = daily_df_minimal[match_id_col].unique()
    all_legs = []
    
    # 4. Process one match at a time to minimize memory
    for match_id in match_ids:
        match_row = daily_df_minimal[daily_df_minimal[match_id_col] == match_id].iloc[0]
        match_details = get_match_details_from_id(match_id, team_resolver)
        
        for _, rule in strategy_guide_df.iterrows():
            prob_col, target_col = rule['prob_col'], rule['target_col']
            
            # Skip if columns don't exist
            if prob_col not in match_row or target_col not in match_row:
                continue
                
            prob_val = match_row[prob_col]
            
            if pd.notna(prob_val) and prob_val >= rule['entry_threshold']:
                leg_info = {
                    'match_id': match_id,
                    'market': rule['market'],
                    'model_used': rule['model_identifier'],
                    'prob_at_bet': float(prob_val),
                    'threshold_used': float(rule['entry_threshold']),
                    'actual_outcome': int(match_row[target_col]),
                    'conflict_group': rule['conflict_group'],
                    'market_group': rule['market_group'],
                }
                leg_info.update(match_details)
                all_legs.append(leg_info)
                
        # 5. Check memory after each match
        if len(all_legs) % 100 == 0:
            should_pause, _ = check_memory_usage()
            if should_pause:
                break
    
    # If memory issues, return early with already collected legs
    should_pause, _ = check_memory_usage()
    if should_pause or not all_legs:
        return []
    
    # 6. Generate parlays with batched processing
    parlay_results = []
    unique_matches = set(leg['match_id'] for leg in all_legs)
    
    # Only process if we have enough matches
    if len(unique_matches) < min_legs:
        return []
        
    # 7. Batch process combinations to control memory
    BATCH_SIZE = 1000  # Process combinations in batches
    
    for num_legs in range(min_legs, min(max_legs + 1, len(unique_matches) + 1)):
        match_combos = list(combinations(unique_matches, num_legs))
        
        # Process in batches
        for i in range(0, len(match_combos), BATCH_SIZE):
            batch = match_combos[i:i+BATCH_SIZE]
            
            for match_combo in batch:
                # Generate parlays for this combo
                parlay_legs = generate_parlay_for_combo(match_combo, all_legs, game_date)
                if parlay_legs:
                    parlay_results.extend(parlay_legs)
            
            # Check memory after each batch
            should_pause, _ = check_memory_usage()
            if should_pause:
                return parlay_results  # Return what we have so far
    
    return parlay_results

# Helper function to generate parlays for a match combination
def generate_parlay_for_combo(match_combo, all_legs, game_date):
    results = []
    legs_by_match = {m_id: [leg for leg in all_legs if leg['match_id'] == m_id] for m_id in match_combo}
    
    # Skip if any match has no legs
    if any(not legs for legs in legs_by_match.values()):
        return []
    
    # Get all combinations of one leg from each match
    leg_options = [legs_by_match[m_id] for m_id in match_combo]
    
    # Limit number of combinations to avoid memory explosion
    MAX_COMBINATIONS = 5000
    total_combinations = np.prod([len(legs) for legs in leg_options])
    
    if total_combinations > MAX_COMBINATIONS:
        return []  # Skip if too many combinations
        
    for leg_combo in product(*leg_options):
        # Check for conflicting market groups
        market_groups = [leg['market_group'] for leg in leg_combo]
        if len(set(market_groups)) < len(market_groups):
            continue  # Skip if duplicate market groups
            
        # Create parlay record
        parlay_won = all(leg['actual_outcome'] == 1 for leg in leg_combo)
        avg_prob = np.mean([leg['prob_at_bet'] for leg in leg_combo])
        
        record = {
            'parlay_date': game_date.strftime('%Y-%m-%d'),
            'num_legs': len(leg_combo),
            'parlay_won': int(parlay_won),
            'avg_prob': float(avg_prob),
            'parlay_country': leg_combo[0].get('match_country', 'Unknown'),
            'parlay_league': leg_combo[0].get('match_league', 'Unknown'),
            'market_combination': '+'.join(sorted(leg['market'] for leg in leg_combo))
        }
        
        # Add leg details
        for i, leg in enumerate(leg_combo, 1):
            record.update({
                f'leg{i}_match_id': leg['match_id'],
                f'leg{i}_market': leg['market'],
                f'leg{i}_model': leg['model_used'],
                f'leg{i}_prob': float(leg['prob_at_bet']),
                f'leg{i}_won': int(leg['actual_outcome']),
                f'leg{i}_country': leg.get('match_country', 'Unknown'),
                f'leg{i}_league': leg.get('match_league', 'Unknown')
            })
            
        results.append(record)
        
    return results

def run_parlay_backtest_parallel(
    combined_oof_df: pd.DataFrame,
    strategy_guide_df: pd.DataFrame,
    parlay_market_definitions: dict,
    team_info_resolver: TeamInfoResolver,
    date_col: str,
    match_id_col: str,
    max_legs: int,
    min_legs: int,
    sample_percentage: float,
    fallback_threshold: float,
    max_workers: int = DEFAULT_CPU_WORKERS,
    output_path_str: Optional[str] = None,
    reuse_checkpoints: bool = True
) -> Optional[pd.DataFrame]:
    
    # Create output directory early
    if output_path_str:
        output_dir = Path(output_path_str).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load existing checkpoints if requested
    existing_df = None
    processed_match_dates = set()
    
    if reuse_checkpoints and output_path_str:
        existing_df = load_existing_checkpoints(Path(output_path_str).parent)
        
        if existing_df is not None and not existing_df.empty:
            # Extract already processed dates
            if 'parlay_date' in existing_df.columns:
                processed_match_dates = set(existing_df['parlay_date'].unique())
                logger.info(f"Found {len(processed_match_dates)} already processed dates in checkpoints.")
    
    # Enhanced memory management
    gc.collect()
    logger.info(f"\nStarting parallel backtest with {max_workers} workers.")
    
    # When filtering dates to process, skip already processed dates
    unique_dates = sorted(combined_oof_df[date_col].unique())
    
    if processed_match_dates:
        # Convert datetime objects to string format matching processed_match_dates
        date_strings = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in unique_dates]
        unprocessed_indices = [i for i, d in enumerate(date_strings) if d not in processed_match_dates]
        unique_dates = [unique_dates[i] for i in unprocessed_indices]
        logger.info(f"After filtering already processed dates, {len(unique_dates)} dates remain to be processed.")
    
    # Rest of the function continues as before...
    # Enhanced memory management
    gc.collect()
    logger.info(f"\nStarting parallel backtest with {max_workers} workers.")
    
    # Create a Manager to share objects between processes
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['team_resolver'] = team_info_resolver
    
    # More aggressive memory parameters for stability
    MAX_DATES = 100 if sample_percentage < 0.2 else 500  # Fewer dates for full run
    CHUNK_SIZE = 3  # Very small chunks to ensure processing completes
    
    # Filter only essential columns before date splitting to reduce memory footprint
    essential_cols = [date_col, match_id_col]
    
    # Get all probability and target columns from strategy guide
    for _, rule in strategy_guide_df.iterrows():
        if 'prob_col' in rule and rule['prob_col']:
            essential_cols.append(rule['prob_col'])
        if 'target_col' in rule and rule['target_col']:
            essential_cols.append(rule['target_col'])
    
    essential_cols = list(set(essential_cols))
    existing_cols = [col for col in essential_cols if col in combined_oof_df.columns]
    
    # Pre-filter the dataframe to dramatically reduce memory usage
    logger.info(f"Filtering combined OOF dataframe to {len(existing_cols)} essential columns")
    combined_oof_df = combined_oof_df[existing_cols].copy()
    
    # Force garbage collection to free memory
    gc.collect()
    
    # Sample dates more aggressively
    unique_dates = sorted(combined_oof_df[date_col].unique())
    
    if sample_percentage < 1:
        num_dates = min(MAX_DATES, int(len(unique_dates) * sample_percentage))
        np.random.seed(42)
        sampled_dates = np.random.choice(unique_dates, size=num_dates, replace=False)
        logger.info(f"Sampled {len(sampled_dates)} dates from {len(unique_dates)} total for processing.")
    else:
        # Hard limit on dates for production runs
        if len(unique_dates) > MAX_DATES:
            np.random.seed(42)
            sampled_dates = np.random.choice(unique_dates, size=MAX_DATES, replace=False)
            logger.info(f"Limited to {MAX_DATES} random dates from {len(unique_dates)} total for memory efficiency.")
        else:
            sampled_dates = unique_dates
            logger.info(f"Using all {len(sampled_dates)} dates for processing.")
    
    # Process in even smaller chunks
    daily_tasks_args_list = []
    for date_val in sampled_dates:
        daily_df_subset = combined_oof_df[combined_oof_df[date_col] == date_val].copy()
        if not daily_df_subset.empty:
            task_args = (
                daily_df_subset, strategy_guide_df.copy(), date_val, 
                min_legs, max_legs, match_id_col, fallback_threshold,
                shared_dict
            )
            daily_tasks_args_list.append(task_args)
    
    # Save incremental results more frequently
    global CHECKPOINT_INTERVAL
    CHECKPOINT_INTERVAL = 100000  # Save after every ~100k parlays
    
    # Process in very small chunks for stability
    all_parlay_results = []
    num_day_chunks = (len(daily_tasks_args_list) + CHUNK_SIZE -1) // CHUNK_SIZE
    
    results_saved = False
    
    try:
        # Process only a manageable number of chunks
        max_chunks_to_process = min(200, num_day_chunks)
        logger.info(f"Will process at most {max_chunks_to_process} chunks out of {num_day_chunks} total")
        
        for i in range(max_chunks_to_process):
            chunk_start_idx = i * CHUNK_SIZE
            chunk_end_idx = min((i + 1) * CHUNK_SIZE, len(daily_tasks_args_list))
            current_chunk_tasks = daily_tasks_args_list[chunk_start_idx:chunk_end_idx]
            
            if not current_chunk_tasks:
                continue
                
            logger.info(f"Processing day chunk {i+1}/{max_chunks_to_process} ({len(current_chunk_tasks)} days)...")
            
            # Use fewer workers for better stability
            actual_workers = min(max_workers, len(current_chunk_tasks))
            
            chunk_results = []
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                futures = [executor.submit(process_daily_parlays, task_args) for task_args in current_chunk_tasks]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Chunk {i+1} Progress"):
                    try:
                        day_results = future.result()
                        chunk_results.extend(day_results)
                    except Exception as e:
                        logger.error(f"Error processing a day in chunk {i+1}: {e}", exc_info=True)
                    
                    # Check memory inside the loop
                    should_pause, _ = check_memory_usage()
                    if should_pause:
                        logger.critical(f"Critical memory during chunk {i+1}. Stopping further processing.")
                        break
            
            # Add to all results
            all_parlay_results.extend(chunk_results)
            
            # Save incremental results
            if output_path_str and chunk_results:
                # Frequent checkpoints with separate files to avoid corruption
                checkpoint_path = Path(f"{os.path.splitext(output_path_str)[0]}_checkpoint_{i+1}.parquet")
                pd.DataFrame(chunk_results).to_parquet(checkpoint_path)
                logger.info(f"Checkpoint saved: {len(chunk_results)} parlays to {checkpoint_path}")
                
                # Also save complete results so far
                if len(all_parlay_results) > 0:
                    complete_df = pd.DataFrame(all_parlay_results)
                    complete_path = Path(output_path_str)
                    complete_df.to_parquet(complete_path.with_suffix('.parquet'))
                    results_saved = True
                    logger.info(f"Saved {len(all_parlay_results)} total parlays to {complete_path.with_suffix('.parquet')}")
            
            # Force garbage collection
            gc.collect()
            
            # Check memory after each chunk
            should_pause, _ = check_memory_usage()
            if should_pause:
                logger.critical("Critical memory after chunk processing. Stopping further chunks.")
                break
    
    except Exception as e:
        logger.critical(f"Error during parallel processing: {e}", exc_info=True)
        
    finally:
        # Always try to save what we have, even if there was an error
        if output_path_str and all_parlay_results and not results_saved:
            try:
                results_df = pd.DataFrame(all_parlay_results)
                output_path = Path(output_path_str)
                results_df.to_parquet(output_path.with_suffix('.parquet'))
                logger.info(f"Saved {len(results_df)} parlays to {output_path.with_suffix('.parquet')}")
            except Exception as save_error:
                logger.error(f"Failed to save final results: {save_error}")
    
    if not all_parlay_results:
        logger.warning("No valid parlay results were generated.")
        return None
    
    logger.info(f"\nGenerated {len(all_parlay_results):,} total parlays from all processed days.")
    
    # Merge new results with existing results if available
    if existing_df is not None and not existing_df.empty and all_parlay_results:
        all_parlay_results_df = pd.DataFrame(all_parlay_results)
        combined_results_df = pd.concat([existing_df, all_parlay_results_df], ignore_index=True)
        logger.info(f"Merged {len(all_parlay_results_df)} new parlays with {len(existing_df)} existing parlays.")
        return combined_results_df
    elif existing_df is not None and not existing_df.empty and not all_parlay_results:
        logger.info("No new parlays generated, returning existing results only.")
        return existing_df
    elif all_parlay_results:
        return pd.DataFrame(all_parlay_results)
    else:
        return None

# --- Analysis and Plotting (Updated for Country/League) ---
def create_parlay_visualizations(results_df: pd.DataFrame, output_viz_dir: Path):
    if results_df is None or results_df.empty:
        logger.warning("No data available for visualization.")
        return
    
    output_viz_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")

    # 1. Win Rate by Number of Legs - Fix the max() error
    plt.figure(figsize=(10, 6))
    summary = results_df.groupby('num_legs')['parlay_won'].agg(['count', 'mean']).reset_index()
    summary['mean'] *= 100 # Convert to percentage
    
    # Fix: Use proper scalar max calculation
    max_mean_value = summary['mean'].max() if not summary.empty else 10
    y_limit = max(max_mean_value * 1.15, 10)
    
    sns.barplot(x='num_legs', y='mean', data=summary, hue='num_legs', palette="viridis", dodge=False, legend=False)
    for index, row in summary.iterrows():
        plt.text(index, row['mean'] + 1, f"{row['mean']:.1f}% (n={row['count']})", 
                 color='black', ha="center", va="bottom")
    plt.title('Parlay Win Rate by Number of Legs')
    plt.xlabel('Number of Legs')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, y_limit)
    plt.tight_layout()
    plt.savefig(output_viz_dir / 'win_rate_by_legs.png')
    plt.close()

    # 2. Win Rate by Predicted Probability (remains similar)
    if 'avg_prob' in results_df.columns and not results_df['avg_prob'].isnull().all():
        plt.figure(figsize=(10, 6))
        results_df['prob_bin'] = pd.cut(results_df['avg_prob'], bins=np.arange(0.4, 1.01, 0.1), right=False) # Ensure bins cover range
        prob_summary = results_df.groupby('prob_bin', observed=False)['parlay_won'].agg(['count', 'mean']).reset_index()
        prob_summary['mean'] *= 100
        sns.barplot(x='prob_bin', y='mean', data=prob_summary, hue='prob_bin', palette="viridis", dodge=False, legend=False)
        plt.title('Parlay Win Rate by Avg. Predicted Probability')
        plt.xlabel('Avg. Predicted Probability Bin')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_viz_dir / 'win_rate_by_avg_prob.png')
        plt.close()

    # 3. Win Rate by Country (New)
    if 'parlay_country' in results_df.columns and not results_df['parlay_country'].isnull().all():
        plt.figure(figsize=(12, 7))
        country_summary = results_df.groupby('parlay_country')['parlay_won'].agg(['count', 'mean']).reset_index()
        country_summary['mean'] *= 100
        country_summary = country_summary[country_summary['count'] >= 50].sort_values('mean', ascending=False).head(15) # Top 15 with min 50 parlays
        
        if not country_summary.empty:
            sns.barplot(x='mean', y='parlay_country', data=country_summary, hue='parlay_country', palette="viridis", dodge=False, legend=False, orient='h')
            plt.title('Top Parlay Win Rates by Match Country (Min 50 Parlays)')
            plt.xlabel('Win Rate (%)')
            plt.ylabel('Country')
            plt.tight_layout()
            plt.savefig(output_viz_dir / 'win_rate_by_country.png')
        else:
            logger.info("Not enough data per country for 'Win Rate by Country' plot.")
        plt.close()

    # 4. Win Rate by League (New)
    if 'parlay_league' in results_df.columns and not results_df['parlay_league'].isnull().all():
        plt.figure(figsize=(14, 8)) # Wider for league names
        league_summary = results_df.groupby('parlay_league')['parlay_won'].agg(['count', 'mean']).reset_index()
        league_summary['mean'] *= 100
        league_summary = league_summary[league_summary['count'] >= 30].sort_values('mean', ascending=False).head(20) # Top 20 with min 30 parlays
        
        if not league_summary.empty:
            sns.barplot(x='mean', y='parlay_league', data=league_summary, hue='parlay_league', palette="magma", dodge=False, legend=False, orient='h')
            plt.title('Top Parlay Win Rates by Match League (Min 30 Parlays)')
            plt.xlabel('Win Rate (%)')
            plt.ylabel('League')
            plt.tight_layout()
            plt.savefig(output_viz_dir / 'win_rate_by_league.png')
        else:
            logger.info("Not enough data per league for 'Win Rate by League' plot.")
        plt.close()

    # 5. Model Usage in Winning vs All Parlays (remains similar)
    all_models, winning_models = analyze_model_usage_in_parlays(results_df, results_df['num_legs'].max() if 'num_legs' in results_df else 4)
    if all_models is not None:
        plt.figure(figsize=(12,7))
        all_models.sort_values().plot(kind='barh', title='Model Usage in All Parlay Legs', color='skyblue')
        plt.xlabel('Percentage of Legs (%)')
        plt.tight_layout()
        plt.savefig(output_viz_dir / 'model_usage_all_parlays.png')
        plt.close()
    if winning_models is not None:
        plt.figure(figsize=(12,7))
        winning_models.sort_values().plot(kind='barh', title='Model Usage in Winning Parlay Legs', color='lightgreen')
        plt.xlabel('Percentage of Legs (%)')
        plt.tight_layout()
        plt.savefig(output_viz_dir / 'model_usage_winning_parlays.png')
        plt.close()

    # Add call to the new selection visualization function
    try:
        create_selection_visualizations(results_df, output_viz_dir)
    except Exception as e:
        logger.error(f"Error creating selection visualizations: {e}", exc_info=True)
    
    logger.info(f"All visualizations saved to {output_viz_dir}")

def analyze_model_usage_in_parlays(parlay_results_df: pd.DataFrame, max_legs_in_data: int) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Analyzes model usage in all and winning parlays."""
    if parlay_results_df is None or parlay_results_df.empty:
        logger.warning("No parlay results to analyze model usage.")
        return None, None

    all_leg_models: List[str] = []
    winning_leg_models: List[str] = []

    for i in range(1, max_legs_in_data + 1):
        model_col = f'leg{i}_model'
        won_col = f'leg{i}_won' # Assuming individual leg win status is stored, or use parlay_won
        
        if model_col in parlay_results_df.columns:
            all_leg_models.extend(parlay_results_df[model_col].dropna().tolist())
            
            # If using parlay_won for leg contribution:
            # This attributes all models in a winning parlay as "winning models"
            # A more granular `leg{i}_won` column would be better for leg-specific model performance
            if 'parlay_won' in parlay_results_df.columns:
                 winning_legs_from_parlay = parlay_results_df[parlay_results_df['parlay_won'] == 1][model_col].dropna().tolist()
                 winning_leg_models.extend(winning_legs_from_parlay)

    all_usage_series = pd.Series(all_leg_models).value_counts(normalize=True) * 100 if all_leg_models else None
    winning_usage_series = pd.Series(winning_leg_models).value_counts(normalize=True) * 100 if winning_leg_models else None
    
    return all_usage_series, winning_usage_series

# Updated helper function for safe visualization
def safe_max(series, default=10):
    """Safely get max value from a series, handling empty series."""
    if series.empty:
        return default
    return series.max()

# --- Main Orchestrator ---
def run_parlay_backtester_orchestrator(
    oof_path_str: str, strategy_path_str: str, markets_def_path_str: str, 
    team_info_path_str: str, output_path_str: str,
    max_legs: int, min_legs: int, sample_perc: float,
    date_col: str, match_id_col: str,
    fallback_threshold: float,
    max_cpu_workers: int = DEFAULT_CPU_WORKERS,
    create_plots: bool = True
):
    logger.info("\n" + "="*30 + " Starting Parlay Backtester V2 " + "="*30)
    overall_start_time = time.time()

    # Create output directory if it doesn't exist
    PARLAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if create_plots:
        VISUALIZATIONS_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)

    # Load data (this will initialize TEAM_INFO_RESOLVER globally)
    data_load_res = load_data(
        Path(oof_path_str), Path(strategy_path_str), 
        Path(markets_def_path_str), Path(team_info_path_str)
    )
    if data_load_res is None:
        logger.critical("Data loading failed. Exiting.")
        return
    combined_oof_df, raw_strat_guide_df, mkt_defs_loaded, _ = data_load_res # team_resolver is now global

    # Preprocess strategy guide
    assert TEAM_INFO_RESOLVER is not None, "TeamInfoResolver was not initialized by load_data."
    strat_guide_processed_df = preprocess_strategy_guide(
        raw_strat_guide_df, mkt_defs_loaded, set(combined_oof_df.columns)
    )
    if strat_guide_processed_df is None:
        logger.critical("Strategy guide processing failed. Exiting.")
        return

    logger.info(f"Configuration: Max Legs={max_legs}, Min Legs={min_legs}, Sample={sample_perc*100}%, CPU Workers={max_cpu_workers}")

    results_df = run_parlay_backtest_parallel(
        combined_oof_df=combined_oof_df,
        strategy_guide_df=strat_guide_processed_df,
        parlay_market_definitions=mkt_defs_loaded,
        team_info_resolver=TEAM_INFO_RESOLVER, # Pass it explicitly
        date_col=date_col,
        match_id_col=match_id_col,
        max_legs=max_legs,
        min_legs=min_legs,
        sample_percentage=sample_perc,
        fallback_threshold=fallback_threshold,
        max_workers=max_cpu_workers,
        output_path_str=output_path_str,
        reuse_checkpoints=True
    )

    # Save to parquet first, then CSV (parquet is faster and safer)
    if results_df is not None and not results_df.empty:
        output_p = Path(output_path_str)
        parquet_path = output_p.with_suffix('.parquet')
        
        # First save as parquet
        results_df.to_parquet(parquet_path)
        logger.info(f"Saved parlay results to: {parquet_path}")
        
        try:
            # Then save as CSV with reduced columns if needed
            if results_df.shape[1] > 100:  # If too many columns for CSV
                essential_cols = ['parlay_date', 'num_legs', 'parlay_won', 'avg_prob', 
                                 'parlay_country', 'parlay_league', 'market_combination']
                results_df[essential_cols].to_csv(output_p, index=False)
            else:
                results_df.to_csv(output_p, index=False)
            logger.info(f"Saved CSV results to: {output_p}")
        except Exception as e:
            logger.warning(f"Could not save CSV version: {e}")
        
        if create_plots:
            try:
                create_parlay_visualizations(results_df, VISUALIZATIONS_DIR_DEFAULT)
            except Exception as e:
                logger.error(f"Error creating visualizations: {e}")
    else:
        logger.error("No valid parlay results generated. Cannot save or visualize.")

    logger.info(f"Parlay Backtester V2 finished in {time.time() - overall_start_time:.2f} seconds.")
    logger.info("="*30 + " Parlay Backtester V2 Complete " + "="*30 + "\n")

def load_existing_checkpoints(output_dir: Path) -> Optional[pd.DataFrame]:
    """Load and combine all existing checkpoint files into a single dataframe."""
    checkpoint_files = list(output_dir.glob("*_checkpoint_*.parquet"))
    main_file = output_dir / "parlay_backtest_results_v2.parquet"
    
    all_files = checkpoint_files + ([main_file] if main_file.exists() else [])
    
    if not all_files:
        logger.info("No existing checkpoint files found.")
        return None
        
    logger.info(f"Found {len(all_files)} existing checkpoint files to load.")
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
            logger.info(f"Loaded {len(df)} parlays from {file}")
        except Exception as e:
            logger.error(f"Error loading checkpoint {file}: {e}")
    
    if not dfs:
        return None
        
    # Combine all dataframes and remove duplicates
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create a unique identifier for each parlay to detect duplicates
    # Use leg details to create a unique signature
    leg_cols = [col for col in combined_df.columns if col.startswith('leg') and '_match_id' in col]
    market_cols = [col for col in combined_df.columns if col.startswith('leg') and '_market' in col]
    
    if leg_cols and market_cols:
        combined_df['parlay_signature'] = combined_df.apply(
            lambda row: '_'.join([str(row[col]) for col in (leg_cols + market_cols) if col in row]), 
            axis=1
        )
        
        # Remove duplicates
        before_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['parlay_signature'])
        after_count = len(combined_df)
        
        logger.info(f"Removed {before_count - after_count} duplicate parlays. Kept {after_count} unique parlays.")
        
        # Drop the signature column as it's no longer needed
        combined_df = combined_df.drop(columns=['parlay_signature'])
    
    return combined_df

def create_selection_visualizations(results_df: pd.DataFrame, output_viz_dir: Path):
    """Create visualizations focused on market selections."""
    if results_df is None or results_df.empty:
        logger.warning("No data available for selection visualizations.")
        return
    
    # Create selection directory
    selection_viz_dir = output_viz_dir / 'selections'
    selection_viz_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Extract all market selections
    all_market_cols = [col for col in results_df.columns if col.startswith('leg') and '_market' in col]
    all_markets = []
    
    for col in all_market_cols:
        all_markets.extend(results_df[col].dropna().unique())
    
    unique_markets = sorted(set(all_markets))
    logger.info(f"Analyzing {len(unique_markets)} unique market selections.")
    
    # 2. Selections usage count and win rate
    market_usage = {}
    market_wins = {}
    
    for market in unique_markets:
        # Count usage across all leg positions
        usage_count = 0
        win_count = 0
        
        for i in range(1, results_df['num_legs'].max() + 1):
            market_col = f'leg{i}_market'
            won_col = f'leg{i}_won'
            
            if market_col in results_df.columns and won_col in results_df.columns:
                market_matches = results_df[results_df[market_col] == market]
                usage_count += len(market_matches)
                win_count += market_matches[won_col].sum()
        
        if usage_count > 0:
            market_usage[market] = usage_count
            market_wins[market] = (win_count / usage_count) * 100
    
    # Sort by usage
    sorted_markets = sorted(market_usage.keys(), key=lambda x: market_usage[x], reverse=True)
    
    # 3. Plot selection usage
    plt.figure(figsize=(14, 10))
    usage_values = [market_usage[m] for m in sorted_markets[:20]]  # Top 20
    usage_labels = sorted_markets[:20]
    
    sns.barplot(x=usage_values, y=usage_labels, hue=usage_labels, palette="viridis", legend=False)
    plt.title('Top 20 Market Selections by Usage')
    plt.xlabel('Number of Times Used in Parlays')
    plt.ylabel('Market Selection')
    plt.tight_layout()
    plt.savefig(selection_viz_dir / 'market_selection_usage.png')
    plt.close()
    
    # 4. Plot selection win rates
    plt.figure(figsize=(14, 10))
    # Filter for selections with sufficient sample size
    min_usage = 50
    filtered_markets = [m for m in sorted_markets if market_usage[m] >= min_usage]
    win_values = [market_wins[m] for m in filtered_markets[:20]]  # Top 20
    win_labels = filtered_markets[:20]
    
    # Create a paired bar chart
    y_pos = np.arange(len(win_labels))
    plt.figure(figsize=(14, 10))
    
    # Sort by win rate
    sorted_indices = np.argsort(win_values)[::-1]
    win_values = [win_values[i] for i in sorted_indices]
    win_labels = [win_labels[i] for i in sorted_indices]
    
    sns.barplot(x=win_values, y=win_labels, hue=win_labels, palette="magma", legend=False)
    plt.title(f'Top 20 Market Selections by Win Rate (Min {min_usage} Occurrences)')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Market Selection')
    plt.tight_layout()
    plt.savefig(selection_viz_dir / 'market_selection_win_rates.png')
    plt.close()
    
    # 5. Selection performance by country
    # Take top 5 most common selections
    top_markets = sorted_markets[:5]
    
    for market in top_markets:
        country_performance = {}
        country_usage = {}
        
        # Gather data
        for i in range(1, results_df['num_legs'].max() + 1):
            market_col = f'leg{i}_market'
            won_col = f'leg{i}_won'
            country_col = f'leg{i}_country'
            
            if all(col in results_df.columns for col in [market_col, won_col, country_col]):
                # Filter rows where this market was used in this leg position
                market_data = results_df[results_df[market_col] == market]
                
                for country in market_data[country_col].unique():
                    if pd.isna(country) or country == 'Unknown':
                        continue
                        
                    country_data = market_data[market_data[country_col] == country]
                    
                    # Update counts
                    if country not in country_usage:
                        country_usage[country] = 0
                        country_performance[country] = 0
                    
                    country_usage[country] += len(country_data)
                    country_performance[country] += country_data[won_col].sum()
        
        # Calculate win rates
        for country in country_performance:
            if country_usage[country] >= 20:  # Minimum threshold
                country_performance[country] = (country_performance[country] / country_usage[country]) * 100
            else:
                country_performance.pop(country, None)
                
        # Plot if we have data
        if country_performance:
            # Sort countries by win rate
            sorted_countries = sorted(country_performance.keys(), 
                                      key=lambda x: country_performance[x], 
                                      reverse=True)[:15]  # Top 15
            
            win_rates = [country_performance[c] for c in sorted_countries]
            
            plt.figure(figsize=(14, 8))
            sns.barplot(x=win_rates, y=sorted_countries, hue=sorted_countries, palette="rocket", legend=False)
            plt.title(f'Win Rate by Country for Selection: {market}')
            plt.xlabel('Win Rate (%)')
            plt.ylabel('Country')
            plt.tight_layout()
            plt.savefig(selection_viz_dir / f'selection_{market}_by_country.png')
            plt.close()
    
    # 6. Selection win rate by number of legs
    for market in top_markets:
        leg_performance = {}
        
        # For each number of legs
        for num_legs in range(2, results_df['num_legs'].max() + 1):
            # Get parlays with exactly this many legs
            parlay_subset = results_df[results_df['num_legs'] == num_legs]
            
            # Check if this market appears in any leg
            market_used = False
            win_count = 0
            total_count = 0
            
            for i in range(1, num_legs + 1):
                market_col = f'leg{i}_market'
                
                if market_col in parlay_subset.columns:
                    market_parlays = parlay_subset[parlay_subset[market_col] == market]
                    
                    if not market_parlays.empty:
                        market_used = True
                        win_count += market_parlays['parlay_won'].sum()
                        total_count += len(market_parlays)
            
            if market_used and total_count >= 20:
                leg_performance[num_legs] = (win_count / total_count) * 100
        
        # Plot if we have data
        if leg_performance:
            leg_numbers = sorted(leg_performance.keys())
            win_rates = [leg_performance[n] for n in leg_numbers]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=leg_numbers, y=win_rates, hue=leg_numbers, palette="Blues_d", legend=False)
            plt.title(f'Win Rate by Number of Legs for Selection: {market}')
            plt.xlabel('Number of Legs')
            plt.ylabel('Win Rate (%)')
            plt.tight_layout()
            plt.savefig(selection_viz_dir / f'selection_{market}_by_legs.png')
            plt.close()
    
    # 7. Selection win rate vs. average probability
    for market in top_markets:
        # Collect probabilities and outcomes
        probs = []
        outcomes = []
        
        for i in range(1, results_df['num_legs'].max() + 1):
            market_col = f'leg{i}_market'
            prob_col = f'leg{i}_prob'
            won_col = f'leg{i}_won'
            
            if all(col in results_df.columns for col in [market_col, prob_col, won_col]):
                market_data = results_df[results_df[market_col] == market]
                
                probs.extend(market_data[prob_col].tolist())
                outcomes.extend(market_data[won_col].tolist())
        
        if probs and outcomes:
            # Create DataFrame for binning
            prob_df = pd.DataFrame({'probability': probs, 'outcome': outcomes})
            
            # Create bins
            bin_edges = np.arange(0.5, 1.01, 0.05)
            bin_labels = [f"{100*x:.0f}-{100*y:.0f}%" for x, y in zip(bin_edges[:-1], bin_edges[1:])]
            prob_df['prob_bin'] = pd.cut(prob_df['probability'], bins=bin_edges, labels=bin_labels, right=False)
            
            # Calculate win rate per bin
            bin_stats = prob_df.groupby('prob_bin', observed=False)['outcome'].agg(['mean', 'count']).reset_index()
            bin_stats['mean'] *= 100  # Convert to percentage
            
            # Filter bins with sufficient data
            bin_stats = bin_stats[bin_stats['count'] >= 20]
            
            if not bin_stats.empty:
                plt.figure(figsize=(12, 6))
                sns.barplot(x='prob_bin', y='mean', data=bin_stats, hue='prob_bin', palette="YlGnBu", legend=False)
                
                # Add count labels
                for i, row in bin_stats.iterrows():
                    plt.text(i, row['mean'] + 1, f"n={row['count']}", ha='center')
                
                plt.title(f'Win Rate by Predicted Probability for Selection: {market}')
                plt.xlabel('Predicted Probability Range')
                plt.ylabel('Actual Win Rate (%)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(selection_viz_dir / f'selection_{market}_by_probability.png')
                plt.close()
    
    logger.info(f"Selection-based visualizations saved to {selection_viz_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parlay Backtester V2 with Enhanced Team Info")
    parser.add_argument("--oof_path", type=str, default=str(OOF_INPUT_PATH_DEFAULT), help="Path to combined OOF predictions parquet file.")
    parser.add_argument("--strategy_path", type=str, default=str(STRATEGY_GUIDE_PATH_DEFAULT), help="Path to strategy guide CSV file.")
    parser.add_argument("--markets_def_path", type=str, default=str(MARKET_DEFINITIONS_PATH_DEFAULT), help="Path to parlay market definitions JSON file.")
    parser.add_argument("--team_info_path", type=str, default=str(CONSOLIDATED_TEAM_INFO_PATH), help="Path to consolidated team info JSON file.")
    parser.add_argument("--output_path", type=str, default=str(PARLAY_RESULTS_PATH_DEFAULT), help="Path to save parlay backtest results CSV.")
    parser.add_argument("--max_legs", type=int, default=6, help="Maximum number of legs in a parlay.")
    parser.add_argument("--min_legs", type=int, default=2, help="Minimum number of legs in a parlay.")
    parser.add_argument("--sample_perc", type=float, default=1, help="Percentage of unique dates to sample (0.0 to 1.0).")
    parser.add_argument("--date_col", type=str, default=DEFAULT_DATE_COL, help="Name of the date column in OOF data.")
    parser.add_argument("--match_id_col", type=str, default=DEFAULT_MATCH_ID_COL, help="Name of the match ID column in OOF data.")
    parser.add_argument("--fallback_thresh", type=float, default=0.0, help="Fallback probability threshold for legs not in strategy guide (0.0 to disable).")
    parser.add_argument("--max_cpu", type=int, default=DEFAULT_CPU_WORKERS, help=f"Maximum CPU workers (cores) to use. Defaults to ~25% of available cores.")
    parser.add_argument("--no_plots", action="store_true", help="Disable generation of plots.")
    parser.add_argument("--sample_rate", type=float, default=0.25,
                   help="Percentage of dates to sample for backtesting (0.0 to 1.0)")
    
    args = parser.parse_args()

    # Assertions for arguments
    assert Path(args.oof_path).exists(), f"OOF file not found: {args.oof_path}"
    assert Path(args.strategy_path).exists(), f"Strategy guide file not found: {args.strategy_path}"
    assert Path(args.markets_def_path).exists(), f"Market definitions file not found: {args.markets_def_path}"
    assert Path(args.team_info_path).exists(), f"Consolidated team info file not found: {args.team_info_path}"
    assert args.min_legs >= 1, "min_legs must be at least 1."
    assert args.max_legs >= args.min_legs, "max_legs must be greater than or equal to min_legs."
    assert 0.0 < args.sample_perc <= 1.0, "sample_perc must be between 0 (exclusive) and 1 (inclusive)."
    assert args.fallback_thresh >= 0.0 and args.fallback_thresh <= 1.0, "fallback_thresh must be between 0.0 and 1.0."
    assert args.max_cpu >= 1, "max_cpu must be at least 1."

    try:
        run_parlay_backtester_orchestrator(
            oof_path_str=args.oof_path,
            strategy_path_str=args.strategy_path,
            markets_def_path_str=args.markets_def_path,
            team_info_path_str=args.team_info_path,
            output_path_str=args.output_path,
            max_legs=args.max_legs,
            min_legs=args.min_legs,
            sample_perc=args.sample_perc,
            date_col=args.date_col,
            match_id_col=args.match_id_col,
            fallback_threshold=args.fallback_thresh,
            max_cpu_workers=args.max_cpu,
            create_plots=not args.no_plots
        )
    except Exception as e:
        logger.critical(f"Parlay backtester orchestrator failed with unhandled error: {e}", exc_info=True)
        # Potentially exit with an error code
        # sys.exit(1)