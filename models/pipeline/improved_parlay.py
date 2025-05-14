#!/usr/bin/env python3
"""
Parlay Backtester - ELITE STRATEGY VERSION.

This script backtests parlay betting strategies using OOF predictions
BUT applies a specific "Elite Leg Strategy" with refined market-model pairs
and higher, data-driven probability thresholds.
"""

import gc
import json
import logging
import os
import re
import time
import warnings
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import argparse
import sys

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

# --- Project Setup (Same as before) ---
try:
    _project_root = Path(__file__).resolve().parent.parent.parent
    if not (_project_root / "models").exists() or not (_project_root / "scripts").exists():
        _project_root = Path(os.getenv("PROJECT_ROOT", "/Users/barroca888/Downloads/Agenticfc/AgenticFC888"))
    sys.path.insert(0, str(_project_root))
    from models.utils.config import TeamNameStrict
except ImportError:
    class TeamNameStrict(str): pass

# --- Configuration (Mostly same, output path will be different) ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

BASE_DIR = _project_root
DATA_OUTPUT_DIR = BASE_DIR / 'models' / 'data' / 'outputs' / 'predictions'
UTILS_DIR = BASE_DIR / 'models' / 'utils' / 'files'
PARLAY_OUTPUT_DIR_ELITE = DATA_OUTPUT_DIR / 'parlay_outputs_V4_elite_strategy' # NEW OUTPUT DIR

OOF_INPUT_PATH_DEFAULT = DATA_OUTPUT_DIR / 'prefiltered_elite_oof.parquet' # Source of all predictions
MARKET_DEFINITIONS_PATH_DEFAULT = UTILS_DIR / 'parlay_market_definitions.json' # Still needed
CONSOLIDATED_TEAM_INFO_PATH_DEFAULT = UTILS_DIR / 'consolidated_team_info.json' # Still needed

PARLAY_RESULTS_PATH_ELITE_DEFAULT = PARLAY_OUTPUT_DIR_ELITE / 'parlay_backtest_results_elite.parquet'
PARLAY_RESULTS_CSV_ELITE_DEFAULT = PARLAY_OUTPUT_DIR_ELITE / 'parlay_backtest_results_elite.csv'
CHECKPOINT_DIR_ELITE_DEFAULT = PARLAY_OUTPUT_DIR_ELITE / 'checkpoints_elite'

DEFAULT_DATE_COL = 'Date'
DEFAULT_MATCH_ID_COL = 'MatchID'
DEFAULT_YEARLY_SAMPLE_RATE = 1.0 # For this elite run, you might want to use all data or a large sample
DAYS_PER_CHUNK = 30
MAX_COMBINATIONS_PER_MATCH_SET = 53_000

# --- ELITE LEG STRATEGY DEFINITION ---
ELITE_LEG_STRATEGY_CONFIG_PLACEHOLDER = [
    {'market': 'U45',        'model_identifier': 'gradient_boosting_V1_NonPCA_NoOdds_L0', 'min_elite_threshold': 0.75},
    {'market': 'O15',        'model_identifier': 'monte_carlo_V1_NonPCA_WithOdds_L0',     'min_elite_threshold': 0.70},
    {'market': 'HomeOrAway', 'model_identifier': 'monte_carlo_V1_NonPCA_WithOdds_L0', 'min_elite_threshold': 0.62},
    {'market': 'HomeOrDraw', 'model_identifier': 'gradient_boosting_V1_NonPCA_NoOdds_L0', 'min_elite_threshold': 0.68}, # Example threshold
    {'market': 'U35',        'model_identifier': 'gradient_boosting_V1_NonPCA_NoOdds_L0', 'min_elite_threshold': 0.62}, # Example threshold
     {'market': '1XU45',      'model_identifier': 'poisson_V2_PCA_NoOdds_L0', 'min_elite_threshold': 0.62}, # Example threshold
    # Let's include BTTSYes as an example, assuming you fix it
     {'market': 'X2O15',      'model_identifier': 'gradient_boosting_V2_PCA_WithOdds_L0', 'min_elite_threshold': 0.62},
     {'market': 'X2O25',      'model_identifier': 'gradient_boosting_V1_NonPCA_NoOdds_L0', 'min_elite_threshold': 0.62},
     {'market': 'X2U45',      'model_identifier': 'random_forest_V2_PCA_WithOdds_L0', 'min_elite_threshold': 0.62},
     {'market': '1XO15',      'model_identifier': 'poisson_V2_PCA_NoOdds_L0', 'min_elite_threshold': 0.53},
     {'market': '1XO25',      'model_identifier': 'random_forest_V2_PCA_WithOdds_L0', 'min_elite_threshold': 0.62},
     {'market': '1XU35',      'model_identifier': 'monte_carlo_V2_PCA_WithOdds_L0', 'min_elite_threshold': 0.62},
     {'market': 'BTTSYes',    'model_identifier': 'gradient_boosting_V1_NonPCA_WithOdds_L0', 'min_elite_threshold': 0.65},
     {'market': '1XU45',      'model_identifier': 'poisson_V2_PCA_NoOdds_L0', 'min_elite_threshold': 0.50},
]


# --- TeamInfoResolver, Memory Check, get_match_details (largely unchanged from previous script) ---
class TeamInfoResolver: # Simplified for brevity, use your full version
    def __init__(self, consolidated_info_path: Path): self._info_path_str = str(consolidated_info_path); #... (full implementation)
    def _load_and_validate_team_data(self): return {} # Placeholder
    def _normalize_name(self, name): return "" # Placeholder
    def _build_lookups(self): pass # Placeholder
    def get_canonical_name(self, name): return None # Placeholder
    def get_team_details(self, name): return None # Placeholder
    def parse_match_id(self, mid): return None, None # Placeholder

def check_memory_usage( # Simplified for brevity
    critical_threshold: float = 92.0, high_threshold: float = 85.0, min_free_gb: float = 1.0
) -> Tuple[bool, str]:
    memory_info = psutil.virtual_memory()
    if memory_info.percent > critical_threshold or (memory_info.available / (1024**3)) < min_free_gb:
        gc.collect()
        return True, "critical"
    return False, "normal"

def get_match_details(match_id: str, resolver: TeamInfoResolver) -> Dict[str, Optional[str]]: # Simplified
    return {"match_country": "Unknown", "match_league": "Unknown"}


# --- MODIFIED Data Loading and Preprocessing ---
def load_and_prepare_elite_strategy(
    elite_strategy_config: List[Dict[str, Any]],
    market_definitions: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Converts the ELITE_LEG_STRATEGY_CONFIG into a DataFrame similar to the
    original strategy_guide_df, adding prob_col and target_col.
    """
    processed_rules = []
    for rule_config in elite_strategy_config:
        market_name = rule_config['market']
        model_id = rule_config['model_identifier']
        entry_threshold = rule_config['min_elite_threshold']

        market_info = market_definitions.get(market_name)
        if not market_info:
            logger.warning(f"Market '{market_name}' from Elite Strategy not in market_definitions. Skipping.")
            continue
        
        # Assert expected keys exist in market_info
        expected_keys = {'prob_suffix', 'target_col', 'conflict_group'}
        missing_keys = expected_keys - market_info.keys()
        if missing_keys:
            logger.warning(f"Market definition for '{market_name}' missing keys: {missing_keys}. Using defaults.")
            # Provide defaults or skip rule if critical keys are missing
            if 'prob_suffix' not in market_info or 'target_col' not in market_info:
                 logger.error(f"CRITICAL: prob_suffix or target_col missing for market {market_name}. Skipping rule.")
                 continue

        prob_col = f"{model_id}_{market_info['prob_suffix']}"
        target_col = market_info['target_col'] # This is the actual outcome column name

        processed_rules.append({
            'market': market_name,
            'model_identifier': model_id,
            'entry_threshold': entry_threshold, # This is our new min_elite_threshold
            'prob_col': prob_col,
            'target_col': target_col,
            'conflict_group': market_info.get('conflict_group', market_name),
        })
    
    if not processed_rules:
        raise ValueError("No valid rules processed from ELITE_LEG_STRATEGY_CONFIG. Check market definitions and config.")
    
    return pd.DataFrame(processed_rules)


# --- Core Parlay Generation Logic (find_eligible_legs_for_day is implicitly using the new strategy_guide_df) ---
# find_eligible_legs_for_day, generate_parlays_for_day, sample_dates_yearly remain THE SAME as in your previous script.
# The change is that the `strategy_guide_df` fed into `run_parlay_backtest_chunked` will be different.

# (Copy paste find_eligible_legs_for_day, generate_parlays_for_day, sample_dates_yearly here from your previous complete script)
# ... these functions are assumed to be present and correct ...
def find_eligible_legs_for_day(
    daily_games_df: pd.DataFrame,
    strategy_guide: pd.DataFrame, # This will now be the elite strategy
    team_resolver: TeamInfoResolver,
    match_id_col: str
) -> List[Dict]:
    # This function remains IDENTICAL to the one in your `parlay_backtester_V3_sampled.py`
    # It will naturally use the new `strategy_guide` (our elite strategy) that's passed to it.
    eligible_legs = []
    required_cols = {match_id_col} | set(strategy_guide['prob_col']) | set(strategy_guide['target_col'])
    available_cols = [col for col in required_cols if col in daily_games_df.columns]
    if not available_cols: # Quick check if no relevant columns are available at all
        # logger.debug(f"No required columns for strategy guide present in daily_games_df. Columns: {daily_games_df.columns.tolist()}")
        return []
    daily_minimal_df = daily_games_df[available_cols].copy()

    match_details_cache = { mid: get_match_details(mid, team_resolver) for mid in daily_minimal_df[match_id_col].unique()}

    for _, rule in strategy_guide.iterrows():
        prob_col, target_col = rule['prob_col'], rule['target_col']
        threshold, market, model_id, conflict_group = rule['entry_threshold'], rule['market'], rule['model_identifier'], rule['conflict_group']

        if prob_col not in daily_minimal_df.columns or target_col not in daily_minimal_df.columns:
            # logger.debug(f"Skipping rule for market {market}, model {model_id}: prob_col '{prob_col}' or target_col '{target_col}' not in daily_df columns.")
            continue

        eligible_matches_for_rule = daily_minimal_df[
            pd.notna(daily_minimal_df[prob_col]) & (daily_minimal_df[prob_col] >= threshold)
        ]

        for _, match_row in eligible_matches_for_rule.iterrows():
            match_id = match_row[match_id_col]
            leg_info = {
                'match_id': match_id, 'market': market, 'model_used': model_id,
                'prob_at_bet': float(match_row[prob_col]),
                'threshold_used': float(threshold),
                'actual_outcome': int(match_row[target_col]) if pd.notna(match_row[target_col]) else -1,
                'conflict_group': conflict_group,
            }
            leg_info.update(match_details_cache.get(match_id, {}))
            eligible_legs.append(leg_info)
    return eligible_legs

def generate_parlays_for_day( # IDENTICAL to previous script
    eligible_legs: List[Dict], game_date: Union[str, pd.Timestamp],
    min_legs: int, max_legs: int, max_combinations_limit: int
) -> List[Dict]:
    if not eligible_legs: return []
    parlay_results = []
    legs_by_match: Dict[str, List[Dict]] = {}
    for leg in eligible_legs:
        legs_by_match.setdefault(leg['match_id'], []).append(leg)
    
    unique_match_ids = list(legs_by_match.keys())
    game_date_str = game_date.strftime('%Y-%m-%d') if hasattr(game_date, 'strftime') else str(game_date)

    for num_legs_iter in range(min_legs, min(max_legs + 1, len(unique_match_ids) + 1)):
        for match_combo_tuple in combinations(unique_match_ids, num_legs_iter):
            leg_options_for_combo: List[List[Dict]] = [legs_by_match[mid] for mid in match_combo_tuple]
            num_potential_parlays = np.prod([len(opts) for opts in leg_options_for_combo], dtype=np.int64)
            if num_potential_parlays == 0: continue
            if num_potential_parlays > max_combinations_limit:
                logger.warning(f"Date {game_date_str}: Combo {match_combo_tuple} excessive combinations: {num_potential_parlays}")
                continue

            for leg_combination in product(*leg_options_for_combo):
                conflict_groups_in_parlay = [leg['conflict_group'] for leg in leg_combination]
                if len(set(conflict_groups_in_parlay)) != len(conflict_groups_in_parlay):
                    continue
                
                parlay_won = all(leg['actual_outcome'] == 1 for leg in leg_combination)
                valid_probs = [leg['prob_at_bet'] for leg in leg_combination if pd.notna(leg.get('prob_at_bet'))]
                avg_prob = np.mean(valid_probs) if valid_probs else 0.0
                first_leg = leg_combination[0]

                record = {
                    'parlay_date': game_date_str, 'num_legs': num_legs_iter, 'parlay_won': int(parlay_won),
                    'avg_prob': float(avg_prob),
                    'parlay_country': first_leg.get('match_country', 'Unknown'),
                    'parlay_league': first_leg.get('match_league', 'Unknown'),
                    'market_combination': '+'.join(sorted({leg['market'] for leg in leg_combination}))
                }
                for i, leg in enumerate(leg_combination, 1):
                    record.update({
                        f'leg{i}_match_id': leg['match_id'], f'leg{i}_market': leg['market'],
                        f'leg{i}_model': leg['model_used'], f'leg{i}_prob': float(leg['prob_at_bet']),
                        f'leg{i}_won': int(leg['actual_outcome']),
                        f'leg{i}_country': leg.get('match_country', 'Unknown'),
                        f'leg{i}_league': leg.get('match_league', 'Unknown')
                    })
                parlay_results.append(record)
    return parlay_results

def sample_dates_yearly( # IDENTICAL to previous script
    all_dates: pd.Series, sample_rate: float
) -> List[Any]: # Changed to List[Any] as pd.Timestamp is not directly available here
    if sample_rate == 1.0: return sorted(all_dates.unique().tolist())
    all_dates_df = pd.DataFrame({'date': pd.to_datetime(all_dates).dt.date}).drop_duplicates()
    all_dates_df['year'] = pd.to_datetime(all_dates_df['date']).dt.year
    sampled_dates = []
    np.random.seed(42)
    for year, group in all_dates_df.groupby('year'):
        n_to_sample = max(1, int(np.round(len(group) * sample_rate)))
        sampled_indices = np.random.choice(group.index, size=n_to_sample, replace=False)
        sampled_dates.extend(group.loc[sampled_indices, 'date'].tolist())
    return sorted(list(set(sampled_dates)))

def limit_legs_per_match(eligible_legs: List[Dict], max_legs_per_match: int = 3) -> List[Dict]:
    """
    Limit the number of eligible legs per match to prevent combinatorial explosion.
    For each match, keep only the top N legs with highest probability.
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

# --- MODIFIED Main Backtesting Orchestration ---
def run_parlay_backtest_elite_strategy(
    combined_oof_df: pd.DataFrame,
    elite_strategy_config_list: List[Dict[str, Any]],
    market_definitions: Dict[str, Dict],
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
    reuse_checkpoints: bool = True,
    process_only_new_chunks: bool = True  # New parameter
) -> Optional[pd.DataFrame]:
    logger.info("\n" + "="*30 + " Starting ELITE STRATEGY Parlay Backtest " + "="*30)
    overall_start_time = time.time()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path_parquet.parent.mkdir(parents=True, exist_ok=True)

    # --- Build Elite Strategy DataFrame ---
    try:
        elite_strategy_guide_df = load_and_prepare_elite_strategy(elite_strategy_config_list, market_definitions)
        logger.info(f"Elite strategy guide processed: {len(elite_strategy_guide_df)} rules.")
        assert not elite_strategy_guide_df.empty, "Elite strategy guide is empty after processing."
        required_oof_cols_for_elite_strategy = set(elite_strategy_guide_df['prob_col']) | set(elite_strategy_guide_df['target_col'])
        missing_oof_cols = required_oof_cols_for_elite_strategy - set(combined_oof_df.columns)
        assert not missing_oof_cols, f"OOF DataFrame is missing columns required by Elite Strategy: {missing_oof_cols}"
    except Exception as e:
        logger.critical(f"Failed to load or prepare elite strategy: {e}", exc_info=True)
        return None

    # --- Date Sampling and Data Filtering ---
    all_available_dates = combined_oof_df[date_col].drop_duplicates()
    dates_to_process = sample_dates_yearly(all_available_dates, yearly_sample_rate)
    if not dates_to_process: return None
    del all_available_dates; gc.collect()

    # --- Define chunks ---
    date_chunks: List[Tuple[str, List[Any]]] = []
    for i in range(0, len(dates_to_process), days_per_chunk):
        chunk_dates_list = dates_to_process[i : i + days_per_chunk]
        if chunk_dates_list:
            start_date_str = chunk_dates_list[0].strftime('%Y%m%d')
            end_date_str = chunk_dates_list[-1].strftime('%Y%m%d')
            chunk_id = f"chunk_elite_{start_date_str}_to_{end_date_str}"
            date_chunks.append((chunk_id, chunk_dates_list))
    
    # --- Identify which chunks to process ---
    if process_only_new_chunks:
        # First, identify existing chunks (valid parquet files)
        existing_chunks = []
        chunks_to_process = []
        
        for chunk_id, chunk_dates_list in date_chunks:
            checkpoint_file = checkpoint_dir / f"{chunk_id}.parquet"
            if reuse_checkpoints and checkpoint_file.exists():
                try:
                    # Try to read the file to make sure it's valid
                    chunk_df = pd.read_parquet(checkpoint_file)
                    if not chunk_df.empty:
                        existing_chunks.append(chunk_id)
                        continue
                except Exception:
                    # If there's an error reading the file, process this chunk
                    pass
            
            # If we get here, either the file doesn't exist or it's invalid/empty
            chunks_to_process.append((chunk_id, chunk_dates_list))
        
        logger.info(f"Found {len(existing_chunks)} existing valid chunks")
        logger.info(f"Will process {len(chunks_to_process)} new or invalid chunks")
        
        if not chunks_to_process:
            logger.info("All chunks have already been processed. Loading existing results.")
            # Load all existing chunks
            all_results_dfs = []
            for chunk_id in existing_chunks:
                checkpoint_file = checkpoint_dir / f"{chunk_id}.parquet"
                try:
                    chunk_df = pd.read_parquet(checkpoint_file)
                    all_results_dfs.append(chunk_df)
                except Exception as e:
                    logger.warning(f"Failed to load existing chunk {chunk_id}: {e}")
            
            if not all_results_dfs:
                logger.warning("No results could be loaded from existing chunks.")
                return None
                
            # Combine all results
            final_df = pd.concat(all_results_dfs, ignore_index=True)
            logger.info(f"Final ELITE DataFrame shape (from existing chunks): {final_df.shape}")
            return final_df
    else:
        # Process all chunks as before
        chunks_to_process = date_chunks
    
    # --- Process the filtered data ---
    oof_filtered_by_date = combined_oof_df[combined_oof_df[date_col].isin(
        [date for _, chunk_dates in chunks_to_process for date in chunk_dates]
    )].copy()
    
    if oof_filtered_by_date.empty:
        logger.info("No new data to process after filtering by dates.")
        # If we have existing chunks, load and return those
        if process_only_new_chunks and existing_chunks:
            all_results_dfs = []
            for chunk_id in existing_chunks:
                checkpoint_file = checkpoint_dir / f"{chunk_id}.parquet"
                try:
                    chunk_df = pd.read_parquet(checkpoint_file)
                    all_results_dfs.append(chunk_df)
                except Exception as e:
                    logger.warning(f"Failed to load existing chunk {chunk_id}: {e}")
            
            if all_results_dfs:
                final_df = pd.concat(all_results_dfs, ignore_index=True)
                logger.info(f"Final ELITE DataFrame shape (from existing chunks): {final_df.shape}")
                return final_df
        return None
    
    del combined_oof_df; gc.collect()
    
    # Essential columns now come from the elite_strategy_guide_df
    essential_cols = {date_col, match_id_col} | set(elite_strategy_guide_df['prob_col']) | set(elite_strategy_guide_df['target_col'])
    available_essential_cols = [col for col in essential_cols if col in oof_filtered_by_date.columns]
    oof_minimal = oof_filtered_by_date[available_essential_cols].copy()
    del oof_filtered_by_date; gc.collect()
    logger.info(f"Minimal OOF DataFrame shape for ELITE processing: {oof_minimal.shape}")
    
    # --- Process chunks ---
    all_results_dfs: List[pd.DataFrame] = []
    
    # First, load all existing chunks if we're only processing new ones
    if process_only_new_chunks:
        for chunk_id in existing_chunks:
            checkpoint_file = checkpoint_dir / f"{chunk_id}.parquet"
            try:
                chunk_df = pd.read_parquet(checkpoint_file)
                all_results_dfs.append(chunk_df)
                logger.info(f"Loaded {len(chunk_df)} elite parlays from existing chunk {chunk_id}")
            except Exception as e:
                logger.warning(f"Failed to load existing chunk {chunk_id}: {e}")
    
    # Process new chunks
    for chunk_index, (chunk_id, chunk_dates_list) in enumerate(tqdm(chunks_to_process, desc="Processing Elite Chunks")):
        checkpoint_file = checkpoint_dir / f"{chunk_id}.parquet"
        
        chunk_oof_df = oof_minimal[oof_minimal[date_col].isin(chunk_dates_list)].copy()
        if chunk_oof_df.empty:
            logger.info(f"Chunk {chunk_id} has no data after filtering. Skipping.")
            continue
        
        chunk_parlay_results: List[Dict] = []
        for game_date_obj in sorted(chunk_oof_df[date_col].unique()):
            is_critical, _ = check_memory_usage()
            if is_critical:
                logger.error(f"CRITICAL memory for date {game_date_obj}. Skipping rest of chunk {chunk_id}.")
                break
            
            daily_df = chunk_oof_df[chunk_oof_df[date_col] == game_date_obj]
            if daily_df.empty: continue

            try:
                eligible_legs = find_eligible_legs_for_day(
                    daily_games_df=daily_df,
                    strategy_guide=elite_strategy_guide_df,
                    team_resolver=team_info_resolver,
                    match_id_col=match_id_col
                )
                if eligible_legs:
                    limited_eligible_legs = limit_legs_per_match(eligible_legs, max_legs_per_match=max_legs)
                    daily_parlays = generate_parlays_for_day(
                        eligible_legs=limited_eligible_legs, game_date=game_date_obj,
                        min_legs=min_legs, max_legs=max_legs,
                        max_combinations_limit=max_combinations_limit
                    )
                    if daily_parlays: chunk_parlay_results.extend(daily_parlays)
            except Exception as e:
                logger.error(f"Error processing elite date {game_date_obj}: {e}", exc_info=True)
            del daily_df
        
        if chunk_parlay_results:
            chunk_df = pd.DataFrame(chunk_parlay_results)
            try:
                chunk_df.to_parquet(checkpoint_file, index=False)
                all_results_dfs.append(chunk_df)
                logger.info(f"Elite Chunk {chunk_id}: Saved {len(chunk_df)} parlays.")
            except Exception as e:
                logger.error(f"Failed to save elite checkpoint {checkpoint_file}: {e}")
        else:
            logger.info(f"Elite Chunk {chunk_id}: No parlays generated.")
        del chunk_oof_df; gc.collect()

    if not all_results_dfs:
        logger.warning("No elite parlay results generated.")
        return None
    
    try:
        final_df = pd.concat(all_results_dfs, ignore_index=True)
        logger.info(f"Final ELITE DataFrame shape: {final_df.shape}")
        assert not final_df.empty
        final_df.to_parquet(output_path_parquet, index=False)
        logger.info(f"Elite parlay backtest completed in {time.time() - overall_start_time:.2f}s.")
        return final_df
    except Exception as e:
        logger.critical(f"FATAL: Error aggregating/saving elite results: {e}", exc_info=True)
        return None


# --- Main Execution (Modified) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ELITE STRATEGY Parlay Backtester.")
    parser.add_argument("--oof_path", type=str, default=str(OOF_INPUT_PATH_DEFAULT))
    parser.add_argument("--markets_def_path", type=str, default=str(MARKET_DEFINITIONS_PATH_DEFAULT))
    parser.add_argument("--team_info_path", type=str, default=str(CONSOLIDATED_TEAM_INFO_PATH_DEFAULT))
    parser.add_argument("--output_dir", type=str, default=str(PARLAY_OUTPUT_DIR_ELITE))
    
    # New arguments for controlling legs for this elite run
    parser.add_argument("--min_legs_elite", type=int, default=2, help="Min legs for elite parlays.")
    parser.add_argument("--max_legs_elite", type=int, default=4, help="Max legs for elite parlays (e.g., 2,3,4).")
    parser.add_argument("--elite_strategy_json_path", type=str, default=None, help="Optional: Path to a JSON file defining the ELITE_LEG_STRATEGY_CONFIG.")
    parser.add_argument("--yearly_sample_rate_elite", type=float, default=DEFAULT_YEARLY_SAMPLE_RATE)
    
    # Add new argument for processing only new chunks
    parser.add_argument("--process_only_new_chunks", action="store_true", help="Only process chunks that don't already exist as checkpoints.")
    parser.add_argument("--no_reuse_checkpoints", action="store_true", help="Don't reuse existing checkpoints, reprocess all data.")
    
    args = parser.parse_args()

    output_base_elite = Path(args.output_dir)
    output_base_elite.mkdir(parents=True, exist_ok=True)
    output_parquet_elite_path = output_base_elite / f"{output_base_elite.name}_results.parquet"
    output_csv_elite_path = output_base_elite / f"{output_base_elite.name}_results.csv"
    output_checkpoint_elite_path = output_base_elite / "checkpoints"

    # --- Determine Elite Strategy Config ---
    current_elite_strategy_config = ELITE_LEG_STRATEGY_CONFIG_PLACEHOLDER # Default
    if args.elite_strategy_json_path:
        try:
            with open(args.elite_strategy_json_path, 'r') as f:
                current_elite_strategy_config = json.load(f)
            logger.info(f"Loaded elite strategy from: {args.elite_strategy_json_path}")
            assert isinstance(current_elite_strategy_config, list), "Elite strategy JSON must be a list of dicts."
            assert all(isinstance(item, dict) for item in current_elite_strategy_config), "All items in elite strategy list must be dicts."
        except Exception as e:
            logger.error(f"Failed to load elite strategy from JSON {args.elite_strategy_json_path}: {e}. Using placeholder.")
    else:
        logger.info("Using hardcoded placeholder for ELITE_LEG_STRATEGY_CONFIG. Please update or provide a JSON.")
    
    assert current_elite_strategy_config, "Elite strategy config is empty!"


    try:
        # 1. Load OOF data, market definitions, team info
        logger.info("--- Loading Base Data for Elite Parlay Run ---")
        assert Path(args.oof_path).exists(), f"OOF file not found: {args.oof_path}"
        full_oof_df = pd.read_parquet(Path(args.oof_path))
        full_oof_df[DEFAULT_DATE_COL] = pd.to_datetime(full_oof_df[DEFAULT_DATE_COL]).dt.date

        assert Path(args.markets_def_path).exists(), f"Market definitions not found: {args.markets_def_path}"
        with open(Path(args.markets_def_path), 'r') as f:
            market_definitions_loaded = json.load(f)
        
        team_resolver_instance = TeamInfoResolver(Path(args.team_info_path))

        # 2. Run the elite strategy backtest
        run_parlay_backtest_elite_strategy(
            combined_oof_df=full_oof_df,
            elite_strategy_config_list=current_elite_strategy_config,
            market_definitions=market_definitions_loaded,
            team_info_resolver=team_resolver_instance,
            date_col=DEFAULT_DATE_COL,
            match_id_col=DEFAULT_MATCH_ID_COL,
            min_legs=args.min_legs_elite,
            max_legs=args.max_legs_elite,
            yearly_sample_rate=args.yearly_sample_rate_elite,
            days_per_chunk=DAYS_PER_CHUNK,
            max_combinations_limit=MAX_COMBINATIONS_PER_MATCH_SET,
            checkpoint_dir=output_checkpoint_elite_path,
            output_path_parquet=output_parquet_elite_path,
            output_path_csv=output_csv_elite_path,
            reuse_checkpoints=not args.no_reuse_checkpoints,
            process_only_new_chunks=args.process_only_new_chunks
        )

    except Exception as e:
        logger.critical(f"Unhandled exception during ELITE parlay backtesting: {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- ELITE STRATEGY Parlay Backtester Finished ---")
    sys.exit(0)

def identify_new_chunks(date_chunks: List[Tuple[str, List[Any]]], checkpoint_dir: Path) -> List[Tuple[str, List[Any]]]:
    """
    Identify which chunks need to be processed by checking if they already exist as checkpoint files.
    
    Args:
        date_chunks: List of tuples containing (chunk_id, chunk_dates_list)
        checkpoint_dir: Directory where checkpoint files are stored
        
    Returns:
        List of tuples containing only the chunks that need to be processed
    """
    new_chunks = []
    existing_chunks = []
    
    for chunk_id, chunk_dates_list in date_chunks:
        checkpoint_file = checkpoint_dir / f"{chunk_id}.parquet"
        if checkpoint_file.exists():
            try:
                # Try to read the file to make sure it's valid
                pd.read_parquet(checkpoint_file)
                existing_chunks.append(chunk_id)
            except Exception:
                # If there's an error reading the file, consider it as a new chunk
                new_chunks.append((chunk_id, chunk_dates_list))
        else:
            new_chunks.append((chunk_id, chunk_dates_list))
    
    logger.info(f"Found {len(existing_chunks)} existing chunks and {len(new_chunks)} new chunks to process.")
    return new_chunks

#python /Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/pipeline/improved_parlay.py \
#  --markets_def_path /Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/utils/files/parlay_market_definitions.json \
#  --team_info_path /Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/utils/files/consolidated_team_info.json \
#  --output_dir /Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/data/outputs/predictions/parlay_outputs_V4_elite_strategy \
#  --min_legs_elite 2 \
#  --max_legs_elite 4 \
#  --elite_strategy_json_path /Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/utils/files/elite_config.json \
#  --yearly_sample_rate_elite 1.0 \
#  --process_only_new_chunks