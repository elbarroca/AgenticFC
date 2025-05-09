# parlay_backtester.py

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
import json
import argparse
import time # For basic timing

# --- Configuration ---
BASE_DIR = Path(__file__).parent.parent.parent  # Go up 3 levels from this file to project root
DATA_OUTPUT_DIR = BASE_DIR / 'models' / 'data' / 'outputs' / 'predictions'
MODELS_SAVE_DIR = BASE_DIR / 'models' / 'data' / 'outputs' / 'joblib' / 'V1'
PARLAY_OUTPUT_DIR = DATA_OUTPUT_DIR / 'parlay_outputs'  # Dedicated dir for parlay results

# Input paths
OOF_INPUT_PATH = DATA_OUTPUT_DIR / 'level0_oof_predictions_pca_combined.parquet'
STRATEGY_GUIDE_PATH = DATA_OUTPUT_DIR / 'best_per_market.csv'
MARKET_DEFINITIONS_PATH = BASE_DIR / 'models' / 'config' / 'parlay_market_definitions.json'

# Output paths
PARLAY_RESULTS_PATH = PARLAY_OUTPUT_DIR / 'parlay_backtest_results.csv'
PARLAY_INSIGHTS_PATH = PARLAY_OUTPUT_DIR / 'parlay_model_insights.json'

DEFAULT_DATE_COL = 'Date'
DEFAULT_MATCH_ID_COL = 'MatchID'

def load_data(combined_oof_path: Path, strategy_guide_path: Path, market_definitions_path: Path) -> tuple | None:
    print("--- Loading Data ---")
    s_time = time.time()
    try:
        # Ensure combined_oof_df is loaded efficiently
        # If it's huge, consider if only necessary columns can be pre-selected if possible,
        # but the strategy guide will dictate which ones are needed.
        combined_oof_df = pd.read_parquet(combined_oof_path)
        print(f"Loaded combined_oof_for_betting: {combined_oof_df.shape} (in {time.time()-s_time:.2f}s)")

        s_time_strat = time.time()
        strategy_guide_df = pd.read_csv(strategy_guide_path)
        print(f"Loaded strategy_guide_df (e.g., best_per_market_df): {strategy_guide_df.shape} (in {time.time()-s_time_strat:.2f}s)")
        
        s_time_mkt = time.time()
        with open(market_definitions_path, 'r') as f:
            parlay_market_definitions = json.load(f)
        print(f"Loaded PARLAY_MARKET_DEFINITIONS: {len(parlay_market_definitions)} entries (in {time.time()-s_time_mkt:.2f}s)")
        
        return combined_oof_df, strategy_guide_df, parlay_market_definitions
    except FileNotFoundError as e:
        print(f"CRITICAL: File not found: {e}. Ensure paths are correct.")
        return None
    except Exception as e:
        print(f"CRITICAL: Error loading data: {e}")
        return None

def preprocess_strategy_guide(raw_guide_df: pd.DataFrame, 
                              market_definitions: dict, 
                              oof_df_columns: pd.Index # Pass the actual columns from oof_df
                             ) -> pd.DataFrame | None:
    print("--- Pre-processing Strategy Guide ---")
    s_time = time.time()
    processed_rules = []
    
    # Convert oof_df_columns to a set for faster lookups (O(1) on average)
    oof_column_set = set(oof_df_columns)

    market_details_map = {
        m_label: {
            'target_col': m_info['target_col'],
            'prob_suffix': m_info['prob_suffix'],
            'conflict_group': m_info.get('conflict_group', m_label)
        } for m_label, m_info in market_definitions.items()
    }

    skipped_rules_prob_col = 0
    skipped_rules_target_col = 0

    for _, rule in raw_guide_df.iterrows(): # This loop count is based on number of markets in your guide
        market_name = rule['market']
        model_id = rule['model_identifier'] # This comes from your 'best_per_market_df'
        
        market_info = market_details_map.get(market_name)
        if not market_info or not market_info.get('prob_suffix'):
            continue # Silently skip if market definition is incomplete
        
        # Construct the specific column name for this model's prediction for this market
        prob_col = f"{model_id}_{market_info['prob_suffix']}"
        target_col = market_info['target_col']

        # Efficient check using the set
        if prob_col not in oof_column_set:
            skipped_rules_prob_col += 1
            continue
        if target_col not in oof_column_set:
            skipped_rules_target_col += 1
            continue
            
        processed_rules.append({
            'market': market_name,
            'model_identifier': model_id, # The "best model" for this market
            'efficient_entry_threshold': rule['efficient_entry_threshold'], # The "best entry point"
            'prob_col_to_check': prob_col, # The exact column in OOF data to get the prob from
            'target_col_to_check': target_col, # The exact column for the actual outcome
            'conflict_group': market_info['conflict_group']
        })
    
    if skipped_rules_prob_col > 0:
        print(f"Warning: Skipped {skipped_rules_prob_col} strategy rules due to missing probability columns in OOF data.")
    if skipped_rules_target_col > 0:
        print(f"Warning: Skipped {skipped_rules_target_col} strategy rules due to missing target columns in OOF data.")

    if not processed_rules:
        print("ERROR: No valid rules after pre-processing strategy guide. Check input files and column name consistency.")
        return None
    
    print(f"Pre-processing strategy guide complete: {len(processed_rules)} valid rules found (in {time.time()-s_time:.2f}s)")
    return pd.DataFrame(processed_rules)

def run_parlay_backtest(
    combined_oof_df: pd.DataFrame, 
    strategy_guide_df: pd.DataFrame, # This is the pre-processed one
    date_col: str, 
    match_id_col: str,
    max_legs: int, 
    min_legs: int,
    sample_percentage: float = 1.0
    ):
    """The core parlay backtesting logic."""
    
    parlay_input_df = combined_oof_df.copy()
    
    if date_col not in parlay_input_df.columns:
        raise KeyError(f"FATAL ERROR: Date column '{date_col}' not found.")
    try:
        parlay_input_df[date_col] = pd.to_datetime(parlay_input_df[date_col])
    except Exception as e:
        raise ValueError(f"FATAL ERROR: Could not convert '{date_col}' to datetime: {e}.")

    if sample_percentage < 1.0 and sample_percentage > 0.0:
        unique_dates = sorted(parlay_input_df[date_col].unique())
        num_sample_dates = max(1, int(len(unique_dates) * sample_percentage))
        # Ensure reproducibility of sampling if desired by setting np.random.seed() before this
        # np.random.seed(42) # Example
        sampled_dates = np.random.choice(unique_dates, size=num_sample_dates, replace=False)
        parlay_input_df = parlay_input_df[parlay_input_df[date_col].isin(sampled_dates)]
        print(f"Using a {sample_percentage*100:.0f}% sample of days: {num_sample_dates} days selected for backtesting.")
    elif sample_percentage == 1.0:
        print("Using 100% of days for backtesting.")
    else:
        print(f"Warning: Invalid sample_percentage ({sample_percentage}). Using 100% of days.")
        sample_percentage = 1.0 # Default to full if invalid


    parlay_results_accumulator = []
    print(f"\nStarting parlay generation for parlays with {min_legs} to {max_legs} legs.")
    
    unique_days_to_process = parlay_input_df[date_col].unique()
    total_days = len(unique_days_to_process)
    processed_days_count = 0
    
    if total_days == 0:
        print("No days to process after sampling (or in original data).")
        return None, None


    for game_date in unique_days_to_process: # Iterate over unique dates to avoid issues with groupby object
        daily_games_df = parlay_input_df[parlay_input_df[date_col] == game_date]
        processed_days_count += 1
        if processed_days_count % 50 == 0 or processed_days_count == 1 or processed_days_count == total_days : # Print progress
             print(f"Processing day {processed_days_count}/{total_days} ({pd.to_datetime(game_date).date()})... Eligible legs found so far today: ", end="")

        eligible_legs_for_this_day = []
        for _, game_row in daily_games_df.iterrows(): # Iterates over games for that specific day
            match_id = game_row[match_id_col]
            # For each game, check against all rules in our (already filtered) strategy guide
            for _, strategy_rule in strategy_guide_df.iterrows():
                entry_thresh = strategy_rule['efficient_entry_threshold']
                prob_col = strategy_rule['prob_col_to_check'] # Already validated to exist
                target_col = strategy_rule['target_col_to_check'] # Already validated to exist

                # Check if the specific game_row actually has a value for this prob_col
                # (it should, as strategy_guide was pre-filtered based on oof_df columns)
                if pd.notna(game_row[prob_col]) and pd.notna(game_row[target_col]):
                    predicted_prob = game_row[prob_col]
                    if predicted_prob >= entry_thresh:
                        actual_outcome = game_row[target_col]
                        eligible_legs_for_this_day.append({
                            'game_date': game_date, 'match_id': match_id, 'market': strategy_rule['market'],
                            'model_used': strategy_rule['model_identifier'], 'prob_at_bet': predicted_prob,
                            'threshold_used': entry_thresh, 'actual_outcome': int(actual_outcome),
                            'conflict_group': strategy_rule['conflict_group']
                        })
        
        print(f"{len(eligible_legs_for_this_day)}") # Complete the progress print for the day

        if not eligible_legs_for_this_day or len(eligible_legs_for_this_day) < min_legs:
            continue

        for num_legs_in_parlay in range(min_legs, max_legs + 1):
            if len(eligible_legs_for_this_day) < num_legs_in_parlay: continue

            for parlay_legs_tuple in combinations(eligible_legs_for_this_day, num_legs_in_parlay):
                match_ids_in_parlay = {leg['match_id'] for leg in parlay_legs_tuple}
                if len(match_ids_in_parlay) != num_legs_in_parlay: continue # Distinct games only

                parlay_won = all(leg['actual_outcome'] == 1 for leg in parlay_legs_tuple)
                record = {'parlay_date': game_date, 'num_legs': num_legs_in_parlay, 'parlay_won': int(parlay_won)}
                for i, leg_info in enumerate(parlay_legs_tuple):
                    record[f'leg{i+1}_match_id'] = leg_info['match_id']
                    record[f'leg{i+1}_market'] = leg_info['market']
                    record[f'leg{i+1}_model'] = leg_info['model_used']
                    record[f'leg{i+1}_won'] = leg_info['actual_outcome']
                    record[f'leg{i+1}_prob'] = leg_info['prob_at_bet']
                parlay_results_accumulator.append(record)
    
    if not parlay_results_accumulator:
        print("No hypothetical parlays were generated that met all criteria.")
        return None # Return None if no results, was (None,None)

    parlay_summary_df = pd.DataFrame(parlay_results_accumulator)
    print(f"\n--- Parlay Backtest Summary (Sample: {sample_percentage*100:.0f}% of days) ---")
    print(f"Total unique parlays generated: {len(parlay_summary_df)}")

    for num_legs_val, group in parlay_summary_df.groupby('num_legs'):
        total_count = len(group)
        wins = group['parlay_won'].sum()
        win_rate = (wins / total_count) * 100 if total_count > 0 else 0
        print(f"  {num_legs_val}-Leg Parlays: Count={total_count}, Wins={wins}, Win Rate={win_rate:.2f}%")
    
    return parlay_summary_df # Only return the df, model usage can be a separate call

def analyze_model_usage_in_parlays(parlay_summary_df: pd.DataFrame, max_legs: int) -> pd.Series | None:
    """Analyzes model frequency in parlay legs.""" # Changed from "successful parlay legs"
    if parlay_summary_df is None or parlay_summary_df.empty:
        print("No parlay summary DataFrame to analyze model usage.")
        return None
        
    leg_model_usage = []
    for i in range(1, max_legs + 1): # Ensure max_legs matches what was generated
        model_col = f'leg{i}_model'
        if model_col in parlay_summary_df.columns:
            leg_model_usage.extend(parlay_summary_df[model_col].dropna().tolist()) # Add dropna()
    
    if not leg_model_usage:
        print("No model usage data found in parlay legs (possibly no parlays or model column missing/empty).")
        return None
        
    model_counts = pd.Series(leg_model_usage).value_counts(normalize=True) * 100
    return model_counts

def run_parlay_backtester(
    oof_path: str = str(OOF_INPUT_PATH),
    strategy_path: str = str(STRATEGY_GUIDE_PATH),
    markets_def_path: str = str(MARKET_DEFINITIONS_PATH),
    output_path: str = str(PARLAY_RESULTS_PATH),
    max_legs: int = 3,
    min_legs: int = 2,
    sample_perc: float = 1.0,
    date_col: str = DEFAULT_DATE_COL,
    match_id_col: str = DEFAULT_MATCH_ID_COL
):
    """Main function to run the parlay backtester with arguments as parameters instead of CLI."""
    
    # --- 1. Load Data ---
    data_load_result = load_data(Path(oof_path), Path(strategy_path), Path(markets_def_path))
    if data_load_result is None:
        print("Exiting due to data loading errors.")
        return None
    combined_oof_df, raw_strategy_guide_df, parlay_market_definitions_loaded = data_load_result

    # --- 2. Pre-process Strategy Guide ---
    strategy_guide_processed_df = preprocess_strategy_guide(
        raw_strategy_guide_df, 
        parlay_market_definitions_loaded,
        combined_oof_df.columns
    )
    if strategy_guide_processed_df is None:
        print("Exiting due to strategy guide pre-processing errors.")
        return None

    # --- 3. Run Backtest ---
    s_time_backtest = time.time()
    results_df = run_parlay_backtest(
        combined_oof_df=combined_oof_df,
        strategy_guide_df=strategy_guide_processed_df,
        date_col=date_col,
        match_id_col=match_id_col,
        max_legs=max_legs,
        min_legs=min_legs,
        sample_percentage=sample_perc
    )
    print(f"Parlay generation and evaluation took {time.time() - s_time_backtest:.2f} seconds.")

    # --- 4. Save Results & Analyze Model Usage ---
    if results_df is not None and not results_df.empty:
        print(f"\nSaving parlay results to: {output_path}")
        results_df.to_csv(output_path, index=False)

        print("\nSample of Final Parlay Results (first 5):")
        print(results_df.head())

        model_usage_stats = analyze_model_usage_in_parlays(results_df, max_legs)
        if model_usage_stats is not None:
            print("\n\n--- Insights for Stacker Model from Parlay Leg Usage ---")
            print("Frequency of Models appearing in generated parlay legs (based on strategy guide):")
            print(model_usage_stats.round(2).to_string())
            print("\nInterpretation for Stacker:")
            print("- Models appearing more frequently here are those your strategy guide relies on often.")
            print("- When building your stacker, features from these frequently used models might be particularly important.")
    else:
        print("Parlay backtesting did not produce any results to save or analyze further.")

    print("\nScript execution complete.")
    return results_df

# --- Main Execution Block ---
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    PARLAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Run Parlay Backtester for Football Predictions.")
    parser.add_argument("--oof_path", type=str, default=str(OOF_INPUT_PATH), 
                       help="Path to the combined OOF predictions Parquet file.")
    parser.add_argument("--strategy_path", type=str, default=str(STRATEGY_GUIDE_PATH),
                       help="Path to the strategy guide CSV file (e.g., best_per_market_df).")
    parser.add_argument("--markets_def_path", type=str, default=str(MARKET_DEFINITIONS_PATH),
                       help="Path to the PARLAY_MARKET_DEFINITIONS JSON file.")
    parser.add_argument("--output_path", type=str, default=str(PARLAY_RESULTS_PATH),
                       help="Path to save the parlay results CSV.")
    parser.add_argument("--max_legs", type=int, default=3,
                       help="Maximum number of legs per parlay.")
    parser.add_argument("--min_legs", type=int, default=2,
                       help="Minimum number of legs per parlay.")
    parser.add_argument("--sample_perc", type=float, default=1.0,
                       help="Percentage of days to sample (0.0 to 1.0). Default 1.0 (all days).")
    parser.add_argument("--date_col", type=str, default=DEFAULT_DATE_COL,
                       help="Name of the date column in OOF data.")
    parser.add_argument("--match_id_col", type=str, default=DEFAULT_MATCH_ID_COL,
                       help="Name of the match ID column.")

    args = parser.parse_args()
    
    # Run the backtester with CLI arguments
    run_parlay_backtester(
        oof_path=args.oof_path,
        strategy_path=args.strategy_path,
        markets_def_path=args.markets_def_path,
        output_path=args.output_path,
        max_legs=args.max_legs,
        min_legs=args.min_legs,
        sample_perc=args.sample_perc,
        date_col=args.date_col,
        match_id_col=args.match_id_col
    )