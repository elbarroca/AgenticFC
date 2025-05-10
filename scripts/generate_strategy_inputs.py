# generate_strategy_inputs.py
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import time
# Import necessary functions if they were defined in your notebook's Cell 2
# For this script, we'll redefine simplified versions or assume they are part of the logic below.

# --- Configuration ---
# Assuming this script is in the root of your AGENTICFC888 project directory
PROJECT_ROOT_PATH = Path('/Users/barroca888/Downloads/Agenticfc/AgenticFC888')
BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = PROJECT_ROOT_PATH / 'models' / 'data' / 'outputs' / 'predictions'
PLOT_OUTPUT_DIR = PREDICTIONS_DIR / 'plots' # New directory for plots
CONFIG_DIR = PROJECT_ROOT_PATH / 'scripts' / 'models' / 'config'
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure plot directory exists

# --- Default File Names ---
# Input: The already merged OOF file from your notebook's Cell 3
DEFAULT_COMBINED_OOF_INPUT_FILE = 'combined_oof_ALL_pipelines.parquet'
# Input: Market definitions
DEFAULT_MARKET_DEFS_FILE = 'parlay_market_definitions.json'
# Output of this script
DEFAULT_STRATEGY_GUIDE_OUTPUT_FILE = 'best_strategy_per_market.csv'
# --- Global Column Names (MUST MATCH YOUR DATA in the input OOF file) ---
MATCH_ID_COL = 'MatchID'
DATE_COL = 'Date'
FTHG_COL = 'FTHG'
FTAG_COL = 'FTAG'
FTR_COL = 'FTR'
# --- Default Parameters ---
# For initial SBR calculation & viability filtering
DEFAULT_SBR_CALC_MIN_BETS_FOR_WR = 1000
DEFAULT_MIN_BETS_FOR_ANY_CONSIDERATION = 100 
DEFAULT_MIN_PEAK_WR_OVERALL = 62.0 # A (model,market) must hit this WR at some point
DEFAULT_EXCLUDED_MARKETS = ['Over4.5'] # Keep Draw exclusion for now

# For "Optimal Trade-off Point" Logic - THIS IS NOW THE PRIMARY LOGIC FOR THE CSV
DEFAULT_DESIRABLE_WR_FOR_TRADEOFF = 62.0 # Target minimum WR for the trade-off point
DEFAULT_MIN_BETS_AT_TRADEOFF_POINT = 1000  # Min bets for a point to be considered a trade-off candidate

# Parameters for the "Overall Top Strategies" and "Best Model (Max WR) per Market" tables (for context)
DEFAULT_OVERALL_TOP_N_STRATEGIES = 30
DEFAULT_OVERALL_MIN_WIN_RATE = 62.0
DEFAULT_OVERALL_MIN_BETS_FILTER = 1000
DEFAULT_BEST_PER_MARKET_MIN_BETS_FILTER = 1000

# --- Default Parameters for Plotting ---
DEFAULT_PLOT_NUM_TOP_STRATEGIES = 30 # Matches the CSV count
# (Using the versions from your last provided script as they were good)
def load_prerequisites(combined_oof_path: Path, market_definitions_path: Path) -> tuple | None:
    print("--- Loading Prerequisite Data ---")
    try:
        s_time = time.time()
        print(f"Attempting to load combined OOF data from: {combined_oof_path}")
        if not combined_oof_path.exists():
            print(f"CRITICAL: OOF File not found: {combined_oof_path}")
            return None
        combined_oof_df = pd.read_parquet(combined_oof_path)
        print(f"Loaded combined_oof_df: {combined_oof_df.shape} (in {time.time()-s_time:.2f}s)")

        s_time_mkt = time.time()
        print(f"Attempting to load market definitions from: {market_definitions_path}")
        if not market_definitions_path.exists():
            print(f"CRITICAL: Market Definitions File not found: {market_definitions_path}")
            print(f"Please create '{market_definitions_path.name}' in '{market_definitions_path.parent}'")
            return None
        with open(market_definitions_path, 'r') as f:
            parlay_market_definitions = json.load(f)
        print(f"Loaded PARLAY_MARKET_DEFINITIONS: {len(parlay_market_definitions)} entries (in {time.time()-s_time_mkt:.2f}s)")
        
        return combined_oof_df, parlay_market_definitions
    except Exception as e:
        print(f"CRITICAL: Error loading prerequisite data: {e}")
        return None

def create_all_binary_targets(df: pd.DataFrame, parlay_market_definitions: dict,
                               match_id_col:str, date_col:str, 
                               fthg_col:str, ftag_col:str, ftr_col:str) -> pd.DataFrame:
    print("--- Creating/Ensuring All Binary Target Columns ---")
    new_columns_dict = {}
    if ftr_col in df.columns:
        if 'target_H' not in df.columns: new_columns_dict['target_H'] = (df[ftr_col] == 'H').astype(int)
        if 'target_D' not in df.columns: new_columns_dict['target_D'] = (df[ftr_col] == 'D').astype(int)
        if 'target_A' not in df.columns: new_columns_dict['target_A'] = (df[ftr_col] == 'A').astype(int)
    if fthg_col in df.columns and ftag_col in df.columns:
        temp_fthg = pd.to_numeric(df[fthg_col], errors='coerce')
        temp_ftag = pd.to_numeric(df[ftag_col], errors='coerce')
        valid_scores_mask = temp_fthg.notna() & temp_ftag.notna()
        if valid_scores_mask.any():
            total_goals_valid = temp_fthg[valid_scores_mask] + temp_ftag[valid_scores_mask]
            for gt_val_float in [0.5, 1.5, 2.5, 3.5, 4.5]:
                gt_val_str = str(gt_val_float).replace('.', '')
                target_o_col, target_u_col = f'target_O{gt_val_str}', f'target_U{gt_val_str}'
                if target_o_col not in df.columns: 
                    series_o = pd.Series(np.nan, index=df.index); series_o.loc[valid_scores_mask] = (total_goals_valid > gt_val_float).astype(int); new_columns_dict[target_o_col] = series_o
                if target_u_col not in df.columns:
                    series_u = pd.Series(np.nan, index=df.index); series_u.loc[valid_scores_mask] = (total_goals_valid < gt_val_float).astype(int); new_columns_dict[target_u_col] = series_u
            if 'target_BTTS_Y' not in df.columns:
                series_btts_y = pd.Series(np.nan, index=df.index); series_btts_y.loc[valid_scores_mask] = ((temp_fthg[valid_scores_mask] > 0) & (temp_ftag[valid_scores_mask] > 0)).astype(int); new_columns_dict['target_BTTS_Y'] = series_btts_y
            if 'target_BTTS_N' not in df.columns:
                btts_y_series_for_n = new_columns_dict.get('target_BTTS_Y', df.get('target_BTTS_Y'))
                if btts_y_series_for_n is not None: new_columns_dict['target_BTTS_N'] = 1 - btts_y_series_for_n
    df_out = df.assign(**new_columns_dict); new_columns_dict.clear()
    for mkt_label, m_info in parlay_market_definitions.items():
        target_col, base_targets, operation = m_info['target_col'], m_info.get('base_targets', []), m_info.get('op')
        if target_col in df_out.columns or not (operation and base_targets): continue
        
        # Ensure all base targets for complex ones are actually present before trying to use them
        if not all(bt in df_out.columns for bt in base_targets):
            # print(f"    Debug Targets: Skipping complex target '{target_col}' for market '{mkt_label}', missing base targets: {[bt for bt in base_targets if bt not in df_out.columns]}")
            continue
        try:
            series_list = [df_out[bt].astype(bool) for bt in base_targets]
            if operation == 'and' and len(series_list) == 2: new_columns_dict[target_col] = (series_list[0] & series_list[1]).astype(int)
            elif operation == 'or' and len(series_list) == 2: new_columns_dict[target_col] = (series_list[0] | series_list[1]).astype(int)
        except KeyError as e: print(f"    KeyError creating complex target '{target_col}' from {base_targets}. Error: {e}")
        except Exception as e: print(f"    Error creating complex target '{target_col}': {e}")
    df_out = df_out.assign(**new_columns_dict)
    print(f"--- Binary target creation/verification complete. Final df shape: {df_out.shape} ---")
    return df_out.copy()

def calculate_comprehensive_sbr(combined_oof_df_with_targets: pd.DataFrame, parlay_market_definitions: dict, 
                                prob_thresholds: list, min_bets_for_wr_stat: int, match_id_col: str) -> pd.DataFrame:
    print("--- Calculating Comprehensive Single Bet Records (SBR) ---")
    s_time = time.time(); sbr_records = []; model_identifiers_present = set()
    all_prob_suffixes_map = {m_info['prob_suffix']: m_label for m_label, m_info in parlay_market_definitions.items()}
    for col in combined_oof_df_with_targets.columns:
        if col.startswith("target_") or col in [MATCH_ID_COL, DATE_COL, FTHG_COL, FTAG_COL, FTR_COL]: continue
        for prob_sfx_key, mkt_label_val in all_prob_suffixes_map.items():
            full_sfx_to_check = f"_{prob_sfx_key}"
            if col.endswith(full_sfx_to_check):
                model_id_candidate = col[:-len(full_sfx_to_check)]
                if model_id_candidate: model_identifiers_present.add(model_id_candidate)
                break 
    print(f"  Found {len(model_identifiers_present)} unique model identifiers (e.g., {list(model_identifiers_present)[:5]}...).")
    if not model_identifiers_present: print("  CRITICAL: No model_ids found."); return pd.DataFrame()
    for model_id in sorted(list(model_identifiers_present)):
        for market_label, market_info in parlay_market_definitions.items(): # Use market_label as the key from parlay_market_definitions
            prob_col, target_col = f"{model_id}_{market_info['prob_suffix']}", market_info['target_col']
            if prob_col not in combined_oof_df_with_targets.columns: continue
            if target_col not in combined_oof_df_with_targets.columns:
                # print(f"DEBUG: Target column '{target_col}' for market '{market_label}' (key from JSON) not found. Skipping SBR.")
                continue
            if not pd.api.types.is_numeric_dtype(combined_oof_df_with_targets[target_col]):
                try: combined_oof_df_with_targets[target_col] = combined_oof_df_with_targets[target_col].astype(int)
                except ValueError: continue
            if not pd.api.types.is_numeric_dtype(combined_oof_df_with_targets[prob_col]) or combined_oof_df_with_targets[prob_col].isnull().all(): continue
            analysis_slice = combined_oof_df_with_targets[[prob_col, target_col]].dropna()
            if analysis_slice.empty : continue
            for conf_thresh in prob_thresholds:
                selected = analysis_slice[analysis_slice[prob_col] >= conf_thresh]
                n_bets, n_wins = len(selected), (selected[target_col].sum() if len(selected) > 0 else 0)
                wr = (n_wins / n_bets) * 100 if n_bets >= min_bets_for_wr_stat else np.nan
                sbr_records.append({'model_identifier': model_id, 'market': market_label, # Use market_label from JSON key
                                    'confidence_threshold': conf_thresh, 'num_bets_placed': n_bets, 
                                    'num_bets_won': n_wins, 'win_rate_actual_perc': wr})
    sbr_df = pd.DataFrame(sbr_records)
    print(f"Comprehensive SBR calculated: {sbr_df.shape} (in {time.time()-s_time:.2f}s)")
    if sbr_df.empty: print("WARNING: SBR DataFrame is empty post-calculation.")
    return sbr_df

# generate_optimal_tradeoff_strategy_per_market_df (same as your last version)
def generate_optimal_tradeoff_strategy_per_market_df(
    sbr_df: pd.DataFrame, 
    args_ns: argparse.Namespace 
) -> pd.DataFrame | None:
    """
    Generate optimal trade-off strategy for each market, prioritizing volume at good win rates.
    
    Args:
        sbr_df: DataFrame containing Single Bet Records
        args_ns: Namespace with configuration parameters
        
    Returns:
        DataFrame with optimal trade-off strategies per market or None if no valid strategies found
    """
    # Validate input parameters
    assert isinstance(sbr_df, pd.DataFrame), "SBR data must be a pandas DataFrame"
    assert hasattr(args_ns, 'desirable_wr_for_tradeoff'), "args_ns missing required 'desirable_wr_for_tradeoff' attribute"
    assert hasattr(args_ns, 'min_bets_at_tradeoff_point'), "args_ns missing required 'min_bets_at_tradeoff_point' attribute"
    
    if sbr_df.empty:
        print("Comprehensive SBR DataFrame is empty. Cannot generate optimal trade-off strategies.")
        return None
    
    # Required columns check
    required_columns = ['model_identifier', 'market', 'confidence_threshold', 
                        'num_bets_placed', 'num_bets_won', 'win_rate_actual_perc']
    missing_cols = [col for col in required_columns if col not in sbr_df.columns]
    if missing_cols:
        print(f"SBR DataFrame missing required columns: {missing_cols}")
        return None

    print(f"\n--- Generating Optimal Trade-off Strategy for Each Market (v2 - Prioritizing Volume at Good WR) ---")
    print(f"Params: Desirable WR >= {args_ns.desirable_wr_for_tradeoff}%, Min Bets at any considered point >= {args_ns.min_bets_at_tradeoff_point}")
    s_time = time.time()

    # 1. Filter for points that meet a baseline "good enough" win rate and minimum bets
    sbr_candidates = sbr_df[
        (sbr_df['win_rate_actual_perc'] >= args_ns.desirable_wr_for_tradeoff) & # Base WR
        (sbr_df['num_bets_placed'] >= args_ns.min_bets_at_tradeoff_point)       # Min volume
    ].copy()

    if sbr_candidates.empty:
        print(f"No SBR points found meeting initial criteria: WR >= {args_ns.desirable_wr_for_tradeoff}% and Bets >= {args_ns.min_bets_at_tradeoff_point}.")
        return None

    # 2. For each (model, market), find its "best trade-off point"
    #    This point maximizes bets among those meeting the desirable WR.
    #    If bets are tied, higher WR is preferred. If still tied, lower threshold.
    
    # Sort to prepare for picking the best trade-off point per (model, market)
    sbr_candidates.sort_values(
        by=['model_identifier', 'market', 'num_bets_placed', 'win_rate_actual_perc', 'confidence_threshold'],
        ascending=[True, True, False, False, True], # Max bets, then max WR, then min threshold
        inplace=True
    )
    
    # Pick the first row for each (model, market) group after sorting
    model_market_best_tradeoff_points_df = sbr_candidates.drop_duplicates(
        subset=['model_identifier', 'market'],
        keep='first'
    ).reset_index(drop=True)

    print(f"  Found {len(model_market_best_tradeoff_points_df)} (model,market) specific best trade-off points.")

    if model_market_best_tradeoff_points_df.empty:
        print("No (model, market) pairs had a qualifying trade-off point.")
        return None
    
    # 3. For each market, select the model that offers the "best overall" trade-off point.
    #    "Best overall" can be defined by highest win_rate_at_tradeoff among these points.
    #    If win rates are tied, pick the one with more bets_at_tradeoff.
    
    model_market_tradeoff_df_sorted_for_final_selection = model_market_best_tradeoff_points_df.sort_values(
        by=['market', 'win_rate_actual_perc', 'num_bets_placed'], # For each market, prefer higher WR, then higher bets
        ascending=[True, False, False]
    )
    
    final_tradeoff_strategies_df = model_market_tradeoff_df_sorted_for_final_selection.drop_duplicates(
        subset=['market'],
        keep='first' # This gets the best model for each market based on the sort above
    ).reset_index(drop=True)

    # Final sort of the resulting table by market name
    final_tradeoff_strategies_df = final_tradeoff_strategies_df.sort_values(
        by=['market']
    ) 
    
    print(f"Generated {len(final_tradeoff_strategies_df)} optimal trade-off strategies (1 per market) (in {time.time()-s_time:.2f}s).")
    
    # Rename columns for the final CSV
    final_tradeoff_strategies_df = final_tradeoff_strategies_df.rename(columns={
        'confidence_threshold': 'tradeoff_threshold',
        'win_rate_actual_perc': 'win_rate_at_tradeoff',
        'num_bets_placed': 'bets_at_tradeoff',
        'num_bets_won': 'wins_at_tradeoff'
    })
    output_columns = ['market', 'model_identifier', 'tradeoff_threshold', 
                      'win_rate_at_tradeoff', 'bets_at_tradeoff', 'wins_at_tradeoff']
    
    # Ensure 'wins_at_tradeoff' column exists or is created
    if 'wins_at_tradeoff' not in final_tradeoff_strategies_df.columns and 'num_bets_won' in final_tradeoff_strategies_df.columns:
         final_tradeoff_strategies_df = final_tradeoff_strategies_df.rename(columns={'num_bets_won': 'wins_at_tradeoff'})
    elif 'wins_at_tradeoff' not in final_tradeoff_strategies_df.columns: # If num_bets_won also missing
        final_tradeoff_strategies_df['wins_at_tradeoff'] = \
            (final_tradeoff_strategies_df['win_rate_at_tradeoff'] / 100 * final_tradeoff_strategies_df['bets_at_tradeoff']).round(0).astype(int)

    # Validate output data
    assert not final_tradeoff_strategies_df.empty, "Generated strategies dataframe should not be empty"
    assert final_tradeoff_strategies_df['tradeoff_threshold'].notna().all(), "All tradeoff thresholds must be valid numbers"
    assert final_tradeoff_strategies_df['win_rate_at_tradeoff'].notna().all(), "All win rates must be valid numbers"
    
    return final_tradeoff_strategies_df[[col for col in output_columns if col in final_tradeoff_strategies_df.columns]]

# plot_tradeoff_strategies (same as your last version)
def plot_tradeoff_strategies(tradeoff_strategies_df: pd.DataFrame, sbr_full_data_df: pd.DataFrame, 
                             num_to_plot: int, plot_output_dir: Path):
    if tradeoff_strategies_df is None or tradeoff_strategies_df.empty or sbr_full_data_df is None or sbr_full_data_df.empty: print("Not enough data for plots."); return
    print(f"\n--- Generating Plots for Top {num_to_plot} Optimal Trade-off Strategies ---")
    strategies_to_plot = tradeoff_strategies_df.head(num_to_plot)
    for _, strategy_row in strategies_to_plot.iterrows():
        model_to_plot, market_to_plot = strategy_row['model_identifier'], strategy_row['market']
        tradeoff_thresh_val, win_rate_at_tradeoff, bets_at_tradeoff = strategy_row['tradeoff_threshold'], strategy_row['win_rate_at_tradeoff'], strategy_row['bets_at_tradeoff']
        plot_data_full_curve = sbr_full_data_df[(sbr_full_data_df['model_identifier'] == model_to_plot) & (sbr_full_data_df['market'] == market_to_plot)].sort_values('confidence_threshold')
        if plot_data_full_curve.empty: print(f"  No SBR data for plotting curve: {model_to_plot}/{market_to_plot}"); continue
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.plot(plot_data_full_curve['confidence_threshold'], plot_data_full_curve['win_rate_actual_perc'], color='dodgerblue', marker='o', linestyle='-', label='Win Rate Trend', linewidth=2, markersize=7)
        ax1.set_xlabel('Confidence Threshold', fontsize=13); ax1.set_ylabel('Actual Win Rate (%)', color='dodgerblue', fontsize=13)
        ax1.tick_params(axis='y', labelcolor='dodgerblue', labelsize=11); ax1.tick_params(axis='x', labelsize=11)
        ax1.grid(True, linestyle=':', alpha=0.7, axis='y')
        ax1.scatter(tradeoff_thresh_val, win_rate_at_tradeoff, color='darkorange', s=300, edgecolor='black', zorder=10, label=f'Optimal Trade-off Point\nThresh: {tradeoff_thresh_val:.2f}\nWR: {win_rate_at_tradeoff:.1f}%\nBets: {int(bets_at_tradeoff)}')
        ax1.axvline(tradeoff_thresh_val, color='darkorange', linestyle='--', alpha=0.8, linewidth=2.5)
        ax1.axhline(win_rate_at_tradeoff, color='darkorange', linestyle=':', alpha=0.6, linewidth=2)
        ax2 = ax1.twinx()
        ax2.plot(plot_data_full_curve['confidence_threshold'], plot_data_full_curve['num_bets_placed'], color='crimson', marker='x', linestyle='--', label='Num Bets Trend', linewidth=1.5, markersize=6)
        ax2.set_ylabel('Number of Bets Placed', color='crimson', fontsize=13); ax2.tick_params(axis='y', labelcolor='crimson', labelsize=11)
        plt.title(f'Efficiency & Trade-off for: {model_to_plot} \nMarket: {market_to_plot}', fontsize=16, fontweight='bold')
        handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.05, 0.95), fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plot_filename = f"tradeoff_{model_to_plot.replace(' ', '_').replace('/', '-')}_{market_to_plot.replace(' ', '_').replace('/', '-')}.png"
        plt.savefig(plot_output_dir / plot_filename); plt.close(fig)
        print(f"  Saved plot: {plot_filename}")

# display_overall_top_strategies and display_best_model_per_market_max_wr (same as your script)
def display_overall_top_strategies(sbr_df: pd.DataFrame, top_n: int, min_wr: float, min_bets: int):
    if sbr_df.empty: print("SBR is empty for overall top strategies."); return
    print(f"\n--- Overall Top Performing Single Bet Strategies (Min WR: {min_wr}%, Min Bets: {min_bets}, Top: {top_n}) ---")
    overall_top = sbr_df[(sbr_df['win_rate_actual_perc'] >= min_wr) & (sbr_df['num_bets_placed'] >= min_bets)].sort_values(by=['win_rate_actual_perc', 'num_bets_placed'], ascending=[False, False]).head(top_n)
    if overall_top.empty: print("No strategies found for overall top performers.")
    else: print(overall_top[['model_identifier', 'market', 'confidence_threshold', 'num_bets_placed', 'num_bets_won', 'win_rate_actual_perc']])

def display_best_model_per_market_max_wr(sbr_df: pd.DataFrame, min_bets_for_selection: int):
    if sbr_df.empty: print("SBR is empty for best model per market (max WR)."); return
    print(f"\n--- Best Model (Max WR) for Each Market (Min Bets for Selection: {min_bets_for_selection}) ---")
    sbr_filtered = sbr_df[(sbr_df['num_bets_placed'] >= min_bets_for_selection) & (sbr_df['win_rate_actual_perc'].notna())].copy()
    if sbr_filtered.empty: print(f"No strategies with >= {min_bets_for_selection} bets."); return
    sbr_filtered.sort_values(['win_rate_actual_perc', 'num_bets_placed'], ascending=[False, False], inplace=True)
    best_per_market = sbr_filtered.drop_duplicates(subset=['market'], keep='first').sort_values(by='market')
    if best_per_market.empty: print("Could not determine best model (max WR) per market.")
    else: print(best_per_market[['market', 'model_identifier', 'confidence_threshold', 'win_rate_actual_perc', 'num_bets_placed', 'num_bets_won']])

def main():
    """Main function to generate strategy inputs"""
    print("--- Starting: Generate Strategy Inputs Script ---")
    class Args: pass    
    args = Args()

    # Paths
    args.combined_oof_input_path = str(PREDICTIONS_DIR / DEFAULT_COMBINED_OOF_INPUT_FILE)
    args.market_definitions_path = str(CONFIG_DIR / DEFAULT_MARKET_DEFS_FILE)
    args.strategy_guide_output_file = DEFAULT_STRATEGY_GUIDE_OUTPUT_FILE
    
    # Column Names
    args.match_id_col = MATCH_ID_COL
    args.date_col = DATE_COL
    
    # SBR Calculation & Initial Viability Filters for SBR
    args.sbr_calc_min_bets_for_wr = DEFAULT_SBR_CALC_MIN_BETS_FOR_WR
    # Filters for generate_optimal_tradeoff_strategy_per_market_df
    args.desirable_wr_for_tradeoff = DEFAULT_DESIRABLE_WR_FOR_TRADEOFF
    args.min_bets_at_tradeoff_point = DEFAULT_MIN_BETS_AT_TRADEOFF_POINT
    
    # Parameters for contextual display tables (not for the main CSV output)
    args.overall_top_n_strategies = DEFAULT_OVERALL_TOP_N_STRATEGIES
    args.overall_min_win_rate = DEFAULT_OVERALL_MIN_WIN_RATE
    args.overall_min_bets_filter = DEFAULT_OVERALL_MIN_BETS_FILTER
    args.best_per_market_min_bets_filter = DEFAULT_BEST_PER_MARKET_MIN_BETS_FILTER
    
    # Excluded markets for the generate_optimal_tradeoff_strategy_per_market_df
    # This filtering is now done *before* calling the function.
    args.excluded_markets = DEFAULT_EXCLUDED_MARKETS 
    args.min_peak_wr = DEFAULT_MIN_PEAK_WR_OVERALL # For initial SBR filtering
    args.min_bets_consideration = DEFAULT_MIN_BETS_FOR_ANY_CONSIDERATION # For initial SBR filtering

    # Load data prerequisites
    load_res = load_prerequisites(Path(args.combined_oof_input_path), Path(args.market_definitions_path))
    if load_res is None: 
        print("Failed to load prerequisites. Exiting.")
        exit(1)
    
    combined_oof_df, parlay_market_definitions = load_res
    assert isinstance(combined_oof_df, pd.DataFrame), "Combined OOF data must be a pandas DataFrame"
    assert isinstance(parlay_market_definitions, dict), "Market definitions must be a dictionary"

    # Create binary target columns
    combined_oof_df_with_targets = create_all_binary_targets(
        combined_oof_df, parlay_market_definitions,
        args.match_id_col, args.date_col, FTHG_COL, FTAG_COL, FTR_COL
    )
    
    # Define thresholds for SBR calculation
    prob_thresholds_for_sbr = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
    
    # Calculate comprehensive single bet records
    comprehensive_sbr_df = calculate_comprehensive_sbr(
        combined_oof_df_with_targets, parlay_market_definitions,
        prob_thresholds_for_sbr, args.sbr_calc_min_bets_for_wr, args.match_id_col
    )
    if comprehensive_sbr_df.empty: 
        print("CRITICAL: Comprehensive SBR is empty. Exiting.")
        exit(1)
    
    # --- Filter SBR based on EXCLUDED_MARKETS and MIN_PEAK_WR before passing to tradeoff function ---
    print(f"\nFiltering SBR for excluded markets: {args.excluded_markets}")
    sbr_for_analysis = comprehensive_sbr_df[~comprehensive_sbr_df['market'].isin(args.excluded_markets)].copy()
    
    # Apply peak win rate filter if configured
    if args.min_peak_wr > 0:
        print(f"Filtering SBR for markets where at least one model achieved peak WR >= {args.min_peak_wr}% (with >= {args.min_bets_consideration} bets)")
        sbr_for_peak_check = sbr_for_analysis[sbr_for_analysis['num_bets_placed'] >= args.min_bets_consideration].copy()
        if not sbr_for_peak_check.empty:
            peak_wr_info = sbr_for_peak_check.groupby(['model_identifier', 'market'])['win_rate_actual_perc'].max().reset_index()
            peak_wr_info = peak_wr_info.rename(columns={'win_rate_actual_perc': 'peak_achieved_wr'})
            viable_model_markets = peak_wr_info[peak_wr_info['peak_achieved_wr'] >= args.min_peak_wr]
            if not viable_model_markets.empty:
                sbr_for_analysis = pd.merge(sbr_for_analysis, viable_model_markets[['model_identifier', 'market']],
                                           on=['model_identifier', 'market'], how='inner')
                print(f"  SBR filtered by peak WR, remaining rows for analysis: {len(sbr_for_analysis)}")
            else: print("  No (model, market) pairs met the peak WR criteria.")
        else: print("  Not enough data points (with sufficient bets) to apply peak WR filter robustly.")

    # Generate optimal trade-off strategies
    if sbr_for_analysis.empty:
        print("CRITICAL: SBR DataFrame is empty after initial viability filters. Cannot generate trade-off guide.")
        optimal_tradeoff_guide_df = None
    else:
        optimal_tradeoff_guide_df = generate_optimal_tradeoff_strategy_per_market_df(
            sbr_for_analysis, # Use the filtered SBR
            args
        )

    # Display contextual tables using the full comprehensive_sbr_df
    display_overall_top_strategies(comprehensive_sbr_df, args.overall_top_n_strategies, 
                                   args.overall_min_win_rate, args.overall_min_bets_filter)
    display_best_model_per_market_max_wr(comprehensive_sbr_df, args.best_per_market_min_bets_filter)

    # Save results and generate plots if strategies were found
    if optimal_tradeoff_guide_df is not None and not optimal_tradeoff_guide_df.empty:
        num_strategies_found = len(optimal_tradeoff_guide_df)
        strategy_guide_output_path = PREDICTIONS_DIR / args.strategy_guide_output_file
        optimal_tradeoff_guide_df.to_csv(strategy_guide_output_path, index=False, float_format='%.4f') # Save with good precision
        print(f"\nSuccessfully saved {num_strategies_found} Optimal Trade-off Strategies by Market to: {strategy_guide_output_path}")
        print("\nSample of Optimal Trade-off Strategies by Market (CSV Content):")
        print(optimal_tradeoff_guide_df.head())
        
        # Generate plots for visualization
        plot_tradeoff_strategies(
            optimal_tradeoff_guide_df, 
            sbr_for_analysis, # Use the filtered SBR for plotting curves, as it's more relevant
            num_to_plot=min(num_strategies_found, DEFAULT_PLOT_NUM_TOP_STRATEGIES),
            plot_output_dir=PLOT_OUTPUT_DIR
        )
    else:
        print(f"\nFailed to generate Optimal Trade-off Strategies by Market. CSV and plots not generated.")

if __name__ == "__main__":
    main()
    print("\n--- Script to generate strategy inputs finished. ---")