#!/usr/bin/env python3
import sys
import pandas as pd
import json
from pathlib import Path
import argparse
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Assuming ELITE_LEG_STRATEGY_CONFIG_PLACEHOLDER and load_and_prepare_elite_strategy
# are either defined here or imported from your improved_parlay.py or a shared utils module.
# For simplicity, let's redefine a minimal version of load_and_prepare_elite_strategy here.

def prepare_elite_strategy_for_filter(
    elite_strategy_config: list[dict[str, any]],
    market_definitions: dict[str, dict]
) -> list[tuple[str, str, float]]:
    """
    Prepares a list of (prob_col, target_col, min_elite_threshold)
    tuples needed for pre-filtering the OOF data.
    """
    filter_criteria = []
    all_needed_oof_cols = set()

    for rule_config in elite_strategy_config:
        market_name = rule_config['market']
        model_id = rule_config['model_identifier']
        min_threshold = rule_config['min_elite_threshold']

        market_info = market_definitions.get(market_name)
        if not market_info or 'prob_suffix' not in market_info or 'target_col' not in market_info:
            logger.warning(f"Skipping market {market_name} for pre-filter: incomplete market definition.")
            continue

        prob_col = f"{model_id}_{market_info['prob_suffix']}"
        target_col = market_info['target_col'] # Actual outcome column

        filter_criteria.append({'prob_col': prob_col, 'target_col': target_col, 'threshold': min_threshold})
        all_needed_oof_cols.add(prob_col)
        all_needed_oof_cols.add(target_col)

    return filter_criteria, list(all_needed_oof_cols)


def prefilter_oof(
    full_oof_path_str: str,
    elite_strategy_config: list[dict[str, any]], # From JSON or placeholder
    market_definitions_path_str: str,
    output_prefiltered_oof_path_str: str,
    date_col: str = 'Date',
    match_id_col: str = 'MatchID'
):
    logger.info("--- Starting OOF Pre-filtering for Elite Strategy ---")
    full_oof_path = Path(full_oof_path_str)
    market_definitions_path = Path(market_definitions_path_str)
    output_path = Path(output_prefiltered_oof_path_str)

    assert full_oof_path.exists(), f"Full OOF file not found: {full_oof_path}"
    assert market_definitions_path.exists(), f"Market definitions file not found: {market_definitions_path}"

    logger.info(f"Loading full OOF from: {full_oof_path}")
    df_full = pd.read_parquet(full_oof_path)
    original_rows = len(df_full)
    logger.info(f"Original OOF shape: {df_full.shape}")

    logger.info(f"Loading market definitions from: {market_definitions_path}")
    with open(market_definitions_path, 'r') as f:
        market_definitions = json.load(f)

    # Prepare the filter criteria based on the elite strategy
    filter_rules, needed_prediction_cols = prepare_elite_strategy_for_filter(elite_strategy_config, market_definitions)
    
    if not filter_rules:
        logger.error("No valid filter rules derived from elite strategy. Aborting pre-filtering.")
        return

    # Columns to always keep + the dynamically identified prediction/target columns
    base_cols_to_keep = [date_col, match_id_col]
    # Also need actual target columns from the original OOF file for other models,
    # so it's safer to select relevant prediction columns and base columns,
    # and then also keep ALL original target columns (like H_actual, A_actual etc.)
    # For simplicity now, let's just keep what's in filter_rules and base_cols.
    # A more robust way would be to list all possible target columns from market_definitions.
    
    all_target_cols_from_defs = list(set(md['target_col'] for md in market_definitions.values() if 'target_col' in md))

    columns_to_select_initially = list(set(base_cols_to_keep + needed_prediction_cols + all_target_cols_from_defs))
    
    # Ensure all selected columns actually exist in df_full
    columns_to_select_initially = [col for col in columns_to_select_initially if col in df_full.columns]
    if not columns_to_select_initially:
        logger.error("No columns to select after matching with OOF. Check elite strategy & market defs.")
        return
        
    logger.info(f"Initially selecting {len(columns_to_select_initially)} columns for potential filtering.")
    df_subset = df_full[columns_to_select_initially].copy() # Work on a copy with fewer columns

    # Build the filter condition
    # A row is kept if ANY of the elite strategy conditions are met for that row
    # (i.e., if that match has at least one potential elite leg)
    combined_filter_condition = pd.Series(False, index=df_subset.index)

    for rule in filter_rules:
        prob_col = rule['prob_col']
        threshold = rule['threshold']
        if prob_col in df_subset.columns:
            condition = (df_subset[prob_col] >= threshold) & pd.notna(df_subset[prob_col])
            combined_filter_condition = combined_filter_condition | condition
            logger.info(f"Applied filter for {prob_col} >= {threshold}. Matches found: {condition.sum()}")
        else:
            logger.warning(f"Probability column '{prob_col}' for pre-filtering not found in OOF subset. Skipping this rule for filter.")

    df_prefiltered = df_subset[combined_filter_condition].copy()
    filtered_rows = len(df_prefiltered)
    logger.info(f"Pre-filtered OOF shape: {df_prefiltered.shape}. Kept {filtered_rows} rows from {original_rows}.")

    if filtered_rows == 0:
        logger.warning("Pre-filtering resulted in an empty DataFrame. No matches met any elite criteria.")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving pre-filtered OOF to: {output_path}")
        df_prefiltered.to_parquet(output_path, index=False)
        logger.info("Pre-filtered OOF saved successfully.")

    logger.info("--- OOF Pre-filtering Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-filter OOF data based on Elite Leg Strategy.")
    parser.add_argument("--full_oof_path", required=True, help="Path to the full combined_oof_ALL_pipelines.parquet file.")
    parser.add_argument("--elite_strategy_json_path", required=True, help="Path to the JSON file defining the ELITE_LEG_STRATEGY_CONFIG.")
    parser.add_argument("--markets_def_path", required=True, help="Path to parlay_market_definitions.json.")
    parser.add_argument("--output_prefiltered_oof_path", required=True, help="Path to save the smaller, pre-filtered OOF parquet file.")
    
    args = parser.parse_args()

    try:
        with open(args.elite_strategy_json_path, 'r') as f:
            elite_config = json.load(f)
        assert isinstance(elite_config, list), "Elite strategy JSON must be a list."
    except Exception as e:
        logger.error(f"Failed to load elite strategy JSON: {e}")
        sys.exit(1)

    prefilter_oof(
        full_oof_path_str=args.full_oof_path,
        elite_strategy_config=elite_config,
        market_definitions_path_str=args.markets_def_path,
        output_prefiltered_oof_path_str=args.output_prefiltered_oof_path
    )