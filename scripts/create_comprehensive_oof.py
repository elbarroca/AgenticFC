# create_comprehensive_oof.py
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import time

# --- Configuration ---
# Assuming this script is in the root of your AGENTICFC888 project directory
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up one level to reach project root
PREDICTIONS_DIR = BASE_DIR / 'models' / 'data' / 'outputs' / 'predictions'
CONFIG_DIR = Path('/Users/barroca888/Downloads/Agenticfc/AgenticFC888/scripts/models/config')
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Default Configuration Files ---
DEFAULT_MARKET_DEFS_FILE = 'parlay_market_definitions.json'
DEFAULT_COMPREHENSIVE_OOF_OUTPUT_FILE = 'combined_oof_ALL_pipelines.parquet'

# --- Global Column Names (MUST MATCH YOUR DATA) ---
MATCH_ID_COL = 'MatchID'
DATE_COL = 'Date'
FTHG_COL = 'FTHG' # Full Time Home Goals
FTAG_COL = 'FTAG' # Full Time Away Goals
FTR_COL = 'FTR'   # Full Time Result (H, D, A)

# --- Model Prefixes for L0 models (used for renaming) ---
# This list should contain the base prefixes of your L0 models as they appear
# in the *original* individual OOF files BEFORE any pipeline labels are added.
# E.g., if an OOF file has "poisson_prob_H", then "poisson" is a prefix.
LEVEL0_MODEL_PREFIXES = ["poisson", "random_forest", "gradient_boosting", "monte_carlo"]


def load_and_rename_individual_oof(
    oof_file_path: Path,
    pipeline_label: str, # e.g., "V1_NonPCA_NoOdds", "V2_PCA_WithOdds_L0"
    match_id_col: str,
    date_col: str,
    parlay_market_definitions: dict, # Needed to identify prob_suffixes
    level0_model_prefixes: list # To identify L0 model columns
    ) -> pd.DataFrame | None:
    """
    Loads an individual OOF file (CSV or Parquet).
    Selects MatchID, Date, FTHG, FTAG, FTR.
    Renames L0 prediction columns to be globally unique using the pipeline_label.
    It assumes stacker predictions (if any) in this file are ALREADY uniquely named
    or this function needs to be adapted if stackers are also in these "L0 OOF" files.
    """
    print(f"  Processing: {oof_file_path.name} with label: {pipeline_label}")
    df = None
    try:
        if str(oof_file_path).lower().endswith(".parquet"):
            df = pd.read_parquet(oof_file_path)
        elif str(oof_file_path).lower().endswith(".csv"):
            df = pd.read_csv(oof_file_path)
            if match_id_col not in df.columns: # Handle CSVs where MatchID might be an index
                if 'Unnamed: 0' in df.columns and df['Unnamed: 0'].nunique() == len(df):
                    df.rename(columns={'Unnamed: 0': match_id_col}, inplace=True)
                elif 'index' in df.columns and df['index'].nunique() == len(df):
                    df.rename(columns={'index': match_id_col}, inplace=True)
        else:
            print(f"    Unsupported file type for {oof_file_path.name}")
            return None
    except FileNotFoundError:
        print(f"    File NOT FOUND: {oof_file_path}")
        return None
    except Exception as e:
        print(f"    Error loading {oof_file_path.name}: {e}")
        return None

    if match_id_col not in df.columns:
        print(f"    CRITICAL: '{match_id_col}' not found in {oof_file_path.name}. Skipping.")
        return None

    # Columns to keep/rename
    cols_to_keep_as_is = {match_id_col, date_col, FTHG_COL, FTAG_COL, FTR_COL}
    renamed_cols_dict = {} # For L0 prediction columns

    # Identify L0 prediction columns and prepare for renaming
    # These are columns like "poisson_prob_H", "random_forest_expected_AG" in the *individual* file
    for col in df.columns:
        if col in cols_to_keep_as_is:
            continue
        
        is_l0_pred_col = False
        for l0_prefix in level0_model_prefixes:
            if col.startswith(f"{l0_prefix}_"): # e.g., "poisson_"
                # Extract original suffix (e.g., "prob_H", "expected_AG")
                original_suffix = col[len(l0_prefix)+1:]
                # New unique name: modelprefix_PIPELINELABEL_L0_originalsuffix
                # The `pipeline_label` itself should contain the V1/V2, PCA/NonPCA, Odds/NoOdds info
                new_col_name = f"{l0_prefix}_{pipeline_label}_L0_{original_suffix}"
                renamed_cols_dict[col] = new_col_name
                is_l0_pred_col = True
                break # Found L0 prefix for this column
        
        # Handle Stacker L1 predictions if they are in these files and need renaming
        # This part assumes stacker columns are already uniquely named or follow a pattern
        # that needs specific handling if they are mixed with L0 OOFs.
        # For now, if a column is not a common key col or a recognized L0 pred col,
        # we might assume it's a stacker col that's already uniquely named by a previous process
        # OR it's an L0 col whose prefix wasn't in LEVEL0_MODEL_PREFIXES.
        if not is_l0_pred_col and col not in cols_to_keep_as_is:
            # If your stacker output columns are already globally unique (e.g. stacker_V1_NonPCA_NoOdds_L1_prob_H)
            # and are present in these input files, they will be carried over.
            # If they are NOT uniquely named yet (e.g. just "stacker_prob_H"), they need renaming too.
            # For simplicity, this example assumes stacker outputs are handled separately or are already unique.
            # print(f"    Column '{col}' not a common key or L0 pred, keeping as is (could be stacker or other).")
            pass


    # Apply renaming
    df_processed = df.rename(columns=renamed_cols_dict)

    # Select only necessary columns: MatchID, Date, Results, and all (renamed) prediction columns
    final_columns_to_select = list(cols_to_keep_as_is.intersection(df_processed.columns)) + \
                              list(renamed_cols_dict.values())
    
    # Add any other columns that were not common keys and not L0 preds (e.g. existing stacker preds)
    other_cols = [col for col in df_processed.columns if col not in final_columns_to_select and col != match_id_col]
    final_columns_to_select.extend(other_cols)
    final_columns_to_select = list(dict.fromkeys(final_columns_to_select)) # Ensure unique and preserve order

    # Filter out columns that might not exist after renaming if some were dropped
    final_columns_to_select = [col for col in final_columns_to_select if col in df_processed.columns]


    print(f"    Processed {oof_file_path.name}. Shape: {df_processed[final_columns_to_select].shape}. Kept/Renamed {len(final_columns_to_select)} cols.")
    return df_processed[final_columns_to_select]

def create_all_binary_targets(df: pd.DataFrame, parlay_market_definitions: dict,
                               match_id_col:str, date_col:str, 
                               fthg_col:str, ftag_col:str, ftr_col:str) -> pd.DataFrame:
    print("--- Creating/Ensuring All Binary Target Columns on Merged DataFrame ---")
    df_out = df.copy() 
    
    # --- Pass 1: Basic Atomic Targets (H/D/A, O/U from scores, BTTS from scores) ---
    print("  Pass 1: Creating basic atomic targets (1X2, O/U, BTTS from raw results)...")
    basic_targets_dict = {}
    if ftr_col in df_out.columns:
        if 'target_H' not in df_out.columns: basic_targets_dict['target_H'] = (df_out[ftr_col] == 'H').astype(int)
        if 'target_D' not in df_out.columns: basic_targets_dict['target_D'] = (df_out[ftr_col] == 'D').astype(int)
        if 'target_A' not in df_out.columns: basic_targets_dict['target_A'] = (df_out[ftr_col] == 'A').astype(int)
    else: print(f"    Warning: FTR column '{ftr_col}' not found for 1X2 targets.")

    if fthg_col in df_out.columns and ftag_col in df_out.columns:
        temp_fthg = pd.to_numeric(df_out[fthg_col], errors='coerce')
        temp_ftag = pd.to_numeric(df_out[ftag_col], errors='coerce')
        valid_scores_mask = temp_fthg.notna() & temp_ftag.notna()
        if valid_scores_mask.any():
            total_goals_valid = temp_fthg[valid_scores_mask] + temp_ftag[valid_scores_mask]
            for gt_val_float in [0.5, 1.5, 2.5, 3.5, 4.5]:
                gt_val_str = str(gt_val_float).replace('.', '')
                for ou_prefix in ['O', 'U']:
                    target_name = f'target_{ou_prefix}{gt_val_str}'
                    if target_name not in df_out.columns: 
                        series_ou = pd.Series(np.nan, index=df_out.index); series_ou.loc[valid_scores_mask] = ((total_goals_valid > gt_val_float) if ou_prefix == 'O' else (total_goals_valid < gt_val_float)).astype(int); basic_targets_dict[target_name] = series_ou
            if 'target_BTTS_Y' not in df_out.columns:
                series_btts_y = pd.Series(np.nan, index=df_out.index); series_btts_y.loc[valid_scores_mask] = ((temp_fthg[valid_scores_mask] > 0) & (temp_ftag[valid_scores_mask] > 0)).astype(int); basic_targets_dict['target_BTTS_Y'] = series_btts_y
            if 'target_BTTS_N' not in df_out.columns:
                btts_y_s = basic_targets_dict.get('target_BTTS_Y', df_out.get('target_BTTS_Y'))
                if btts_y_s is not None: basic_targets_dict['target_BTTS_N'] = 1 - btts_y_s
    else: print(f"    Warning: FTHG/FTAG columns ('{fthg_col}', '{ftag_col}') not found for O/U, BTTS targets.")
    
    if basic_targets_dict:
        df_out = df_out.assign(**basic_targets_dict)
    print(f"  Pass 1 complete. DF shape: {df_out.shape}. Columns added: {list(basic_targets_dict.keys())}")
    print(f"    DEBUG Pass 1: 'target_H' in df_out: {'target_H' in df_out.columns}, Sum: {df_out['target_H'].sum() if 'target_H' in df_out.columns else 'N/A'}")
    print(f"    DEBUG Pass 1: 'target_D' in df_out: {'target_D' in df_out.columns}, Sum: {df_out['target_D'].sum() if 'target_D' in df_out.columns else 'N/A'}")
    print(f"    DEBUG Pass 1: 'target_A' in df_out: {'target_A' in df_out.columns}, Sum: {df_out['target_A'].sum() if 'target_A' in df_out.columns else 'N/A'}")

    # --- Pass 2: Derived Single Targets (like DC: 1X, X2, 12) ---
    print("  Pass 2: Creating derived single targets (e.g., Double Chance)...")
    derived_single_targets_dict = {}
    # Expected keys for DC markets from your generate_market_definitions.py
    dc_market_keys = ["HomeOrDraw", "DrawOrAway", "HomeOrAway"] 

    for mkt_label, m_info in parlay_market_definitions.items():
        target_col = m_info['target_col']
        base_targets = m_info.get('base_targets', [])
        operation = m_info.get('op')
        
        if target_col in df_out.columns:
            if mkt_label in dc_market_keys: # If it's a DC market that already exists
                print(f"    DEBUG Pass 2: DC Target '{target_col}' for market '{mkt_label}' already exists. Skipping its creation.")
            continue
        
        if mkt_label in dc_market_keys: # Specific debug for DC markets
            print(f"\n    DEBUG Pass 2: Processing Potential DC Market Key from JSON: '{mkt_label}'")
            print(f"      Target Col: {target_col}, Operation: {operation}, Base Targets: {base_targets}")
            print(f"      'target_H' in df_out? {'target_H' in df_out.columns}")
            print(f"      'target_D' in df_out? {'target_D' in df_out.columns}")
            print(f"      'target_A' in df_out? {'target_A' in df_out.columns}")

        if operation == 'or' and len(base_targets) == 2:
            print(f"    DEBUG Pass 2: Attempting to create '{target_col}' (market '{mkt_label}') as 'or' op.")
            base_targets_exist = all(bt in df_out.columns for bt in base_targets)
            if not base_targets_exist:
                print(f"    WARNING: For DC target '{target_col}' (market '{mkt_label}'), missing base: {[bt for bt in base_targets if bt not in df_out.columns]}. Skipping.")
                continue
            
            print(f"    DEBUG Pass 2: Base targets for '{target_col}' ({base_targets}) FOUND in df_out.")
            try:
                # Check for all NaNs in base targets
                is_base1_all_nan = df_out[base_targets[0]].isnull().all()
                is_base2_all_nan = df_out[base_targets[1]].isnull().all()
                if is_base1_all_nan or is_base2_all_nan:
                    print(f"    WARNING: For DC target '{target_col}', base1 all NaN: {is_base1_all_nan}, base2 all NaN: {is_base2_all_nan}. Result will be NaN.")
                    derived_single_targets_dict[target_col] = pd.Series(np.nan, index=df_out.index)
                    continue

                series_list = [df_out[bt].astype(bool) for bt in base_targets]
                derived_single_targets_dict[target_col] = (series_list[0] | series_list[1]).astype(int)
                print(f"    SUCCESS Pass 2: Prepared '{target_col}' for dict. Sum: {derived_single_targets_dict[target_col].sum()}")
            except Exception as e: 
                print(f"    Error creating DC target '{target_col}': {e}")
                derived_single_targets_dict[target_col] = pd.Series(np.nan, index=df_out.index)
    
    if derived_single_targets_dict:
        print(f"    DEBUG Pass 2: Assigning derived_single_targets_dict with keys: {list(derived_single_targets_dict.keys())}")
        df_out = df_out.assign(**derived_single_targets_dict)
    else:
        print("    DEBUG Pass 2: derived_single_targets_dict is empty. No DC targets were prepared.")
        
    print(f"  Pass 2 complete. DF shape: {df_out.shape}. Columns added in Pass 2: {list(derived_single_targets_dict.keys())}")
    print(f"    DEBUG Pass 2: 'target_1X' in df_out: {'target_1X' in df_out.columns}, Sum: {df_out['target_1X'].sum() if 'target_1X' in df_out.columns else 'N/A'}")
    print(f"    DEBUG Pass 2: 'target_X2' in df_out: {'target_X2' in df_out.columns}, Sum: {df_out['target_X2'].sum() if 'target_X2' in df_out.columns else 'N/A'}")
    print(f"    DEBUG Pass 2: 'target_12' in df_out: {'target_12' in df_out.columns}, Sum: {df_out['target_12'].sum() if 'target_12' in df_out.columns else 'N/A'}")

    # --- Pass 3: Dual Outcome Targets (e.g., 1X_and_O15) ---
    print("  Pass 3: Creating dual outcome targets...")
    dual_targets_dict = {}
    for mkt_label, m_info in parlay_market_definitions.items():
        target_col, base_targets, operation = m_info['target_col'], m_info.get('base_targets', []), m_info.get('op')
        if target_col in df_out.columns: continue 
        if operation == 'and' and len(base_targets) == 2:
            if mkt_label == "1XO15": # Example specific debug
                print(f"    DEBUG Pass 3: Attempting '{target_col}'. Base1 '{base_targets[0]}' in df_out: {base_targets[0] in df_out.columns}. Base2 '{base_targets[1]}' in df_out: {base_targets[1] in df_out.columns}")
            if not all(bt in df_out.columns for bt in base_targets):
                print(f"    WARNING: For dual target '{target_col}' (market key: '{mkt_label}'), missing base: {[bt for bt in base_targets if bt not in df_out.columns]}. Skipping.")
                continue
            try:
                if df_out[base_targets[0]].isnull().all() or df_out[base_targets[1]].isnull().all():
                    dual_targets_dict[target_col] = pd.Series(np.nan, index=df_out.index)
                    continue
                series_list = [df_out[bt].astype(bool) for bt in base_targets]
                dual_targets_dict[target_col] = (series_list[0] & series_list[1]).astype(int)
            except Exception as e: 
                print(f"    Error creating dual target '{target_col}': {e}")
                dual_targets_dict[target_col] = pd.Series(np.nan, index=df_out.index)
    if dual_targets_dict:
        df_out = df_out.assign(**dual_targets_dict)
    print(f"  Pass 3 complete. DF shape: {df_out.shape}. Columns added: {list(dual_targets_dict.keys())}")
    print("--- Binary target creation/verification complete. ---")
    return df_out.copy()


def main(args):
    print("--- Starting: Create Comprehensive OOF Script ---")

    # --- Define Input OOF File Configurations ---
    # Each entry: (relative_path_to_oof_file_from_PREDICTIONS_DIR, unique_pipeline_label)
    # The `unique_pipeline_label` is what will be used to form model_identifier prefixes.
    # It MUST match what your strategy guide generation process will expect.
    oof_file_configs = [
        (args.oof_v1_no_pca_with_odds, "V1_NonPCA_WithOdds"),
        (args.oof_v1_no_pca_without_odds, "V1_NonPCA_NoOdds"),
        (args.oof_v2_pca_with_odds, "V2_PCA_WithOdds"),
        (args.oof_v2_pca_without_odds, "V2_PCA_NoOdds"),
        # Add entries for your stacker OOF files if they are separate
        # Example: (args.oof_stacker_v1_no_odds, "Stacker_V1_NonPCA_NoOdds_L1"),
        #          (args.oof_stacker_v2_pca_odds, "Stacker_V2_PCA_WithOdds_L1"),
    ]
    
    # Load PARLAY_MARKET_DEFINITIONS
    market_definitions_path = CONFIG_DIR / args.market_definitions_file
    try:
        with open(market_definitions_path, 'r') as f:
            parlay_market_definitions = json.load(f)
        print(f"Loaded PARLAY_MARKET_DEFINITIONS from {market_definitions_path}")
    except FileNotFoundError:
        print(f"CRITICAL: Market definitions file not found: {market_definitions_path}")
        return
    except Exception as e:
        print(f"CRITICAL: Error loading market definitions: {e}")
        return

    all_prepared_dfs = []
    print("\n--- Loading and Pre-processing Individual OOF Files ---")
    for file_name, pipeline_label in oof_file_configs:
        if not file_name: # Skip if path argument was not provided
            print(f"Skipping pipeline with label '{pipeline_label}' as no file path was given.")
            continue
        file_path = PREDICTIONS_DIR / file_name
        prepared_df = load_and_rename_individual_oof(
            file_path, 
            pipeline_label, 
            args.match_id_col, 
            args.date_col,
            parlay_market_definitions,
            LEVEL0_MODEL_PREFIXES
        )
        if prepared_df is not None:
            all_prepared_dfs.append(prepared_df)

    if not all_prepared_dfs:
        print("CRITICAL: No OOF files were successfully loaded. Cannot create combined file.")
        return

    # --- Merge DataFrames ---
    print("\n--- Merging All Processed DataFrames ---")
    if len(all_prepared_dfs) == 1:
        merged_df = all_prepared_dfs[0]
    else:
        merged_df = all_prepared_dfs[0].drop_duplicates(subset=[args.match_id_col])
        for i in range(1, len(all_prepared_dfs)):
            next_df = all_prepared_dfs[i].drop_duplicates(subset=[args.match_id_col])
            # Identify columns to merge from next_df: MatchID and any columns NOT in merged_df
            cols_to_bring_from_next = [args.match_id_col] + \
                                      [col for col in next_df.columns if col not in merged_df.columns or col == args.match_id_col]
            # Ensure no duplicate columns other than merge key
            cols_to_bring_from_next = list(dict.fromkeys(cols_to_bring_from_next))

            merged_df = pd.merge(merged_df, next_df[cols_to_bring_from_next], on=args.match_id_col, how='outer')
    
    print(f"  Merged DataFrame shape before final duplicate check: {merged_df.shape}")
    merged_df.drop_duplicates(subset=[args.match_id_col], keep='first', inplace=True)
    print(f"  Shape after dropping potential duplicates by MatchID: {merged_df.shape}")

    # --- Create All Binary Targets on the fully merged DataFrame ---
    merged_df_with_targets = create_all_binary_targets(
        merged_df, parlay_market_definitions,
        args.match_id_col, args.date_col,
        FTHG_COL, FTAG_COL, FTR_COL
    )

    # --- Save Combined OOF File ---
    combined_oof_output_path = PREDICTIONS_DIR / args.combined_oof_output_file
    combined_oof_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df_with_targets.to_parquet(combined_oof_output_path, index=False)
    print(f"\nSuccessfully saved COMPREHENSIVE OOF data to: {combined_oof_output_path}")
    print("This file can now be used as input for 'generate_strategy_inputs.py' and 'parlay_backtester.py'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a comprehensive OOF DataFrame by merging individual OOF files.")
    
    # --- Arguments for input OOF files ---
    # You need to add an argument for EACH individual OOF file you want to merge.
    parser.add_argument("--oof_v1_no_pca_with_odds", type=str, default="level0_oof_predictions_with_odds.csv",
                        help="Filename of V1 NonPCA WithOdds OOF (in PREDICTIONS_DIR)")
    parser.add_argument("--oof_v1_no_pca_without_odds", type=str, default="level0_oof_predictions_without_odds.csv",
                        help="Filename of V1 NonPCA NoOdds OOF (in PREDICTIONS_DIR)")
    parser.add_argument("--oof_v2_pca_with_odds", type=str, default="level0_oof_predictions_pca_with_odds.parquet",
                        help="Filename of V2 PCA WithOdds OOF (in PREDICTIONS_DIR)")
    parser.add_argument("--oof_v2_pca_without_odds", type=str, default="level0_oof_predictions_pca_without_odds.parquet",
                        help="Filename of V2 PCA NoOdds OOF (in PREDICTIONS_DIR)")
    # Add more for stacker OOFs if they are separate files:
    # parser.add_argument("--oof_stacker_v1_no_odds", type=str, default="stacker_v1_noodds_oof.parquet")
    # parser.add_argument("--oof_stacker_v2_pca_odds", type=str, default="stacker_v2_pcaodds_oof.parquet")


    # Config and Output files
    parser.add_argument("--market_definitions_file", type=str, default=DEFAULT_MARKET_DEFS_FILE, 
                        help="Filename for PARLAY_MARKET_DEFINITIONS JSON (in CONFIG_DIR)")
    parser.add_argument("--combined_oof_output_file", type=str, default=DEFAULT_COMPREHENSIVE_OOF_OUTPUT_FILE, 
                        help="Filename for the final merged OOF Parquet (in PREDICTIONS_DIR)")

    # Column names
    parser.add_argument("--match_id_col", type=str, default=MATCH_ID_COL)
    parser.add_argument("--date_col", type=str, default=DATE_COL)

    args = parser.parse_args()
    main(args)

    print("\n--- Script to create comprehensive OOF finished. ---")