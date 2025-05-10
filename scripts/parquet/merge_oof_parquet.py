# merge_oof_parquets.py

import pandas as pd
from pathlib import Path
import argparse

# --- Configuration (Define these based on your file structure) ---
# Assuming this script is in the root of your AGENTICFC888 project directory
BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = BASE_DIR / 'models' / 'data' / 'outputs' / 'predictions'

# Default input file names (relative to PREDICTIONS_DIR)
DEFAULT_OOF_WITH_ODDS_FILE = 'level0_oof_predictions_pca_with_odds.parquet'
DEFAULT_OOF_WITHOUT_ODDS_FILE = 'level0_oof_predictions_pca_without_odds.parquet'

# Default output file name (relative to PREDICTIONS_DIR)
DEFAULT_OUTPUT_COMBINED_FILE = 'level0_oof_predictions_pca_combined_from_script.parquet' # Different name to avoid overwriting notebook output initially

# Columns that are expected to be common identifiers or targets, and should NOT be renamed with suffixes
# Ensure these match your actual column names
# MATCH_ID_COL and DATE_COL should be defined globally in your notebook Cell 1
# and used consistently. For the script, we'll define them here or take as args.
DEFAULT_MATCH_ID_COL = 'MatchID'
DEFAULT_DATE_COL = 'Date'
# Add other common columns that should not be suffixed if they exist in both
# (e.g., FTHG, FTAG, FTR, and all 'target_*' columns)
# It's often safer to let the merge handle these and then verify.
# For simplicity, we'll primarily focus on MatchID for merging.

def get_common_and_prediction_columns(df, match_id_col, date_col):
    """Identifies common ID/target columns and prediction columns."""
    common_cols = {match_id_col, date_col}
    # Add known target column patterns
    for col in df.columns:
        if col.startswith('target_') or col in ['FTHG', 'FTAG', 'FTR']:
            common_cols.add(col)
    
    prediction_cols = [col for col in df.columns if col not in common_cols]
    return list(common_cols.intersection(df.columns)), prediction_cols


def merge_parquets(path_with_odds: Path, 
                   path_without_odds: Path, 
                   output_path: Path,
                   match_id_col: str,
                   date_col: str):
    """
    Merges two OOF Parquet files, renaming prediction columns to avoid clashes.
    """
    print(f"--- Merging Parquet Files ---")
    print(f"File 1 (With Odds): {path_with_odds}")
    print(f"File 2 (Without Odds): {path_without_odds}")

    try:
        df_with_odds = pd.read_parquet(path_with_odds)
        print(f"  Loaded df_with_odds: {df_with_odds.shape}")
        df_without_odds = pd.read_parquet(path_without_odds)
        print(f"  Loaded df_without_odds: {df_without_odds.shape}")
    except FileNotFoundError as e:
        print(f"CRITICAL: File not found: {e}. Cannot proceed with merge.")
        return
    except Exception as e:
        print(f"CRITICAL: Error loading Parquet files: {e}")
        return

    # Identify common columns (MatchID, Date, targets) and prediction columns for each df
    common_cols_wo, pred_cols_wo = get_common_and_prediction_columns(df_with_odds, match_id_col, date_col)
    common_cols_no, pred_cols_no = get_common_and_prediction_columns(df_without_odds, match_id_col, date_col)

    # Ensure MatchID is present for merging
    if match_id_col not in common_cols_wo or match_id_col not in common_cols_no:
        print(f"CRITICAL: Match ID column '{match_id_col}' not found in one or both DataFrames. Cannot merge.")
        return

    # Prepare for merging: Set MatchID as index for easier column management if needed,
    # but pandas merge on column is fine.

    # Rename prediction columns to make them unique before merging
    # Suffixes should reflect the source pipeline labels used in your notebook's Cell 3 merging
    # Example: If your notebook used "V2_PCA_WithOdds" and "V2_PCA_NoOdds" as labels
    suffix_with_odds = "_V2_PCA_WithOdds" # Or derive from filename / a label column if it exists
    suffix_without_odds = "_V2_PCA_NoOdds"

    rename_map_wo = {col: f"{col.split('_expected_')[0] if '_expected_' in col else col.split('_prob_')[0]}{suffix_with_odds}_{col.split('_', 1)[1] if '_' in col else col}" 
                     for col in pred_cols_wo}
    # More robust renaming: modelprefix_pipelinelabel_L0_suffix
    # Example: poisson_V2_PCA_WithOdds_L0_expected_HG
    # This requires knowing the model prefixes (e.g., "poisson", "random_forest")
    
    # Simpler renaming for this script: just add a suffix indicating source
    # This assumes prediction columns are like "model_prob_H" or "model_expected_HG"
    def generate_rename_map(pred_cols, pipeline_suffix_label):
        new_map = {}
        for col in pred_cols:
            parts = col.split("_", 1) # Split on the first underscore
            model_prefix = parts[0]
            original_suffix = parts[1] if len(parts) > 1 else ""
            # Construct new name: modelprefix_PIPELINELABEL_originalsuffix
            # This structure is similar to what your notebook's Cell 3 merging logic created.
            new_map[col] = f"{model_prefix}{pipeline_suffix_label}_L0_{original_suffix}"
        return new_map

    # Define the pipeline labels as used in your notebook's `combined_oof_for_betting`
    # These should match the `pipeline_label` argument from `process_and_analyze_pipeline_version`
    # or the labels you used when constructing unique column names in Cell 3.
    # For these specific files, the labels would likely be based on "PCA_WithOdds" and "PCA_NoOdds"
    # and potentially a version like "V2".
    
    # Let's assume the model_identifiers in your strategy guide will look like:
    # e.g., "poisson_V2_PCA_WithOdds_L0" or "random_forest_V2_PCA_NoOdds_L0"
    # So, the suffixes to add to the original OOF columns need to match this pattern.
    
    # The OOF files `level0_oof_predictions_pca_with_odds.parquet` likely contain columns like:
    # `poisson_expected_HG`, `random_forest_prob_H`, etc.
    # We need to transform them into `poisson_V2_PCA_WithOdds_L0_expected_HG`, etc.
    
    # This requires knowing the "pipeline version" label (e.g., V2_PCA_WithOdds)
    # For this script, let's assume fixed labels.
    # You would need to ensure these match what your strategy guide expects.
    
    # If your strategy guide uses model_identifiers like "poisson_V2_PCA_WithOdds_L0",
    # then the columns in the merged OOF file need to be exactly that + "_prob_H" or "_expected_HG".
    # The input OOF files probably have columns like "poisson_prob_H".
    # So, the renaming should construct the full `model_identifier` string.

    # Let's assume the pipeline labels are:
    pipeline_label_with_odds = "_V2_PCA_WithOdds_L0" # Example, adjust to your actual full label
    pipeline_label_without_odds = "_V2_PCA_NoOdds_L0" # Example

    rename_map_df1 = {col: f"{col.split('_', 1)[0]}{pipeline_label_with_odds}_{col.split('_', 1)[1]}"
                      for col in pred_cols_wo if '_' in col} # Handles "model_suffix"
    rename_map_df1_no_underscore = {col: f"{col}{pipeline_label_with_odds}"
                                   for col in pred_cols_wo if '_' not in col} # Handles "model"
    rename_map_df1.update(rename_map_df1_no_underscore)


    rename_map_df2 = {col: f"{col.split('_', 1)[0]}{pipeline_label_without_odds}_{col.split('_', 1)[1]}"
                      for col in pred_cols_no if '_' in col}
    rename_map_df2_no_underscore = {col: f"{col}{pipeline_label_without_odds}"
                                   for col in pred_cols_no if '_' not in col}
    rename_map_df2.update(rename_map_df2_no_underscore)


    df_with_odds_renamed = df_with_odds.rename(columns=rename_map_df1)
    df_without_odds_renamed = df_without_odds.rename(columns=rename_map_df2)

    print(f"  Sample renamed columns from df_with_odds: {list(df_with_odds_renamed.columns[:5])}")
    print(f"  Sample renamed columns from df_without_odds: {list(df_without_odds_renamed.columns[:5])}")

    # Merge the DataFrames
    # Perform an outer merge to keep all matches.
    # Common columns (MatchID, Date, targets) should align.
    # If target columns differ slightly (e.g., one df has more than other), outer merge handles it.
    
    # Identify the truly common columns (MatchID, Date, and actual results like FTHG, FTAG, FTR, target_*)
    # These should NOT be renamed and should be used as the basis for merging.
    # The `get_common_and_prediction_columns` already identified these.
    # We need to ensure we are merging on MatchID and that other common columns are handled correctly.

    # Let's select only the MatchID and the renamed prediction columns from each,
    # then merge, and then add back common target columns from one of them (assuming they are consistent).

    df1_to_merge = df_with_odds_renamed[[match_id_col] + [rename_map_df1.get(c, c) for c in pred_cols_wo]]
    df2_to_merge = df_without_odds_renamed[[match_id_col] + [rename_map_df2.get(c, c) for c in pred_cols_no]]

    merged_df = pd.merge(df1_to_merge, df2_to_merge, on=match_id_col, how='outer')
    print(f"  Shape after merging prediction columns: {merged_df.shape}")

    # Add back the common/target columns from one of the original dataframes (e.g., df_with_odds)
    # Assuming common_cols_wo contains MatchID, Date, FTHG, FTAG, FTR, target_* etc.
    # We need to be careful not to duplicate MatchID.
    common_data_to_add = df_with_odds[common_cols_wo].drop_duplicates(subset=[match_id_col])
    
    # Merge common data back
    final_merged_df = pd.merge(merged_df, common_data_to_add, on=match_id_col, how='left') # Use left to keep all rows from merged_df
    print(f"  Shape after adding back common/target columns: {final_merged_df.shape}")
    
    # Verification
    if final_merged_df[match_id_col].duplicated().any():
        print(f"WARNING: Duplicates found in '{match_id_col}' after merge. Review merging logic.")
        final_merged_df.drop_duplicates(subset=[match_id_col], keep='first', inplace=True)
        print(f"  Shape after dropping duplicates: {final_merged_df.shape}")

    # Save the merged DataFrame
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        final_merged_df.to_parquet(output_path, index=False)
        print(f"Successfully merged and saved to: {output_path}")
    except Exception as e:
        print(f"CRITICAL: Error saving merged file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two OOF Parquet files.")
    parser.add_argument("--with_odds_path", type=str, 
                        default=str(PREDICTIONS_DIR / DEFAULT_OOF_WITH_ODDS_FILE),
                        help="Path to the OOF Parquet file with odds features.")
    parser.add_argument("--without_odds_path", type=str, 
                        default=str(PREDICTIONS_DIR / DEFAULT_OOF_WITHOUT_ODDS_FILE),
                        help="Path to the OOF Parquet file without odds features.")
    parser.add_argument("--output_path", type=str, 
                        default=str(PREDICTIONS_DIR / DEFAULT_OUTPUT_COMBINED_FILE),
                        help="Path to save the merged Parquet file.")
    parser.add_argument("--match_id_col", type=str, default=DEFAULT_MATCH_ID_COL,
                        help="Name of the Match ID column.")
    parser.add_argument("--date_col", type=str, default=DEFAULT_DATE_COL,
                        help="Name of the Date column.")
    
    args = parser.parse_args()

    merge_parquets(
        path_with_odds=Path(args.with_odds_path),
        path_without_odds=Path(args.without_odds_path),
        output_path=Path(args.output_path),
        match_id_col=args.match_id_col,
        date_col=args.date_col
    )

    print("\nMerge script execution complete.")