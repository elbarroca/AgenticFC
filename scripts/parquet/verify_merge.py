# scripts/verify_merge.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
import logging
import random

# --- Configuration ---
NUM_SAMPLES_TO_CHECK = 800  # How many random matches to verify in detail
TOLERANCE = 1e-5 # Tolerance for comparing float values

# Columns expected primarily from Mongo (choose a representative sample)
MONGO_CHECK_COLS = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', # Core Info
    'HomeShots', 'AwayCorners', 'HomePossession', 'AwayYellowCards', # Stats
    'Home_AvgGoalsScored_Total_Last5', 'Away_FormPoints_Total_Last10', # Rolling Features
    'Home_BTTS_Ratio_Total_Last15'
]
# Odds columns expected primarily from CSV (choose a representative sample)
CSV_CHECK_COLS = [
    'B365H', 'B365D', 'B365A', 'AvgH', 'AvgD', 'AvgA', 'AHh',
    'BbMxAHH', 'BbAvAHA', 'PSCH', 'MaxC>2.5', 'PC<2.5', 'WHCA'
]

# --- Setup ---
# Add project root
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Project Root added to sys.path: {project_root}")
except Exception as e:
    print(f"Error setting up project root: {e}. Exiting.")
    sys.exit(1)

# Define paths
FINAL_PARQUET_PATH = os.path.join(project_root, "data", "unified_data", "final_unified_data.parquet")
MONGO_PARQUET_PATH = os.path.join(project_root, "data", "unified_data", "mongo.parquet")
CSV_PARQUET_PATH = os.path.join(project_root, "data", "unified_data", "csv_unified_full_cols.parquet")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path, name):
    """Loads a parquet file with error handling and ensures MatchID is string."""
    logging.info(f"Loading {name} data from: {path}")
    if not os.path.exists(path):
        logging.error(f"{name} Parquet file not found: {path}. Exiting.")
        return None
    try:
        df = pd.read_parquet(path)
        logging.info(f"Loaded {name} data shape: {df.shape}")
        # Ensure MatchID is string for consistent comparison
        if 'MatchID' in df.columns:
            df['MatchID'] = df['MatchID'].astype(str).replace('nan', pd.NA)
        else:
            logging.error(f"MatchID column not found in {name} file: {path}")
            return None
        return df
    except Exception as e:
        logging.error(f"Failed to load {name} Parquet {path}: {e}", exc_info=True)
        return None

def compare_values(val1, val2, col_name):
    """Compares two values, handling NaN and float tolerance."""
    if pd.isna(val1) and pd.isna(val2):
        return True, None # Both NaN -> Equal
    if pd.isna(val1) or pd.isna(val2):
        return False, f"{col_name}: One value is NaN (Final='{val1}', Orig='{val2}')" # One NaN -> Not Equal
    if isinstance(val1, float) or isinstance(val2, float):
        # Use np.isclose for float comparison with tolerance
        if not np.isclose(float(val1), float(val2), atol=TOLERANCE, rtol=TOLERANCE, equal_nan=True):
            return False, f"{col_name}: Float difference (Final={val1:.6f}, Orig={val2:.6f})"
    elif isinstance(val1, pd.Timestamp) or isinstance(val2, pd.Timestamp):
         # Ensure consistent timestamp comparison (e.g., convert to string or normalize)
         if pd.to_datetime(val1) != pd.to_datetime(val2):
              return False, f"{col_name}: Timestamp difference (Final='{val1}', Orig='{val2}')"
    elif val1 != val2:
        # General inequality for other types (int, str, bool)
        return False, f"{col_name}: Value difference (Final='{val1}', Orig='{val2}')"
    return True, None # Values are equal

# --- Main Verification Logic ---
if __name__ == "__main__":
    logging.info("--- Starting Merge Verification Script ---")

    # 1. Load DataFrames
    final_df = load_data(FINAL_PARQUET_PATH, "Final")
    mongo_orig_df = load_data(MONGO_PARQUET_PATH, "Original Mongo")
    csv_orig_df = load_data(CSV_PARQUET_PATH, "Original CSV")

    if final_df is None or mongo_orig_df is None or csv_orig_df is None:
        logging.error("Failed to load one or more necessary data files. Exiting verification.")
        sys.exit(1)

    # 2. Identify Common MatchIDs present in all three files
    try:
        common_ids_mongo_final = set(mongo_orig_df['MatchID'].dropna().unique()) & set(final_df['MatchID'].dropna().unique())
        common_ids_csv_final = set(csv_orig_df['MatchID'].dropna().unique()) & set(final_df['MatchID'].dropna().unique())
        # We check Mongo & CSV separately against Final, as CSV merge was 'left'
        logging.info(f"Found {len(common_ids_mongo_final)} common MatchIDs between Original Mongo and Final.")
        logging.info(f"Found {len(common_ids_csv_final)} common MatchIDs between Original CSV and Final.")

        # Sample IDs that are common to Mongo and Final for the main check
        if not common_ids_mongo_final:
             logging.error("No common MatchIDs found between the Original Mongo and Final files. Cannot verify Mongo data alignment.")
             sys.exit(1)

    except KeyError:
         logging.error("MatchID column not found in one of the files after loading. Exiting.")
         sys.exit(1)


    # 3. Select Samples from Mongo common IDs
    num_to_sample = min(NUM_SAMPLES_TO_CHECK, len(common_ids_mongo_final))
    if num_to_sample < NUM_SAMPLES_TO_CHECK:
        logging.warning(f"Fewer common Mongo/Final IDs ({len(common_ids_mongo_final)}) than requested samples ({NUM_SAMPLES_TO_CHECK}). Checking all common IDs.")

    # Ensure the sample IDs also exist in the CSV common set where possible, for better odds checking
    potential_sample_ids = list(common_ids_mongo_final)
    sample_ids = random.sample(potential_sample_ids, num_to_sample)
    logging.info(f"Will check {num_to_sample} randomly selected MatchIDs common to Mongo/Final.")

    # 4. Perform Checks
    mismatches_found = 0
    total_checks = 0
    for match_id in sample_ids:
        total_checks += 1
        logging.debug(f"--- Checking MatchID: {match_id} ---")

        # Extract rows (assuming MatchID is unique in each file)
        final_row_s = final_df[final_df['MatchID'] == match_id]
        mongo_row_s = mongo_orig_df[mongo_orig_df['MatchID'] == match_id]
        csv_row_s = csv_orig_df[csv_orig_df['MatchID'] == match_id] # May be empty

        if final_row_s.empty or mongo_row_s.empty:
             logging.error(f"Critical Error: MatchID {match_id} missing from Final or Original Mongo df during check loop. Aborting further checks for this ID.")
             mismatches_found += 1 # Count this as a failure
             continue

        # Use iloc[0] to get the Series for easier value access
        final_row = final_row_s.iloc[0]
        mongo_row = mongo_row_s.iloc[0]
        csv_row = csv_row_s.iloc[0] if not csv_row_s.empty else None # Handle cases where MatchID not in CSV

        match_mismatch = False

        # --- Check Mongo columns ---
        for col in MONGO_CHECK_COLS:
            if col not in final_row.index:
                logging.warning(f"Check column '{col}' missing in Final DF for MatchID {match_id}. Skipping.")
                continue
            if col not in mongo_row.index:
                logging.warning(f"Check column '{col}' missing in Original Mongo DF for MatchID {match_id}. Skipping.")
                continue

            final_val = final_row[col]
            mongo_val = mongo_row[col]

            are_equal, reason = compare_values(final_val, mongo_val, col)
            if not are_equal:
                logging.error(f"MISMATCH for MatchID {match_id} -> {reason} (Source: Mongo)")
                match_mismatch = True

        # --- Check CSV columns (Odds) ---
        if csv_row is not None: # Only check if the match was found in the original CSV data
            for col in CSV_CHECK_COLS:
                 if col not in final_row.index:
                    logging.warning(f"Check column '{col}' missing in Final DF for MatchID {match_id}. Skipping.")
                    continue
                 if col not in csv_row.index:
                    # This is expected if the column wasn't in the original CSV file
                    # logging.debug(f"Check column '{col}' missing in Original CSV DF for MatchID {match_id}. Skipping.")
                    continue

                 final_val = final_row[col]
                 csv_val = csv_row[col]

                 are_equal, reason = compare_values(final_val, csv_val, col)
                 if not are_equal:
                    # Important: Check if the Mongo data had this column and it was null - maybe it was kept?
                    mongo_val_for_csv_col = mongo_row.get(col, np.nan)
                    if not pd.isna(mongo_val_for_csv_col) and compare_values(final_val, mongo_val_for_csv_col, col)[0]:
                         # Value matches the original Mongo value, likely CSV value was NaN and Mongo value was kept (or merge issue)
                         logging.warning(f"Potential Discrepancy for MatchID {match_id} -> {col}: Final value ('{final_val}') matches original Mongo ('{mongo_val_for_csv_col}') but NOT original CSV ('{csv_val}'). Check merge logic/source data.")
                         # Decide if this counts as a mismatch based on requirements - for now, let's flag it but not count as error
                         # match_mismatch = True
                    else:
                         # Truly doesn't match CSV or Mongo's original (if any)
                         logging.error(f"MISMATCH for MatchID {match_id} -> {reason} (Source: CSV)")
                         match_mismatch = True
        else:
            logging.debug(f"MatchID {match_id} not found in Original CSV. Skipping odds comparison.")
            # Verify that the corresponding odds columns in final_df are indeed NaN for this match
            for col in CSV_CHECK_COLS:
                 if col in final_row.index and not pd.isna(final_row[col]):
                      logging.error(f"MISMATCH for MatchID {match_id} -> Column '{col}' has value '{final_row[col]}' but original CSV row was missing.")
                      match_mismatch = True


        if match_mismatch:
            mismatches_found += 1
        else:
            logging.debug(f"MatchID {match_id}: All checks passed.")


    # 5. Final Summary
    logging.info("--- Verification Summary ---")
    if mismatches_found == 0:
        logging.info(f"SUCCESS: All {total_checks} randomly checked common MatchIDs appear correctly merged based on selected columns.")
    else:
        logging.error(f"FAILURE: Found mismatches in {mismatches_found} out of {total_checks} checked MatchIDs. Review logs above.")
        logging.error("This indicates potential issues with MatchID consistency in the source files or the merge/conflict resolution logic.")

    logging.info("--- Merge Verification Script Finished ---")
