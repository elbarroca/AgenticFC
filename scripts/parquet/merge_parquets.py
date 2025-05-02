# scripts/merge_parquets.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import logging
import sys
import io
import re # Keep re for helper function if needed later

# --- Configuration Setup ---
# Add project root to sys.path (adjust if your script location differs)
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Project Root added to sys.path: {project_root}")

    # Define input and output paths
    # <<< CONFIRM these paths are correct >>>
    CSV_PARQUET_PATH = os.path.join(project_root, "data", "unified_data", "csv_unified_full_cols.parquet")
    MONGO_PARQUET_PATH = os.path.join(project_root, "data", "unified_data", "mongo.parquet")
    OUTPUT_DIR = os.path.join(project_root, "data", "unified_data")
    OUTPUT_FILENAME = "final_unified_data.parquet" # Name for the final merged output

except Exception as e:
    print(f"Error setting up project root or paths: {e}. Using defaults.")
    CSV_PARQUET_PATH = "data/unified_data/csv_unified.parquet"
    MONGO_PARQUET_PATH = "data/unified_data/mongo.parquet"
    OUTPUT_DIR = "data/unified_data"
    OUTPUT_FILENAME = "final_unified_data.parquet"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
FINAL_OUTPUT_PARQUET_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Final Cleaning (ADAPTED based on mongo.parquet info) ---

# Core columns expected primarily from MongoDB source
CORE_COLS_FINAL = [
    'MatchID', 'Date', 'Timestamp', 'Season', 'LeagueID', 'LeagueName', 'Country', 'Round',
    'HomeTeam', 'AwayTeam', 'HomeTeamWinner', 'AwayTeamWinner', 'HomeTeamID', 'AwayTeamID',
    'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
    'Referee', 'VenueName', 'VenueCity',
    'StatusLong', 'StatusShort', 'StatusElapsed',
    'HomeFormation', 'AwayFormation'
]

# Stats columns expected *only* from MongoDB source (based on mongo.parquet info)
# Exclude low-coverage stats like xG, maybe RedCards if coverage is too low? User decision.
# For now, include all stats found in mongo.parquet info with > ~50k coverage.
ALL_EXPECTED_STATS_COLS = sorted([
    'AwayBlockedShots', 'AwayCorners', 'AwayFouls', 'AwayOffsides', 'AwayPassAccuracy',
    'AwayPassesAccurate', 'AwayPossession', 'AwaySaves', 'AwayShots', 'AwayShotsInsideBox',
    'AwayShotsOffTarget', 'AwayShotsOutsideBox', 'AwayShotsTarget', 'AwayTotalPasses', 'AwayYellowCards', #'AwayRedCards', # Low coverage
    'HomeBlockedShots', 'HomeCorners', 'HomeFouls', 'HomeOffsides', 'HomePassAccuracy',
    'HomePassesAccurate', 'HomePossession', 'HomeSaves', 'HomeShots', 'HomeShotsInsideBox',
    'HomeShotsOffTarget', 'HomeShotsOutsideBox', 'HomeShotsTarget', 'HomeTotalPasses', 'HomeYellowCards', #'HomeRedCards', # Low coverage
    # 'AwayExpectedGoals', 'HomeExpectedGoals' # Very low coverage, exclude for now
])

# Odds columns expected primarily from the merged CSV data (better coverage)
# We use the 'valuable_csv_cols' list defined in merge_data (excluding MatchID)
# Note: This list needs to be consistent with the one in merge_data. For robustness,
# it might be better to define this list once globally, but for now, we repeat it conceptually.
ALL_EXPECTED_ODDS_COLS = sorted([
    'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA',
    'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA',
    'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5', 'B365AHH', 'B365AHA',
    'PAHH', 'PAHA', 'GBH', 'GBD', 'GBA', 'LBH', 'LBD', 'LBA', 'SBH',
    'SBD', 'SBA', 'SJH', 'SJD', 'SJA', 'BSH', 'BSD', 'BSA',
    'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'Max>2.5', 'Max<2.5',
    'Avg>2.5', 'Avg<2.5', 'AHh', 'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA',
    'B365CH', 'B365CD', 'B365CA', 'BWCH', 'BWCD', 'BWCA', 'IWCH', 'IWCD',
    'IWCA', 'PSCH', 'PSCD', 'PSCA', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD',
    'VCCA', 'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'B365CAHH',
    'B365CAHA', 'PCAHH', 'PCAHA',
    'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA', 'MaxC>2.5', 'MaxC<2.5',
    'AvgC>2.5', 'AvgC<2.5', 'AHCh', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA',
    'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU',
    'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbAH', 'BbAHh',
    'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA',
])

# Rolling Features Columns - We expect these from mongo.parquet
# Define dynamically later in final_clean_and_order based on prefix/suffix

# --- Helper Function ---
# Keep create_match_id in case we need to regenerate/verify IDs post-merge, although unlikely necessary
def create_match_id(row, date_col='Date', home_col='HomeTeam', away_col='AwayTeam'):
    """Creates a unique identifier string for a match. Uses Date."""
    try:
        if pd.isna(row[date_col]) or not isinstance(row[date_col], pd.Timestamp): return None
        date_str = row[date_col].strftime('%Y%m%d')
        home = re.sub(r'\W+', '', str(row[home_col])); away = re.sub(r'\W+', '', str(row[away_col]))
        if not home or not away or home == 'nan' or away == 'nan': return None
        return f"{date_str}_{home}_{away}"
    except Exception: return None


# --- Merging Function ---

def merge_data(csv_df: pd.DataFrame, mongo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges valuable, well-populated odds columns from the CSV data onto the
    MongoDB data (left merge), prioritizing MongoDB as the base for core/stats/rolling features.
    Selects CSV columns based on relevance (odds) and data coverage (>15k non-null).
    If odds columns conflict, prioritizes the CSV version due to better coverage.
    """
    if mongo_df.empty:
        logging.error("MongoDB DataFrame is empty. Cannot use as base for merge. Returning empty.")
        return pd.DataFrame()
    if csv_df.empty:
        logging.warning("CSV DataFrame is empty. No additional columns to merge. Returning MongoDB data.")
        return mongo_df

    logging.info(f"Starting merge. Base Mongo shape: {mongo_df.shape}, CSV shape: {csv_df.shape}")

    # --- Prepare for Merge ---
    if 'MatchID' not in csv_df.columns or 'MatchID' not in mongo_df.columns:
        logging.error("MatchID column missing in one or both DataFrames. Cannot merge.")
        return pd.DataFrame() # Exit if MatchID missing

    try:
        mongo_df['MatchID'] = mongo_df['MatchID'].astype(str)
        csv_df['MatchID'] = csv_df['MatchID'].astype(str)
    except Exception as e:
        logging.error(f"Error converting MatchID to string: {e}", exc_info=True)
        return pd.DataFrame()

    # --- Define Valuable Columns from CSV (Odds Focus, >15k non-null from previous analysis) ---
    # This list is based on the CSV info provided earlier and the >15k threshold
    valuable_csv_cols = [
        'MatchID',
        'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA',
        'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA',
        'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5', 'B365AHH', 'B365AHA',
        'PAHH', 'PAHA', 'GBH', 'GBD', 'GBA', 'LBH', 'LBD', 'LBA', 'SBH',
        'SBD', 'SBA', 'SJH', 'SJD', 'SJA', 'BSH', 'BSD', 'BSA',
        'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'Max>2.5', 'Max<2.5',
        'Avg>2.5', 'Avg<2.5', 'AHh', 'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA',
        'B365CH', 'B365CD', 'B365CA', 'BWCH', 'BWCD', 'BWCA', 'IWCH', 'IWCD',
        'IWCA', 'PSCH', 'PSCD', 'PSCA', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD',
        'VCCA', 'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'B365CAHH',
        'B365CAHA', 'PCAHH', 'PCAHA',
        'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA', 'MaxC>2.5', 'MaxC<2.5',
        'AvgC>2.5', 'AvgC<2.5', 'AHCh', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA',
        'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU',
        'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbAH', 'BbAHh',
        'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA',
    ]

    actual_valuable_csv_cols = [col for col in valuable_csv_cols if col in csv_df.columns]
    missing_cols = set(valuable_csv_cols) - set(actual_valuable_csv_cols)
    if 'MatchID' not in actual_valuable_csv_cols:
         logging.error("MatchID column is missing from CSV data after filtering. Cannot merge.")
         return mongo_df
    if missing_cols:
        logging.warning(f"Columns defined as valuable but not found in CSV data: {missing_cols}")
    if len(actual_valuable_csv_cols) <= 1:
        logging.warning("No valuable columns (beyond MatchID) found in CSV data to merge. Returning MongoDB data.")
        return mongo_df

    logging.info(f"Selected {len(actual_valuable_csv_cols) - 1} valuable columns (plus MatchID) from CSV for merge.")

    # --- Perform Left Merge ---
    try:
        # Use suffixes to detect collisions
        merged_df = pd.merge(
            mongo_df,
            csv_df[actual_valuable_csv_cols],
            on='MatchID',
            how='left',
            suffixes=('_mongo_orig', '_csv_valuable') # Identify source if collision
        )
        logging.info(f"Shape after initial left merge: {merged_df.shape}")

        # --- Resolve Conflicts: Prioritize CSV columns ---
        cols_to_drop = []
        cols_to_rename = {}
        for col_valuable in actual_valuable_csv_cols:
            if col_valuable == 'MatchID': continue # Skip join key

            suffixed_csv_col = f"{col_valuable}_csv_valuable"
            suffixed_mongo_col = f"{col_valuable}_mongo_orig"

            if suffixed_csv_col in merged_df.columns:
                # Collision occurred
                logging.debug(f"Collision detected for {col_valuable}. Prioritizing CSV version.")
                # Rename CSV column to the original name
                cols_to_rename[suffixed_csv_col] = col_valuable
                # Mark the original Mongo column (if it exists) for dropping
                if suffixed_mongo_col in merged_df.columns:
                    cols_to_drop.append(suffixed_mongo_col)
                elif col_valuable in mongo_df.columns: # Check original mongo_df if suffix wasn't applied
                     # This case is less likely with suffixes but check for safety
                     if col_valuable in merged_df.columns and col_valuable not in actual_valuable_csv_cols:
                          cols_to_drop.append(col_valuable)


        if cols_to_rename:
            merged_df.rename(columns=cols_to_rename, inplace=True)
            logging.info(f"Renamed {len(cols_to_rename)} columns from CSV to replace Mongo versions.")
        if cols_to_drop:
             # Ensure columns to drop actually exist before dropping
            cols_to_drop = [col for col in cols_to_drop if col in merged_df.columns]
            merged_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            logging.info(f"Dropped {len(cols_to_drop)} original Mongo columns due to collision resolution.")

        logging.info(f"Shape after conflict resolution: {merged_df.shape}")

    except pd.errors.MergeError as me:
         logging.error(f"Pandas MergeError during left merge: {me}. Check MatchID types and uniqueness.", exc_info=True)
         return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error during pandas left merge: {e}", exc_info=True)
        return pd.DataFrame()

    # --- Final Checks & Return ---
    initial_rows = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['MatchID'], keep='first') # Keep first (Mongo base)
    rows_dropped = initial_rows - len(merged_df)
    if rows_dropped > 0:
        logging.warning(f"Dropped {rows_dropped} duplicate MatchIDs after merge (unexpected for left merge, check base Mongo data).")

    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep='first')]

    logging.info(f"Merge completed. Final shape before cleaning: {merged_df.shape}")
    return merged_df


# --- Feature Calculation and Cleaning Functions ---

# Update final_clean_and_order for robust casting and ML optimization
def final_clean_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies final type casting, sorting, and column ordering based on the
    expected final schema. Optimizes types for memory and ML (float32, category).
    Handles potential type casting errors robustly.
    """
    logging.info(f"Starting final cleaning and ordering. Input shape: {df.shape}")
    original_cols = set(df.columns)

    # --- Type Casting (Optimized for ML) ---
    logging.debug("Applying type casting...")

    # Datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Nullable Boolean
    bool_cols = ['HomeTeamWinner', 'AwayTeamWinner']
    for col in bool_cols:
        if col in df.columns:
            try:
                # Attempt direct cast first, may work if already bool/numeric
                df[col] = df[col].astype('boolean')
            except TypeError:
                # Fallback for strings 'True'/'False' etc.
                logging.debug(f"Attempting flexible boolean conversion for {col}")
                df[col] = df[col].map({'True': True, 'False': False, 'true': True, 'false': False, 1: True, 0: False, 1.0: True, 0.0: False, True: True, False: False}).astype('boolean')
            except Exception as e:
                 logging.warning(f"Could not cast column '{col}' to boolean: {e}. Leaving as is.")
        else:
            logging.warning(f"Expected Boolean column '{col}' not found for casting.")

    # Nullable Integers (Robust Casting)
    # Identify potential integer columns (adjust list as needed)
    int_col_candidates = ['Season', 'LeagueID', 'HomeTeamID', 'AwayTeamID', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'StatusElapsed']
    # Add rolling count columns and other potential ints
    int_col_candidates.extend([c for c in df.columns if ('_Count_' in c or 'Bb1X2' in c or 'BbOU' in c or 'BbAH' in c) and c not in ['MatchID']])
    # Maybe card counts? (Check if these stats are reliable integers)
    # int_col_candidates.extend([c for c in ALL_EXPECTED_STATS_COLS if 'YellowCards' in c or 'RedCards' in c])

    processed_as_int = []
    for col in int_col_candidates:
        if col in df.columns:
            # Coerce to numeric first, handling errors by turning invalid entries to NaN
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            # Check if all *non-missing* values are effectively integers
            if numeric_series.isna().all() or numeric_series.dropna().apply(lambda x: x == np.round(x)).all():
                try:
                    # Attempt cast to Int64 (safe default nullable integer)
                    df[col] = numeric_series.astype('Int64')
                    processed_as_int.append(col)
                    # logging.debug(f"Successfully cast '{col}' to Int64.")
                except TypeError as te:
                    # This catch might be redundant with the check above, but keep for safety
                    logging.warning(f"Could not cast column '{col}' to Int64 despite checks ({te}). Casting to float32.")
                    df[col] = numeric_series.astype('float32') # Fallback to float32
                except Exception as e:
                    logging.error(f"Unexpected error casting column '{col}' to Int64: {e}. Casting to float32.")
                    df[col] = numeric_series.astype('float32')
            else:
                # Contains non-integer floats, cast to float32
                logging.warning(f"Column '{col}' contains non-integer values or could not be coerced. Casting to float32 instead of Int64.")
                df[col] = numeric_series.astype('float32')
        # else: # No need to warn for every potential candidate not present
             # logging.debug(f"Integer candidate column '{col}' not found for casting.")

    # Floats (Optimized to float32)
    # Identify remaining numeric columns (Stats, Odds, Rolling Features)
    potential_float_cols = []
    potential_float_cols.extend(ALL_EXPECTED_STATS_COLS)
    potential_float_cols.extend(ALL_EXPECTED_ODDS_COLS)
    potential_float_cols.extend([c for c in df.columns if ('_Avg' in c or '_Ratio' in c)])
    # Add any other numeric columns not yet processed
    potential_float_cols.extend([c for c in df.select_dtypes(include=np.number).columns if c not in processed_as_int and c not in bool_cols and c != 'Timestamp']) # Exclude already handled types

    float_cols = list(set(potential_float_cols) - set(processed_as_int)) # Ensure no overlap with successfully cast integers

    for col in float_cols:
         if col in df.columns:
            # Cast remaining numeric types to float32
            if pd.api.types.is_numeric_dtype(df[col]):
                 try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
                 except Exception as e:
                    logging.error(f"Could not cast column '{col}' to float32: {e}. Leaving as is.")
            # else: # Column might be object/string type but expected numeric - pd.to_numeric handles this
            #      df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
         # else: # Don't warn for every potential column not present
              # logging.debug(f"Expected Float column '{col}' not found for casting.")

    # Categorical Types (for low-cardinality strings)
    categorical_cols = [
        'LeagueName', 'Country', 'Round', # Round might be high cardinality depending on format
        'FTR', 'HTR', 'StatusShort', # StatusLong might be too verbose
        'HomeFormation', 'AwayFormation', # Can have many unique values, but maybe useful as category
        # 'Referee', # Often high cardinality
    ]
    for col in categorical_cols:
         if col in df.columns:
            # Check cardinality before converting
            if df[col].nunique(dropna=False) < 1000: # Adjust threshold as needed
                try:
                    df[col] = df[col].astype('category')
                    # logging.debug(f"Successfully cast '{col}' to category.")
                except Exception as e:
                    logging.error(f"Could not cast column '{col}' to category: {e}. Leaving as string.")
                    df[col] = df[col].astype(str).replace('nan', pd.NA).astype('string') # Use Pandas string type
            else:
                 logging.warning(f"Column '{col}' has high cardinality ({df[col].nunique(dropna=False)} unique values). Keeping as string.")
                 df[col] = df[col].astype(str).replace('nan', pd.NA).astype('string')
         # else:
         #    logging.debug(f"Categorical candidate column '{col}' not found.")


    # Remaining String/Object columns -> Use Pandas nullable string type 'string'
    # Start with remaining core cols, EXCLUDING Date and already processed types
    string_cols_candidates = list(set(CORE_COLS_FINAL) - set(['Date']) - set(bool_cols) - set(processed_as_int) - set(categorical_cols))
    string_cols_candidates.extend(['MatchID', 'HomeTeam', 'AwayTeam', 'Referee', 'VenueName', 'VenueCity', 'StatusLong']) # Add others explicitly
    string_cols_candidates = list(set(string_cols_candidates)) # Unique

    for col in string_cols_candidates:
        if col in df.columns and not pd.api.types.is_categorical_dtype(df[col]): # Check if not already category
             try:
                 # Use Pandas nullable string type <--- Ensure this line uses pd.NA
                 df[col] = df[col].astype(str).replace('nan', pd.NA).astype('string')
             except Exception as e:
                 logging.error(f"Could not cast column '{col}' to string: {e}. Leaving as is.")
        # else:
        #     logging.debug(f"String candidate column '{col}' not found or already category.")

    # Final check on Timestamp type if it exists
    if 'Timestamp' in df.columns:
        if not pd.api.types.is_integer_dtype(df['Timestamp']):
            df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce').astype('Int64')

    logging.info("Type casting finished.")


    # --- Sort by Date ---
    if 'Date' in df.columns and df['Date'].notna().any():
        logging.debug("Sorting by Date...")
        df = df.sort_values(by='Date').reset_index(drop=True)
    else:
        logging.warning("Date column not found or all nulls, skipping sort.")

    # --- Reorder Columns ---
    logging.debug("Reordering columns...")
    present_cols = df.columns.tolist()

    # Use constants, filtering by columns actually present in the dataframe
    core_ordered = [col for col in CORE_COLS_FINAL if col in present_cols]
    stats_ordered = [col for col in ALL_EXPECTED_STATS_COLS if col in present_cols]
    odds_ordered = [col for col in ALL_EXPECTED_ODDS_COLS if col in present_cols]

    # Dynamically find rolling features based on common patterns (adjust patterns as needed)
    home_rolling_ordered = sorted([col for col in present_cols if col.startswith('Home_Avg') or col.startswith('Home_Form') or col.startswith('Home_BTTS') or col.startswith('Home_W_Count') or col.startswith('Home_D_Count') or col.startswith('Home_L_Count') ])
    away_rolling_ordered = sorted([col for col in present_cols if col.startswith('Away_Avg') or col.startswith('Away_Form') or col.startswith('Away_BTTS') or col.startswith('Away_W_Count') or col.startswith('Away_D_Count') or col.startswith('Away_L_Count') ])

    # Combine ordered groups
    known_ordered_cols = core_ordered + stats_ordered + odds_ordered + home_rolling_ordered + away_rolling_ordered

    # Find any remaining columns not captured above
    remaining_cols = sorted(list(set(present_cols) - set(known_ordered_cols)))
    if remaining_cols:
        logging.warning(f"Found {len(remaining_cols)} columns not explicitly categorized during reordering: {remaining_cols}. Appending them at the end.")

    # Final column order
    final_ordered_cols = known_ordered_cols + remaining_cols
    final_ordered_cols = list(dict.fromkeys(final_ordered_cols)) # Ensure unique cols in final list

    # Verify and apply order
    if set(final_ordered_cols) != set(present_cols):
        logging.error("Column mismatch during reordering! Columns expected vs present differ.")
        logging.error(f"Expected based on ordering: {len(final_ordered_cols)} cols") #{final_ordered_cols}") # Avoid logging huge list
        logging.error(f"Present in DataFrame: {len(present_cols)} cols") #{present_cols}")
        logging.warning("Proceeding with original column order due to error.")
        # Optionally return df as is, or raise an error
    else:
        df = df[final_ordered_cols]
        logging.debug("Column reordering applied successfully.")

    lost_cols = original_cols - set(df.columns)
    if lost_cols:
         logging.warning(f"Columns lost during cleaning/ordering: {lost_cols}")

    gained_cols = set(df.columns) - original_cols
    if gained_cols:
         logging.warning(f"Columns gained during cleaning/ordering (unexpected): {gained_cols}")


    logging.info(f"Final cleaning and ordering complete. Output shape: {df.shape}")
    return df


# Placeholder for calculate_rolling_features - Not needed if using Mongo's features
def calculate_rolling_features(df: pd.DataFrame, windows: list = [5, 10, 15]) -> pd.DataFrame:
     logging.warning("Rolling feature calculation is SKIPPED as features are expected from MongoDB data.")
     # No operation, just return the dataframe as is
     return df

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Merge and Clean Script ---")

    # 1. Load Input Parquet Files
    logging.info(f"Loading CSV data from: {CSV_PARQUET_PATH}")
    if not os.path.exists(CSV_PARQUET_PATH):
        logging.error(f"CSV Parquet file not found: {CSV_PARQUET_PATH}. Exiting.")
        sys.exit(1)
    try:
        csv_data = pd.read_parquet(CSV_PARQUET_PATH)
        logging.info(f"Loaded CSV data shape: {csv_data.shape}")
        # Log CSV columns for debugging
        # logging.debug(f"CSV Columns: {csv_data.columns.tolist()}")
    except Exception as e:
        logging.error(f"Failed to load CSV Parquet {CSV_PARQUET_PATH}: {e}", exc_info=True)
        sys.exit(1)

    logging.info(f"Loading MongoDB data from: {MONGO_PARQUET_PATH}")
    if not os.path.exists(MONGO_PARQUET_PATH):
        logging.error(f"Mongo Parquet file not found: {MONGO_PARQUET_PATH}. Exiting.")
        sys.exit(1)
    try:
        mongo_data = pd.read_parquet(MONGO_PARQUET_PATH)
        logging.info(f"Loaded MongoDB data shape: {mongo_data.shape}")
        # Log Mongo columns for debugging
        # logging.debug(f"Mongo Columns: {mongo_data.columns.tolist()}")
    except Exception as e:
        logging.error(f"Failed to load Mongo Parquet {MONGO_PARQUET_PATH}: {e}", exc_info=True)
        sys.exit(1)

    # 2. Merge Data (Prioritizing Mongo base, CSV odds)
    unified_data = merge_data(csv_data, mongo_data)
    if unified_data.empty:
        logging.error("Merging failed or resulted in an empty DataFrame. Exiting.")
        sys.exit(1)
    logging.info(f"Unified data shape after merge: {unified_data.shape}")

    # 3. Calculate Rolling Average/Form Features - SKIPPED
    # unified_data_with_features = calculate_rolling_features(unified_data, windows=[5, 10, 15])
    # logging.info(f"Skipped rolling feature calculation. Shape remains: {unified_data_with_features.shape}")
    # Use unified_data directly in the next step
    unified_data_with_features = unified_data # Assign for consistency if needed downstream

    # 4. Final Cleaning (Type casting, sorting, column reordering)
    final_data = final_clean_and_order(unified_data_with_features)
    if final_data.empty:
         logging.error("Final cleaning resulted in an empty DataFrame. Exiting.")
         sys.exit(1)
    logging.info(f"Final data shape after cleaning: {final_data.shape}")

    # 5. Save Final Unified Data to Parquet
    try:
        # Final check for duplicate columns before saving (should be handled earlier)
        final_data = final_data.loc[:, ~final_data.columns.duplicated(keep='first')]
        logging.info(f"Shape before saving to Parquet: {final_data.shape}")

        final_data.to_parquet(FINAL_OUTPUT_PARQUET_PATH, index=False, engine='pyarrow', compression='snappy')
        logging.info(f"Successfully saved final unified data to: {FINAL_OUTPUT_PARQUET_PATH}")
        logging.info(f"Final DataFrame columns saved ({len(final_data.columns)} columns).") # Avoid logging all columns

        # Log detailed info
        buffer = io.StringIO()
        final_data.info(buf=buffer, verbose=True, show_counts=True, memory_usage='deep')
        logging.info(f"Saved DataFrame info:\n{buffer.getvalue()}")

    except Exception as e:
        logging.error(f"Failed to save final unified data to {FINAL_OUTPUT_PARQUET_PATH}: {e}", exc_info=True)

    logging.info("--- Merge and Clean Script Finished ---")
