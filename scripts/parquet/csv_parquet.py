# scripts/create_csv_parquet.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import glob
from dateutil.parser import parse
import logging
import re
import sys
import io

# --- Configuration Setup ---
# Add project root to sys.path (adjust if your script location differs)
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Project Root added to sys.path: {project_root}")

    # Attempt to import shared config if it exists, otherwise define fallback
    try:
        from models.utils import config as project_config
        print("Successfully imported config from models.utils")
        RAW_CSV_DIR = project_config.RAW_CSV_DIR
        TEAM_NAME_MAPPING = project_config.TEAM_NAME_MAPPING
        OUTPUT_DIR = os.path.join(project_root, "data", "unified_data")
        OUTPUT_FILENAME = "csv_unified_full_cols.parquet"
    except ImportError:
        print("Could not import project config. Using default settings in script.")
        RAW_CSV_DIR = os.path.join(project_root, "football_data_db")
        TEAM_NAME_MAPPING = {}
        OUTPUT_DIR = os.path.join(project_root, "data", "unified_data")
        OUTPUT_FILENAME = "csv_unified_full_cols.parquet"

except Exception as e:
    print(f"Error setting up project root or config: {e}. Using defaults.")
    RAW_CSV_DIR = "football_data_db"
    TEAM_NAME_MAPPING = {}
    OUTPUT_DIR = "data/unified_data"
    OUTPUT_FILENAME = "csv_unified_full_cols.parquet"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PARQUET_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Define desired standardized ODDS columns (map source names to these)
ODDS_COLS_MAP = {
    # Target Name : [List of potential source column names in order of preference]
    'OddsH': ['B365H', 'PSH', 'AvgH', 'MaxH', 'WHH', 'VCH', 'LBH'],
    'OddsD': ['B365D', 'PSD', 'AvgD', 'MaxD', 'WHD', 'VCD', 'LBD'],
    'OddsA': ['B365A', 'PSA', 'AvgA', 'MaxA', 'WHA', 'VCA', 'LBA'],
    'OddsOver2.5': ['B365>2.5', 'P>2.5', 'Max>2.5', 'Avg>2.5'],
    'OddsUnder2.5': ['B365<2.5', 'P<2.5', 'Max<2.5', 'Avg<2.5'],
    'OddsAHH': ['AHH', 'PSCH', 'AvgAHH', 'MaxAHH', 'B365AHH'],
    'OddsAHA': ['AHA', 'PSCA', 'AvgAHA', 'MaxAHA', 'B365AHA'],
    'OddsAHh': ['AHh', 'AHLine', 'HandicapLine', 'B365AHh', 'BbAHh']
}

# Standardized names for common CSV stats columns
CSV_STATS_MAP = {
    'HS': 'HomeShots', 'AS': 'AwayShots',
    'HST': 'HomeShotsTarget', 'AST': 'AwayShotsTarget',
    'HF': 'HomeFouls', 'AF': 'AwayFouls',
    'HC': 'HomeCorners', 'AC': 'AwayCorners',
    'HY': 'HomeYellowCards', 'AY': 'AwayYellowCards',
    'HR': 'HomeRedCards', 'AR': 'AwayRedCards'
}
ALL_STANDARDIZED_CSV_STATS_COLS = list(CSV_STATS_MAP.values())


# --- Helper Functions ---

def parse_date(date_input):
    """Attempts to parse various date formats and return tz-naive UTC datetime."""
    if pd.isna(date_input): return pd.NaT
    dt = None
    if isinstance(date_input, (int, float)): # Handle numeric timestamps (seconds or ms)
        try: dt = pd.to_datetime(date_input, unit='s', errors='raise')
        except (ValueError, TypeError, OverflowError):
            try: dt = pd.to_datetime(date_input, unit='ms', errors='raise')
            except (ValueError, TypeError, OverflowError): return pd.NaT
    elif isinstance(date_input, str): # Handle string formats
        common_formats = ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%Y/%m/%d",
                          "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", # ISO with TZ
                          "%Y-%m-%d %H:%M:%S"] # Common DB format
        for fmt in common_formats:
            try: dt = pd.to_datetime(date_input, format=fmt, errors='raise'); break
            except (ValueError, TypeError): continue
        if dt is None: # Fallback to flexible parsing
            try: dt = pd.to_datetime(date_input, errors='raise', infer_datetime_format=True)
            except (ValueError, TypeError, OverflowError):
                try: parsed_dt = parse(date_input); dt = pd.Timestamp(parsed_dt) # dateutil fallback
                except (ValueError, TypeError, OverflowError): return pd.NaT
    elif isinstance(date_input, (pd.Timestamp, datetime.datetime)): # Handle existing objects
        dt = pd.to_datetime(date_input)
    else: return pd.NaT # Unrecognized type

    if pd.isna(dt): return pd.NaT
    # Ensure UTC and naive
    try:
        if not isinstance(dt, pd.Timestamp): dt = pd.Timestamp(dt)
        if dt.tzinfo is not None: dt = dt.tz_convert('UTC').tz_localize(None)
        # else: dt = dt.tz_localize('UTC').tz_localize(None) # Assume UTC if naive - Let's keep naive dates as they are from CSVs unless TZ is explicit
        return dt.normalize() # Keep only date part if no time info, make naive
    except Exception as e:
        logging.warning(f"Error during final date conversion/tz handling for {dt}: {e}")
        return pd.NaT

def standardize_team_names(name: str, mapping: dict) -> str:
    """Applies cleaning and mapping to team names."""
    if pd.isna(name): return None
    name = str(name).strip()
    return mapping.get(name, name) # Return mapped name or original if no mapping

def create_match_id(row, date_col='Date', home_col='HomeTeam', away_col='AwayTeam'):
    """Creates a unique identifier string for a match. Uses Date."""
    try:
        # Ensure Date is valid datetime before formatting
        if pd.isna(row[date_col]) or not isinstance(row[date_col], pd.Timestamp):
            return None
        date_str = row[date_col].strftime('%Y%m%d')
        # Sanitize team names for use in ID
        home = re.sub(r'\W+', '', str(row[home_col])) # Remove non-alphanumeric
        away = re.sub(r'\W+', '', str(row[away_col]))
        if not home or not away or home == 'nan' or away == 'nan':
             return None
        return f"{date_str}_{home}_{away}"
    except Exception as e:
        logging.error(f"Error creating MatchID. Row: {row.get('HomeTeam','?')}-{row.get('AwayTeam','?')}, Date: {row.get('Date','?')}. Error: {e}", exc_info=False)
        return None

# --- Main Processing Function (Revised Logic) ---

def load_process_and_collect_csv_data(csv_dir: str, team_mapping: dict) -> pd.DataFrame:
    """
    Loads each CSV individually, standardizes core fields, keeps all other columns,
    and returns a concatenated DataFrame of all processed files.
    """
    all_files = glob.glob(os.path.join(csv_dir, "**/*.csv"), recursive=True)
    if not all_files:
        logging.warning(f"No CSV files found recursively in directory: {csv_dir}")
        return pd.DataFrame()
    logging.info(f"Found {len(all_files)} CSV files to process in '{csv_dir}'.")

    processed_dfs = [] # List to hold processed DataFrames from each file

    for f in all_files:
        logging.debug(f"Attempting to read CSV: {f}")
        try:
            df_temp = None
            try:
                # Read with warn for bad lines, let pandas infer columns for THIS file
                df_temp = pd.read_csv(f, low_memory=False, on_bad_lines='warn', engine='c')
            except UnicodeDecodeError:
                logging.warning(f"UnicodeDecodeError reading {f}. Trying ISO-8859-1.")
                try: df_temp = pd.read_csv(f, encoding='ISO-8859-1', low_memory=False, on_bad_lines='warn', engine='c')
                except Exception as inner_e: logging.error(f"Failed read {f} with ISO-8859-1: {inner_e}. Skip."); continue
            except pd.errors.EmptyDataError: logging.warning(f"Skipping empty CSV: {f}"); continue
            except pd.errors.ParserError as pe: logging.error(f"Parser error {f}: {pe}. Skip."); continue
            except Exception as e: logging.error(f"Unexpected error reading {f}: {e}. Skip."); continue

            if df_temp is None or df_temp.empty:
                logging.warning(f"DataFrame is None or empty after read attempt for {f}. Skip.")
                continue

            logging.info(f"Read {len(df_temp)} rows from {os.path.basename(f)}. Standardizing...")

            # --- Standardize Core Columns ---
            original_columns = df_temp.columns.tolist() # Keep track of original names
            rename_map = {}
            standardized_cols_added = [] # Track which standard names we created

            # Define potential names for core columns
            potential_date_cols = ['Date', 'datetime', 'date', 'Time']
            potential_home_cols = ['HomeTeam', 'Home', 'HT', 'home']
            potential_away_cols = ['AwayTeam', 'Away', 'AT', 'away']
            potential_fthg_cols = ['FTHG', 'HG', 'HomeGoals', 'home_score']
            potential_ftag_cols = ['FTAG', 'AG', 'AwayGoals', 'away_score']
            potential_hthg_cols = ['HTHG']
            potential_htag_cols = ['HTAG']
            potential_ftr_cols = ['FTR']
            potential_htr_cols = ['HTR']
            potential_div_cols = ['Div', 'League', 'league']
            potential_referee_cols = ['Referee']

            def find_and_map(potential_names, target_name, current_df):
                for name in potential_names:
                    if name in current_df.columns:
                        if name != target_name: rename_map[name] = target_name
                        standardized_cols_added.append(target_name)
                        return True
                return False

            # Check and map essential columns
            if not find_and_map(potential_date_cols, 'Date', df_temp): logging.warning(f"Skipping {f}: No 'Date' column found."); continue
            if not find_and_map(potential_home_cols, 'HomeTeam', df_temp): logging.warning(f"Skipping {f}: No 'HomeTeam' column found."); continue
            if not find_and_map(potential_away_cols, 'AwayTeam', df_temp): logging.warning(f"Skipping {f}: No 'AwayTeam' column found."); continue
            if not find_and_map(potential_fthg_cols, 'FTHG', df_temp): logging.warning(f"Skipping {f}: No 'FTHG' column found."); continue
            if not find_and_map(potential_ftag_cols, 'FTAG', df_temp): logging.warning(f"Skipping {f}: No 'FTAG' column found."); continue
            # Map optional core columns
            find_and_map(potential_hthg_cols, 'HTHG', df_temp); find_and_map(potential_htag_cols, 'HTAG', df_temp)
            find_and_map(potential_ftr_cols, 'FTR', df_temp); find_and_map(potential_htr_cols, 'HTR', df_temp)
            find_and_map(potential_div_cols, 'LeagueID', df_temp); find_and_map(potential_referee_cols, 'Referee', df_temp)

            df_temp.rename(columns=rename_map, inplace=True)

            # Perform essential processing (Date, Teams, Scores)
            df_temp.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'], inplace=True)
            if df_temp.empty: logging.warning(f"Dropped all rows from {f} after NA check on core columns."); continue

            df_temp['Date'] = df_temp['Date'].apply(parse_date)
            df_temp.dropna(subset=['Date'], inplace=True)
            if df_temp.empty: continue

            df_temp['HomeTeam'] = df_temp['HomeTeam'].apply(lambda x: standardize_team_names(x, team_mapping))
            df_temp['AwayTeam'] = df_temp['AwayTeam'].apply(lambda x: standardize_team_names(x, team_mapping))
            df_temp.dropna(subset=['HomeTeam', 'AwayTeam'], inplace=True)
            if df_temp.empty: continue

            df_temp['FTHG'] = pd.to_numeric(df_temp['FTHG'], errors='coerce').astype('Int64')
            df_temp['FTAG'] = pd.to_numeric(df_temp['FTAG'], errors='coerce').astype('Int64')
            df_temp.dropna(subset=['FTHG', 'FTAG'], inplace=True)
            if df_temp.empty: continue

            # Infer FTR if missing/not mapped
            if 'FTR' not in df_temp.columns: df_temp['FTR'] = None
            df_temp['FTR'] = df_temp.apply(lambda r: np.select([r['FTHG'] > r['FTAG'], r['FTHG'] < r['FTAG']], ['H', 'A'], 'D') if pd.isna(r['FTR']) else r['FTR'], axis=1).astype('string')

            # Handle Optional Half Time
            if 'HTHG' in df_temp.columns: df_temp['HTHG'] = pd.to_numeric(df_temp['HTHG'], errors='coerce').astype('Int64')
            else: df_temp['HTHG'] = pd.NA
            if 'HTAG' in df_temp.columns: df_temp['HTAG'] = pd.to_numeric(df_temp['HTAG'], errors='coerce').astype('Int64')
            else: df_temp['HTAG'] = pd.NA
            if 'HTR' not in df_temp.columns: df_temp['HTR'] = None
            df_temp['HTR'] = df_temp.apply(lambda r: np.select([r['HTHG'] > r['HTAG'], r['HTHG'] < r['HTAG']], ['H', 'A'], 'D') if pd.notna(r['HTHG']) and pd.notna(r['HTAG']) and pd.isna(r['HTR']) else r['HTR'], axis=1).astype('string')


            # --- Standardize Known Stats/Odds (if source exists in THIS file) ---
            stats_cols_standardized_this_file = []
            for source_csv_name, target_name in CSV_STATS_MAP.items():
                 if source_csv_name in df_temp.columns:
                     if target_name not in df_temp.columns: # Avoid overwriting if already named target_name
                          df_temp[target_name] = pd.to_numeric(df_temp[source_csv_name], errors='coerce')
                          standardized_cols_added.append(target_name)
                          stats_cols_standardized_this_file.append(target_name)
                     # Ensure numeric type even if target existed (e.g. direct rename)
                     elif target_name in df_temp.columns:
                          df_temp[target_name] = pd.to_numeric(df_temp[target_name], errors='coerce')
                          stats_cols_standardized_this_file.append(target_name)


            odds_cols_standardized_this_file = []
            for target_name, source_options in ODDS_COLS_MAP.items():
                selected_col = None
                for source_col in source_options:
                    # Check if source_col exists in the current df's columns
                    if source_col in df_temp.columns:
                        selected_col = source_col
                        break
                # If a preferred source was found in this file, create the standardized column
                if selected_col:
                     # Avoid overwriting if target_name already exists and wasn't the selected_col
                     if target_name not in df_temp.columns or selected_col == target_name:
                         df_temp[target_name] = pd.to_numeric(df_temp[selected_col], errors='coerce')
                         standardized_cols_added.append(target_name)
                         odds_cols_standardized_this_file.append(target_name)
                # Optional: ensure target column exists as NaN if no source found?
                # elif target_name not in df_temp.columns:
                #     df_temp[target_name] = np.nan


            # --- Standardize League/Season ---
            if 'LeagueID' not in df_temp.columns:
                match = re.match(r"([A-Z][A-Z0-9]?[0-9]?)(?:_(\d{4}))?", os.path.basename(f))
                league_id_from_file = match.group(1) if match else 'UnknownCSV'
                df_temp['LeagueID'] = league_id_from_file
                standardized_cols_added.append('LeagueID')
            df_temp['LeagueName'] = df_temp['LeagueID'] # Use ID as name
            standardized_cols_added.append('LeagueName')

            df_temp['Season'] = df_temp['Date'].dt.year
            df_temp['Season'] = df_temp.apply(lambda r: r['Season'] - 1 if r['Date'].month < 7 else r['Season'], axis=1).astype('Int64')
            standardized_cols_added.append('Season')

            # --- Create MatchID ---
            df_temp['MatchID'] = df_temp.apply(create_match_id, axis=1, date_col='Date', home_col='HomeTeam', away_col='AwayTeam')
            df_temp.dropna(subset=['MatchID'], inplace=True)
            if df_temp.empty: continue
            standardized_cols_added.append('MatchID')

            # --- Convert remaining non-standardized columns to appropriate types ---
            # Identify columns that were NOT part of the standardization process
            original_cols_kept = [col for col in original_columns if col not in rename_map and col in df_temp.columns]
            other_cols_to_process = [col for col in df_temp.columns if col not in standardized_cols_added]

            for col in other_cols_to_process:
                try:
                    numeric_col = pd.to_numeric(df_temp[col], errors='coerce')
                    if numeric_col.notna().any():
                        df_temp[col] = numeric_col.astype('float64')
                    else:
                        df_temp[col] = df_temp[col].astype('string').replace(['', 'NA', 'N/A', 'NaN', 'nan'], pd.NA)
                except (ValueError, TypeError):
                    df_temp[col] = df_temp[col].astype('string').replace(['', 'NA', 'N/A', 'NaN', 'nan'], pd.NA)


            processed_dfs.append(df_temp)
            logging.info(f"Finished standardizing {os.path.basename(f)}. Shape: {df_temp.shape}, Columns: {len(df_temp.columns)}")

        except Exception as e:
            logging.error(f"Failed processing/standardizing file {f}: {e}", exc_info=True)

    # --- Concatenate All Processed DataFrames ---
    if not processed_dfs:
        logging.error("No CSV files could be successfully processed.")
        return pd.DataFrame()

    logging.info(f"Concatenating {len(processed_dfs)} processed DataFrames...")
    try:
        # Concatenate: aligns columns based on name, fills missing with NaN
        combined_df = pd.concat(processed_dfs, ignore_index=True, sort=False)
        logging.info(f"Combined shape before deduplication: {combined_df.shape}, Total columns: {len(combined_df.columns)}")
    except Exception as e:
        logging.error(f"Error during final concatenation: {e}", exc_info=True)
        return pd.DataFrame()

    # --- Final Deduplication & Sort ---
    initial_rows = len(combined_df)
    combined_df['MatchID'] = combined_df['MatchID'].astype(str) # Ensure string type
    combined_df = combined_df.drop_duplicates(subset=['MatchID'], keep='first')
    rows_dropped = initial_rows - len(combined_df)
    if rows_dropped > 0:
        logging.info(f"Dropped {rows_dropped} duplicate MatchIDs from combined data.")

    # --- Remove Empty 'Unnamed: X' Columns ---
    unnamed_cols = [col for col in combined_df.columns if 'Unnamed:' in str(col)]
    if unnamed_cols:
        logging.info(f"Found {len(unnamed_cols)} 'Unnamed:' columns. Checking if empty...")
        cols_to_drop = []
        for col in unnamed_cols:
            if combined_df[col].isnull().all(): # Check if ALL values in the column are null/NaN
                cols_to_drop.append(col)
        if cols_to_drop:
            combined_df.drop(columns=cols_to_drop, inplace=True)
            logging.info(f"Dropped {len(cols_to_drop)} completely empty 'Unnamed:' columns.")
        else:
            logging.info("No completely empty 'Unnamed:' columns found to drop.")

    # --- Final Sort ---
    combined_df = combined_df.sort_values(by='Date').reset_index(drop=True)
    logging.info(f"Final DataFrame shape after cleanup: {combined_df.shape}")
    # logging.debug(f"Final columns: {combined_df.columns.tolist()}")

    return combined_df


# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"--- Starting CSV Data Standardization Script (Keeps All Columns) ---")
    logging.info(f"Reading CSV files from: {RAW_CSV_DIR}")
    logging.info(f"Output will be saved to: {OUTPUT_PARQUET_PATH}")

    # 1. Load, Process, and Concatenate CSV Data
    csv_data_full = load_process_and_collect_csv_data(
        csv_dir=RAW_CSV_DIR,
        team_mapping=TEAM_NAME_MAPPING
    )

    # 2. Save Unified CSV Data to Parquet
    if not csv_data_full.empty:
        try:
            # Final check for duplicate column *names* before saving (safety check)
            csv_data_full = csv_data_full.loc[:, ~csv_data_full.columns.duplicated(keep='first')]
            logging.info(f"Shape before saving to Parquet: {csv_data_full.shape}")

            csv_data_full.to_parquet(OUTPUT_PARQUET_PATH, index=False, engine='pyarrow', compression='snappy')
            logging.info(f"Successfully saved unified CSV data (all columns) to: {OUTPUT_PARQUET_PATH}")
            logging.info(f"Final DataFrame columns saved ({len(csv_data_full.columns)}). Sample: {csv_data_full.columns.tolist()[:10]}...") # Log sample cols

            # Log detailed info
            buffer = io.StringIO()
            csv_data_full.info(buf=buffer, verbose=False, show_counts=True, memory_usage='deep') # Less verbose info
            logging.info(f"Saved DataFrame info:\n{buffer.getvalue()}")

        except Exception as e:
            logging.error(f"Failed to save unified CSV data to {OUTPUT_PARQUET_PATH}: {e}", exc_info=True)
    else:
        logging.error("No data available after processing CSVs. Output file not saved.")

    logging.info("--- CSV Data Standardization Script Finished ---")
