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

def apply_feature_scaling(df: pd.DataFrame) -> tuple:
    """
    Apply appropriate feature scaling to different feature categories and create feature metadata.
    """
    if df.empty: 
        logging.warning("Input DF to feature scaling is empty.")
        return df, pd.DataFrame()
    
    logging.info("Applying feature scaling to harmonize feature magnitudes...")
    
    # Initialize metadata tracking
    metadata_records = []
    
    # Create a copy for scaled features only
    scaled_df = df.copy()
    
    # --- 1. Categorize Features ---
    # Identify different feature types based on naming patterns
    count_features = [col for col in df.columns if ('_Count' in col or 'FormPoints' in col 
                                                   or col in ['HS', 'AS', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'])]
    
    ratio_features = [col for col in df.columns if ('_Ratio' in col or 'CleanSheet_Ratio' in col 
                                                   or 'BTTS_Ratio' in col or 'Possession' in col)]
    
    # Keep odds in original decimal format - remove them from scaling
    odds_columns = [col for col in df.columns if col.startswith(('B365', 'PS', 'WH', 'VC', 'IW', 'Bb', 'Avg', 'Max', 'Odds'))]
    
    # CSV-specific stats that should be treated as raw counts
    csv_count_stats = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    
    # Core features that shouldn't be scaled
    core_features = ['MatchID', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                     'HTHG', 'HTAG', 'HTR', 'LeagueID', 'LeagueName', 'Referee']

    # --- 2. Handle Odds Columns (Keep as Decimal) ---
    for col in odds_columns:
        if col in df.columns:
            # Keep odds as float64 but don't scale them
            scaled_df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            
            # Add to metadata
            metadata_records.append({
                'feature_name': col,
                'feature_type': 'Odds (Decimal)',
                'scaling_method': 'None - kept as decimal odds',
                'original_mean': df[col].mean(),
                'original_std': df[col].std(),
                'original_min': df[col].min(),
                'original_max': df[col].max(),
                'original_nunique': df[col].nunique(),
                'is_scaled': False
            })

    # --- 3. Apply Min-Max Scaling to Count Features ---
    for col in count_features:
        if col in df.columns:
            # Calculate min and max, handling NaNs
            col_min = df[col].min()
            col_max = df[col].max()
            
            if pd.notna(col_min) and pd.notna(col_max) and (col_max - col_min) > 0:
                metadata_records.append({
                    'feature_name': col,
                    'feature_type': 'Count',
                    'scaling_method': 'Min-Max Scaling',
                    'original_mean': df[col].mean(),
                    'original_std': df[col].std(),
                    'original_min': col_min,
                    'original_max': col_max,
                    'original_nunique': df[col].nunique(),
                    'is_scaled': True
                })
                
                scaled_df[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                metadata_records.append({
                    'feature_name': col,
                    'feature_type': 'Count',
                    'scaling_method': 'None - invalid statistics',
                    'original_mean': df[col].mean(),
                    'original_std': df[col].std(),
                    'original_min': col_min,
                    'original_max': col_max,
                    'original_nunique': df[col].nunique(),
                    'is_scaled': False
                })

    # --- 4. Apply Ratio Features (already in [0,1] range) ---
    for col in ratio_features:
        if col in df.columns:
            # Add to metadata (preserved as-is)
            metadata_records.append({
                'feature_name': col,
                'feature_type': 'Ratio',
                'scaling_method': 'None - already in [0,1] range',
                'original_mean': df[col].mean(),
                'original_std': df[col].std(),
                'original_min': df[col].min(),
                'original_max': df[col].max(),
                'original_nunique': df[col].nunique(),
                'is_scaled': False
            })

    # --- 5. Add Metadata for Core Features (not scaled) ---
    for col in core_features:
        if col in df.columns:
            metadata_records.append({
                'feature_name': col,
                'feature_type': 'Core',
                'scaling_method': 'None - core feature',
                'original_mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'original_std': df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'original_min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'original_max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'original_nunique': df[col].nunique(),
                'is_scaled': False
            })

    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata_records)
    
    # Sort metadata by feature type and name for better organization
    if not metadata_df.empty:
        metadata_df = metadata_df.sort_values(['feature_type', 'feature_name']).reset_index(drop=True)
    
    logging.info(f"Feature scaling complete. Generated metadata for {len(metadata_df)} features.")
    
    return scaled_df, metadata_df

def final_clean_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies final type casting with optimized types for ML and analysis.
    """
    logging.info("Starting final cleaning and type optimization...")

    # --- Core Match Info (Int64) ---
    score_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG']
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # --- Match Stats (Int64) ---
    int_stats = [
        'HomeShots', 'AwayShots', 
        'HomeShotsTarget', 'AwayShotsTarget',
        'HomeFouls', 'AwayFouls',
        'HomeCorners', 'AwayCorners',
        'HomeYellowCards', 'AwayYellowCards',
        'HomeRedCards', 'AwayRedCards'
    ]
    for col in int_stats:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # --- Odds and Probabilities (float64) ---
    # Main odds columns (high coverage >80%)
    main_odds = [
        'B365H', 'B365D', 'B365A',  # Coverage: 88.71%
        'BWH', 'BWD', 'BWA',        # Coverage: 86.03%
        'IWH', 'IWD', 'IWA',        # Coverage: 81.79%
        'WHH', 'WHD', 'WHA',        # Coverage: 85.64%
        'VCH', 'VCD', 'VCA',        # Coverage: 81.97%
        'OddsH', 'OddsD', 'OddsA'   # Coverage: 88.97%
    ]
    
    # Additional odds (lower coverage but still important)
    additional_odds = [
        'PSH', 'PSD', 'PSA',
        'MaxH', 'MaxD', 'MaxA',
        'AvgH', 'AvgD', 'AvgA',
        'B365>2.5', 'B365<2.5',
        'P>2.5', 'P<2.5',
        'Max>2.5', 'Max<2.5',
        'Avg>2.5', 'Avg<2.5',
        'OddsOver2.5', 'OddsUnder2.5',
        'OddsAHH', 'OddsAHA', 'OddsAHh'
    ]

    # Asian Handicap related
    ah_odds = [
        'AHh', 'B365AHH', 'B365AHA',
        'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA',
        'AvgAHH', 'AvgAHA'
    ]

    # All odds columns combined
    all_odds_cols = main_odds + additional_odds + ah_odds

    for col in all_odds_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    # --- Categorical Columns (low cardinality) ---
    categorical_cols = {
        'FTR': ['H', 'D', 'A'],           # Match result
        'HTR': ['H', 'D', 'A'],           # Half-time result
        'LeagueName': None,                # Will check cardinality
        'Country': None                    # Will check cardinality
    }

    for col, allowed_values in categorical_cols.items():
        if col in df.columns:
            if allowed_values:
                # For columns with known values, enforce them
                df[col] = pd.Categorical(df[col], categories=allowed_values)
            else:
                # For others, check cardinality first
                n_unique = df[col].nunique()
                if n_unique < 100:  # Conservative threshold
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype('string')

    # --- String Columns (high cardinality) ---
    string_cols = [
        'HomeTeam', 'AwayTeam',    # Team names
        'Referee',                  # Officials
        'Time',                     # Match time
        'MatchID'                   # Unique identifier
    ]
    
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None', ''], pd.NA).astype('string')

    # --- Date Column ---
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # --- Season/ID Columns (Int64) ---
    id_cols = ['Season']  # Add other ID columns if needed
    for col in id_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # --- Betting Exchange Columns (float64) ---
    # These have low coverage but need precision
    exchange_cols = [col for col in df.columns if col.startswith(('Bb', 'BF', '1XB'))]
    for col in exchange_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    logging.info("Type optimization complete.")
    return df

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

    # 2. Apply Feature Scaling
    if not csv_data_full.empty:
        try:
            # Apply feature scaling
            scaled_csv_data, feature_metadata = apply_feature_scaling(csv_data_full)
            logging.info(f"Feature scaling applied. Shape: {scaled_csv_data.shape}")
            
            # Save raw data
            raw_output_path = os.path.join(os.path.dirname(OUTPUT_PARQUET_PATH), "csv_unified_raw.parquet")
            csv_data_full = csv_data_full.loc[:, ~csv_data_full.columns.duplicated(keep='first')]
            csv_data_full.to_parquet(raw_output_path, index=False, engine='pyarrow', compression='snappy')
            logging.info(f"Successfully saved raw CSV data to: {raw_output_path}")
            
            # Save scaled data
            scaled_csv_data = scaled_csv_data.loc[:, ~scaled_csv_data.columns.duplicated(keep='first')]
            scaled_csv_data.to_parquet(OUTPUT_PARQUET_PATH, index=False, engine='pyarrow', compression='snappy')
            logging.info(f"Successfully saved scaled CSV data to: {OUTPUT_PARQUET_PATH}")
            
            # Save feature metadata
            if not feature_metadata.empty:
                metadata_output_path = os.path.join(os.path.dirname(OUTPUT_PARQUET_PATH), "csv_feature_metadata.parquet")
                feature_metadata.to_parquet(metadata_output_path, index=False, engine='pyarrow', compression='snappy')
                
                # Also save as CSV for easier viewing
                csv_metadata_path = os.path.join(os.path.dirname(OUTPUT_PARQUET_PATH), "csv_feature_metadata.csv")
                feature_metadata.to_csv(csv_metadata_path, index=False)
                logging.info(f"Saved feature metadata to: {metadata_output_path} and {csv_metadata_path}")

            # Log detailed info
            buffer = io.StringIO()
            scaled_csv_data.info(buf=buffer, verbose=False, show_counts=True, memory_usage='deep')
            logging.info(f"Saved DataFrame info:\n{buffer.getvalue()}")

        except Exception as e:
            logging.error(f"Failed to save unified CSV data: {e}", exc_info=True)
    else:
        logging.error("No data available after processing CSVs. Output file not saved.")

    logging.info("--- CSV Data Standardization Script Finished ---")
