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

    # Update paths to match your actual file locations
    CSV_PARQUET_PATH = "/Users/barroca888/Downloads/Agenticfc/AgenticFC888/scripts/data/unified_data/csv_normalized.parquet"
    MONGO_PARQUET_PATH = "/Users/barroca888/Downloads/Agenticfc/AgenticFC888/output/parquet/mongo_normalized.parquet"
    OUTPUT_DIR = os.path.join(project_root, "data", "unified_data")
    OUTPUT_FILENAME = "final_unified_data.parquet"

except Exception as e:
    print(f"Error setting up project root or paths: {e}. Using defaults.")
    # Use the same absolute paths as fallback
    CSV_PARQUET_PATH = "/Users/barroca888/Downloads/Agenticfc/AgenticFC888/scripts/data/unified_data/csv_normalized.parquet"
    MONGO_PARQUET_PATH = "/Users/barroca888/Downloads/Agenticfc/AgenticFC888/output/parquet/mongo_normalized.parquet"
    OUTPUT_DIR = os.path.join(os.path.dirname(CSV_PARQUET_PATH))
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

# Move valuable_csv_cols to global scope at the top of the file
VALUABLE_CSV_COLS = [
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
    'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA'
]

def analyze_dataframes(mongo_df: pd.DataFrame, csv_df: pd.DataFrame) -> dict:
    """Analyze both dataframes to understand column overlap and data quality"""
    analysis = {
        'mongo_only': set(mongo_df.columns) - set(csv_df.columns),
        'csv_only': set(csv_df.columns) - set(mongo_df.columns),
        'common': set(mongo_df.columns) & set(csv_df.columns),
        'column_stats': {}
    }
    
    # Analyze fill rates for common columns
    for col in analysis['common']:
        mongo_fill = (mongo_df[col].notna().sum() / len(mongo_df)) * 100
        csv_fill = (csv_df[col].notna().sum() / len(csv_df)) * 100
        analysis['column_stats'][col] = {
            'mongo_fill_rate': mongo_fill,
            'csv_fill_rate': csv_fill,
            'preferred_source': 'csv' if csv_fill > mongo_fill else 'mongo'
        }
    
    return analysis

def merge_data(csv_df: pd.DataFrame, mongo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced merge function with better column handling and validation
    """
    if mongo_df.empty or csv_df.empty:
        logging.error("One of the input DataFrames is empty")
        return pd.DataFrame()

    # Analyze dataframes
    analysis = analyze_dataframes(mongo_df, csv_df)
    logging.info(f"Found {len(analysis['common'])} common columns")
    logging.info(f"MongoDB-only columns: {len(analysis['mongo_only'])}")
    logging.info(f"CSV-only columns: {len(analysis['csv_only'])}")

    # Ensure MatchID presence and type consistency
    for df_name, df in [('mongo', mongo_df), ('csv', csv_df)]:
        if 'MatchID' not in df.columns:
            logging.error(f"MatchID missing in {df_name} DataFrame")
            return pd.DataFrame()
        df['MatchID'] = df['MatchID'].astype('string')

    try:
        # First merge with only odds columns from CSV
        odds_cols = ['MatchID'] + [col for col in VALUABLE_CSV_COLS if col != 'MatchID']
        csv_odds = csv_df[odds_cols]
        
        # Perform merge
        merged_df = pd.merge(
            mongo_df,
            csv_odds,
            on='MatchID',
            how='left',
            indicator=True
        )
        
        # Log merge results
        merge_counts = merged_df['_merge'].value_counts()
        logging.info(f"Merge results: {merge_counts.to_dict()}")
        merged_df.drop('_merge', axis=1, inplace=True)
        
        # Fill missing values appropriately
        for col in merged_df.columns:
            if col.startswith(('B365', 'BW', 'PS', 'WH', 'VC', 'Avg', 'Max')):
                merged_df[col] = merged_df[col].fillna(-1)  # Use -1 for missing odds
            elif '_Count' in col or col.endswith(('Cards', 'FTHG', 'FTAG', 'HTHG', 'HTAG')):
                merged_df[col] = merged_df[col].fillna(0)  # Use 0 for missing counts
        
        logging.info(f"Final merged shape: {merged_df.shape}")
        return merged_df

    except Exception as e:
        logging.error(f"Merge failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

# --- Feature Calculation and Cleaning Functions ---

# Update final_clean_and_order for robust casting and ML optimization
def final_clean_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies final type casting with optimized types for ML and analysis.
    """
    logging.info(f"Starting final cleaning and ordering. Input shape: {df.shape}")
    
    # --- Handle float columns first ---
    odds_cols = [
        'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 
        'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA',
        'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5',
        'B365AHH', 'B365AHA', 'PAHH', 'PAHA',
        'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA'
    ]
    
    # Process float columns in batches
    float_cols = [col for col in odds_cols if col in df.columns]
    float_cols.extend([c for c in df.columns if ('_Ratio' in c or '_Probability' in c)])
    
    for col in float_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        except Exception as e:
            logging.warning(f"Could not convert {col} to float32: {e}")
    
    logging.info(f"Converted {len(float_cols)} columns to float32")

    # --- Handle integer columns ---
    int_cols = {
        'int8': ['FTHG', 'FTAG', 'HTHG', 'HTAG'],  # Game scores usually 0-10
        'int16': ['Season', 'LeagueID', 'HomeTeamID', 'AwayTeamID'],  # Larger IDs
        'int32': ['StatusElapsed']  # Time values
    }
    
    # Add dynamic columns to appropriate types
    for col in df.columns:
        if '_Count' in col or 'YellowCards' in col or 'RedCards' in col:
            int_cols['int8'].append(col)
    
    # Convert integers safely
    for dtype, cols in int_cols.items():
        existing_cols = [col for col in cols if col in df.columns]
        for col in existing_cols:
            try:
                # First convert to float64 to handle any decimal values
                temp = pd.to_numeric(df[col], errors='coerce')
                # Round to nearest integer
                temp = temp.round()
                # Convert to integer, handling NaN values
                df[col] = pd.array(temp, dtype=f"Int{dtype.replace('int', '')}")
            except Exception as e:
                logging.warning(f"Could not convert {col} to {dtype}: {e}")
    
    # --- Handle categorical columns ---
    cat_cols = [
        'LeagueName', 'Country', 'Round',
        'FTR', 'HTR', 'StatusShort',
        'HomeFormation', 'AwayFormation'
    ]
    
    for col in cat_cols:
        if col in df.columns and df[col].nunique() < 1000:
            try:
                df[col] = df[col].astype('category')
            except Exception as e:
                logging.warning(f"Could not convert {col} to category: {e}")
    
    # --- Handle string columns ---
    str_cols = [
        'MatchID', 'HomeTeam', 'AwayTeam',
        'Referee', 'VenueName', 'VenueCity',
        'StatusLong'
    ]
    
    for col in str_cols:
        if col in df.columns:
            try:
                df[col] = df[col].astype(str).replace(['nan', 'None', ''], pd.NA).astype('string')
            except Exception as e:
                logging.warning(f"Could not convert {col} to string: {e}")
    
    # --- Handle datetime ---
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception as e:
            logging.warning(f"Could not convert Date to datetime: {e}")
    
    if 'Timestamp' in df.columns:
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        except Exception as e:
            logging.warning(f"Could not convert Timestamp to datetime: {e}")
    
    # Log summary
    logging.info("Column type conversion summary:")
    for dtype in df.dtypes.unique():
        count = (df.dtypes == dtype).sum()
        logging.info(f"{dtype}: {count} columns")
    
    return df


# Placeholder for calculate_rolling_features - Not needed if using Mongo's features
def calculate_rolling_features(df: pd.DataFrame, windows: list = [5, 10, 15]) -> pd.DataFrame:
     logging.warning("Rolling feature calculation is SKIPPED as features are expected from MongoDB data.")
     # No operation, just return the dataframe as is
     return df

# Add these checks after merge
def validate_merged_data(df: pd.DataFrame) -> bool:
    # Check for expected columns
    required_cols = ['MatchID', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for duplicate MatchIDs
    duplicates = df['MatchID'].duplicated().sum()
    if duplicates > 0:
        logging.error(f"Found {duplicates} duplicate MatchIDs")
        return False
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        logging.error("Date column is not datetime type")
        return False
    
    return True

def log_memory_usage(df: pd.DataFrame, stage: str):
    mem_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    logging.info(f"Memory usage at {stage}: {mem_usage:.2f} MB")
    logging.info(f"Shape at {stage}: {df.shape}")

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting enhanced merge process...")
    
    try:
        # Load data with optimized settings
        csv_data = pd.read_parquet(
            CSV_PARQUET_PATH,
            engine='pyarrow',
            columns=VALUABLE_CSV_COLS,  # Only load needed columns
            use_threads=True
        )
        
        mongo_data = pd.read_parquet(
            MONGO_PARQUET_PATH,
            engine='pyarrow',
            use_threads=True
        )
        
        # Log initial data info
        log_memory_usage(csv_data, "csv_initial")
        log_memory_usage(mongo_data, "mongo_initial")
        
        # Perform merge
        merged_data = merge_data(csv_data, mongo_data)
        if merged_data.empty:
            logging.error("Merge failed - empty result")
            sys.exit(1)
            
        # Clean and optimize
        final_data = final_clean_and_order(merged_data)
        
        # Save with optimization - removed use_threads parameter
        final_data.to_parquet(
            FINAL_OUTPUT_PARQUET_PATH,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        # Log final statistics
        logging.info(f"Successfully saved merged data to {FINAL_OUTPUT_PARQUET_PATH}")
        log_memory_usage(final_data, "final")
        
        # Print column type summary
        type_counts = final_data.dtypes.value_counts()
        logging.info("\nFinal column type distribution:")
        for dtype, count in type_counts.items():
            logging.info(f"{dtype}: {count} columns")
        
        # Print sample of columns for each type
        logging.info("\nSample columns by type:")
        for dtype in final_data.dtypes.unique():
            cols = final_data.select_dtypes(include=[dtype]).columns
            logging.info(f"{dtype}: {list(cols[:5])}...")
        
    except Exception as e:
        logging.error(f"Process failed: {str(e)}", exc_info=True)
        sys.exit(1)

# Add path existence checks
for path in [CSV_PARQUET_PATH, MONGO_PARQUET_PATH]:
    if not os.path.exists(path):
        logging.error(f"Input file not found: {path}")
        sys.exit(1)
