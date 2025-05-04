import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_column_variance(df):
    """Analyze column variance and identify potential issues"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    zero_var_cols = []
    near_zero_var_cols = []
    missing_stats = {}
    
    for col in numeric_cols:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        std = df[col].std()
        if std == 0:
            zero_var_cols.append((col, missing_pct))
        elif std < 1e-6:
            near_zero_var_cols.append((col, std, missing_pct))
        missing_stats[col] = missing_pct
    
    return zero_var_cols, near_zero_var_cols, missing_stats

def handle_missing_values(df, col, strategy='zero'):
    """Handle missing values with specified strategy"""
    if strategy == 'zero':
        df[col] = df[col].fillna(0)
    elif strategy == 'mean':
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val if pd.notnull(mean_val) else 0)
    elif strategy == 'median':
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if pd.notnull(median_val) else 0)
    elif strategy == 'mode':
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode_val)
    return df[col]

def optimize_football_dtypes(df):
    """Optimize DataFrame dtypes specifically for football betting data"""
    
    # 1. Define column groups based on our data structure
    categorical_cols = [
        'LeagueName', 'Country', 'Round', 'FTR', 'HTR', 
        'StatusShort', 'HomeFormation', 'AwayFormation',
        'Referee', 'VenueName', 'VenueCity', 'StatusLong'
    ]
    
    date_cols = ['Date', 'Timestamp']
    
    boolean_cols = ['HomeTeamWinner', 'AwayTeamWinner']
    
    # Match statistics (small integers)
    int8_cols = [
        'FTHG', 'FTAG', 'HTHG', 'HTAG',
        'HomeYellowCards', 'AwayYellowCards',
        'HomeRedCards', 'AwayRedCards'
    ]
    
    # Add form points and count columns to int8
    int8_cols.extend([col for col in df.columns if '_Count_' in col or 'FormPoints_' in col])
    
    # IDs and larger integers
    int16_cols = ['Season', 'LeagueID', 'HomeTeamID', 'AwayTeamID']
    
    # Time-related integers
    int32_cols = ['StatusElapsed', 'fixture_id']
    
    # String columns
    string_cols = [
        'MatchID', 'HomeTeam', 'AwayTeam',
        'Referee', 'VenueName', 'VenueCity'
    ]
    
    # Odds and probabilities (float32)
    float_patterns = [
        'B365', 'BW', 'IW', 'PS', 'WH', 'VC', 'GB', 'Avg', 'Max',
        '_Ratio', '_Probability', 'ELO', 'Expected', 'Accuracy',
        'Possession', 'Pass', 'Shots', 'Corners', 'Fouls'
    ]

    # Features that should be scaled with StandardScaler
    standardize_patterns = ['ELO', 'Expected', 'Total', 'Passes', 'Shots', 'Corners']
    
    # Features that should be scaled with MinMaxScaler
    minmax_patterns = ['_Count_', 'FormPoints_', 'YellowCards', 'RedCards']

    try:
        # Handle categorical columns first
        for col in categorical_cols:
            if col in df.columns:
                # First convert to string to handle any numeric values
                df[col] = df[col].astype(str)
                # Fill missing values with 'Unknown'
                df[col] = df[col].fillna('Unknown')
                
                nunique = df[col].nunique()
                if nunique < len(df) * 0.6:  # Less than 60% unique values
                    df[col] = df[col].astype('category')
                    logging.info(f"Converted {col} to category (cardinality: {nunique})")
                else:
                    logging.info(f"Keeping {col} as object due to high cardinality: {nunique} unique values")

        # New boolean handling
        for col in boolean_cols:
            if col in df.columns:
                logging.info(f"Processing binary column: {col}")
                try:
                    # 1. Handle potential string representations first if dtype is object/string
                    if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                        # Define mappings (case-insensitive)
                        map_dict = {'true': 1, 'yes': 1, '1': 1, 't': 1, 'y': 1,
                                    'false': 0, 'no': 0, '0': 0, 'f': 0, 'n': 0}
                        # Apply mapping, keep non-matches as NaN
                        df[col] = df[col].astype(str).str.lower().map(map_dict)  # Will produce NaN for non-matches

                    # 2. Convert the entire column to numeric, coercing errors
                    numeric_col = pd.to_numeric(df[col], errors='coerce')

                    # 3. Fill NaN values (use 0 as a safe default for binary flags)
                    numeric_col = numeric_col.fillna(0)

                    # 4. Check if values are only 0 or 1 (or just one of them)
                    unique_vals = numeric_col.unique()
                    is_binary_numeric = all(val in [0, 1] for val in unique_vals)

                    if is_binary_numeric:
                        # 5. Convert to nullable Int8
                        df[col] = numeric_col.astype(pd.Int8Dtype())
                        logging.info(f"Successfully converted {col} to Int8 binary indicator.")
                    else:
                        logging.warning(f"Column {col} contains non-binary values after cleaning: {unique_vals}. Keeping as float32.")
                        df[col] = numeric_col.astype('float32')  # Fallback if not strictly 0/1

                except Exception as e:
                    logging.error(f"Error processing binary column {col}: {e}")
                    # Fallback: try converting to float32 directly after coerce/fillna
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')
                        logging.warning(f"Fallback: converted {col} to float32 due to error.")
                    except Exception as final_e:
                        logging.error(f"FATAL: Could not convert {col} even to float32: {final_e}. Leaving as is: {df[col].dtype}")

        # Handle numeric missing values
        for col in df.columns:
            if col not in categorical_cols and col not in string_cols and col not in boolean_cols:
                if any(pattern in col for pattern in float_patterns):
                    df[col] = handle_missing_values(df, col, 'mean')
                else:
                    df[col] = handle_missing_values(df, col, 'zero')

        # Handle date columns
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                logging.info(f"Converted {col} to datetime")

        # Special handling for fixture_id before other integer conversions
        if 'fixture_id' in df.columns:
            try:
                # First clean any potential string issues
                df['fixture_id'] = df['fixture_id'].astype(str).str.strip()
                # Convert to numeric, coerce any errors to NaN
                df['fixture_id'] = pd.to_numeric(df['fixture_id'], errors='coerce')
                # Fill NaN with -1 as a sentinel value
                df['fixture_id'] = df['fixture_id'].fillna(-1)
                # Convert to Int32
                df['fixture_id'] = df['fixture_id'].astype('Int32')
                logging.info("Successfully converted fixture_id to Int32")
            except Exception as e:
                logging.warning(f"Could not convert fixture_id to Int32: {e}")
                # Keep as string if conversion fails
                df['fixture_id'] = df['fixture_id'].astype(str)

        # Handle integer columns with validation
        for col in int8_cols:
            if col in df.columns:
                try:
                    non_null = df[col].dropna()
                    if non_null.apply(lambda x: float(x) if pd.notnull(x) else True).apply(lambda x: x == x and (isinstance(x, bool) or float(x).is_integer())).all():
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int8')
                        logging.info(f"Converted {col} to Int8")
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
                        logging.info(f"Kept {col} as float32 due to non-integer values")
                except Exception as e:
                    logging.warning(f"Error converting {col}: {e}")
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

        for col in int16_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int16')
                logging.info(f"Converted {col} to Int16")

        for col in int32_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int32')
                logging.info(f"Converted {col} to Int32")

        # Handle string columns
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype('string')
                logging.info(f"Converted {col} to string")

        # Handle remaining numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        # Initialize scalers
        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()
        # Apply appropriate scaling
        scaled_cols = []
        for col in numeric_cols:
            if col not in boolean_cols:  # Skip boolean columns
                if any(pattern in col for pattern in standardize_patterns):
                    df[col] = standard_scaler.fit_transform(df[[col]])
                    scaled_cols.append((col, 'standard'))
                    logging.info(f"Applied StandardScaler to {col}")
                elif any(pattern in col for pattern in minmax_patterns):
                    df[col] = minmax_scaler.fit_transform(df[[col]])
                    scaled_cols.append((col, 'minmax'))
                    logging.info(f"Applied MinMaxScaler to {col}")

        # Analyze variance and missing values
        zero_var_cols, near_zero_var_cols, missing_stats = analyze_column_variance(df)
        
        if zero_var_cols:
            logging.warning("\nColumns with zero variance (consider dropping):")
            for col, missing_pct in zero_var_cols:
                logging.warning(f"- {col} (missing: {missing_pct:.2f}%)")
        
        if near_zero_var_cols:
            logging.warning("\nColumns with near-zero variance (review for importance):")
            for col, std, missing_pct in near_zero_var_cols:
                logging.warning(f"- {col} (std: {std:.2e}, missing: {missing_pct:.2f}%)")

        # Count conversions
        dtype_counts = df.dtypes.value_counts()
        logging.info("\nFinal dtype distribution:")
        for dtype, count in dtype_counts.items():
            logging.info(f"{dtype}: {count} columns")

        # Log scaling summary
        logging.info("\nScaling summary:")
        logging.info(f"StandardScaler applied to: {len([c for c, t in scaled_cols if t == 'standard'])} columns")
        logging.info(f"MinMaxScaler applied to: {len([c for c, t in scaled_cols if t == 'minmax'])} columns")

    except Exception as e:
        logging.error(f"Error during optimization: {str(e)}")
        raise
    
    return df

def optimize_parquet_for_ml(input_path, output_path=None):
    """Optimize a football betting parquet file for ML training"""
    try:
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_ml_ready.parquet"
        output_path = Path(output_path)
        
        logging.info(f"Reading file: {input_path}")
        df = pd.read_parquet(input_path)
        initial_size = df.memory_usage(deep=True).sum() / 1024**2
        logging.info(f"Initial size: {initial_size:.2f} MB")
        initial_shape = df.shape
        logging.info(f"Initial shape: {initial_shape}")
        
        # Optimize dtypes and handle missing values
        logging.info("Optimizing for ML training...")
        df = optimize_football_dtypes(df)
        
        # Calculate memory savings
        optimized_size = df.memory_usage(deep=True).sum() / 1024**2
        logging.info(f"Optimized size: {optimized_size:.2f} MB")
        logging.info(f"Memory reduction: {((initial_size - optimized_size) / initial_size * 100):.1f}%")
        
        # Save metadata about the columns
        zero_var_cols, near_zero_var_cols, missing_stats = analyze_column_variance(df)
        metadata = {
            'zero_variance_columns': [col[0] for col in zero_var_cols],
            'near_zero_variance_columns': [col[0] for col in near_zero_var_cols],
            'missing_value_percentages': {k: float(v) for k, v in missing_stats.items()},
            'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
            'memory_usage_mb': float(optimized_size),
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns))
        }
        
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved column metadata to: {metadata_path}")
        
        # Save optimized file
        logging.info(f"Saving optimized file to: {output_path}")
        df.to_parquet(
            output_path,
            compression='snappy',
            index=False,
            engine='pyarrow',
            row_group_size=100000  # Good for typical ML batch sizes
        )
        
        logging.info("Optimization complete!")
        return True
        
    except Exception as e:
        logging.error(f"Error optimizing file: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/parquets/final_data_with_elo.parquet'
    output_file = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/parquets/final_data_with_elo_ml_ready.parquet'
    
    optimize_parquet_for_ml(input_file, output_file)