import pandas as pd
import numpy as np
import gc # Garbage Collector
import os
import time

# --- Configuration ---
parquet_file_path = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/scripts/data/unified_data/csv_unified_raw_optimized.parquet'
optimized_file_path = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/scripts/data/unified_data/csv_unified_raw_optimized2.parquet'
save_optimized = True # Set to True to save the optimized file
fill_nan_strategy = 'mean' # Options: 'mean', 'median', 'zero', -1 (or any other number)

# --- Helper Function for Cleaning Numeric Features ---
def clean_numeric_column(series, strategy='mean', preserve_integers=True):
    """
    Cleans a pandas Series intended to be numeric for ML:
        1. Converts to numeric, coercing errors to NaN.
        2. Handles infinite values by replacing them with NaN.
        3. Fills NaN values based on the strategy.
        4. Preserves integer types when appropriate.
    """
    # Convert numpy array to pandas Series if needed
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    # Store if the original was integer type
    was_integer = pd.api.types.is_integer_dtype(series.dtype)
    
    # First convert to float64 to avoid overflow issues
    series = pd.to_numeric(series, errors='coerce').astype('float64')

    # Handle infinities
    series = series.replace([np.inf, -np.inf], np.nan)

    # Calculate fill value
    if pd.isna(series).all():
        fill_value = 0 if strategy in ['mean', 'median', 'zero'] else strategy
    elif strategy == 'mean':
        fill_value = series.mean()
    elif strategy == 'median':
        fill_value = series.median()
    elif strategy == 'zero':
        fill_value = 0
    elif isinstance(strategy, (int, float)):
        fill_value = strategy

    # Fill NaN values
    series = series.fillna(fill_value)

    # Try to preserve integer type if original was integer and all values are whole numbers
    if preserve_integers and was_integer:
        try:
            # Check if all values can be safely converted to integers
            if series.apply(lambda x: abs(x - round(x)) < 1e-10).all():
                # Round to remove any floating-point imprecision
                series = series.round()
                # Check if values are within Int16 range
                if (series >= np.iinfo(np.int16).min).all() and (series <= np.iinfo(np.int16).max).all():
                    return series.astype(pd.Int16Dtype())
                else:
                    print(f"Warning: Values in column exceed Int16 range, converting to float32")
        except Exception as e:
            print(f"Warning: Could not convert to Int16, using float32 instead: {e}")
    
    # Default to float32 for other numeric columns
    return series.astype(np.float32)

# --- Main Data Type Optimization Function ---
def optimize_dtypes_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes the data types of a pandas DataFrame after cleaning,
    preparing it for ML models and memory efficiency.
    """
    print(f"\nStarting dtype optimization...")
    print(f"Memory usage before optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    start_time = time.time()

    # 1. Categorical Columns with improved handling
    categorical_cols = [
        'FTR', 'HTR', 'LeagueID', 'Country', 'Res', 'LeagueName',
        'HomeTeam', 'AwayTeam', 'Season'
    ]
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    for col in categorical_cols:
        nunique = df[col].nunique()
        if nunique < (len(df) * 0.6):  # Less than 60% unique values
            print(f"Converting '{col}' to category (cardinality: {nunique})")
            df[col] = df[col].astype('category')
        else:
            print(f"Keeping '{col}' as object due to high cardinality: {nunique} unique values")

    # 2. Identifier Columns (as strings)
    identifier_cols = ['MatchID']
    for col in identifier_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # 3. Count/Integer Columns (with strict integer checking)
    int_cols = [
        'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HR', 'AR', 'HY', 'AY',
        'HomeFouls', 'HomeOffsides', 'HomeRedCards', 'HomeSaves', 'HomeShots',
        'HomeShotsInsideBox', 'HomeShotsOffTarget', 'HomeShotsOutsideBox',
        'HomeShotsTarget', 'HomeYellowCards',
        'AwayFouls', 'AwayOffsides', 'AwayRedCards', 'AwaySaves', 'AwayShots',
        'AwayShotsInsideBox', 'AwayShotsOffTarget', 'AwayShotsOutsideBox',
        'AwayShotsTarget', 'AwayYellowCards'
    ]
    
    # Add derived count columns
    count_patterns = ['_Count_', '_FormPoints_']
    for pattern in count_patterns:
        int_cols.extend([col for col in df.columns if pattern in col])
    
    int_cols = list(set([col for col in int_cols if col in df.columns]))
    for col in int_cols:
        try:
            if df[col].dropna().apply(lambda x: float(x).is_integer()).all():
                df[col] = df[col].astype(pd.Int16Dtype())
                print(f"Converted '{col}' to Int16")
            else:
                print(f"Warning: Count column '{col}' contains non-integer values, keeping as float32")
                df[col] = df[col].astype(np.float32)
        except Exception as e:
            print(f"Could not convert '{col}' to Int16: {e}")

    # 4. DateTime Columns
    datetime_cols = ['Date']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 5. Check for zero-variance columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    zero_var_cols = []
    near_zero_var_cols = []
    for col in numeric_cols:
        std = df[col].std()
        if std == 0:
            zero_var_cols.append(col)
        elif std < 1e-6:
            near_zero_var_cols.append(col)
    
    if zero_var_cols:
        print("\nWarning: Found columns with zero variance (consider dropping):")
        for col in zero_var_cols:
            print(f"- {col}")
    
    if near_zero_var_cols:
        print("\nWarning: Found columns with near-zero variance:")
        for col in near_zero_var_cols:
            print(f"- {col} (std: {df[col].std():.2e})")

    # 6. Convert remaining numeric columns to float32
    remaining_numeric = [col for col in numeric_cols if col not in int_cols + zero_var_cols]
    for col in remaining_numeric:
        try:
            df[col] = df[col].astype(np.float32)
        except Exception as e:
            print(f"Could not convert '{col}' to float32: {e}")

    gc.collect()
    end_time = time.time()
    print(f"\nOptimization process took: {end_time - start_time:.2f} seconds")
    print(f"Optimized DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df

# --- Main Execution Block ---
if __name__ == "__main__":

    if not os.path.exists(parquet_file_path):
        print(f"Error: File not found at '{parquet_file_path}'")
        print("Please update the 'parquet_file_path' variable in the script.")
    else:
        print(f"Loading data from: {parquet_file_path}...")
        try:
             df = pd.read_parquet(parquet_file_path)
             print(f"Initial DataFrame shape: {df.shape}")
             print(f"Initial memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

             # --- Step 1: Clean Numeric Features for ML ---
             print("\n--- Step 1: Cleaning numeric columns ---")
             numeric_cols_to_clean = df.select_dtypes(include=np.number).columns
             print(f"Found {len(numeric_cols_to_clean)} potentially numeric columns to clean.")

             cleaning_start_time = time.time()
             for col in numeric_cols_to_clean:
                 # Apply cleaning function
                 df[col] = clean_numeric_column(df[col], strategy=fill_nan_strategy)
                 # Optional: check if NaNs remain (shouldn't if strategy handles all-NaN cols)
                 # if df[col].isnull().any():
                 #    print(f"Warning: NaNs still present in '{col}' after cleaning with strategy '{fill_nan_strategy}'.")
             cleaning_end_time = time.time()
             print(f"Numeric cleaning took: {cleaning_end_time - cleaning_start_time:.2f} seconds")
             print(f"Memory usage after cleaning: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


             # --- Step 2: Optimize Data Types ---
             print("\n--- Step 2: Optimizing data types ---")
             df_optimized = optimize_dtypes_for_ml(df)

             # --- Final Verification ---
             print("\n--- Final Optimized DataFrame Info: ---")
             # Check dtypes and non-null counts
             df_optimized.info(verbose=True, show_counts=True)

             # Verify no NaNs/Infs in numeric columns (critical for most ML)
             numeric_cols_final = df_optimized.select_dtypes(include=np.number).columns
             nan_check = df_optimized[numeric_cols_final].isnull().sum().sum()
             inf_check = np.isinf(df_optimized[numeric_cols_final]).sum().sum()

             if nan_check == 0 and inf_check == 0:
                 print("\nVerification successful: No NaN or Inf values found in numeric columns.")
             else:
                 print(f"\nVerification WARNING: Found {nan_check} NaN and {inf_check} Inf values in numeric columns. Review cleaning step.")
                 # Optional: print columns with issues
                 # print("NaN counts:\n", df_optimized[numeric_cols_final].isnull().sum()[df_optimized[numeric_cols_final].isnull().sum() > 0])
                 # print("Inf counts:\n", np.isinf(df_optimized[numeric_cols_final]).sum()[np.isinf(df_optimized[numeric_cols_final]).sum() > 0])


             # --- Step 3: Save Optimized File (Optional) ---
             if save_optimized:
                 print(f"\n--- Step 3: Saving optimized data to: {optimized_file_path} ---")
                 try:
                     df_optimized.to_parquet(optimized_file_path, index=False)
                     print("Optimized file saved successfully.")
                 except Exception as e:
                     print(f"Error saving optimized file: {e}")
             else:
                 print("\nOptimized file not saved (set 'save_optimized = True' to save).")

        except Exception as e:
            print(f"\nAn error occurred during processing: {e}")
            import traceback
            traceback.print_exc()