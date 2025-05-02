import pandas as pd

# Define the path to your Parquet file
file_path = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/data/unified_data/csv_unified_full_cols.parquet'

try:
    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(file_path)
    
    # Display DataFrame info
    print("--- DataFrame Info ---")
    df.info(verbose=True, show_counts=True, memory_usage='deep')
    
    # Display info for the first row
    print("--- First Row Example ---")
    if not df.empty:
        print(df.iloc[0].to_string())
    else:
        print("DataFrame is empty.")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except KeyError as e:
     print(f"KeyError: {e}. 'MatchID' column might be missing or named differently.")
     if 'df' in locals(): print(f"Available columns: {df.columns.tolist()}")
except Exception as e:
    print(f"An error occurred while reading or processing the Parquet file: {e}")
