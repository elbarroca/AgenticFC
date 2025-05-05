import pandas as pd
import random

# Define the path to your Parquet file
file_path = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/parquets/mongo_normalized.parquet'

try:
    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(file_path)
    
    # Display DataFrame info
    print("--- DataFrame Info ---")
    df.info(verbose=True, show_counts=True, memory_usage='deep')
    
    # Display raw data count
    print(f"--- Raw Data Count ---\nTotal rows: {len(df)}")
    
    # Display an example of the data
    print("--- Example Data ---")
    print(df.head(1).to_string(index=False))
    
    # Display column fulfillment with number of filled entries and a random example
    print("--- Column Fulfillment ---")
    if not df.empty:
        if not df.empty:
            for column in df.columns:
                filled_count = df[column].notna().sum()
                fill_rate = (filled_count / len(df)) * 100
                if filled_count > 0:
                    random_example = df[column].dropna().sample(n=1, random_state=random.randint(0, 1000)).iloc[0]
                else:
                    random_example = "No data"
                print(f"{column}: {filled_count} filled ({fill_rate:.2f}%), Example: {random_example}")
            
            # Create a dictionary to count column types
            column_types_count = {}
            for column in df.columns:
                col_type = df[column].dtype
                column_types_count[col_type] = column_types_count.get(col_type, 0) + 1
            
            # Display the count of each column type
            print("--- Column Types Count ---")
            for col_type, count in column_types_count.items():
                print(f"{count} columns of type {col_type}")
        else:
            print("DataFrame is empty.")
    
        # Display additional statistics
        print("--- Additional Statistics ---")
        total_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"Total size of DataFrame: {total_size_mb:.2f} MB")
        print(f"Number of columns: {len(df.columns)}")
        print(f"Number of data points: {df.notna().sum().sum()}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except KeyError as e:
     print(f"KeyError: {e}. 'MatchID' column might be missing or named differently.")
     if 'df' in locals(): print(f"Available columns: {df.columns.tolist()}")
except Exception as e:
    print(f"An error occurred while reading or processing the Parquet file: {e}")
