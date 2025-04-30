import pandas as pd
import os

# Define the path to your Parquet file
file_path = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/data/unified_data/unified_matches.parquet'

try:
    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(file_path)

    # Display the first 5 rows to get a feel for the data
    print("First 5 rows of the data:")
    print(df.head())

    # Display information about the DataFrame, including column names, data types, and non-null counts
    print("\nDataFrame Info:")
    df.info()

    # Display summary statistics for numerical columns
    print("\nSummary Statistics:")
    print(df.describe())

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred while reading the Parquet file: {e}")
