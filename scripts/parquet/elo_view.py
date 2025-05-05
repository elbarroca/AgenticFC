import pandas as pd
import os
# --- Configuration ---
PARQUET_FILE_PATH = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/parquets/final_data_with_elo.parquet'

def view_parquet_data():
    print(f"\nChecking Parquet file: {PARQUET_FILE_PATH}")
    
    if not os.path.exists(PARQUET_FILE_PATH):
        print("Error: File not found!")
        return
    
    try:
        # Read the full parquet file
        df = pd.read_parquet(PARQUET_FILE_PATH)
        
        print("\n=== File Information ===")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print("\nColumns:", df.columns.tolist())
        
        # Check for ELO columns
        elo_columns = [col for col in df.columns if 'elo' in col.lower()]
        print("\n=== ELO Columns ===")
        if elo_columns:
            print("Found ELO columns:", elo_columns)
            print("\nELO Data Sample:")
            print(df[elo_columns].head())
            
            # Show statistics for ELO columns
            print("\nELO Statistics:")
            print(df[elo_columns].describe())
            
            # Count non-null values
            print("\nNon-null counts:")
            print(df[elo_columns].count())
        else:
            print("No ELO columns found!")
            
        # Show sample of the data
        print("\n=== Data Sample ===")
        print(df.head())
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    view_parquet_data()