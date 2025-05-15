from typing import Optional
import pandas as pd
from pathlib import Path
import argparse

def inspect_parquet_columns(parquet_path_str: str, search_term: Optional[str] = None):
    """
    Loads a Parquet file and prints its columns.
    If a search_term is provided, it prints columns containing that term.

    Args:
        parquet_path_str (str): Path to the Parquet file.
        search_term (Optional[str]): A term to search for within column names (case-insensitive).
    """
    parquet_path = Path(parquet_path_str)

    if not parquet_path.exists():
        print(f"ERROR: File not found at {parquet_path}")
        return
    if not parquet_path.is_file() or parquet_path.suffix.lower() != '.parquet':
        print(f"ERROR: Path is not a Parquet file: {parquet_path}")
        return

    print(f"--- Inspecting Parquet File: {parquet_path.name} ---")
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Successfully loaded. Shape: {df.shape}")
        
        all_columns = df.columns.tolist()
        print(f"\nTotal columns found: {len(all_columns)}")
        
        if len(all_columns) < 200: # Print all if not too many
            print("\nAll Columns:")
            for col in all_columns:
                print(f"  - {col}")
        else:
            print("\nFirst 50 Columns:")
            for col in all_columns[:50]:
                print(f"  - {col}")
            print("  ...")
            print("Last 50 Columns:")
            for col in all_columns[-50:]:
                print(f"  - {col}")
            print("  ...")


        if search_term:
            print(f"\n--- Columns containing '{search_term}' (case-insensitive) ---")
            found_columns = [col for col in all_columns if search_term.lower() in col.lower()]
            if found_columns:
                for col in found_columns:
                    print(f"  - {col}")
                print(f"Found {len(found_columns)} columns matching '{search_term}'.")
            else:
                print(f"No columns found containing '{search_term}'.")
        
        # Specifically check for common Monte Carlo prefixes based on our discussion
        print("\n--- Checking for common Monte Carlo prefixes ---")
        mc_prefixes_to_check = [
            "monte_carlo_enhanced_pca_without_odds", # Expected by train_stacker.py
            "monte_carlo_pca_without_odds",          # A possible simpler name
            "montecarlo_pca_without_odds",           # Variation
            "mc_pca_without_odds",                   # Shorter variation
            "monte_carlo_enhanced_"                  # General prefix from MC class output
        ]
        
        for prefix_check in mc_prefixes_to_check:
            mc_cols = [col for col in all_columns if col.startswith(prefix_check)]
            if mc_cols:
                print(f"\nFound columns starting with '{prefix_check}':")
                for i, col in enumerate(mc_cols):
                    print(f"  - {col}")
                    if i >= 9 and len(mc_cols) > 10 : # Print first 10 then count
                        print(f"  ... and {len(mc_cols) - 10} more.")
                        break
            else:
                print(f"No columns found starting with '{prefix_check}'.")


    except Exception as e:
        print(f"ERROR loading or processing Parquet file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect columns of a Parquet file, optionally searching for a term.")
    parser.add_argument("parquet_path", type=str, help="Full path to the Parquet file to inspect.")
    parser.add_argument("--search", type=str, default=None, help="Optional: Case-insensitive term to search for in column names.")
    
    args = parser.parse_args()
    
    inspect_parquet_columns(args.parquet_path, args.search)