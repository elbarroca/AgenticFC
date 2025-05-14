#!/usr/bin/env python3
"""
Script to merge multiple Parquet files from specified directories
into a single output Parquet file.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
# numpy is implicitly used by pandas, but explicit import for type hinting if needed
# import numpy as np

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def find_parquet_files_in_directory(directory: Path) -> List[Path]:
    """
    Finds all .parquet files in a given directory.

    Args:
        directory: The directory path to search.

    Returns:
        A list of Path objects for all .parquet files found.
        Returns an empty list if the directory does not exist or is not a directory.
    """
    if not directory.is_dir():
        logger.warning(f"Input path is not a directory or does not exist: {directory}")
        return []
    
    parquet_files = sorted(list(directory.glob("*.parquet"))) # Sort for consistent order (optional, but good for debugging)
    logger.info(f"Found {len(parquet_files)} .parquet files in {directory}.")
    return parquet_files

def load_and_combine_parquet_files(file_paths: List[Path]) -> Optional[pd.DataFrame]:
    """
    Loads multiple parquet files and concatenates them into a single Pandas DataFrame.

    Args:
        file_paths: A list of Path objects pointing to .parquet files.

    Returns:
        A Pandas DataFrame containing the combined data, or None if no files
        were loaded or an error occurred during concatenation.
    """
    if not file_paths:
        logger.warning("No parquet files provided to load and combine.")
        return None

    dataframes: List[pd.DataFrame] = []
    for file_path in file_paths:
        logger.debug(f"Attempting to load data from: {file_path}")
        try:
            df = pd.read_parquet(file_path)
            
            assert isinstance(df, pd.DataFrame), \
                f"Expected a Pandas DataFrame from {file_path}, but received type {type(df)}."
            
            if df.empty:
                logger.info(f"File {file_path} is empty. Skipping.")
                continue
            
            dataframes.append(df)
            logger.debug(f"Successfully loaded {df.shape[0]} rows from {file_path}.")

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}. Skipping.")
        except pd.errors.EmptyDataError: # PyArrow might raise different errors for malformed files
            logger.error(f"Empty or malformed Parquet file (EmptyDataError): {file_path}. Skipping.")
        except Exception as e: # Catch other potential read errors
            logger.error(f"Error loading or validating file {file_path}: {e}", exc_info=False) # Set exc_info=True for full stack trace
            # Depending on desired robustness, you might want to re-raise critical errors or sys.exit()
            logger.warning(f"Skipping file {file_path} due to load error.")


    if not dataframes:
        logger.warning("No DataFrames were successfully loaded. Cannot combine.")
        return None

    logger.info(f"Successfully loaded {len(dataframes)} DataFrames. Starting concatenation...")
    try:
        # ignore_index=True is crucial to re-index the combined DataFrame
        combined_df = pd.concat(dataframes, ignore_index=True)
    except Exception as e: # Catch potential errors during concat (e.g., out of memory)
        logger.critical(f"Error during DataFrame concatenation: {e}", exc_info=True)
        return None
        
    logger.info(f"Concatenated DataFrame shape: {combined_df.shape}")
    assert not combined_df.empty, "Concatenated DataFrame is empty after processing." # Should not happen if dataframes list was not empty
    
    # Deduplication based on identifying columns
    # We're assuming these columns uniquely identify a parlay entry
    identifying_cols = ['parlay_date', 'market_combination', 'leg1_match_id']
    if all(col in combined_df.columns for col in identifying_cols):
        original_rows = len(combined_df)
        combined_df.drop_duplicates(subset=identifying_cols, keep='first', inplace=True)
        logger.info(f"Removed {original_rows - len(combined_df)} duplicate rows based on {identifying_cols}.")
    else:
        logger.warning(f"Could not perform deduplication. One or more identifying columns not present in data.")
        # Try to show available columns to help troubleshoot
        logger.debug(f"Available columns: {list(combined_df.columns)}")

    return combined_df

# --- Main Script Logic ---
def merge_parquets_main(
    input_directory_paths_str: List[str],
    output_file_path_str: str
) -> None:
    """
    Main function to find, load, combine, and save parquet files.
    """
    process_start_time = pd.Timestamp.now()
    logger.info("--- Starting Parquet File Merge Script ---")

    # Validate and convert input directory paths
    input_directories: List[Path] = []
    for dir_str in input_directory_paths_str:
        path = Path(dir_str)
        assert path.exists(), f"Input directory does not exist: {path}"
        assert path.is_dir(), f"Input path is not a directory: {path}"
        input_directories.append(path)
    
    # Prepare output file path and ensure parent directory exists
    output_file_path = Path(output_file_path_str)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_file_path}")

    # Collect all parquet file paths from all specified input directories
    all_parquet_file_paths: List[Path] = []
    for directory in input_directories:
        all_parquet_file_paths.extend(find_parquet_files_in_directory(directory))

    if not all_parquet_file_paths:
        logger.warning("No .parquet files found in any of the specified input directories. Nothing to merge. Exiting.")
        return

    logger.info(f"Total of {len(all_parquet_file_paths)} .parquet files found to process.")

    # Load and combine the data
    combined_dataframe = load_and_combine_parquet_files(all_parquet_file_paths)

    # Save the combined data if successful
    if combined_dataframe is not None and not combined_dataframe.empty:
        try:
            logger.info(f"Saving merged DataFrame with shape {combined_dataframe.shape} to {output_file_path}...")
            # index=False is important as we've already re-indexed during concat
            combined_dataframe.to_parquet(output_file_path, index=False, engine='pyarrow')
            logger.info(f"Successfully saved merged DataFrame to {output_file_path}.")
        except Exception as e:
            logger.critical(f"Failed to save merged DataFrame to {output_file_path}: {e}", exc_info=True)
    elif combined_dataframe is None:
        logger.error("Merging process failed. No DataFrame was returned for saving.")
    else: # combined_dataframe is not None but empty
        logger.warning("Merged DataFrame is empty. Nothing to save.")

    process_end_time = pd.Timestamp.now()
    duration = (process_end_time - process_start_time).total_seconds()
    logger.info(f"--- Parquet File Merge Script Finished in {duration:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple Parquet files from specified directories into a single Parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Use nargs='+' to accept one or more input directories
    parser.add_argument(
        "-i", "--input_dirs",
        nargs='+',
        required=True,
        help="List of input directories containing .parquet files to merge. "
             "Example: /path/to/dir1 /path/to/dir2",
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        required=True,
        help="Path for the output merged .parquet file. Example: /path/to/merged_output.parquet",
    )

    args = parser.parse_args()

    # The paths you provided:
    # Path 1: /Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/data/outputs/predictions/parlay_outputs_V3_sampled/checkpoints/
    # Path 2: /Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/data/outputs/predictions/parlay_outputs_V2/checkpoint_parlays/
    
    # You would run this script from your terminal like this:
    # python your_script_name.py \
    #   -i /Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/data/outputs/predictions/parlay_outputs_V3_sampled/checkpoints/ \
    #      /Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/data/outputs/predictions/parlay_outputs_V2/checkpoint_parlays/ \
    #   -o /Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/data/outputs/predictions/merged_all_parlay_results.parquet

    merge_parquets_main(
        input_directory_paths_str=args.input_dirs,
        output_file_path_str=args.output_file
    )