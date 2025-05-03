import pandas as pd
import numpy as np
from pathlib import Path

def optimize_dtypes(df):
    # Downcast numerical columns
    for col in df.select_dtypes(include='float64').columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
        
    for col in df.select_dtypes(include='int64').columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        
    # Convert object columns with few unique values to category
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() < df.shape[0] * 0.1:
            df[col] = df[col].astype('category')
            
    # Convert string columns where appropriate
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() < df.shape[0] * 0.1:
            df[col] = df[col].astype('category')
            
    return df

def optimize_parquet_file(input_path, output_path=None):
    """
    Optimize a parquet file for size and reading speed.
    
    Args:
        input_path (str): Path to input parquet file
        output_path (str, optional): Path to save optimized file. If None, will append '_optimized' to input name
    """
    try:
        # Validate input path
        input_path = Path(input_path)
        assert input_path.exists(), f"Input file not found: {input_path}"
        
        # Set output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_optimized.parquet"
        output_path = Path(output_path)
        
        # Read the parquet file
        print(f"Reading file: {input_path}")
        df = pd.read_parquet(input_path)
        
        # Get initial size
        initial_size = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Initial size: {initial_size:.2f} MB")
        
        # Optimize dtypes
        print("Optimizing data types...")
        df = optimize_dtypes(df)
        
        # Get optimized size
        optimized_size = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Optimized size: {optimized_size:.2f} MB")
        print(f"Size reduction: {((initial_size - optimized_size) / initial_size * 100):.1f}%")
        
        # Save optimized file with Snappy compression
        print(f"Saving optimized file to: {output_path}")
        df.to_parquet(
            output_path,
            compression='snappy',
            index=False,
            engine='pyarrow'
        )
        
        # Verify file size reduction
        final_size = output_path.stat().st_size / 1024**2
        print(f"Final file size: {final_size:.2f} MB")
        print(f"Total size reduction: {((initial_size - final_size) / initial_size * 100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Error optimizing file: {str(e)}")
        return False

if __name__ == "__main__":
    # Use your existing file path
    input_file = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/scripts/data/unified_data/csv_unified_full_cols.parquet'
    optimize_parquet_file(input_file)