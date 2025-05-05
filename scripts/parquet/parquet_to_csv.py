import pandas as pd
import os

# Define the fixed input and output paths
FIXED_INPUT_PATH = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/parquets/final_data_with_elo.parquet'
OUTPUT_CSV_PATH = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/csvs/final_data_with_elo.csv'

def parquet_to_csv() -> None:
    """
    Reads a Parquet file and writes its content to a CSV file with columns sorted alphabetically.
    """
    input_path = FIXED_INPUT_PATH
    output_path = OUTPUT_CSV_PATH

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- Read Parquet ---
    print("Reading Parquet file into memory...")
    df = pd.read_parquet(input_path, engine='pyarrow')
    print(f"Read {len(df)} rows and {len(df.columns)} columns.")

    # Print the first raw value of each column in a vertical format
    print("First raw value of each column:")
    for column in df.columns:
        print(f"{column}: {df[column].iloc[0]}")

    # Sort columns alphabetically
    df = df.reindex(sorted(df.columns), axis=1)

    # Transform the DataFrame to have each row as a column name and its first value
    transformed_df = pd.DataFrame({
        'Column': df.columns,
        'Value': [f"({col}: {df[col].iloc[0]})" for col in df.columns]
    })

    # --- Write to CSV ---
    print(f"Writing transformed DataFrame to CSV at {output_path}...")
    transformed_df.to_csv(output_path, index=False, header=False)
    print("CSV file has been created successfully in the desired format.")

if __name__ == "__main__":
    parquet_to_csv()