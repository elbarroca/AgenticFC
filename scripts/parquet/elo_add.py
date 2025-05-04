import pandas as pd
import numpy as np
from pymongo import MongoClient
from pathlib import Path
import logging
import time
import os

# --- Configuration ---

# MongoDB Connection Details
MONGO_URI = "mongodb://admin888:admin888@127.0.0.1:27017/?authSource=admin"  # Replace with your MongoDB connection string if different
MONGO_DB_NAME = "agenticfc"      # Replace with your database name
MONGO_COLLECTION_NAME = "matches" # Replace with the collection containing ELO data

# Parquet File Paths
PARQUET_INPUT_PATH = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/parquets/final_data.parquet'
PARQUET_OUTPUT_PATH = '/Users/barroca888/Downloads/Agenticfc/AgenticFC888/models/parquets/final_data_with_elo.parquet' # Output file name

# Column Names for Joining and ELO Data
PARQUET_ID_COLUMN = 'MatchID'  # Column in Parquet file used for matching
MONGO_ID_COLUMN = 'fixture_id'    # MongoDB ID field
PARQUET_HOME_ELO_COLUMN = 'HomeTeamELO'  # Parquet column for home team ELO
PARQUET_AWAY_ELO_COLUMN = 'AwayTeamELO'  # Parquet column for away team ELO
MONGO_HOME_ELO_COLUMN = 'home_team_elo'  # MongoDB field for home team ELO
MONGO_AWAY_ELO_COLUMN = 'away_team_elo'  # MongoDB field for away team ELO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Script ---

# Add debug function
def print_sample_data(df, collection):
    """Debug function to print sample data from both sources"""
    logging.info("=== DEBUG INFO ===")
    # Print sample from Parquet
    logging.info("Parquet Sample IDs:")
    logging.info(df[PARQUET_ID_COLUMN].head().tolist())
    
    # Print sample from MongoDB
    sample = collection.find_one()
    if sample:
        logging.info("MongoDB Sample Document:")
        logging.info(f"fixture_id: {sample.get('fixture_id')}")
        logging.info(f"home_team_elo: {sample.get('home_team_elo')}")
        logging.info(f"away_team_elo: {sample.get('away_team_elo')}")

def add_elo_to_parquet(mongo_uri, db_name, collection_name, parquet_in, parquet_out,
                       parquet_id_col, mongo_id_col, mongo_home_elo_col, mongo_away_elo_col):
    try:
        # 1. Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        logging.info("MongoDB connection successful.")

        # 2. Get MongoDB data
        query = {
            mongo_home_elo_col: {"$exists": True},
            mongo_away_elo_col: {"$exists": True}
        }
        projection = {
            "_id": 0,
            mongo_id_col: 1,
            mongo_home_elo_col: 1,
            mongo_away_elo_col: 1,
            "date_str": 1,
            "home_team_name": 1,
            "away_team_name": 1
        }
        
        elo_matches = list(collection.find(query, projection))
        df_elo = pd.DataFrame(elo_matches)
        
        # 3. Read Parquet file
        df_parquet = pd.read_parquet(parquet_in)
        
        # Debug information before matching
        logging.info("\n=== Debug Information ===")
        logging.info(f"MongoDB date range: {df_elo['date_str'].min()} to {df_elo['date_str'].max()}")
        logging.info(f"Parquet date range: {df_parquet['Date'].min()} to {df_parquet['Date'].max()}")
        
        # Sample of team names from both sources
        logging.info("\nMongoDB team names sample:")
        mongo_teams = set(df_elo['home_team_name'].unique()) | set(df_elo['away_team_name'].unique())
        logging.info(f"Total unique teams in MongoDB: {len(mongo_teams)}")
        logging.info(f"Sample teams: {list(mongo_teams)[:5]}")
        
        parquet_teams = set(df_parquet['HomeTeam'].unique()) | set(df_parquet['AwayTeam'].unique())
        logging.info(f"\nTotal unique teams in Parquet: {len(parquet_teams)}")
        logging.info(f"Sample teams: {list(parquet_teams)[:5]}")
        
        # Find team name differences
        teams_only_in_mongo = mongo_teams - parquet_teams
        teams_only_in_parquet = parquet_teams - mongo_teams
        
        logging.info("\nTeam name mismatches:")
        logging.info(f"Teams only in MongoDB (first 5): {list(teams_only_in_mongo)[:5]}")
        logging.info(f"Teams only in Parquet (first 5): {list(teams_only_in_parquet)[:5]}")

        # 4. Prepare data for matching
        # Convert date formats to match
        df_parquet['Date'] = pd.to_datetime(df_parquet['Date']).dt.strftime('%Y-%m-%d')
        
        # Create match keys
        df_elo['match_key'] = df_elo['date_str'] + '_' + df_elo['home_team_name'] + '_' + df_elo['away_team_name']
        df_parquet['match_key'] = df_parquet['Date'] + '_' + df_parquet['HomeTeam'] + '_' + df_parquet['AwayTeam']
        
        # Debug match keys
        logging.info("\nSample match keys from MongoDB:")
        logging.info(df_elo['match_key'].head().tolist())
        logging.info("\nSample match keys from Parquet:")
        logging.info(df_parquet['match_key'].head().tolist())

        # 5. Merge and update
        df_merged = pd.merge(
            df_parquet,
            df_elo[[mongo_id_col, mongo_home_elo_col, mongo_away_elo_col, 'match_key']],
            on='match_key',
            how='left'
        )

        # Update ELO columns
        df_merged[PARQUET_HOME_ELO_COLUMN] = df_merged[mongo_home_elo_col]
        df_merged[PARQUET_AWAY_ELO_COLUMN] = df_merged[mongo_away_elo_col]

        # Drop temporary columns
        df_merged = df_merged.drop(['match_key', mongo_home_elo_col, mongo_away_elo_col], axis=1)

        # Log match statistics
        total_matches = len(df_parquet)
        matched_matches = df_merged[df_merged[PARQUET_HOME_ELO_COLUMN].notna()].shape[0]
        match_rate = (matched_matches / total_matches) * 100
        
        logging.info("\n=== Match Statistics ===")
        logging.info(f"Total matches in Parquet: {total_matches}")
        logging.info(f"Matches with ELO data in MongoDB: {len(df_elo)}")
        logging.info(f"Successfully matched matches: {matched_matches}")
        logging.info(f"Match rate: {match_rate:.2f}%")

        # Sample of unmatched games
        unmatched = df_merged[df_merged[PARQUET_HOME_ELO_COLUMN].isna()].head()
        logging.info("\nSample of unmatched games:")
        for _, row in unmatched.iterrows():
            logging.info(f"Date: {row['Date']}, Home: {row['HomeTeam']}, Away: {row['AwayTeam']}")

        # Save the result
        df_merged.to_parquet(parquet_out, index=False)
        logging.info(f"\nSaved updated data to {parquet_out}")

        return True

    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        return False
    finally:
        if client:
            client.close()


# --- Run the script ---
if __name__ == "__main__":
    success = add_elo_to_parquet(
        mongo_uri=MONGO_URI,
        db_name=MONGO_DB_NAME,
        collection_name=MONGO_COLLECTION_NAME,
        parquet_in=PARQUET_INPUT_PATH,
        parquet_out=PARQUET_OUTPUT_PATH,
        parquet_id_col=PARQUET_ID_COLUMN,
        mongo_id_col=MONGO_ID_COLUMN,
        mongo_home_elo_col=MONGO_HOME_ELO_COLUMN,
        mongo_away_elo_col=MONGO_AWAY_ELO_COLUMN
    )

    if success:
        logging.info("Script finished successfully.")
    else:
        logging.error("Script finished with errors.")