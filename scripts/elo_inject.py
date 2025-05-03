import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
import soccerdata as sd
import pandas as pd
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv() # Load environment variables from .env file

# Ensure MONGO_URI, DB_NAME, and COLLECTION_NAME are set in your .env file
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "agenticfc")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "matches")

# --- Team Name Mapping ---
# Add mappings from your MongoDB team names to ClubElo official names
# Example: TEAM_NAME_MAPPING = { "Man Utd": "Man United", "Inter": "Internazionale" }
# Populate this based on the errors you see in the logs.
TEAM_NAME_MAPPING = {
    "FC OSS": "TOP Oss", # Example: Found 'TOP Oss' on clubelo.com for FC Oss
    "FC Copenhagen": "FC København", # Example: Found 'FC København'
    "Randers FC": "Randers", # Example: Found 'Randers'
    "FC Heidenheim": "Heidenheim", # Example: Found 'Heidenheim'
    "Neftchi Baku": "Neftchi", # Example: Found 'Neftchi'
    "Vikingur Reykjavik": "Víkingur", # Example: Found 'Víkingur'
    "Lech Poznan": "Lech", # Example: Found 'Lech'
    # Add more mappings here as needed based on logs
}

# --- Helper Functions ---

# Cache for team Elo histories to minimize API calls via soccerdata's cache
TEAM_ELO_HISTORIES_CACHE = {}
FAILED_ELO_FETCH_TEAMS = set() # Store original names that failed

def fetch_team_elo_history(team_name: str, elo_reader: sd.ClubElo):
    """
    Fetches and caches Elo history for a team, applying name mapping.
    Now with better error handling and fallback options.
    """
    original_team_name = team_name

    # Early return if we already know about this team
    if original_team_name in TEAM_ELO_HISTORIES_CACHE:
        return TEAM_ELO_HISTORIES_CACHE[original_team_name]
    if original_team_name in FAILED_ELO_FETCH_TEAMS:
        return None

    # Try different name variations
    name_variations = [
        TEAM_NAME_MAPPING.get(original_team_name, original_team_name),  # First try mapped name
        original_team_name.replace(" & ", " "),  # Remove ampersands
        original_team_name.split(" & ")[0].strip(),  # Take first team name before ampersand
        original_team_name.replace("FC ", "").replace(" FC", ""),  # Try without "FC"
    ]

    for name_variant in name_variations:
        if not name_variant:
            continue
            
        logging.debug(f"Trying variant '{name_variant}' for team '{original_team_name}'")
        try:
            history_df = elo_reader.read_team_history(name_variant)
            if history_df is not None and not history_df.empty:
                history_df.index = pd.to_datetime(history_df.index)
                history_df['to'] = pd.to_datetime(history_df['to'])
                history_df.index = history_df.index.tz_localize(None)
                history_df['to'] = history_df['to'].dt.tz_localize(None)
                TEAM_ELO_HISTORIES_CACHE[original_team_name] = history_df
                logging.info(f"Successfully found Elo data for '{original_team_name}' using variant '{name_variant}'")
                return history_df
        except ValueError as ve:
            continue  # Try next variant
        except Exception as e:
            logging.debug(f"Error trying variant '{name_variant}' for '{original_team_name}': {e}")
            continue

    # If we get here, all variants failed
    logging.warning(f"Could not find Elo data for team '{original_team_name}' after trying all variants")
    FAILED_ELO_FETCH_TEAMS.add(original_team_name)
    return None

def get_elo_on_date(team_name: str, match_date: datetime, elo_reader: sd.ClubElo):
    """
    Finds the ClubElo rating for a team on a specific date using cached history.
    Uses fetch_team_elo_history which handles name mapping.

    Args:
        team_name (str): The original name of the team from MongoDB.
        match_date (datetime): The date of the match (timezone-aware UTC).
        elo_reader (sd.ClubElo): Instance of the ClubElo reader.

    Returns:
        Optional[int]: The Elo rating as an integer, or None if not found/error.
    """
    # Fetch history (handles mapping and caching internally)
    history_df = fetch_team_elo_history(team_name, elo_reader)

    if history_df is None:
        return None # Failure already logged by fetch function

    # Convert match_date to timezone-naive for comparison
    match_date_naive = match_date.replace(tzinfo=None)

    try:
        # Find the row where the match date falls within the [from, to] interval
        relevant_elo_row = history_df[
            (history_df.index <= match_date_naive) & (history_df['to'] >= match_date_naive)
        ]

        if not relevant_elo_row.empty:
            elo_value = relevant_elo_row.iloc[0]['elo']
            return int(round(elo_value))
        else:
            # Fallback: Find the latest Elo rating *before* the match date
            past_elos = history_df[history_df.index < match_date_naive]
            if not past_elos.empty:
                latest_past_elo_row = past_elos.iloc[-1]
                # Check if the 'to' date is reasonably close (e.g., within 90 days)
                if pd.notna(latest_past_elo_row['to']) and match_date_naive - latest_past_elo_row['to'] <= timedelta(days=90):
                     elo_value = latest_past_elo_row['elo']
                     logging.debug(f"Using fallback Elo (latest before {match_date_naive.date()}) for {team_name}: {int(round(elo_value))}")
                     return int(round(elo_value))
                else:
                    logging.warning(f"Found past Elo for {team_name}, but latest period ended >90 days before {match_date_naive.date()} or 'to' date invalid. Skipping.")
                    return None
            else:
                logging.warning(f"Could not find any Elo rating interval or prior rating for {team_name} on or before {match_date_naive.date()}")
                return None

    except Exception as e:
        logging.error(f"Error processing Elo history lookup for {team_name} on {match_date_naive.date()}: {e}")
        return None

def validate_elo_data(match_id, team_name, match_date, elo_value, elo_reader):
    """
    Validates that the ELO value is actually from the correct time period.
    
    Args:
        match_id: The match identifier
        team_name: Team to validate
        match_date: Date of the match
        elo_value: The ELO value we calculated
        elo_reader: ClubElo reader instance
    
    Returns:
        bool: True if valid, False if invalid
    """
    # Double check using read_by_date directly
    try:
        # Get ELO scores directly from the API for that specific date
        date_str = match_date.strftime('%Y-%m-%d')
        direct_elo_df = elo_reader.read_by_date(date=date_str)
        
        # Find the team in the result
        team_row = direct_elo_df[direct_elo_df.index == team_name]
        if team_row.empty:
            # Try mapped versions
            for variant in [
                TEAM_NAME_MAPPING.get(team_name, team_name),
                team_name.replace(" & ", " "),
                team_name.split(" & ")[0].strip(),
                team_name.replace("FC ", "").replace(" FC", "")
            ]:
                team_row = direct_elo_df[direct_elo_df.index == variant]
                if not team_row.empty:
                    break
        
        if not team_row.empty:
            direct_elo = int(round(team_row.iloc[0]['elo']))
            # Compare with a tolerance of 5 points for rounding differences
            if abs(direct_elo - elo_value) <= 5:
                logging.info(f"✅ VALIDATED: Match {match_id}, {team_name} on {date_str}: API ELO={direct_elo}, Calculated ELO={elo_value}")
                return True
            else:
                logging.warning(f"⚠️ MISMATCH: Match {match_id}, {team_name} on {date_str}: API ELO={direct_elo}, but calculated ELO={elo_value}")
                return False
        else:
            logging.warning(f"⚠️ VALIDATION FAILED: Match {match_id}, {team_name} not found in direct ELO data for {date_str}")
            return False
    except Exception as e:
        logging.error(f"⚠️ VALIDATION ERROR: Match {match_id}, {team_name} on {match_date}: {str(e)}")
        return False

# --- Main Script Logic ---
def main():
    """Connects to MongoDB, fetches Elo data, and updates match documents."""
    logging.info("Starting Elo injection script...")

    # --- Initialize ClubElo ---
    try:
        elo_reader = sd.ClubElo()
        logging.info("ClubElo reader initialized. Using cache path: %s", elo_reader.data_dir)
    except Exception as e:
        logging.error(f"Failed to initialize ClubElo reader: {e}")
        sys.exit(1)

    # --- Connect to MongoDB ---
    client = None
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        client.admin.command('ping')
        logging.info(f"Connected to MongoDB: {DB_NAME}/{COLLECTION_NAME}")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        if client:
            client.close()
        sys.exit(1)

    # --- Iterate and Update Matches ---
    logging.info("Processing matches and injecting Elo ratings...")
    update_count = 0
    error_count = 0
    skipped_exist_count = 0
    skipped_missing_data_count = 0
    skipped_elo_lookup_count = 0

    query = {"home_team_elo": {"$exists": False}}
    projection = {"_id": 1, "home_team_name": 1, "away_team_name": 1, "date_utc": 1, "date_str": 1}

    successful_teams = set()
    failed_teams = set()

    try:
        total_matches_to_process = collection.count_documents(query)
        logging.info(f"Found {total_matches_to_process} matches potentially needing Elo injection.")

        if total_matches_to_process == 0:
             logging.info("No matches require Elo injection. Exiting.")
             if client: client.close()
             sys.exit(0)

        matches_cursor = collection.find(query, projection, no_cursor_timeout=True)

        with matches_cursor:
            for match in tqdm(matches_cursor, total=total_matches_to_process, desc="Updating Matches"):
                match_id = match.get('_id')
                home_team = match.get('home_team_name')
                away_team = match.get('away_team_name')
                match_date_utc = match.get('date_utc')

                if not all([match_id, home_team, away_team, match_date_utc]):
                    logging.warning(f"Skipping match {match_id}: Missing required fields (ID, team names or date_utc).")
                    skipped_missing_data_count += 1
                    continue

                # Date handling logic (improved slightly)
                if not isinstance(match_date_utc, datetime):
                    date_str = match.get('date_str')
                    logging.warning(f"Match {match_id}: 'date_utc' is not a datetime object ({type(match_date_utc)}). Attempting parse from 'date_str': {date_str}")
                    try:
                        parsed_date = pd.to_datetime(date_str, errors='coerce') # Coerce invalid dates to NaT
                        if pd.isna(parsed_date):
                             raise ValueError(f"Could not parse date string: {date_str}")
                        # Assume UTC if naive, otherwise keep original timezone
                        if parsed_date.tzinfo is None:
                            match_date_utc = parsed_date.tz_localize(timezone.utc)
                        else:
                            match_date_utc = parsed_date.tz_convert(timezone.utc) # Ensure UTC
                    except Exception as parse_e:
                        logging.error(f"Skipping match {match_id}: Error parsing date ('{match_date_utc}' or '{date_str}'): {parse_e}")
                        skipped_missing_data_count += 1
                        continue
                elif match_date_utc.tzinfo is None:
                    match_date_utc = match_date_utc.replace(tzinfo=timezone.utc)
                else:
                     # Ensure it's UTC for consistency before converting to naive later
                     match_date_utc = match_date_utc.astimezone(timezone.utc)


                # Get Elo ratings (now uses mapping)
                home_elo = get_elo_on_date(home_team, match_date_utc, elo_reader)
                away_elo = get_elo_on_date(away_team, match_date_utc, elo_reader)

                if home_elo is not None and away_elo is not None:
                    try:
                        # Validate ELO values
                        home_valid = False
                        away_valid = False
                        
                        if home_elo is not None:
                            home_valid = validate_elo_data(match_id, home_team, match_date_utc, home_elo, elo_reader)
                        
                        if away_elo is not None:
                            away_valid = validate_elo_data(match_id, away_team, match_date_utc, away_elo, elo_reader)
                        
                        # Only update if BOTH values are valid
                        if home_elo is not None and away_elo is not None and home_valid and away_valid:
                            # Add log to verify date-specific ELO retrieval
                            logging.info(f"Match {match_id} on {match_date_utc.date()}: {home_team} ELO = {home_elo}, {away_team} ELO = {away_elo}")
                            
                            result = collection.update_one(
                                {'_id': match_id},
                                {'$set': {
                                    'home_team_elo': home_elo,
                                    'away_team_elo': away_elo,
                                    'elo_fetch_timestamp_utc': datetime.now(timezone.utc),
                                    # Add reference to match date for verification
                                    'elo_match_date_utc': match_date_utc
                                }}
                            )
                            if result.matched_count > 0:
                                 if result.modified_count > 0:
                                     update_count += 1
                                     successful_teams.add(home_team)
                                     successful_teams.add(away_team)
                                 else:
                                     logging.warning(f"Match {match_id} matched but was not modified.")
                                     skipped_exist_count +=1
                            else:
                                 logging.warning(f"Match {match_id} was not found during update attempt.")
                                 error_count += 1
                                 failed_teams.add(home_team)
                                 failed_teams.add(away_team)
                    except Exception as e:
                        logging.error(f"Failed to update match {match_id}: {e}")
                        error_count += 1
                        failed_teams.add(home_team)
                        failed_teams.add(away_team)
                else:
                    # Failure reason already logged in get_elo_on_date or fetch_team_elo_history
                    logging.warning(f"Skipping update for match {match_id} due to Elo lookup failure for one or both teams.")
                    skipped_elo_lookup_count += 1
                    failed_teams.add(home_team)
                    failed_teams.add(away_team)

    except Exception as e:
        logging.error(f"An error occurred during match processing loop: {e}")
    finally:
        if client:
            client.close()
            logging.info("MongoDB connection closed.")

    # Final Summary
    logging.info(f"--- Elo Injection Summary ---")
    logging.info(f"Successfully Updated: {update_count}")
    logging.info(f"Skipped (Already Exist/Race Condition): {skipped_exist_count}")
    logging.info(f"Skipped (Missing Input Data/Bad Date): {skipped_missing_data_count}")
    logging.info(f"Skipped (Elo Lookup Failed/Not Found/Stale): {skipped_elo_lookup_count}")
    logging.info(f"Errors during DB Update/Query: {error_count}")
    logging.info(f"Total Teams Failed Fetch (Unique): {len(FAILED_ELO_FETCH_TEAMS)}")
    if FAILED_ELO_FETCH_TEAMS:
         logging.info(f"Failed Team Names (Original): {sorted(list(FAILED_ELO_FETCH_TEAMS))}")
    logging.info(f"-----------------------------")

    # Enhanced final summary
    logging.info("\n=== Final Processing Summary ===")
    logging.info(f"Total Unique Teams Processed Successfully: {len(successful_teams)}")
    logging.info(f"Total Unique Teams Failed: {len(failed_teams)}")
    logging.info(f"Success Rate: {(len(successful_teams)/(len(successful_teams)+len(failed_teams))*100):.1f}%")
    
    if len(failed_teams) > 0:
        logging.info("\nFailed Teams (sample of first 10):")
        for team in list(failed_teams)[:10]:
            logging.info(f"- {team}")


if __name__ == "__main__":
    main()