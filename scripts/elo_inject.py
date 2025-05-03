import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
import soccerdata as sd
import pandas as pd
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import logging
import pathlib

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
    "FC OSS": "TOP Oss",
    "FC Copenhagen": "FC København",
    "Randers FC": "Randers",
    "FC Heidenheim": "Heidenheim",
    "Neftchi Baku": "Neftchi",
    "Vikingur Reykjavik": "Víkingur",
    "Lech Poznan": "Lech",
    "Pacos Ferreira": "Paços Ferreira",
    "Puszcza Niepołomice": "Puszcza",
    "Hanácká": "Hanácká Slavia",
    "Valmiera BSS": "Valmiera",
    "Krems Rehberg": "Krems Rehberg",
    "Manchester United": "Man United",
    "Inter": "Internazionale",
    "Real Madrid": "Real",
    "Bayern Munich": "Bayern",
    "Juventus": "Juve",
    "Paris Saint-Germain": "PSG",
    "Barcelona": "Barça",
    "Chelsea": "Chelsea FC",
    "Liverpool": "Liverpool FC",
    "Arsenal": "Arsenal FC",
    # Add more mappings here as needed based on logs
}

# --- Helper Functions ---

# Cache for team Elo histories to minimize API calls via soccerdata's cache
TEAM_ELO_HISTORIES_CACHE = {}
FAILED_ELO_FETCH_TEAMS = set() # Store original names that failed

def create_parent_dirs(filepath):
    """Ensure parent directories exist for the given filepath"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return filepath

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
        TEAM_NAME_MAPPING.get(original_team_name, original_team_name),
        original_team_name.replace(" & ", " "),
        original_team_name.split(" & ")[0].strip(),
        original_team_name.replace("FC ", "").replace(" FC", ""),
        original_team_name.split(" ")[0],  # Try just first word
        ' '.join(original_team_name.split(" ")[:2]),  # Try first two words
        original_team_name.replace("BSS", "").strip(),  # Remove BSS suffix
        original_team_name.replace("Rehberg", "").strip(),  # Remove Rehberg suffix
    ]

    if "/" in original_team_name:
        parts = original_team_name.split("/")
        name_variations.extend(parts)

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
            logging.debug(f"Value error for variant '{name_variant}': {ve}")
            continue
        except FileNotFoundError as fnf:
            logging.debug(f"File not found for variant '{name_variant}': {fnf}")
            # Try to create directory structure
            try:
                expected_path = pathlib.Path(elo_reader.data_dir) / f"{name_variant}.csv"
                os.makedirs(expected_path.parent, exist_ok=True)
            except:
                pass
            continue
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

def check_elo_file_exists(team_name, elo_reader):
    """Check if the ELO file for a team exists in cache"""
    # ClubElo files are stored as TeamName.csv in the data directory
    filepath = pathlib.Path(elo_reader.data_dir) / f"{team_name}.csv"
    return filepath.exists()

# --- Main Script Logic ---
def main():
    """Connects to MongoDB, fetches Elo data for all teams, and updates match documents."""
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

    # --- Fetch All Unique Team Names from DB ---
    logging.info("Fetching unique team names from the database...")
    unique_teams = set()
    try:
        home_teams = collection.distinct("home_team_name")
        away_teams = collection.distinct("away_team_name")
        unique_teams = set(home_teams) | set(away_teams)
        # Filter out potential None or empty strings if they exist
        unique_teams = {team for team in unique_teams if team}
        logging.info(f"Found {len(unique_teams)} unique team names in the database.")
    except Exception as e:
        logging.error(f"Failed to fetch unique team names from MongoDB: {e}")
        if client: client.close()
        sys.exit(1)

    # --- Pre-fetch Elo Histories for All Teams ---
    logging.info("Attempting to pre-fetch Elo histories for all unique teams...")
    prefetched_count = 0
    prefetch_failed_count = 0
    for team_name in tqdm(unique_teams, desc="Pre-fetching Elo Data"):
        history = fetch_team_elo_history(team_name, elo_reader)
        if history is not None:
            prefetched_count += 1
        # fetch_team_elo_history automatically adds to FAILED_ELO_FETCH_TEAMS on failure
    prefetch_failed_count = len(FAILED_ELO_FETCH_TEAMS)
    logging.info(f"Finished pre-fetching: Successfully cached {prefetched_count}, Failed/Not Found {prefetch_failed_count} teams.")
    # Log failed teams discovered during pre-fetch if any
    if FAILED_ELO_FETCH_TEAMS:
        logging.warning(f"Teams failing pre-fetch: {sorted(list(FAILED_ELO_FETCH_TEAMS))[:20]}...") # Log first 20


    # --- Iterate and Update Matches ---
    logging.info("Processing matches and injecting Elo ratings...")
    update_count = 0
    error_count = 0
    skipped_exist_count = 0
    skipped_missing_data_count = 0
    skipped_elo_lookup_count = 0 # Count matches skipped due to ELO lookup failure (now likely means one team failed pre-fetch)

    # Original query to find matches needing ELO
    query = {"home_team_elo": {"$exists": False}}
    projection = {"_id": 1, "home_team_name": 1, "away_team_name": 1, "date_utc": 1, "date_str": 1}

    successful_teams_in_matches = set() # Track teams successfully updated *in matches*
    failed_teams_in_matches = set()     # Track teams involved in failed match updates

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
                    failed_teams_in_matches.add(home_team)
                    failed_teams_in_matches.add(away_team)
                    continue

                # Date handling logic (improved slightly) - Keep existing logic
                if not isinstance(match_date_utc, datetime):
                    date_str = match.get('date_str')
                    logging.warning(f"Match {match_id}: 'date_utc' is not a datetime object ({type(match_date_utc)}). Attempting parse from 'date_str': {date_str}")
                    try:
                        # Use pandas for robust parsing, coerce errors
                        parsed_date = pd.to_datetime(date_str, errors='coerce', utc=True)
                        if pd.isna(parsed_date):
                             raise ValueError(f"Could not parse date string: {date_str}")
                        match_date_utc = parsed_date.to_pydatetime() # Convert back to datetime object
                    except Exception as parse_e:
                        logging.error(f"Skipping match {match_id}: Error parsing date ('{date_str}'): {parse_e}")
                        skipped_missing_data_count += 1
                        failed_teams_in_matches.add(home_team)
                        failed_teams_in_matches.add(away_team)
                        continue
                elif match_date_utc.tzinfo is None:
                    # Assume UTC if naive
                    match_date_utc = match_date_utc.replace(tzinfo=timezone.utc)
                else:
                     # Ensure it's UTC for consistency
                     match_date_utc = match_date_utc.astimezone(timezone.utc)


                # Get Elo ratings (now uses pre-populated cache primarily)
                # No changes needed here, get_elo_on_date uses the cache
                home_elo = get_elo_on_date(home_team, match_date_utc, elo_reader)
                away_elo = get_elo_on_date(away_team, match_date_utc, elo_reader)

                # Check if *either* team failed the lookup (likely due to pre-fetch failure)
                if home_elo is None or away_elo is None:
                    logging.warning(f"Skipping update for match {match_id}: Elo lookup failed for '{home_team}' ({'Found' if home_elo is not None else 'Not Found'}) or '{away_team}' ({'Found' if away_elo is not None else 'Not Found'}).")
                    skipped_elo_lookup_count += 1
                    # Add teams even if one succeeded, as the match update failed
                    failed_teams_in_matches.add(home_team)
                    failed_teams_in_matches.add(away_team)
                    continue # Skip to next match

                # --- Proceed with Update only if BOTH ELO values were found ---
                try:
                    # Optional: Keep validation if desired, but it might be slow
                    # home_valid = validate_elo_data(match_id, home_team, match_date_utc, home_elo, elo_reader)
                    # away_valid = validate_elo_data(match_id, away_team, match_date_utc, away_elo, elo_reader)
                    # if not (home_valid and away_valid):
                    #      logging.warning(f"Skipping match {match_id} due to ELO validation failure.")
                    #      error_count += 1 # Count validation failures as errors perhaps?
                    #      failed_teams_in_matches.add(home_team)
                    #      failed_teams_in_matches.add(away_team)
                    #      continue

                    # Log successful retrieval before update
                    logging.info(f"Match {match_id} on {match_date_utc.date()}: Found ELO {home_team}={home_elo}, {away_team}={away_elo}. Attempting update.")

                    result = collection.update_one(
                        {'_id': match_id},
                        {'$set': {
                            'home_team_elo': home_elo,
                            'away_team_elo': away_elo,
                            'elo_fetch_timestamp_utc': datetime.now(timezone.utc),
                            'elo_match_date_utc': match_date_utc # Keep reference date
                        }}
                    )
                    if result.matched_count > 0:
                         if result.modified_count > 0:
                             update_count += 1
                             successful_teams_in_matches.add(home_team)
                             successful_teams_in_matches.add(away_team)
                         else:
                             # Match found but not modified (e.g., race condition, already updated?)
                             logging.warning(f"Match {match_id} matched but was not modified (already updated?).")
                             skipped_exist_count +=1
                             # Add to successful if already updated
                             successful_teams_in_matches.add(home_team)
                             successful_teams_in_matches.add(away_team)
                    else:
                         # Should not happen if we iterate based on find() results, but handle defensively
                         logging.warning(f"Match {match_id} was not found during update attempt (unexpected).")
                         error_count += 1
                         failed_teams_in_matches.add(home_team)
                         failed_teams_in_matches.add(away_team)

                except Exception as update_e:
                    logging.error(f"Failed to update match {match_id}: {update_e}")
                    error_count += 1
                    failed_teams_in_matches.add(home_team)
                    failed_teams_in_matches.add(away_team)
                # The 'else' case for ELO lookup failure is handled above by 'continue'

    except Exception as loop_e:
        logging.error(f"An error occurred during match processing loop: {loop_e}", exc_info=True)
    finally:
        if client:
            client.close()
            logging.info("MongoDB connection closed.")

    # --- Final Summary ---
    logging.info(f"\n--- Elo Injection Summary ---")
    logging.info(f"Total Unique Teams Found in DB: {len(unique_teams)}")
    logging.info(f"Teams Pre-fetched Successfully: {prefetched_count}")
    logging.info(f"Teams Failed Pre-fetch (Not Found/Error): {prefetch_failed_count}") # This uses len(FAILED_ELO_FETCH_TEAMS)
    logging.info(f"--- Match Update Results ---")
    logging.info(f"Matches Successfully Updated with Elo: {update_count}")
    logging.info(f"Matches Skipped (Already Had Elo/Race Condition): {skipped_exist_count}")
    logging.info(f"Matches Skipped (Missing Input Data/Bad Date): {skipped_missing_data_count}")
    logging.info(f"Matches Skipped (Elo Lookup Failed for >=1 Team): {skipped_elo_lookup_count}")
    logging.info(f"Errors during DB Update/Query/Validation: {error_count}")
    logging.info(f"--- Team Statistics (from Match Processing) ---")
    # Calculate unique teams involved in failed updates, excluding those that also had successes
    truly_failed_teams = failed_teams_in_matches - successful_teams_in_matches
    logging.info(f"Unique Teams in Successfully Updated Matches: {len(successful_teams_in_matches)}")
    logging.info(f"Unique Teams Involved in At Least One Failed Update: {len(failed_teams_in_matches)}")
    logging.info(f"Unique Teams Involved ONLY in Failed Updates: {len(truly_failed_teams)}")
    # Combine pre-fetch failures and match update failures for a comprehensive list
    overall_failed_teams = FAILED_ELO_FETCH_TEAMS | truly_failed_teams
    logging.info(f"Total Unique Teams with Issues (Pre-fetch Fail OR Only in Failed Updates): {len(overall_failed_teams)}")
    if overall_failed_teams:
        logging.warning(f"Problematic Team Names (Sample): {sorted(list(overall_failed_teams))[:50]}...") # Show more sample failures
    logging.info(f"-----------------------------")


if __name__ == "__main__":
    main()