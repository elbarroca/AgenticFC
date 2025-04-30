import pandas as pd
import numpy as np
import os
import glob  # For finding CSV files
from pymongo import MongoClient # For MongoDB connection
from dateutil.parser import parse # More flexible date parsing
import logging
import re # For potential column name cleaning
import sys

# Add the project root to Python path to allow imports from any module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import config from models/utils
from models.utils import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Define a set of CORE columns expected in the final unified DataFrame
CORE_COLS = [
    'MatchID', 'Date', 'Timestamp', 'Season', 'LeagueID', 'LeagueName', 'Country', 'Round',
    'HomeTeam', 'AwayTeam', 'HomeTeamWinner', 'AwayTeamWinner', # Added Winner flags
    'FTHG', 'FTAG', 'FTR',
    'HTHG', 'HTAG', 'HTR',
    'Referee', 'VenueName', 'VenueCity', # Added fixture details
    'StatusLong', 'StatusShort', 'StatusElapsed', # Added status
    'HomeFormation', 'AwayFormation' # Added formation
]

# Define desired standardized STATS columns (map source names to these)
STATS_COLS_MAP = {
    # CSV potential names -> Standard Name
    'HS': 'HomeShots', 'AS': 'AwayShots',
    'HST': 'HomeShotsTarget', 'AST': 'AwayShotsTarget',
    'HF': 'HomeFouls', 'AF': 'AwayFouls',
    'HC': 'HomeCorners', 'AC': 'AwayCorners',
    'HY': 'HomeYellowCards', 'AY': 'AwayYellowCards',
    'HR': 'HomeRedCards', 'AR': 'AwayRedCards',
    # MongoDB potential names (from statistics array) -> Standard Name
    'Total Shots_home': 'HomeShots', 'Total Shots_away': 'AwayShots',
    'Shots on Goal_home': 'HomeShotsTarget', 'Shots on Goal_away': 'AwayShotsTarget',
    'Fouls_home': 'HomeFouls', 'Fouls_away': 'AwayFouls',
    'Corner Kicks_home': 'HomeCorners', 'Corner Kicks_away': 'AwayCorners',
    'Yellow Cards_home': 'HomeYellowCards', 'Yellow Cards_away': 'AwayYellowCards',
    'Red Cards_home': 'HomeRedCards', 'Red Cards_away': 'AwayRedCards',
    # Add others like possession if available
    'Ball Possession_home': 'HomePossession', 'Ball Possession_away': 'AwayPossession',
    # Add Expected Goals
    'expected_goals_home': 'HomeExpectedGoals', 'expected_goals_away': 'AwayExpectedGoals',
}

# Define desired standardized ODDS columns (map source names to these)
# Prioritize certain bookmakers if multiple exist (e.g., B365, Pinnacle)
ODDS_COLS_MAP = {
    # Target Name : [List of potential source column names in order of preference]
    'OddsH': ['B365H', 'PSH', 'AvgH', 'MaxH', 'WHH', 'VCH', 'LBH'], # Add others
    'OddsD': ['B365D', 'PSD', 'AvgD', 'MaxD', 'WHD', 'VCD', 'LBD'],
    'OddsA': ['B365A', 'PSA', 'AvgA', 'MaxA', 'WHA', 'VCA', 'LBA'],
    'OddsOver2.5': ['B365>2.5', 'P>2.5', 'Max>2.5', 'Avg>2.5'],
    'OddsUnder2.5': ['B365<2.5', 'P<2.5', 'Max<2.5', 'Avg<2.5'],
    # Add Asian Handicap odds if needed
}

# Simplified concept for extract_mongo_stats
MONGO_STATS_TYPE_MAP = {
    'Shots on Goal': ('HomeShotsTarget', 'AwayShotsTarget'),
    'Total Shots': ('HomeShots', 'AwayShots'),
    'Corner Kicks': ('HomeCorners', 'AwayCorners'),
    'Fouls': ('HomeFouls', 'AwayFouls'),
    'Ball Possession': ('HomePossession', 'AwayPossession'),
    'Yellow Cards': ('HomeYellowCards', 'AwayYellowCards'),
    'Red Cards': ('HomeRedCards', 'AwayRedCards'),
    'expected_goals': ('HomeExpectedGoals', 'AwayExpectedGoals'),
}

# --- Helper Functions ---

def parse_date(date_str):
    """Attempts to parse various date formats."""
    if pd.isna(date_str):
        return None
    try:
        # Try standard formats first
        return pd.to_datetime(date_str, errors='raise')
    except (ValueError, TypeError):
        try:
            # Use dateutil for more flexible parsing
            return parse(str(date_str), dayfirst=True) # Assume day first for ambiguous like 01/02/03
        except (ValueError, TypeError, OverflowError):
            logging.warning(f"Could not parse date: {date_str}")
            return None

def standardize_team_names(name: str, mapping: dict) -> str:
    """Applies cleaning and mapping to team names."""
    if pd.isna(name):
        return None
    name = str(name).strip()
    return mapping.get(name, name) # Return mapped name or original if no mapping

def create_match_id(row, date_col='Date', home_col='HomeTeam', away_col='AwayTeam'):
    """Creates a unique identifier string for a match."""
    try:
        # Format date consistently to avoid issues with timezones etc.
        date_str = pd.to_datetime(row[date_col]).strftime('%Y%m%d')
        home = str(row[home_col]).replace(" ", "")
        away = str(row[away_col]).replace(" ", "")
        return f"{date_str}_{home}_{away}"
    except Exception:
        return None # Handle cases where date or teams are missing

def extract_mongo_stats(stats_list: list) -> dict:
    """Extracts stats from the MongoDB statistics list/dict structure."""
    # CORRECTED INITIALIZATION: Flatten the list of tuples into a list of strings
    all_target_stat_names = [name for tpl in MONGO_STATS_TYPE_MAP.values() for name in tpl]
    extracted = {std_name: np.nan for std_name in all_target_stat_names} # Initialize with string keys

    if not isinstance(stats_list, list):
        logging.debug("Statistics data is not a list, returning empty dict.")
        return extracted # Return empty if format is unexpected

    for team_stats in stats_list:
        venue = team_stats.get('_team_context_venue') # 'home' or 'away'
        if not venue: continue

        stats_items = team_stats.get('statistics', [])
        if not isinstance(stats_items, list):
             logging.debug(f"Statistics for team with venue '{venue}' is not a list.")
             continue

        for stat_item in stats_items:
            if isinstance(stat_item, dict):
                stat_type = stat_item.get('type')
                value = stat_item.get('value')

                if stat_type in MONGO_STATS_TYPE_MAP:
                    target_col_tuple = MONGO_STATS_TYPE_MAP[stat_type]
                    target_col_name = target_col_tuple[0] if venue == 'home' else target_col_tuple[1]

                    # Process value (handle %, None, conversion errors)
                    processed_value = None
                    if isinstance(value, str) and '%' in value:
                        try: processed_value = float(value.replace('%',''))
                        except (ValueError, TypeError): pass
                    elif value is not None:
                        # Try converting to numeric, coercing errors to NaN
                        processed_value = pd.to_numeric(value, errors='coerce')
                        # If coercion resulted in NaN, keep it as NaN, otherwise proceed
                        if pd.isna(processed_value):
                           processed_value = np.nan # Explicitly set to NaN if coercion failed

                    # Store the processed value if it's not NaN (or keep existing NaN if already set)
                    if pd.notna(processed_value):
                        extracted[target_col_name] = processed_value
                    # No need for explicit else NaN, as they are pre-initialized

                # else: # Optional: Log unknown stat types if needed for debugging
                #     logging.debug(f"Stat type '{stat_type}' not found in MONGO_STATS_TYPE_MAP.")

            # else: # Optional: Log invalid stat_item format
            #      logging.debug(f"Skipping invalid stat_item: {stat_item}")


    return extracted


# --- Main Processing Functions ---

def load_and_standardize_csv_data(csv_dir: str, team_mapping: dict) -> pd.DataFrame:
    """Loads all CSVs, standardizes columns, types, and team names."""
    all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not all_files:
        logging.warning(f"No CSV files found in directory: {csv_dir}")
        return pd.DataFrame()

    logging.info(f"Found {len(all_files)} CSV files to process.")
    df_list = []

    for f in all_files:
        try:
            # Try common encodings if default fails
            try:
                df_temp = pd.read_csv(f, low_memory=False)
            except UnicodeDecodeError:
                df_temp = pd.read_csv(f, encoding='ISO-8859-1', low_memory=False)

            logging.info(f"Processing {os.path.basename(f)} with {len(df_temp)} rows.")

            # 1. Date Parsing (handle multiple potential column names and formats)
            date_col = next((col for col in ['Date', 'datetime', 'date'] if col in df_temp.columns), None)
            if date_col:
                df_temp['Date'] = df_temp[date_col].apply(parse_date)
                df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
                # Drop original date column if it wasn't 'Date'
                if date_col != 'Date':
                    df_temp = df_temp.drop(columns=[date_col])
            else:
                logging.warning(f"No date column found in {f}. Skipping file or add handling.")
                continue # Skip file if no date

            # Drop rows where date parsing failed
            df_temp = df_temp.dropna(subset=['Date'])
            if df_temp.empty: continue

            # 2. Team Name Standardization
            home_col = next((col for col in ['HomeTeam', 'Home', 'HT'] if col in df_temp.columns), None)
            away_col = next((col for col in ['AwayTeam', 'Away', 'AT'] if col in df_temp.columns), None)
            if not home_col or not away_col:
                 logging.warning(f"Missing HomeTeam/AwayTeam columns in {f}. Skipping.")
                 continue

            df_temp['HomeTeam'] = df_temp[home_col].apply(lambda x: standardize_team_names(x, team_mapping))
            df_temp['AwayTeam'] = df_temp[away_col].apply(lambda x: standardize_team_names(x, team_mapping))
            if home_col != 'HomeTeam': df_temp = df_temp.drop(columns=[home_col])
            if away_col != 'AwayTeam': df_temp = df_temp.drop(columns=[away_col])

            # 3. Standardize Core Result Columns
            # FTR (Full Time Result)
            if 'FTR' not in df_temp.columns: df_temp['FTR'] = None # Add if missing
            # FTHG, FTAG (Full Time Goals) - Crucial
            hg_col = next((col for col in ['FTHG', 'HG'] if col in df_temp.columns), None)
            ag_col = next((col for col in ['FTAG', 'AG'] if col in df_temp.columns), None)
            if not hg_col or not ag_col:
                 logging.warning(f"Missing FTHG/FTAG columns in {f}. Skipping.")
                 continue
            df_temp['FTHG'] = pd.to_numeric(df_temp[hg_col], errors='coerce')
            df_temp['FTAG'] = pd.to_numeric(df_temp[ag_col], errors='coerce')
            if hg_col != 'FTHG': df_temp = df_temp.drop(columns=[hg_col])
            if ag_col != 'FTAG': df_temp = df_temp.drop(columns=[ag_col])
            # Infer FTR if missing and goals are present
            df_temp['FTR'] = df_temp.apply(
                lambda row: np.select([row['FTHG'] > row['FTAG'], row['FTHG'] < row['FTAG']], ['H', 'A'], default='D')
                            if pd.isna(row['FTR']) and pd.notna(row['FTHG']) and pd.notna(row['FTAG']) else row['FTR'],
                axis=1
            )

            # HTHG, HTAG, HTR (Half Time) - Optional but good
            for ht_prefix in ['HTHG', 'HTAG', 'HTR']:
                ht_col = next((col for col in df_temp.columns if col.startswith(ht_prefix[:2]) and col.endswith(ht_prefix[2:])), None)
                if ht_col and ht_col != ht_prefix:
                     df_temp[ht_prefix] = df_temp[ht_col]
                     df_temp = df_temp.drop(columns=[ht_col])
                elif ht_prefix not in df_temp.columns:
                     df_temp[ht_prefix] = None # Add if missing
            df_temp['HTHG'] = pd.to_numeric(df_temp['HTHG'], errors='coerce')
            df_temp['HTAG'] = pd.to_numeric(df_temp['HTAG'], errors='coerce')


            # 4. Standardize Stats Columns
            for source_name, target_name in STATS_COLS_MAP.items():
                # Check if any potential source column exists
                source_col = next((col for col in df_temp.columns if col == source_name), None)
                if source_col and target_name not in df_temp.columns: # Only rename if target doesn't exist
                    df_temp[target_name] = pd.to_numeric(df_temp[source_col], errors='coerce')
                    # Optionally drop original if name changed significantly (e.g., HS -> HomeShots)
                    if source_col != target_name:
                         df_temp = df_temp.drop(columns=[source_col])
                elif target_name not in df_temp.columns:
                     df_temp[target_name] = np.nan # Add column with NaN if not found

            # 5. Standardize Odds Columns (select preferred bookmaker)
            for target_name, source_options in ODDS_COLS_MAP.items():
                selected_col = None
                for source_col in source_options:
                    if source_col in df_temp.columns:
                        selected_col = source_col
                        break # Found preferred source
                if selected_col:
                    df_temp[target_name] = pd.to_numeric(df_temp[selected_col], errors='coerce')
                    # Optionally drop original odds columns after standardization
                elif target_name not in df_temp.columns:
                     df_temp[target_name] = np.nan # Add column with NaN if not found


            # 6. Add League/Season Info (Extract from filename or 'Div' column)
            if 'Div' in df_temp.columns:
                df_temp['LeagueID'] = df_temp['Div'] # Assuming Div is a league identifier
                df_temp['LeagueName'] = df_temp['Div'] # Placeholder, might need mapping
            else:
                # Try extracting from filename (e.g., 'E0_2022_2023.csv')
                match = re.match(r"([A-Z0-9]+)(_(\d{4})_(\d{4}))?", os.path.basename(f))
                if match:
                    df_temp['LeagueID'] = match.group(1)
                    df_temp['LeagueName'] = match.group(1) # Placeholder
                    if match.group(3):
                         df_temp['Season'] = f"{match.group(3)}/{match.group(4)}"
                else:
                     df_temp['LeagueID'] = 'Unknown'
                     df_temp['LeagueName'] = 'Unknown'

            if 'Season' not in df_temp.columns: df_temp['Season'] = None # Add if missing
            if 'Country' not in df_temp.columns: df_temp['Country'] = None # Add if missing (might infer from LeagueID)


            # 7. Create MatchID
            df_temp['MatchID'] = df_temp.apply(create_match_id, axis=1, date_col='Date', home_col='HomeTeam', away_col='AwayTeam')
            df_temp = df_temp.dropna(subset=['MatchID']) # Need ID for merging

            # 8. Select and Reorder Columns
            cols_to_keep = CORE_COLS + list(STATS_COLS_MAP.values()) + list(ODDS_COLS_MAP.keys())
            final_cols_in_df = [col for col in cols_to_keep if col in df_temp.columns]
            df_list.append(df_temp[final_cols_in_df])

        except Exception as e:
            logging.error(f"Failed to process file {f}: {e}", exc_info=True)

    if not df_list:
        logging.error("No CSV data could be processed.")
        return pd.DataFrame()

    # Concatenate all processed DataFrames
    combined_csv_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined CSV data shape: {combined_csv_df.shape}")
    # Drop exact duplicates just in case
    combined_csv_df = combined_csv_df.drop_duplicates(subset=['MatchID'], keep='first')
    logging.info(f"Shape after dropping duplicate MatchIDs: {combined_csv_df.shape}")
    return combined_csv_df


def load_and_standardize_mongo_data(mongo_uri: str, db_name: str, collection_name: str, team_mapping: dict) -> pd.DataFrame:
    """Loads data from MongoDB, extracts relevant fields, standardizes them."""
    logging.info(f"Connecting to MongoDB: {mongo_uri}...")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000) # Timeout
        client.server_info() # Trigger exception if connection failed
        db = client[db_name]
        collection = db[collection_name]
        logging.info(f"Connected to DB: {db_name}, Collection: {collection_name}")
    except Exception as e:
        logging.error(f"MongoDB connection failed: {e}")
        return pd.DataFrame()

    # Load data - consider batching for very large collections using find().batch_size()
    logging.info("Loading data from MongoDB...")
    try:
        # Define projection to load only necessary fields
        projection = {
            "_id": 1, # Keep ID for error tracking
            "fixture_details.fixture.id": 1,
            "fixture_details.fixture.referee": 1,
            "fixture_details.fixture.timezone": 1,
            "fixture_details.fixture.date": 1,
            "fixture_details.fixture.timestamp": 1,
            "fixture_details.fixture.venue": 1,
            "fixture_details.fixture.status": 1,
            "fixture_details.league.id": 1,
            "fixture_details.league.name": 1,
            "fixture_details.league.country": 1,
            "fixture_details.league.season": 1,
            "fixture_details.league.round": 1,
            "fixture_details.teams.home": 1,
            "fixture_details.teams.away": 1,
            "fixture_details.goals": 1,
            "fixture_details.score.halftime": 1,
            "statistics_full": 1, # Keep the original name from your query structure
            "lineups": 1, # Keep the original name
            "odds_data": 1 # Assuming this exists based on previous code
        }
        # It's safer to explicitly request fields than load everything
        # mongo_docs = list(collection.find({})) # Original: Load all
        mongo_docs = list(collection.find({}, projection)) # Load only projected fields
        logging.info(f"Attempted to load {len(mongo_docs)} documents from MongoDB using projection.")

    except Exception as e:
         logging.error(f"Error loading data from MongoDB: {e}", exc_info=True)
         client.close()
         return pd.DataFrame()

    client.close()


    if not mongo_docs:
        logging.warning("No documents found or loaded from MongoDB.")
        return pd.DataFrame()

    processed_rows = []
    for i, doc in enumerate(mongo_docs):
        if i % 1000 == 0 and i > 0: # Log progress
             logging.info(f"Processing MongoDB document {i}...")

        row = {}
        try:
            # Use .get() extensively to avoid KeyErrors on missing fields/nested structures
            fixture_details = doc.get('fixture_details', {})
            fixture = fixture_details.get('fixture', {})
            league = fixture_details.get('league', {})
            teams = fixture_details.get('teams', {})
            goals = fixture_details.get('goals', {})
            score = fixture_details.get('score', {})
            venue = fixture.get('venue', {})
            status = fixture.get('status', {})
            home_team_details = teams.get('home', {})
            away_team_details = teams.get('away', {})


            # --- Core Match Info ---
            row['Date'] = parse_date(fixture.get('date'))
            if pd.isna(row['Date']):
                 logging.debug(f"Skipping doc ID {doc.get('_id', 'N/A')} due to missing/invalid date.")
                 continue # Skip if no valid date

            row['Timestamp'] = pd.to_numeric(fixture.get('timestamp'), errors='coerce') # Added Timestamp

            row['HomeTeam'] = standardize_team_names(home_team_details.get('name'), team_mapping)
            row['AwayTeam'] = standardize_team_names(away_team_details.get('name'), team_mapping)
            if not row['HomeTeam'] or not row['AwayTeam']:
                logging.debug(f"Skipping doc ID {doc.get('_id', 'N/A')} due to missing team names.")
                continue # Skip if teams missing

            row['HomeTeamWinner'] = home_team_details.get('winner') # Added Winner Flag
            row['AwayTeamWinner'] = away_team_details.get('winner') # Added Winner Flag


            # --- Score ---
            row['FTHG'] = pd.to_numeric(goals.get('home'), errors='coerce')
            row['FTAG'] = pd.to_numeric(goals.get('away'), errors='coerce')
            # Skip if full time goals are missing, as FTR depends on them
            if pd.isna(row['FTHG']) or pd.isna(row['FTAG']):
                 logging.debug(f"Skipping doc ID {doc.get('_id', 'N/A')} due to missing FTHG/FTAG.")
                 continue

            # Determine FTR
            if row['FTHG'] > row['FTAG']: row['FTR'] = 'H'
            elif row['FTHG'] < row['FTAG']: row['FTR'] = 'A'
            else: row['FTR'] = 'D'

            halftime_score = score.get('halftime', {})
            row['HTHG'] = pd.to_numeric(halftime_score.get('home'), errors='coerce')
            row['HTAG'] = pd.to_numeric(halftime_score.get('away'), errors='coerce')
            # Determine HTR
            if pd.notna(row['HTHG']) and pd.notna(row['HTAG']):
                 if row['HTHG'] > row['HTAG']: row['HTR'] = 'H'
                 elif row['HTHG'] < row['HTAG']: row['HTR'] = 'A'
                 else: row['HTR'] = 'D'
            else: row['HTR'] = None

            # --- League Info ---
            row['LeagueID'] = league.get('id')
            row['LeagueName'] = league.get('name')
            row['Country'] = league.get('country')
            row['Season'] = league.get('season') # Assuming season is available
            row['Round'] = league.get('round') # Added Round

            # --- Fixture Details ---
            row['Referee'] = fixture.get('referee') # Added Referee
            row['VenueName'] = venue.get('name') # Added Venue Name
            row['VenueCity'] = venue.get('city') # Added Venue City
            row['StatusLong'] = status.get('long') # Added Status Long
            row['StatusShort'] = status.get('short') # Added Status Short
            row['StatusElapsed'] = pd.to_numeric(status.get('elapsed'), errors='coerce') # Added Status Elapsed


            # --- Create MatchID ---
            # Ensure Date, HomeTeam, AwayTeam are used for ID creation after potential standardization
            row['MatchID'] = create_match_id(row, date_col='Date', home_col='HomeTeam', away_col='AwayTeam')
            if not row['MatchID']:
                logging.debug(f"Skipping doc ID {doc.get('_id', 'N/A')} due to MatchID creation failure.")
                continue

            # --- Extract Stats ---
            stats_list = doc.get('statistics_full', [])
            home_team_id = home_team_details.get('id')
            away_team_id = away_team_details.get('id')

            # Add context needed by extract_mongo_stats (venue)
            if isinstance(stats_list, list):
                 for team_stat_block in stats_list:
                     if isinstance(team_stat_block, dict) and 'team' in team_stat_block:
                          current_team_id = team_stat_block['team'].get('id')
                          if current_team_id == home_team_id:
                               team_stat_block['_team_context_venue'] = 'home'
                          elif current_team_id == away_team_id:
                               team_stat_block['_team_context_venue'] = 'away'
                          else:
                               # Log if a stat block team doesn't match home or away
                               logging.warning(f"Stat block team ID {current_team_id} in doc {doc.get('_id', 'N/A')} does not match Home ({home_team_id}) or Away ({away_team_id}).")


            extracted_stats = extract_mongo_stats(stats_list)
            row.update(extracted_stats) # Add the extracted stats to the row


             # --- Extract Lineups/Formation ---
            lineups = doc.get('lineups', [])
            row['HomeFormation'] = None
            row['AwayFormation'] = None
            if isinstance(lineups, list):
                for lineup_block in lineups:
                    if isinstance(lineup_block, dict):
                         team_info = lineup_block.get('team', {})
                         formation = lineup_block.get('formation')
                         current_team_id = team_info.get('id')
                         if formation:
                              if current_team_id == home_team_id:
                                   row['HomeFormation'] = formation
                              elif current_team_id == away_team_id:
                                   row['AwayFormation'] = formation


            # --- Extract Odds ---
            odds_data = doc.get('odds_data', {}) # Replace 'odds_data' with your actual key if different
            # Ensure all target odds columns are initialized
            for target_name in ODDS_COLS_MAP.keys():
                 row[target_name] = np.nan

            if isinstance(odds_data, dict): # Process only if odds_data is a dictionary
                for target_name, source_options in ODDS_COLS_MAP.items():
                     selected_val = np.nan # Default to NaN
                     for source_key in source_options:
                         # Check if the key exists *and* the value is not None or empty string
                         if source_key in odds_data and odds_data[source_key] not in [None, '']:
                             temp_val = pd.to_numeric(odds_data[source_key], errors='coerce')
                             # Use the value only if it's a valid number
                             if pd.notna(temp_val):
                                 selected_val = temp_val
                                 break # Found valid preferred odd
                     row[target_name] = selected_val
            else:
                 if odds_data: # Log only if odds_data exists but isn't a dict
                      logging.warning(f"Expected 'odds_data' to be a dict, but got {type(odds_data)} for doc ID {doc.get('_id', 'N/A')}")


            processed_rows.append(row)

        except Exception as e:
            # Log the specific document ID for easier debugging
            logging.error(f"Error processing MongoDB document ID {doc.get('_id', 'N/A')} at index {i}: {e}", exc_info=True) # Add exc_info for traceback


    if not processed_rows:
        logging.warning("No MongoDB documents were successfully processed.")
        return pd.DataFrame()

    mongo_df = pd.DataFrame(processed_rows)
    logging.info(f"Successfully processed {len(mongo_df)} MongoDB documents into DataFrame.")
    logging.info(f"Processed MongoDB data shape: {mongo_df.shape}")

    # Optional: Log columns present in the Mongo DF before merging
    logging.debug(f"MongoDB DataFrame columns before merge: {mongo_df.columns.tolist()}")

    return mongo_df

def merge_data(csv_df: pd.DataFrame, mongo_df: pd.DataFrame) -> pd.DataFrame:
    """Merges standardized CSV and MongoDB data based on MatchID."""
    if csv_df.empty and mongo_df.empty:
        logging.error("Both CSV and MongoDB dataframes are empty. Cannot merge.")
        return pd.DataFrame()
    elif csv_df.empty:
        logging.info("CSV data is empty. Returning only MongoDB data.")
        # Ensure mongo_df has at least the MatchID column if it exists
        if 'MatchID' not in mongo_df.columns and not mongo_df.empty:
             logging.warning("MatchID column missing in MongoDB data when CSV is empty.")
             # Decide handling: return empty, error, or proceed without MatchID? Returning as is for now.
        return mongo_df
    elif mongo_df.empty:
        logging.info("MongoDB data is empty. Returning only CSV data.")
        # Ensure csv_df has at least the MatchID column if it exists
        if 'MatchID' not in csv_df.columns and not csv_df.empty:
             logging.warning("MatchID column missing in CSV data when MongoDB is empty.")
        return csv_df

    logging.info(f"Merging CSV ({len(csv_df)}) and MongoDB ({len(mongo_df)}) data using MatchID...")
    logging.debug(f"CSV columns before merge: {csv_df.columns.tolist()}")
    logging.debug(f"MongoDB columns before merge: {mongo_df.columns.tolist()}")

    # --- Merge Strategy: Outer merge to keep all matches, then combine info ---
    # Assumes MatchID is the reliable unique key across sources
    try:
        merged_df = pd.merge(
            csv_df.set_index('MatchID'), # Set index for easier coalescing
            mongo_df.set_index('MatchID'), # Set index for easier coalescing
            left_index=True,
            right_index=True,
            how='outer',
            suffixes=('_csv', '_mongo') # Add suffixes to identify source of conflicting columns
        )
        merged_df = merged_df.reset_index() # Reset index to bring MatchID back as column
        logging.info(f"Shape after outer merge: {merged_df.shape}")
        logging.debug(f"Columns after merge: {merged_df.columns.tolist()}")

    except KeyError as e:
        logging.error(f"Merge failed. 'MatchID' column missing in one of the dataframes? Error: {e}")
        # Decide recovery strategy: return empty, or one of the inputs?
        # Returning empty DF for safety.
        return pd.DataFrame()
    except Exception as e:
         logging.error(f"An unexpected error occurred during merge: {e}", exc_info=True)
         return pd.DataFrame()


    # --- Combine/Coalesce Columns ---
    # Identify columns with suffixes
    csv_cols = [col for col in merged_df.columns if col.endswith('_csv')]
    mongo_cols = [col for col in merged_df.columns if col.endswith('_mongo')]
    base_cols_csv = [col.replace('_csv', '') for col in csv_cols]
    base_cols_mongo = [col.replace('_mongo', '') for col in mongo_cols]

    # Columns that exist in both sources (need coalescing)
    common_base_cols = list(set(base_cols_csv) & set(base_cols_mongo))
    # Columns that only exist in one source (suffix needs removing)
    unique_csv_cols = [col for col in csv_cols if col.replace('_csv', '') not in common_base_cols]
    unique_mongo_cols = [col for col in mongo_cols if col.replace('_mongo', '') not in common_base_cols]


    logging.info(f"Coalescing {len(common_base_cols)} common columns...")
    # Prioritize MongoDB data where available, then CSV for common columns
    for base_col in common_base_cols:
        col_mongo = f"{base_col}_mongo"
        col_csv = f"{base_col}_csv"
        # Use mongo column if exists, otherwise csv column. Handles cases where one might be all NaN.
        merged_df[base_col] = merged_df[col_mongo].combine_first(merged_df[col_csv])
        # Drop the original suffixed columns
        merged_df = merged_df.drop(columns=[col_mongo, col_csv])
        logging.debug(f"Coalesced column: {base_col}")


    # Rename unique columns by removing suffix
    logging.info(f"Renaming {len(unique_csv_cols)} unique CSV columns...")
    for col_csv in unique_csv_cols:
        base_col = col_csv.replace('_csv', '')
        merged_df = merged_df.rename(columns={col_csv: base_col})
        logging.debug(f"Renamed unique CSV column: {col_csv} -> {base_col}")

    logging.info(f"Renaming {len(unique_mongo_cols)} unique MongoDB columns...")
    for col_mongo in unique_mongo_cols:
         base_col = col_mongo.replace('_mongo', '')
         merged_df = merged_df.rename(columns={col_mongo: base_col})
         logging.debug(f"Renamed unique Mongo column: {col_mongo} -> {base_col}")


    # Verify no remaining suffixed columns (shouldn't happen with above logic)
    remaining_suffixed = [col for col in merged_df.columns if col.endswith('_csv') or col.endswith('_mongo')]
    if remaining_suffixed:
        logging.warning(f"Found unexpected remaining suffixed columns after coalescing: {remaining_suffixed}. Dropping them.")
        merged_df = merged_df.drop(columns=remaining_suffixed)


    # Handle potential full duplicate rows after merging/coalescing if MatchID wasn't unique originally (unlikely with prior processing)
    # Keep 'first' assumes the first occurrence (often earlier data if sorted before merge, or arbitrary otherwise) is preferred.
    # Consider 'last' if later data (e.g., Mongo data if it came second) is preferred in case of exact duplicates beyond MatchID.
    initial_rows = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['MatchID'], keep='first') # Keep first occurrence based on MatchID
    rows_dropped = initial_rows - len(merged_df)
    if rows_dropped > 0:
         logging.warning(f"Dropped {rows_dropped} rows due to duplicate MatchID after merging/coalescing.")

    logging.info(f"Shape after coalescing and final deduplication: {merged_df.shape}")
    logging.debug(f"Columns after coalescing: {merged_df.columns.tolist()}")


    return merged_df


def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Performs final cleaning, type casting, and sorting."""
    if df.empty:
        logging.warning("Input DataFrame to final_clean is empty. Returning empty DataFrame.")
        return df

    logging.info("Performing final cleaning...")

    # Ensure correct types for known columns
    if 'Date' in df.columns:
         df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Drop rows where Date parsing failed completely after merge/coalesce
    initial_rows = len(df)
    df = df.dropna(subset=['Date'])
    if len(df) < initial_rows:
         logging.warning(f"Dropped {initial_rows - len(df)} rows due to invalid Date after merge.")
         if df.empty: return df # Return early if no valid dates remain


    # --- Numeric Columns (using nullable Int64 for counts/goals) ---
    int_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'StatusElapsed']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # --- Numeric Columns (using float for stats/odds/timestamp) ---
    # Combine stats, odds, and other potential float columns
    float_cols = list(STATS_COLS_MAP.values()) + list(ODDS_COLS_MAP.keys()) + ['Timestamp']
    for col in float_cols:
        if col in df.columns:
            # Check if column is already numeric to avoid unnecessary conversion warnings
            if not pd.api.types.is_numeric_dtype(df[col]):
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            # Optionally cast to a specific float type like 'float64' if needed
            # df[col] = df[col].astype('float64') # Example


    # --- Boolean Columns ---
    bool_cols = ['HomeTeamWinner', 'AwayTeamWinner']
    for col in bool_cols:
        if col in df.columns:
             # Map various potential inputs to boolean, coerce errors to None (Nullable Boolean)
             # True values: True, 'true', 'True', 1, '1'
             # False values: False, 'false', 'False', 0, '0'
             # Others/Errors -> pd.NA
             df[col] = df[col].map({
                 True: True, 1: True, '1': True, 'true': True, 'True': True,
                 False: False, 0: False, '0': False, 'false': False, 'False': False
             }).astype('boolean') # Use pandas nullable boolean type



    # --- Categorical/String Columns ---
    # Convert potential numeric/other types to string, handle NaN representations
    string_cols = ['HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'LeagueID', 'LeagueName', 'Country',
                   'Season', 'Referee', 'VenueName', 'VenueCity', 'StatusLong', 'StatusShort',
                   'Round', 'HomeFormation', 'AwayFormation', 'MatchID'] # MatchID should be string
    for col in string_cols:
         if col in df.columns:
             # Convert to string first to handle mixed types, then replace 'nan'/None representations
             df[col] = df[col].astype(str).replace(['nan', 'NaN', 'None', '<NA>', ''], pd.NA).astype('string') # Use pandas nullable string


    # --- Sort chronologically ---
    # Use Timestamp if available and reliable, otherwise fallback to Date
    sort_col = 'Timestamp' if 'Timestamp' in df.columns and df['Timestamp'].notna().all() else 'Date'
    logging.info(f"Sorting final DataFrame by '{sort_col}'.")
    try:
         df = df.sort_values(by=sort_col).reset_index(drop=True)
    except KeyError:
         logging.error(f"Sort column '{sort_col}' not found. Skipping sort.")
    except Exception as e:
         logging.error(f"Error sorting DataFrame: {e}. Skipping sort.", exc_info=True)


    # --- Reorder columns (Core first, then Stats, then Odds, then others) ---
    core_ordered = [col for col in CORE_COLS if col in df.columns]
    stats_ordered = sorted([col for col in STATS_COLS_MAP.values() if col in df.columns and col not in core_ordered])
    odds_ordered = sorted([col for col in ODDS_COLS_MAP.keys() if col in df.columns and col not in core_ordered])
    other_cols = sorted([col for col in df.columns if col not in core_ordered + stats_ordered + odds_ordered])

    final_ordered_cols = core_ordered + stats_ordered + odds_ordered + other_cols

    # Ensure no columns were lost/duplicated
    if set(final_ordered_cols) != set(df.columns):
        logging.warning("Column mismatch during reordering. Using original columns.")
        final_ordered_cols = df.columns.tolist() # Fallback to original order

    df = df[final_ordered_cols]

    logging.info("Final cleaning and type casting complete.")
    logging.debug(f"Final DataFrame columns after cleaning: {df.columns.tolist()}")
    # Log info including dtypes
    # Capture info() output to string to avoid large console print
    # buffer = io.StringIO()
    # df.info(buf=buffer)
    # logging.info(f"Final DataFrame info:\n{buffer.getvalue()}")


    return df

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Data Preparation Script ---")

    # 1. Load CSV Data
    csv_data = load_and_standardize_csv_data(
        csv_dir=config.RAW_CSV_DIR,
        team_mapping=config.TEAM_NAME_MAPPING
    )

    # 2. Load MongoDB Data
    mongo_data = load_and_standardize_mongo_data(
        mongo_uri=config.MONGO_URI,
        db_name=config.MONGO_DB_NAME,
        collection_name=config.MONGO_COLLECTION_NAME,
        team_mapping=config.TEAM_NAME_MAPPING
    )

    # 3. Merge Data
    unified_data = merge_data(csv_data, mongo_data)

    # 4. Final Cleaning
    if not unified_data.empty:
        unified_data = final_clean(unified_data)

        # 5. Save Unified Data
        output_path = config.UNIFIED_DATA_PATH
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # --- FIX: Remove duplicate columns before saving ---
            # Keep the first occurrence of each column name
            unified_data = unified_data.loc[:, ~unified_data.columns.duplicated(keep='first')]
            logging.info(f"Shape after removing duplicate columns (if any): {unified_data.shape}")
            # --- End FIX ---

            unified_data.to_parquet(output_path, index=False, engine='pyarrow') # Use pyarrow for better compatibility
            logging.info(f"Successfully saved unified data to: {output_path}")
            logging.info(f"Final DataFrame shape: {unified_data.shape}")
            # Use pandas info() method with verbose=False to avoid excessively long output if needed
            # Or capture full info() to string buffer if required for detailed debug logging
            logging.info(f"Final DataFrame columns: {unified_data.columns.tolist()}")
            # Consider logging df.info() output for detailed structure confirmation
            # Example:
            # import io
            # buffer = io.StringIO()
            # unified_data.info(buf=buffer)
            # logging.info(f"Final DataFrame info:\n{buffer.getvalue()}")


        except Exception as e:
            logging.error(f"Failed to save unified data to {output_path}: {e}", exc_info=True)
    else:
        logging.error("No data available after merging. Output file not saved.")

    logging.info("--- Data Preparation Script Finished ---")