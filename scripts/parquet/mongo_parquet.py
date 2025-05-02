# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
from pymongo import MongoClient # For MongoDB connection
from dateutil.parser import parse # More flexible date parsing
import logging
import re # For potential column name cleaning
import sys
import io # For capturing df.info()

# --- Configuration Setup (Assuming config is importable) ---
# Add the project root to Python path (adjust if needed)
project_root_potential = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(project_root_potential) == 'scripts': # Common structure
    project_root = os.path.dirname(project_root_potential)
else:
    project_root = project_root_potential # Assume current dir's parent is root
sys.path.insert(0, project_root)
print(f"Project Root added to sys.path: {project_root}")

try:
    # Attempt to import configuration from your project structure
    from models.utils import config
    print("Successfully imported config from models.utils")
except ImportError as e:
    print(f"Error importing config: {e}")
    print("Could not find 'models.utils.config'. Using fallback paths defined in script.")
    # Define fallback paths/details if config import fails - **UPDATE THESE**
    class MockConfig:
        MONGO_URI = "mongodb://admin888:admin888@127.0.0.1:27017/?authSource=admin" # <<<--- UPDATE MONGODB URI if needed
        MONGO_DB_NAME = "agenticfc"         # <<<--- UPDATE DB NAME if needed
        MONGO_COLLECTION_NAME = "matches"      # <<<--- UPDATE COLLECTION NAME if needed
        TEAM_NAME_MAPPING = {}                  # Add team name mappings if necessary, e.g., {'B Mgladbach': 'Borussia Monchengladbach'}
        OUTPUT_PARQUET_PATH = "data/processed/unified_data_with_rolling_features.parquet" # <<<--- UPDATE OUTPUT PATH
    config = MockConfig()
    print(f"Using Fallback Config - Output Path: {config.OUTPUT_PARQUET_PATH}")
    # Consider adding: sys.exit(1) # Exit if config is essential and fails to load


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Define a set of CORE columns expected in the final unified DataFrame
# These are columns directly extracted or derived from the main fixture/league/team info
CORE_COLS = [
    'MatchID', 'Date', 'Timestamp', 'Season', 'LeagueID', 'LeagueName', 'Country', 'Round',
    'HomeTeam', 'AwayTeam', 'HomeTeamID', 'AwayTeamID', # Added Team IDs
    'HomeTeamWinner', 'AwayTeamWinner',
    'FTHG', 'FTAG', 'FTR',
    'HTHG', 'HTAG', 'HTR',
    'Referee', 'VenueName', 'VenueCity',
    'StatusLong', 'StatusShort', 'StatusElapsed',
    'HomeFormation', 'AwayFormation'
]

# Map MongoDB stat 'type' to corresponding Home/Away standardized column names
MONGO_STATS_TYPE_MAP = {
    'Shots on Goal': ('HomeShotsTarget', 'AwayShotsTarget'), 'Shots off Goal': ('HomeShotsOffTarget', 'AwayShotsOffTarget'),
    'Total Shots': ('HomeShots', 'AwayShots'), 'Blocked Shots': ('HomeBlockedShots', 'AwayBlockedShots'),
    'Shots insidebox': ('HomeShotsInsideBox', 'AwayShotsInsideBox'), 'Shots outsidebox': ('HomeShotsOutsideBox', 'AwayShotsOutsideBox'),
    'Fouls': ('HomeFouls', 'AwayFouls'), 'Corner Kicks': ('HomeCorners', 'AwayCorners'),
    'Offsides': ('HomeOffsides', 'AwayOffsides'), 'Ball Possession': ('HomePossession', 'AwayPossession'),
    'Yellow Cards': ('HomeYellowCards', 'AwayYellowCards'), 'Red Cards': ('HomeRedCards', 'AwayRedCards'),
    'Goalkeeper Saves': ('HomeSaves', 'AwaySaves'), 'Total passes': ('HomeTotalPasses', 'AwayTotalPasses'),
    'Passes accurate': ('HomePassesAccurate', 'AwayPassesAccurate'), 'Passes %': ('HomePassAccuracy', 'AwayPassAccuracy'),
    'expected_goals': ('HomeExpectedGoals', 'AwayExpectedGoals'),
}
ALL_MATCH_SPECIFIC_STATS_COLS = sorted([col for pair in MONGO_STATS_TYPE_MAP.values() for col in pair])

ODDS_COLS_MAP = {
    'OddsH': ['B365H', 'PSH', 'AvgH', 'MaxH'], 'OddsD': ['B365D', 'PSD', 'AvgD', 'MaxD'], 'OddsA': ['B365A', 'PSA', 'AvgA', 'MaxA'],
    'OddsOver2.5': ['B365>2.5', 'P>2.5', 'Max>2.5', 'Avg>2.5'], 'OddsUnder2.5': ['B365<2.5', 'P<2.5', 'Max<2.5', 'Avg<2.5'],
    'OddsAHH': ['AHH', 'PSCH', 'AvgAHH', 'MaxAHH'], 'OddsAHA': ['AHA', 'PSCA', 'AvgAHA', 'MaxAHA'], 'OddsAHh': ['AHh', 'AHLine']
}
ALL_ODDS_COLS = sorted(list(ODDS_COLS_MAP.keys()))

# --- Helper Functions --- (parse_date, standardize_team_names, create_match_id, extract_mongo_stats - unchanged from previous correct version)

def parse_date(date_input):
    """Attempts to parse various date formats and return tz-naive UTC datetime."""
    if pd.isna(date_input): return pd.NaT
    dt = None
    if isinstance(date_input, (int, float)):
        try: dt = pd.to_datetime(date_input, unit='s', errors='raise')
        except (ValueError, TypeError, OverflowError):
            try: dt = pd.to_datetime(date_input, unit='ms', errors='raise')
            except (ValueError, TypeError, OverflowError): return pd.NaT
    elif isinstance(date_input, str):
        common_formats = ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"]
        for fmt in common_formats:
            try: dt = pd.to_datetime(date_input, format=fmt, errors='raise'); break
            except (ValueError, TypeError): continue
        if dt is None:
            try: dt = pd.to_datetime(date_input, errors='raise', infer_datetime_format=True)
            except (ValueError, TypeError, OverflowError):
                try: parsed_dt = parse(date_input); dt = pd.Timestamp(parsed_dt)
                except (ValueError, TypeError, OverflowError): return pd.NaT
    elif isinstance(date_input, (pd.Timestamp, datetime.datetime)):
        dt = pd.to_datetime(date_input)
    else: return pd.NaT
    if pd.isna(dt): return pd.NaT
    try:
        if not isinstance(dt, pd.Timestamp): dt = pd.Timestamp(dt)
        if dt.tzinfo is not None: dt = dt.tz_convert('UTC').tz_localize(None)
        return dt
    except Exception as e:
        logging.warning(f"Error final date conversion {dt}: {e}"); return pd.NaT

def standardize_team_names(name: str, mapping: dict) -> str:
    if pd.isna(name): return None
    name = str(name).strip()
    return mapping.get(name, name)

def create_match_id(row, date_col='Date', home_col='HomeTeam', away_col='AwayTeam'):
    try:
        if pd.isna(row[date_col]) or not isinstance(row[date_col], pd.Timestamp): return None
        date_str = row[date_col].strftime('%Y%m%d')
        home = re.sub(r'\W+', '', str(row[home_col]))
        away = re.sub(r'\W+', '', str(row[away_col]))
        if not home or not away or home == 'nan' or away == 'nan': return None
        return f"{date_str}_{home}_{away}"
    except Exception as e:
        logging.error(f"Err creating MatchID. Row: H={row.get('HomeTeam','?')}, A={row.get('AwayTeam','?')}, Dt={row.get('Date','?')}. Err: {e}", exc_info=False)
        return None

def extract_mongo_stats(stats_list_with_context: list) -> dict:
    extracted = {std_name: np.nan for std_name in ALL_MATCH_SPECIFIC_STATS_COLS}
    if not isinstance(stats_list_with_context, list): return extracted
    for team_stats_block in stats_list_with_context:
        venue = team_stats_block.get('_team_context_venue')
        if not venue: continue
        stats_items = team_stats_block.get('statistics', [])
        if not isinstance(stats_items, list): continue
        for stat_item in stats_items:
            if isinstance(stat_item, dict):
                stat_type = stat_item.get('type'); value = stat_item.get('value')
                if stat_type in MONGO_STATS_TYPE_MAP:
                    target_col_tuple = MONGO_STATS_TYPE_MAP[stat_type]
                    target_col_name = target_col_tuple[0] if venue == 'home' else target_col_tuple[1]
                    processed_value = np.nan
                    if value is not None:
                        if isinstance(value, str):
                            if '%' in value:
                                try: processed_value = float(value.replace('%',''))
                                except: pass
                            elif value.strip().lower() in ['n/a', '-', '']: processed_value = np.nan
                            else: processed_value = pd.to_numeric(value, errors='coerce')
                        else: processed_value = pd.to_numeric(value, errors='coerce')
                    if pd.notna(processed_value): extracted[target_col_name] = processed_value
    return extracted

# --- Main Processing Functions ---

def load_and_standardize_mongo_data(mongo_uri: str, db_name: str, collection_name: str, team_mapping: dict) -> pd.DataFrame:
    """Loads data from MongoDB, extracts relevant fields, standardizes them."""
    logging.info(f"Connecting to MongoDB: {mongo_uri}...")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=15000) # Increased timeout
        client.server_info(); db = client[db_name]; collection = db[collection_name]
        logging.info(f"Connected to DB: {db_name}, Collection: {collection_name}")
    except Exception as e: logging.error(f"MongoDB connection failed: {e}"); return pd.DataFrame()

    logging.info("Loading data from MongoDB...")
    try:
        projection = { # Projection based on sample data
            "_id": 1, "fixture_details.fixture.id": 1, "fixture_details.fixture.referee": 1,
            "fixture_details.fixture.date": 1, "fixture_details.fixture.timestamp": 1,
            "fixture_details.fixture.venue": 1, "fixture_details.fixture.status": 1,
            "fixture_details.league.id": 1, "fixture_details.league.name": 1, "fixture_details.league.country": 1,
            "fixture_details.league.season": 1, "fixture_details.league.round": 1,
            "fixture_details.teams.home.id": 1, "fixture_details.teams.home.name": 1, "fixture_details.teams.home.winner": 1,
            "fixture_details.teams.away.id": 1, "fixture_details.teams.away.name": 1, "fixture_details.teams.away.winner": 1,
            "fixture_details.goals.home": 1, "fixture_details.goals.away": 1,
            "fixture_details.score.halftime.home": 1, "fixture_details.score.halftime.away": 1,
            "statistics_full": 1, "statistics_half": 1, "lineups": 1, "predictions": 1, "odds_data": 1,
        }
        mongo_docs = list(collection.find({}, projection))
        logging.info(f"Loaded {len(mongo_docs)} documents from MongoDB.")
    except Exception as e: logging.error(f"Error loading data: {e}", exc_info=True); client.close(); return pd.DataFrame()
    finally: client.close()

    if not mongo_docs: logging.warning("No documents found."); return pd.DataFrame()

    processed_rows = []; processed_fixture_ids = set()
    skipped_count = 0; stat_fail_count = 0

    for i, doc in enumerate(mongo_docs):
        # Simplified progress logging
        # if i % 5000 == 0 and i > 0: logging.info(f"Processing MongoDB doc {i}/{len(mongo_docs)}...")
        row = {}
        try:
            fd = doc.get('fixture_details'); fx = fd.get('fixture') if isinstance(fd, dict) else None
            lg = fd.get('league') if isinstance(fd, dict) else None; tm = fd.get('teams') if isinstance(fd, dict) else None
            gl = fd.get('goals') if isinstance(fd, dict) else None; sc = fd.get('score', {}) if isinstance(fd, dict) else {}
            vn = fx.get('venue', {}) if isinstance(fx, dict) else {}; st = fx.get('status', {}) if isinstance(fx, dict) else {}
            hm = tm.get('home') if isinstance(tm, dict) else None; aw = tm.get('away') if isinstance(tm, dict) else None
            if not all([isinstance(o, dict) for o in [fx, lg, tm, gl, hm, aw]]): skipped_count+=1; continue

            fixture_id = fx.get('id'); home_team_id = hm.get('id'); away_team_id = aw.get('id')
            home_name = hm.get('name'); away_name = aw.get('name'); league_id = lg.get('id'); season = lg.get('season')
            if not all([fixture_id, home_team_id, away_team_id, home_name, away_name, league_id, season]): skipped_count+=1; continue
            if fixture_id in processed_fixture_ids: skipped_count+=1; continue

            date_str = fx.get('date'); timestamp = fx.get('timestamp')
            parsed_dt = pd.NaT
            if date_str: parsed_dt = parse_date(date_str)
            if pd.isna(parsed_dt) and timestamp: parsed_dt = parse_date(timestamp)
            if pd.isna(parsed_dt): skipped_count+=1; continue

            row['Date'] = parsed_dt; row['Timestamp'] = pd.to_numeric(timestamp, errors='coerce') if timestamp else parsed_dt.timestamp()
            row['HomeTeam'] = standardize_team_names(home_name, config.TEAM_NAME_MAPPING); row['AwayTeam'] = standardize_team_names(away_name, config.TEAM_NAME_MAPPING)
            row['HomeTeamID'] = home_team_id; row['AwayTeamID'] = away_team_id
            row['HomeTeamWinner'] = hm.get('winner'); row['AwayTeamWinner'] = aw.get('winner')

            row['FTHG'] = pd.to_numeric(gl.get('home'), errors='coerce'); row['FTAG'] = pd.to_numeric(gl.get('away'), errors='coerce')
            if pd.isna(row['FTHG']) or pd.isna(row['FTAG']): skipped_count+=1; continue
            row['FTR'] = 'H' if row['FTHG'] > row['FTAG'] else ('A' if row['FTHG'] < row['FTAG'] else 'D')

            ht_score = sc.get('halftime', {}); row['HTHG'] = pd.to_numeric(ht_score.get('home'), errors='coerce') if isinstance(ht_score, dict) else pd.NA
            row['HTAG'] = pd.to_numeric(ht_score.get('away'), errors='coerce') if isinstance(ht_score, dict) else pd.NA
            row['HTR'] = ('H' if row['HTHG'] > row['HTAG'] else ('A' if row['HTHG'] < row['HTAG'] else 'D')) if pd.notna(row['HTHG']) and pd.notna(row['HTAG']) else None

            row['LeagueID'] = league_id; row['LeagueName'] = lg.get('name'); row['Country'] = lg.get('country'); row['Season'] = season
            row['Round'] = lg.get('round'); row['Referee'] = fx.get('referee'); row['VenueName'] = vn.get('name'); row['VenueCity'] = vn.get('city')
            row['StatusLong'] = st.get('long'); row['StatusShort'] = st.get('short'); row['StatusElapsed'] = pd.to_numeric(st.get('elapsed'), errors='coerce')

            row['MatchID'] = create_match_id(row, date_col='Date', home_col='HomeTeam', away_col='AwayTeam')
            if not row['MatchID']: skipped_count+=1; continue

            stats_list_full = doc.get('statistics_full', []); stats_list_half = doc.get('statistics_half', [])
            stats_to_process = stats_list_full if isinstance(stats_list_full, list) and stats_list_full else (stats_list_half if isinstance(stats_list_half, list) and stats_list_half else None)
            extracted_stats = {std_name: np.nan for std_name in ALL_MATCH_SPECIFIC_STATS_COLS}
            if stats_to_process:
                stats_list_with_context = []
                for team_stat_block in stats_to_process:
                    if isinstance(team_stat_block, dict) and 'team' in team_stat_block and isinstance(team_stat_block['team'], dict):
                        current_team_id = team_stat_block['team'].get('id')
                        venue_context = 'home' if current_team_id == home_team_id else ('away' if current_team_id == away_team_id else None)
                        if venue_context:
                            context_block = team_stat_block.copy(); context_block['_team_context_venue'] = venue_context
                            stats_list_with_context.append(context_block)
                if stats_list_with_context: extracted_stats = extract_mongo_stats(stats_list_with_context)
            if pd.Series(extracted_stats).isna().all(): stat_fail_count += 1
            row.update(extracted_stats)

            lineups_list = doc.get('lineups', []); row['HomeFormation'] = None; row['AwayFormation'] = None
            if isinstance(lineups_list, list):
                for lineup_block in lineups_list:
                    if isinstance(lineup_block, dict):
                         team_info = lineup_block.get('team', {}); formation = lineup_block.get('formation')
                         current_team_id = team_info.get('id') if isinstance(team_info, dict) else None
                         if formation and current_team_id:
                             if current_team_id == home_team_id: row['HomeFormation'] = formation
                             elif current_team_id == away_team_id: row['AwayFormation'] = formation

            for target_name in ALL_ODDS_COLS: row[target_name] = np.nan
            odds_data = doc.get('odds_data'); predictions_data = doc.get('predictions', {}).get('predictions', {})
            if isinstance(odds_data, dict):
                for target_name, source_options in ODDS_COLS_MAP.items():
                     for source_key in source_options:
                         if source_key in odds_data and odds_data[source_key] not in [None, '']:
                             row[target_name] = pd.to_numeric(odds_data[source_key], errors='coerce'); break
            elif isinstance(predictions_data, dict) and 'percent' in predictions_data:
                 perc = predictions_data.get('percent', {}); hp=np.nan; dp=np.nan; ap=np.nan
                 if isinstance(perc, dict):
                    try: hp = float(str(perc.get('home', '')).replace('%', ''))
                    except: pass
                    try: dp = float(str(perc.get('draw', '')).replace('%', ''))
                    except: pass
                    try: ap = float(str(perc.get('away', '')).replace('%', ''))
                    except: pass
                    if pd.notna(hp) and hp > 0: row['OddsH'] = 100 / hp
                    if pd.notna(dp) and dp > 0: row['OddsD'] = 100 / dp
                    if pd.notna(ap) and ap > 0: row['OddsA'] = 100 / ap

            processed_rows.append(row); processed_fixture_ids.add(fixture_id)
        except Exception as e:
            logging.error(f"Error processing Mongo doc ID {doc.get('_id', 'N/A')} (Fixture: {fixture_id}): {e}", exc_info=False)
            skipped_count += 1

    if skipped_count > 0: logging.warning(f"Skipped {skipped_count} MongoDB documents due to missing data or errors.")
    if stat_fail_count > 0: logging.warning(f"{stat_fail_count} matches had no extractable stats (check 'statistics_full'/'half').")
    if not processed_rows: logging.error("No MongoDB documents successfully processed."); return pd.DataFrame()

    mongo_df = pd.DataFrame(processed_rows)
    logging.info(f"Successfully processed {len(mongo_df)} MongoDB documents into DataFrame.")
    return mongo_df

def calculate_rolling_features(df: pd.DataFrame, windows: list = [5, 10, 15]) -> pd.DataFrame:
    """ Calculates rolling features including Avg Goals Scored and BTTS Ratio, grouped by TeamID. """
    if df.empty: logging.warning("Input DF empty, skipping rolling features."); return df

    # Essential columns now include Team IDs
    essential_cols = ['Date', 'HomeTeam', 'AwayTeam', 'HomeTeamID', 'AwayTeamID', 'FTR', 'MatchID', 'FTHG', 'FTAG']
    existing_match_stats = [col for col in ALL_MATCH_SPECIFIC_STATS_COLS if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    missing_essentials = [col for col in essential_cols if col not in df.columns]
    if missing_essentials: logging.error(f"Missing essential columns for ID-based rolling features: {missing_essentials}. Skipping."); return df

    logging.info(f"Calculating rolling features using TeamID (Windows: {windows}). Includes AvgGoalsScored & BTTS Ratio.")

    # 1. Ensure Chronological Order & Unique MatchID
    df = df.sort_values(by='Date').reset_index(drop=True)
    if df['MatchID'].duplicated().any():
        logging.warning(f"Found {df['MatchID'].duplicated().sum()} duplicate MatchIDs. Keeping first occurrence.")
        df = df.drop_duplicates(subset=['MatchID'], keep='first')

    # Ensure Team IDs are suitable for grouping (e.g., Int64 or string, handle potential NA)
    for id_col in ['HomeTeamID', 'AwayTeamID']:
        if id_col in df:
            df[id_col] = pd.to_numeric(df[id_col], errors='coerce').astype('Int64') # Use nullable Int
        else:
            logging.error(f"Required column '{id_col}' not found. Cannot proceed with ID-based grouping.")
            return df
    # Drop rows where Team IDs are missing, as they cannot be grouped
    initial_rows = len(df)
    df.dropna(subset=['HomeTeamID', 'AwayTeamID'], inplace=True)
    if len(df) < initial_rows:
         logging.warning(f"Dropped {initial_rows - len(df)} rows due to missing HomeTeamID or AwayTeamID.")
    if df.empty:
        logging.error("No valid rows remain after dropping matches with missing Team IDs.")
        return df


    # 2. Identify Stats for Canonical View
    stats_for_canonical = sorted(list(set(existing_match_stats + ['FTHG', 'FTAG'])))

    # 3. Create Canonical View using Team IDs
    df_home = df.copy(); df_away = df.copy()
    df_home['TeamID'] = df['HomeTeamID']; df_home['OpponentID'] = df['AwayTeamID']; df_home['Venue'] = 'Home'
    df_away['TeamID'] = df['AwayTeamID']; df_away['OpponentID'] = df['HomeTeamID']; df_away['Venue'] = 'Away'
    # Keep original Team Names for reference if needed later, but don't use for grouping
    df_home['TeamName'] = df['HomeTeam']; df_home['OpponentName'] = df['AwayTeam']
    df_away['TeamName'] = df['AwayTeam']; df_away['OpponentName'] = df['HomeTeam']

    df_home['Result'] = df['FTR'].map({'H': 'W', 'D': 'D', 'A': 'L'}); df_away['Result'] = df['FTR'].map({'H': 'L', 'D': 'D', 'A': 'W'})
    df_home['Points'] = df_home['Result'].map({'W': 3, 'D': 1, 'L': 0}).astype('Int64'); df_away['Points'] = df_away['Result'].map({'W': 3, 'D': 1, 'L': 0}).astype('Int64')

    canonical_stat_names = set()
    # Use TeamID and OpponentID as the core identifiers
    canonical_base_cols = ['MatchID', 'Date', 'TeamID', 'OpponentID', 'Venue', 'Result', 'Points', 'TeamName', 'OpponentName'] # Added names for context
    for col in stats_for_canonical: # Map stats to For/Against
        base_name = None
        if col.startswith('Home'): base_name = col[4:]
        elif col.startswith('Away'): base_name = col[4:]
        elif col == 'FTHG': base_name = 'Goals'
        elif col == 'FTAG': base_name = 'Goals'
        if not base_name: continue

        stat_for = f"{base_name}For"; stat_against = f"{base_name}Against"
        canonical_stat_names.add(stat_for); canonical_stat_names.add(stat_against)

        if (col.startswith('Home') or col == 'FTHG') and col in df_home:
             df_home[stat_for] = pd.to_numeric(df_home[col], errors='coerce')
        elif (col.startswith('Away') or col == 'FTAG') and col in df_away:
             df_away[stat_for] = pd.to_numeric(df_away[col], errors='coerce')

        if col.startswith('Home') or col == 'FTHG':
            away_equiv = 'Away' + base_name if col.startswith('Home') else ('FTAG' if col == 'FTHG' else None)
            if away_equiv and away_equiv in df_home:
                df_home[stat_against] = pd.to_numeric(df_home[away_equiv], errors='coerce')
            elif stat_against not in df_home: df_home[stat_against] = np.nan
        if col.startswith('Away') or col == 'FTAG':
            home_equiv = 'Home' + base_name if col.startswith('Away') else ('FTHG' if col == 'FTAG' else None)
            if home_equiv and home_equiv in df_away:
                 df_away[stat_against] = pd.to_numeric(df_away[home_equiv], errors='coerce')
            elif stat_against not in df_away: df_away[stat_against] = np.nan

    list_of_canonical_stats_for_avg = sorted([s for s in canonical_stat_names if 'Points' not in s and 'GoalsAgainst' not in s])
    list_of_canonical_stats_incl_points = sorted(list(canonical_stat_names) + ['Points'])
    final_canonical_cols = sorted(list(set(canonical_base_cols + list_of_canonical_stats_incl_points)))

    # Ensure all necessary columns exist before selection
    actual_home_cols = [c for c in final_canonical_cols if c in df_home.columns]
    actual_away_cols = [c for c in final_canonical_cols if c in df_away.columns]
    df_home_subset = df_home[actual_home_cols]
    df_away_subset = df_away[actual_away_cols]
    df_canonical = pd.concat([df_home_subset, df_away_subset], ignore_index=True, sort=False)


    # --- Add BTTS Flag ---
    if 'GoalsFor' in df_canonical:
        df_canonical['GoalsFor'] = pd.to_numeric(df_canonical['GoalsFor'], errors='coerce')
    else:
        logging.warning("Canonical 'GoalsFor' column missing for BTTS calculation.")
        df_canonical['GoalsFor'] = np.nan
    if 'GoalsAgainst' in df_canonical:
        df_canonical['GoalsAgainst'] = pd.to_numeric(df_canonical['GoalsAgainst'], errors='coerce')
    else:
         logging.warning("Canonical 'GoalsAgainst' column missing for BTTS calculation.")
         df_canonical['GoalsAgainst'] = np.nan

    # Calculate BTTS flag using np.select for potentially better handling of pd.NA
    conditions = [
        (df_canonical['GoalsFor'] > 0) & (df_canonical['GoalsAgainst'] > 0), # Both scored -> 1
        pd.notna(df_canonical['GoalsFor']) & pd.notna(df_canonical['GoalsAgainst']) # Neither NA, but not both > 0 -> 0
    ]
    choices = [1, 0]
    # Create the column with np.select, default is NaN which pd.NA represents better later
    # Assign to a temporary column first
    df_canonical['BTTS_Flag_temp'] = np.select(conditions, choices, default=np.nan)

    # Now convert the temporary column to Int64, handling NaN correctly
    try:
        # Explicitly convert to numeric (which handles NaN), then cast to Int64
        df_canonical['BTTS_Flag'] = pd.to_numeric(df_canonical['BTTS_Flag_temp'], errors='coerce').astype('Int64')
    except Exception as e:
         logging.error(f"Failed to convert BTTS_Flag_temp to Int64: {e}. Check intermediate values.")
         # Fallback: Keep as float64 if Int64 cast fails
         df_canonical['BTTS_Flag'] = pd.to_numeric(df_canonical['BTTS_Flag_temp'], errors='coerce')

    # Drop the temporary column if it exists
    if 'BTTS_Flag_temp' in df_canonical.columns:
        df_canonical.drop(columns=['BTTS_Flag_temp'], inplace=True)
    # Log dtype for verification
    if 'BTTS_Flag' in df_canonical.columns:
        logging.info(f"BTTS_Flag column created with dtype: {df_canonical['BTTS_Flag'].dtype}")
    else:
        logging.warning("BTTS_Flag column creation failed or was skipped.")


    # 4. Sort by TeamID, then Date (Crucial!)
    # Ensure Date is datetime for sorting
    df_canonical['Date'] = pd.to_datetime(df_canonical['Date'])
    df_canonical = df_canonical.sort_values(by=['TeamID', 'Date']).reset_index(drop=True)
    logging.info(f"Canonical DataFrame created using TeamID (Shape: {df_canonical.shape}). Added GoalsFor/Against/BTTS_Flag.")

    # 5. Pre-calculate W/D/L flags
    for r_type in ['W', 'D', 'L']: df_canonical[f'Is_{r_type}'] = (df_canonical['Result'] == r_type).astype(int)


    # --- Calculate Rolling Features - Corrected Approach (Inner function mostly unchanged, uses grouped data) ---
    # The calculate_team_rolling_features_corrected function definition remains the same
    # as it operates on the 'team_data' passed to it, which will be grouped by TeamID later.
    # We might update logging inside it if needed.
    def calculate_team_rolling_features_corrected(team_data, window_list, stats_to_avg):
        # Get TeamID for logging
        team_id_for_log = team_data['TeamID'].iloc[0] if not team_data.empty and 'TeamID' in team_data.columns else 'Unknown ID'

        team_data = team_data.set_index('Date', drop=False) # Use Date index for rolling
        if team_data.index.has_duplicates:
             logging.warning(f"Team ID {team_id_for_log} has duplicate dates. Aggregating values before rolling.")
             # Aggregate values for duplicate dates
             agg_funcs = {col: 'mean' for col in stats_to_avg if col != 'BTTS_Flag'} # Avg most stats
             agg_funcs.update({col: 'sum' for col in ['Points', 'Is_W', 'Is_D', 'Is_L'] if col in team_data.columns})
             agg_funcs.update({'BTTS_Flag': 'mean'})
             # Keep other identifying columns like MatchID, Venue, TeamID etc. using 'first'
             id_cols = ['MatchID', 'TeamID', 'OpponentID', 'Venue', 'Result', 'Date', 'TeamName', 'OpponentName'] # Added names
             agg_funcs.update({col: 'first' for col in id_cols if col in team_data.columns and col != 'Date'}) # Exclude Date as it's index
             other_cols = [c for c in team_data.columns if c not in agg_funcs and c not in id_cols]
             agg_funcs.update({col: 'first' for col in other_cols})

             try:
                 team_data = team_data.groupby(level=0).agg(agg_funcs)
                 if team_data.index.name != 'Date':
                      team_data = team_data.reset_index().set_index('Date', drop=False)
                 logging.info(f"Aggregated duplicate dates for team ID {team_id_for_log}.")
             except Exception as agg_e:
                  logging.error(f"Error aggregating duplicate dates for team ID {team_id_for_log}: {agg_e}. Skipping aggregation.")
                  team_data = team_data[~team_data.index.duplicated(keep='first')]

        features_list = []
        shifted_data = team_data.shift(1)

        for current_date, current_row in team_data.iterrows():
             # Use MatchID and Venue from the current row for linking features back
             row_features = {'MatchID': current_row['MatchID'], 'Venue': current_row['Venue']}
             if pd.isna(row_features['MatchID']) or pd.isna(row_features['Venue']):
                 logging.warning(f"Missing MatchID or Venue for TeamID {team_id_for_log} on date {current_date}. Skipping row feature calculation.")
                 continue # Skip if key identifiers are missing

             shifted_data.index = pd.to_datetime(shifted_data.index)
             current_date_ts = pd.to_datetime(current_date)
             hist_data = shifted_data[shifted_data.index < current_date_ts]
             hist_home = hist_data[hist_data['Venue'] == 'Home']
             hist_away = hist_data[hist_data['Venue'] == 'Away']

             for W in window_list:
                ws = f"_Last{W}"
                hist_data_w = hist_data.tail(W)
                hist_home_w = hist_home.tail(W)
                hist_away_w = hist_away.tail(W)

                # --- Total Context ---
                ctx = '_Total'; pts_col = 'Points'; wdl_cols = ['Is_W', 'Is_D', 'Is_L']
                if not hist_data_w.empty:
                    if pts_col in hist_data_w: row_features[f'FormPoints{ctx}{ws}'] = pd.to_numeric(hist_data_w[pts_col], errors='coerce').sum(min_count=1)
                    for r_col in wdl_cols:
                        if r_col in hist_data_w: row_features[f'{r_col[3:]}_Count{ctx}{ws}'] = pd.to_numeric(hist_data_w[r_col], errors='coerce').sum(min_count=1)
                    for stat in stats_to_avg:
                        if stat in hist_data_w and stat != 'BTTS_Flag':
                            avg_col_name = f'AvgGoalsScored{ctx}{ws}' if stat == 'GoalsFor' else f'Avg{stat}{ctx}{ws}'
                            numeric_stat = pd.to_numeric(hist_data_w[stat], errors='coerce')
                            row_features[avg_col_name] = numeric_stat.mean() if numeric_stat.notna().any() else np.nan
                    if 'GoalsAgainst' in hist_data_w:
                        numeric_stat_against = pd.to_numeric(hist_data_w['GoalsAgainst'], errors='coerce')
                        row_features[f'AvgGoalsConceded{ctx}{ws}'] = numeric_stat_against.mean() if numeric_stat_against.notna().any() else np.nan
                    if 'BTTS_Flag' in hist_data_w:
                        btts_flag_numeric = pd.to_numeric(hist_data_w['BTTS_Flag'], errors='coerce')
                        row_features[f'BTTS_Ratio{ctx}{ws}'] = btts_flag_numeric.mean() if btts_flag_numeric.notna().any() else np.nan
                    else: row_features[f'BTTS_Ratio{ctx}{ws}'] = np.nan
                # Add else block to explicitly set NaNs if hist_data_w is empty
                else:
                    if pts_col in team_data.columns: row_features[f'FormPoints{ctx}{ws}'] = np.nan
                    for r_col in wdl_cols:
                         if r_col in team_data.columns: row_features[f'{r_col[3:]}_Count{ctx}{ws}'] = np.nan
                    for stat in stats_to_avg:
                        if stat in team_data.columns and stat != 'BTTS_Flag':
                             avg_col_name = f'AvgGoalsScored{ctx}{ws}' if stat == 'GoalsFor' else f'Avg{stat}{ctx}{ws}'
                             row_features[avg_col_name] = np.nan
                    if 'GoalsAgainst' in team_data.columns: row_features[f'AvgGoalsConceded{ctx}{ws}'] = np.nan
                    if 'BTTS_Flag' in team_data.columns: row_features[f'BTTS_Ratio{ctx}{ws}'] = np.nan


             features_list.append(row_features)

        team_features_df = pd.DataFrame(features_list)
        if 'MatchID' not in team_features_df.columns and not team_features_df.empty:
             logging.error(f"MatchID lost during feature calculation for TeamID {team_id_for_log}. Check aggregation/processing logic.")
             # Attempt recovery or return empty to signal failure
             return pd.DataFrame() # Return empty if MatchID is lost

        return team_features_df


    logging.info("Applying rolling calculations per team ID (corrected method)...")
    # Group by TeamID now
    df_canonical = df_canonical.sort_values(by=['TeamID', 'Date', 'MatchID'], ascending=[True, True, True]).reset_index(drop=True)
    # Check if TeamID column exists before grouping
    if 'TeamID' not in df_canonical.columns:
         logging.error("'TeamID' column not found in df_canonical before grouping. Cannot proceed.")
         return df
    grouped_data = df_canonical.groupby('TeamID', group_keys=False, sort=False)

    all_teams_features_list = []
    for team_id, group in grouped_data: # Iterate over team_id and group
         if group.empty:
             logging.warning(f"Group for team ID '{team_id}' is empty. Skipping rolling calculation.")
             continue
         try:
             team_features = calculate_team_rolling_features_corrected(group.copy(), windows, list_of_canonical_stats_for_avg)
             if not team_features.empty:
                  all_teams_features_list.append(team_features)
             else:
                  logging.warning(f"No features calculated for team ID '{team_id}'.")
         except Exception as e:
             logging.error(f"Error calculating rolling features for team ID '{team_id}': {e}", exc_info=True)


    logging.info("Finished applying rolling calculations.")

    if not all_teams_features_list:
        logging.error("No features calculated across all teams using TeamID. Problem during apply step or no valid data.")
        return df

    # Concatenate features from all teams
    rolling_features_df = pd.concat(all_teams_features_list, ignore_index=True)
    logging.info(f"Concatenated rolling features shape (grouped by TeamID): {rolling_features_df.shape}")
    if rolling_features_df.empty:
         logging.error("Concatenated rolling features DataFrame is empty.")
         return df

    # --- Merge features back to original match-centric DataFrame (using MatchID) ---
    # This part remains the same as merging is based on MatchID, not TeamID
    if 'MatchID' not in rolling_features_df.columns or 'Venue' not in rolling_features_df.columns:
        logging.error("MatchID or Venue missing from calculated features DataFrame (TeamID grouped). Cannot merge back.")
        logging.info(f"Columns in rolling_features_df: {rolling_features_df.columns.tolist()[:20]}...")
        return df

    rolling_features_df['MatchID'] = rolling_features_df['MatchID'].astype(str)
    df['MatchID'] = df['MatchID'].astype(str)

    feature_cols = [col for col in rolling_features_df.columns if col not in ['MatchID', 'Venue']]
    if not feature_cols:
        logging.error("No feature columns identified in rolling_features_df (TeamID grouped) after excluding MatchID/Venue.")
        return df

    df_home_rf = rolling_features_df[rolling_features_df['Venue'] == 'Home'].drop(columns=['Venue'])
    df_away_rf = rolling_features_df[rolling_features_df['Venue'] == 'Away'].drop(columns=['Venue'])

    if df_home_rf['MatchID'].duplicated().any():
        logging.warning(f"Duplicate MatchIDs found in HOME rolling features (TeamID grouped). Keeping first.")
        df_home_rf = df_home_rf.drop_duplicates(subset=['MatchID'], keep='first')
    if df_away_rf['MatchID'].duplicated().any():
        logging.warning(f"Duplicate MatchIDs found in AWAY rolling features (TeamID grouped). Keeping first.")
        df_away_rf = df_away_rf.drop_duplicates(subset=['MatchID'], keep='first')

    home_rename = {col: f'Home_{col}' for col in feature_cols if col in df_home_rf}
    away_rename = {col: f'Away_{col}' for col in feature_cols if col in df_away_rf}

    df_home_rf = df_home_rf.rename(columns=home_rename)
    df_away_rf = df_away_rf.rename(columns=away_rename)

    merge_count_before = len(df)
    original_cols = set(df.columns) # Store original columns before merge

    if not df_home_rf.empty:
        # Explicitly check if MatchID column exists before merge
        if 'MatchID' not in df_home_rf.columns:
             logging.error("MatchID missing from df_home_rf before merge.")
             return df # Or handle error appropriately
        df = pd.merge(df, df_home_rf, on='MatchID', how='left', suffixes=('', '_DROP_H'))
        cols_to_drop_h = [c for c in df.columns if c.endswith('_DROP_H')]
        if cols_to_drop_h:
             logging.info(f"Dropping columns due to merge suffix _DROP_H: {cols_to_drop_h}")
             df = df.drop(columns=cols_to_drop_h)

    if not df_away_rf.empty:
         # Explicitly check if MatchID column exists before merge
        if 'MatchID' not in df_away_rf.columns:
             logging.error("MatchID missing from df_away_rf before merge.")
             return df # Or handle error appropriately
        df = pd.merge(df, df_away_rf, on='MatchID', how='left', suffixes=('', '_DROP_A'))
        cols_to_drop_a = [c for c in df.columns if c.endswith('_DROP_A')]
        if cols_to_drop_a:
             logging.info(f"Dropping columns due to merge suffix _DROP_A: {cols_to_drop_a}")
             df = df.drop(columns=cols_to_drop_a)


    if len(df) != merge_count_before:
        logging.warning(f"Row count changed during merge (Before: {merge_count_before}, After: {len(df)}). Check for MatchID issues.")

    logging.info(f"Finished merging rolling features (grouped by TeamID). Final DF shape: {df.shape}")
    new_cols_added = set(df.columns) - original_cols # Use original cols before merge
    logging.info(f"Sample of newly added columns (TeamID grouped): {sorted(list(new_cols_added))[:20]}...") # Increased sample size

    return df


def final_clean_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """Performs final cleaning, type casting, sorting, and column reordering."""
    if df.empty: logging.warning("Input DF to final_clean is empty."); return df
    logging.info("Performing final cleaning, type casting, and sorting...")

    # --- Date/Timestamp ---
    if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    initial_rows = len(df); df = df.dropna(subset=['Date']) # Date is essential
    if len(df) < initial_rows: logging.warning(f"Dropped {initial_rows - len(df)} rows due to invalid Date.")
    if df.empty: return df
    if 'Timestamp' in df.columns: df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')

    # --- Integer Columns ---
    rolling_count_cols = [col for col in df.columns if 'Count_Last' in col or 'FormPoints_Last' in col]
    match_int_stats = [col for col in df.columns if 'YellowCards' in col or 'RedCards' in col] # Example: Cards are usually int
    int_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'StatusElapsed', 'Season'] + rolling_count_cols + match_int_stats
    int_cols.extend(['HomeTeamID', 'AwayTeamID']) # Add integer Team IDs if keeping them
    int_cols = sorted(list(set([c for c in int_cols if c in df.columns]))) # Keep only existing cols

    for col in int_cols:
        # Convert to numeric first, coercing errors. This might create floats (due to NaN).
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        # Round the series to handle any potential non-integer floats introduced.
        # This assumes rounding is the correct approach for these integer columns.
        # NaN values remain NaN after rounding.
        rounded_series = numeric_series.round(0)
        # Now, safely cast to nullable Int64. NaNs will become pd.NA.
        try:
            df[col] = rounded_series.astype('Int64')
        except TypeError as e:
             logging.error(f"Failed to cast column '{col}' to Int64 after rounding. Error: {e}. Check data for unexpected non-numeric values not handled by coerce/round.")
             # Decide on fallback: leave as float, fillna(0).astype(int), etc. Leaving as float for now.
             df[col] = rounded_series # Keep as float if Int64 cast fails even after rounding

    # --- Float Columns ---
    rolling_avg_cols = [col for col in df.columns if 'Avg' in col and '_Last' in col]
    match_float_stats = [col for col in ALL_MATCH_SPECIFIC_STATS_COLS if col in df.columns and col not in int_cols and col not in ['FTHG', 'FTAG', 'HTHG', 'HTAG']]
    odds_cols = [col for col in ALL_ODDS_COLS if col in df.columns]
    float_cols = match_float_stats + odds_cols + rolling_avg_cols
    # Ensure we don't try to convert columns that failed the Int64 cast and remained float
    float_cols = sorted(list(set([c for c in float_cols if c in df.columns and c not in int_cols])))


    for col in float_cols:
         # Check if it's not already numeric (it might be if it came from numeric_series)
         if not pd.api.types.is_numeric_dtype(df[col]):
             df[col] = pd.to_numeric(df[col], errors='coerce')
         # Ensure it's float64 for consistency
         df[col] = df[col].astype('float64')


    # --- Boolean Columns ---
    bool_cols = ['HomeTeamWinner', 'AwayTeamWinner']
    for col in bool_cols:
        if col in df.columns: df[col] = df[col].map({True: True, 1: True, False: False, 0: False}).astype('boolean')

    # --- String Columns ---
    string_cols = ['HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'LeagueName', 'Country', 'Referee', 'VenueName', 'VenueCity', 'StatusLong', 'StatusShort', 'Round', 'HomeFormation', 'AwayFormation', 'MatchID']
    if 'LeagueID' in df.columns and not pd.api.types.is_numeric_dtype(df['LeagueID']): string_cols.append('LeagueID')
    string_cols = sorted(list(set([c for c in string_cols if c in df.columns]))) # Keep only existing cols

    for col in string_cols: df[col] = df[col].astype(object).replace(['nan', 'NaN', 'None', '', np.nan], pd.NA).astype('string')

    # --- Sort chronologically by Date ---
    try: df = df.sort_values(by='Date').reset_index(drop=True)
    except Exception as e: logging.error(f"Error sorting by Date: {e}. Skipping sort.")

    # --- Reorder columns ---
    core_ordered = [col for col in CORE_COLS if col in df.columns]
    match_stats_ordered = [col for col in ALL_MATCH_SPECIFIC_STATS_COLS if col in df.columns]
    odds_ordered = [col for col in ALL_ODDS_COLS if col in df.columns]
    rolling_features_ordered = sorted([col for col in df.columns if '_Last' in col]) # Group rolling features

    # Recalculate final_ordered_cols based on actual dtypes after potential casting failures
    current_int_cols = df.select_dtypes(include=['Int64', 'int64']).columns.tolist()
    current_float_cols = df.select_dtypes(include=['float64']).columns.tolist()
    current_bool_cols = df.select_dtypes(include=['boolean']).columns.tolist()
    current_string_cols = df.select_dtypes(include=['string']).columns.tolist()
    current_datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    current_other_cols = df.select_dtypes(exclude=['Int64', 'int64', 'float64', 'boolean', 'string', 'datetime64[ns]']).columns.tolist()

    # Define order preference
    col_order_preference = core_ordered + match_stats_ordered + odds_ordered + rolling_features_ordered

    # Build final list, ensuring all columns are included
    final_ordered_cols = [col for col in col_order_preference if col in df.columns]
    processed_cols = set(final_ordered_cols)
    final_ordered_cols.extend(sorted([col for col in df.columns if col not in processed_cols]))


    if set(final_ordered_cols) != set(df.columns):
        logging.warning(f"Column mismatch during reordering. Expected {len(final_ordered_cols)}, Found {len(df.columns)}. Using original columns as fallback.")
        final_ordered_cols = df.columns.tolist() # Fallback
    else:
        logging.info(f"Reordered {len(final_ordered_cols)} columns.")

    df = df[final_ordered_cols]


    logging.info("Final cleaning and type casting complete.")
    buffer = io.StringIO(); df.info(buf=buffer, verbose=True, show_counts=True, memory_usage='deep')
    logging.info(f"Final DataFrame info:\n{buffer.getvalue()}")
    return df


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Data Preparation Script ---")

    # 1. Load MongoDB Data
    master_data = load_and_standardize_mongo_data(
        mongo_uri=config.MONGO_URI, db_name=config.MONGO_DB_NAME,
        collection_name=config.MONGO_COLLECTION_NAME, team_mapping=config.TEAM_NAME_MAPPING
    )
    if master_data.empty: logging.error("Failed to load data from MongoDB. Exiting."); sys.exit(1)
    logging.info(f"Loaded MongoDB data shape: {master_data.shape}")

    # 2. Calculate Rolling Features
    master_data_with_features = calculate_rolling_features(master_data, windows=[5, 10, 15])
    if master_data_with_features.empty: logging.error("Rolling feature calculation failed. Exiting."); sys.exit(1)
    logging.info(f"Shape after adding rolling features: {master_data_with_features.shape}")

    # 3. Final Cleaning
    final_data = final_clean_and_order(master_data_with_features)
    if final_data.empty:
        logging.error("Final cleaning failed. Exiting.")
        sys.exit(1)
    logging.info(f"Shape after final cleaning: {final_data.shape}")

    # 4. Save Data
    try:
        # Define the output path for the Parquet file
        output_dir = "output/parquet"
        output_file = "mongo.parquet"
        output_path = os.path.join(output_dir, output_file)

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Final safety check to remove duplicated columns
        final_data = final_data.loc[:, ~final_data.columns.duplicated(keep='first')]
        logging.info(f"Shape before saving to Parquet: {final_data.shape}")

        # Save the DataFrame to a Parquet file
        final_data.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
        logging.info(f"Successfully saved data to: {output_path}")
        logging.info(f"Final DataFrame columns saved ({len(final_data.columns)}).")
    except Exception as e:
        logging.error(f"Failed to save data: {e}", exc_info=True)
        sys.exit(1)

    logging.info("--- Data Preparation Script Finished ---")