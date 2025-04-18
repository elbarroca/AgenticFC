import os
import logging
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure
from typing import Optional, Dict, Any, List, Set
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pymongo.results import UpdateResult # Import UpdateResult for type hinting

logger = logging.getLogger(__name__)

class MongoDBManager:
    _instance = None
    _client = None
    _db = None
    _matches_collection = None
    _standings_collection = None
    _odds_collection = None
    _team_fixtures_collection = None
    _statarea_collection = None # Added for StatArea data
    _daily_games_collection = None  # Collection for daily games summaries
    _match_processor_collection = None # <<< New Collection for MatchProcessor data
    _max_retries = 3

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MongoDBManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_name="agenticfc"):
        if hasattr(self, '_initialized') and self._initialized:
            # Prevent re-initialization if already initialized
            if self._db and self._db.name == db_name:
                 logger.debug(f"MongoDBManager already initialized with database: {db_name}")
                 return
            else:
                 logger.warning(f"Re-initializing MongoDBManager with different DB: {db_name}. Closing previous connection.")
                 self._reset_state() # Reset before re-initializing with new DB

        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        dotenv_path = project_root / '.env'
        logger.info(f"Attempting to load .env file from: {dotenv_path}")
        loaded = load_dotenv(dotenv_path=dotenv_path)
        if not loaded:
             logger.warning(f".env file not found or not loaded from {dotenv_path}. Ensure it exists in the project root.")

        mongo_uri = os.getenv("MONGO_URI")

        if not mongo_uri:
            logger.error("MONGO_URI environment variable not set or not found in .env.")
            raise ValueError("MONGO_URI environment variable not set or not found in .env.")
        else:
             # Basic obfuscation for logging
             uri_parts = mongo_uri.split('@')
             if len(uri_parts) == 2:
                  creds_part = uri_parts[0].split(':')
                  if len(creds_part) > 1:
                       logged_uri = f"{creds_part[0].split('://')[-1]}:*****@{uri_parts[1]}"
                  else: # Handle URI without password
                      logged_uri = f"{creds_part[0].split('://')[-1]}@******" # Mask potentially username only part
             else: # Handle URI without credentials part
                  logged_uri = mongo_uri # Log as is if no '@' sign
             logger.info(f"Found MONGO_URI: {logged_uri}")


        retry_count = 0
        connected = False
        
        while not connected and retry_count < self._max_retries:
            try:
                retry_count += 1
                logger.info(f"Attempting to connect to MongoDB (attempt {retry_count}/{self._max_retries})...")
                self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000, appname="AgenticFC") # Added appname
                # The ismaster command is cheap and does not require auth.
                self._client.admin.command('ping') # Use ping instead of ismaster for modern MongoDB
                logger.info(f"Successfully connected to MongoDB server")
                
                self._db = self._client[db_name]
                logger.info(f"Using database: {db_name}")

                # Initialize all collections
                self._matches_collection = self._db['matches']
                self._standings_collection = self._db['standings']
                self._odds_collection = self._db['odds']
                self._team_fixtures_collection = self._db['team_season_fixtures']
                self._statarea_collection = self._db['statarea_stats']
                self._daily_games_collection = self._db['daily_games']
                self._match_processor_collection = self._db['match_processor'] # <<< Initialize new collection handle
                logger.info("Initialized collections: matches, standings, odds, team_season_fixtures, statarea_stats, daily_games, match_processor") # <<< Updated log

                # Ensure initialization is marked true *before* creating indexes
                self._initialized = True
                # Create necessary indexes only after successful initialization
                self._create_indexes()

                connected = True
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"MongoDB connection attempt {retry_count} failed: {e}")
                if retry_count < self._max_retries:
                    wait_time = 2 ** retry_count
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"MongoDB connection failed after {self._max_retries} attempts: {e}")
                    self._reset_state()
                    # Re-raise the error after exhausting retries
                    raise ConnectionFailure(f"MongoDB connection failed after {self._max_retries} attempts: {e}")
            except Exception as e: # Catch broader exceptions during init
                logger.error(f"An unexpected error occurred during MongoDB initialization: {e}", exc_info=True)
                self._reset_state()
                raise # Re-raise the caught exception

    def _create_indexes(self):
        """Creates necessary indexes for the collections if they don't exist."""
        if not self._initialized:
            logger.error("Cannot create indexes: DB not initialized.")
            return
        try:
            # Example indexes (add more as needed)
            if self._matches_collection is not None:
                self._matches_collection.create_index("date_utc", name="match_date_utc_idx")
                self._matches_collection.create_index("fixture_id", name="match_fixture_id_idx", unique=True) # Assuming fixture_id is unique in API data

            if self._odds_collection is not None:
                self._odds_collection.create_index("fixture_id", name="odds_fixture_id_idx", unique=True)
                self._odds_collection.create_index("match_date_utc", name="odds_match_date_utc_idx")

            if self._standings_collection is not None:
                self._standings_collection.create_index([("league_id", 1), ("season", 1), ("date_retrieved_utc", -1)], name="standings_league_season_date_idx")

            if self._team_fixtures_collection is not None:
                 self._team_fixtures_collection.create_index([("team_id", 1), ("season", 1)], name="team_season_idx", unique=True) # _id is already unique

            # Indexes for the StatArea collection (updated for time-series approach)
            if self._statarea_collection is not None:
                 # Index for finding recent/historical data by team, game type, and period
                 self._statarea_collection.create_index(
                     [("api_id", 1), ("game_type", 1), ("period", 1), ("scrape_date_utc", -1)], 
                     name="statarea_team_game_period_date_idx"
                 )
                 # Index for looking up by content hash (to check for duplicate content)
                 self._statarea_collection.create_index(
                     [("api_id", 1), ("game_type", 1), ("period", 1), ("content_hash", 1)],
                     name="statarea_content_hash_idx"
                 )
                 # Index for quick lookup by team api_id
                 self._statarea_collection.create_index("api_id", name="statarea_api_id_idx")
                 # Index for checking scrape date
                 self._statarea_collection.create_index("scrape_date_utc", name="statarea_scrape_date_idx")

            if self._daily_games_collection is not None:
                self._daily_games_collection.create_index(
                    "date", name="daily_games_date_idx", unique=True
                )
            
            # <<< Add index for the new collection
            if self._match_processor_collection is not None:
                 self._match_processor_collection.create_index("fixture_id", name="proc_fixture_id_idx", unique=True)
                 self._match_processor_collection.create_index("match_date_str", name="proc_match_date_str_idx")
                 self._match_processor_collection.create_index("processed_at_utc", name="proc_processed_at_utc_idx")

            logger.info("Finished creating/ensuring indexes.")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}", exc_info=True)
            # Decide if this should be fatal; often it's not critical for basic operation but impacts performance

    def _reset_state(self):
        """Helper to reset connection state variables."""
        logger.debug("Resetting MongoDBManager state...")
        if self._client:
            try:
                self._client.close()
                logger.debug("MongoDB client closed.")
            except Exception as close_e:
                logger.error(f"Error closing MongoDB client during reset: {close_e}")
        self._client = None
        self._db = None
        self._matches_collection = None
        self._standings_collection = None
        self._odds_collection = None
        self._team_fixtures_collection = None
        self._statarea_collection = None # Reset new collection handle
        self._daily_games_collection = None
        self._match_processor_collection = None # <<< Reset new collection handle
        self._initialized = False
        # Important: Reset the class-level singleton instance so __new__ creates a fresh one next time
        MongoDBManager._instance = None
        logger.debug("MongoDBManager state reset complete.")

    def close_connection(self):
        logger.info("Closing MongoDB connection.")
        self._reset_state()

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse date string into a datetime object."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
            return None

    def save_match_data(self, match_data: Dict[str, Any]):
        """Saves or updates detailed match data using the provided dictionary."""
        if not self._initialized or self._matches_collection is None:
            logger.error("MongoDBManager not initialized. Cannot save match data.")
            return False

        fixture_id = match_data.get("fixture_id") or match_data.get("_id")
        date_str = match_data.get("date_str") # For logging

        if not fixture_id:
            logger.error("Match data is missing 'fixture_id' or '_id'. Cannot save.")
            return False
        if not date_str:
            logger.warning(f"Match data for fixture {fixture_id} is missing 'date_str'. Proceeding with save.")

        try:
            match_data["_id"] = str(fixture_id)
            if not isinstance(match_data.get("date_utc"), datetime):
                parsed_date = self._parse_date_string(date_str)
                if parsed_date:
                     match_data["date_utc"] = datetime(parsed_date.year, parsed_date.month, parsed_date.day, 0, 0, 0, tzinfo=timezone.utc) # Ensure UTC
                else:
                    logger.error(f"Cannot determine valid date_utc for fixture {fixture_id}. Save might fail or have null date.")

            if not isinstance(match_data.get("fetch_timestamp_utc"), datetime):
                 match_data["fetch_timestamp_utc"] = datetime.now(timezone.utc)

            result = self._matches_collection.update_one(
                {"_id": str(fixture_id)},
                {"$set": match_data},
                upsert=True
            )
            op_type = "updated" if result.matched_count > 0 else "inserted"
            if result.upserted_id: op_type = "inserted"
            logger.info(f"Successfully {op_type} match data for fixture {fixture_id} ({date_str}). Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving/updating match data for fixture {fixture_id} to MongoDB: {e}", exc_info=True)
            return False

    def check_match_exists(self, fixture_id: str) -> bool:
        """Checks if a match with the given fixture ID exists in the 'matches' collection."""
        if not self._initialized or self._matches_collection is None: return False
        try:
            count = self._matches_collection.count_documents({"_id": str(fixture_id)})
            return count > 0
        except Exception as e:
            logger.error(f"Error checking match existence for fixture {fixture_id}: {e}")
            return False

    def get_match_summary(self, fixture_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves minimal summary data (e.g., _id, date_str) for a specific fixture."""
        if not self._initialized or self._matches_collection is None: return None
        try:
             return self._matches_collection.find_one(
                 {"_id": str(fixture_id)},
                 {"_id": 1, "date_str": 1}
            )
        except Exception as e:
            logger.error(f"Error retrieving match summary for fixture {fixture_id}: {e}")
            return None

    def get_match_fixture_ids_for_date(self, date_str: str) -> List[str]:
        """Gets all fixture IDs for matches on a specific date from the 'matches' collection."""
        if not self._initialized or self._matches_collection is None: return []
        try:
            parsed_date = self._parse_date_string(date_str)
            if not parsed_date: return []
            start_date = datetime(parsed_date.year, parsed_date.month, parsed_date.day, 0, 0, 0, tzinfo=timezone.utc)
            end_date = start_date + timedelta(days=1)
            cursor = self._matches_collection.find(
                {"date_utc": {"$gte": start_date, "$lt": end_date}},
                {"fixture_id": 1, "_id": 0}
            )
            fixture_ids = [doc["fixture_id"] for doc in cursor if "fixture_id" in doc]
            return fixture_ids
        except Exception as e:
            logger.error(f"Error getting fixture IDs for date {date_str}: {e}")
            return []

    def get_matches_for_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Retrieves all match documents for a specific date."""
        if not self._initialized or self._matches_collection is None: return []
        try:
            parsed_date = self._parse_date_string(date_str)
            if not parsed_date: return []
            start_date = datetime(parsed_date.year, parsed_date.month, parsed_date.day, 0, 0, 0, tzinfo=timezone.utc)
            end_date = start_date + timedelta(days=1)
            cursor = self._matches_collection.find(
                {"date_utc": {"$gte": start_date, "$lt": end_date}}
            )
            return list(cursor)
        except Exception as e:
            logger.error(f"Error getting matches for date {date_str}: {e}")
            return []

    def get_match_data(self, fixture_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves detailed match data for a specific fixture using _id."""
        if not self._initialized or self._matches_collection is None: return None
        try:
            return self._matches_collection.find_one({"_id": str(fixture_id)})
        except Exception as e:
            logger.error(f"Error retrieving match data for fixture {fixture_id}: {e}")
            return None

    def save_standings_data(self, date_str: str, league_id: str, season: int, standings_payload: Dict[str, Any]):
        """Saves league standings snapshot in the 'standings' collection."""
        if not self._initialized or self._standings_collection is None: return False
        try:
            parsed_date = self._parse_date_string(date_str)
            if not parsed_date: return False
            standings_payload["date_retrieved_utc"] = datetime(parsed_date.year, parsed_date.month, parsed_date.day, 0, 0, 0, tzinfo=timezone.utc)
            standings_payload["date_retrieved_str"] = date_str
            standings_payload["league_id"] = str(league_id)
            standings_payload["season"] = int(season)
            result = self._standings_collection.insert_one(standings_payload)
            logger.info(f"Saved standings snapshot for league {league_id}, season {season} on {date_str}. Inserted ID: {result.inserted_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving standings for league {league_id}, season {season} to MongoDB: {e}")
            return False

    def get_standings_data(self, league_id: str, season: int, date_str: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves the latest league standings or standings from a specific date."""
        if not self._initialized or self._standings_collection is None: return None
        try:
            query = {"league_id": str(league_id), "season": int(season)}
            sort_order = [("date_retrieved_utc", -1)]
            if date_str:
                parsed_date = self._parse_date_string(date_str)
                if parsed_date:
                    query_date = datetime(parsed_date.year, parsed_date.month, parsed_date.day, 23, 59, 59, tzinfo=timezone.utc)
                    query["date_retrieved_utc"] = {"$lte": query_date}
                else:
                    logger.warning(f"Invalid date {date_str} provided. Fetching latest.")
            return self._standings_collection.find_one(query, sort=sort_order)
        except Exception as e:
            logger.error(f"Error retrieving standings for league {league_id}, season {season}: {e}")
            return None

    def save_odds_data(self, date_str: str, fixture_id: str, odds_payload: Dict[str, Any]):
        """Saves or updates odds data for a specific fixture in the 'odds' collection."""
        if not self._initialized or self._odds_collection is None: return False
        try:
            parsed_match_date = self._parse_date_string(date_str)
            if not parsed_match_date:
                logger.error(f"Invalid match date format '{date_str}' for odds fixture {fixture_id}. Cannot save odds reliably.")
                return False
            odds_payload["_id"] = str(fixture_id)
            odds_payload["fixture_id"] = str(fixture_id)
            odds_payload["match_date_utc"] = datetime(parsed_match_date.year, parsed_match_date.month, parsed_match_date.day, 0, 0, 0, tzinfo=timezone.utc)
            odds_payload["match_date_str"] = date_str
            odds_payload["retrieved_at_utc"] = datetime.now(timezone.utc)
            result = self._odds_collection.update_one({"_id": str(fixture_id)}, {"$set": odds_payload}, upsert=True)
            op_type = "updated" if result.matched_count > 0 else "inserted"
            if result.upserted_id: op_type = "inserted"
            logger.info(f"Successfully {op_type} odds for fixture {fixture_id} (Match Date {date_str}). Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving odds for fixture {fixture_id} to MongoDB: {e}", exc_info=True)
            return False

    def get_odds_data(self, fixture_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves odds data for a specific fixture using _id."""
        if not self._initialized or self._odds_collection is None: return None
        try:
            return self._odds_collection.find_one({"_id": str(fixture_id)})
        except Exception as e:
            logger.error(f"Error retrieving odds for fixture {fixture_id}: {e}")
            return None

    def save_team_season_fixture_list(self, team_id: int, season: int, fixture_ids: List[int]):
        """Saves or updates the list of fixture IDs for a specific team and season."""
        if not self._initialized or self._team_fixtures_collection is None:
            logger.error("MongoDBManager not initialized or team_fixtures_collection is None. Cannot save fixture list.")
            return False

        doc_id = f"{team_id}_{season}" # Unique identifier for the team-season pair

        try:
            document_payload = {
                "_id": doc_id,
                "team_id": team_id,
                "season": season,
                "fixture_ids": fixture_ids,
                "count": len(fixture_ids),
                "last_updated_utc": datetime.now(timezone.utc)
            }

            result = self._team_fixtures_collection.update_one(
                {"_id": doc_id},
                {"$set": document_payload},
                upsert=True
            )

            op_type = "updated" if result.matched_count > 0 else "inserted"
            if result.upserted_id: op_type = "inserted"

            logger.info(f"Successfully {op_type} fixture list for Team {team_id}, Season {season}. Count: {len(fixture_ids)}")
            return True

        except OperationFailure as of:
             logger.error(f"MongoDB operation failure saving fixture list for {doc_id}: {of.details}", exc_info=True)
             return False
        except Exception as e:
            logger.error(f"Error saving/updating fixture list for {doc_id} to MongoDB: {e}", exc_info=True)
            return False

    def get_fixture_ids_for_teams_seasons(self, team_ids: List[int], seasons: List[int]) -> Set[int]:
        """
        Retrieves a set of unique fixture IDs from the 'team_season_fixtures'
        collection for the specified teams and seasons.
        """
        if not self._initialized or self._team_fixtures_collection is None:
            logger.error("DB not initialized or team_fixtures_collection missing.")
            return set()

        fixture_ids: Set[int] = set()
        try:
            # Construct query to find documents matching any of the team_ids AND any of the seasons
            query = {
                "team_id": {"$in": team_ids},
                "season": {"$in": seasons}
            }
            # Project only the fixture_ids field
            cursor = self._team_fixtures_collection.find(query, {"fixture_ids": 1, "_id": 0})

            for doc in cursor:
                ids_in_doc = doc.get("fixture_ids", [])
                if ids_in_doc: # Add IDs from the list in the document
                    fixture_ids.update(ids_in_doc)

            logger.info(f"Retrieved {len(fixture_ids)} unique fixture IDs from DB for {len(team_ids)} teams across {len(seasons)} seasons.")
            return fixture_ids
        except Exception as e:
            logger.error(f"Error retrieving fixture IDs for teams {team_ids}, seasons {seasons}: {e}", exc_info=True)
            return set() # Return empty set on error

    def get_existing_match_ids(self, fixture_ids_to_check: Optional[Set[int]] = None) -> Set[str]:
        """
        Retrieves a set of existing _id values (as strings) from the 'matches' collection.
        Optionally filters to only check IDs within the provided set.
        """
        if not self._initialized or self._matches_collection is None:
            logger.error("DB not initialized or matches_collection missing.")
            return set()

        existing_ids: Set[str] = set()
        try:
            query = {}
            # If a set of IDs is provided, optimize the query
            if fixture_ids_to_check:
                # Convert int IDs to strings for matching MongoDB _id (which is stored as string)
                string_ids_to_check = {str(fid) for fid in fixture_ids_to_check}
                query = {"_id": {"$in": list(string_ids_to_check)}}
                logger.info(f"Checking existence against {len(string_ids_to_check)} potential fixture IDs.")
            else:
                 logger.info("Checking existence against all documents in matches collection.")


            # Project only the _id field
            cursor = self._matches_collection.find(query, {"_id": 1})

            for doc in cursor:
                existing_ids.add(doc["_id"]) # _id is already a string

            logger.info(f"Found {len(existing_ids)} existing match documents based on the query.")
            return existing_ids
        except Exception as e:
            logger.error(f"Error retrieving existing match IDs: {e}", exc_info=True)
            return set()

    def check_team_season_fixture_list_exists(self, team_id: int, season: int) -> bool:
        """Checks if a fixture list document exists for the given team and season."""
        if not self._initialized or self._team_fixtures_collection is None:
            logger.error("DB not initialized or team_fixtures_collection missing. Assuming list doesn't exist.")
            return False

        doc_id = f"{team_id}_{season}"
        try:
            count = self._team_fixtures_collection.count_documents({"_id": doc_id}, limit=1)
            exists = count > 0
            if exists:
                logger.debug(f"Fixture list document {doc_id} already exists in DB.")
            else:
                logger.debug(f"Fixture list document {doc_id} does NOT exist in DB.")
            return exists
        except Exception as e:
            logger.error(f"Error checking existence for fixture list {doc_id}: {e}", exc_info=True)
            return False # Assume it doesn't exist on error to be safe

    # --- New methods for StatArea ---

    def save_statarea_data(self, statarea_doc: Dict[str, Any]) -> bool:
        """
        Saves StatArea data as a new document if it differs from the most recent version.
        Uses a time-series approach to preserve historical snapshots.
        """
        if not self._initialized or self._statarea_collection is None:
            logger.error("MongoDBManager not initialized or statarea_collection is None. Cannot save StatArea data.")
            return False

        # Extract key fields to identify the team and data type
        api_id = statarea_doc.get("api_id")
        game_type = statarea_doc.get("game_type")
        period = statarea_doc.get("period")
        team = statarea_doc.get("team", "Unknown team")

        if not all([api_id, game_type, period is not None]):
            logger.error(f"StatArea document is missing key fields (api_id, game_type, period): {team}")
            return False
            
        # Ensure scrape_date is a datetime object for proper querying
        scrape_date_str = statarea_doc.get("scrape_date")
        scrape_date_utc = None
        if scrape_date_str and isinstance(scrape_date_str, str):
            try:
                scrape_date_utc = datetime.fromisoformat(scrape_date_str).replace(tzinfo=timezone.utc)
                statarea_doc["scrape_date_utc"] = scrape_date_utc
            except ValueError:
                logger.error(f"Invalid ISO format for scrape_date '{scrape_date_str}' in StatArea data for {team}. Setting to current time.")
                scrape_date_utc = datetime.now(timezone.utc)
                statarea_doc["scrape_date_utc"] = scrape_date_utc
        elif not statarea_doc.get("scrape_date_utc"):
            logger.warning(f"Missing scrape_date for StatArea doc {team}. Setting to current time.")
            scrape_date_utc = datetime.now(timezone.utc)
            statarea_doc["scrape_date_utc"] = scrape_date_utc

        # Add a timestamp for the operation itself
        current_time = datetime.now(timezone.utc)
        statarea_doc["last_updated_db_utc"] = current_time
        
        # Create a document ID that includes the timestamp to ensure uniqueness
        timestamp_str = current_time.strftime("%Y%m%d%H%M%S")
        doc_id = f"{api_id}_{game_type}_{period}_{timestamp_str}"
        statarea_doc["_id"] = doc_id
        
        # Also create a content hash of the actual data (match history, statistics)
        content_to_hash = {
            "match_history": statarea_doc.get("match_history", []),
            "general_statistics": statarea_doc.get("general_statistics", {}),
            "team_bet_statistics": statarea_doc.get("team_bet_statistics", {})
        }
        import hashlib
        import json
        content_hash = hashlib.md5(json.dumps(content_to_hash, sort_keys=True).encode()).hexdigest()
        statarea_doc["content_hash"] = content_hash
        
        # Check if we already have this exact data (based on content hash)
        try:
            # Find the most recent document for this team/game_type/period
            query = {
                "api_id": api_id,
                "game_type": game_type,
                "period": period
            }
            most_recent_doc = self._statarea_collection.find_one(
                query,
                sort=[("scrape_date_utc", -1)]
            )
            
            # If we have a recent document with the same content hash, don't save a new one
            if most_recent_doc and most_recent_doc.get("content_hash") == content_hash:
                logger.info(f"StatArea data unchanged for {team} ({game_type}, period {period}). Skipping save.")
                return True  # Return true as this isn't an error condition
            
            # Insert the new document (it has different content or we have no previous record)
            result = self._statarea_collection.insert_one(statarea_doc)
            
            if result.inserted_id:
                if most_recent_doc:
                    logger.info(f"Saved new StatArea data snapshot for {team} ({game_type}, period {period}) - content changed")
                else:
                    logger.info(f"Saved first StatArea data snapshot for {team} ({game_type}, period {period})")
                return True
            else:
                logger.error(f"Failed to insert StatArea data for {team}")
                return False
                
        except OperationFailure as of:
            logger.error(f"MongoDB operation failure saving StatArea data for {team}: {of.details}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error saving/updating StatArea data for {team} to MongoDB: {e}", exc_info=True)
            return False

    def check_statarea_data_needs_update(self, api_id: str, game_type: str, period: int, cache_expire_days: int = 1) -> bool:
        """
        Checks if StatArea data for a specific team, game_type, and period needs updating
        based on the 'scrape_date_utc' field and cache duration.
        Returns True if data doesn't exist or is older than cache_expire_days.
        """
        if not self._initialized or self._statarea_collection is None:
            logger.error("MongoDBManager not initialized. Assuming update needed.")
            return True # Assume update needed if DB isn't ready

        try:
            # Find the most recent document for this team/game_type/period
            query = {
                "api_id": api_id,
                "game_type": game_type,
                "period": period
            }
            
            most_recent_doc = self._statarea_collection.find_one(
                query,
                sort=[("scrape_date_utc", -1)]
            )

            if not most_recent_doc:
                logger.debug(f"StatArea data for team ID {api_id} ({game_type}, period {period}) not found. Needs update.")
                return True # Data doesn't exist, needs scraping

            last_scraped_utc = most_recent_doc.get("scrape_date_utc")
            if not last_scraped_utc or not isinstance(last_scraped_utc, datetime):
                 logger.warning(f"StatArea data for team ID {api_id} found but missing valid 'scrape_date_utc'. Needs update.")
                 return True # Data exists but lacks a valid timestamp

            # Ensure last_scraped_utc is offset-aware for comparison
            if last_scraped_utc.tzinfo is None:
                last_scraped_utc = last_scraped_utc.replace(tzinfo=timezone.utc) # Assume UTC if naive

            expiry_threshold = datetime.now(timezone.utc) - timedelta(days=cache_expire_days)

            if last_scraped_utc < expiry_threshold:
                logger.debug(f"StatArea data for team ID {api_id} is stale (last scraped: {last_scraped_utc}). Needs update.")
                return True # Data is older than cache duration
            else:
                logger.debug(f"StatArea data for team ID {api_id} is recent (last scraped: {last_scraped_utc}). No update needed.")
                return False # Data is recent

        except Exception as e:
            logger.error(f"Error checking StatArea data update status for team ID {api_id}: {e}", exc_info=True)
            return True # Assume update needed on error

    def get_statarea_match_history(self, api_id: str, game_type: str, period: int = 15, date_str: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieves the 'match_history' array for a specific StatArea document.
        Typically used with period=15 as that's where history is stored.
        If date_str is provided, retrieves the historical snapshot closest to that date.
        """
        if not self._initialized or self._statarea_collection is None:
            logger.error("MongoDBManager not initialized. Cannot retrieve StatArea match history.")
            return None

        try:
            query = {
                "api_id": api_id,
                "game_type": game_type,
                "period": period
            }
            
            # If date specified, find the snapshot closest to that date
            sort_order = [("scrape_date_utc", -1)]  # Default to most recent
            if date_str:
                try:
                    target_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    # Find documents before or on the target date
                    query["scrape_date_utc"] = {"$lte": target_date}
                    # Sort to get the closest one to the target date
                    sort_order = [("scrape_date_utc", -1)]
                except ValueError:
                    logger.error(f"Invalid date format {date_str}. Using most recent.")

            document = self._statarea_collection.find_one(
                query,
                sort=sort_order,
                projection={"match_history": 1, "scrape_date_utc": 1, "_id": 0}
            )

            if document and "match_history" in document:
                logger.debug(f"Retrieved match history for team ID {api_id} from {document.get('scrape_date_utc')}.")
                return document["match_history"]
            else:
                logger.debug(f"No match history found for team ID {api_id}.")
                return None # Document or history field doesn't exist

        except Exception as e:
            logger.error(f"Error retrieving StatArea match history for team ID {api_id}: {e}", exc_info=True)
            return None

    def get_statarea_historical_snapshots(self, api_id: str, game_type: str, period: int = 15, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves multiple historical snapshots for a team's StatArea data.
        Returns a list of documents ordered from newest to oldest.
        """
        if not self._initialized or self._statarea_collection is None:
            logger.error("MongoDBManager not initialized. Cannot retrieve StatArea history.")
            return []

        try:
            query = {
                "api_id": api_id,
                "game_type": game_type,
                "period": period
            }
            
            cursor = self._statarea_collection.find(
                query,
                sort=[("scrape_date_utc", -1)],
                limit=limit
            )
            
            snapshots = list(cursor)
            logger.info(f"Retrieved {len(snapshots)} historical snapshots for team ID {api_id} ({game_type}, period {period}).")
            return snapshots
            
        except Exception as e:
            logger.error(f"Error retrieving StatArea historical snapshots for team ID {api_id}: {e}", exc_info=True)
            return []

    def verify_collection_integrity(self):
        """Verify all collections exist and none are unexpectedly empty"""
        expected_collections = ['matches', 'standings', 'odds', 'team_season_fixtures', 'statarea_stats']
        missing_collections = []
        
        for coll_name in expected_collections:
            coll_attr = f"_{coll_name}_collection"
            if not hasattr(self, coll_attr) or getattr(self, coll_attr) is None:
                missing_collections.append(coll_name)
        
        if missing_collections:
            logger.error(f"Missing collections: {missing_collections}")
            return False
        return True

    def save_daily_games(self, date_str: str, daily_payload: Dict[str, Any]) -> bool:
        """Upsert the daily games summary for the given date."""
        if not self._initialized or self._daily_games_collection is None:
            logger.error("Cannot save daily games data: DB not initialized.")
            return False
        try:
            doc = daily_payload.copy()
            doc['date'] = date_str
            doc['last_updated_utc'] = datetime.now(timezone.utc)
            self._daily_games_collection.replace_one(
                {'date': date_str},
                doc,
                upsert=True
            )
            logger.info(f"Daily games for {date_str} saved to 'daily_games'.")
            return True
        except Exception as e:
            logger.error(f"Error saving daily games for {date_str}: {e}", exc_info=True)
            return False

    def get_daily_games(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Retrieve the daily games summary for the given date."""
        if not self._initialized or self._daily_games_collection is None:
            logger.error("Cannot load daily games data: DB not initialized.")
            return None
        try:
            return self._daily_games_collection.find_one({'date': date_str})
        except Exception as e:
            logger.error(f"Error loading daily games for {date_str}: {e}", exc_info=True)
            return None

    # <<< New method to save MatchProcessor data >>>
    def save_match_processor_data(self, processor_payload: Dict[str, Any]) -> bool:
        """Saves the fetched predictions, stats, and standings snapshot to the 'match_processor' collection."""
        if not self._initialized or self._match_processor_collection is None:
            logger.error("MongoDBManager not initialized or match_processor_collection is None. Cannot save processor data.")
            return False

        fixture_id = processor_payload.get("fixture_id")
        if not fixture_id:
            logger.error("Match processor payload missing 'fixture_id'. Cannot save.")
            return False

        try:
            # Use fixture_id as the document _id
            processor_payload["_id"] = str(fixture_id)
            # Ensure a timestamp exists
            if "processed_at_utc" not in processor_payload:
                processor_payload["processed_at_utc"] = datetime.now(timezone.utc)

            result = self._match_processor_collection.update_one(
                {"_id": str(fixture_id)},
                {"$set": processor_payload},
                upsert=True
            )
            op_type = "updated" if result.matched_count > 0 else "inserted"
            if result.upserted_id: op_type = "inserted"
            logger.info(f"Successfully {op_type} match processor data for fixture {fixture_id}. Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")
            return True
        except OperationFailure as of:
             logger.error(f"MongoDB operation failure saving processor data for fixture {fixture_id}: {of.details}", exc_info=True)
             return False
        except Exception as e:
            logger.error(f"Error saving match processor data for fixture {fixture_id}: {e}", exc_info=True)
            return False

    def check_match_processor_data_exists(self, fixture_id: str) -> bool:
        """Checks if data for a fixture ID exists in the 'match_processor' collection."""
        if not self._initialized or self._match_processor_collection is None:
            logger.error("Cannot check match processor data: DB not initialized.")
            return False
        try:
            count = self._match_processor_collection.count_documents({"_id": str(fixture_id)})
            return count > 0
        except Exception as e:
            logger.error(f"Error checking match processor data existence for fixture {fixture_id}: {e}")
            return False


# Singleton instance (initialization happens on first call)
# Ensure initialization uses the desired DB name if not default
db_manager = MongoDBManager(db_name="agenticfc") # Explicitly set db_name

# --- Basic Test Function Update ---
async def run_db_test():
    """Tests writing and reading from the MongoDB structure, including StatArea."""
    logger.info("--- Running MongoDB Test ---")
    try:
        # Ensure manager is initialized (it should be by accessing the singleton)
        if not db_manager._initialized:
             logger.error("Test failed: MongoDBManager singleton could not be initialized.")
             # Try explicit init? Depends on desired behavior. Let's assume singleton access initializes.
             # db_manager.__init__(db_name="agenticfc") # Or try explicit init
             # if not db_manager._initialized: return # Exit if still not initialized
             return


        test_date = datetime.now().strftime("%Y-%m-%d")
        test_fixture_id = "test_api_fixture_123"
        test_league_id = "test_league_999"
        test_season = datetime.now().year
        test_statarea_api_id = "statarea_test_team_1"
        test_statarea_team = "Statarea Test United"
        test_statarea_country = "Testland"

        # --- Test StatArea ---
        logger.info(f"Testing save_statarea_data for {test_statarea_api_id}...")
        statarea_doc_host_10 = {
            "api_id": test_statarea_api_id,
            "team": test_statarea_team,
            "country": test_statarea_country,
            "game_type": "host",
            "period": 10,
            "scrape_date": datetime.now(timezone.utc).isoformat(),
            "general_statistics": {"Goals Scored": "1.5"},
            "team_bet_statistics": {"Over/Under 2.5": {"Over": "60%"}}
        }
        statarea_saved_host_10 = db_manager.save_statarea_data(statarea_doc_host_10)
        if statarea_saved_host_10: logger.info(" StatArea save successful (host, 10).")
        else: logger.error(" StatArea save failed (host, 10).")

        statarea_doc_guest_15 = {
            "api_id": test_statarea_api_id,
            "team": test_statarea_team,
            "country": test_statarea_country,
            "game_type": "guest",
            "period": 15,
            "scrape_date": datetime.now(timezone.utc).isoformat(),
            "general_statistics": {"Goals Conceded": "1.2"},
            "team_bet_statistics": {"Both Teams To Score": {"Yes": "55%"}},
            "match_history": [{"date": "2023-10-01", "opponent": "Rival FC", "result": "win"}, {"date": "2023-09-25", "opponent": "Local XI", "result": "draw"}]
        }
        statarea_saved_guest_15 = db_manager.save_statarea_data(statarea_doc_guest_15)
        if statarea_saved_guest_15: logger.info(" StatArea save successful (guest, 15 with history).")
        else: logger.error(" StatArea save failed (guest, 15).")

        logger.info(f"Testing check_statarea_data_needs_update...")
        needs_update_recent = db_manager.check_statarea_data_needs_update(test_statarea_api_id, "host", 10, cache_expire_days=1)
        if not needs_update_recent: logger.info(" Check needs update successful (recent data -> False).")
        else: logger.error(" Check needs update failed (recent data -> True).")

        needs_update_old_cache = db_manager.check_statarea_data_needs_update(test_statarea_api_id, "guest", 15, cache_expire_days=0) # Force expiry
        if needs_update_old_cache: logger.info(" Check needs update successful (expired cache -> True).")
        else: logger.error(" Check needs update failed (expired cache -> False).")

        needs_update_missing = db_manager.check_statarea_data_needs_update("nonexistent_id", "host", 10, cache_expire_days=1)
        if needs_update_missing: logger.info(" Check needs update successful (missing data -> True).")
        else: logger.error(" Check needs update failed (missing data -> False).")

        logger.info(f"Testing get_statarea_match_history...")
        retrieved_history = db_manager.get_statarea_match_history(test_statarea_api_id, "guest", 15)
        if retrieved_history and len(retrieved_history) == 2:
            logger.info(f" StatArea history retrieve successful: Found {len(retrieved_history)} matches.")
        elif retrieved_history is not None:
             logger.error(f" StatArea history retrieve failed (wrong count: {len(retrieved_history)}).")
        else:
            logger.error(" StatArea history retrieve failed (returned None).")

        retrieved_history_wrong_period = db_manager.get_statarea_match_history(test_statarea_api_id, "host", 10)
        if retrieved_history_wrong_period is None:
            logger.info(" StatArea history retrieve successful (no history expected for period 10 -> None).")
        else:
            logger.error(" StatArea history retrieve failed (expected None for period 10, got data).")


        # --- Test Other Collections (Keep brief) ---
        logger.info(f"Testing save_match_data for fixture {test_fixture_id}...")
        # Use a realistic structure matching game_details if possible
        match_data_payload = {
            "fixture_id": int(test_fixture_id.split('_')[-1]), # Assuming integer ID if possible
            "teams": {"home": {"id": 1, "name": "Test Home"}, "away": {"id": 2, "name": "Test Away"}},
            "goals": {"home": 1, "away": 0},
            "league": {"id": int(test_league_id.split('_')[-1]), "season": test_season},
            "fixture": {"date": f"{test_date}T12:00:00+00:00"}, # ISO string for date
            "_id": test_fixture_id, # Keep string _id for consistency
            "date_str": test_date, # Maintain for potential use elsewhere
            # Add date_utc and fetch_timestamp_utc which save_match_data expects or creates
        }
        match_saved = db_manager.save_match_data(match_data_payload)
        if match_saved: logger.info(" Match save successful.")
        else: logger.error(" Match save failed.")

        # ... (Keep other tests brief or remove if redundant for this task) ...

        logger.info("Cleaning up test data...")
        if db_manager._matches_collection:
             # Use _id which is string
            delete_match_result = db_manager._matches_collection.delete_one({"_id": test_fixture_id})
            logger.debug(f"Deleted {delete_match_result.deleted_count} match test documents.")
        # ... (Keep other cleanup) ...
        if db_manager._statarea_collection:
             # Use _id which is composite string
            delete_statarea_result = db_manager._statarea_collection.delete_many({"api_id": test_statarea_api_id})
            logger.debug(f"Deleted {delete_statarea_result.deleted_count} StatArea test documents.")

        logger.info("Test data cleanup finished.")

    except Exception as e:
        logger.error(f"An error occurred during the DB test: {e}", exc_info=True)
    finally:
        logger.info("--- MongoDB Test Finished ---")
        # Consider closing the connection if this test runs standalone
        # db_manager.close_connection()

# Keep the main block for running tests if needed
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    import asyncio
    # Make sure the event loop is running for async operations if any dependencies require it
    # Although run_db_test itself is async, the calls inside might not be if db_manager methods are synchronous
    # asyncio.run(run_db_test()) should work fine here.
    asyncio.run(run_db_test())