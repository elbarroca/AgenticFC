import os
import logging
import time
from pymongo import MongoClient, UpdateOne, ReturnDocument
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure, BulkWriteError
from typing import Optional, Dict, Any, List, Set, Tuple
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class MongoDBManager:
    _instance = None
    _client: Optional[MongoClient] = None
    _db: Optional[Any] = None
    _matches_collection: Optional[Any] = None
    _standings_collection: Optional[Any] = None
    _odds_collection: Optional[Any] = None
    _team_fixtures_collection: Optional[Any] = None
    _statarea_collection: Optional[Any] = None
    _daily_games_collection: Optional[Any] = None
    _ml_ready_collection: Optional[Any] = None
    _match_processor_collection: Optional[Any] = None
    _max_retries: int = 3
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MongoDBManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_name: str = "agenticfc"):
        if self._initialized:
            assert self._db is not None, "DB should be initialized if _initialized is True"
            if self._db.name == db_name:
                 return
            else:
                logger.warning(f"Re-initializing MongoDBManager (existing db: {self._db.name}, requested: {db_name}). Closing previous connection.")
                self._reset_state()

        self._initialized = False

        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        dotenv_path = project_root / '.env'
        load_dotenv(dotenv_path=dotenv_path)

        mongo_uri = os.getenv("MONGO_URI")
        assert mongo_uri, "MONGO_URI environment variable is required and must be set in .env"

        retry_count = 0
        connected = False

        while not connected and retry_count < self._max_retries:
            try:
                retry_count += 1
                logger.info(f"Attempting to connect to MongoDB (attempt {retry_count}/{self._max_retries})...")
                self._client = MongoClient(
                    mongo_uri,
                    serverSelectionTimeoutMS=15000,
                    connectTimeoutMS=15000,
                    socketTimeoutMS=60000,
                    maxPoolSize=50,
                    appname="AgenticFC-ML",
                    tls=False,
                )
                self._client.admin.command('ping')
                logger.info(f"Successfully connected to MongoDB server")

                self._db = self._client[db_name]
                assert self._db is not None, "Database object not obtained after connection"
                logger.info(f"Using database: {db_name}")

                self._matches_collection = self._db['matches']
                self._standings_collection = self._db['standings']
                self._odds_collection = self._db['odds']
                self._team_fixtures_collection = self._db['team_season_fixtures']
                self._statarea_collection = self._db['statarea_stats']
                self._daily_games_collection = self._db['daily_games']
                self._ml_ready_collection = self._db['ml_ready']
                self._match_processor_collection = self._db['match_processor']

                assert self._matches_collection is not None
                assert self._standings_collection is not None
                assert self._odds_collection is not None
                assert self._team_fixtures_collection is not None
                assert self._statarea_collection is not None
                assert self._daily_games_collection is not None
                assert self._ml_ready_collection is not None
                assert self._match_processor_collection is not None

                logger.info("Initialized collections: matches, standings, odds, team_season_fixtures, statarea_stats, daily_games, ml_ready, match_processor")

                self._initialized = True
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
                    raise ConnectionFailure(f"MongoDB connection failed after {self._max_retries} attempts: {e}")

    def _create_indexes(self):
        """Creates or ensures necessary indexes using consistent field paths."""
        assert self._initialized, "DB Manager not initialized"
        logger.info("Attempting to create/ensure MongoDB indexes with corrected paths...")

        def _create_index_safely(collection, keys, name, **kwargs):
            """Helper to create index and log warnings on failure."""
            if collection is None:
                logger.warning(f"Collection object is None. Cannot create index '{name}'.")
                return
            try:
                collection.create_index(keys, name=name, **kwargs)
                # logger.debug(f"Index '{name}' ensured for collection '{collection.name}'.")
            except OperationFailure as e:
                 # Check for specific conflict codes
                 if e.code in [85, 86]: # IndexOptionsConflict, IndexKeySpecsConflict
                     logger.warning(f"Index conflict for '{name}' on '{collection.name}': {e.details}. Check definition matches DB.")
                 elif e.code == 11000: # DuplicateKey
                     logger.warning(f"Duplicate key error for index '{name}' on '{collection.name}': {e.details}. Data violates uniqueness or index should not be unique.")
                 else:
                     logger.error(f"OperationFailure creating index '{name}' on '{collection.name}': {e.details}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error creating index '{name}' on '{collection.name}': {e}", exc_info=True)

        # --- Matches Collection (Using fixture_details paths consistently) ---
        # Assuming _id is string version of fixture_id
        _create_index_safely(self._matches_collection, [("_id", 1)], name="match_fixture_id_str_idx") # Removed unique=True
        _create_index_safely(self._matches_collection, [("fixture_details.fixture.date", 1)], name="match_date_idx")
        _create_index_safely(self._matches_collection, [("fixture_details.teams.home.id", 1), ("fixture_details.fixture.date", -1)], name="match_home_team_date_idx")
        _create_index_safely(self._matches_collection, [("fixture_details.teams.away.id", 1), ("fixture_details.fixture.date", -1)], name="match_away_team_date_idx")
        _create_index_safely(self._matches_collection, [("fixture_details.league.id", 1)], name="match_league_id_idx") # Index on int league ID


        # --- Odds Collection ---
        # Assuming _id is string version of fixture_id
        _create_index_safely(self._odds_collection, [("_id", 1)], name="odds_fixture_id_str_idx") # Removed unique=True
        _create_index_safely(self._odds_collection, [("match_date_str", 1)], name="odds_match_date_str_idx")
        _create_index_safely(self._odds_collection, [("last_updated_utc", -1)], name="odds_last_updated_idx") # Index update time


        # --- Daily Games Collection ---
        # Assuming _id is the date string "YYYY-MM-DD"
        _create_index_safely(self._daily_games_collection, [("_id", 1)], name="daily_games_date_idx") # Removed unique=True

        # --- ML Ready Collection ---
        # Assuming _id is MatchID (preferred) or string fixture_id
        _create_index_safely(self._ml_ready_collection, [("_id", 1)], name="ml_doc_id_idx") # Removed unique=True # Primary lookup
        _create_index_safely(self._ml_ready_collection, [("fixture_id", 1)], name="ml_fixture_id_idx", sparse=True) # If fixture_id is stored separately
        _create_index_safely(self._ml_ready_collection, [("Date", 1)], name="ml_date_idx")
        _create_index_safely(self._ml_ready_collection, [("LeagueID", 1)], name="ml_league_id_idx")
        _create_index_safely(self._ml_ready_collection, [("processing_timestamp_utc", -1)], name="ml_proc_timestamp_idx")


        # --- Standings Collection ---
        # Index for finding latest standings for a league/season before a date
        _create_index_safely(self._standings_collection, [("league_id", 1), ("season", 1), ("date_retrieved_str", -1)], name="standings_league_season_date_idx")


        # --- StatArea Collection ---
        # Assuming _id is "{api_id}_{game_type}_{period}"
        _create_index_safely(self._statarea_collection, [("_id", 1)], name="statarea_doc_id_idx") # Should be unique now
        _create_index_safely(self._statarea_collection, [("api_id", 1)], name="statarea_api_id_idx")
        _create_index_safely(self._statarea_collection, [("api_id", 1), ("game_type", 1), ("period", 1), ("scrape_date_utc", -1)], name="statarea_timeseries_lookup_idx") # For historical lookups
        _create_index_safely(self._statarea_collection, [("scrape_date_utc", -1)], name="statarea_scrape_date_idx")


        # --- Match Processor Collection (If still used) ---
        # Assuming _id is string version of fixture_id
        if hasattr(self, '_match_processor_collection') and self._match_processor_collection is not None:
             _create_index_safely(self._match_processor_collection, [("_id", 1)], name="proc_fixture_id_str_idx") # Removed unique=True
             _create_index_safely(self._match_processor_collection, [("match_date_str", 1)], name="proc_match_date_str_idx")
             _create_index_safely(self._match_processor_collection, [("league_id", 1)], name="proc_league_id_idx")
             _create_index_safely(self._match_processor_collection, [("last_updated_utc", -1)], name="proc_last_updated_idx") # Use consistent naming

        logger.info("Finished attempting to create/ensure indexes.")

    def _reset_state(self):
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
        self._statarea_collection = None
        self._daily_games_collection = None
        self._ml_ready_collection = None
        self._match_processor_collection = None
        self._initialized = False
        MongoDBManager._instance = None
        logger.debug("MongoDBManager state reset complete.")

    def close_connection(self):
        logger.info("Closing MongoDB connection.")
        self._reset_state()

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        assert isinstance(date_str, str) and date_str, "date_str must be a non-empty string"

        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.astimezone(timezone.utc)
        except ValueError:
            try:
                dt_naive = datetime.strptime(date_str, "%Y-%m-%d")
                return dt_naive.replace(tzinfo=timezone.utc)
            except ValueError:
                logger.error(f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD or ISO 8601 format.")
                return None

    def get_match_data(self, fixture_id: str) -> Optional[Dict[str, Any]]:
        assert self._initialized and self._matches_collection is not None, "DB not initialized or matches collection missing"
        assert isinstance(fixture_id, str) and fixture_id, "Fixture ID must be a non-empty string"

        return self._matches_collection.find_one({"_id": fixture_id})

    def get_historical_matches(self, team_id: int, before_date: datetime, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Retrieves historical matches for a team played strictly *before* a given date.
        Uses the `matches` collection structure.
        """
        assert self._initialized and self._matches_collection is not None, "DB not initialized or matches collection missing"
        assert isinstance(team_id, int), "Team ID must be an integer"
        assert isinstance(before_date, datetime), "before_date must be a datetime object"
        assert limit > 0, "Limit must be positive"

        # Ensure before_date is timezone-aware UTC for comparison
        if before_date.tzinfo is None or before_date.tzinfo.utcoffset(before_date) != timezone.utc.utcoffset(before_date):
            before_date_utc = before_date.astimezone(timezone.utc)
            # logger.warning(f"Provided 'before_date' was naive or not UTC. Converted to UTC: {before_date_utc.isoformat()}")
        else:
             before_date_utc = before_date

        try:
            # --- !!! ADJUST FIELD PATHS AND TYPES HERE !!! ---
            query = {
                "$and": [
                    # Compare against the correct date field (ISODate object preferred)
                    {"fixture_details.fixture.date": {"$lt": before_date_utc.isoformat()}}, # Use ISO string if date is stored as string
                    # {"fixture_details.fixture.date": {"$lt": before_date_utc}}, # Use datetime object if date is ISODate

                    # Compare against the correct team ID fields and ensure TYPE matches (int vs string)
                    {"$or": [
                        {"fixture_details.teams.home.id": team_id}, # Assumes ID is stored as INT
                        {"fixture_details.teams.away.id": team_id}  # Assumes ID is stored as INT
                        # {"fixture_details.teams.home.id": str(team_id)}, # Use if ID is stored as STRING
                        # {"fixture_details.teams.away.id": str(team_id)}  # Use if ID is stored as STRING
                    ]}
                ]
            }
            # Sort by date descending to get the most recent matches first
            sort_order = [("fixture_details.fixture.date", -1)] # Use the correct date field path
            # --- End Adjustments ---

            cursor = self._matches_collection.find(query).sort(sort_order).limit(limit)
            matches = list(cursor)
            if not matches:
                 logger.warning(f"DB Query found 0 historical matches for team {team_id} before {before_date_utc.isoformat()}. Check query/data.")
            else:
                 logger.info(f"DB Query found {len(matches)} historical matches for team {team_id} before {before_date_utc.isoformat()}")
            return matches
        except Exception as e:
            logger.error(f"Error retrieving historical matches for team {team_id} before {before_date_utc.isoformat()}: {e}", exc_info=True)
            return []

    def save_ml_ready_data(self, ml_data: Dict[str, Any]) -> bool:
        assert self._initialized and self._ml_ready_collection is not None, "DB not initialized or ml_ready collection missing"
        assert isinstance(ml_data, dict), "ml_data must be a dictionary"

        match_id = ml_data.get("MatchID")
        fixture_id = ml_data.get("fixture_id")
        assert match_id or fixture_id, "ML data must contain 'MatchID' or 'fixture_id' for identification"

        doc_id = match_id if match_id else str(fixture_id)
        ml_data_to_save = ml_data.copy()
        ml_data_to_save["_id"] = doc_id
        ml_data_to_save["processing_timestamp_utc"] = datetime.now(timezone.utc)

        try:
            result = self._ml_ready_collection.update_one(
                {"_id": doc_id},
                {"$set": ml_data_to_save},
                upsert=True
            )
            op_type = "updated" if result.matched_count > 0 else "inserted"
            if result.upserted_id: op_type = "inserted"
            logger.info(f"Successfully {op_type} ML ready data for ID {doc_id}. Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")
            return True
        except OperationFailure as op_fail:
            logger.error(f"MongoDB operation failure saving ML ready data for {doc_id}: {op_fail.details}", exc_info=True)
            return False

    def save_ml_ready_data_bulk(self, ml_data_list: List[Dict[str, Any]]) -> bool:
        assert self._initialized and self._ml_ready_collection is not None, "DB not initialized or ml_ready collection missing"
        assert isinstance(ml_data_list, list), "ml_data_list must be a list of dictionaries"

        if not ml_data_list:
            logger.info("No ML ready data provided for bulk save.")
            return True

        operations = []
        current_time = datetime.now(timezone.utc)
        processed_ids: Set[str] = set()
        skipped_duplicates = 0

        for ml_data in ml_data_list:
            assert isinstance(ml_data, dict), "Each item in ml_data_list must be a dictionary"
            match_id = ml_data.get("MatchID")
            fixture_id = ml_data.get("fixture_id")
            assert match_id or fixture_id, "ML data must contain 'MatchID' or 'fixture_id'"

            doc_id = match_id if match_id else str(fixture_id)

            if doc_id in processed_ids:
                logger.warning(f"Duplicate ID '{doc_id}' found within the bulk ML data list. Skipping this entry.")
                skipped_duplicates += 1
                continue
            processed_ids.add(doc_id)

            ml_data_to_save = ml_data.copy()
            ml_data_to_save["_id"] = doc_id
            ml_data_to_save["processing_timestamp_utc"] = current_time

            operations.append(
                UpdateOne({"_id": doc_id}, {"$set": ml_data_to_save}, upsert=True)
            )

        if not operations:
            logger.info(f"No valid operations generated for bulk ML data save (initial list size: {len(ml_data_list)}, duplicates skipped: {skipped_duplicates}).")
            return True

        logger.info(f"Executing bulk write for {len(operations)} ML ready documents...")
        try:
            result = self._ml_ready_collection.bulk_write(operations, ordered=False)
            logger.info(
                f"Bulk ML ready data write complete. "
                f"Inserted: {result.upserted_count}, Updated: {result.modified_count}, "
                f"Matched: {result.matched_count}. "
                f"(Duplicates skipped in input list: {skipped_duplicates})"
            )
            return True
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error saving ML ready data: {bwe.details}", exc_info=True)
            return False

    def check_ml_ready_data_exists(self, identifier: str) -> bool:
        assert self._initialized and self._ml_ready_collection is not None, "DB not initialized or ml_ready collection missing"
        assert isinstance(identifier, str) and identifier, "Identifier must be a non-empty string"

        count = self._ml_ready_collection.count_documents({"_id": identifier}, limit=1)
        return count > 0

    def save_daily_games(self, date_str: str, daily_payload: Dict[str, Any]) -> bool:
        assert self._initialized and self._daily_games_collection is not None, "DB not initialized or daily_games collection missing"
        assert isinstance(date_str, str) and len(date_str) == 10, "Date string must be in YYYY-MM-DD format"
        assert isinstance(daily_payload, dict), "Payload must be a dictionary"

        doc_to_save = daily_payload.copy()
        doc_to_save['date'] = date_str
        doc_to_save['_id'] = date_str
        doc_to_save['last_updated_utc'] = datetime.now(timezone.utc)

        try:
            result = self._daily_games_collection.replace_one(
                {'_id': date_str},
                doc_to_save,
                upsert=True
            )
            op_type = "replaced" if result.matched_count > 0 else "inserted"
            if result.upserted_id: op_type = "inserted"
            logger.info(f"Daily games for {date_str} {op_type} in 'daily_games'. Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")
            return True
        except OperationFailure as op_fail:
            logger.error(f"MongoDB operation failure saving daily games for {date_str}: {op_fail.details}", exc_info=True)
            return False

    def get_daily_games(self, date_str: str) -> Optional[Dict[str, Any]]:
        assert self._initialized and self._daily_games_collection is not None, "DB not initialized or daily_games collection missing"
        assert isinstance(date_str, str) and len(date_str) == 10, "Date string must be in YYYY-MM-DD format"

        return self._daily_games_collection.find_one({'_id': date_str})

    def save_match_data(self, match_data: Dict[str, Any]) -> bool:
        assert self._initialized and self._matches_collection is not None, "DB not initialized or matches collection missing"
        assert isinstance(match_data, dict), "match_data must be a dictionary"

        fixture_id = match_data.get("fixture_id") or match_data.get("_id")
        if not fixture_id and 'basic_info' in match_data:
            try:
                fixture_id = match_data['basic_info'][0]['fixture']['id']
                restructured_data = {
                    "_id": str(fixture_id),
                    "fixture_id": str(fixture_id),
                    "fixture_details_raw": match_data,
                    "last_updated_utc": datetime.now(timezone.utc)
                }
                match_data_to_save = restructured_data
                logger.debug(f"Restructured data from FixtureDetailsFetcher for saving fixture {fixture_id}")
            except (IndexError, KeyError, TypeError) as e:
                logger.error(f"Could not extract fixture_id from FixtureDetailsFetcher structure: {e}")
                assert False, "Match data structure from FixtureDetailsFetcher unrecognized or missing fixture ID."
        elif fixture_id:
            fixture_id = str(fixture_id)
            match_data_to_save = match_data.copy()
            match_data_to_save["_id"] = fixture_id
            match_data_to_save["last_updated_utc"] = datetime.now(timezone.utc)
        else:
            assert False, "Match data must contain 'fixture_id', '_id', or be structured like FixtureDetailsFetcher output."

        try:
            result = self._matches_collection.update_one(
                {"_id": fixture_id},
                {"$set": match_data_to_save},
                upsert=True
            )
            op_type = "updated" if result.matched_count > 0 else "inserted"
            if result.upserted_id: op_type = "inserted"
            log_id_info = f"fixture ID {fixture_id}"
            if 'fixture_details_raw' in match_data_to_save:
                log_id_info += " (raw details structure)"

            logger.info(f"Successfully {op_type} match data for {log_id_info}. Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")
            return True
        except OperationFailure as op_fail:
            logger.error(f"MongoDB operation failure saving match data for {fixture_id}: {op_fail.details}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving match data for {fixture_id}: {e}", exc_info=True)
            return False

    def check_match_exists(self, fixture_id: str) -> bool:
        assert self._initialized and self._matches_collection is not None, "DB not initialized or matches collection missing"
        assert isinstance(fixture_id, str) and fixture_id, "Fixture ID must be a non-empty string"

        count = self._matches_collection.count_documents({"_id": fixture_id}, limit=1)
        return count > 0

    def save_odds_data(self, date_str: str, fixture_id: str, odds_payload: Dict[str, Any]) -> bool:
        assert self._initialized and self._odds_collection is not None, "DB not initialized or odds collection missing"
        assert isinstance(date_str, str) and len(date_str) == 10, "Date string must be in YYYY-MM-DD format"
        assert isinstance(fixture_id, str) and fixture_id, "Fixture ID must be a non-empty string"
        assert isinstance(odds_payload, dict), "odds_payload must be a dictionary"
        assert "bookmakers" in odds_payload, "'bookmakers' key is required in odds_payload"

        doc_id = fixture_id
        odds_payload_to_save = odds_payload.copy()
        odds_payload_to_save['_id'] = doc_id
        odds_payload_to_save['fixture_id'] = fixture_id
        odds_payload_to_save['match_date_str'] = date_str
        odds_payload_to_save['last_updated_utc'] = datetime.now(timezone.utc)

        try:
            result = self._odds_collection.update_one(
                {'_id': doc_id},
                {'$set': odds_payload_to_save},
                upsert=True
            )
            op_type = "updated" if result.matched_count > 0 else "inserted"
            if result.upserted_id: op_type = "inserted"
            logger.info(f"Successfully {op_type} odds data for fixture ID {fixture_id} (Date: {date_str}). Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")
            return True
        except OperationFailure as op_fail:
            logger.error(f"MongoDB operation failure saving odds for fixture {fixture_id}: {op_fail.details}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving odds for fixture {fixture_id}: {e}", exc_info=True)
            return False

    def get_odds_data(self, fixture_id: str) -> Optional[Dict[str, Any]]:
        assert self._initialized and self._odds_collection is not None, "DB not initialized or odds collection missing"
        assert isinstance(fixture_id, str) and fixture_id, "Fixture ID must be a non-empty string"

        return self._odds_collection.find_one({'_id': fixture_id})

    def save_standings_data(self, date_str: str, league_id: str, season: int, standings_payload: Dict[str, Any]) -> bool:
        assert self._initialized and self._standings_collection is not None, "DB not initialized or standings collection missing"
        assert isinstance(date_str, str) and len(date_str) == 10, "Date string must be in YYYY-MM-DD format"
        assert isinstance(league_id, str) and league_id, "League ID must be a non-empty string"
        assert isinstance(season, int) and season > 1900, "Season must be a valid year integer"
        assert isinstance(standings_payload, dict), "standings_payload must be a dictionary"

        standings_to_save = standings_payload.copy()
        standings_to_save['league_id'] = league_id
        standings_to_save['season'] = season
        standings_to_save['date_retrieved_str'] = date_str
        standings_to_save['saved_at_utc'] = datetime.now(timezone.utc)

        try:
            result = self._standings_collection.insert_one(standings_to_save)
            logger.info(f"Successfully inserted standings data for League {league_id}, Season {season}, Date {date_str}. Inserted ID: {result.inserted_id}")
            return True
        except OperationFailure as op_fail:
            logger.error(f"MongoDB operation failure saving standings for League {league_id}, Season {season}, Date {date_str}: {op_fail.details}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving standings for League {league_id}, Season {season}, Date {date_str}: {e}", exc_info=True)
            return False

    def get_latest_standings(self, league_id: str, season: int, before_date_str: Optional[str] = None) -> Optional[Dict[str, Any]]:
        assert self._initialized and self._standings_collection is not None, "DB not initialized or standings collection missing"
        assert isinstance(league_id, str) and league_id, "League ID must be a non-empty string"
        assert isinstance(season, int) and season > 1900, "Season must be a valid year integer"

        query = {"league_id": league_id, "season": season}
        if before_date_str:
            assert isinstance(before_date_str, str) and len(before_date_str) == 10, "before_date_str must be in YYYY-MM-DD format"
            query["date_retrieved_str"] = {"$lte": before_date_str}

        sort_order = [("date_retrieved_str", -1), ("saved_at_utc", -1)]

        return self._standings_collection.find_one(query, sort=sort_order)

    def save_statarea_data(self, stats_data: Dict[str, Any]) -> bool:
        assert self._initialized and self._statarea_collection is not None, "DB not initialized or statarea collection missing"
        assert isinstance(stats_data, dict), "stats_data must be a dictionary"
        api_id = stats_data.get("api_id")
        game_type = stats_data.get("game_type")
        period = stats_data.get("period")
        assert api_id and game_type and period is not None, "StatArea data must contain 'api_id', 'game_type', and 'period'"

        doc_id = f"{api_id}_{game_type}_{period}"
        stats_data_to_save = stats_data.copy()
        stats_data_to_save["_id"] = doc_id
        if "scrape_date_utc" not in stats_data_to_save or not isinstance(stats_data_to_save["scrape_date_utc"], datetime):
             stats_data_to_save["scrape_date_utc"] = datetime.now(timezone.utc)

        try:
            result = self._statarea_collection.update_one(
                {"_id": doc_id},
                {"$set": stats_data_to_save},
                upsert=True
            )
            op_type = "updated" if result.matched_count > 0 else "inserted"
            if result.upserted_id: op_type = "inserted"
            logger.info(f"Successfully {op_type} StatArea data for ID {doc_id}. Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")
            return True
        except OperationFailure as op_fail:
            logger.error(f"MongoDB operation failure saving StatArea data for {doc_id}: {op_fail.details}", exc_info=True)
            return False
        except Exception as e:
             logger.error(f"Unexpected error saving StatArea data for {doc_id}: {e}", exc_info=True)
             return False

    def check_statarea_data_needs_update(self, api_id: str, game_type: str, period: int, cache_expire_days: int = 1) -> bool:
        assert self._initialized and self._statarea_collection is not None, "DB not initialized or statarea collection missing"
        assert api_id and game_type and period is not None, "api_id, game_type, and period are required"
        assert isinstance(cache_expire_days, int) and cache_expire_days >= 0, "cache_expire_days must be a non-negative integer"

        doc_id = f"{api_id}_{game_type}_{period}"
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=cache_expire_days)

        document = self._statarea_collection.find_one(
            {"_id": doc_id},
            {"scrape_date_utc": 1}
        )
        if document:
            last_scrape_time = document.get("scrape_date_utc")
            if isinstance(last_scrape_time, datetime):
                if last_scrape_time.tzinfo is None:
                    last_scrape_time = last_scrape_time.replace(tzinfo=timezone.utc)
                if last_scrape_time >= cutoff_time:
                    return False
        return True

    def save_match_processor_data(self, processor_data: Dict[str, Any]) -> bool:
        assert self._initialized and self._match_processor_collection is not None, "DB not initialized or match_processor collection missing"
        assert isinstance(processor_data, dict), "processor_data must be a dictionary"
        fixture_id = processor_data.get("fixture_id")
        assert fixture_id, "Processor data must contain 'fixture_id'"
        fixture_id = str(fixture_id)

        processor_data_to_save = processor_data.copy()
        processor_data_to_save["_id"] = fixture_id
        processor_data_to_save["last_updated_utc"] = datetime.now(timezone.utc)

        try:
            result = self._match_processor_collection.update_one(
                {"_id": fixture_id},
                {"$set": processor_data_to_save},
                upsert=True
            )
            op_type = "updated" if result.matched_count > 0 else "inserted"
            if result.upserted_id: op_type = "inserted"
            logger.info(f"Successfully {op_type} match processor data for fixture ID {fixture_id}. Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")
            return True
        except OperationFailure as op_fail:
            logger.error(f"MongoDB operation failure saving match processor data for {fixture_id}: {op_fail.details}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving match processor data for {fixture_id}: {e}", exc_info=True)
            return False

    def check_match_processor_data_exists(self, fixture_id: str) -> bool:
        assert self._initialized and self._match_processor_collection is not None, "DB not initialized or match_processor collection missing"
        assert isinstance(fixture_id, str) and fixture_id, "Fixture ID must be a non-empty string"

        count = self._match_processor_collection.count_documents({"_id": fixture_id}, limit=1)
        return count > 0

    def get_fixture_ids_from_daily_games_range(self, start_date: datetime, end_date: datetime) -> List[int]:
        assert self._initialized and self._daily_games_collection is not None, "DB not initialized or daily_games collection missing"
        assert isinstance(start_date, datetime) and isinstance(end_date, datetime), "Start and end dates must be datetime objects"
        assert start_date <= end_date, "Start date must be before or equal to end date"

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        logger.info(f"Querying daily_games for fixture IDs between {start_date_str} and {end_date_str}")

        query = {"date": {"$gte": start_date_str, "$lte": end_date_str}}
        projection = {"leagues": 1, "_id": 0}

        cursor = self._daily_games_collection.find(query, projection)
        all_fixture_ids: Set[int] = set()

        for daily_doc in cursor:
            leagues_dict = daily_doc.get("leagues", {})
            if not isinstance(leagues_dict, dict): continue
            for league_data in leagues_dict.values():
                if not isinstance(league_data, dict): continue
                matches_list = league_data.get("matches", [])
                if not isinstance(matches_list, list): continue
                for match in matches_list:
                    if not isinstance(match, dict): continue
                    fixture_id = match.get("id")
                    if fixture_id is not None:
                        try:
                            all_fixture_ids.add(int(fixture_id))
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert fixture ID '{fixture_id}' to int in daily_games doc.")

        logger.info(f"Found {len(all_fixture_ids)} unique fixture IDs in daily_games between {start_date_str} and {end_date_str}.")
        return sorted(list(all_fixture_ids))

    def find_missing_fixture_ids_in_matches(self, fixture_ids_to_check: List[int]) -> List[int]:
        assert self._initialized and self._matches_collection is not None, "DB not initialized or matches collection missing"
        assert isinstance(fixture_ids_to_check, list), "Input must be a list of fixture IDs"

        if not fixture_ids_to_check:
            return []

        fixture_ids_str = [str(fid) for fid in fixture_ids_to_check]
        logger.info(f"Checking existence of {len(fixture_ids_str)} fixture IDs in 'matches' collection...")

        query = {"_id": {"$in": fixture_ids_str}}
        projection = {"_id": 1}

        cursor = self._matches_collection.find(query, projection)
        found_ids_str: Set[str] = {doc["_id"] for doc in cursor}
        logger.info(f"Found {len(found_ids_str)} existing matches in the collection.")

        missing_ids_int: List[int] = [
            fid for fid, fid_str in zip(fixture_ids_to_check, fixture_ids_str)
            if fid_str not in found_ids_str
        ]

        logger.info(f"Identified {len(missing_ids_int)} missing fixture IDs in 'matches'.")
        return missing_ids_int

    def get_match_processor_data(self, fixture_id: str) -> Optional[Dict[str, Any]]:
        assert self._initialized and self._match_processor_collection is not None, "DB not initialized or match_processor collection missing"
        assert isinstance(fixture_id, str) and fixture_id, "Fixture ID must be a non-empty string"
        logger.debug(f"Fetching data from match_processor collection for fixture_id: {fixture_id}")
        return self._match_processor_collection.find_one({"_id": fixture_id})


db_manager = MongoDBManager(db_name="agenticfc")

