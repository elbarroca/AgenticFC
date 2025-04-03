import os
import logging
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from datetime import datetime

logger = logging.getLogger(__name__)

class MongoDBManager:
    _instance = None
    _client = None
    _dbs = {}  # Dictionary to store database references
    _max_retries = 3  # Maximum number of connection retries

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MongoDBManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        load_dotenv()
        mongo_uri = os.getenv("MONGO_URI")

        if not mongo_uri:
            logger.error("MONGO_URI environment variable not set.")
            raise ValueError("MONGO_URI environment variable not set.")

        retry_count = 0
        connected = False
        
        while not connected and retry_count < self._max_retries:
            try:
                retry_count += 1
                logger.info(f"Attempting to connect to MongoDB (attempt {retry_count}/{self._max_retries})...")
                self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)  # Increased timeout from 5000 to 10000
                self._client.admin.command('ping')
                logger.info(f"Successfully connected to MongoDB")
                
                # Initialize database references
                self._dbs = {
                    'games': self._client['games'],
                    'matches': self._client['matches'],
                    'standings': self._client['standings'],
                    'odds': self._client['odds']
                }
                
                self._initialized = True
                connected = True
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"MongoDB connection attempt {retry_count} failed: {e}")
                if retry_count < self._max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"MongoDB connection failed after {self._max_retries} attempts: {e}")
                    self._client = None
                    self._dbs = {}
                    self._initialized = False
                    raise ConnectionFailure(f"MongoDB connection failed after {self._max_retries} attempts: {e}")
            except Exception as e:
                logger.error(f"An error occurred during MongoDB initialization: {e}")
                self._client = None
                self._dbs = {}
                self._initialized = False
                raise e

    def close_connection(self):
        if self._client is not None:
            self._client.close()
            logger.info("MongoDB connection closed.")
        # Reset state fully on close
        self._client = None
        self._dbs = {}
        self._initialized = False
        MongoDBManager._instance = None

    def _parse_date_components(self, date_str: str) -> Dict[str, str]:
        """Extract year, month, day from date string"""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return {
                "year": str(date_obj.year),
                "month": str(date_obj.month).zfill(2),
                "day": str(date_obj.day).zfill(2)
            }
        except ValueError:
            logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
            return {"year": "0000", "month": "00", "day": "00"}

    def _get_month_collection(self, db_type: str, date_str: str):
        """Get the monthly collection from the appropriate database"""
        date_parts = self._parse_date_components(date_str)
        month = date_parts["month"]
        collection_name = f"month_{month}"
        
        # Ensure the database type is valid
        if db_type not in self._dbs:
            logger.error(f"Invalid database type: {db_type}")
            return None
            
        return self._dbs[db_type][collection_name]

    # --- Games Methods ---
    
    def save_daily_games(self, date_str: str, games_data: Dict[str, Any]):
        """Saves the daily games list for a specific date."""
        try:
            date_parts = self._parse_date_components(date_str)
            day = date_parts["day"]
            
            # Get the appropriate monthly collection
            games_collection = self._get_month_collection('games', date_str)
            
            # Prepare data with day prefix in _id
            doc_id = f"day_{day}"
            
            # Save games data
            result = games_collection.update_one(
                {"_id": doc_id},
                {"$set": {
                    "date": date_str,
                    "data": games_data
                }},
                upsert=True
            )
            
            logger.info(f"Saved/Updated daily games for {date_str}. Modified: {result.modified_count}")
            return True
        except Exception as e:
            logger.error(f"Error saving daily games for {date_str} to MongoDB: {e}")
            return False

    def get_daily_games(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Retrieves the daily games list for a specific date."""
        try:
            date_parts = self._parse_date_components(date_str)
            day = date_parts["day"]
            
            # Get the appropriate monthly collection
            games_collection = self._get_month_collection('games', date_str)
            
            # Find by day ID
            doc_id = f"day_{day}"
            result = games_collection.find_one({"_id": doc_id})
            
            return result["data"] if result else None
        except Exception as e:
            logger.error(f"Error getting daily games for {date_str} from MongoDB: {e}")
            return None

    # --- Match Methods ---
    
    def save_match_data(self, date_str: str, fixture_id: str, match_data: Dict[str, Any]):
        """Saves or updates detailed match data."""
        try:
            date_parts = self._parse_date_components(date_str)
            day = date_parts["day"]
            
            # Get the appropriate monthly collection
            matches_collection = self._get_month_collection('matches', date_str)
            
            # Create document ID with day and fixture ID
            doc_id = f"day_{day}_fixture_{fixture_id}"
            
            # Include date and fixture ID in data
            match_data["date"] = date_str
            match_data["fixture_id"] = fixture_id
            
            # Save match data
            result = matches_collection.update_one(
                {"_id": doc_id},
                {"$set": match_data},
                upsert=True
            )
            
            logger.info(f"Saved/Updated match data for fixture {fixture_id} ({date_str}). Modified: {result.modified_count}")
            return True
        except Exception as e:
            logger.error(f"Error saving/updating match data for fixture {fixture_id} to MongoDB: {e}")
            return False

    def check_match_exists(self, date_str: str, fixture_id: str) -> bool:
        """Checks if a match with the given fixture ID exists."""
        try:
            date_parts = self._parse_date_components(date_str)
            day = date_parts["day"]
            
            # Get the appropriate monthly collection
            matches_collection = self._get_month_collection('matches', date_str)
            
            # Check if document exists
            doc_id = f"day_{day}_fixture_{fixture_id}"
            result = matches_collection.find_one({"_id": doc_id}, {"_id": 1})
            
            return result is not None
        except Exception as e:
            logger.error(f"Error checking match existence for fixture {fixture_id}: {e}")
            return False

    def get_match_fixture_ids_for_date(self, date_str: str) -> List[str]:
        """Gets all fixture IDs for matches saved on a specific date."""
        try:
            date_parts = self._parse_date_components(date_str)
            day = date_parts["day"]
            
            # Get the appropriate monthly collection
            matches_collection = self._get_month_collection('matches', date_str)
            
            # Find all documents for this day
            day_prefix = f"day_{day}_fixture_"
            cursor = matches_collection.find({"_id": {"$regex": f"^{day_prefix}"}}, {"fixture_id": 1})
            
            # Extract fixture IDs
            fixture_ids = []
            for doc in cursor:
                if "fixture_id" in doc:
                    fixture_ids.append(doc["fixture_id"])
                else:
                    # Fallback: extract from _id if fixture_id field is missing
                    doc_id = doc["_id"]
                    fixture_id = doc_id.replace(day_prefix, "")
                    fixture_ids.append(fixture_id)
                    
            return fixture_ids
        except Exception as e:
            logger.error(f"Error getting fixture IDs for date {date_str}: {e}")
            return []

    def get_match_data(self, date_str: str, fixture_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves detailed match data for a specific fixture."""
        try:
            date_parts = self._parse_date_components(date_str)
            day = date_parts["day"]
            
            # Get the appropriate monthly collection
            matches_collection = self._get_month_collection('matches', date_str)
            
            # Find the document
            doc_id = f"day_{day}_fixture_{fixture_id}"
            return matches_collection.find_one({"_id": doc_id})
        except Exception as e:
            logger.error(f"Error retrieving match data for fixture {fixture_id}: {e}")
            return None

    # --- Standings Methods ---
    
    def save_standings_data(self, date_str: str, league_id: str, season: int, standings_payload: Dict[str, Any]):
        """Saves league standings API response."""
        try:
            date_parts = self._parse_date_components(date_str)
            day = date_parts["day"]
            
            # Get the appropriate monthly collection
            standings_collection = self._get_month_collection('standings', date_str)
            
            # Create document ID
            doc_id = f"day_{day}_league_{league_id}_season_{season}"
            
            # Ensure required fields
            standings_payload["date"] = date_str
            standings_payload["league_id"] = str(league_id)
            standings_payload["season"] = season
            
            # Save standings data
            result = standings_collection.update_one(
                {"_id": doc_id},
                {"$set": standings_payload},
                upsert=True
            )
            
            logger.info(f"Saved/Updated standings for league {league_id}, season {season}. Modified: {result.modified_count}")
            return True
        except Exception as e:
            logger.error(f"Error saving standings for league {league_id}, season {season} to MongoDB: {e}")
            return False

    def get_standings_data(self, date_str: str, league_id: str, season: int) -> Optional[Dict[str, Any]]:
        """Retrieves league standings API response."""
        try:
            date_parts = self._parse_date_components(date_str)
            day = date_parts["day"]
            
            # Get the appropriate monthly collection
            standings_collection = self._get_month_collection('standings', date_str)
            
            # Find document
            doc_id = f"day_{day}_league_{league_id}_season_{season}"
            return standings_collection.find_one({"_id": doc_id})
        except Exception as e:
            logger.error(f"Error retrieving standings for league {league_id}, season {season}: {e}")
            return None

    def get_league_standings(self, league_id: str, season: int) -> Optional[Dict[str, Any]]:
        """
        Get the most recent standings data for a specific league and season.
        This method searches through recent collections to find the latest standings.
        
        Args:
            league_id: The ID of the league
            season: The season year
            
        Returns:
            Optional[Dict[str, Any]]: The most recent standings data for the league, or None if not found
        """
        try:
            # Get all month collections
            db = self._client['games']
            collections = sorted(
                [coll for coll in db.list_collection_names() if coll.startswith('month_standings_')],
                reverse=True  # Sort in reverse order to check most recent first
            )
            
            for collection_name in collections:
                collection = db[collection_name]
                # Search for any standings document for this league and season
                query = {
                    "_id": {"$regex": f"^day_\\d+_league_{league_id}_season_{season}$"}
                }
                standings = collection.find_one(query, sort=[("_id", -1)])  # Get most recent
                
                if standings:
                    logger.info(f"Found standings for league {league_id} in collection {collection_name}")
                    return standings
            
            logger.warning(f"No standings found for league {league_id}, season {season} in any collection")
            return None
            
        except Exception as e:
            logger.error(f"Error getting league standings for league {league_id}, season {season}: {e}")
            return None

    # --- Odds Methods ---
    
    def save_odds_data(self, date_str: str, fixture_id: str, odds_payload: Dict[str, Any]):
        """Saves odds data for a specific fixture."""
        try:
            date_parts = self._parse_date_components(date_str)
            day = date_parts["day"]
            
            # Get the appropriate monthly collection
            odds_collection = self._get_month_collection('odds', date_str)
            
            # Create document ID
            doc_id = f"day_{day}_fixture_{fixture_id}"
            
            # Include date reference
            odds_payload["date"] = date_str
            odds_payload["fixture_id"] = fixture_id
            
            # Save odds data
            result = odds_collection.update_one(
                {"_id": doc_id},
                {"$set": odds_payload},
                upsert=True
            )
            
            logger.info(f"Saved/Updated odds for fixture {fixture_id} ({date_str}). Modified: {result.modified_count}")
            return True
        except Exception as e:
            logger.error(f"Error saving odds for fixture {fixture_id} to MongoDB: {e}")
            return False

    def get_odds_data(self, date_str: str, fixture_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves odds data for a specific fixture."""
        try:
            date_parts = self._parse_date_components(date_str)
            day = date_parts["day"]
            
            # Get the appropriate monthly collection
            odds_collection = self._get_month_collection('odds', date_str)
            
            # Find document
            doc_id = f"day_{day}_fixture_{fixture_id}"
            return odds_collection.find_one({"_id": doc_id})
        except Exception as e:
            logger.error(f"Error retrieving odds for fixture {fixture_id}: {e}")
            return None

    def get_day_summary(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Retrieves a summary of all data available for a specific date."""
        try:
            date_parts = self._parse_date_components(date_str)
            day = date_parts["day"]
            day_prefix = f"day_{day}"
            
            # Create summary
            summary = {
                "_id": date_str,
                "date": date_str,
                "has_games": False,
                "has_matches": False,
                "has_standings": False,
                "has_odds": False,
                "games_count": 0,
                "matches_count": 0,
                "standings_count": 0,
                "odds_count": 0
            }
            
            # Check games
            games_collection = self._get_month_collection('games', date_str)
            games_doc = games_collection.find_one({"_id": day_prefix})
            if games_doc:
                summary["has_games"] = True
                if "data" in games_doc and "total_matches" in games_doc["data"]:
                    summary["games_count"] = games_doc["data"]["total_matches"]
            
            # Count matches
            matches_collection = self._get_month_collection('matches', date_str)
            matches_count = matches_collection.count_documents({"_id": {"$regex": f"^{day_prefix}"}})
            summary["matches_count"] = matches_count
            summary["has_matches"] = matches_count > 0
            
            # Count standings
            standings_collection = self._get_month_collection('standings', date_str)
            standings_count = standings_collection.count_documents({"_id": {"$regex": f"^{day_prefix}"}})
            summary["standings_count"] = standings_count
            summary["has_standings"] = standings_count > 0
            
            # Count odds
            odds_collection = self._get_month_collection('odds', date_str)
            odds_count = odds_collection.count_documents({"_id": {"$regex": f"^{day_prefix}"}})
            summary["odds_count"] = odds_count
            summary["has_odds"] = odds_count > 0
            
            return summary
        except Exception as e:
            logger.error(f"Error retrieving day summary for {date_str}: {e}")
            return None

# Singleton instance
db_manager = MongoDBManager()