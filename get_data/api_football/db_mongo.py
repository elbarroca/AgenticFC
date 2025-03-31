import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from datetime import datetime

logger = logging.getLogger(__name__)

class MongoDBManager:
    _instance = None
    _client = None
    _db = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MongoDBManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_name: str = "pregame_data"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        load_dotenv()  # Load environment variables from .env file
        mongo_uri = os.getenv("MONGO_URI")

        if not mongo_uri:
            logger.error("MONGO_URI environment variable not set.")
            raise ValueError("MONGO_URI environment variable not set.")

        try:
            logger.info("Attempting to connect to MongoDB...")
            # Explicitly set serverSelectionTimeoutMS to handle connection delays
            self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # The ismaster command is cheap and does not require auth.
            self._client.admin.command('ping') # Use ping for modern MongoDB versions
            self._db = self._client[db_name]
            logger.info(f"Successfully connected to MongoDB database: {db_name}")
            self._initialized = True
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            self._client = None
            self._db = None
            self._initialized = False # Ensure not marked as initialized
            raise ConnectionFailure(f"MongoDB connection failed: {e}")
        except Exception as e:
            logger.error(f"An error occurred during MongoDB initialization: {e}")
            self._client = None
            self._db = None
            self._initialized = False # Ensure not marked as initialized
            raise e

    def get_db(self):
        if self._db is None:
            logger.error("Database not initialized. Attempting to re-initialize...")
            # Attempt re-initialization if needed, potentially raising an error
            self.__init__()
            if self._db is None:
                 raise ConnectionFailure("Database could not be initialized.")
        return self._db

    def close_connection(self):
        if self._client is not None:
            self._client.close()
            logger.info("MongoDB connection closed.")
        # Reset state fully on close
        self._client = None
        self._db = None
        self._initialized = False
        MongoDBManager._instance = None

    # --- Helper Methods ---
    
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

    def _get_collection_path(self, date_str: str, collection_type: str) -> str:
        """Generate hierarchical collection path"""
        date_parts = self._parse_date_components(date_str)
        # Ensure collection type is part of the name
        return f"pregame.{date_parts['year']}.{date_parts['month']}.{date_parts['day']}.{collection_type}"

    # --- Collection Specific Methods ---

    def save_daily_games(self, date_str: str, games_data: Dict[str, Any]):
        """Saves the daily games list for a specific date using the new structure."""
        db = self.get_db()
        try:
            # Use hierarchical collection structure
            collection_name = self._get_collection_path(date_str, "daily_games")
            collection = db.get_collection(collection_name)
            
            # Clean up data - remove redundant information to optimize storage
            optimized_data = {
                "_id": date_str,
                "date": date_str,
                "total_matches": games_data.get("total_matches", 0),
                "leagues": {}
            }
            
            # Optimize league data structure
            for league_id, league_data in games_data.get("leagues", {}).items():
                optimized_data["leagues"][league_id] = {
                    "name": league_data.get("name", ""),
                    "country": league_data.get("country", ""),
                    "tier": league_data.get("tier", 0),
                    "matches": []
                }
                
                # Optimize match data
                for match in league_data.get("matches", []):
                    optimized_match = {
                        "id": match.get("id", ""),
                        "time": match.get("time", ""),
                        "home_team": {
                            "id": match.get("home_team", {}).get("id", ""),
                            "name": match.get("home_team", {}).get("name", "")
                        },
                        "away_team": {
                            "id": match.get("away_team", {}).get("id", ""),
                            "name": match.get("away_team", {}).get("name", "")
                        },
                        "status": match.get("status", {})
                    }
                    optimized_data["leagues"][league_id]["matches"].append(optimized_match)
            
            # Save optimized data
            result = collection.replace_one({"_id": date_str}, optimized_data, upsert=True)
            logger.info(f"Saved/Updated daily games for {date_str} in {collection_name}. Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted: {result.upserted_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving daily games for {date_str} to MongoDB: {e}")
            return False

    def get_daily_games(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Retrieves the daily games list for a specific date using the new structure."""
        db = self.get_db()
        try:
            collection_name = self._get_collection_path(date_str, "daily_games")
            collection = db.get_collection(collection_name)
            return collection.find_one({"_id": date_str})
        except Exception as e:
            logger.error(f"Error getting daily games for {date_str} from MongoDB: {e}")
            return None

    def save_match_data(self, date_str: str, fixture_id: str, match_data: Dict[str, Any]):
        """Saves or updates detailed match data (including raw stats/preds/odds/standings snapshot) using the new structure."""
        db = self.get_db()
        try:
            collection_name = self._get_collection_path(date_str, "matches")
            collection = db.get_collection(collection_name)
            
            fixture_id = str(fixture_id)
            
            # Use replace_one with upsert=True to handle both creation and updates
            # The match_data payload passed in should contain ALL fields for the match document
            # Ensure _id is set correctly within the match_data if creating new
            if "_id" not in match_data:
                match_data["_id"] = fixture_id
                
            result = collection.replace_one({"_id": fixture_id}, match_data, upsert=True)
            
            if result.upserted_id:
                logger.info(f"Created match data for fixture {fixture_id} ({date_str}) in {collection_name}")
            elif result.modified_count > 0:
                logger.info(f"Updated match data for fixture {fixture_id} ({date_str}) in {collection_name}")
            else:
                logger.info(f"Match data for fixture {fixture_id} ({date_str}) already up-to-date in {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving/updating match data for fixture {fixture_id} to MongoDB: {e}")
            return False

    def check_match_exists(self, date_str: str, fixture_id: str) -> bool:
        """Checks if a match with the given fixture ID exists in the new structure."""
        db = self.get_db()
        try:
            collection_name = self._get_collection_path(date_str, "matches")
            collection = db.get_collection(collection_name)
            return collection.count_documents({"_id": str(fixture_id)}) > 0
        except Exception as e:
            logger.error(f"Error checking match existence for fixture {fixture_id}: {e}")
            return False

    def get_match_fixture_ids_for_date(self, date_str: str) -> List[str]:
        """Gets all fixture IDs for matches saved on a specific date using the new structure."""
        db = self.get_db()
        try:
            collection_name = self._get_collection_path(date_str, "matches")
            collection = db.get_collection(collection_name)
            # Find documents matching the date and project only the _id field
            cursor = collection.find({}, {"_id": 1})
            return [doc["_id"] for doc in cursor]
        except Exception as e:
            logger.error(f"Error getting fixture IDs for date {date_str}: {e}")
            return []

    def get_match_data(self, date_str: str, fixture_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves detailed match data for a specific fixture using the new structure."""
        db = self.get_db()
        try:
            collection_name = self._get_collection_path(date_str, "matches")
            collection = db.get_collection(collection_name)
            return collection.find_one({"_id": str(fixture_id)})
        except Exception as e:
            logger.error(f"Error retrieving match data for fixture {fixture_id}: {e}")
            return None

    def save_standings_data(self, date_str: str, league_id: str, season: int, standings_payload: Dict[str, Any]):
        """Saves raw league standings API response using the new structure."""
        db = self.get_db()
        try:
            # Store standings in their dedicated collection per day
            collection_name = self._get_collection_path(date_str, "standings")
            collection = db.get_collection(collection_name)
            
            # Use a composite key of league_id and season as the document ID
            doc_id = f"{league_id}_{season}"
            
            # Ensure the payload has the _id field set
            standings_payload["_id"] = doc_id 
            # Ensure league_id, season, date are present for querying
            standings_payload["league_id"] = str(league_id)
            standings_payload["season"] = season
            standings_payload["date"] = date_str
            
            # The standings_payload already contains the raw `standings_api_response` field
            
            result = collection.replace_one({"_id": doc_id}, standings_payload, upsert=True)
            logger.info(f"Saved/Updated raw standings for league {league_id}, season {season} in {collection_name}. Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted: {result.upserted_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving raw standings for league {league_id}, season {season} to MongoDB: {e}")
            return False

    def get_standings_data(self, date_str: str, league_id: str, season: int) -> Optional[Dict[str, Any]]:
        """Retrieves raw league standings API response using the new structure."""
        db = self.get_db()
        try:
            collection_name = self._get_collection_path(date_str, "standings")
            collection = db.get_collection(collection_name)
            doc_id = f"{league_id}_{season}"
            # Return the full document containing the raw response
            return collection.find_one({"_id": doc_id})
        except Exception as e:
            logger.error(f"Error retrieving raw standings for league {league_id}, season {season}: {e}")
            return None

# Singleton instance
db_manager = MongoDBManager() 