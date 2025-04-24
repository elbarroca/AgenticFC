import os
import logging
from datetime import date, datetime, timezone
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import re
import sys
import time
import requests
from dotenv import load_dotenv

# Import MongoDB manager from existing code
try:
    from api_football.db_mongo import MongoDBManager
    from api_football.db_ids.team_id_mappings import TEAM_ID_MAPPING
    from api_football.db_ids.league_id_mappings import LEAGUE_ID_MAPPING
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'unified_data')
OPENWEATHERMAP_API_KEY = os.environ.get('OPENWEATHERMAP_API_KEY', "6bc36c1099f96f6a1e985c9d3b959744")
OPENWEATHERMAP_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHERMAP_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# --- Rate Limiting Tracking ---
WEATHER_API_CALL_COUNT = 0
WEATHER_API_LIMIT_PER_DAY = 990
LAST_RESET_DATE = date.today()

# Text normalization mapping
TERM_STANDARDIZATION = {
    'loses': 'losses',
    'lose': 'loss',
    'scored goals': 'goals_scored',
    'conceded goals': 'goals_conceded',
    'clean sheet': 'clean_sheets',
    'failure to score': 'failed_to_score',
    'matches over': 'over',
    'matches under': 'under',
    'winner': 'win',
    'wind_speed': 'wind_speed',
    'wind_deg': 'wind_direction',
    'temp': 'temperature',
    'pressure': 'air_pressure',
}

class DailyGameExtractor:
    """
    Extracts game data from MongoDB database for a specific day,
    including previous matches and weather forecasts.
    """
    
    def __init__(self, use_mongo=True):
        """Initialize the extractor with MongoDB connection."""
        self.mongo_db = None
        self.openweathermap_api_key = OPENWEATHERMAP_API_KEY

        if use_mongo:
            try:
                # Ensure MongoDBManager is configured for the Agentic FC Workflow database
                self.mongo_db = MongoDBManager() # Assuming it connects to the correct DB
                logger.info("✅ MongoDB connection successful (ensure it's configured for Agentic FC Workflow DB)")
            except Exception as e:
                logger.error(f"❌ Failed to connect to MongoDB: {e}")
                # For Agentic FC Workflow, MongoDB is crucial
                raise ConnectionError("MongoDB connection is required for DailyGameExtractor") from e
        else:
             raise ValueError("`use_mongo` must be True for Agentic FC Workflow")
    
    def get_current_date_str(self) -> str:
        """Get today's date as a string in YYYY-MM-DD format."""
        return date.today().strftime("%Y-%m-%d")
    
    def extract_games_for_date(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts all game data strictly from MongoDB for a specific date,
        fetches previous matches for involved teams, and processes it.
        
        Args:
            date_str: Date string in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary containing all the extracted and processed game data for the day.
        """
        # Use today's date if none is provided
        if not date_str:
            date_str = self.get_current_date_str()
            
        logger.info(f"🚀 Starting extraction for date: {date_str} from MongoDB (Agentic FC Workflow)")
        
        processed_games_output = [] # Store final processed game objects
        fixture_ids = []

        if not self.mongo_db:
            logger.error("❌ MongoDB connection not available. Cannot extract daily games.")
            # Return an empty structure or raise an error
            return {"date": date_str, "games": [], "total_games": 0, "error": "MongoDB connection failed"}

        try:
            # Step 1: Get the list of fixture IDs for the target date directly from MongoDB
            fixture_ids = self.mongo_db.get_match_fixture_ids_for_date(date_str)

            if not fixture_ids:
                logger.warning(f"⚠️ No fixture IDs found in MongoDB for {date_str}.")
                # Return early if no games are scheduled
                return {"date": date_str, "games": [], "total_games": 0, "message": "No games found in MongoDB for this date."}

            logger.info(f"✅ Found {len(fixture_ids)} fixture IDs for {date_str} in MongoDB.")

            # Step 2: Fetch detailed data for each fixture ID from MongoDB
            for fixture_id in fixture_ids:
                logger.debug(f"Fetching details for fixture ID: {fixture_id}")
                match_data = None
                try:
                    # Fetch detailed data using the correct method and ensure fixture_id is a string
                    match_data = self.mongo_db.get_match_data(str(fixture_id))
                    
                    if match_data:
                        # Process the fetched data (includes fetching previous games)
                        logger.info(f"Processing game data for fixture {fixture_id}")
                        processed_game = self.process_game_data(match_data)

                        if processed_game:
                            # Save individual game file (will use data within processed_game)
                            self.save_individual_game_file(processed_game)
                            processed_games_output.append({
                                "fixture_id": processed_game.get("fixture_info", {}).get("id"),
                                "status": "Processed and saved"
                            })
                        else:
                            logger.warning(f"⚠️ Processing failed or returned None for fixture {fixture_id}.")

                    else:
                        logger.warning(f"⚠️ No detailed match data found in MongoDB for fixture {fixture_id}.")
                except Exception as e:
                    logger.error(f"❌ Error processing fixture {fixture_id}: {e}", exc_info=True)
                    # Continue with the next fixture_id
                    continue
        except Exception as e:
            logger.error(f"❌ Failed to retrieve or process games from MongoDB for {date_str}: {e}", exc_info=True)
            # Return partial results or indicate failure
            return {
                "date": date_str,
                "games_processed_summary": processed_games_output, # Return summary of processed games
                "total_games_attempted": len(fixture_ids),
                "error": f"Failed during MongoDB fetch/process: {str(e)}"
            }

        # Prepare the final result dictionary
        result = {
            "date": date_str,
            "games_processed_summary": processed_games_output,
            "total_games_processed": len(processed_games_output),
            "total_games_found_for_date": len(fixture_ids)
        }

        logger.info(f"✅ Successfully attempted processing for {len(fixture_ids)} games found for {date_str} from MongoDB.")
        logger.info(f"✅ {len(processed_games_output)} games were successfully processed and saved.")
        return result
    
    def _get_previous_team_matches(self, team_id: int, current_game_timestamp: int, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Fetches the details of the previous N matches for a given team before a specific game time
        by querying the 'matches' collection.

        Args:
            team_id: The ID of the team.
            current_game_timestamp: The timestamp of the current game (to find games before this).
            limit: The maximum number of previous matches to retrieve.

        Returns:
            A list of dictionaries, each representing a raw previous game document from 'matches'.
        """
        if not self.mongo_db:
            logger.error("MongoDB connection not available. Cannot fetch previous matches.")
            return []

        logger.info(f"Fetching previous {limit} matches for team ID {team_id} from 'matches' collection before timestamp {current_game_timestamp}")
        try:
            # Call the DB method which queries the 'matches' collection
            previous_matches = self.mongo_db.get_previous_matches_for_team(
                team_id=team_id,
                date_before_timestamp=current_game_timestamp,
                limit=limit
            )
            # The result is already a list of raw match documents
            logger.info(f"Found {len(previous_matches)} previous matches in 'matches' collection for team ID {team_id}.")

            # Optional: Clean internal _id from these results if they exist
            for match in previous_matches:
                if "_id" in match:
                    try:
                        # Attempt removal, ignore if it fails (e.g., already removed)
                        del match["_id"]
                    except Exception:
                        pass
            return previous_matches
        except Exception as e:
            logger.error(f"❌ Error fetching previous matches for team {team_id} from 'matches' collection: {e}", exc_info=True)
            return []

    def _fetch_weather_forecast_by_city(self, city: str, country: Optional[str] = None, timestamp: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Fetches weather data for a specific city and optional country using OpenWeatherMap API.

        Args:
            city: City name.
            country: Optional country code (2-letter ISO code).
            timestamp: Unix timestamp for the desired forecast time.

        Returns:
            Dictionary containing the weather data from the API response, or error info if it fails.
        """
        global WEATHER_API_CALL_COUNT, LAST_RESET_DATE, WEATHER_API_LIMIT_PER_DAY

        # --- Rate Limiting Check ---
        today = date.today()
        if today != LAST_RESET_DATE:
            logger.info(f"Resetting weather API call count for new day: {today}")
            WEATHER_API_CALL_COUNT = 0
            LAST_RESET_DATE = today

        if WEATHER_API_CALL_COUNT >= WEATHER_API_LIMIT_PER_DAY:
            logger.warning(f"OpenWeatherMap API daily limit ({WEATHER_API_LIMIT_PER_DAY}) reached. Skipping weather fetch.")
            return {"error": "API limit reached"}

        # Determine whether to use current weather or forecast API based on timestamp
        use_forecast = False
        if timestamp:
            current_timestamp = int(time.time())
            # If timestamp is more than 3 hours in the future, use forecast API
            if timestamp > current_timestamp + (3 * 3600):
                use_forecast = True
                logger.info(f"Using forecast API for timestamp {timestamp} ({datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')})")

        # Build the query parameter for city/country
        location_query = city
        if country:
            location_query = f"{city},{country}"

        params = {
            'q': location_query,
            'appid': self.openweathermap_api_key,
            'units': 'metric'  # Use metric units (Celsius, m/s)
        }

        # For forecast endpoint, we might need to filter based on timestamp later
        api_url = OPENWEATHERMAP_CURRENT_URL if not use_forecast else OPENWEATHERMAP_FORECAST_URL
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()

            # Increment count only on successful fetch
            WEATHER_API_CALL_COUNT += 1
            logger.info(f"Weather API call successful for {location_query}. Count: {WEATHER_API_CALL_COUNT}/{WEATHER_API_LIMIT_PER_DAY}")

            weather_data = response.json()
            
            if use_forecast and 'list' in weather_data:
                # Filter forecast data to find closest to requested timestamp
                forecast_items = weather_data['list']
                closest_forecast = None
                min_diff = float('inf')
                
                for item in forecast_items:
                    item_ts = item.get('dt', 0)
                    diff = abs(item_ts - timestamp)
                    if diff < min_diff:
                        min_diff = diff
                        closest_forecast = item
                
                if closest_forecast:
                    # Add city info to the forecast item
                    closest_forecast['city'] = {
                        'name': weather_data.get('city', {}).get('name', city),
                        'country': weather_data.get('city', {}).get('country', country)
                    }
                    return closest_forecast
                else:
                    return {"error": "Could not find forecast item close to requested timestamp"}
            
            # For current weather, return the response directly
            return weather_data

        except requests.exceptions.Timeout:
            logger.error(f"Weather API request timed out for {location_query}.")
            return {"error": "Request timed out"}
        except requests.exceptions.HTTPError as http_err:
            # Error handling for HTTP errors
            if response.status_code == 401:
                logger.error(f"Weather API Error 401: Unauthorized. Check API key.")
                return {"error": "Unauthorized - Check API Key"}
            elif response.status_code == 404:
                logger.warning(f"Weather API Error 404: City not found: {location_query}")
                return {"error": "City not found"}
            elif response.status_code == 429:
                logger.warning(f"Weather API Error 429: Rate limit exceeded (API side).")
                WEATHER_API_CALL_COUNT = WEATHER_API_LIMIT_PER_DAY  # Assume we hit the limit
                return {"error": "API rate limit exceeded (OpenWeatherMap)"}
            else:
                logger.error(f"Weather API HTTP error occurred: {http_err} - Status Code: {response.status_code}")
                return {"error": f"HTTP error: {response.status_code}"}
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Weather API request failed: {req_err}")
            return {"error": "Request failed"}
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON response from Weather API for {location_query}.")
            return {"error": "Invalid JSON response"}
        except Exception as e:
            logger.error(f"An unexpected error occurred during weather fetch: {e}", exc_info=True)
            return {"error": f"Unexpected error during weather fetch: {str(e)}"}

    def _normalize_string_value(self, value: Any) -> Any:
        """
        Normalize string values to appropriate types:
        - "80%" -> 0.8 (float)
        - "1.4" -> 1.4 (float)
        - "6" -> 6 (int)
        
        Args:
            value: The value to normalize
            
        Returns:
            Normalized value (int, float, or original value)
        """
        if not isinstance(value, str):
            return value
            
        # Strip whitespace
        value = value.strip()
        
        # Try to convert percentages
        if value.endswith('%'):
            try:
                # Convert "80%" to 0.8
                percentage = value.rstrip('%')
                return float(percentage) / 100
            except (ValueError, TypeError):
                pass
                
        # Try to convert to integer
        try:
            if '.' not in value:
                return int(value)
        except (ValueError, TypeError):
            pass
            
        # Try to convert to float
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
            
        # Return original value if conversion fails
        return value

    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text by:
        - Removing leading/trailing whitespace
        - Standardizing vocabulary terms
        - Converting to lowercase
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return text
            
        # Trim whitespace
        sanitized = text.strip()
        
        # Convert to lowercase (cautiously - only if not a proper noun)
        if not (sanitized and sanitized[0].isupper() and len(sanitized) > 1 and sanitized[1:].islower()):
            sanitized = sanitized.lower()
        
        # Replace standardized terms
        for original, replacement in TERM_STANDARDIZATION.items():
            # Only replace full words or phrases, not substrings
            pattern = r'\b' + re.escape(original) + r'\b'
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
            
        return sanitized

    def _sanitize_keys_recursively(self, obj: Any) -> Any:
        """
        Recursively sanitize dictionary keys.
        
        Args:
            obj: Object to process (dict, list, or scalar)
            
        Returns:
            Processed object with sanitized keys
        """
        if isinstance(obj, dict):
            return {self._sanitize_text(k): self._sanitize_keys_recursively(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_keys_recursively(item) for item in obj]
        else:
            return obj

    def _normalize_values_recursively(self, obj: Any) -> Any:
        """
        Recursively normalize values in a nested structure.
        
        Args:
            obj: Object to process (dict, list, or scalar)
            
        Returns:
            Processed object with normalized values
        """
        if isinstance(obj, dict):
            return {k: self._normalize_values_recursively(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_values_recursively(item) for item in obj]
        else:
            return self._normalize_string_value(obj)

    def process_game_data(self, match_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process match data and structure it in the new format:
        {
          "fixture_id": "string",
          "date": "YYYY-MM-DD",
          "league": { ... },
          "fixture_meta": { ... },
          "team_A": { ... },
          "team_B": { ... },
          "standings": { ... }
        }
        """
        if "_id" in match_data:
            try:
                del match_data["_id"]
            except Exception:
                logger.warning("Could not remove _id field from base match_data.")

        fixture_id = match_data.get('fixture_id') or match_data.get('fixture', {}).get('id')
        if not fixture_id:
            logger.error("❌ Missing fixture ID in base match data. Cannot process.")
            return None
        fixture_id_str = str(fixture_id)
        logger.info(f"Processing unified data for fixture ID: {fixture_id_str}")

        # --- 1. Extract Core Information ---
        core_info = self._extract_core_match_info(match_data)
        if not core_info:
            logger.error(f"❌ Failed to extract core info for fixture {fixture_id_str}. Skipping.")
            return None

        fixture_info = core_info["fixture_info"]
        league_info = core_info["league"]
        home_team_base = core_info["teams"]["home"]
        away_team_base = core_info["teams"]["away"]
        current_game_timestamp = fixture_info.get("timestamp")
        home_team_id = home_team_base.get("id")
        away_team_id = away_team_base.get("id")
        
        # --- Get City & Country for Weather ---
        venue_data = fixture_info.get("venue", {})
        city = venue_data.get("city")
        country = league_info.get("country")
        
        if not all([current_game_timestamp, home_team_id, away_team_id]):
            logger.error(f"❌ Missing timestamp or team IDs in core info for fixture {fixture_id_str}. Skipping.")
            return None

        # --- 2. Add StatArea ID Mappings ---
        self._add_statarea_id_mappings_basic(home_team_base, away_team_base, league_info)
        home_statarea_id = home_team_base.get("statarea_id", "unknown")
        away_statarea_id = away_team_base.get("statarea_id", "unknown")

        # --- 3. Fetch Weather Data for Fixture Meta ---
        weather_forecast = None
        if city:
            logger.info(f"Attempting to fetch weather forecast for fixture {fixture_id_str} in {city}, {country}")
            weather_forecast = self._fetch_weather_forecast_by_city(city, country, current_game_timestamp)
            if not weather_forecast or "error" in weather_forecast:
                error_msg = weather_forecast.get("error", "Failed to fetch") if weather_forecast else "None returned"
                logger.warning(f"Could not retrieve valid weather forecast for fixture {fixture_id_str} in {city}. Reason: {error_msg}")
                weather_forecast = {"error": error_msg}
            else:
                logger.info(f"Successfully fetched weather forecast for fixture {fixture_id_str} in {city}")
                # Add a summary for easier consumption
                main_weather = weather_forecast.get("weather", [{}])[0].get("main", "Unknown")
                description = weather_forecast.get("weather", [{}])[0].get("description", "Unknown conditions")
                temp = weather_forecast.get("main", {}).get("temp", "Unknown")
                weather_forecast["summary"] = f"{main_weather}: {description}, Temp: {temp}°C"
        else:
            logger.warning(f"Cannot fetch weather: Missing city information for fixture {fixture_id_str}")

        # --- 4. Fetch Additional Data from MongoDB ---
        # Match processor data (contains team stats for both teams)
        match_processor_data = self.mongo_db.get_match_processor_data(fixture_id_str) or {}
        
        # Extract standings from match_processor_data if available
        standings_list = []
        if match_processor_data:
             standings_list = match_processor_data.get("standings_snapshot", {}).get("standings", [])
             if standings_list and isinstance(standings_list, list) and len(standings_list) > 0:
                 # Often standings are wrapped in another list, handle that
                 standings_list = standings_list[0] if isinstance(standings_list[0], list) else standings_list

        # Find specific team standings
        home_standings_info = None
        away_standings_info = None
        if standings_list and isinstance(standings_list, list):
            for team_standing in standings_list:
                if isinstance(team_standing, dict) and team_standing.get("team", {}).get("id") == home_team_id:
                    home_standings_info = team_standing
                if isinstance(team_standing, dict) and team_standing.get("team", {}).get("id") == away_team_id:
                    away_standings_info = team_standing

        # Extract team-specific processor data (after getting standings)
        home_processor_data = match_processor_data.get("home_team_stats", {}) if match_processor_data else {}
        away_processor_data = match_processor_data.get("away_team_stats", {}) if match_processor_data else {}
        
        # Include predictions for individual teams if available
        if "predictions" in match_processor_data:
            pred_data = match_processor_data.get("predictions", {})
            # Include team-specific prediction data if available
            if "teams" in pred_data:
                teams_pred = pred_data.get("teams", {})
                if "home" in teams_pred:
                    home_processor_data["predictions"] = teams_pred.get("home", {})
                if "away" in teams_pred:
                    away_processor_data["predictions"] = teams_pred.get("away", {})

        # --- Get StatArea Data ---
        home_statarea_data = None
        if home_statarea_id != "unknown":
            home_statarea_data = self.mongo_db.get_latest_statarea_data(home_statarea_id, "host", 15)
            if not home_statarea_data:
                logger.warning(f"⚠️ No StatArea data found for home team {home_statarea_id} (host, 15)")

        away_statarea_data = None
        if away_statarea_id != "unknown":
            away_statarea_data = self.mongo_db.get_latest_statarea_data(away_statarea_id, "guest", 15)
            if not away_statarea_data:
                logger.warning(f"⚠️ No StatArea data found for away team {away_statarea_id} (guest, 15)")

        # --- Fetch Previous Matches ---
        home_previous_matches = self._get_previous_team_matches(home_team_id, current_game_timestamp, limit=15)
        away_previous_matches = self._get_previous_team_matches(away_team_id, current_game_timestamp, limit=15)

        # --- 5. Calculate Engineered Features ---
        # Pass the specific standings info for each team
        home_engineered_features = self._generate_engineered_features(
            team_id=home_team_id,
            processor_data=home_processor_data,
            statarea_data=home_statarea_data,
            previous_matches=home_previous_matches,
            standings_info=home_standings_info # Pass specific standings
        )

        away_engineered_features = self._generate_engineered_features(
            team_id=away_team_id,
            processor_data=away_processor_data,
            statarea_data=away_statarea_data,
            previous_matches=away_previous_matches,
            standings_info=away_standings_info # Pass specific standings
        )

        # --- 6. Construct the New Unified JSON Structure ---
        # Extract date part (YYYY-MM-DD) from fixture info
        match_date = fixture_info.get("date", "").split("T")[0] if fixture_info.get("date") else None
        
        # Create fixture metadata
        fixture_meta = {
            "id": fixture_id_str,
            "date": fixture_info.get("date"),
            "timestamp": fixture_info.get("timestamp"),
            "referee": fixture_info.get("referee"),
            "venue": venue_data,
            "status": fixture_info.get("status"),
            "weather_forecast": weather_forecast
        }
        
        # Format home team as team_A
        team_A = {
            "team_id": home_team_id,
            "name": home_team_base.get("name"),
            "logo": home_team_base.get("logo"),
            "statarea_id": home_statarea_id,
            "is_home": True,
            "engineered_features": home_engineered_features,
            "raw_data": {
                "match_processor": home_processor_data,
                "statarea_stats": home_statarea_data or {},
                "recent_matches": [
                    {"source": "matches", "data": match} 
                    for match in home_previous_matches
                ]
            }
        }
        
        # Format away team as team_B
        team_B = {
            "team_id": away_team_id,
            "name": away_team_base.get("name"),
            "logo": away_team_base.get("logo"),
            "statarea_id": away_statarea_id,
            "is_home": False,
            "engineered_features": away_engineered_features,
            "raw_data": {
                "match_processor": away_processor_data,
                "statarea_stats": away_statarea_data or {},
                "recent_matches": [
                    {"source": "matches", "data": match} 
                    for match in away_previous_matches
                ]
            }
        }
        
        # Construct the final structure
        unified_data = {
            "fixture_id": fixture_id_str,
            "date": match_date,
            "league": league_info,
            "fixture_meta": fixture_meta,
            "team_a": team_A, # Contains engineered_features with standings_snapshot
            "team_b": team_B, # Contains engineered_features with standings_snapshot
            # Decide if you still need the full raw standings here, or remove it
            # "standings": match_processor_data.get("standings_snapshot", {})
        }

        # --- 7. Normalize and Sanitize the Data ---
        # First sanitize keys recursively
        sanitized_data = self._sanitize_keys_recursively(unified_data)
        
        # Then normalize string values to appropriate types
        normalized_data = self._normalize_values_recursively(sanitized_data)
        
        # --- 8. Final Cleanup ---
        cleaned_data = self._remove_none_unknown_empty(normalized_data)

        if not cleaned_data:
            logger.error(f"❌ Data structure became empty after cleaning for fixture {fixture_id_str}. Skipping save.")
            return None

        logger.info(f"✅ Successfully processed and structured unified data for fixture {fixture_id_str}.")
        return cleaned_data
    
    def _generate_engineered_features(self, team_id: int, processor_data: Dict[str, Any],
                                      statarea_data: Optional[Dict[str, Any]],
                                      previous_matches: List[Dict[str, Any]],
                                      standings_info: Optional[Dict[str, Any]],
                                      form_pattern: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive engineered features for a team, prioritizing StatArea history
        and including relevant standings information.

        Args:
            team_id: Team ID
            processor_data: Data from match_processor (already sanitized and normalized)
            statarea_data: Data from StatArea (already sanitized and normalized), can be None.
            previous_matches: Previous matches from 'matches' collection (fallback history)
            standings_info: Dictionary containing standings data for this specific team, can be None.
            form_pattern: Optional explicit form string pattern (for specific analysis)

        Returns:
            Dictionary of engineered features.
        """
        features = {}
        history_analysis = {}

        # --- Prioritize StatArea Match History ---
        statarea_history = []
        if isinstance(statarea_data, dict):
            statarea_history = statarea_data.get("match_history", [])

        if statarea_history and isinstance(statarea_history, list) and len(statarea_history) > 0:
            logger.info(f"Analyzing StatArea match history ({len(statarea_history)} games) for team {team_id}")
            history_analysis = self._analyze_match_history(statarea_history, team_id, source="statarea")
        else:
            # --- Fallback to Previous Matches from 'matches' collection ---
            if previous_matches and isinstance(previous_matches, list) and len(previous_matches) > 0:
                logger.info(f"Analyzing 'matches' collection history ({len(previous_matches)} games) for team {team_id} (StatArea history missing)")
                matches_to_analyze = [m.get("data", m) for m in previous_matches]
                history_analysis = self._analyze_match_history(matches_to_analyze, team_id, source="matches")
            else:
                logger.warning(f"No match history found from StatArea or 'matches' collection for team {team_id}. Limited features.")

        # Add the analyzed history stats to features
        features.update(history_analysis)

        # --- Add Standings Information ---
        if isinstance(standings_info, dict):
             # Add relevant fields, ensuring keys are sanitized if needed
             standing_features = {
                 "rank": standings_info.get("rank"),
                 "points": standings_info.get("points"),
                 "goals_diff": standings_info.get("goalsdiff"), # Assuming 'goalsdiff' key
                 "form": standings_info.get("form"), # Standing form might differ from calculated history
                 "group": standings_info.get("group")
             }
             # Remove None values from standing_features before adding
             features["standings_snapshot"] = {k: v for k, v in standing_features.items() if v is not None}

        # --- Add Supplemental Features from Processor Data (Goal Timings as %) ---
        if processor_data and "goals" in processor_data:
            goal_data = processor_data.get("goals", {})
            proc_total_scored = goal_data.get("for", {}).get("total", {}).get("total", 0)
            proc_total_conceded = goal_data.get("against", {}).get("total", {}).get("total", 0)
            timing_features = {}

            for_per_minute = goal_data.get("for", {}).get("minute", {})
            if for_per_minute and proc_total_scored and proc_total_scored > 0:
                for bracket, value in for_per_minute.items():
                    count = value.get("total")
                    if count is not None:
                        timing_features[f"proc_goals_scored_{bracket}_pct"] = count / proc_total_scored

            against_per_minute = goal_data.get("against", {}).get("minute", {})
            if against_per_minute and proc_total_conceded and proc_total_conceded > 0:
                for bracket, value in against_per_minute.items():
                    count = value.get("total")
                    if count is not None:
                        timing_features[f"proc_goals_conceded_{bracket}_pct"] = count / proc_total_conceded

            if timing_features:
                 features["processor_goal_timing_stats"] = timing_features


        # --- Add Supplemental Features from StatArea (General / Betting) ---
        if isinstance(statarea_data, dict):
             gen_stats = statarea_data.get("general_statistics", {})
             bet_stats = statarea_data.get("team_bet_statistics", {}) # Should be sanitized key
             statarea_supplemental = {}

             keys_to_check = [
                 "chance_to_score_goal_next_match", "chance_to_conceded_goal_next_match",
                 "time_without_scored_goal", "time_without_conceded_goal"
             ]
             for key in keys_to_check:
                 if key in gen_stats:
                     statarea_supplemental[f"statarea_gen_{key}"] = gen_stats[key]

             ou_25_key = "over/under_2.5_for_all_goals_in_matches"
             if bet_stats and ou_25_key in bet_stats:
                  statarea_supplemental[f"statarea_bet_o25_over"] = bet_stats[ou_25_key].get("over")
                  statarea_supplemental[f"statarea_bet_o25_under"] = bet_stats[ou_25_key].get("under")

             if statarea_supplemental:
                  features["statarea_supplemental_stats"] = statarea_supplemental

        # --- Form pattern parsing (if provided explicitly) ---
        if form_pattern:
            pattern_features = self._parse_form_pattern(form_pattern)
            if pattern_features:
                 features["explicit_form_pattern_analysis"] = pattern_features

        return features

    def _analyze_match_history(self, match_history: List[Dict[str, Any]], team_id: int, source: str = "unknown") -> Dict[str, Any]:
        """
        Analyzes a list of past matches (up to 15) to extract detailed statistics,
        including overall, home/away splits, rolling stats, performance by result,
        opponent names, and scores.

        Note: Retrieving opponent's historical standing for each past match is
              not feasible with the current data sources. Opponent context is
              approximated via performance_by_result analysis.

        Args:
            match_history: List of match dictionaries (StatArea or Matches format).
            team_id: The ID of the team to analyze for.
            source: Identifier for the source ('statarea' or 'matches').

        Returns:
            A dictionary containing structured engineered statistics.
        """
        # --- Initialization ---
        stats = {
            "overall": {"m": 0, "w": 0, "d": 0, "l": 0, "sc": 0, "cn": 0, "btts": 0, "cs": 0, "fts": 0, "tg_list": []},
            "home":    {"m": 0, "w": 0, "d": 0, "l": 0, "sc": 0, "cn": 0, "btts": 0, "cs": 0, "fts": 0, "tg_list": []},
            "away":    {"m": 0, "w": 0, "d": 0, "l": 0, "sc": 0, "cn": 0, "btts": 0, "cs": 0, "fts": 0, "tg_list": []},
            "by_result": {
                "win":  {"m": 0, "matches": []},
                "draw": {"m": 0, "matches": []},
                "loss": {"m": 0, "matches": []}
            },
            "form_overall": "", "form_home": "", "form_away": "",
            "recent_matches_overall": [],
            "recent_matches_home": [],
            "recent_matches_away": []
        }
        MAX_HISTORY = 15
        history_to_analyze = match_history[:MAX_HISTORY]
        games_analyzed_count = len(history_to_analyze)

        # --- Main Loop (Oldest to Newest for Form String Construction) ---
        for match in reversed(history_to_analyze):
            venue, result, scored, conceded, opponent_name = None, None, None, None, "Unknown Opponent"

            # --- Extract data based on source ---
            if source == "statarea" and isinstance(match, dict):
                venue = match.get("venue")
                result = match.get("result")
                scored = match.get("team_goals")
                conceded = match.get("opponent_goals")
                opponent_name = match.get("opponent", opponent_name)

            elif source == "matches" and isinstance(match, dict):
                teams_data = match.get("teams", {})
                home_team = teams_data.get("home", {})
                away_team = teams_data.get("away", {})
                home_team_id = home_team.get("id")
                away_team_id = away_team.get("id")
                goals_data = match.get("goals", {})
                home_goals, away_goals = goals_data.get("home"), goals_data.get("away")

                if home_team_id == team_id:
                    venue, scored, conceded = "home", home_goals, away_goals
                    opponent_name = away_team.get("name", opponent_name)
                elif away_team_id == team_id:
                    venue, scored, conceded = "away", away_goals, home_goals
                    opponent_name = home_team.get("name", opponent_name)
                else:
                    continue

                if scored is not None and conceded is not None:
                    try:
                        sc_int, cn_int = int(scored), int(conceded)
                        if sc_int > cn_int: result = "win"
                        elif sc_int < cn_int: result = "loss"
                        else: result = "draw"
                    except (ValueError, TypeError): result = None

            # Skip if essential data is missing
            if venue not in ["home", "away"] or result not in ["win", "draw", "loss"] or scored is None or conceded is None:
                logger.debug(f"Skipping match in history analysis due to missing data")
                continue

            # Aggregate stats
            try:
                sc, cn = int(scored), int(conceded)
                res_char = result[0].upper()
                match_summary = {"result": res_char, "sc": sc, "cn": cn, "opponent": opponent_name, "venue": venue}

                # Overall
                o = stats["overall"]; o["m"] += 1; o["sc"] += sc; o["cn"] += cn; o["tg_list"].append(sc + cn)
                if result == "win": o["w"] += 1
                elif result == "draw": o["d"] += 1
                else: o["l"] += 1
                if sc > 0 and cn > 0: o["btts"] += 1
                if cn == 0: o["cs"] += 1
                if sc == 0: o["fts"] += 1
                stats["form_overall"] += res_char
                stats["recent_matches_overall"].append(match_summary)

                # Venue specific
                v = stats[venue]; v["m"] += 1; v["sc"] += sc; v["cn"] += cn; v["tg_list"].append(sc + cn)
                if result == "win": v["w"] += 1
                elif result == "draw": v["d"] += 1
                else: v["l"] += 1
                if sc > 0 and cn > 0: v["btts"] += 1
                if cn == 0: v["cs"] += 1
                if sc == 0: v["fts"] += 1
                stats[f"form_{venue}"] += res_char
                if venue == 'home': stats["recent_matches_home"].append(match_summary)
                else: stats["recent_matches_away"].append(match_summary)

                # By Result
                r = stats["by_result"][result]; r["m"] += 1; r["matches"].append(match_summary)

            except (ValueError, TypeError) as e:
                 logger.warning(f"Error processing match stats: {e}")
                 continue

        # Reverse form strings (most recent first)
        stats["form_overall"] = stats["form_overall"][::-1]
        stats["form_home"] = stats["form_home"][::-1]
        stats["form_away"] = stats["form_away"][::-1]

        # Get recent matches (most recent first)
        recent_overall = stats["recent_matches_overall"][-5:][::-1]
        recent_home = stats["recent_matches_home"][-3:][::-1]
        recent_away = stats["recent_matches_away"][-3:][::-1]


        # --- Calculate Final Metrics ---
        final_features = {"source": source, "games_analyzed": games_analyzed_count}
        metrics_cache = {}

        # Function to calculate and cache metrics
        def calculate_scope_metrics(scope_key, scope_data):
            if scope_key in metrics_cache: return metrics_cache[scope_key]
            metrics = {}
            m = scope_data["m"]
            if m > 0:
                sc, cn = scope_data["sc"], scope_data["cn"]
                w, d, l = scope_data["w"], scope_data["d"], scope_data["l"]
                metrics["avg_scored"] = sc / m
                metrics["avg_conceded"] = cn / m
                metrics["win_rate"] = w / m
                metrics["draw_rate"] = d / m
                metrics["loss_rate"] = l / m
                metrics["btts_rate"] = scope_data["btts"] / m
                metrics["clean_sheet_rate"] = scope_data["cs"] / m
                metrics["failed_to_score_rate"] = scope_data["fts"] / m
                # Over/Under
                tg = scope_data["tg_list"]
                metrics["over_1_5_rate"] = sum(1 for g in tg if g > 1.5) / m
                metrics["over_2_5_rate"] = sum(1 for g in tg if g > 2.5) / m
                metrics["over_3_5_rate"] = sum(1 for g in tg if g > 3.5) / m
                # Form
                form_str_key = f"form_{scope_key}"
                if form_str_key in stats and stats[form_str_key]:
                     metrics["form_string"] = stats[form_str_key]
                     metrics["form_rating"] = self._calculate_form_rating(metrics["form_string"])
            metrics_cache[scope_key] = metrics
            return metrics


        # Calculate Overall, Home, Away metrics
        if stats["overall"]["m"] > 0: final_features["overall_stats"] = calculate_scope_metrics("overall", stats["overall"])
        if stats["home"]["m"] > 0: final_features["home_stats"] = calculate_scope_metrics("home", stats["home"])
        if stats["away"]["m"] > 0: final_features["away_stats"] = calculate_scope_metrics("away", stats["away"])

        # --- Rolling Stats with Details ---
        rolling_stats = {}
        if recent_overall:
             overall_l5_form = "".join(m["result"] for m in recent_overall)
             rolling_stats["overall_l5"] = {
                 "form_string": overall_l5_form,
                 "form_rating": self._calculate_form_rating(overall_l5_form),
                 "matches": [f"{m['result']} {m['sc']}-{m['cn']} vs {m['opponent']} ({m['venue']})" for m in recent_overall]
             }
        if recent_home:
            home_l3_form = "".join(m["result"] for m in recent_home)
            home_l3_sc = sum(m["sc"] for m in recent_home)
            home_l3_cn = sum(m["cn"] for m in recent_home)
            home_l3_m = len(recent_home)
            rolling_stats["home_l3"] = {
                "form_string": home_l3_form,
                "form_rating": self._calculate_form_rating(home_l3_form),
                "avg_scored": home_l3_sc / home_l3_m,
                "avg_conceded": home_l3_cn / home_l3_m,
                "matches": [f"{m['result']} {m['sc']}-{m['cn']} vs {m['opponent']}" for m in recent_home]
            }
        if recent_away:
            away_l3_form = "".join(m["result"] for m in recent_away)
            away_l3_sc = sum(m["sc"] for m in recent_away)
            away_l3_cn = sum(m["cn"] for m in recent_away)
            away_l3_m = len(recent_away)
            rolling_stats["away_l3"] = {
                "form_string": away_l3_form,
                "form_rating": self._calculate_form_rating(away_l3_form),
                "avg_scored": away_l3_sc / away_l3_m,
                "avg_conceded": away_l3_cn / away_l3_m,
                "matches": [f"{m['result']} {m['sc']}-{m['cn']} vs {m['opponent']}" for m in recent_away]
            }
        if rolling_stats: final_features["rolling_stats"] = rolling_stats


        # --- Performance by Result with Avg Scorelines ---
        perf_by_result = {}
        for result_type, data in stats["by_result"].items():
            m = data["m"]
            if m > 0:
                total_sc = sum(match["sc"] for match in data["matches"])
                total_cn = sum(match["cn"] for match in data["matches"])
                perf_by_result[result_type] = {
                    "matches": m,
                    "avg_scored": total_sc / m,
                    "avg_conceded": total_cn / m,
                    "avg_scoreline": f"{total_sc / m:.2f} - {total_cn / m:.2f}"
                }
        if perf_by_result: final_features["performance_by_result"] = perf_by_result

        # --- Probabilities ---
        prob_stats = {}
        home_m, away_m = stats["home"]["m"], stats["away"]["m"]
        if home_m > 0:
             prob_stats["win_home_prob"] = stats["home"]["w"] / home_m
             prob_stats["win_or_draw_home_prob"] = (stats["home"]["w"] + stats["home"]["d"]) / home_m
        if away_m > 0:
             prob_stats["win_away_prob"] = stats["away"]["w"] / away_m
             prob_stats["win_or_draw_away_prob"] = (stats["away"]["w"] + stats["away"]["d"]) / away_m
        if prob_stats: final_features["probability_stats"] = prob_stats

        return final_features

    def _calculate_form_rating(self, form_string: str) -> float:
        """
        Calculate form rating from a form string like "WDLWWLDW".

        Args:
            form_string: String of W (win), D (draw), L (loss) characters

        Returns:
            Form rating as percentage of max possible points
        """
        if not form_string:
            return 0.0
            
        # Count points (3 for W, 1 for D, 0 for L)
        points = 0
        for char in form_string:
            if char.upper() == 'W':
                points += 3
            elif char.upper() == 'D':
                points += 1
                
        # Calculate percentage of max possible points
        max_points = len(form_string) * 3
        return points / max_points if max_points > 0 else 0.0

    def _extract_core_match_info(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Helper to extract and minimally clean core fixture, league, and team info.
        
        Args:
            game_data: Raw game data dictionary
            
        Returns:
            Dictionary containing cleaned fixture_info, league, and teams data
        """
        try:
            # Extract Fixture Info
            fixture_source = game_data.get("fixture", {})
            if not fixture_source and "match_info" in game_data:
             fixture_source = game_data.get("match_info", {})

            fixture_info = {
                "id": str(game_data.get("fixture_id") or fixture_source.get("id")),
            "referee": fixture_source.get("referee"),
            "timezone": fixture_source.get("timezone"),
            "date": fixture_source.get("date"),
            "timestamp": fixture_source.get("timestamp"),
            "venue": fixture_source.get("venue"),
            "status": fixture_source.get("status"),
        }

            # Extract League Info
            league_source = game_data.get("league", {})
            league_info = {
                "id": str(league_source.get("id")),
            "name": league_source.get("name"),
            "country": league_source.get("country"),
            "logo": league_source.get("logo"),
            "flag": league_source.get("flag"),
            "season": league_source.get("season"),
            "round": league_source.get("round"),
            }

            # Extract Teams Info
            teams_source = game_data.get("teams", {})
            home_team_source = teams_source.get("home", {})
            away_team_source = teams_source.get("away", {})
            
            # If teams are at root level
            if not home_team_source and "home_team" in game_data:
                home_team_source = game_data.get("home_team", {})
            if not away_team_source and "away_team" in game_data:
                away_team_source = game_data.get("away_team", {})

            teams_info = {
                "home": {
                    "id": home_team_source.get("id"),
                    "name": home_team_source.get("name"),
                    "logo": home_team_source.get("logo"),
                },
                "away": {
                    "id": away_team_source.get("id"),
                    "name": away_team_source.get("name"),
                    "logo": away_team_source.get("logo"),
                }
            }
            
            # Basic validation
            if not fixture_info.get("id"):
                logger.error("Missing fixture ID in core info extraction.")
                return None
            
            if not teams_info["home"].get("id") or not teams_info["away"].get("id"):
                logger.error("Missing team IDs in core info extraction.")
                return None
            
            if not fixture_info.get("timestamp"):
                logger.error("Missing timestamp in core info extraction.")
                return None
            
            # Clean any None values
            fixture_info = {k: v for k, v in fixture_info.items() if v is not None}
            league_info = {k: v for k, v in league_info.items() if v is not None}
            teams_info["home"] = {k: v for k, v in teams_info["home"].items() if v is not None}
            teams_info["away"] = {k: v for k, v in teams_info["away"].items() if v is not None}
             
            return {
                "fixture_info": fixture_info,
                "league": league_info,
                "teams": teams_info
            }
        except Exception as e:
            logger.error(f"Error extracting core match info: {e}", exc_info=True)
            return None

    def _add_statarea_id_mappings_basic(self, home_team_data: Dict[str, Any], away_team_data: Dict[str, Any], league_data: Dict[str, Any]):
        """
        Adds StatArea IDs to team and league data dictionaries in place.
        """
        # Map league ID
        league_id_str = league_data.get("id")
        if league_id_str:
            league_statarea_id = "unknown"
            for _, ids in LEAGUE_ID_MAPPING.items():
                if ids.get("mongodb_id") == league_id_str:
                    league_statarea_id = ids.get("statarea_id", "unknown")
                    break
            league_data["statarea_id"] = league_statarea_id

        # Map home team ID
        home_id_str = str(home_team_data.get("id", ""))
        home_name = home_team_data.get("name", "")
        home_statarea_id = "unknown"
        if home_id_str:
            for team_name_map, ids in TEAM_ID_MAPPING.items():
                # Prioritize ID match, fall back to name match
                if ids.get("mongodb_id") == home_id_str or (home_name and team_name_map == home_name):
                    home_statarea_id = ids.get("statarea_id", "unknown")
                    break
            home_team_data["statarea_id"] = home_statarea_id

        # Map away team ID
        away_id_str = str(away_team_data.get("id", ""))
        away_name = away_team_data.get("name", "")
        away_statarea_id = "unknown"
        if away_id_str:
             for team_name_map, ids in TEAM_ID_MAPPING.items():
                 if ids.get("mongodb_id") == away_id_str or (away_name and team_name_map == away_name):
                     away_statarea_id = ids.get("statarea_id", "unknown")
                     break
             away_team_data["statarea_id"] = away_statarea_id

    def _remove_none_unknown_empty(self, data: Any) -> Any:
        """
        Recursively remove keys whose values are None, "unknown", "None",
        or empty dictionaries/lists.
        """
        if isinstance(data, dict):
            cleaned_dict = {}
            for k, v in data.items():
                # Check if this is a non-empty important structure we want to preserve
                is_important_field = k in ("statarea_stats", "match_processor", "recent_matches", 
                                           "raw_data", "engineered_features", "standings")
                
                cleaned_v = self._remove_none_unknown_empty(v)
                
                # Keep important fields even if they're empty
                if is_important_field and isinstance(cleaned_v, (dict, list)) and not cleaned_v:
                    cleaned_dict[k] = {} if isinstance(cleaned_v, dict) else []
                    continue
                
                # Keep the value if it's not None/"unknown"/"None" and not an empty container
                keep_value = False
                if cleaned_v is not None and cleaned_v != "unknown" and cleaned_v != "None":
                    if isinstance(cleaned_v, (dict, list)):
                        if cleaned_v:  # Keep non-empty lists/dicts
                            keep_value = True
                    else:
                        keep_value = True  # Keep all other values

                if keep_value:
                    cleaned_dict[k] = cleaned_v

            return cleaned_dict if cleaned_dict else None
        elif isinstance(data, list):
            cleaned_list = [
                item_cleaned for item in data
                if (item_cleaned := self._remove_none_unknown_empty(item)) is not None
                   and item_cleaned != "unknown" and item_cleaned != "None"
                   and not (isinstance(item_cleaned, (dict, list)) and not item_cleaned)
            ]
            return cleaned_list if cleaned_list else None
        elif data is None or data == "unknown" or data == "None":
            return None
        else:
            return data

    def save_individual_game_file(self, processed_game_data: Dict[str, Any]):
        """
        Save the fully processed and structured game data (unified format)
        to a JSON file in the unified data directory.

        Args:
            processed_game_data: The final, cleaned game data dictionary.
        """
        # Ensure the main output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            logger.info(f"Created unified output directory: {OUTPUT_DIR}")

        # Get components for filename from the processed data
        # Use the new structure: team_A and team_B
        team_a_info = processed_game_data.get("team_a", {}) # Use lowercase keys due to sanitization
        team_b_info = processed_game_data.get("team_b", {}) # Use lowercase keys due to sanitization

        # Sanitize team names for filename
        # Default to Unknown if name is missing after processing
        home_team_name = self._sanitize_filename(team_a_info.get("name", "UnknownHome"))
        away_team_name = self._sanitize_filename(team_b_info.get("name", "UnknownAway"))

        # Get date and fixture ID from top-level fields
        date_part = processed_game_data.get("date", "unknown_date") # Should be YYYY-MM-DD
        fixture_id = processed_game_data.get("fixture_id", "unknown_fixture")

        # Handle potential date issues if it wasn't set correctly
        date_str = None # Initialize date_str before using it
        if not date_part or date_part == "unknown_date":
            fixture_meta = processed_game_data.get("fixture_meta", {})
            date_str = fixture_meta.get("date") # Fallback to full timestamp string
            
        if date_str and isinstance(date_str, str):
             try:
                 date_part = date_str.split('T')[0]
             except Exception:
                 logger.warning(f"Could not parse date '{date_str}' for filename, using fallback.")
                 date_part = self.get_current_date_str() # Final fallback
             if date_part == "unknown_date":
                 date_part = self.get_current_date_str() # Final fallback

        # Create filename: YYYY-MM-DD_HomeTeam_vs_AwayTeam_fixtureID.json
        filename = f"{date_part}_{home_team_name}_vs_{away_team_name}_{fixture_id}.json"
        file_path = os.path.join(OUTPUT_DIR, filename)

        # Save the processed game data to file
        try:
            # Convert any remaining MongoDB specific types (like $numberInt)
            # Note: data should already be normalized, but this is a safe fallback
            data_to_save = self._convert_mongodb_types(processed_game_data)
            with open(file_path, 'w', encoding='utf-8') as f:
                # Use default=str for robustness against non-serializable types
                json.dump(data_to_save, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Saved unified game data to {file_path}")
        except TypeError as e:
            logger.error(f"Error saving JSON data to {file_path}: {e}. Check for non-serializable types.")
            # Attempt to save with default=str as a fallback
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_game_data, f, indent=2, ensure_ascii=False, default=str)
                logger.warning(f"Saved {file_path} using default=str due to TypeError.")
            except Exception as e_fallback:
                 logger.error(f"Fallback save also failed for {file_path}: {e_fallback}")
        except Exception as e:
            logger.error(f"An unexpected error occurred saving {file_path}: {e}")

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize a string to be used as a filename or directory name.
        """
        # Remove characters that are not alphanumeric, underscore, hyphen, or space
        sanitized = re.sub(r'[^\w\s-]', '', name)
        # Replace sequences of hyphens or spaces with a single underscore
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length to avoid issues (e.g., max 100 chars)
        sanitized = sanitized[:100]
        # Ensure it's not empty
        if not sanitized:
            return "invalid_name"
        return sanitized

    def _convert_mongodb_types(self, data):
        """Convert MongoDB specific types (like $numberInt) to standard Python types."""
        if isinstance(data, dict):
            # Check for $numberInt structure BEFORE iterating items
            if len(data) == 1 and "$numberInt" in data:
                 try:
                    return int(data["$numberInt"])
                 except (ValueError, TypeError):
                     logger.warning(f"Could not convert $numberInt: {data['$numberInt']} to int.")
                     return data["$numberInt"] # Return original value if conversion fails

            # Check for $date structure
            if len(data) == 1 and "$date" in data:
                date_val = data["$date"]
                if isinstance(date_val, dict) and "$numberLong" in date_val:
                     try:
                         # Convert milliseconds timestamp to ISO 8601 string
                         ts_ms = int(date_val["$numberLong"])
                         dt_obj = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                         return dt_obj.isoformat()
                     except (ValueError, TypeError):
                         logger.warning(f"Could not convert $date $numberLong: {date_val['$numberLong']}")
                         return date_val # Return original if conversion fails
                elif isinstance(date_val, str): # Handle ISO date strings directly in $date
                     return date_val

            # Recursively process other dictionary items
            return {key: self._convert_mongodb_types(value) for key, value in data.items()}

        elif isinstance(data, list):
            return [self._convert_mongodb_types(item) for item in data]
        # Handle potential datetime objects already converted - return as ISO string
        elif isinstance(data, datetime):
             return data.isoformat()
        return data

    def _parse_form_pattern(self, form_pattern: str) -> Dict[str, Any]:
        """
        Parse a form pattern string like "DDLWWLDWLWWDDLLDWDLLWDLWWWWWWDLW"
        and extract meaningful features.
        
        Args:
            form_pattern: String of W/D/L characters
            
        Returns:
            Dictionary of features extracted from the pattern
        """
        if not form_pattern or not isinstance(form_pattern, str):
            return {}
        
        # Clean the string to contain only W, D, L characters
        clean_pattern = re.sub(r'[^WDL]', '', form_pattern.upper())
        
        if not clean_pattern:
            return {}
        
        # Count overall occurrences
        win_count = clean_pattern.count('W')
        draw_count = clean_pattern.count('D')
        loss_count = clean_pattern.count('L')
        total_matches = len(clean_pattern)
        
        # Calculate streaks
        longest_win_streak = 0
        longest_draw_streak = 0
        longest_loss_streak = 0
        current_win_streak = 0
        current_draw_streak = 0
        current_loss_streak = 0
        
        for char in clean_pattern:
            if char == 'W':
                current_win_streak += 1
                current_draw_streak = 0
                current_loss_streak = 0
                longest_win_streak = max(longest_win_streak, current_win_streak)
            elif char == 'D':
                current_draw_streak += 1
                current_win_streak = 0
                current_loss_streak = 0
                longest_draw_streak = max(longest_draw_streak, current_draw_streak)
            elif char == 'L':
                current_loss_streak += 1
                current_win_streak = 0
                current_draw_streak = 0
                longest_loss_streak = max(longest_loss_streak, current_loss_streak)
        
        # Get recent form (last 5)
        recent_form = clean_pattern[:5] if len(clean_pattern) >= 5 else clean_pattern
        
        # Calculate points earned
        total_points = win_count * 3 + draw_count * 1
        max_possible_points = total_matches * 3
        points_percentage = total_points / max_possible_points if max_possible_points > 0 else 0
        
        return {
            "win_rate": win_count / total_matches if total_matches > 0 else 0,
            "draw_rate": draw_count / total_matches if total_matches > 0 else 0,
            "loss_rate": loss_count / total_matches if total_matches > 0 else 0,
            "longest_win_streak": longest_win_streak,
            "longest_draw_streak": longest_draw_streak,
            "longest_loss_streak": longest_loss_streak,
            "recent_form": recent_form,
            "form_points_percentage": points_percentage,
            "total_matches_analyzed": total_matches
        }
        
# Main execution block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Ensure OUTPUT_DIR exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")

    try:
        extractor = DailyGameExtractor(use_mongo=True)

        # Extract data for today
        today_date_str = extractor.get_current_date_str()
        logger.info(f"--- Running unified extraction for today ({today_date_str}) ---")
        
        # extract_games_for_date now handles fetching, processing (new structure), and saving individual files
        daily_result_summary = extractor.extract_games_for_date(today_date_str)
        
        if daily_result_summary.get("error"):
            logger.error(f"Extraction failed: {daily_result_summary['error']}")
        else:
            processed_count = daily_result_summary.get('total_games_processed', 0)
            found_count = daily_result_summary.get('total_games_found_for_date', 0)
            
            if processed_count > 0:
                logger.info(f"--- Unified Extraction Complete for {today_date_str} ---")
                logger.info(f"Attempted: {found_count}, Successfully processed and saved: {processed_count}")
                # Note: The save_summary_file and _save_standings_files logic might need review
                # if they depended on the old data structure or are no longer needed.
                # For now, commenting them out as the primary goal was the individual unified file.
                # extractor.save_summary_file(daily_data) # Review this function's input/logic
                # extractor._save_standings_files(today_date_str) # Review this function's logic
            elif found_count > 0:
                logger.warning(f"--- Unified Extraction attempted for {today_date_str}, found {found_count} games but processed 0. Check logs for errors. ---")
            else:
                logger.info(f"--- Unified Extraction completed for {today_date_str}, but no games were found for this date. ---")

    except ConnectionError as ce:
        logger.critical(f"❌ MongoDB Connection Error during initialization: {ce}", exc_info=True)
        sys.exit(1) # Exit if DB connection fails
    except Exception as e:
        logger.critical(f"❌ An unexpected error occurred during the main execution: {e}", exc_info=True)
        sys.exit(1)
