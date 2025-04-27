import os
import logging
from datetime import date, datetime, timezone
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import re
import sys
from pathlib import Path

# Add project root to system path
project_root = str(Path(__file__).resolve().parent.parent) # Assuming project root is 2 levels up
sys.path.insert(0, project_root)

import time
import requests
from dotenv import load_dotenv

# Import MongoDB manager from existing code
try:
    from get_data.api_football.db_mongo import MongoDBManager
    from get_data.api_football.db_ids.team_id_mappings import TEAM_ID_MAPPING
    from get_data.api_football.db_ids.league_id_mappings import LEAGUE_ID_MAPPING
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
    'm': 'matches_played',
    'w': 'wins',
    'd': 'draws',
    'l': 'losses',
    'sc': 'goals_scored',
    'cn': 'goals_conceded',
    'cs': 'clean_sheets',
    'fts': 'failed_to_score',
    'btts': 'both_teams_scored',
    'tg_list': 'total_goals_list',
    'avg_scored': 'average_goals_scored',
    'avg_conceded': 'average_goals_conceded',
    'goalsdiff': 'goals_difference',
    'form': 'form_string',
    'rank': 'league_rank',
    'points': 'league_points',
    '1 x 2': 'bet_1x2_probabilities',
    'over/under 1.5 for all goals in matches': 'bet_o1.5_total_probabilities',
    'over/under 2.5 for all goals in matches': 'bet_o2.5_total_probabilities',
    'over/under 3.5 for all goals in matches': 'bet_o3.5_total_probabilities',
    'goal bands': 'bet_goal_bands_probabilities',
    'team to score': 'bet_both_teams_to_score_probabilities',
    'odd/even goals in matches': 'bet_odd_even_probabilities',
    'first goal in matches': 'bet_first_goal_scorer_probabilities',
    'win after 45 minutes': 'bet_halftime_result_probabilities',
    'win after 90 minutes': 'bet_fulltime_result_probabilities',
    'goal difference in match': 'bet_goal_difference_probabilities',
    'half with most goals': 'bet_half_most_goals_probabilities',
    'temp_min': 'temperature_min',
    'temp_max': 'temperature_max',
    'feels_like': 'temperature_feels_like',
    'sea_level': 'pressure_sea_level',
    'grnd_level': 'pressure_ground_level',
    'deg': 'direction_degrees',
    'speed': 'speed_mps',
    'all': 'coverage_percentage',
    'pop': 'probability_of_precipitation'
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
        
        # Convert to lowercase (cautiously - only if not a proper noun or acronym)
        # Check if it's likely a proper noun (starts uppercase, rest lower) or all uppercase (acronym)
        is_proper_noun = (sanitized and sanitized[0].isupper() and len(sanitized) > 1 and sanitized[1:].islower())
        is_acronym = sanitized.isupper() and len(sanitized) > 1

        if not (is_proper_noun or is_acronym):
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
        Process match data and structure it, including a flattened feature set for models.
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
        home_team_base = core_info["teams"]["home"] # Basic info: id, name, logo
        away_team_base = core_info["teams"]["away"] # Basic info: id, name, logo
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
        # Add statarea IDs to the base info dicts
        home_team_base["statarea_id"] = home_statarea_id
        away_team_base["statarea_id"] = away_statarea_id

        # --- 3. Fetch Weather Data for Fixture Meta ---
        weather_forecast = None
        if city:
            logger.info(f"Attempting to fetch weather forecast for fixture {fixture_id_str} in {city}, {country}")
            weather_forecast = self._fetch_weather_forecast_by_city(city, country, current_game_timestamp)
            # Normalize weather data values
            if weather_forecast and "error" not in weather_forecast:
                weather_forecast = self._normalize_values_recursively(weather_forecast)
                logger.info(f"Successfully fetched weather forecast for fixture {fixture_id_str} in {city}")
                # Add a summary for easier consumption
                main_weather = weather_forecast.get("weather", [{}])[0].get("main", "Unknown")
                description = weather_forecast.get("weather", [{}])[0].get("description", "Unknown conditions")
                temp = weather_forecast.get("main", {}).get("temperature", "Unknown") # Use normalized key 'temperature'
                weather_forecast["summary"] = f"{main_weather}: {description}, Temp: {temp}°C"
            elif weather_forecast: # Error case
                error_msg = weather_forecast.get("error", "Failed to fetch")
                logger.warning(f"Could not retrieve valid weather forecast for fixture {fixture_id_str} in {city}. Reason: {error_msg}")
                weather_forecast = {"error": error_msg} # Keep error info
            else: # None returned
                logger.warning(f"Could not retrieve weather forecast for fixture {fixture_id_str} in {city}. Reason: None returned")
                weather_forecast = {"error": "None returned"}

        # Apply sanitization to weather keys
        if weather_forecast and isinstance(weather_forecast, dict) and "error" not in weather_forecast:
            weather_forecast = self._sanitize_keys_recursively(weather_forecast)
            # Re-add summary after key sanitization if necessary
            main_weather = weather_forecast.get("weather", [{}])[0].get("main", "Unknown")
            description = weather_forecast.get("weather", [{}])[0].get("description", "Unknown conditions")
            temp_key = self._sanitize_text('temperature') # Find the sanitized key
            temp = weather_forecast.get("main", {}).get(temp_key, "Unknown")
            weather_forecast["summary"] = f"{main_weather}: {description}, Temp: {temp}°C"


        else:
            logger.warning(f"Cannot fetch weather: Missing city information for fixture {fixture_id_str}")
            weather_forecast = {"error": "Missing city info"}


        # --- 4. Fetch Additional Data from MongoDB ---
        match_processor_data = self.mongo_db.get_match_processor_data(fixture_id_str) or {}
        
        # Extract standings from match_processor_data if available
        standings_list = []
        home_standings_info = None
        away_standings_info = None
        if match_processor_data:
             standings_snapshot_raw = match_processor_data.get("standings_snapshot", {})
             # Check if standings_snapshot_raw itself is the list (handle older formats)
             if isinstance(standings_snapshot_raw, list) and len(standings_snapshot_raw) > 0:
                 # Assuming the first element is the relevant league's standings list
                 potential_standings = standings_snapshot_raw[0]
             elif isinstance(standings_snapshot_raw, dict):
                 # Assume standard structure { "league": {...}, "standings": [[...]] }
                 standings_list_raw = standings_snapshot_raw.get("standings", [])
                 # Often standings are wrapped in another list, handle that
                 if standings_list_raw and isinstance(standings_list_raw, list) and len(standings_list_raw) > 0:
                    potential_standings = standings_list_raw[0] if isinstance(standings_list_raw[0], list) else standings_list_raw
                 else:
                     potential_standings = []
             else:
                 potential_standings = []


             # Ensure it's actually a list of standings dicts
             if isinstance(potential_standings, list):
                standings_list = potential_standings
             else:
                logger.warning(f"Unexpected format for standings data in fixture {fixture_id_str}. Expected list.")


        # Find specific team standings
        if standings_list and isinstance(standings_list, list):
            for team_standing in standings_list:
                if isinstance(team_standing, dict):
                    team_info = team_standing.get("team", {})
                    standing_team_id = team_info.get("id")
                    # Compare IDs robustly (handle potential type differences)
                    try:
                         if standing_team_id is not None and int(standing_team_id) == int(home_team_id):
                            home_standings_info = team_standing
                            logger.info(f"Found standings for home team {home_team_id}")
                         elif standing_team_id is not None and int(standing_team_id) == int(away_team_id):
                            away_standings_info = team_standing
                            logger.info(f"Found standings for away team {away_team_id}")
                    except (ValueError, TypeError):
                         logger.warning(f"Could not compare team ID for standings: {standing_team_id} vs {home_team_id}/{away_team_id}")


        # Extract team-specific processor data
        home_processor_data = match_processor_data.get("home_team_stats", {}) if match_processor_data else {}
        away_processor_data = match_processor_data.get("away_team_stats", {}) if match_processor_data else {}
        
        # Include predictions for individual teams if available (attach to processor data)
        if "predictions" in match_processor_data:
            pred_data = match_processor_data.get("predictions", {})
            if "teams" in pred_data:
                teams_pred = pred_data.get("teams", {})
                if "home" in teams_pred:
                    home_processor_data["predictions_snapshot"] = teams_pred.get("home", {}) # Use distinct key
                if "away" in teams_pred:
                    away_processor_data["predictions_snapshot"] = teams_pred.get("away", {}) # Use distinct key


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


        # --- 5. Calculate Engineered Features (Pass Normalized and Sanitized Data) ---
        # Sanitize keys FIRST, then normalize values for consistency
        logger.debug(f"Raw home standings before sanitization: {home_standings_info}")
        home_standings_sanitized = self._normalize_values_recursively(self._sanitize_keys_recursively(home_standings_info)) if home_standings_info else None
        logger.debug(f"Sanitized home standings before passing to features: {home_standings_sanitized}")

        logger.debug(f"Raw away standings before sanitization: {away_standings_info}")
        away_standings_sanitized = self._normalize_values_recursively(self._sanitize_keys_recursively(away_standings_info)) if away_standings_info else None
        logger.debug(f"Sanitized away standings before passing to features: {away_standings_sanitized}")


        home_eng_features = self._generate_engineered_features(
            team_basic_info=home_team_base,
            league_info=league_info,
            weather_info=weather_forecast,
            processor_data=self._normalize_values_recursively(self._sanitize_keys_recursively(home_processor_data)),
            statarea_data=self._normalize_values_recursively(self._sanitize_keys_recursively(home_statarea_data)),
            previous_matches=[self._normalize_values_recursively(self._sanitize_keys_recursively(m)) for m in home_previous_matches],
            standings_info=home_standings_sanitized,
            current_game_timestamp=current_game_timestamp
        )

        away_eng_features = self._generate_engineered_features(
            team_basic_info=away_team_base,
            league_info=league_info,
            weather_info=weather_forecast,
            processor_data=self._normalize_values_recursively(self._sanitize_keys_recursively(away_processor_data)),
            statarea_data=self._normalize_values_recursively(self._sanitize_keys_recursively(away_statarea_data)),
            previous_matches=[self._normalize_values_recursively(self._sanitize_keys_recursively(m)) for m in away_previous_matches],
            standings_info=away_standings_sanitized,
            current_game_timestamp=current_game_timestamp
        )
        
        # --- 5b. Flatten Engineered Features for Model Input ---
        logger.info(f"Flattening engineered features for fixture {fixture_id_str}")
        model_input_features = self._flatten_engineered_features(home_eng_features or {}, away_eng_features or {})

        # --- 6. Construct the New Unified JSON Structure ---
        match_date = fixture_info.get("date", "").split("T")[0] if fixture_info.get("date") else None
        
        # --- ADDED: Extract original goals data ---
        original_goals_data = match_data.get('goals') # Get goals dict from the input data
        # --- End ADDED section ---
        
        fixture_meta = {
            "id": fixture_id_str,
            "date_utc": fixture_info.get("date"),
            "timestamp_utc": fixture_info.get("timestamp"),
            "referee": fixture_info.get("referee"),
            "venue": venue_data,
            "status": fixture_info.get("status"),
            "weather_forecast": weather_forecast
        }

        engineered_features_nested = { # Rename for clarity
            "home": home_eng_features or {},
            "away": away_eng_features or {}
        }

        raw_data = {
             "home": {
                "basic_info": home_team_base,
                "match_processor_snapshot": self._convert_mongodb_types(home_processor_data),
                "statarea_snapshot": self._convert_mongodb_types(home_statarea_data or {}),
                "recent_matches_raw": [
                    {"source": "matches", "data": self._convert_mongodb_types(match)}
                    for match in home_previous_matches
                ],
                 "standings_snapshot_raw": self._convert_mongodb_types(home_standings_info or {})
            },
            "away": {
                 "basic_info": away_team_base,
                "match_processor_snapshot": self._convert_mongodb_types(away_processor_data),
                "statarea_snapshot": self._convert_mongodb_types(away_statarea_data or {}),
                "recent_matches_raw": [
                    {"source": "matches", "data": self._convert_mongodb_types(match)}
                    for match in away_previous_matches
                ],
                "standings_snapshot_raw": self._convert_mongodb_types(away_standings_info or {})
            }
        }
        
        unified_data = {
            "fixture_id": fixture_id_str,
            "match_date": match_date,
            "league": league_info,
            "engineered_features": engineered_features_nested, # Keep nested version
            "model_input_features": model_input_features, # Add flattened version
            "fixture_meta": fixture_meta,
            "raw_data": raw_data,
            "goals": original_goals_data # <-- ADDED: Include original goals data
        }

        # --- 7. Final Normalize/Sanitize/Clean the *entire* structure ---
        # Keys should already be sanitized mostly, apply normalize first then clean
        normalized_data = self._normalize_values_recursively(unified_data)
        sanitized_normalized_data = self._sanitize_keys_recursively(normalized_data)
        # Use the less aggressive cleaner
        cleaned_data = self._remove_none_values(sanitized_normalized_data)

        if not cleaned_data:
            logger.error(f"❌ Data structure became empty after cleaning for fixture {fixture_id_str}. Skipping save.")
            return None
            
        logger.info(f"✅ Successfully processed and structured unified data for fixture {fixture_id_str}.")
        return cleaned_data
        
    def _flatten_engineered_features(self, home_features: Dict[str, Any], away_features: Dict[str, Any], none_placeholder: Any = -999.0, div_zero_placeholder: Any = -998.0) -> Dict[str, Any]:
        """
        Flattens the home and away engineered features into a single dictionary
        suitable for model input, calculating differences and ratios by matching keys.

        Args:
            home_features: Dictionary of engineered features for the home team.
            away_features: Dictionary of engineered features for the away team.
            none_placeholder: Value to use for missing numerical data.
            div_zero_placeholder: Value to use for division by zero in ratios.

        Returns:
            A flat dictionary with prefixed features ('eng_h_', 'eng_a_', 'eng_diff_', 'eng_ratio_').
        """
        flat_features = {}
        home_keys_processed = set() # Track home keys added
        away_keys_processed = set() # Track away keys added

        # Helper to safely add features and handle None/types
        def add_feature(prefix: str, key: str, value: Any, processed_set: set):
            full_key = f"{prefix}{key}"
            processed_set.add(key) # Add the original key to the processed set
            if isinstance(value, (int, float)):
                flat_features[full_key] = float(value) # Standardize to float
            elif value is None:
                flat_features[full_key] = none_placeholder # Use placeholder for None
            # Basic handling for known simple categoricals (add more as needed)
            elif key in ["weather_condition_main", "match_history_source", "match_history_venue_context"]:
                 # Ensure value is converted to string, handle potential non-string types gracefully
                try:
                    flat_features[full_key] = str(value)
                except Exception:
                     logger.warning(f"Could not convert value to string for key {full_key}. Assigning placeholder.")
                     flat_features[full_key] = "encoding_error" # Or another suitable placeholder string
            # Skip complex types (dict, list) or unknown string types by default

        # Process home features
        for key, value in home_features.items():
            add_feature("eng_h_", key, value, home_keys_processed)

        # Process away features
        for key, value in away_features.items():
            add_feature("eng_a_", key, value, away_keys_processed)

        # --- Calculate difference and ratio features by finding matching keys ---
        # Iterate through the original keys found in home_features
        for key in home_keys_processed:
            # Check if the same original key exists for the away team
            if key in away_keys_processed:
                h_key = f"eng_h_{key}"
                a_key = f"eng_a_{key}"

                # Check if both prefixed keys ended up in flat_features (they should have if processed)
                if h_key in flat_features and a_key in flat_features:
                    h_val = flat_features[h_key]
                    a_val = flat_features[a_key]

                    # Only proceed if both are numerical (not placeholders or strings)
                    is_h_numeric = isinstance(h_val, (int, float)) and h_val != none_placeholder
                    is_a_numeric = isinstance(a_val, (int, float)) and a_val != none_placeholder

                    if is_h_numeric and is_a_numeric:
                        # Difference
                        flat_features[f"eng_diff_{key}"] = h_val - a_val

                        # Ratio
                        if a_val != 0.0:
                            flat_features[f"eng_ratio_{key}"] = h_val / a_val
                        else:
                            flat_features[f"eng_ratio_{key}"] = div_zero_placeholder # Division by zero
                    else:
                        # If one or both are not numeric, set diff/ratio to placeholder
                        flat_features[f"eng_diff_{key}"] = none_placeholder
                        flat_features[f"eng_ratio_{key}"] = none_placeholder
                else:
                     # This case should ideally not happen if keys were processed correctly
                     logger.debug(f"Skipping diff/ratio for base key '{key}' - prefixed keys missing.")

            # Handle keys present only in home features (optional: add with placeholder diff/ratio)
            # else:
            #     h_key = f"eng_h_{key}"
            #     if h_key in flat_features:
            #         h_val = flat_features[h_key]
            #         is_h_numeric = isinstance(h_val, (int, float)) and h_val != none_placeholder
            #         if is_h_numeric:
            #              flat_features[f"eng_diff_{key}"] = none_placeholder # No away value to diff
            #              flat_features[f"eng_ratio_{key}"] = none_placeholder

        # Handle keys present only in away features (optional: add with placeholder diff/ratio)
        # for key in away_keys_processed:
        #      if key not in home_keys_processed:
        #         a_key = f"eng_a_{key}"
        #         if a_key in flat_features:
        #             a_val = flat_features[a_key]
        #             is_a_numeric = isinstance(a_val, (int, float)) and a_val != none_placeholder
        #             if is_a_numeric:
        #                 flat_features[f"eng_diff_{key}"] = none_placeholder # No home value to diff
        #                 flat_features[f"eng_ratio_{key}"] = none_placeholder


        return flat_features

    def _generate_engineered_features(self, team_basic_info: Dict[str, Any],
                                      league_info: Dict[str, Any],
                                      weather_info: Optional[Dict[str, Any]],
                                      processor_data: Dict[str, Any],
                                      statarea_data: Optional[Dict[str, Any]],
                                      previous_matches: List[Dict[str, Any]],
                                      standings_info: Optional[Dict[str, Any]],
                                      current_game_timestamp: Optional[int] = None,
                                      form_pattern: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive engineered features for a team, including basic ID info, league, weather,
        prioritizing StatArea history and including relevant standings information directly.
        Includes relevant info from the processor_data snapshot. Adds Elo rating retrieval.

        Args:
            team_basic_info: Basic team info (id, name, logo, statarea_id).
            league_info: Basic league info (id, name, country).
            weather_info: Sanitized and normalized weather forecast data, or None/error dict.
            processor_data: Data from match_processor (already sanitized and normalized).
            statarea_data: Data from StatArea (already sanitized and normalized), can be None.
            previous_matches: Previous matches from 'matches' collection (fallback history).
            standings_info: Dictionary containing standings data for this specific team (already normalized/sanitized).
            # ADD current_game_timestamp argument documentation
            current_game_timestamp: Timestamp of the game for fetching point-in-time Elo.
            form_pattern: Optional explicit form string pattern (for specific analysis).

        Returns:
            Dictionary of engineered features.
        """
        team_id = team_basic_info.get("id")
        features = {}
        history_analysis = {}

        # --- Add Basic Team Info ---
        features["team_id"] = team_id
        features["team_name"] = team_basic_info.get("name")
        # features["team_logo"] = team_basic_info.get("logo") # Logo likely not needed for model
        features["team_statarea_id"] = team_basic_info.get("statarea_id")

        # --- Fetch PRE-CALCULATED Elo Rating ---
        # Assumes an external process calculates and stores Elo ratings historically.
        # This function FETCHE S the relevant rating just before the match.
        elo_rating = None
        if team_id and current_game_timestamp and self.mongo_db:
             # Verify the MongoDBManager method exists
             if not hasattr(self.mongo_db, 'get_elo_rating'):
                 logger.warning(f"MongoDBManager does not have a 'get_elo_rating' method. Cannot fetch Elo for team {team_id}.")
                 features["elo_rating"] = None
             else:
                 try:
                     # Fetch the pre-calculated Elo rating valid *before* this game's timestamp
                     fetched_elo = self.mongo_db.get_elo_rating(team_id, current_game_timestamp)

                     if fetched_elo is not None:
                         # Attempt conversion to float, handle potential errors
                         try:
                             elo_rating = float(fetched_elo)
                             logger.info(f"Fetched Elo rating for team {team_id} (timestamp {current_game_timestamp}): {elo_rating}")
                             features["elo_rating"] = elo_rating
                         except (ValueError, TypeError):
                             logger.error(f"Fetched Elo rating for team {team_id} is not a valid number: {fetched_elo}. Setting Elo to None.")
                             features["elo_rating"] = None
                     else:
                         logger.warning(f"No pre-calculated Elo rating found for team {team_id} before timestamp {current_game_timestamp}.")
                         features["elo_rating"] = None # Explicitly set None if not found by the DB method

                 except Exception as e:
                     logger.error(f"Error fetching Elo for team {team_id} from DB: {e}", exc_info=True)
                     features["elo_rating"] = None
        else:
             if not team_id: logger.debug("Cannot fetch Elo: Missing team_id.")
             if not current_game_timestamp: logger.debug("Cannot fetch Elo: Missing current_game_timestamp.")
             if not self.mongo_db: logger.debug("Cannot fetch Elo: Missing MongoDB connection.")
             features["elo_rating"] = None # Set None if prerequisites missing

        # --- Add Base League & Round Info ---
        features["base_league_id"] = league_info.get("id") # Renamed to avoid clash
        features["base_league_name"] = league_info.get("name") # Renamed
        features["base_league_country"] = league_info.get("country") # Renamed
        features["base_league_season"] = league_info.get("season") # Renamed
        features["base_league_round"] = league_info.get("round") # Renamed

        # --- Add League/Team Info from Processor Snapshot ---
        if processor_data:
            proc_league_info = processor_data.get(self._sanitize_text("league"), {})
            proc_team_info = processor_data.get(self._sanitize_text("team"), {})
            if proc_league_info:
                 features["proc_league_id"] = proc_league_info.get("id")
                 features["proc_league_name"] = proc_league_info.get("name")
                 features["proc_league_country"] = proc_league_info.get("country")
                 features["proc_league_season"] = proc_league_info.get("season")
                 # features["proc_league_logo"] = proc_league_info.get("logo") # Likely not needed
                 # features["proc_league_flag"] = proc_league_info.get("flag") # Likely not needed
            if proc_team_info:
                 features["proc_team_id"] = proc_team_info.get("id") # Should match team_id
                 features["proc_team_name"] = proc_team_info.get("name") # Should match team_name
                 # features["proc_team_logo"] = proc_team_info.get("logo") # Likely not needed

        # --- Add Weather Features ---
        if isinstance(weather_info, dict) and "error" not in weather_info:
            # Ensure weather keys are sanitized using TERM_STANDARDIZATION map
            main_weather = weather_info.get("main", {})
            wind_weather = weather_info.get("wind", {})
            clouds_weather = weather_info.get("clouds", {})
            precip_prob = weather_info.get(self._sanitize_text("pop")) # Already sanitized key
            
            # Extract using sanitized keys where possible
            features["weather_temp_celsius"] = main_weather.get(self._sanitize_text("temperature"))
            features["weather_temp_feels_like_celsius"] = main_weather.get(self._sanitize_text("temperature_feels_like"))
            features["weather_pressure_hpa"] = main_weather.get(self._sanitize_text("air_pressure")) # Or 'pressure' if standardized
            features["weather_humidity_pct"] = main_weather.get("humidity") # Key 'humidity' might not be in map
            features["weather_wind_speed_mps"] = wind_weather.get(self._sanitize_text("speed_mps")) # Use standardized key
            features["weather_wind_direction_deg"] = wind_weather.get(self._sanitize_text("direction_degrees")) # Use standardized key
            features["weather_cloud_coverage_pct"] = clouds_weather.get(self._sanitize_text("coverage_percentage")) # Use standardized key
            features["weather_precipitation_probability"] = precip_prob # Already sanitized key from 'pop'

            # Add a general description if useful
            weather_desc_list = weather_info.get("weather", [])
            if weather_desc_list and isinstance(weather_desc_list, list):
                 features["weather_description"] = weather_desc_list[0].get("description")
                 features["weather_condition_main"] = weather_desc_list[0].get("main") # e.g., "Clouds", "Rain"

        else:
            logger.warning(f"Weather info missing or contains error for fixture {features.get('fixture_id', 'unknown')}. Skipping weather features.")
            # Optionally add placeholder values or skip
            features["weather_temp_celsius"] = None # Indicate missing data explicitly

        # --- Prioritize StatArea Match History ---
        statarea_history = []
        if isinstance(statarea_data, dict):
            statarea_history = statarea_data.get("match_history", [])
        team_name_statarea_raw = statarea_data.get("team") if isinstance(statarea_data, dict) else None

        if statarea_history and isinstance(statarea_history, list) and len(statarea_history) > 0:
            logger.info(f"Analyzing StatArea match history ({len(statarea_history)} games) for team {team_id}")
            venue_context = statarea_data.get("game_type")
            history_analysis = self._analyze_match_history(statarea_history, team_id, source="statarea", venue_context=venue_context)
        else:
            if previous_matches and isinstance(previous_matches, list) and len(previous_matches) > 0:
                logger.info(f"Analyzing 'matches' collection history ({len(previous_matches)} games) for team {team_id} (StatArea history missing)")
                matches_to_analyze = [m.get("data", m) for m in previous_matches]
                history_analysis = self._analyze_match_history(matches_to_analyze, team_id, source="matches")
            else:
                logger.warning(f"No match history found from StatArea or 'matches' collection for team {team_id}. Limited features.")

        features.update(history_analysis)

        # --- Add Standings Information Directly to Features ---
        if isinstance(standings_info, dict):
             logger.info(f"Adding standings info for team {team_id}: {list(standings_info.keys())}")
             # Use sanitized keys based on TERM_STANDARDIZATION
             features["standings_league_rank"] = standings_info.get(self._sanitize_text("league_rank"))
             features["standings_league_points"] = standings_info.get(self._sanitize_text("league_points"))
             # Double-check the source key for goals difference ('goalsdiff' or 'goals_difference')
             features["standings_goals_difference"] = standings_info.get(self._sanitize_text("goals_difference"))
             features["standings_form_string"] = standings_info.get(self._sanitize_text("form_string"))
             features["standings_group_name"] = standings_info.get(self._sanitize_text("group"))
             all_stats = standings_info.get("all", {})
             if isinstance(all_stats, dict):
                 # Add mappings to TERM_STANDARDIZATION if these keys aren't standardized yet
                 features["standings_matches_played"] = all_stats.get(self._sanitize_text("played")) # Need mapping or use 'matches_played'
                 features["standings_wins"] = all_stats.get(self._sanitize_text("wins"))
                 features["standings_draws"] = all_stats.get(self._sanitize_text("draws"))
                 features["standings_losses"] = all_stats.get(self._sanitize_text("losses"))
                 features["standings_goals_scored"] = all_stats.get("goals", {}).get(self._sanitize_text("for")) # Need mapping or use 'goals_scored'
                 features["standings_goals_conceded"] = all_stats.get("goals", {}).get(self._sanitize_text("against")) # Need mapping or use 'goals_conceded'
                 # Add win percentage directly from standings if available
                 m_played = features.get("standings_matches_played")
                 wins = features.get("standings_wins")
                 if m_played is not None and wins is not None and m_played > 0:
                     try:
                        features["standings_win_percentage"] = round(float(wins) / float(m_played), 3)
                     except (ValueError, TypeError, ZeroDivisionError):
                         logger.debug(f"Could not calculate standings win percentage for team {team_id}")
        else:
             logger.warning(f"No valid standings_info dict found for team {team_id} when generating features.")
             # Add placeholders if needed
             features["standings_league_rank"] = None


        # --- Add Supplemental Features from Processor Data ---
        processor_supplemental = {}
        if processor_data:
            # Goal Timings as %
            # Use sanitized keys for access
            goal_data = processor_data.get(self._sanitize_text("goals"), {})
            proc_total_scored = goal_data.get(self._sanitize_text("for"), {}).get("total", {}).get("total", 0)
            proc_total_conceded = goal_data.get(self._sanitize_text("against"), {}).get("total", {}).get("total", 0)
            timing_features = {}

            # Example using explicit keys:
            for_per_minute = goal_data.get(self._sanitize_text("for"), {}).get("minute", {})
            if for_per_minute and proc_total_scored and proc_total_scored > 0:
                for bracket, value in for_per_minute.items():
                    # bracket might be '0-15', '16-30', etc.
                    count = value.get("total")
                    if count is not None:
                        try:
                            count_num = int(count) # Ensure it's a number
                            # Use explicit naming convention
                            sanitized_bracket_key = self._sanitize_filename(bracket) # Sanitize '0-15' etc. for safety
                            timing_features[f"proc_goals_scored_min_{sanitized_bracket_key}_count"] = count_num
                            timing_features[f"proc_goals_scored_min_{sanitized_bracket_key}_pct"] = round(count_num / proc_total_scored, 4)
                        except (ValueError, TypeError):
                             logger.warning(f"Could not convert goal timing count: {count} for bracket {bracket}")

            # Similar logic for conceded goals... using 'proc_goals_conceded_min_{bracket}_...'

            if timing_features:
                 processor_supplemental["goal_timing_stats"] = timing_features

            # Card Stats
            # ... (existing logic, ensure keys like 'proc_yellow_cards_avg' are used) ...
            # Example:
            card_data = processor_data.get(self._sanitize_text("cards"), {})
            fixtures_played = processor_data.get(self._sanitize_text("fixtures"), {}).get(self._sanitize_text("played"), {}).get("total", 0)
            card_stats = {}
            if card_data and fixtures_played > 0:
                yellow_cards = 0
                # Iterate through yellow card time brackets
                for bracket, bracket_data in card_data.get("yellow", {}).items():
                    # Check if bracket_data is a dict and get the total
                    if isinstance(bracket_data, dict):
                        total = bracket_data.get("total")
                        # Ensure total is not None before adding
                        if total is not None:
                            try:
                                yellow_cards += int(total)
                            except (ValueError, TypeError):
                                 logger.warning(f"Could not convert yellow card total to int: {total} for bracket {bracket}")

                red_cards = 0
                 # Iterate through red card time brackets
                for bracket, bracket_data in card_data.get("red", {}).items():
                     # Check if bracket_data is a dict and get the total
                    if isinstance(bracket_data, dict):
                        total = bracket_data.get("total")
                        # Ensure total is not None before adding
                        if total is not None:
                            try:
                                red_cards += int(total)
                            except (ValueError, TypeError):
                                 logger.warning(f"Could not convert red card total to int: {total} for bracket {bracket}")


                # Calculate averages only if cards were found
                if yellow_cards is not None: # yellow_cards will be an int >= 0
                     card_stats["yellow_avg"] = round(yellow_cards / fixtures_played, 3)
                if red_cards is not None: # red_cards will be an int >= 0
                     card_stats["red_avg"] = round(red_cards / fixtures_played, 3)

            if card_stats:
                 processor_supplemental["card_stats"] = card_stats


            # Penalty Stats
            penalty_data = processor_data.get(self._sanitize_text("penalty"), {})
            if penalty_data:
                 proc_penalty_stats = {
                     "proc_penalty_scored_pct": penalty_data.get(self._sanitize_text("scored"), {}).get("percentage"),
                     "proc_penalty_missed_pct": penalty_data.get(self._sanitize_text("missed"), {}).get("percentage"),
                     "proc_penalty_total_taken": penalty_data.get("total")
                 }
                 processor_supplemental["penalty_stats"] = {k: v for k, v in proc_penalty_stats.items() if v is not None}


        if processor_supplemental:
             # Add directly to features dict, maybe flatten or keep nested
             features["processor_supplemental_stats"] = processor_supplemental # Keeping nested for now


        # --- Add Supplemental Features from StatArea (General / Betting) ---
        statarea_supplemental = {}
        if isinstance(statarea_data, dict):
             gen_stats = statarea_data.get(self._sanitize_text("general_statistics"), {})
             bet_stats = statarea_data.get(self._sanitize_text("team_bet_statistics"), {}) # Use sanitized key
             team_name_statarea_raw = statarea_data.get("team") # Get team name for dynamic keys

             # General Stats
             # ... (existing logic, use explicit keys like 'statarea_gen_chance_to_score_next') ...
             keys_to_check_gen = [
                 "chance_to_score_goal_next_match", "chance_to_conceded_goal_next_match",
                 "time_without_scored_goal", "time_without_conceded_goal"
             ]
             for key in keys_to_check_gen:
                 s_key = self._sanitize_text(key)
                 if s_key in gen_stats:
                     value = gen_stats[s_key]
                     # ... (potential time parsing logic) ...
                     statarea_supplemental[f"statarea_gen_{s_key}"] = value # Use sanitized key + prefix

             # Betting Stats
             # ... (existing logic to map statarea betting keys to explicit feature names) ...
             # Example using betting_keys_map:
             # Ensure the target_key in betting_keys_map uses prefixes like 'statarea_bet_...'
             # The logic needs careful review based on TERM_STANDARDIZATION for betting keys.
             # Example target key could be 'statarea_bet_o2.5_total_over_prob'

             if statarea_supplemental: # Only add if stats were found
                 features["statarea_supplemental_stats"] = statarea_supplemental # Keeping nested for now


        # --- Form pattern parsing (if provided explicitly) ---
        if form_pattern:
            pattern_features = self._parse_form_pattern(form_pattern)
            if pattern_features:
                 # Add with prefix, e.g., 'explicit_form_win_rate'
                 for key, value in pattern_features.items():
                     features[f"explicit_form_{key}"] = value

        # --- Final Cleanup of Engineered Features ---
        features = self._normalize_values_recursively(features)
        features = self._remove_none_values(features)

        return features if features else {}

    def _analyze_match_history(self, match_history: List[Dict[str, Any]], team_id: int, source: str = "unknown", venue_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes a list of past matches (up to 15) to extract detailed statistics using explicit keys,
        including overall, home/away splits, rolling stats, performance by result,
        opponent names, and scores. Also calculates home/away differentials.

        Args:
            match_history: List of match dictionaries (StatArea or Matches format).
            team_id: The ID of the team to analyze for.
            source: Identifier for the source ('statarea' or 'matches').
            venue_context: For 'statarea', indicates if the history is 'host' or 'guest'.

        Returns:
            A dictionary containing structured engineered statistics with explicit keys.
        """
        # --- Initialization using explicit keys ---
        stats = {
            "overall": {"matches_played": 0, "wins": 0, "draws": 0, "losses": 0, "goals_scored": 0, "goals_conceded": 0, "both_teams_scored": 0, "clean_sheets": 0, "failed_to_score": 0, "total_goals_list": []},
            "home":    {"matches_played": 0, "wins": 0, "draws": 0, "losses": 0, "goals_scored": 0, "goals_conceded": 0, "both_teams_scored": 0, "clean_sheets": 0, "failed_to_score": 0, "total_goals_list": []},
            "away":    {"matches_played": 0, "wins": 0, "draws": 0, "losses": 0, "goals_scored": 0, "goals_conceded": 0, "both_teams_scored": 0, "clean_sheets": 0, "failed_to_score": 0, "total_goals_list": []},
            "by_result": {
                "win":  {"matches_played": 0, "matches": []},
                "draw": {"matches_played": 0, "matches": []},
                "loss": {"matches_played": 0, "matches": []}
            },
            "form_string_overall": "", "form_string_home": "", "form_string_away": "",
            # Store detailed match summaries for recent games analysis
            "recent_match_summaries_overall": [], # Stores summaries regardless of venue
            # No need for separate home/away lists here, derive later
            # "recent_matches_home": [],
            # "recent_matches_away": []
        }
        MAX_HISTORY = 15
        # Determine if the source history is ALREADY filtered by venue (e.g., StatArea host/guest)
        is_venue_specific_history = (source == "statarea" and venue_context in ["host", "guest"])

        history_to_analyze = match_history[:MAX_HISTORY]
        games_analyzed_count = len(history_to_analyze)

        # --- Main Loop (Oldest to Newest for Form String Construction) ---
        for match in reversed(history_to_analyze):
            venue, result, scored, conceded, opponent_name = None, None, None, None, "Unknown Opponent"
            # --- Extract data based on source ---
            if source == "statarea" and isinstance(match, dict):
                statarea_venue = match.get("venue") # Usually 'home' or 'away' relative to the team
                if venue_context == "host":
                    venue = "home"
                elif venue_context == "guest":
                    venue = "away"
                elif statarea_venue: # Fallback if venue_context wasn't set but match has venue
                    venue = statarea_venue.lower()
                else:
                    logger.warning(f"StatArea history missing venue context for team {team_id}. Cannot determine home/away.")
                    continue # Skip if venue is indeterminable

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

                # Convert potential string IDs from older data
                try:
                    current_team_id_int = int(team_id)
                    home_team_id_int = int(home_team_id) if home_team_id is not None else None
                    away_team_id_int = int(away_team_id) if away_team_id is not None else None
                except (ValueError, TypeError):
                     logger.warning(f"Could not compare team IDs due to type mismatch in match history: {team_id}, {home_team_id}, {away_team_id}")
                     continue


                if home_team_id_int == current_team_id_int:
                    venue, scored, conceded = "home", home_goals, away_goals
                    opponent_name = away_team.get("name", opponent_name)
                elif away_team_id_int == current_team_id_int:
                    venue, scored, conceded = "away", away_goals, home_goals
                    opponent_name = home_team.get("name", opponent_name)
                else:
                    # This match doesn't involve the target team_id (shouldn't happen with DB query)
                    logger.debug(f"Match history entry does not involve team {team_id}. Skipping.")
                    continue

                # Determine result from scores
                if scored is not None and conceded is not None:
                    try:
                        # Handle potential None or non-numeric scores
                        sc_int = int(scored) if scored is not None else None
                        cn_int = int(conceded) if conceded is not None else None

                        if sc_int is not None and cn_int is not None:
                            if sc_int > cn_int:
                                result = "win"
                            elif sc_int < cn_int:
                                result = "loss"
                            else:
                                result = "draw"
                        else:
                            result = None # Cannot determine result if scores are missing
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert scores to int for result calculation: {scored}, {conceded}")
                        result = None

            # Skip if essential data is missing
            if venue not in ["home", "away"] or result not in ["win", "draw", "loss"] or scored is None or conceded is None:
                logger.debug(f"Skipping match in history analysis due to missing data (venue={venue}, result={result}, sc={scored}, cn={conceded})")
                continue

            # Aggregate stats using explicit keys
            try:
                sc, cn = int(scored), int(conceded)
                res_char = result[0].upper()
                match_summary = {"result": res_char, "goals_scored": sc, "goals_conceded": cn, "opponent": opponent_name, "venue": venue}

                # Store overall summary regardless of source or venue filtering
                stats["recent_match_summaries_overall"].append(match_summary)

                # Aggregate overall stats ONLY IF the history source is NOT venue-specific
                if not is_venue_specific_history:
                    o = stats["overall"]; o["matches_played"] += 1; o["goals_scored"] += sc; o["goals_conceded"] += cn; o["total_goals_list"].append(sc + cn)
                    if result == "win": o["wins"] += 1
                    elif result == "draw": o["draws"] += 1
                    else: o["losses"] += 1
                    if sc > 0 and cn > 0: o["both_teams_scored"] += 1
                    if cn == 0: o["clean_sheets"] += 1
                    if sc == 0: o["failed_to_score"] += 1
                    stats["form_string_overall"] += res_char


                # Aggregate venue-specific stats (always update home/away based on the match's venue)
                venue_to_update = venue # 'home' or 'away' determined from the match data

                if venue_to_update in ["home", "away"]:
                     v = stats[venue_to_update]; v["matches_played"] += 1; v["goals_scored"] += sc; v["goals_conceded"] += cn; v["total_goals_list"].append(sc + cn)
                     if result == "win": v["wins"] += 1
                     elif result == "draw": v["draws"] += 1
                     else: v["losses"] += 1
                     if sc > 0 and cn > 0: v["both_teams_scored"] += 1
                     if cn == 0: v["clean_sheets"] += 1
                     if sc == 0: v["failed_to_score"] += 1
                     stats[f"form_string_{venue_to_update}"] += res_char
                     # Note: Recent match summaries are handled globally above


                # Aggregate By Result (using explicit keys)
                r = stats["by_result"][result]; r["matches_played"] += 1; r["matches"].append(match_summary)

            except (ValueError, TypeError) as e:
                 logger.warning(f"Error processing match stats: {e}, Scores: sc={scored}, cn={conceded}")
                 continue

        # Reverse form strings
        stats["form_string_overall"] = stats["form_string_overall"][::-1]
        stats["form_string_home"] = stats["form_string_home"][::-1]
        stats["form_string_away"] = stats["form_string_away"][::-1]

        # Get recent matches (most recent first) from the stored summaries
        # Recent summaries are always based on the actual venue of the match in history
        recent_overall_summaries = stats["recent_match_summaries_overall"][::-1]
        recent_home_summaries = [m for m in recent_overall_summaries if m.get("venue") == "home"]
        recent_away_summaries = [m for m in recent_overall_summaries if m.get("venue") == "away"]

        # ... (rest of the function remains the same, calculating metrics based on aggregated stats) ...
        # calculate_scope_metrics("overall", ...) will now correctly reflect either:
        # 1. Combined home/away history if source='matches'
        # 2. Empty/zero stats if source='statarea' (since overall wasn't updated)

        recent_overall_l5 = recent_overall_summaries[:5]
        recent_home_l3 = recent_home_summaries[:3]
        recent_away_l3 = recent_away_summaries[:3]
        recent_overall_l10 = recent_overall_summaries[:10] if len(recent_overall_summaries) >= 10 else None

        # --- Calculate Final Metrics ---
        final_features = {"match_history_source": source, "match_history_games_analyzed": games_analyzed_count}
        if is_venue_specific_history:
            final_features["match_history_venue_context"] = venue_context # Add context if used

        metrics_cache = {}

        # Function to calculate and cache metrics using explicit keys
        def calculate_scope_metrics(scope_key, scope_data):
            # ... calculation logic unchanged ...
            if scope_key in metrics_cache: return metrics_cache[scope_key]

            metrics = {}
            # Use explicit keys with .get for safety
            m = scope_data.get("matches_played", 0)

            if m > 0:
                sc = scope_data.get("goals_scored", 0)
                cn = scope_data.get("goals_conceded", 0)
                w = scope_data.get("wins", 0)
                d = scope_data.get("draws", 0)
                l = scope_data.get("losses", 0)
                btts = scope_data.get("both_teams_scored", 0)
                cs = scope_data.get("clean_sheets", 0)
                fts = scope_data.get("failed_to_score", 0)
                tg_list = scope_data.get("total_goals_list", [])

                metrics["matches_played"] = m
                metrics["average_goals_scored"] = round(sc / m, 3) if m else 0.0
                metrics["average_goals_conceded"] = round(cn / m, 3) if m else 0.0
                metrics["win_rate"] = round(w / m, 3) if m else 0.0
                metrics["draw_rate"] = round(d / m, 3) if m else 0.0
                metrics["loss_rate"] = round(l / m, 3) if m else 0.0
                metrics["both_teams_scored_rate"] = round(btts / m, 3) if m else 0.0
                metrics["clean_sheet_rate"] = round(cs / m, 3) if m else 0.0
                metrics["failed_to_score_rate"] = round(fts / m, 3) if m else 0.0
                # Add goal difference avg
                metrics["average_goal_difference"] = round((sc - cn) / m, 3) if m else 0.0


                # Over/Under Rates
                if tg_list:
                    metrics["over_0_5_goals_rate"] = round(sum(1 for g in tg_list if g > 0.5) / m, 3) if m else 0.0
                    metrics["over_1_5_goals_rate"] = round(sum(1 for g in tg_list if g > 1.5) / m, 3) if m else 0.0
                    metrics["over_2_5_goals_rate"] = round(sum(1 for g in tg_list if g > 2.5) / m, 3) if m else 0.0
                    metrics["over_3_5_goals_rate"] = round(sum(1 for g in tg_list if g > 3.5) / m, 3) if m else 0.0
                    metrics["over_4_5_goals_rate"] = round(sum(1 for g in tg_list if g > 4.5) / m, 3) if m else 0.0

                # Form String and Rating
                form_str_key = f"form_string_{scope_key}"
                if form_str_key in stats and stats[form_str_key]:
                     form_string = stats[form_str_key]
                     metrics["form_string"] = form_string # Already explicit key from stats dict
                     metrics["form_rating_points_pct"] = round(self._calculate_form_rating(form_string), 3)

            metrics_cache[scope_key] = metrics
            return metrics


        # Calculate Overall, Home, Away metrics
        # Note: overall_metrics might be empty/zero if is_venue_specific_history was True
        overall_metrics = calculate_scope_metrics("overall", stats["overall"])
        home_metrics = calculate_scope_metrics("home", stats["home"])
        away_metrics = calculate_scope_metrics("away", stats["away"])


        # Add metrics directly to final_features with scope prefixes
        if overall_metrics:
             for k, v in overall_metrics.items(): final_features[f"hist_overall_{k}"] = v
        if home_metrics:
             for k, v in home_metrics.items(): final_features[f"hist_home_{k}"] = v
        if away_metrics:
             for k, v in away_metrics.items(): final_features[f"hist_away_{k}"] = v

        # --- Calculate Home/Away Differentials ---
        # Only calculate if both home and away metrics have data
        if home_metrics and away_metrics and home_metrics.get("matches_played", 0) > 0 and away_metrics.get("matches_played", 0) > 0:
            diff_stats = {}
            # Use explicit keys from the calculated metrics
            # Add 'form_rating_points_pct' if it exists in both
            keys_to_diff = ["average_goals_scored", "average_goals_conceded", "win_rate", "draw_rate", "loss_rate", "both_teams_scored_rate", "clean_sheet_rate", "failed_to_score_rate", "average_goal_difference"]
            if "form_rating_points_pct" in home_metrics and "form_rating_points_pct" in away_metrics:
                keys_to_diff.append("form_rating_points_pct")

            for key in keys_to_diff:
                 if key in home_metrics and key in away_metrics: # Check presence again just in case
                     diff = home_metrics[key] - away_metrics[key]
                     # Add explicit prefix 'hist_diff_home_away_'
                     diff_stats[f"hist_diff_home_away_{key}"] = round(diff, 3)
            if diff_stats:
                 final_features.update(diff_stats) # Add diffs directly


        # --- Rolling Stats with Details ---
        # ... (rest of the logic for rolling stats remains the same) ...
        rolling_stats_detail = {} # Keep detailed match strings separate if needed
        if recent_overall_l5:
             overall_l5_form = "".join(m["result"] for m in recent_overall_l5)
             final_features["hist_overall_l5_form_string"] = overall_l5_form
             final_features["hist_overall_l5_form_rating_points_pct"] = round(self._calculate_form_rating(overall_l5_form), 3)
             # Generate descriptive strings for details section
             rolling_stats_detail["overall_l5_matches"] = [
                 f"{m['result']} {m['goals_scored']}-{m['goals_conceded']} vs {m['opponent']} ({m['venue']})"
                 for m in recent_overall_l5
             ]

        # ... similar updates for overall_l10, home_l3, away_l3 using explicit keys like 'hist_home_l3_avg_scored' ...
        if recent_home_l3:
            home_l3_form = "".join(m["result"] for m in recent_home_l3)
            home_l3_sc = sum(m["goals_scored"] for m in recent_home_l3)
            home_l3_cn = sum(m["goals_conceded"] for m in recent_home_l3)
            home_l3_m = len(recent_home_l3)
            final_features["hist_home_l3_form_string"] = home_l3_form
            final_features["hist_home_l3_form_rating_points_pct"] = round(self._calculate_form_rating(home_l3_form), 3)
            final_features["hist_home_l3_avg_scored"] = round(home_l3_sc / home_l3_m, 3) if home_l3_m else 0.0
            final_features["hist_home_l3_avg_conceded"] = round(home_l3_cn / home_l3_m, 3) if home_l3_m else 0.0
            # Generate descriptive strings for details section
            rolling_stats_detail["home_l3_matches"] = [
                 f"{m['result']} {m['goals_scored']}-{m['goals_conceded']} vs {m['opponent']}"
                 for m in recent_home_l3
             ]

        if recent_away_l3: # Calculate for away L3
             away_l3_form = "".join(m["result"] for m in recent_away_l3)
             away_l3_sc = sum(m["goals_scored"] for m in recent_away_l3)
             away_l3_cn = sum(m["goals_conceded"] for m in recent_away_l3)
             away_l3_m = len(recent_away_l3)
             final_features["hist_away_l3_form_string"] = away_l3_form
             final_features["hist_away_l3_form_rating_points_pct"] = round(self._calculate_form_rating(away_l3_form), 3)
             final_features["hist_away_l3_avg_scored"] = round(away_l3_sc / away_l3_m, 3) if away_l3_m else 0.0
             final_features["hist_away_l3_avg_conceded"] = round(away_l3_cn / away_l3_m, 3) if away_l3_m else 0.0
             # Generate descriptive strings for details section
             rolling_stats_detail["away_l3_matches"] = [
                 f"{m['result']} {m['goals_scored']}-{m['goals_conceded']} vs {m['opponent']}"
                 for m in recent_away_l3
             ]


        if rolling_stats_detail: final_features["rolling_stats_details"] = rolling_stats_detail # Add detailed strings if needed


        # --- Performance by Result with Avg Scorelines ---
        # ... (rest of the logic remains the same) ...
        perf_by_result = {}
        for result_type, data in stats["by_result"].items():
            m = data.get("matches_played", 0)
            if m > 0:
                total_sc = sum(match.get("goals_scored", 0) for match in data.get("matches", []))
                total_cn = sum(match.get("goals_conceded", 0) for match in data.get("matches", []))
                # Use explicit keys with prefix 'hist_perf_by_result_{result_type}_'
                final_features[f"hist_perf_by_result_{result_type}_matches"] = m
                final_features[f"hist_perf_by_result_{result_type}_avg_scored"] = round(total_sc / m, 3) if m else 0.0
                final_features[f"hist_perf_by_result_{result_type}_avg_conceded"] = round(total_cn / m, 3) if m else 0.0
                # Keep avg scoreline separate if desired
                # perf_by_result[result_type] = { ... "avg_scoreline": ... }
        # if perf_by_result: final_features["performance_by_result_details"] = perf_by_result



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
        league_id_str = league_data.get("id") # Should be a string now from _extract_core_match_info
        league_statarea_id = "unknown"
        if league_id_str and league_id_str != "None":
            found = False
            for _, ids in LEAGUE_ID_MAPPING.items():
                 # Compare string IDs directly
                if ids.get("mongodb_id") == league_id_str:
                    league_statarea_id = ids.get("statarea_id", "unknown")
                    found = True
                    break
            if not found:
                 logger.warning(f"Could not find StatArea mapping for MongoDB league ID: {league_id_str}")
        league_data["statarea_id"] = league_statarea_id

        # Map home team ID
        home_id = home_team_data.get("id") # Should be an int now
        home_name = home_team_data.get("name", "")
        home_statarea_id = "unknown"
        if home_id is not None:
            found = False
            for team_name_map, ids in TEAM_ID_MAPPING.items():
                mongodb_id_mapped = ids.get("mongodb_id")
                statarea_id_mapped = ids.get("statarea_id")
                # Attempt ID match first (convert mapped ID to int for comparison)
                try:
                    if mongodb_id_mapped is not None and int(mongodb_id_mapped) == home_id:
                        home_statarea_id = statarea_id_mapped or "unknown"
                        found = True
                        break
                except (ValueError, TypeError):
                     # logger.debug(f"Could not compare home team ID {home_id} with mapped ID {mongodb_id_mapped}")
                    pass # Ignore comparison error and try name next

            # Fallback to name match if ID match failed and name exists
            if not found and home_name:
                 for team_name_map, ids in TEAM_ID_MAPPING.items():
                      if team_name_map == home_name:
                          home_statarea_id = ids.get("statarea_id", "unknown")
                          found = True
                          break
            if not found:
                 logger.warning(f"Could not find StatArea mapping for home team ID: {home_id} or Name: {home_name}")

        home_team_data["statarea_id"] = home_statarea_id

        # Map away team ID (similar logic)
        away_id = away_team_data.get("id") # Should be an int now
        away_name = away_team_data.get("name", "")
        away_statarea_id = "unknown"
        if away_id is not None:
            found = False
            for team_name_map, ids in TEAM_ID_MAPPING.items():
                mongodb_id_mapped = ids.get("mongodb_id")
                statarea_id_mapped = ids.get("statarea_id")
                # Attempt ID match first
                try:
                    if mongodb_id_mapped is not None and int(mongodb_id_mapped) == away_id:
                        away_statarea_id = statarea_id_mapped or "unknown"
                        found = True
                        break
                except (ValueError, TypeError):
                    # logger.debug(f"Could not compare away team ID {away_id} with mapped ID {mongodb_id_mapped}")
                    pass

            # Fallback to name match
            if not found and away_name:
                 for team_name_map, ids in TEAM_ID_MAPPING.items():
                     if team_name_map == away_name:
                         away_statarea_id = ids.get("statarea_id", "unknown")
                         found = True
                         break
            if not found:
                 logger.warning(f"Could not find StatArea mapping for away team ID: {away_id} or Name: {away_name}")

        away_team_data["statarea_id"] = away_statarea_id


    def _remove_none_values(self, data: Any) -> Any:
        """Recursively remove keys whose values are None."""
        if isinstance(data, dict):
            return {k: self._remove_none_values(v) for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            # Keep the list structure, just clean items within it
            return [self._remove_none_values(item) for item in data if item is not None]
        else:
            return data

    def save_individual_game_file(self, processed_game_data: Dict[str, Any]):
        """
        Save the fully processed and structured game data (unified format)
        to a JSON file in the unified data directory.

        Args:
            processed_game_data: The final, cleaned game data dictionary with the new structure.
        """
        # Ensure the main output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            logger.info(f"Created unified output directory: {OUTPUT_DIR}")

        # Get components for filename from the processed data (new structure)
        # Access raw_data -> home/away -> basic_info for names
        raw_data_section = processed_game_data.get("raw_data", {})
        home_raw_info = raw_data_section.get("home", {}).get("basic_info", {})
        away_raw_info = raw_data_section.get("away", {}).get("basic_info", {})


        # Sanitize team names for filename
        home_team_name = self._sanitize_filename(home_raw_info.get("name", "UnknownHome"))
        away_team_name = self._sanitize_filename(away_raw_info.get("name", "UnknownAway"))

        # Get date and fixture ID from top-level fields
        match_date = processed_game_data.get("match_date", None) # Use "match_date" (YYYY-MM-DD)
        fixture_id = processed_game_data.get("fixture_id", "unknown_fixture")

        # Handle potential date issues
        date_part = None
        if match_date and isinstance(match_date, str) and len(match_date) >= 10:
            date_part = match_date[:10] # Take YYYY-MM-DD
        else:
             # Fallback to fixture_meta date
             fixture_meta = processed_game_data.get("fixture_meta", {})
             date_str_from_meta = fixture_meta.get("date_utc") # Use date_utc key
             if date_str_from_meta and isinstance(date_str_from_meta, str):
                 try:
                     date_part = date_str_from_meta.split('T')[0]
                 except Exception:
                     logger.warning(f"Could not parse date_utc '{date_str_from_meta}' for filename.")

        # Final fallback if date is still missing
        if not date_part:
            logger.warning(f"Date part is still invalid for fixture {fixture_id}. Using today's date as fallback.")
            date_part = self.get_current_date_str()


        # Create filename: YYYY-MM-DD_HomeTeam_vs_AwayTeam_fixtureID.json
        filename = f"{date_part}_{home_team_name}_vs_{away_team_name}_{fixture_id}.json"
        file_path = os.path.join(OUTPUT_DIR, filename)

        # Save the processed game data to file
        try:
            # Data should already be normalized/cleaned, final conversion for Mongo types
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
                elif isinstance(date_val, int): # Handle timestamp int directly in $date
                     try:
                         dt_obj = datetime.fromtimestamp(date_val / 1000, tz=timezone.utc)
                         return dt_obj.isoformat()
                     except (ValueError, TypeError):
                         logger.warning(f"Could not convert integer timestamp $date: {date_val}")
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
