import os
import logging
from datetime import date, datetime
import json
from typing import Dict, List, Any, Optional
import re
import sys
import time

# Import MongoDB manager from existing code
try:
    from get_data.api_football.db_mongo import MongoDBManager
    from api_football.db_ids.team_id_mappings import TEAM_ID_MAPPING
    from api_football.db_ids.league_id_mappings import LEAGUE_ID_MAPPING
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'daily_output', 'daily_games')

class DailyGameExtractor:
    """Extracts game data from MongoDB database for a specific day."""
    
    def __init__(self, use_mongo=True):
        """Initialize the extractor with MongoDB connection."""
        self.mongo_db = None
        self.league_standings_map = {} # Added to store standings { (league_id, season): standings_data }
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
        then processes it.
        
        Args:
            date_str: Date string in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary containing all the extracted and processed game data for the day.
        """
        # Use today's date if none is provided
        if not date_str:
            date_str = self.get_current_date_str()
            
        logger.info(f"🚀 Starting extraction for date: {date_str} from MongoDB (Agentic FC Workflow)")
        
        detailed_games = []
        fixture_ids = []

        if not self.mongo_db:
            logger.error("❌ MongoDB connection not available. Cannot extract daily games.")
            # Return an empty structure or raise an error
            return {"date": date_str, "games": [], "total_games": 0, "error": "MongoDB connection failed"}

        try:
            # Step 1: Get the list of fixture IDs for the target date directly from MongoDB
            fixture_ids = self.mongo_db.get_fixture_ids_for_date(date_str)

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
                    # Fetch detailed data from the 'game_details' collection using the fixture ID
                    match_data = self.mongo_db.get_game_details_by_fixture_id(fixture_id)
                    
                    if match_data:
                        # Process the fetched data
                        logger.info(f"Processing game data for fixture {fixture_id}")
                        game_info = self.process_game_data(match_data)
                        detailed_games.append(game_info)
                        
                        # Save individual game file
                        self.save_individual_game_file(game_info)
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
                "games": detailed_games, # Return any games processed so far
                "total_games": len(detailed_games),
                "error": f"Failed during MongoDB fetch/process: {str(e)}"
            }

        # Prepare the final result dictionary
        result = {
            "date": date_str,
            "games": detailed_games,
            "total_games": len(detailed_games)
        }
        
        # Save standings files after processing all games
        if self.league_standings_map:
            self._save_standings_files(date_str)

        # Clear the standings map for the next potential run
        self.league_standings_map = {}
            
        logger.info(f"✅ Successfully extracted and processed {len(detailed_games)} games for {date_str} from MongoDB.")
        return result
    
    def process_game_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process match data fetched from MongoDB and handle standings extraction.

        Args:
            match_data: Raw match data from MongoDB's 'game_details' collection.

        Returns:
            Processed game data dictionary ready for saving/output.
        """
        # Remove MongoDB internal ID if present
        if "_id" in match_data:
            try:
                del match_data["_id"]
            except Exception:
                logger.warning("Could not remove _id field, might not exist or be standard type.")

        logger.debug(f"Starting processing for fixture ID: {match_data.get('fixture_id', 'N/A')}")

        # --- Map StatArea IDs for cross-referencing ---
        self._add_statarea_id_mappings(match_data)

        # --- Standings Handling ---
        # Extract league ID and season for standings lookup
        league_id = None
        season = None

        # Prioritize 'league' object if it exists in game_details
        if "league" in match_data:
            league_obj = match_data.get("league", {})
            league_id = league_obj.get("id")
            season = league_obj.get("season")

        if not league_id:
            league_id = match_data.get("league_id")
        if not season:
            season = match_data.get("season")

        # Default season if absolutely necessary
        if league_id and not season:
            current_year = datetime.now().year
            logger.warning(f"⚠️ Season not found for league {league_id}, defaulting to {current_year}.")
            season = current_year

        logger.debug(f"Extracted league ID: {league_id}, season: {season} for standings lookup.")

        # Fetch and cache standings if league_id and season are valid
        if league_id and season:
            league_key = (league_id, season)
            if league_key not in self.league_standings_map:
                # Standings might be embedded in 'game_details' or need separate fetching
                standings = None
                # Check if standings are already embedded
                if "standings_snapshot" in match_data:
                    standings = match_data["standings_snapshot"]
                    logger.info(f"Using embedded standings snapshot for league {league_id}, season {season}")
                elif "standings" in match_data:
                    standings = match_data["standings"]
                    logger.info(f"Using embedded standings for league {league_id}, season {season}")

                # Fetch from MongoDB 'standings' collection if not embedded
                if not standings and self.mongo_db:
                    logger.info(f"Fetching standings from MongoDB for league {league_id}, season {season}")
                    standings = self.mongo_db.get_standings_by_league_season(league_id, season)
                    if standings:
                        logger.info(f"✅ Successfully fetched and cached standings for league {league_id}, season {season}")
                    else:
                        logger.warning(f"🤷 Could not find/fetch standings for league {league_id}, season {season} from MongoDB.")

                # Store whatever standings we found (or None) in the map
                self.league_standings_map[league_key] = standings
            else:
                logger.debug(f"Standings for league {league_id}, season {season} already in cache.")
        else:
            logger.warning(f"⚠️ Missing league ID ({league_id}) or season ({season}) in match data. Cannot process standings.")

        # Remove potentially embedded standings from the main game data before final cleaning/saving
        if "standings" in match_data:
            del match_data["standings"]
        if "standings_snapshot" in match_data:
            del match_data["standings_snapshot"]

        # Return the processed match_data
        return match_data
    
    def _add_statarea_id_mappings(self, match_data: Dict[str, Any]):
        """
        Add StatArea ID mappings to the match data for teams and leagues.
        This modifies the match_data dictionary in place.
        
        Args:
            match_data: Match data to enrich with StatArea IDs
        """
        # Map league ID if present
        if "league" in match_data and "id" in match_data["league"]:
            league_id = str(match_data["league"]["id"])
            
            # Find corresponding statarea ID from mappings
            statarea_id = "unknown"
            for league, ids in LEAGUE_ID_MAPPING.items():
                if ids.get("mongodb_id") == league_id:
                    statarea_id = ids.get("statarea_id", "unknown")
                    break
                    
            # Add StatArea ID to league data
            match_data["league"]["statarea_id"] = statarea_id
        
        # Map home team ID if present
        if "home_team" in match_data and "id" in match_data["home_team"]:
            home_id = str(match_data["home_team"]["id"])
            home_name = match_data["home_team"].get("name", "")
            
            # Find corresponding statarea ID
            home_statarea_id = "unknown"
            for team_name, ids in TEAM_ID_MAPPING.items():
                if ids.get("mongodb_id") == home_id or team_name == home_name:
                    home_statarea_id = ids.get("statarea_id", "unknown")
                    break
                    
            # Add StatArea ID to home team data
            match_data["home_team"]["statarea_id"] = home_statarea_id
            
        # Map away team ID if present
        if "away_team" in match_data and "id" in match_data["away_team"]:
            away_id = str(match_data["away_team"]["id"])
            away_name = match_data["away_team"].get("name", "")
            
            # Find corresponding statarea ID
            away_statarea_id = "unknown"
            for team_name, ids in TEAM_ID_MAPPING.items():
                if ids.get("mongodb_id") == away_id or team_name == away_name:
                    away_statarea_id = ids.get("statarea_id", "unknown")
                    break
                    
            # Add StatArea ID to away team data
            match_data["away_team"]["statarea_id"] = away_statarea_id
        
        # Map teams in H2H data if present
        if "h2h" in match_data:
            for h2h_match in match_data["h2h"]:
                if "teams" in h2h_match:
                    if "home" in h2h_match["teams"] and "id" in h2h_match["teams"]["home"]:
                        team_id = str(h2h_match["teams"]["home"]["id"])
                        team_name = h2h_match["teams"]["home"].get("name", "")
                        
                        # Find corresponding statarea ID
                        statarea_id = "unknown"
                        for name, ids in TEAM_ID_MAPPING.items():
                            if ids.get("mongodb_id") == team_id or name == team_name:
                                statarea_id = ids.get("statarea_id", "unknown")
                                break
                                
                        # Add StatArea ID
                        h2h_match["teams"]["home"]["statarea_id"] = statarea_id
                        
                    if "away" in h2h_match["teams"] and "id" in h2h_match["teams"]["away"]:
                        team_id = str(h2h_match["teams"]["away"]["id"])
                        team_name = h2h_match["teams"]["away"].get("name", "")
                        
                        # Find corresponding statarea ID
                        statarea_id = "unknown"
                        for name, ids in TEAM_ID_MAPPING.items():
                            if ids.get("mongodb_id") == team_id or name == team_name:
                                statarea_id = ids.get("statarea_id", "unknown")
                                break
                                
                        # Add StatArea ID
                        h2h_match["teams"]["away"]["statarea_id"] = statarea_id
    
    def _process_team(self, team_data_full_match: Dict[str, Any], team_key: str):
        """
        Processes team data from the full match document, prioritizing MongoDB stats.

        Args:
            team_data_full_match: The full match data dictionary from MongoDB.
            team_key: 'home_team' or 'away_team'.

        Returns:
            Processed team dictionary or None if team data is missing.
        """
        team_info = team_data_full_match.get(team_key)
        if not team_info: return None

        # Use the nested stats object from MongoDB (e.g., 'home_team_stats')
        mongodb_stats_key = f"{team_key}_stats"
        mongodb_stats_raw = team_data_full_match.get(mongodb_stats_key, {})

        # Extract team details
        team_id = team_info.get("id")
        team_name = team_info.get("name")

        # --- Organize MongoDB Stats ---
        mongodb_stats_processed = {}

        # Form - Directly from the stats object
        mongodb_stats_processed["form"] = mongodb_stats_raw.get("form")

        # Fixtures - Use the structure from MongoDB stats
        if "fixtures" in mongodb_stats_raw:
            mongodb_stats_processed["fixtures"] = mongodb_stats_raw.get("fixtures")

        # Goals - Use the structure from MongoDB stats
        if "goals" in mongodb_stats_raw:
            mongodb_stats_processed["goals"] = mongodb_stats_raw.get("goals")

        # Biggest - Use the structure from MongoDB stats
        if "biggest" in mongodb_stats_raw:
            mongodb_stats_processed["biggest"] = mongodb_stats_raw.get("biggest")

        # Performance Metrics - Use from MongoDB stats
        mongodb_stats_processed["performance"] = {
            "clean_sheet": mongodb_stats_raw.get("clean_sheet"),
            "failed_to_score": mongodb_stats_raw.get("failed_to_score"),
            "penalty": mongodb_stats_raw.get("penalty"),
            "lineups": mongodb_stats_raw.get("lineups"),
            "cards": mongodb_stats_raw.get("cards"),
        }
        
        # --- Final Team Structure ---
        processed_team = {
            "id": team_id,
            "name": team_name,
            "logo": team_info.get("logo"),
            "winner": team_info.get("winner"),
            "coach": team_info.get("coach") or team_info.get("lineups", [{}])[0].get("coach"),
            "formation": team_info.get("formation") or team_info.get("lineups", [{}])[0].get("formation"),
            "mongodb_stats": self._remove_none_values(mongodb_stats_processed),
            "statarea_id": team_info.get("statarea_id", "unknown")
        }
        return self._remove_none_values(processed_team)

    def _clean_final_game_data(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cleans and organizes the final game data before saving.
        """
        cleaned = {}

        # 1. Fixture Info
        fixture_source = game_data.get("match_info", game_data.get("fixture", {}))
        cleaned["fixture_info"] = {
            "id": game_data.get("fixture_id"),
            "referee": fixture_source.get("referee"),
            "timezone": fixture_source.get("timezone"),
            "date": fixture_source.get("date"),
            "timestamp": fixture_source.get("timestamp"),
            "venue": fixture_source.get("venue"),
            "status": fixture_source.get("status"),
        }

        # 2. League Info
        league_info = game_data.get("league", {})
        cleaned["league"] = {
            "id": league_info.get("id") or game_data.get("league_id"),
            "name": league_info.get("name") or game_data.get("league_name"),
            "country": league_info.get("country"),
            "logo": league_info.get("logo"),
            "flag": league_info.get("flag"),
            "season": league_info.get("season") or game_data.get("season"),
            "round": league_info.get("round"),
            "statarea_id": league_info.get("statarea_id"),
        }

        # 3. Teams Info
        cleaned["teams"] = {
            "home": self._process_team(game_data, "home_team"),
            "away": self._process_team(game_data, "away_team"),
        }

        # 4. Score and Goals
        cleaned["score"] = game_data.get("score")
        cleaned["goals"] = game_data.get("goals")

        # 5. Match Events, Lineups, Statistics
        if game_data.get("events"):
            cleaned["events"] = game_data.get("events")
        if game_data.get("lineups"):
            cleaned["lineups"] = game_data.get("lineups")
        if game_data.get("statistics"):
            cleaned["statistics"] = game_data.get("statistics")

        # 6. Predictions
        predictions_raw = game_data.get("predictions", {})
        pred_content = predictions_raw.get("predictions", predictions_raw)

        cleaned["predictions"] = {
            "winner": pred_content.get("winner"),
            "win_or_draw": pred_content.get("win_or_draw"),
            "advice": pred_content.get("advice"),
            "percent": pred_content.get("percent"),
            "under_over_prediction": pred_content.get("under_over"),
            "goals_prediction": pred_content.get("goals"),
            "comparison": predictions_raw.get("comparison"),
            "h2h_prediction_details": pred_content.get("h2h"),
            "winning_odds": predictions_raw.get("winning_odds"),
        }

        # 7. H2H Data
        h2h_raw = []
        if isinstance(predictions_raw.get("h2h"), list):
            h2h_raw = predictions_raw["h2h"]
        elif isinstance(predictions_raw.get("h2h"), dict):
            h2h_raw = predictions_raw.get("h2h", {}).get("matches", [])

        if not h2h_raw and isinstance(game_data.get("h2h"), list):
            h2h_raw = game_data["h2h"]

        h2h_list = []
        for match in h2h_raw[:10]:
            h2h_fixture = match.get("fixture", {})
            h2h_teams = match.get("teams", {})
            h2h_goals = match.get("goals", {})
            h2h_league = match.get("league", {})
            h2h_home = h2h_teams.get("home", {})
            h2h_away = h2h_teams.get("away", {})

            h2h_list.append({
                "fixture_id": h2h_fixture.get("id"),
                "date": h2h_fixture.get("date"),
                "home_team": {
                    "id": h2h_home.get("id"),
                    "name": h2h_home.get("name"),
                    "logo": h2h_home.get("logo"),
                    "winner": h2h_home.get("winner"),
                },
                "away_team": {
                    "id": h2h_away.get("id"),
                    "name": h2h_away.get("name"),
                    "logo": h2h_away.get("logo"),
                    "winner": h2h_away.get("winner"),
                },
                "score": {
                    "home": h2h_goals.get("home"),
                    "away": h2h_goals.get("away")
                },
                "league": {
                    "id": h2h_league.get("id"),
                    "name": h2h_league.get("name"),
                    "country": h2h_league.get("country")
                },
                "venue": h2h_fixture.get("venue"),
                "status": h2h_fixture.get("status")
            })
        cleaned["h2h"] = h2h_list

        # 8. Final Cleanup
        cleaned = self._remove_none_values(cleaned)

        return cleaned

    def _remove_none_values(self, data):
        """
        Recursively remove keys with None values from dictionaries and lists.
        """
        if isinstance(data, dict):
            return {
                k: self._remove_none_values(v)
                for k, v in data.items()
                if v is not None
            }
        elif isinstance(data, list):
            return [self._remove_none_values(item) for item in data if item is not None]
        else:
            return data

    def save_individual_game_file(self, game_data: Dict[str, Any]):
        """
        Save cleaned game data to an individual JSON file in the league-specific directory.

        Args:
            game_data: Raw game data after initial processing
        """
        # Clean the data before determining the filename components
        cleaned_data = self._clean_final_game_data(game_data)

        # Get league info for directory name
        league_info = cleaned_data.get("league", {})
        league_name = league_info.get("name", "UnknownLeague")
        league_country = league_info.get("country", "")

        # Use standardized league name for the directory
        standardized_league_name = self._standardize_league_name(league_name, league_country)
        league_dir_name = self._sanitize_filename(standardized_league_name)
        league_dir_path = os.path.join(OUTPUT_DIR, league_dir_name)

        # Create league directory if it doesn't exist
        if not os.path.exists(league_dir_path):
            os.makedirs(league_dir_path)

        # Get team and fixture info for filename
        home_team_info = cleaned_data.get("teams", {}).get("home", {})
        away_team_info = cleaned_data.get("teams", {}).get("away", {})
        fixture_info = cleaned_data.get("fixture_info", {})

        # Get team names and sanitize for filename
        home_team_name = home_team_info.get("name", "UnknownHome")
        away_team_name = away_team_info.get("name", "UnknownAway")
        home_team_file_part = self._sanitize_filename(home_team_name)
        away_team_file_part = self._sanitize_filename(away_team_name)

        # Get date and fixture ID
        date_str = fixture_info.get("date", self.get_current_date_str())
        date_part = date_str.split('T')[0] if date_str and isinstance(date_str, str) else self.get_current_date_str()
        fixture_id = fixture_info.get("id", "unknown_fixture")

        # Create filename
        filename = f"{date_part}_{home_team_file_part}_vs_{away_team_file_part}_{fixture_id}.json"
        file_path = os.path.join(league_dir_path, filename)

        # Save the cleaned game data to file
        try:
            cleaned_data = self._convert_mongodb_types(cleaned_data)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved cleaned game data to {file_path}")
        except TypeError as e:
            logger.error(f"Error saving JSON data to {file_path}: {e}. Check for non-serializable types.")
        except Exception as e:
            logger.error(f"An unexpected error occurred saving {file_path}: {e}")

    def _save_standings_files(self, date_str: str):
        """Saves league standings to their respective league directories."""
        logger.info(f"Saving standings for {len(self.league_standings_map)} leagues...")

        for (league_id, season), standings_data in self.league_standings_map.items():
            if not standings_data:
                logger.warning(f"Skipping league {league_id} season {season} due to missing standings data.")
                continue

            # Get league info
            league_info = {}
            standings_array = []

            if "league" in standings_data:
                league_info = standings_data.get("league", {})
            elif "standings_api_response" in standings_data and standings_data["standings_api_response"]:
                api_response = standings_data["standings_api_response"][0] if isinstance(standings_data["standings_api_response"], list) else {}
                if "league" in api_response:
                    league_info = api_response.get("league", {})

            # Get standings data
            if "standings" in standings_data:
                standings_array = standings_data.get("standings", [])
            elif "standings_api_response" in standings_data and standings_data["standings_api_response"]:
                api_response = standings_data["standings_api_response"][0] if isinstance(standings_data["standings_api_response"], list) else {}
                if "league" in api_response and "standings" in api_response["league"]:
                    standings_array = api_response["league"].get("standings", [])

            # Create league directory name using standardization
            league_name = league_info.get("name", f"League_{league_id}")
            league_country = league_info.get("country", "")
            standardized_league_name = self._standardize_league_name(league_name, league_country)
            league_dir_name = self._sanitize_filename(standardized_league_name)
            league_dir_path = os.path.join(OUTPUT_DIR, league_dir_name)

            # Create league directory if it doesn't exist
            if not os.path.exists(league_dir_path):
                os.makedirs(league_dir_path)

            # Create standings file in league directory
            standings_filename = f"{date_str}_standings.json"
            standings_file_path = os.path.join(league_dir_path, standings_filename)

            # Prepare standings data
            standings_data_to_save = {
                "league_info": league_info,
                "standings": standings_array,
                "date": date_str,
                "season": season
            }

            # Save standings to file
            try:
                with open(standings_file_path, 'w', encoding='utf-8') as f:
                    json.dump(standings_data_to_save, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved standings for {league_name} ({standardized_league_name}) to {standings_file_path}")
            except Exception as e:
                logger.error(f"Failed to save standings for {league_name} ({standardized_league_name}) to {standings_file_path}: {e}")

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
        # Ensure it's not empty
        if not sanitized:
            return "invalid_name"
        return sanitized

    def _convert_mongodb_types(self, data):
        """Convert MongoDB specific types to standard Python types."""
        if isinstance(data, dict):
            for key, value in list(data.items()):
                if key == "$numberInt" and isinstance(value, str):
                    return int(value)
                data[key] = self._convert_mongodb_types(value)
        elif isinstance(data, list):
            return [self._convert_mongodb_types(item) for item in data]
        return data

    def save_summary_file(self, data: Dict[str, Any], output_file: Optional[str] = None):
        """
        Save a summary of all extracted data to a JSON file in each league directory.
        """
        # Group games by league
        games_by_league = {}

        for raw_game in data.get("games", []):
            # Clean the game data first
            cleaned_game_for_summary = self._clean_final_game_data(raw_game)
            league_info = cleaned_game_for_summary.get("league", {})
            league_name = league_info.get("name", "UnknownLeague")
            league_country = league_info.get("country", "")

            # Create standardized league key for grouping and directory naming
            standardized_league_name = self._standardize_league_name(league_name, league_country)
            league_dir_name = self._sanitize_filename(standardized_league_name)

            if league_dir_name not in games_by_league:
                games_by_league[league_dir_name] = {
                    "league_info": league_info,
                    "standardized_name": standardized_league_name,
                    "games": []
                }

            # Add game to league group using cleaned data
            fixture_info = cleaned_game_for_summary.get("fixture_info", {})
            home_team_info = cleaned_game_for_summary.get("teams", {}).get("home", {})
            away_team_info = cleaned_game_for_summary.get("teams", {}).get("away", {})

            # Extract StatArea ID if available
            home_statarea_id = home_team_info.get("statarea_id", "unknown")
            away_statarea_id = away_team_info.get("statarea_id", "unknown")

            games_by_league[league_dir_name]["games"].append({
                "fixture_id": fixture_info.get("id"),
                "kickoff_time": fixture_info.get("date"),
                "home_team": {
                    "id": home_team_info.get("id"),
                    "name": home_team_info.get("name"),
                    "statarea_id": home_statarea_id
                },
                "away_team": {
                    "id": away_team_info.get("id"),
                    "name": away_team_info.get("name"),
                    "statarea_id": away_statarea_id
                }
            })

        # Save summary file for each league
        date_str = data.get("date", self.get_current_date_str())

        for league_dir_name, league_data in games_by_league.items():
            # Use the derived league_dir_name for the path
            league_dir_path = os.path.join(OUTPUT_DIR, league_dir_name)

            # Create league directory if it doesn't exist
            if not os.path.exists(league_dir_path):
                os.makedirs(league_dir_path)

            # Create summary file
            summary_filename = f"games_summary_{date_str}.json"
            summary_file_path = os.path.join(league_dir_path, summary_filename)

            summary = {
                "date": date_str,
                "league": league_data["league_info"],
                "standardized_directory_name": league_data["standardized_name"],
                "total_games": len(league_data["games"]),
                "games": league_data["games"]
            }

            try:
                with open(summary_file_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved summary for {league_data['league_info'].get('name')} ({league_dir_name}) to {summary_file_path}")
            except Exception as e:
                logger.error(f"Failed to save summary for {league_data['league_info'].get('name')} ({league_dir_name}) to {summary_file_path}: {e}")

    def _standardize_league_name(self, league_name: str, country: str) -> str:
        """
        Standardize league names for consistent directory naming.
        """
        # Combine name and country for initial check, handling missing country
        base_name = f"{league_name}_{country}" if country else league_name
        lower_name = league_name.lower()
        lower_country = country.lower() if country else ""

        # Define mappings for standardization
        standardization_map = {
            ('eredivisie 2', 'netherlands'): 'Eerste_Divisie_Netherlands',
            ('eerste divisie', 'netherlands'): 'Eerste_Divisie_Netherlands',
            ('super lig', 'turkey'): 'Süper_Lig_Turkey',
            ('süper lig', 'turkey'): 'Süper_Lig_Turkey',
            ('liga 1', 'romania'): 'Liga_1_Romania', 
            ('liga i', 'romania'): 'Liga_1_Romania',
        }

        # Check for country-specific matches
        if country:
            for (pattern_name, pattern_country), replacement in standardization_map.items():
                if pattern_name in lower_name and pattern_country == lower_country:
                    logger.debug(f"Standardizing '{base_name}' to '{replacement}'")
                    return replacement

        # Default cleanup for the base name
        default_standardized = re.sub(r'[^\w\s-]', '', base_name)
        default_standardized = re.sub(r'[-\s]+', '_', default_standardized).strip('_')
        return default_standardized if default_standardized else "Unknown_League"
        
# Main execution block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Ensure OUTPUT_DIR exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")

    extractor = DailyGameExtractor(use_mongo=True)

    # Extract data for today
    today_date_str = extractor.get_current_date_str()
    logger.info(f"--- Running extraction for today ({today_date_str}) ---")
    daily_data = extractor.extract_games_for_date(today_date_str)

    if daily_data.get("error"):
        logger.error(f"Extraction failed: {daily_data['error']}")
    elif daily_data.get("total_games", 0) > 0:
        logger.info(f"--- Extraction Complete for {today_date_str} ---")
        logger.info(f"Total games processed: {daily_data['total_games']}")
        # Save league-specific summaries
        extractor.save_summary_file(daily_data)
    else:
        logger.info(f"--- Extraction completed for {today_date_str}, but no games were found or processed. ---")

