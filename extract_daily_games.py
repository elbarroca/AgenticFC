import os
import logging
import sqlite3
from datetime import date
import json
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
import re
import sys
import time

# Import MongoDB manager from existing code
try:
    from get_data.api_football.db_mongo import MongoDBManager
    # Import ID mappings for teams and leagues
    from get_data.db_ids.team_id_mappings import TEAM_ID_MAPPING
    from get_data.db_ids.league_id_mappings import LEAGUE_ID_MAPPING
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SQLITE_TIMEOUT = 30  # seconds
SQLITE_DB_PATH = 'get_data/statarea/statarea_stats.db'
OUTPUT_DIR = 'daily_games'

class StatAreaDBManager:
    """Manages access to the StatArea SQLite database."""
    
    def __init__(self):
        """Initialize the StatArea database manager and ensure tables exist."""
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database with required tables."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Create teams table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS teams (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    country TEXT NOT NULL,
                    last_scraped TEXT,
                    content_hash TEXT,
                    status TEXT DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    UNIQUE(name, country)
                )
                ''')
                
                # Create general_stats table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS general_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT,
                    game_type TEXT CHECK(game_type IN ('host', 'guest')),
                    period INTEGER CHECK(period IN (5, 10, 15)),
                    scrape_date TEXT,
                    stat_name TEXT,
                    stat_value TEXT,
                    FOREIGN KEY (team_id) REFERENCES teams(id),
                    UNIQUE(team_id, game_type, period, stat_name)
                )
                ''')
                
                # Create bet_stats table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS bet_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT,
                    game_type TEXT CHECK(game_type IN ('host', 'guest')),
                    period INTEGER CHECK(period IN (5, 10, 15)),
                    scrape_date TEXT,
                    category TEXT,
                    stat_name TEXT,
                    stat_value TEXT,
                    FOREIGN KEY (team_id) REFERENCES teams(id),
                    UNIQUE(team_id, game_type, period, category, stat_name)
                )
                ''')
                
                # Create match_history table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT,
                    game_type TEXT CHECK(game_type IN ('host', 'guest')),
                    match_date TEXT,
                    competition TEXT,
                    opponent TEXT,
                    team_goals INTEGER,
                    opponent_goals INTEGER,
                    result TEXT CHECK(result IN ('win', 'loss', 'draw')),
                    venue TEXT CHECK(venue IN ('home', 'away')),
                    scrape_date TEXT,
                    FOREIGN KEY (team_id) REFERENCES teams(id),
                    UNIQUE(team_id, match_date, opponent)
                )
                ''')
                
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_match_history_team ON match_history(team_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_match_history_date ON match_history(match_date)')
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize StatArea database: {e}")
            raise
    
    @contextmanager
    def get_db_connection(self):
        """Get a connection to the StatArea SQLite database."""
        conn = None
        try:
            conn = sqlite3.connect(
                SQLITE_DB_PATH,
                timeout=SQLITE_TIMEOUT,
                check_same_thread=False
            )
            yield conn
        except sqlite3.Error as e:
            logger.error(f"StatArea database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def get_team_stats(self, team_id: str) -> Dict[str, Any]:
        """
        Get comprehensive team statistics from the StatArea database.
        
        Args:
            team_id: StatArea team ID
            
        Returns:
            Dictionary containing team statistics
        """
        team_data = {"id": team_id, "stats": {}}
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get team basic info
                cursor.execute(
                    "SELECT name, country, last_scraped FROM teams WHERE id = ?", 
                    (team_id,)
                )
                team_info = cursor.fetchone()
                if not team_info:
                    logger.warning(f"No team found with ID {team_id} in StatArea database")
                    return team_data
                
                team_data["name"] = team_info[0]
                team_data["country"] = team_info[1]
                team_data["last_scraped"] = team_info[2]
                
                # Get general statistics
                cursor.execute("""
                    SELECT game_type, period, stat_name, stat_value 
                    FROM general_stats 
                    WHERE team_id = ? 
                    ORDER BY period DESC
                """, (team_id,))
                
                general_stats = cursor.fetchall()
                for game_type, period, stat_name, stat_value in general_stats:
                    key = f"{game_type}_{period}"
                    if key not in team_data["stats"]:
                        team_data["stats"][key] = {}
                    team_data["stats"][key][stat_name] = stat_value
                
                # Get betting statistics
                cursor.execute("""
                    SELECT game_type, period, category, stat_name, stat_value 
                    FROM bet_stats 
                    WHERE team_id = ? 
                    ORDER BY period DESC
                """, (team_id,))
                
                bet_stats = cursor.fetchall()
                for game_type, period, category, stat_name, stat_value in bet_stats:
                    key = f"{game_type}_{period}"
                    if key not in team_data["stats"]:
                        team_data["stats"][key] = {}
                    if category not in team_data["stats"][key]:
                        team_data["stats"][key][category] = {}
                    team_data["stats"][key][category][stat_name] = stat_value
                
                # Get recent match history
                cursor.execute("""
                    SELECT match_date, competition, opponent, team_goals, 
                           opponent_goals, result, venue
                    FROM match_history 
                    WHERE team_id = ? 
                    ORDER BY match_date DESC 
                    LIMIT 10
                """, (team_id,))
                
                matches = []
                match_history = cursor.fetchall()
                for match_date, competition, opponent, team_goals, opponent_goals, result, venue in match_history:
                    matches.append({
                        "date": match_date,
                        "competition": competition,
                        "opponent": opponent,
                        "team_goals": team_goals,
                        "opponent_goals": opponent_goals,
                        "result": result,
                        "venue": venue
                    })
                
                team_data["match_history"] = matches
                
                return team_data
                
        except sqlite3.Error as e:
            logger.error(f"Error fetching team stats for {team_id}: {str(e)}")
            return team_data
        
    def save_summary_file(self, data: Dict[str, Any], output_file: Optional[str] = None):
        """
        Save a summary of all extracted data to a JSON file.
        
        Args:
            data: Extracted game data
            output_file: Output file path (default: games_summary_YYYY-MM-DD.json)
        """
        if not output_file:
            date_str = data.get("date", self.get_current_date_str())
            output_file = f"games_summary_{date_str}.json"
        
        # Create a simplified summary with just the most essential information
        summary = {
            "date": data.get("date"),
            "total_games": data.get("total_games"),
            "games": []
        }
        
        for game in data.get("games", []):
            summary["games"].append({
                "fixture_id": game.get("fixture_id"),
                "home_team": {
                    "id": game.get("home_team", {}).get("id"),
                    "name": game.get("home_team", {}).get("name"),
                    "statarea_id": game.get("home_team", {}).get("statarea_id")
                },
                "away_team": {
                    "id": game.get("away_team", {}).get("id"),
                    "name": game.get("away_team", {}).get("name"),
                    "statarea_id": game.get("away_team", {}).get("statarea_id")
                },
                "league": {
                    "id": game.get("league", {}).get("id"),
                    "name": game.get("league", {}).get("name"),
                    "country": game.get("league", {}).get("country")
                },
                "kickoff_time": game.get("match_info", {}).get("date"),
                "file": f"{data.get('date')}_{self._sanitize_filename(game.get('home_team', {}).get('name', 'Home'))}_vs_{self._sanitize_filename(game.get('away_team', {}).get('name', 'Away'))}_{game.get('fixture_id')}.json"
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved summary data to {output_file}")


class DailyGameExtractor:
    """Extracts and merges game data from MongoDB and StatArea databases for a specific day."""
    
    def __init__(self, use_mongo=True):
        """Initialize the extractor with database connections."""
        self.statarea_db = StatAreaDBManager()
        self.mongo_db = None
        if use_mongo:
            try:
                self.mongo_db = MongoDBManager()
                logger.info("MongoDB connection successful")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                logger.warning("Continuing with StatArea DB only")
    
    def get_current_date_str(self) -> str:
        """Get today's date as a string in YYYY-MM-DD format."""
        return date.today().strftime("%Y-%m-%d")
    
    def extract_games_for_date(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract all game data for a specific date.
        
        Args:
            date_str: Date string in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary containing all the extracted game data
        """
        # Use today's date if none is provided
        if not date_str:
            date_str = self.get_current_date_str()
            
        logger.info(f"Extracting games data for {date_str}")
        
        detailed_games = []
        mongodb_connection_error = None
        
        # First attempt to get data from MongoDB
        if self.mongo_db:
            try:
                # Get daily games summary from MongoDB
                games_data = self.mongo_db.get_daily_games(date_str)
                
                if games_data:
                    # Get fixture IDs for the date
                    fixture_ids = self.mongo_db.get_match_fixture_ids_for_date(date_str)
                    logger.info(f"Found {len(fixture_ids)} fixtures for {date_str} in MongoDB")
                    
                    # Extract detailed game data for each fixture
                    for fixture_id in fixture_ids:
                        try:
                            # Get match data from MongoDB with retries
                            match_data = None
                            retry_count = 0
                            while match_data is None and retry_count < 3:
                                try:
                                    match_data = self.mongo_db.get_match_data(date_str, fixture_id)
                                    if not match_data and retry_count < 2:
                                        logger.warning(f"No match data found for fixture {fixture_id} (attempt {retry_count+1}), retrying...")
                                        time.sleep(1)  # Wait before retry
                                except Exception as e:
                                    logger.warning(f"Error fetching match data (attempt {retry_count+1}): {e}")
                                    if retry_count < 2:
                                        time.sleep(1)  # Wait before retry
                                retry_count += 1
                            
                            if match_data:
                                # Process match data and add StatArea data
                                logger.info(f"Processing match data for fixture {fixture_id}")
                                game_info = self.process_game_data(match_data)
                                detailed_games.append(game_info)
                                
                                # Save individual game file
                                self.save_individual_game_file(game_info)
                            else:
                                logger.warning(f"Failed to get detailed match data for fixture {fixture_id} after retries")
                        except Exception as e:
                            logger.error(f"Error processing fixture {fixture_id}: {e}")
                            continue  # Continue with next fixture
                else:
                    logger.warning(f"No games found in MongoDB for {date_str}")
            except Exception as e:
                mongodb_connection_error = str(e)
                logger.error(f"Error retrieving data from MongoDB: {e}")
                logger.warning("Falling back to StatArea data only")
        else:
            logger.warning("MongoDB connection not available, using StatArea data only")
        
        # If we have a MongoDB connection error, add it to the output
        result = {
            "date": date_str,
            "games": detailed_games,
            "total_games": len(detailed_games)
        }
        
        if mongodb_connection_error:
            result["mongodb_error"] = mongodb_connection_error
            
        return result
    
    def process_game_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process match data and enrich it with StatArea data and standings.
        
        Args:
            match_data: Raw match data from MongoDB
            
        Returns:
            Processed and enriched game data
        """
        # Keep existing code to enrich with StatArea data
        if "_id" in match_data:
            del match_data["_id"]
            
        self._add_statarea_id_mappings(match_data)
        self._add_statarea_team_data(match_data)
        
        # Add standings if needed
        if "league_standings" not in match_data and "standings" not in match_data:
            league_id = match_data.get("league", {}).get("id")
            season = match_data.get("league", {}).get("season")
            
            if league_id and season:
                standings = self.mongo_db.get_league_standings(league_id, season)
                if standings:
                    match_data["standings"] = standings
        
        return match_data  # Return the enriched data, cleaning will happen during save
    
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
            league_name = match_data["league"].get("name", "")
            
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
        
        # Map teams in H2H data if present (but don't fetch StatArea data for these)
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
    
    def _add_statarea_team_data(self, match_data: Dict[str, Any]):
        """
        Add StatArea team data to the match data.
        This modifies the match_data dictionary in place.
        
        Args:
            match_data: Match data to enrich with StatArea team data
        """
        # Add StatArea data for home team if we have a valid ID
        if "home_team" in match_data and "statarea_id" in match_data["home_team"] and match_data["home_team"]["statarea_id"] != "unknown":
            home_statarea_id = match_data["home_team"]["statarea_id"]
            home_statarea_data = self.statarea_db.get_team_stats(home_statarea_id)
            match_data["home_team"]["statarea_data"] = home_statarea_data
        
        # Add StatArea data for away team if we have a valid ID
        if "away_team" in match_data and "statarea_id" in match_data["away_team"] and match_data["away_team"]["statarea_id"] != "unknown":
            away_statarea_id = match_data["away_team"]["statarea_id"]
            away_statarea_data = self.statarea_db.get_team_stats(away_statarea_id)
            match_data["away_team"]["statarea_data"] = away_statarea_data
    
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

    def _process_team(self, team_data_full_match: Dict[str, Any], team_key: str, league_standings: Any):
        """
        Processes team data from the full match document, prioritizing MongoDB stats
        and adding unique StatArea insights.

        Args:
            team_data_full_match: The full match data dictionary from MongoDB.
            team_key: 'home_team' or 'away_team'.
            league_standings: The fetched league standings data.

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

        # Get standing info for this team if available
        team_standing = None
        if league_standings:
            standings_list = league_standings if isinstance(league_standings, list) else league_standings.get("standings", [])
            # Ensure standings_list is a list before iterating
            if isinstance(standings_list, list):
                for group in standings_list:
                     # Ensure group is a list before iterating
                    if isinstance(group, list):
                        for team_standing_entry in group:
                            if isinstance(team_standing_entry, dict) and team_standing_entry.get("team", {}).get("id") == team_id:
                                team_standing = team_standing_entry
                                break
                    if team_standing: break # Found it

        # --- Organize MongoDB Stats ---
        mongodb_stats_processed = {}

        # Form - Directly from the stats object
        mongodb_stats_processed["form"] = mongodb_stats_raw.get("form") # Use the main form string

        # Fixtures - Use the structure from MongoDB stats
        if "fixtures" in mongodb_stats_raw:
            mongodb_stats_processed["fixtures"] = mongodb_stats_raw.get("fixtures")

        # Goals - Use the structure from MongoDB stats
        if "goals" in mongodb_stats_raw:
            # You might want to clean/simplify this further if needed
            mongodb_stats_processed["goals"] = mongodb_stats_raw.get("goals")

        # Biggest - Use the structure from MongoDB stats
        if "biggest" in mongodb_stats_raw:
            mongodb_stats_processed["biggest"] = mongodb_stats_raw.get("biggest")

        # Performance Metrics - Use from MongoDB stats
        mongodb_stats_processed["performance"] = {
            "clean_sheet": mongodb_stats_raw.get("clean_sheet"),
            "failed_to_score": mongodb_stats_raw.get("failed_to_score"),
            "penalty": mongodb_stats_raw.get("penalty"),
            "lineups": mongodb_stats_raw.get("lineups"), # Formations played stats
            "cards": mongodb_stats_raw.get("cards"),
        }
        
        # --- Process StatArea Data ---
        statarea_data_raw = team_info.get("statarea_data") # This was added earlier
        statarea_analysis = {}
        if statarea_data_raw and isinstance(statarea_data_raw, dict):
            # Keep the ID and last_scraped info
            statarea_analysis["statarea_id"] = statarea_data_raw.get("id")
            statarea_analysis["last_scraped"] = statarea_data_raw.get("last_scraped")
            
            # Include ALL the raw StatArea stats without filtering
            statarea_analysis["raw_stats"] = statarea_data_raw.get("stats", {})
            
            # Include the complete match history without limiting to 5
            statarea_analysis["match_history"] = statarea_data_raw.get("match_history", [])
            
            # For backward compatibility, also keep the calculated analysis fields
            statarea_stats = statarea_data_raw.get("stats", {})
            is_home = (team_key == "home_team")
            
            for period in [15, 10, 5]:
                period_key = f"{'host' if is_home else 'guest'}_{period}"
                stats_for_period = statarea_stats.get(period_key, {})
                if stats_for_period:
                    analysis_key = f"analysis_{period}_games"
                    statarea_analysis[analysis_key] = {
                        "chance_to_score": stats_for_period.get("Chance to score goal next match"),
                        "chance_to_concede": stats_for_period.get("Chance to conceded goal next match"),
                    }
                    # Remove None values from this specific period analysis
                    statarea_analysis[analysis_key] = {k: v for k, v in statarea_analysis[analysis_key].items() if v is not None}


        # --- Final Team Structure ---
        processed_team = {
            "id": team_id,
            "name": team_name,
            "logo": team_info.get("logo"),
            "winner": team_info.get("winner"), # From the top-level team object if present
            # Coach/Formation might be in the main team object OR under lineups
            "coach": team_info.get("coach") or team_info.get("lineups", [{}])[0].get("coach"), # Check lineups too
            "formation": team_info.get("formation") or team_info.get("lineups", [{}])[0].get("formation"), # Check lineups too
            "standing": team_standing,
            "mongodb_stats": self._remove_none_values(mongodb_stats_processed), # Cleaned MongoDB core stats
            "statarea_analysis": self._remove_none_values(statarea_analysis) # Cleaned unique StatArea insights
        }
        return self._remove_none_values(processed_team)


    def _clean_final_game_data(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cleans and organizes the final game data before saving, prioritizing MongoDB
        and integrating unique StatArea data.
        """
        cleaned = {}

        # 1. Fixture Info (from MongoDB match_info or fixture)
        # Adjust based on whether 'fixture' or 'match_info' holds the primary data
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

        # 2. League Info & Standings (Prioritize MongoDB 'standings' or 'standings_snapshot')
        league_info = game_data.get("league", {}) # Base league info might be separate
        # Find the most complete standings object
        standings_data = game_data.get("standings", game_data.get("standings_snapshot", {}))

        cleaned["league"] = {
            "id": league_info.get("id") or standings_data.get("league", {}).get("id") or game_data.get("league_id"),
            "name": league_info.get("name") or standings_data.get("league", {}).get("name") or game_data.get("league_name"),
            "country": league_info.get("country") or standings_data.get("league", {}).get("country"),
            "logo": league_info.get("logo") or standings_data.get("league", {}).get("logo"),
            "flag": league_info.get("flag") or standings_data.get("league", {}).get("flag"),
            "season": league_info.get("season") or standings_data.get("league", {}).get("season") or standings_data.get("season"),
            "round": league_info.get("round"), # May not be in standings object
            "statarea_id": league_info.get("statarea_id"), # Added earlier
            # Include the actual standings data, preferring the more detailed one
            "standings": standings_data.get("standings")
        }

        # 3. Teams Info (Use the updated _process_team)
        # Pass the full game_data so _process_team can access team_stats keys
        cleaned["teams"] = {
            "home": self._process_team(game_data, "home_team", cleaned["league"].get("standings")),
            "away": self._process_team(game_data, "away_team", cleaned["league"].get("standings")),
        }

        # 4. Score and Goals (from MongoDB top level)
        cleaned["score"] = game_data.get("score") # Contains halftime, fulltime etc.
        cleaned["goals"] = game_data.get("goals") # Contains current home/away goals

        # 5. Match Events, Lineups, Statistics (if present at match level in MongoDB)
        if game_data.get("events"):
            cleaned["events"] = game_data.get("events")
        if game_data.get("lineups"): # Match specific lineups
            cleaned["lineups"] = game_data.get("lineups")
        if game_data.get("statistics"): # Match specific stats
            cleaned["statistics"] = game_data.get("statistics")

        # 6. Predictions (Process MongoDB predictions object)
        predictions_raw = game_data.get("predictions", {})
        # Handle potential nesting (e.g., predictions within predictions)
        pred_content = predictions_raw.get("predictions", predictions_raw) # Use inner obj if exists, else outer

        cleaned["predictions"] = {
            "winner": pred_content.get("winner"),
            "win_or_draw": pred_content.get("win_or_draw"),
            "advice": pred_content.get("advice"),
            "percent": pred_content.get("percent"),
            "under_over_prediction": pred_content.get("under_over"), # Renamed to avoid clash if any
            "goals_prediction": pred_content.get("goals"), # Predicted goals
            # Comparison data might be at the top level of predictions_raw
            "comparison": predictions_raw.get("comparison"),
            # Other potential fields from the example
             "h2h_prediction_details": pred_content.get("h2h"), # Might contain form/att/def stats used for prediction h2h part
             "winning_odds": predictions_raw.get("winning_odds"), # Example field
             # Add others as identified from your specific MongoDB structure
             # "league_position": predictions_raw.get("league_position"), # Example
             # "attacks": predictions_raw.get("attacks"),             # Example
             # "defenses": predictions_raw.get("defenses"),           # Example
             # "poisson_distribution": predictions_raw.get("poisson_distribution") # Example
        }

        # 7. H2H Data (Prioritize from MongoDB predictions.h2h)
        h2h_raw = []
        if isinstance(predictions_raw.get("h2h"), list): # Check if h2h is a list directly under predictions
             h2h_raw = predictions_raw["h2h"]
        elif isinstance(predictions_raw.get("h2h"), dict): # Check if it's dict containing matches (like example)
             h2h_raw = predictions_raw.get("h2h", {}).get("matches", [])

        if not h2h_raw and isinstance(game_data.get("h2h"), list): # Fallback to top-level h2h if needed
             h2h_raw = game_data["h2h"]

        h2h_list = []
        # Use the existing cleaning logic structure but apply to h2h_raw source
        for match in h2h_raw[:10]: # Limit to 10 most recent
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
                    # Add statarea ID if mapped earlier, but maybe not essential here
                    # "statarea_id": h2h_home.get("statarea_id")
                },
                "away_team": {
                    "id": h2h_away.get("id"),
                    "name": h2h_away.get("name"),
                    "logo": h2h_away.get("logo"),
                    "winner": h2h_away.get("winner"),
                    # "statarea_id": h2h_away.get("statarea_id")
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

        # 8. Final Cleanup & Parameter Summary
        cleaned = self._remove_none_values(cleaned)
        # Regenerate summary based on the *new* structure
        parameter_counts = self._count_parameters_by_category(cleaned) # Needs update to reflect new structure
        cleaned["parameter_summary"] = parameter_counts

        return cleaned

    def _count_parameters_by_category(self, data_dict: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Counts parameters by category in the *cleaned* data structure.
        Needs adjustment to match the new structure from _clean_final_game_data.
        """
        categories = {
            "fixture": [],
            "league": [],
            "team_shared": [], # Basic team info like id, name, logo
            "home_standing": [],
            "away_standing": [],
            "home_mongodb_stats": [],
            "away_mongodb_stats": [],
            "home_statarea_analysis": [],
            "away_statarea_analysis": [],
            "score_goals": [],
            "predictions": [],
            "h2h": [],
            "events": [],
            "lineups": [],
            "match_stats": [],
            "other": []
        }

        # Extract fixture info parameters
        if "fixture_info" in data_dict:
            self._extract_parameters(data_dict["fixture_info"], "fixture", categories["fixture"])

        # Extract league parameters (excluding standings list itself)
        if "league" in data_dict:
            league_data = data_dict["league"]
            for key in league_data.keys():
                if key != "standings":
                    categories["league"].append(f"league.{key}")
            if league_data.get("standings"):
                 categories["league"].append("league.standings_present")


        # Extract score/goals parameters
        if "score" in data_dict:
             self._extract_parameters(data_dict["score"], "score", categories["score_goals"])
        if "goals" in data_dict:
             self._extract_parameters(data_dict["goals"], "goals", categories["score_goals"])


        # --- Team Data ---
        home_team = data_dict.get("teams", {}).get("home")
        away_team = data_dict.get("teams", {}).get("away")

        # Home Team
        if home_team:
            # Basic Info + Shared Keys
            for key in home_team.keys():
                 if key not in ["standing", "mongodb_stats", "statarea_analysis"]:
                    categories["team_shared"].append(f"team.{key}") # Add shared keys once
            # Standing
            if "standing" in home_team:
                self._extract_parameters(home_team["standing"], "home.standing", categories["home_standing"])
            # MongoDB Stats
            if "mongodb_stats" in home_team:
                self._extract_parameters(home_team["mongodb_stats"], "home.mongodb_stats", categories["home_mongodb_stats"])
            # StatArea Analysis
            if "statarea_analysis" in home_team:
                self._extract_parameters(home_team["statarea_analysis"], "home.statarea_analysis", categories["home_statarea_analysis"])

        # Away Team (only non-shared keys)
        if away_team:
             # Standing
            if "standing" in away_team:
                self._extract_parameters(away_team["standing"], "away.standing", categories["away_standing"])
            # MongoDB Stats
            if "mongodb_stats" in away_team:
                self._extract_parameters(away_team["mongodb_stats"], "away.mongodb_stats", categories["away_mongodb_stats"])
            # StatArea Analysis
            if "statarea_analysis" in away_team:
                self._extract_parameters(away_team["statarea_analysis"], "away.statarea_analysis", categories["away_statarea_analysis"])

        # Remove duplicates from shared keys
        categories["team_shared"] = sorted(list(set(categories["team_shared"])))


        # Extract prediction parameters
        if "predictions" in data_dict:
             self._extract_parameters(data_dict["predictions"], "predictions", categories["predictions"])

        # Count H2H parameters (just the count + keys of first entry)
        if "h2h" in data_dict and data_dict["h2h"]:
            categories["h2h"].append(f"h2h.count: {len(data_dict['h2h'])}")
            self._extract_parameters(data_dict['h2h'][0], "h2h_entry", categories["h2h"]) # Keys from first match

        # Count events parameters (count + keys of first entry)
        if "events" in data_dict and data_dict["events"]:
            categories["events"].append(f"events.count: {len(data_dict['events'])}")
            self._extract_parameters(data_dict['events'][0], "event_entry", categories["events"])

        # Count lineups parameters (count + keys of first entry)
        if "lineups" in data_dict and data_dict["lineups"]:
            categories["lineups"].append(f"lineups.count: {len(data_dict['lineups'])}")
            self._extract_parameters(data_dict['lineups'][0], "lineup_entry", categories["lineups"])

        # Count match stats parameters (count + keys of first entry)
        if "statistics" in data_dict and data_dict["statistics"]:
            categories["match_stats"].append(f"match_stats.count: {len(data_dict['statistics'])}")
            self._extract_parameters(data_dict['statistics'][0], "match_stat_entry", categories["match_stats"])


        # Return only non-empty categories
        return {k: sorted(list(set(v))) for k, v in categories.items() if v}

    # Make sure _extract_parameters handles non-dict values gracefully in recursion base case
    def _extract_parameters(self, data, prefix, param_list):
        """
        Helper method to recursively extract parameters from nested dictionaries/lists.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                param_name = f"{prefix}.{key}"
                if isinstance(value, (dict, list)):
                    # Recurse for nested structures
                    self._extract_parameters(value, param_name, param_list)
                else:
                    # Add the parameter name for simple values
                    param_list.append(param_name)
        elif isinstance(data, list):
             # If it's a list, maybe just indicate its presence or process the first item as sample
            if data:
                 # Example: Process first item to get structure keys
                 self._extract_parameters(data[0], f"{prefix}[0]", param_list)
        # Base case: If it's not a dict or list, do nothing (already handled by caller)


    def save_individual_game_file(self, game_data: Dict[str, Any]):
        """
        Save cleaned game data to an individual JSON file named after the teams.

        Args:
            game_data: Raw game data after initial processing and enrichment
        """
        # Create directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # --- Get necessary info for filename ---
        # Need to access potentially cleaned data OR the original enriched data
        # Let's clean FIRST, then get names/ids for the filename from cleaned data

        # Clean the data *before* determining the filename components
        cleaned_data = self._clean_final_game_data(game_data) # Perform cleaning

        # --- Determine filename components from CLEANED data ---
        home_team_info = cleaned_data.get("teams", {}).get("home", {})
        away_team_info = cleaned_data.get("teams", {}).get("away", {})
        fixture_info = cleaned_data.get("fixture_info", {})

        home_team_name = home_team_info.get("name", "UnknownHome")
        away_team_name = away_team_info.get("name", "UnknownAway")
        home_team_file_part = self._sanitize_filename(home_team_name)
        away_team_file_part = self._sanitize_filename(away_team_name)

        date_str = fixture_info.get("date", self.get_current_date_str())
        if date_str and isinstance(date_str, str):
            date_part = date_str.split('T')[0]
        else:
            date_part = self.get_current_date_str()

        fixture_id = fixture_info.get("id", "unknown_fixture")

        # Create filename
        filename = f"{date_part}_{home_team_file_part}_vs_{away_team_file_part}_{fixture_id}.json"
        # Ensure OUTPUT_DIR is defined correctly
        output_dir = OUTPUT_DIR # Make sure this constant is accessible or passed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, filename)


        # Save the CLEANED game data to file
        try:
            cleaned_data = self._convert_mongodb_types(cleaned_data)
            with open(file_path, 'w', encoding='utf-8') as f:
                # Use a custom encoder if you encounter non-serializable types (like $numberInt)
                # Or ensure data is converted before this step
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved cleaned game data to {file_path}")
        except TypeError as e:
             logger.error(f"Error saving JSON data to {file_path}: {e}. Check for non-serializable types.")
             # Consider logging problematic part of cleaned_data if possible
        except Exception as e:
             logger.error(f"An unexpected error occurred saving {file_path}: {e}")


    def save_summary_file(self, data: Dict[str, Any], output_file: Optional[str] = None):
        """
        Save a summary of all extracted data to a JSON file.
        Uses the *cleaned* data for consistency.

        Args:
            data: The dictionary returned by extract_games_for_date, containing raw game entries.
            output_file: Output file path (default: games_summary_YYYY-MM-DD.json)
        """
        # ... (output_file determination logic) ...

        summary = {
            "date": data.get("date"),
            "total_games": data.get("total_games"),
            "games": []
        }

        for raw_game in data.get("games", []):
            # Clean each game *first* to get consistent info for the summary
            cleaned_game = self._clean_final_game_data(raw_game) # Clean the game data

            # --- Extract info from CLEANED data ---
            fixture_info = cleaned_game.get("fixture_info", {})
            league_info = cleaned_game.get("league", {})
            home_team_info = cleaned_game.get("teams", {}).get("home", {})
            away_team_info = cleaned_game.get("teams", {}).get("away", {})

            # Construct filename based on cleaned data (same logic as save_individual_game_file)
            home_name = home_team_info.get("name", "UnknownHome")
            away_name = away_team_info.get("name", "UnknownAway")
            home_file_part = self._sanitize_filename(home_name)
            away_file_part = self._sanitize_filename(away_name)
            fixture_id = fixture_info.get("id", "unknown_fixture")
            date_str = fixture_info.get("date", self.get_current_date_str())
            date_part = date_str.split('T')[0] if date_str and isinstance(date_str, str) else self.get_current_date_str()
            filename = f"{date_part}_{home_file_part}_vs_{away_file_part}_{fixture_id}.json"


            summary["games"].append({
                "fixture_id": fixture_id,
                "kickoff_time": fixture_info.get("date"), # Use cleaned kickoff time
                "league": {
                    "id": league_info.get("id"),
                    "name": league_info.get("name"),
                    "country": league_info.get("country")
                },
                "home_team": {
                    "id": home_team_info.get("id"),
                    "name": home_name,
                    "statarea_id": home_team_info.get("statarea_analysis", {}).get("statarea_id") # Get from analysis section
                },
                "away_team": {
                    "id": away_team_info.get("id"),
                    "name": away_name,
                    "statarea_id": away_team_info.get("statarea_analysis", {}).get("statarea_id") # Get from analysis section
                },
                "file": filename # Use the constructed filename
            })

        # ... (saving logic with error handling) ...
        try:
            # Ensure OUTPUT_DIR is handled correctly if summary is saved elsewhere
            output_dir = os.path.dirname(output_file) if output_file else "."
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir)
                 
            actual_output_file = output_file if output_file else f"games_summary_{data.get('date', self.get_current_date_str())}.json"

            with open(actual_output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved summary data to {actual_output_file}")
        except TypeError as e:
             logger.error(f"Error saving JSON summary data to {actual_output_file}: {e}. Check for non-serializable types.")
        except Exception as e:
             logger.error(f"An unexpected error occurred saving summary {actual_output_file}: {e}")

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize a string to be used as part of a filename.
        
        Args:
            name: String to sanitize
            
        Returns:
            Sanitized string
        """
        # Remove special characters and replace spaces with underscores
        return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')

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


def main():
    """Main function to extract game data for a specific day."""
    import argparse
    parser = argparse.ArgumentParser(description='Extract game data from MongoDB and StatArea for a specific day')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--output', type=str, help='Output summary file path')
    parser.add_argument('--no-mongo', action='store_true', help='Skip MongoDB and use StatArea DB only')
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = DailyGameExtractor(use_mongo=not args.no_mongo)
        
        # Extract data
        data = extractor.extract_games_for_date(args.date)
        
        if data['total_games'] > 0:
            # Save summary file
            extractor.save_summary_file(data, args.output)
            
            print(f"Successfully extracted {data['total_games']} games for {data['date']}")
            print(f"Individual game files saved to the '{OUTPUT_DIR}' directory")
        else:
            print(f"No games extracted for {data['date']}")
        
    except Exception as e:
        logger.error(f"Error extracting game data: {e}")
        raise
    finally:
        # Close MongoDB connection if it exists
        if extractor.mongo_db:
            try:
                extractor.mongo_db.close_connection()
            except:
                pass  # Ignore errors during cleanup

if __name__ == "__main__":
    main()