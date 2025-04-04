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
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'statarea_stats.db')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'daily_output', 'daily_games')

class StatAreaDBManager:
    """Manages access to the StatArea SQLite database."""
    
    def __init__(self, db_path=None):
        """
        Initialize the StatArea database manager and ensure tables exist.
        
        Args:
            db_path: Optional custom path to the database file
        """
        self.db_path = db_path or SQLITE_DB_PATH
        logger.info(f"StatArea database path set to: {self.db_path}")
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
                self.db_path,  # Use instance variable instead of constant
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
            # First print the database path for debugging
            logger.info(f"Retrieving team stats for ID {team_id} from database: {self.db_path}")
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Count total teams for debugging
                cursor.execute("SELECT COUNT(*) FROM teams")
                total_teams = cursor.fetchone()[0]
                logger.info(f"Total teams in StatArea database: {total_teams}")
                
                # Show sample team IDs for debugging
                cursor.execute("SELECT id, name, country FROM teams LIMIT 5")
                sample_teams = cursor.fetchall()
                teams_str = ", ".join([f"{t[0]}({t[1]})" for t in sample_teams])
                logger.info(f"Sample teams in database: {teams_str}")
                
                # Get team basic info
                cursor.execute(
                    "SELECT name, country, last_scraped FROM teams WHERE id = ?", 
                    (team_id,)
                )
                team_info = cursor.fetchone()
                if not team_info:
                    logger.warning(f"No team found with ID {team_id} in StatArea database")
                    # Try to find team by partial ID match for debugging
                    cursor.execute("SELECT id, name, country FROM teams WHERE id LIKE ?", (f"%{team_id}%",))
                    similar_ids = cursor.fetchall()
                    if similar_ids:
                        logger.info(f"Found similar IDs: {similar_ids}")
                    
                    # Try to find teams with similar names
                    cursor.execute("SELECT id, name, country FROM teams WHERE name LIKE ?", (f"%{team_id}%",))
                    similar_names = cursor.fetchall()
                    if similar_names:
                        logger.info(f"Found teams with similar names: {similar_names}")
                        
                    return team_data
                
                team_data["name"] = team_info[0]
                team_data["country"] = team_info[1]
                team_data["last_scraped"] = team_info[2]
                
                logger.info(f"Found team {team_info[0]} (ID: {team_id}) from {team_info[1]}, last scraped: {team_info[2]}")
                
                # Get general statistics
                cursor.execute("""
                    SELECT game_type, period, stat_name, stat_value 
                    FROM general_stats 
                    WHERE team_id = ? 
                    ORDER BY period DESC
                """, (team_id,))
                
                general_stats = cursor.fetchall()
                general_count = len(general_stats)
                logger.info(f"Found {general_count} general stats records for team ID {team_id}")
                
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
                bet_count = len(bet_stats)
                logger.info(f"Found {bet_count} betting stats records for team ID {team_id}")
                
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
                match_count = len(match_history)
                logger.info(f"Found {match_count} match history records for team ID {team_id}")
                
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
                
                # If we didn't find any data, log a warning
                if general_count == 0 and bet_count == 0 and match_count == 0:
                    logger.warning(f"Team ID {team_id} exists in database but has no stats data")
                
                return team_data
                
        except sqlite3.Error as e:
            logger.error(f"Error fetching team stats for {team_id}: {str(e)}")
            return team_data
    
    def list_all_teams(self) -> List[Dict[str, Any]]:
        """
        Get a list of all teams in the StatArea database.
        
        Returns:
            List of team dictionaries with id, name and country
        """
        teams = []
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, country, last_scraped FROM teams")
                for row in cursor.fetchall():
                    teams.append({
                        "id": row[0],
                        "name": row[1],
                        "country": row[2],
                        "last_scraped": row[3]
                    })
                logger.info(f"Found {len(teams)} teams in StatArea database")
                return teams
        except sqlite3.Error as e:
            logger.error(f"Error listing teams: {str(e)}")
            return []
    
    def save_summary_file(self, data: Dict[str, Any], output_file: Optional[str] = None):
        """
        Save a summary of all extracted data to a JSON file in each league directory.
        """
        # Group games by league
        games_by_league = {}
        
        for raw_game in data.get("games", []):
            cleaned_game = self._clean_final_game_data(raw_game)
            league_info = cleaned_game.get("league", {})
            league_name = league_info.get("name", "UnknownLeague")
            league_country = league_info.get("country", "")
            
            # Create league key
            league_dir_name = self._sanitize_filename(f"{league_name}_{league_country}" if league_country else league_name)
            
            if league_dir_name not in games_by_league:
                games_by_league[league_dir_name] = {
                    "league_info": league_info,
                    "games": []
                }
            
            # Add game to league group
            fixture_info = cleaned_game.get("fixture_info", {})
            home_team_info = cleaned_game.get("teams", {}).get("home", {})
            away_team_info = cleaned_game.get("teams", {}).get("away", {})
            
            games_by_league[league_dir_name]["games"].append({
                "fixture_id": fixture_info.get("id"),
                "kickoff_time": fixture_info.get("date"),
                "home_team": {
                    "id": home_team_info.get("id"),
                    "name": home_team_info.get("name"),
                    "statarea_id": home_team_info.get("statarea_analysis", {}).get("statarea_id")
                },
                "away_team": {
                    "id": away_team_info.get("id"),
                    "name": away_team_info.get("name"),
                    "statarea_id": away_team_info.get("statarea_analysis", {}).get("statarea_id")
                }
            })
        
        # Save summary file for each league
        date_str = data.get("date", self.get_current_date_str())
        
        for league_dir_name, league_data in games_by_league.items():
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
                "total_games": len(league_data["games"]),
                "games": league_data["games"]
            }
            
            try:
                with open(summary_file_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved summary for {league_dir_name} to {summary_file_path}")
            except Exception as e:
                logger.error(f"Failed to save summary for {league_dir_name} to {summary_file_path}: {e}")


class DailyGameExtractor:
    """Extracts and merges game data from MongoDB and StatArea databases for a specific day."""
    
    def __init__(self, use_mongo=True):
        """Initialize the extractor with database connections."""
        self.statarea_db = StatAreaDBManager()
        self.mongo_db = None
        self.league_standings_map = {} # Added to store standings { (league_id, season): standings_data }
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
        
        # Save standings files after processing all games
        if self.mongo_db and self.league_standings_map:
             self._save_standings_files(date_str)

        # Clear the standings map for the next potential run
        self.league_standings_map = {}
            
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
        
        # Add correct league ID and season extraction - modified code here
        # Look in multiple possible locations for league ID and season
        league_id = None
        season = None
        
        # Check direct league object
        if "league" in match_data:
            league_obj = match_data.get("league", {})
            league_id = league_obj.get("id")
            season = league_obj.get("season")
        
        # If not found, check fixture_info -> league
        if (not league_id or not season) and "fixture_info" in match_data:
            fixture_info = match_data.get("fixture_info", {})
            if "league" in fixture_info:
                league_obj = fixture_info.get("league", {})
                league_id = league_obj.get("id")
                season = league_obj.get("season")
        
        # If still not found, check league_id field directly
        if not league_id:
            league_id = match_data.get("league_id")
        
        # If season still not found, try to use current year
        if not season:
            season = match_data.get("season")
            if not season:
                # Default to current year if all else fails
                from datetime import datetime
                season = datetime.now().year
        
        # Log the extracted values for debugging
        logger.info(f"Extracted league ID: {league_id}, season: {season} for match data")
        
        # Only proceed if we have valid league ID and season
        if league_id and season:
            league_key = (league_id, season)
            
            if league_key not in self.league_standings_map:
                # Check if standings are already in the match data
                existing_standings = match_data.get("standings")
                if not existing_standings:
                    existing_standings = match_data.get("standings_snapshot")
                
                if existing_standings:
                    logger.info(f"Using existing standings snapshot for league {league_id}, season {season}")
                    # Ensure it has the necessary league info for saving later
                    if "league" not in existing_standings:
                        existing_standings["league"] = match_data.get("league", {})
                    self.league_standings_map[league_key] = existing_standings
                elif self.mongo_db:
                    # Try to fetch standings from MongoDB directly using league ID
                    logger.info(f"Fetching standings for league {league_id}, season {season}")
                    standings = self.mongo_db.get_league_standings(league_id, season)
                    if standings:
                        # Store fetched standings in the map
                        self.league_standings_map[league_key] = standings
                        logger.info(f"✅ Successfully stored standings for league {league_id}, season {season}")
                    else:
                        # Try to fetch from current date if latest not found
                        current_date = match_data.get("date", self.get_current_date_str())
                        logger.info(f"Trying to fetch standings for date {current_date}")
                        standings = self.mongo_db.get_standings_data(current_date, league_id, season)
                        if standings:
                            self.league_standings_map[league_key] = standings
                            logger.info(f"✅ Successfully stored standings for league {league_id} using current date")
                        else:
                            logger.warning(f"Could not fetch standings for league {league_id}, season {season}")
            else:
                 logger.debug(f"Standings for league {league_id}, season {season} already processed.")
        else:
            logger.warning(f"Missing league ID ({league_id}) or season ({season}) for match, can't fetch standings")

        # Remove standings from the game data itself before returning
        if "standings" in match_data:
            del match_data["standings"]
        if "standings_snapshot" in match_data:
            del match_data["standings_snapshot"]
        
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
        # First print all mapped IDs for debugging
        if "home_team" in match_data and "id" in match_data["home_team"] and "away_team" in match_data and "id" in match_data["away_team"]:
            home_id = str(match_data["home_team"]["id"])
            away_id = str(match_data["away_team"]["id"])
            home_name = match_data["home_team"].get("name", "")
            away_name = match_data["away_team"].get("name", "")
            home_statarea_id = match_data["home_team"].get("statarea_id", "unknown")
            away_statarea_id = match_data["away_team"].get("statarea_id", "unknown")
            
            logger.info(f"Match: {home_name} vs {away_name}")
            logger.info(f"Home team: MongoDB ID={home_id}, StatArea ID={home_statarea_id}")
            logger.info(f"Away team: MongoDB ID={away_id}, StatArea ID={away_statarea_id}")
            
            # Verify the mappings against the TEAM_ID_MAPPING dictionary
            home_found = False
            away_found = False
            for team_name, mapping in TEAM_ID_MAPPING.items():
                if mapping.get("mongodb_id") == home_id:
                    expected_id = mapping.get("statarea_id")
                    home_found = True
                    if expected_id != home_statarea_id:
                        logger.warning(f"Home team statarea_id mismatch: expected {expected_id}, got {home_statarea_id}")
                    break
            
            for team_name, mapping in TEAM_ID_MAPPING.items():
                if mapping.get("mongodb_id") == away_id:
                    expected_id = mapping.get("statarea_id")
                    away_found = True
                    if expected_id != away_statarea_id:
                        logger.warning(f"Away team statarea_id mismatch: expected {expected_id}, got {away_statarea_id}")
                    break
                    
            if not home_found:
                logger.warning(f"Home team with MongoDB ID {home_id} not found in TEAM_ID_MAPPING")
            if not away_found:
                logger.warning(f"Away team with MongoDB ID {away_id} not found in TEAM_ID_MAPPING")
        
        # Add StatArea data for home team if we have a valid ID
        if "home_team" in match_data and "statarea_id" in match_data["home_team"] and match_data["home_team"]["statarea_id"] != "unknown":
            home_statarea_id = match_data["home_team"]["statarea_id"]
            home_team_name = match_data["home_team"].get("name", "")
            
            logger.info(f"Fetching StatArea data for home team: {home_team_name} (ID: {home_statarea_id})")
            
            # First check if team exists in database
            team_exists = self._check_team_exists_in_db(home_statarea_id)
            if not team_exists:
                logger.warning(f"Home team ID {home_statarea_id} ('{home_team_name}') not found in StatArea database")
                # Try to find by team name
                alt_id = self._find_team_by_name(home_team_name)
                if alt_id:
                    logger.info(f"Found alternative ID {alt_id} for '{home_team_name}' by name lookup")
                    home_statarea_id = alt_id
                    match_data["home_team"]["statarea_id"] = alt_id
                    match_data["home_team"]["statarea_id_by_name"] = True
            
            # Get team stats with potentially updated ID
            home_statarea_data = self.statarea_db.get_team_stats(home_statarea_id)
            
            # Verify data quality
            stats_populated = bool(home_statarea_data.get("stats") and any(home_statarea_data["stats"].values()))
            history_populated = bool(home_statarea_data.get("match_history"))
            
            if stats_populated and history_populated:
                logger.info(f"Retrieved complete StatArea data for home team {home_team_name}")
            elif stats_populated or history_populated:
                logger.warning(f"Retrieved partial StatArea data for home team {home_team_name}")
            else:
                logger.warning(f"Retrieved empty StatArea data for home team {home_team_name}")
                
            match_data["home_team"]["statarea_data"] = home_statarea_data
        
        # Add StatArea data for away team if we have a valid ID
        if "away_team" in match_data and "statarea_id" in match_data["away_team"] and match_data["away_team"]["statarea_id"] != "unknown":
            away_statarea_id = match_data["away_team"]["statarea_id"]
            away_team_name = match_data["away_team"].get("name", "")
            
            logger.info(f"Fetching StatArea data for away team: {away_team_name} (ID: {away_statarea_id})")
            
            # First check if team exists in database
            team_exists = self._check_team_exists_in_db(away_statarea_id)
            if not team_exists:
                logger.warning(f"Away team ID {away_statarea_id} ('{away_team_name}') not found in StatArea database")
                # Try to find by team name
                alt_id = self._find_team_by_name(away_team_name)
                if alt_id:
                    logger.info(f"Found alternative ID {alt_id} for '{away_team_name}' by name lookup")
                    away_statarea_id = alt_id
                    match_data["away_team"]["statarea_id"] = alt_id
                    match_data["away_team"]["statarea_id_by_name"] = True
            
            # Get team stats with potentially updated ID
            away_statarea_data = self.statarea_db.get_team_stats(away_statarea_id)
            
            # Verify data quality
            stats_populated = bool(away_statarea_data.get("stats") and any(away_statarea_data["stats"].values()))
            history_populated = bool(away_statarea_data.get("match_history"))
            
            if stats_populated and history_populated:
                logger.info(f"Retrieved complete StatArea data for away team {away_team_name}")
            elif stats_populated or history_populated:
                logger.warning(f"Retrieved partial StatArea data for away team {away_team_name}")
            else:
                logger.warning(f"Retrieved empty StatArea data for away team {away_team_name}")
                
            match_data["away_team"]["statarea_data"] = away_statarea_data
    
    def _check_team_exists_in_db(self, team_id: str) -> bool:
        """
        Check if a team exists in the StatArea database.
        
        Args:
            team_id: StatArea team ID
            
        Returns:
            True if team exists, False otherwise
        """
        try:
            with self.statarea_db.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM teams WHERE id = ?", (team_id,))
                count = cursor.fetchone()[0]
                return count > 0
        except sqlite3.Error as e:
            logger.error(f"Error checking if team {team_id} exists: {str(e)}")
            return False
    
    def _find_team_by_name(self, team_name: str) -> Optional[str]:
        """
        Find a team ID by name in the StatArea database.
        
        Args:
            team_name: Team name to search for
            
        Returns:
            Team ID if found, None otherwise
        """
        try:
            with self.statarea_db.get_db_connection() as conn:
                cursor = conn.cursor()
                # Try exact match first
                cursor.execute("SELECT id FROM teams WHERE name = ?", (team_name,))
                result = cursor.fetchone()
                if result:
                    return result[0]
                
                # Try LIKE match
                cursor.execute("SELECT id, name FROM teams WHERE name LIKE ?", (f"%{team_name}%",))
                results = cursor.fetchall()
                if results:
                    # Log possible matches
                    logger.info(f"Found {len(results)} possible matches for '{team_name}':")
                    for team_id, name in results:
                        logger.info(f"  - {name} (ID: {team_id})")
                    # Return the first match
                    return results[0][0]
                
                return None
        except sqlite3.Error as e:
            logger.error(f"Error finding team by name '{team_name}': {str(e)}")
            return None
    
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

    def _process_team(self, team_data_full_match: Dict[str, Any], team_key: str):
        """
        Processes team data from the full match document, prioritizing MongoDB stats
        and adding unique StatArea insights.

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
        statarea_data_status = "missing"
        
        if statarea_data_raw and isinstance(statarea_data_raw, dict):
            # Check if we have any actual stats data in the statarea data
            has_stats = bool(statarea_data_raw.get("stats") and any(statarea_data_raw["stats"].values()))
            has_match_history = bool(statarea_data_raw.get("match_history"))
            
            # Set status based on data quality
            if has_stats and has_match_history:
                statarea_data_status = "complete"
            elif has_stats or has_match_history:
                statarea_data_status = "partial"
            else:
                statarea_data_status = "empty"
                
            # Keep the ID and last_scraped info
            statarea_analysis["statarea_id"] = statarea_data_raw.get("id")
            statarea_analysis["last_scraped"] = statarea_data_raw.get("last_scraped")
            statarea_analysis["status"] = statarea_data_status
            
            # Include if the ID was found by name lookup instead of direct mapping
            if team_info.get("statarea_id_by_name"):
                statarea_analysis["id_by_name_lookup"] = True
            
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
        else:
            # Include information about missing data
            statarea_id = team_info.get("statarea_id", "unknown")
            statarea_analysis["statarea_id"] = statarea_id
            statarea_analysis["status"] = statarea_data_status
            statarea_analysis["error"] = f"No StatArea data found for ID: {statarea_id}"


        # --- Final Team Structure ---
        processed_team = {
            "id": team_id,
            "name": team_name,
            "logo": team_info.get("logo"),
            "winner": team_info.get("winner"), # From the top-level team object if present
            # Coach/Formation might be in the main team object OR under lineups
            "coach": team_info.get("coach") or team_info.get("lineups", [{}])[0].get("coach"), # Check lineups too
            "formation": team_info.get("formation") or team_info.get("lineups", [{}])[0].get("formation"), # Check lineups too
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
            "id": league_info.get("id") or game_data.get("league_id"), # Simplified: Get ID from league info or game_data
            "name": league_info.get("name") or game_data.get("league_name"), # Simplified
            "country": league_info.get("country"), # Simplified
            "logo": league_info.get("logo"), # Simplified
            "flag": league_info.get("flag"), # Simplified
            "season": league_info.get("season") or game_data.get("season"), # Simplified
            "round": league_info.get("round"), # May not be in standings object
            "statarea_id": league_info.get("statarea_id"), # Added earlier
        }

        # 3. Teams Info (Use the updated _process_team)
        # Pass the full game_data so _process_team can access team_stats keys
        cleaned["teams"] = {
            "home": self._process_team(game_data, "home_team"), # Removed standings arg
            "away": self._process_team(game_data, "away_team"), # Removed standings arg
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

    def save_individual_game_file(self, game_data: Dict[str, Any]):
        """
        Save cleaned game data to an individual JSON file in the league-specific directory.

        Args:
            game_data: Raw game data after initial processing and enrichment
        """
        # Clean the data before determining the filename components
        cleaned_data = self._clean_final_game_data(game_data)

        # Get league info for directory name
        league_info = cleaned_data.get("league", {})
        league_name = league_info.get("name", "UnknownLeague")
        league_country = league_info.get("country", "")

        # Use standardized league name for the directory
        standardized_league_name = self._standardize_league_name(league_name, league_country)
        league_dir_name = self._sanitize_filename(standardized_league_name) # Sanitize the *standardized* name
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
            league_dir_name = self._sanitize_filename(standardized_league_name) # Sanitize the *standardized* name
            league_dir_path = os.path.join(OUTPUT_DIR, league_dir_name)

            # Create league directory if it doesn't exist
            if not os.path.exists(league_dir_path):
                os.makedirs(league_dir_path)

            # Create standings file in league directory
            standings_filename = f"{date_str}_standings.json"
            standings_file_path = os.path.join(league_dir_path, standings_filename)

            # Prepare standings data
            standings_data_to_save = { # Renamed variable to avoid conflict
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
        Removes unsafe characters and replaces spaces/hyphens with underscores.

        Args:
            name: String to sanitize

        Returns:
            Sanitized string safe for use in filenames
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
            # It's better to clean the game data once first
            cleaned_game_for_summary = self._clean_final_game_data(raw_game) # Use cleaned data
            league_info = cleaned_game_for_summary.get("league", {})
            league_name = league_info.get("name", "UnknownLeague")
            league_country = league_info.get("country", "")

            # Create standardized league key for grouping and directory naming
            standardized_league_name = self._standardize_league_name(league_name, league_country)
            league_dir_name = self._sanitize_filename(standardized_league_name) # Sanitize the *standardized* name

            if league_dir_name not in games_by_league:
                games_by_league[league_dir_name] = {
                    "league_info": league_info, # Store original info for the summary file content
                    "standardized_name": standardized_league_name, # Store standardized name used for dir
                    "games": []
                }

            # Add game to league group using cleaned data
            fixture_info = cleaned_game_for_summary.get("fixture_info", {})
            home_team_info = cleaned_game_for_summary.get("teams", {}).get("home", {})
            away_team_info = cleaned_game_for_summary.get("teams", {}).get("away", {})

            # Extract StatArea ID from the correct location in cleaned data
            home_statarea_id = home_team_info.get("statarea_analysis", {}).get("statarea_id")
            away_statarea_id = away_team_info.get("statarea_analysis", {}).get("statarea_id")

            games_by_league[league_dir_name]["games"].append({
                "fixture_id": fixture_info.get("id"),
                "kickoff_time": fixture_info.get("date"),
                "home_team": {
                    "id": home_team_info.get("id"),
                    "name": home_team_info.get("name"),
                    "statarea_id": home_statarea_id # Use extracted ID
                },
                "away_team": {
                    "id": away_team_info.get("id"),
                    "name": away_team_info.get("name"),
                    "statarea_id": away_statarea_id # Use extracted ID
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
                "league": league_data["league_info"], # Use original league info here
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
        Combines league name and country, then applies specific known standardizations.

        Args:
            league_name: Original league name
            country: Country name

        Returns:
            Standardized league name string (e.g., Süper_Lig_Turkey)
        """
        # Combine name and country for initial check, handling missing country
        base_name = f"{league_name}_{country}" if country else league_name
        lower_name = league_name.lower()
        lower_country = country.lower()

        # Define mappings for standardization (Pattern: Standard Name)
        # Use lower case for matching patterns
        standardization_map = {
             # Netherlands Eerste Divisie (ID 89)
            ('eredivisie 2', 'netherlands'): 'Eerste_Divisie_Netherlands',
            ('eerste divisie', 'netherlands'): 'Eerste_Divisie_Netherlands', # Ensure target name is also mapped

             # Turkey Süper Lig (ID 203)
            ('super lig', 'turkey'): 'Süper_Lig_Turkey',
            ('süper lig', 'turkey'): 'Süper_Lig_Turkey', # Ensure target name is also mapped

             # Romania Liga 1 (ID 283)
            ('liga 1', 'romania'): 'Liga_1_Romania',
            ('liga i', 'romania'): 'Liga_1_Romania',

            # Add other potential cases if needed
            # ('league name pattern', 'country pattern'): 'Standard_Name_Country',
        }

        # Check for matches in standardization map using (lower_name, lower_country)
        # First check for country-specific matches
        if country:
            for (pattern_name, pattern_country), replacement in standardization_map.items():
                if pattern_name in lower_name and pattern_country == lower_country:
                    logger.debug(f"Standardizing '{base_name}' to '{replacement}' based on name and country match.")
                    return replacement

        # Fallback: Check name patterns without country if no country-specific match found
        # (Less common, but might be useful for international leagues or missing country data)
        # Example: ('champions league', ''): 'UEFA_Champions_League'
        # Add specific patterns here if needed

        # If no specific standardization rule matched, return the combined name/country
        logger.debug(f"No specific standardization rule found for '{base_name}'. Using default combined name.")
        # Basic cleanup for the default case before returning
        default_standardized = re.sub(r'[^\w\s-]', '', base_name)
        default_standardized = re.sub(r'[-\s]+', '_', default_standardized).strip('_')
        return default_standardized if default_standardized else "Unknown_League"

