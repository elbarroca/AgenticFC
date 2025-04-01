import os
import logging
import sqlite3
from datetime import date
import json
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
import re

# Import MongoDB manager from existing code
from get_data.api_football.db_mongo import MongoDBManager
# Import ID mappings for teams and leagues
from get_data.db_ids.team_id_mappings import TEAM_ID_MAPPING
from get_data.db_ids.league_id_mappings import LEAGUE_ID_MAPPING

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

    def _clean_game_data(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean game data to remove duplicative content and organize it better,
        while ensuring essential data from MongoDB is retained.

        Args:
            game_data: Original game data fetched from MongoDB and enriched
                       with StatArea info.

        Returns:
            Cleaned game data ready for JSON output.
        """
        # Start with essential fixture and date info
        cleaned_data = {
            "fixture_id": game_data.get("fixture_id"),
            "date": game_data.get("date"), # This should be the match date
            "league": { # Extract league info carefully
                "id": game_data.get("league", {}).get("id"),
                "name": game_data.get("league", {}).get("name"),
                "country": game_data.get("league", {}).get("country"),
                "logo": game_data.get("league", {}).get("logo"),
                "flag": game_data.get("league", {}).get("flag"),
                "season": game_data.get("league", {}).get("season"),
                "round": game_data.get("league", {}).get("round"),
                "statarea_id": game_data.get("league", {}).get("statarea_id") # Added during processing
            },
            "teams": { # Process teams using the dedicated function
                "home": self._clean_team_data(game_data.get("home_team", {}), is_home=True),
                "away": self._clean_team_data(game_data.get("away_team", {}), is_home=False)
            },
            # Extract venue, status, referee etc. from 'match_info' or 'fixture'
            "venue": game_data.get("fixture", {}).get("venue"),
            "status": game_data.get("fixture", {}).get("status"),
            "referee": game_data.get("fixture", {}).get("referee"),
            "timestamp": game_data.get("fixture", {}).get("timestamp"),
            "timezone": game_data.get("fixture", {}).get("timezone"),
            "goals": game_data.get("goals"), # Should contain home/away goals
            "score": game_data.get("score"), # Should contain halftime, fulltime etc.
            "events": game_data.get("events"), # Keep events data
            "lineups": game_data.get("lineups"), # Keep lineups if available
            "statistics": game_data.get("statistics"), # Keep detailed match stats if available
            "predictions": self._clean_predictions_data(game_data.get("predictions", {})), # Process predictions
            "h2h": self._clean_h2h_data(game_data.get("h2h", [])) # Process H2H
        }

        # Remove None values for cleaner output if desired, but be cautious
        # cleaned_data = {k: v for k, v in cleaned_data.items() if v is not None}

        return cleaned_data

    def _clean_team_data(self, team_data: Dict[str, Any], is_home: bool) -> Dict[str, Any]:
        """
        Clean team data, preserving info from MongoDB and adding StatArea structure.

        Args:
            team_data: Original team data from the enriched match_data.
            is_home: Boolean indicating if this is the home team.

        Returns:
            Cleaned team data.
        """
        if not team_data:
            return {}

        cleaned = {
            "id": team_data.get("id"),
            "name": team_data.get("name"),
            "logo": team_data.get("logo"),
            "winner": team_data.get("winner"), # Keep winner status if available
            "statarea_id": team_data.get("statarea_id", "unknown"), # Added during processing
            # Preserve league form if available from MongoDB source
            "league_form": team_data.get("league_form"),
             # Preserve last 5 matches form if available from MongoDB source
            "form_last_5": team_data.get("form_last_5"),
            # Preserve coach info if available
             "coach": team_data.get("coach"),
            # Preserve formation if available (might be under lineups though)
            "formation": team_data.get("formation"),
            # Add basic stats structure (will be populated from potentially different keys)
            "basic_stats": {},
            # Add statarea data structure
            "statarea_data": {
                "last_scraped": None,
                "recent_form": "",
                "key_stats": {},
                "bet_stats": {},
                "recent_matches": []
            }
        }

        # --- Try to find and populate basic_stats from MongoDB data ---
        # The original code looked for 'home_team_stats'/'away_team_stats' keys
        # directly within team_data, which might be incorrect.
        # Let's assume the stats might be directly under the team object
        # or nested differently depending on the source MongoDB structure.
        # We'll check common patterns.

        # Pattern 1: Stats directly under the team object
        if "fixtures" in team_data or "goals" in team_data:
             cleaned["basic_stats"] = {
                "fixtures": team_data.get("fixtures"),
                "goals": team_data.get("goals"), # Keep the whole goals structure
                "biggest": team_data.get("biggest"),
                "clean_sheet": team_data.get("clean_sheet"),
                "failed_to_score": team_data.get("failed_to_score"),
                "penalty": team_data.get("penalty"),
                "lineups": team_data.get("lineups"), # Team-specific lineups stats
                "cards": team_data.get("cards"),
            }
        # Pattern 2: Check if stats are nested (e.g., team_data['stats']) - adapt if needed
        elif "stats" in team_data and isinstance(team_data["stats"], dict):
             cleaned["basic_stats"] = team_data["stats"]


        # --- Process and structure StatArea data ---
        if "statarea_data" in team_data and team_data["statarea_data"].get("stats"):
            statarea_raw = team_data["statarea_data"]
            statarea_stats = statarea_raw.get("stats", {})
            statarea_matches = statarea_raw.get("match_history", [])

            # Prioritize host/guest stats based on whether it's home/away team
            # Use last 15 games (period 15) as primary source
            primary_key = "host_15" if is_home else "guest_15"
            secondary_key = "guest_15" if is_home else "host_15"
            fallback_key_10 = primary_key.replace("15", "10")
            fallback_key_5 = primary_key.replace("15", "5")

            relevant_stats = statarea_stats.get(primary_key,
                                           statarea_stats.get(secondary_key,
                                           statarea_stats.get(fallback_key_10,
                                           statarea_stats.get(fallback_key_5, {}))))

            recent_form = self._calculate_recent_form(statarea_matches) # Calculate from StatArea matches

            cleaned["statarea_data"]["last_scraped"] = statarea_raw.get("last_scraped")
            cleaned["statarea_data"]["recent_form"] = recent_form
            cleaned["statarea_data"]["key_stats"] = {
                "average_goals_scored": relevant_stats.get("Average goals scored"),
                "average_goals_conceded": relevant_stats.get("Average goals conceded"),
                "clean_sheet_percentage": relevant_stats.get("Clean sheet percentage"),
                "win_percentage": relevant_stats.get("Win percentage"),
                "both_teams_scored_percentage": relevant_stats.get("Both teams scored percentage")
            }
            # Include betting stats (e.g., Over/Under) if available
            if "Total Goals" in relevant_stats:
                 cleaned["statarea_data"]["bet_stats"]["over_under"] = relevant_stats.get("Total Goals", {})
            if "Result / Both teams scored" in relevant_stats:
                 cleaned["statarea_data"]["bet_stats"]["result_btts"] = relevant_stats.get("Result / Both teams scored", {})
            # Add more bet stats categories as needed...

            # Include only the 5 most recent matches from StatArea
            cleaned["statarea_data"]["recent_matches"] = statarea_matches[:5]

        # If StatArea recent_form is empty, try using the form from MongoDB data
        if not cleaned["statarea_data"]["recent_form"]:
             cleaned["statarea_data"]["recent_form"] = cleaned.get("form_last_5") or cleaned.get("league_form")


        return cleaned

    def _clean_predictions_data(self, predictions_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean predictions data, trying different possible structures from MongoDB.

        Args:
            predictions_data: Original predictions data object from MongoDB source.

        Returns:
            Cleaned predictions data.
        """
        if not predictions_data:
            return {}

        # The original code assumed a nested "predictions" key.
        # Let's check if the required fields are directly under predictions_data
        # or nested as originally thought.

        # Try accessing directly first
        winner = predictions_data.get("winner")
        advice = predictions_data.get("advice")
        percent = predictions_data.get("percent")
        comparison = predictions_data.get("comparison")
        under_over = predictions_data.get("under_over")
        goals_pred = predictions_data.get("goals") # Might conflict with match goals, check source
        h2h_pred = predictions_data.get("h2h") # Might conflict with actual h2h, check source

        # If direct access failed, try the nested structure
        if winner is None and "predictions" in predictions_data:
            nested_pred = predictions_data.get("predictions", {})
            winner = nested_pred.get("winner")
            advice = nested_pred.get("advice")
            percent = nested_pred.get("percent")
            # Comparison might still be top-level
            comparison = predictions_data.get("comparison", comparison) # Keep existing if found
            under_over = predictions_data.get("under_over", under_over)
            goals_pred = predictions_data.get("goals", goals_pred)
            h2h_pred = predictions_data.get("h2h", h2h_pred)


        cleaned = {
            "winner": winner,
            "advice": advice,
            "percent": percent,
            "comparison": comparison,
            "under_over": under_over,
            "goals_prediction": goals_pred, # Renamed to avoid clash
            "h2h_prediction": h2h_pred # Renamed to avoid clash
            # Add other prediction fields like 'results', 'winning_odds' etc. if they exist
        }

        # Add other potential top-level prediction fields if they exist
        other_keys = ["results", "winning_odds", "league_position", "attacks", "defenses", "poisson_distribution"]
        for key in other_keys:
            if key in predictions_data:
                cleaned[key] = predictions_data[key]


        # Remove None values if desired
        # cleaned = {k: v for k, v in cleaned.items() if v is not None}

        return cleaned

    def _clean_h2h_data(self, h2h_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean head-to-head data to have a more concise structure.
        
        Args:
            h2h_data: Original head-to-head data
            
        Returns:
            Cleaned head-to-head data
        """
        if not h2h_data:
            return []
            
        cleaned_h2h = []
        for match in h2h_data:
            cleaned_match = {
                "date": match.get("fixture", {}).get("date"),
                "home_team": {
                    "id": match.get("teams", {}).get("home", {}).get("id"),
                    "name": match.get("teams", {}).get("home", {}).get("name"),
                    "winner": match.get("teams", {}).get("home", {}).get("winner")
                },
                "away_team": {
                    "id": match.get("teams", {}).get("away", {}).get("id"),
                    "name": match.get("teams", {}).get("away", {}).get("name"),
                    "winner": match.get("teams", {}).get("away", {}).get("winner")
                },
                "score": {
                    "home": match.get("goals", {}).get("home"),
                    "away": match.get("goals", {}).get("away")
                },
                "league": {
                    "id": match.get("league", {}).get("id"),
                    "name": match.get("league", {}).get("name"),
                    "country": match.get("league", {}).get("country")
                }
            }
            cleaned_h2h.append(cleaned_match)
            
        # Limit to the 5 most recent matches
        return cleaned_h2h[:5]
    
    def _calculate_recent_form(self, matches: List[Dict[str, Any]]) -> str:
        """
        Calculate recent form string (W-D-L) from match history.
        
        Args:
            matches: List of recent matches
            
        Returns:
            Form string (e.g., "WDLWW")
        """
        if not matches:
            return ""
            
        form = ""
        for match in matches[:5]:  # Last 5 matches
            result = match.get("result")
            if result == "win":
                form += "W"
            elif result == "draw":
                form += "D"
            elif result == "loss":
                form += "L"
        
        return form
    
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
    
    def __init__(self):
        """Initialize the extractor with database connections."""
        self.mongo_db = MongoDBManager()
        self.statarea_db = StatAreaDBManager()
        
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
        
        # Get daily games summary from MongoDB
        games_data = self.mongo_db.get_daily_games(date_str)
        
        if not games_data:
            logger.warning(f"No games found for {date_str}")
            return {"date": date_str, "games": [], "total_games": 0}
        
        # Get fixture IDs for the date
        fixture_ids = self.mongo_db.get_match_fixture_ids_for_date(date_str)
        logger.info(f"Found {len(fixture_ids)} fixtures for {date_str}")
        
        # Extract detailed game data for each fixture
        detailed_games = []
        
        for fixture_id in fixture_ids:
            # Get match data from MongoDB (contains all the comprehensive data)
            match_data = self.mongo_db.get_match_data(date_str, fixture_id)
            
            if match_data:
                # Process match data and add StatArea data
                game_info = self.process_game_data(match_data)
                detailed_games.append(game_info)
                
                # Save individual game file
                self.save_individual_game_file(game_info)
        
        return {
            "date": date_str,
            "games": detailed_games,
            "total_games": len(detailed_games)
        }
    
    def process_game_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process match data and enrich it with StatArea data.
        
        Args:
            match_data: Raw match data from MongoDB
            
        Returns:
            Processed and enriched game data
        """
        # Make a copy of match data (excluding MongoDB _id field)
        if "_id" in match_data:
            del match_data["_id"]
            
        # Add mapped StatArea IDs for teams and leagues
        self._add_statarea_id_mappings(match_data)
        
        # Add StatArea data for teams
        self._add_statarea_team_data(match_data)
        
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
    
    def save_individual_game_file(self, game_data: Dict[str, Any]):
        """
        Save game data to an individual JSON file named after the teams.
        
        Args:
            game_data: Game data to save
        """
        # Create directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Get team names and date
        home_team = self._sanitize_filename(game_data.get("home_team", {}).get("name", "Home"))
        away_team = self._sanitize_filename(game_data.get("away_team", {}).get("name", "Away"))
        date_str = game_data.get("date", self.get_current_date_str())
        fixture_id = game_data.get("fixture_id", "unknown")
        
        # Create filename
        filename = f"{date_str}_{home_team}_vs_{away_team}_{fixture_id}.json"
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the raw game data to file - no cleaning needed
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved game data to {file_path}")
    
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


def main():
    """Main function to extract game data for a specific day."""
    import argparse
    parser = argparse.ArgumentParser(description='Extract game data from MongoDB and StatArea for a specific day')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--output', type=str, help='Output summary file path')
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = DailyGameExtractor()
        
        # Extract data
        data = extractor.extract_games_for_date(args.date)
        
        # Save summary file
        extractor.save_summary_file(data, args.output)
        
        print(f"Successfully extracted {data['total_games']} games for {data['date']}")
        print(f"Individual game files saved to the '{OUTPUT_DIR}' directory")
        
    except Exception as e:
        logger.error(f"Error extracting game data: {e}")
        raise
    finally:
        # Close MongoDB connection
        extractor.mongo_db.close_connection()

if __name__ == "__main__":
    main()