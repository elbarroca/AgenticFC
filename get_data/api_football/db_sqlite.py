import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)

class SQLiteDBManager:
    """SQLite Database Manager for football data"""
    
    def __init__(self, db_path: str = "football_data.db"):
        """
        Initialize the SQLite database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Create directory for DB if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        # Configure to return rows as dictionaries
        self.conn.row_factory = sqlite3.Row
        
        # Create required tables
        self.create_tables()
        logger.info(f"Initialized SQLite database at {db_path}")

    def create_tables(self):
        """Create required tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Teams table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            name TEXT,
            country TEXT,
            logo TEXT,
            last_updated TEXT
        )
        """)
        
        # Leagues table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS leagues (
            league_id INTEGER PRIMARY KEY,
            name TEXT,
            country TEXT,
            logo TEXT,
            season INTEGER,
            last_updated TEXT
        )
        """)
        
        # Team seasons table (for teams participating in specific seasons)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_seasons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER,
            season INTEGER,
            league_id INTEGER,
            last_updated TEXT,
            UNIQUE(team_id, season, league_id),
            FOREIGN KEY(team_id) REFERENCES teams(team_id)
        )
        """)
        
        # Fixtures table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fixtures (
            fixture_id INTEGER PRIMARY KEY,
            team_home_id INTEGER,
            team_away_id INTEGER,
            league_id INTEGER,
            season INTEGER,
            date TEXT,
            timestamp INTEGER,
            status TEXT,
            data TEXT,
            last_updated TEXT,
            FOREIGN KEY(team_home_id) REFERENCES teams(team_id),
            FOREIGN KEY(team_away_id) REFERENCES teams(team_id)
        )
        """)
        
        # Fixture details table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fixture_details (
            fixture_id INTEGER PRIMARY KEY,
            basic_info TEXT,
            statistics TEXT,
            events TEXT,
            lineups TEXT,
            last_updated TEXT,
            FOREIGN KEY(fixture_id) REFERENCES fixtures(fixture_id)
        )
        """)
        
        # Odds table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER,
            bookmaker TEXT,
            bet_type TEXT,
            odds_values TEXT,
            last_updated TEXT,
            UNIQUE(fixture_id, bookmaker, bet_type),
            FOREIGN KEY(fixture_id) REFERENCES fixtures(fixture_id)
        )
        """)
        
        # Cache control table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache_control (
            request_key TEXT PRIMARY KEY,
            endpoint TEXT,
            params TEXT,
            response_hash TEXT,
            timestamp INTEGER,
            valid_until INTEGER
        )
        """)
        
        self.conn.commit()
        
    def save_team(self, team_id: int, team_data: Dict[str, Any]) -> bool:
        """
        Save team information to the database.
        
        Args:
            team_id: The team ID
            team_data: Dictionary containing team data including name, country, logo
            
        Returns:
            bool: Success status
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.utcnow().isoformat()
            
            # Extract team data
            name = team_data.get('name', '')
            country = team_data.get('country', '')
            logo = team_data.get('logo', '')
            
            cursor.execute("""
            INSERT OR REPLACE INTO teams (team_id, name, country, logo, last_updated)
            VALUES (?, ?, ?, ?, ?)
            """, (team_id, name, country, logo, now))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving team {team_id}: {str(e)}")
            return False
    
    def save_team_season(self, team_id: int, season: int, league_id: int) -> bool:
        """
        Save team-season mapping to the database.
        
        Args:
            team_id: The team ID
            season: The season year
            league_id: The league ID
            
        Returns:
            bool: Success status
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.utcnow().isoformat()
            
            cursor.execute("""
            INSERT OR REPLACE INTO team_seasons (team_id, season, league_id, last_updated)
            VALUES (?, ?, ?, ?)
            """, (team_id, season, league_id, now))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving team_season {team_id}-{season}: {str(e)}")
            return False
    
    def save_fixture(self, fixture_data: Dict[str, Any]) -> bool:
        """
        Save fixture information to the database.
        
        Args:
            fixture_data: Dictionary containing fixture data
            
        Returns:
            bool: Success status
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.utcnow().isoformat()
            
            # Extract data from the fixture data
            fixture_id = fixture_data.get('fixture', {}).get('id')
            if not fixture_id:
                logger.error("Missing fixture ID in fixture data")
                return False
                
            date = fixture_data.get('fixture', {}).get('date', '')
            timestamp = fixture_data.get('fixture', {}).get('timestamp', 0)
            status = fixture_data.get('fixture', {}).get('status', {}).get('short', '')
            
            teams = fixture_data.get('teams', {})
            team_home_id = teams.get('home', {}).get('id', 0)
            team_away_id = teams.get('away', {}).get('id', 0)
            
            league = fixture_data.get('league', {})
            league_id = league.get('id', 0)
            season = league.get('season', 0)
            
            # Save the teams if they don't exist
            if team_home_id:
                self.save_team(team_home_id, teams.get('home', {}))
            if team_away_id:
                self.save_team(team_away_id, teams.get('away', {}))
                
            # Save full data as JSON
            data_json = json.dumps(fixture_data)
            
            cursor.execute("""
            INSERT OR REPLACE INTO fixtures 
            (fixture_id, team_home_id, team_away_id, league_id, season, date, timestamp, status, data, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (fixture_id, team_home_id, team_away_id, league_id, season, date, timestamp, status, data_json, now))
            
            # Also save team-season mapping
            if team_home_id and season and league_id:
                self.save_team_season(team_home_id, season, league_id)
            if team_away_id and season and league_id:
                self.save_team_season(team_away_id, season, league_id)
                
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving fixture: {str(e)}")
            return False
    
    def save_fixture_detail(self, fixture_id: int, details_data: Dict[str, Any]) -> bool:
        """
        Save fixture details to the database.
        
        Args:
            fixture_id: The fixture ID
            details_data: Dictionary containing fixture details including basic_info, statistics, events, lineups
            
        Returns:
            bool: Success status
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.utcnow().isoformat()
            
            # Check if fixture exists
            cursor.execute("SELECT 1 FROM fixtures WHERE fixture_id = ?", (fixture_id,))
            if not cursor.fetchone():
                logger.warning(f"Cannot save fixture details: Fixture {fixture_id} does not exist in database")
                return False
                
            basic_info = json.dumps(details_data.get('basic_info', []))
            statistics = json.dumps(details_data.get('statistics', []))
            events = json.dumps(details_data.get('events', []))
            lineups = json.dumps(details_data.get('lineups', []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO fixture_details 
            (fixture_id, basic_info, statistics, events, lineups, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (fixture_id, basic_info, statistics, events, lineups, now))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving fixture details for {fixture_id}: {str(e)}")
            return False
            
    def save_odds(self, fixture_id: int, bookmaker: str, bet_type: str, odds_values: Dict[str, Any]) -> bool:
        """
        Save odds data to the database.
        
        Args:
            fixture_id: The fixture ID
            bookmaker: The bookmaker name
            bet_type: The type of bet (1X2, over/under, etc.)
            odds_values: Dictionary containing odds values
            
        Returns:
            bool: Success status
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.utcnow().isoformat()
            
            # Check if fixture exists
            cursor.execute("SELECT 1 FROM fixtures WHERE fixture_id = ?", (fixture_id,))
            if not cursor.fetchone():
                logger.warning(f"Cannot save odds: Fixture {fixture_id} does not exist in database")
                return False
                
            values_json = json.dumps(odds_values)
            
            cursor.execute("""
            INSERT OR REPLACE INTO odds 
            (fixture_id, bookmaker, bet_type, odds_values, last_updated)
            VALUES (?, ?, ?, ?, ?)
            """, (fixture_id, bookmaker, bet_type, values_json, now))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving odds for fixture {fixture_id}: {str(e)}")
            return False
            
    def cache_api_response(self, endpoint: str, params: Dict, response_hash: str, valid_minutes: int = 60) -> bool:
        """
        Cache an API response hash to avoid duplicate requests.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            response_hash: Hash of the response
            valid_minutes: Cache validity in minutes
            
        Returns:
            bool: Success status
        """
        try:
            cursor = self.conn.cursor()
            now = int(datetime.utcnow().timestamp())
            valid_until = now + (valid_minutes * 60)
            
            # Create a unique key from endpoint and params
            request_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
            
            cursor.execute("""
            INSERT OR REPLACE INTO cache_control 
            (request_key, endpoint, params, response_hash, timestamp, valid_until)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (request_key, endpoint, json.dumps(params), response_hash, now, valid_until))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error caching API response: {str(e)}")
            return False
            
    def is_request_cached(self, endpoint: str, params: Dict) -> bool:
        """
        Check if a request is cached and still valid.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            bool: True if cached and valid
        """
        try:
            cursor = self.conn.cursor()
            now = int(datetime.utcnow().timestamp())
            
            # Create a unique key from endpoint and params
            request_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
            
            cursor.execute("""
            SELECT valid_until FROM cache_control 
            WHERE request_key = ? AND valid_until > ?
            """, (request_key, now))
            
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking cache: {str(e)}")
            return False
    
    def get_fixture(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """
        Get fixture data by ID.
        
        Args:
            fixture_id: The fixture ID
            
        Returns:
            Optional[Dict]: Fixture data or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT data FROM fixtures WHERE fixture_id = ?", (fixture_id,))
            result = cursor.fetchone()
            
            if result:
                return json.loads(result[0])
            return None
        except Exception as e:
            logger.error(f"Error getting fixture {fixture_id}: {str(e)}")
            return None
            
    def get_fixture_details(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """
        Get fixture details by fixture ID.
        
        Args:
            fixture_id: The fixture ID
            
        Returns:
            Optional[Dict]: Fixture details or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT basic_info, statistics, events, lineups, last_updated 
            FROM fixture_details 
            WHERE fixture_id = ?
            """, (fixture_id,))
            
            result = cursor.fetchone()
            if not result:
                return None
                
            return {
                'basic_info': json.loads(result[0]),
                'statistics': json.loads(result[1]),
                'events': json.loads(result[2]),
                'lineups': json.loads(result[3]),
                'last_updated': result[4]
            }
        except Exception as e:
            logger.error(f"Error getting fixture details for {fixture_id}: {str(e)}")
            return None
    
    def get_team_fixtures(self, team_id: int, season: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all fixtures for a team.
        
        Args:
            team_id: The team ID
            season: Optional season filter
            
        Returns:
            List[Dict]: List of fixtures
        """
        try:
            cursor = self.conn.cursor()
            query = """
            SELECT data FROM fixtures 
            WHERE team_home_id = ? OR team_away_id = ?
            """
            params = [team_id, team_id]
            
            if season:
                query += " AND season = ?"
                params.append(season)
                
            cursor.execute(query, params)
            
            fixtures = []
            for row in cursor.fetchall():
                fixtures.append(json.loads(row[0]))
                
            return fixtures
        except Exception as e:
            logger.error(f"Error getting team fixtures for {team_id}: {str(e)}")
            return []
    
    def get_all_team_ids(self) -> List[int]:
        """
        Get all team IDs from the database.
        
        Returns:
            List[int]: List of team IDs
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT team_id FROM teams")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting all team IDs: {str(e)}")
            return []
    
    def get_teams_without_fixtures(self, season: Optional[int] = None) -> List[int]:
        """
        Get team IDs that don't have fixtures in the database.
        
        Args:
            season: Optional season filter
            
        Returns:
            List[int]: List of team IDs
        """
        try:
            cursor = self.conn.cursor()
            query = """
            SELECT t.team_id FROM teams t
            LEFT JOIN (
                SELECT DISTINCT team_home_id as team_id FROM fixtures
                UNION
                SELECT DISTINCT team_away_id as team_id FROM fixtures
            ) f ON t.team_id = f.team_id
            WHERE f.team_id IS NULL
            """
            
            if season:
                query = """
                SELECT ts.team_id FROM team_seasons ts
                LEFT JOIN (
                    SELECT DISTINCT team_home_id as team_id, season FROM fixtures
                    UNION
                    SELECT DISTINCT team_away_id as team_id, season FROM fixtures
                ) f ON ts.team_id = f.team_id AND ts.season = f.season
                WHERE f.team_id IS NULL AND ts.season = ?
                """
                cursor.execute(query, (season,))
            else:
                cursor.execute(query)
                
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting teams without fixtures: {str(e)}")
            return []
    
    def get_fixtures_without_details(self) -> List[int]:
        """
        Get fixture IDs that don't have detailed data in the database.
        
        Returns:
            List[int]: List of fixture IDs
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT f.fixture_id FROM fixtures f
            LEFT JOIN fixture_details fd ON f.fixture_id = fd.fixture_id
            WHERE fd.fixture_id IS NULL
            """)
            
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting fixtures without details: {str(e)}")
            return []
    
    def check_fixture_exists(self, fixture_id: int) -> bool:
        """
        Check if a fixture exists in the database.
        
        Args:
            fixture_id: The fixture ID
            
        Returns:
            bool: True if exists
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM fixtures WHERE fixture_id = ?", (fixture_id,))
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking fixture existence: {str(e)}")
            return False
    
    def check_fixture_details_exists(self, fixture_id: int) -> bool:
        """
        Check if fixture details exist in the database.
        
        Args:
            fixture_id: The fixture ID
            
        Returns:
            bool: True if exists
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM fixture_details WHERE fixture_id = ?", (fixture_id,))
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking fixture details existence: {str(e)}")
            return False

    def check_odds_exist(self, fixture_id: int) -> bool:
        """
        Check if odds data exist in the database for a specific fixture.

        Args:
            fixture_id: The fixture ID

        Returns:
            bool: True if exists
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM odds WHERE fixture_id = ?", (fixture_id,))
            exists = cursor.fetchone() is not None
            if exists:
                logger.debug(f"Odds found in DB for fixture {fixture_id}")
            else:
                logger.debug(f"No odds found in DB for fixture {fixture_id}")
            return exists
        except Exception as e:
            logger.error(f"Error checking odds existence for fixture {fixture_id}: {str(e)}")
            return False
            
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

# Create a singleton instance
db_manager = SQLiteDBManager()