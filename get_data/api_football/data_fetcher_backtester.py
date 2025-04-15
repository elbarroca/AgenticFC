import logging
import json
import hashlib
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple

from get_data.api_football.db_sqlite import SQLiteDBManager, db_manager
from get_data.api_football.endpoints.api_manager import api_manager
from get_data.api_football.endpoints.team_fixtures import TeamFixturesFetcher
from get_data.api_football.endpoints.fixture_details import FixtureDetailsFetcher

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DataFetcherBacktester:
    """Backtester class for fetching historical football data efficiently."""
    
    def __init__(self, db_path: str = "football_data.db", team_ids: List[int] = None, seasons: List[int] = None):
        """
        Initialize the backtester.
        
        Args:
            db_path: Path to SQLite database
            team_ids: List of team IDs to fetch data for
            seasons: List of seasons to fetch data for
        """
        self.db = SQLiteDBManager(db_path)
        self.team_fixtures_fetcher = TeamFixturesFetcher()
        self.fixture_details_fetcher = FixtureDetailsFetcher()
        self.team_ids = team_ids or []
        
        # Default to current and previous season if none provided
        current_year = datetime.now().year
        self.seasons = seasons or [current_year, current_year - 1]
        
        # Stats tracking
        self.api_calls = 0
        self.new_fixtures_found = 0
        self.new_details_processed = 0
        self.cached_fixtures_count = 0
        self.start_time = None
        self.end_time = None
        
        # Initialize API key management
        api_manager.initialize()
        logger.info(f"DataFetcherBacktester initialized with DB: {db_path}")
        if team_ids:
            logger.info(f"Will process {len(team_ids)} teams for seasons: {self.seasons}")
        
    async def fetch_teams_data(self):
        """Fetch team data if not already provided."""
        if not self.team_ids:
            logger.info("No team IDs provided, loading from database")
            self.team_ids = self.db.get_all_team_ids()
            
            if not self.team_ids:
                logger.warning("No teams found in database. Using default Premier League teams.")
                # Default to top Premier League teams if DB is empty
                self.team_ids = [
                    33,   # Manchester United
                    40,   # Liverpool
                    50,   # Manchester City
                    42,   # Arsenal
                    47,   # Tottenham
                    49,   # Chelsea
                    51,   # Brighton
                    46,   # Leicester
                    48,   # West Ham
                    35,   # Bournemouth
                ]
                logger.info(f"Using {len(self.team_ids)} default teams")
                
        return self.team_ids
                
    async def process_team_fixtures(self, team_id: int, season: int) -> List[int]:
        """
        Fetch and process fixtures for a specific team and season.
        Only fetches new data if not already in database.
        
        Args:
            team_id: Team ID to fetch fixtures for
            season: Season year
            
        Returns:
            List[int]: List of fixture IDs processed
        """
        logger.info(f"Processing fixtures for team {team_id}, season {season}")
        
        # Check if we already have fixtures for this team and season
        existing_fixtures = self.db.get_team_fixtures(team_id, season)
        if existing_fixtures:
            logger.info(f"Found {len(existing_fixtures)} existing fixtures for team {team_id}, season {season}")
            self.cached_fixtures_count += len(existing_fixtures)
            
            # If we already have data and it's less than 7 days old, use that
            # Check the most recent fixture's last_updated timestamp
            recent_fixtures = [f for f in existing_fixtures if f.get("last_updated")]
            if recent_fixtures:
                most_recent = max(recent_fixtures, key=lambda x: x.get("last_updated", ""))
                last_updated = datetime.fromisoformat(most_recent.get("last_updated"))
                if (datetime.now() - last_updated) < timedelta(days=7):
                    logger.info(f"Using existing fixtures data (updated {last_updated.isoformat()})")
                    return [fixture.get("fixture", {}).get("id") for fixture in existing_fixtures if "fixture" in fixture]
                else:
                    logger.info(f"Existing data is older than 7 days, refreshing from API")
        
        # Fetch fixtures from API
        logger.info(f"Fetching fixtures for team {team_id}, season {season} from API")
        fixtures_response = self.team_fixtures_fetcher.get_team_fixtures(team_id, season)
        self.api_calls += 1
        
        if not fixtures_response.get("response"):
            logger.warning(f"No fixtures found for team {team_id} in season {season}")
            return []
            
        # Process each fixture
        fixture_ids = []
        for fixture_data in fixtures_response.get("response", []):
            fixture_id = fixture_data.get("fixture", {}).get("id")
            if not fixture_id:
                continue
                
            # Save fixture to database using SQLiteDBManager's save_fixture method
            success = self.db.save_fixture(fixture_data)
            if success:
                fixture_ids.append(fixture_id)
                self.new_fixtures_found += 1
                logger.debug(f"Saved new/updated fixture {fixture_id} for team {team_id}")
            else:
                logger.warning(f"Failed to save fixture {fixture_id} for team {team_id}")
        
        logger.info(f"Processed {len(fixture_ids)} fixtures for team {team_id}, season {season}")
        return fixture_ids
    
    def _hash_fixture_data(self, fixture_data: Dict) -> str:
        """Create a hash of fixture data for comparison, excluding timestamps."""
        # Make a copy to avoid modifying the original
        data_copy = fixture_data.copy() if isinstance(fixture_data, dict) else {}
        
        # Remove fields that change frequently but don't affect core data
        if "last_updated" in data_copy:
            del data_copy["last_updated"]
        
        # Also remove fixture.timestamp if present
        if "fixture" in data_copy and isinstance(data_copy["fixture"], dict):
            if "timestamp" in data_copy["fixture"]:
                del data_copy["fixture"]["timestamp"]
        
        # Create hash of stable data
        return hashlib.md5(json.dumps(data_copy, sort_keys=True).encode()).hexdigest()
        
    async def process_fixture_details(self, fixture_id: int, force_refresh: bool = False) -> bool:
        """
        Fetch and process details for a specific fixture.
        Only fetches new data if not already in database or if forced.
        
        Args:
            fixture_id: Fixture ID to fetch details for
            force_refresh: If True, fetches details even if they already exist
            
        Returns:
            bool: Success status
        """
        # Skip if we already have details for this fixture and not forcing refresh
        if not force_refresh and self.db.check_fixture_details_exists(fixture_id):
            logger.debug(f"Skipping fixture {fixture_id}: details already exist")
            return True
            
        # Skip if the fixture itself doesn't exist
        if not self.db.check_fixture_exists(fixture_id):
            logger.warning(f"Skipping fixture {fixture_id}: fixture does not exist in database")
            return False
            
        logger.info(f"Fetching details for fixture {fixture_id}")
        
        # Fetch fixture details from API
        details = self.fixture_details_fetcher.get_fixture_details(fixture_id)
        self.api_calls += 4  # Basic info, stats, events, lineups
        
        if not details or "basic_info" not in details:
            logger.warning(f"No valid details found for fixture {fixture_id}")
            return False
            
        # Save fixture details
        success = self.db.save_fixture_detail(fixture_id, details)
        if success:
            self.new_details_processed += 1
            logger.debug(f"Saved details for fixture {fixture_id}")
            return True
        else:
            logger.warning(f"Failed to save details for fixture {fixture_id}")
            return False
            
    async def process_team(self, team_id: int) -> int:
        """
        Process all fixtures for a specific team across all configured seasons.
        
        Args:
            team_id: Team ID to process
            
        Returns:
            int: Number of fixtures processed
        """
        logger.info(f"Processing team {team_id}")
        fixtures_processed = 0
        
        for season in self.seasons:
            # Get fixture IDs for this team and season
            fixture_ids = await self.process_team_fixtures(team_id, season)
            fixtures_processed += len(fixture_ids)
            
            # Process fixture details with rate limiting
            missing_details = []
            for fixture_id in fixture_ids:
                if not self.db.check_fixture_details_exists(fixture_id):
                    missing_details.append(fixture_id)
            
            if missing_details:
                logger.info(f"Found {len(missing_details)} fixtures missing details for team {team_id}, season {season}")
                
                for i, fixture_id in enumerate(missing_details):
                    success = await self.process_fixture_details(fixture_id)
                    
                    # Apply rate limiting to prevent API key exhaustion
                    if (i + 1) % 5 == 0 and i + 1 < len(missing_details):
                        logger.info(f"Rate limiting: sleeping for 2 seconds after processing {i+1}/{len(missing_details)} fixtures")
                        await asyncio.sleep(2)
            else:
                logger.info(f"All {len(fixture_ids)} fixtures already have details for team {team_id}, season {season}")
                    
        logger.info(f"Completed processing team {team_id}: {fixtures_processed} fixtures")
        return fixtures_processed
        
    async def find_missing_fixture_details(self) -> List[int]:
        """
        Find fixtures that are missing detailed data.
        
        Returns:
            List[int]: List of fixture IDs missing details
        """
        return self.db.get_fixtures_without_details()
        
    async def process_missing_details(self) -> int:
        """
        Process fixtures that are missing detailed data.
        
        Returns:
            int: Number of fixtures processed
        """
        missing_fixtures = await self.find_missing_fixture_details()
        if not missing_fixtures:
            logger.info("No fixtures missing details")
            return 0
            
        logger.info(f"Processing {len(missing_fixtures)} fixtures missing details")
        processed = 0
        
        for i, fixture_id in enumerate(missing_fixtures):
            success = await self.process_fixture_details(fixture_id)
            if success:
                processed += 1
                
            # Apply rate limiting
            if (i + 1) % 5 == 0 and i + 1 < len(missing_fixtures):
                logger.info(f"Rate limiting: sleeping for 2 seconds after processing {i+1}/{len(missing_fixtures)} fixtures")
                await asyncio.sleep(2)
                
        logger.info(f"Completed processing {processed}/{len(missing_fixtures)} missing fixture details")
        return processed
    
    async def run(self, force_refresh: bool = False):
        """
        Run the backtester to fetch and process data.
        
        Args:
            force_refresh: If True, fetches all data even if it already exists
            
        Returns:
            Dict: Results summary
        """
        self.start_time = time.time()
        logger.info(f"Starting data fetcher backtester run (force_refresh: {force_refresh})")
        
        # Reset counters
        self.api_calls = 0
        self.new_fixtures_found = 0
        self.new_details_processed = 0
        self.cached_fixtures_count = 0
        
        # Ensure we have team IDs
        await self.fetch_teams_data()
        
        # Process each team
        total_fixtures = 0
        for i, team_id in enumerate(self.team_ids):
            fixtures = await self.process_team(team_id)
            total_fixtures += fixtures
            
            # Sleep between teams to avoid rate limiting (except for last team)
            if i + 1 < len(self.team_ids):
                logger.info(f"Sleeping for 3 seconds between teams ({i+1}/{len(self.team_ids)} completed)")
                await asyncio.sleep(3)
            
        # Process any remaining missing fixture details
        if not force_refresh:
            missing_processed = await self.process_missing_details()
        else:
            missing_processed = 0
        
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        result = {
            "success": True,
            "duration_seconds": duration,
            "total_fixtures": total_fixtures,
            "cached_fixtures": self.cached_fixtures_count,
            "new_fixtures": self.new_fixtures_found,
            "new_details": self.new_details_processed,
            "missing_details_processed": missing_processed,
            "api_calls": self.api_calls,
            "teams_processed": len(self.team_ids),
            "seasons_processed": self.seasons,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Data fetcher backtester completed in {duration:.2f} seconds")
        logger.info(f"API calls: {self.api_calls}")
        logger.info(f"Fixtures: {total_fixtures} total, {self.cached_fixtures_count} cached, {self.new_fixtures_found} new/updated")
        logger.info(f"Details: {self.new_details_processed} new, {missing_processed} missing processed")
        
        return result

async def main():
    """Main async entry point with error handling."""
    try:
        # You can customize these parameters
        db_path = "football_data.db"
        
        # Premier League teams
        premier_league_teams = [
            33,   # Manchester United
            40,   # Liverpool
            50,   # Manchester City
            42,   # Arsenal
            47,   # Tottenham
            49,   # Chelsea
            51,   # Brighton
            46,   # Leicester
            48,   # West Ham
        ]
        
        # Current and previous season
        current_year = datetime.now().year
        seasons = [current_year, current_year - 1]
        
        # Initialize and run backtester
        backtester = DataFetcherBacktester(
            db_path=db_path,
            team_ids=premier_league_teams,
            seasons=seasons
        )
        
        # Run backtester
        results = await backtester.run(force_refresh=False)
        
        # Save results to a log file
        with open("backtest_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to backtest_results.json")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
    finally:
        # Close database connection
        if 'backtester' in locals() and hasattr(backtester, 'db'):
            backtester.db.close()

if __name__ == "__main__":
    asyncio.run(main())