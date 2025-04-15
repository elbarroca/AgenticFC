import asyncio
import logging
import sqlite3
import os
import json
import sys
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from get_data.api_football.data_fetcher_backtester import DataFetcherBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktesterTest:
    """Test class for DataFetcherBacktester to verify functionality with minimal API calls."""
    
    def __init__(self, db_path: str = "test_football_data.db"):
        """Initialize the test with a test database."""
        self.db_path = db_path
        self.backtester = None
        
        # Delete test database if it exists to start clean
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"Removed existing test database: {db_path}")
    
    async def setup_backtester(self):
        """Set up a backtester instance with minimal configuration."""
        # Use only one team and one recent season to minimize API calls
        test_team_ids = [33]  # Manchester United only
        current_year = datetime.now().year
        test_seasons = [current_year - 1]  # Just last season
        
        self.backtester = DataFetcherBacktester(
            db_path=self.db_path,
            team_ids=test_team_ids,
            seasons=test_seasons
        )
        logger.info(f"Initialized test backtester with team ID {test_team_ids} for season {test_seasons}")
        
        return self.backtester
    
    async def run_minimal_test(self) -> Dict[str, Any]:
        """Run a minimal test to verify the backtester functionality."""
        if not self.backtester:
            await self.setup_backtester()
        
        # Run the backtester with minimal configuration
        logger.info("Starting minimal backtester test run")
        results = await self.backtester.run(force_refresh=False)
        
        # Log results
        logger.info(f"Test completed in {results['duration_seconds']:.2f} seconds")
        logger.info(f"API calls: {results['api_calls']}")
        logger.info(f"Total fixtures: {results['total_fixtures']}")
        logger.info(f"New fixtures: {results['new_fixtures']}")
        logger.info(f"New details: {results['new_details']}")
        
        return results
    
    def verify_database(self) -> bool:
        """Verify that the database was created and has expected tables and data."""
        if not os.path.exists(self.db_path):
            logger.error(f"Test database not found: {self.db_path}")
            return False
        
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            required_tables = [
                'teams', 'leagues', 'team_seasons', 'fixtures', 
                'fixture_details', 'odds', 'cache_control'
            ]
            
            missing_tables = [table for table in required_tables if table not in tables]
            if missing_tables:
                logger.error(f"Missing tables in database: {missing_tables}")
                return False
            
            # Check if we have team data
            cursor.execute("SELECT COUNT(*) FROM teams")
            team_count = cursor.fetchone()[0]
            
            # Check if we have fixture data
            cursor.execute("SELECT COUNT(*) FROM fixtures")
            fixture_count = cursor.fetchone()[0]
            
            # Check if we have fixture details
            cursor.execute("SELECT COUNT(*) FROM fixture_details")
            detail_count = cursor.fetchone()[0]
            
            logger.info(f"Database verification: {team_count} teams, {fixture_count} fixtures, {detail_count} fixture details")
            
            # Success if we have at least one team and one fixture
            return team_count > 0 and fixture_count > 0
            
        except Exception as e:
            logger.error(f"Database verification failed: {str(e)}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    async def test_single_fixture(self) -> bool:
        """Test fetching and processing a single fixture to minimize API calls."""
        if not self.backtester:
            await self.setup_backtester()
            
        # First get team fixtures
        team_id = 33  # Manchester United
        season = datetime.now().year - 1
        
        logger.info(f"Testing single fixture processing for team {team_id}, season {season}")
        
        # Get first fixture ID
        fixture_ids = await self.backtester.process_team_fixtures(team_id, season)
        
        if not fixture_ids:
            logger.error("No fixtures found for team")
            return False
            
        # Process just the first fixture
        logger.info(f"Testing details fetching for fixture {fixture_ids[0]}")
        success = await self.backtester.process_fixture_details(fixture_ids[0])
        
        if success:
            logger.info(f"Successfully processed fixture {fixture_ids[0]}")
        else:
            logger.error(f"Failed to process fixture {fixture_ids[0]}")
            
        return success
        
    def cleanup(self):
        """Clean up after tests."""
        if self.backtester and hasattr(self.backtester, 'db'):
            self.backtester.db.close()
            logger.info("Closed backtester database connection")
            
        # You can choose to remove the test database here or keep it for inspection
        # if os.path.exists(self.db_path):
        #     os.remove(self.db_path)
        #     logger.info(f"Removed test database: {self.db_path}")

async def main():
    """Main entry point for the backtester test."""
    try:
        logger.info("Starting DataFetcherBacktester test")
        test = BacktesterTest(db_path="test_football_data.db")
        
        # Test a single fixture first
        single_fixture_success = await test.test_single_fixture()
        
        if single_fixture_success:
            logger.info("Single fixture test passed!")
            
            # Verify database has expected data
            db_valid = test.verify_database()
            
            if db_valid:
                logger.info("Database verification passed!")
                logger.info("All tests PASSED! The DataFetcherBacktester is working properly.")
            else:
                logger.error("Database verification failed.")
        else:
            logger.error("Single fixture test failed.")
            
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}", exc_info=True)
    finally:
        if 'test' in locals():
            test.cleanup()

if __name__ == "__main__":
    asyncio.run(main())