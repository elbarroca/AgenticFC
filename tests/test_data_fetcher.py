import os
import logging
from datetime import datetime, timezone
import asyncio

# Import the components we need
from get_data.api_football.data_fetcher import fetch_all_data
from get_data.api_football.db_mongo import db_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_single_fixture(date=None, fixture_id=None, force_reprocess=True):
    """Run the data fetching pipeline for a single fixture on a specific date"""
    try:
        # Ensure we have a DB connection
        if db_manager._client is None:
            logger.error("Database connection not initialized")
            return False
        
        if date is None:
            date = datetime.now(timezone.utc)
        
        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"Testing data storage for date: {date_str}")
        
        # Run the data fetching pipeline which now includes games, matches, and odds
        logger.info("Running integrated data fetching pipeline...")
        results = await fetch_all_data(target_date=date, force_reprocess=force_reprocess)
        
        # Log the results summary
        logger.info("Data fetch results:")
        logger.info(f"Overall Success: {results['success']}")
        for step, details in results["steps"].items():
            logger.info(f"  {step}: {details['success']} - {details['message']}")
        
        # Verify database structure and data
        logger.info("\nValidating database structure and records...")
        
        # Parse date components
        date_parts = db_manager._parse_date_components(date_str)
        month = date_parts["month"]
        day = date_parts["day"]
        
        # Log database info
        logger.info(f"Monthly collections with data for date {date_str}:")
        
        # Check games data
        logger.info("\nChecking games collection...")
        games_collection = db_manager._get_month_collection('games', date_str)
        games_data = db_manager.get_daily_games(date_str)
        if games_data:
            total_games = games_data.get('total_matches', 0)
            league_count = len(games_data.get('leagues', {}))
            logger.info(f"Found {total_games} games across {league_count} leagues")
            
            # If no fixture ID provided, try to find one
            if not fixture_id and league_count > 0:
                for league_id, league_data in games_data['leagues'].items():
                    if league_data.get('matches'):
                        fixture_id = league_data['matches'][0]['id']
                        logger.info(f"Selected fixture ID {fixture_id} for testing")
                        break
        else:
            logger.warning(f"No games data found in database for {date_str}")
        
        # Check match data for the selected fixture
        if fixture_id:
            logger.info(f"\nChecking match data for fixture {fixture_id}...")
            
            # Check if match exists
            match_exists = db_manager.check_match_exists(date_str, fixture_id)
            logger.info(f"Match exists in database: {match_exists}")
            
            if match_exists:
                match_data = db_manager.get_match_data(date_str, fixture_id)
                if match_data:
                    teams = match_data.get('teams', {})
                    home_team = teams.get('home', {}).get('name', 'Unknown')
                    away_team = teams.get('away', {}).get('name', 'Unknown')
                    logger.info(f"Match details: {home_team} vs {away_team}")
            
            # Check odds data separately
            logger.info(f"\nChecking odds data for fixture {fixture_id}...")
            odds_data = db_manager.get_odds_data(date_str, fixture_id)
            if odds_data:
                bookmaker_count = len(odds_data.get('bookmakers', []))
                logger.info(f"Found odds data with {bookmaker_count} bookmakers")
            else:
                logger.warning(f"No odds data found for fixture {fixture_id}")
        else:
            logger.warning("No fixture ID available for testing")
            
        # Final validation of database structure
        logger.info("\nFinal database structure validation:")
        all_dbs = await asyncio.to_thread(lambda: db_manager._client.list_database_names())
        logger.info(f"All databases: {all_dbs}")
        
        for db_type in ['games', 'matches', 'standings', 'odds']:
            if db_type in all_dbs:
                logger.info(f"✅ Database '{db_type}' exists")
                
                # Check collections
                collection_names = await asyncio.to_thread(lambda: db_manager._dbs[db_type].list_collection_names())
                logger.info(f"   Collections: {collection_names}")
                
                # Monthly collection we're working with
                month_collection_name = f"month_{month}"
                if month_collection_name in collection_names:
                    logger.info(f"✅ Collection '{month_collection_name}' exists")
                    # Count documents
                    day_prefix = f"day_{day}"
                    count = await asyncio.to_thread(lambda: db_manager._dbs[db_type][month_collection_name].count_documents(
                        {"_id": {"$regex": f"^{day_prefix}"}}
                    ))
                    logger.info(f"   - Contains {count} documents for day {day}")
                else:
                    logger.warning(f"❌ Collection '{month_collection_name}' does not exist")
            else:
                logger.warning(f"❌ Database '{db_type}' does not exist")
        
        # Get day summary
        day_summary = db_manager.get_day_summary(date_str)
        if day_summary:
            logger.info(f"\nDay summary: {day_summary}")
        
        return results
        
    finally:
        # Cleanup after running
        db_manager.close_connection()
        logger.info("Database connection closed")

async def main():
    """Main function to execute the data fetcher test"""
    # Set a specific date for testing (April 1, 2025)
    test_date = datetime(2025, 4, 1, tzinfo=timezone.utc)
    
    # You can specify a fixture ID if known, otherwise the test will try to find one
    test_fixture_id = "1213657"  # Change this to a known fixture ID or set to None
    
    logger.info(f"Running database test for date: {test_date} with fixture: {test_fixture_id}")
    await test_single_fixture(date=test_date, fixture_id=test_fixture_id, force_reprocess=True)

if __name__ == "__main__":
    asyncio.run(main())