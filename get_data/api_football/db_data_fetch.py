import os
import json
import hashlib
import asyncio
import time
import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Set, Tuple

# Import required modules from endpoints
from get_data.api_football.db_sqlite import SQLiteDBManager, db_manager
from get_data.api_football.endpoints.api_manager import api_manager
from get_data.api_football.endpoints.game_scraper import GameScraper
from get_data.api_football.endpoints.match_processor import MatchProcessor
from get_data.api_football.endpoints.odds_fetcher import OddsFetcher
from get_data.api_football.endpoints.team_fixtures import TeamFixturesFetcher
from get_data.api_football.endpoints.fixture_details import FixtureDetailsFetcher

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class UnifiedDataFetcher:
    """Unified data fetcher class for both regular fetching and backtesting."""
    
    def __init__(self, db_path: str = "football_data.db"):
        """
        Initialize the data fetcher.
        
        Args:
            db_path: Path to SQLite database
        """
        # Initialize database
        self.db = db_manager if db_path == "football_data.db" else SQLiteDBManager(db_path)
        
        # Initialize endpoints
        self.game_scraper = GameScraper()
        self.match_processor = MatchProcessor()
        self.odds_fetcher = OddsFetcher()
        self.team_fixtures_fetcher = TeamFixturesFetcher()
        self.fixture_details_fetcher = FixtureDetailsFetcher()
        
        # Initialize API key management
        api_manager.initialize()
        logger.info(f"UnifiedDataFetcher initialized with DB: {db_path}")
        
        # Stats tracking for backtesting
        self.api_calls = 0
        self.new_fixtures_found = 0
        self.new_details_processed = 0
        self.cached_fixtures_count = 0
        self.start_time = None
        self.end_time = None

    # --- Regular Data Fetching Methods ---
    
    async def fetch_daily_data(self, target_date: Optional[datetime] = None, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Fetch all necessary data for a specific date: games, matches, and odds.
        Uses SQLite database for storage.
        
        Args:
            target_date: The date to fetch data for. If None, uses today's date.
            force_reprocess: If True, reprocess data even if it already exists.
            
        Returns:
            dict: Results containing success status and data processing summary.
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc)

        date_str = target_date.strftime("%Y-%m-%d")
        logger.info(f"🗓️ Processing data for: {date_str}")

        # Make sure API manager is initialized
        try:
            api_manager.initialize()
            logger.info("API Manager Initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize API Manager: {str(e)}")
            return {"success": False, "error": "API Manager initialization failed", "date": date_str}

        results = {
            "success": True,
            "date": date_str,
            "steps": {
                "scrape_games": {"success": False, "message": "Not run"},
                "process_matches": {"success": False, "message": "Not run"},
                "fetch_odds": {"success": False, "message": "Not run"}
            },
            "games_data": None, # Store fetched games data for subsequent steps
        }

        # --- Step 1: Get games data --- 
        try:
            logger.info("--- Running Step 1: Scraping Games ---")
            
            # Convert target_date to naive datetime if scraper requires it
            date_obj = target_date.replace(tzinfo=None) if target_date.tzinfo else target_date
            games_data = self.game_scraper.get_games(date_obj)

            # Adapt the game_scraper to use SQLite
            # Assuming the scraper is modified to return data dict but not save directly
            
            if not games_data or games_data.get("total_matches", 0) == 0:
                # In a real implementation, should check SQLite DB for existing data
                logger.warning(f"⚠️ No games found for date {date_str}")
                results["steps"]["scrape_games"]["success"] = True  # Not a failure if no games exist
                results["steps"]["scrape_games"]["message"] = "No games found for this date."
            else:
                # Data scraped successfully
                results["games_data"] = games_data
                results["steps"]["scrape_games"]["success"] = True
                results["steps"]["scrape_games"]["message"] = f"Successfully scraped {games_data.get('total_matches', 0)} games."
                logger.info(f"✅ Step 1 finished: {results['steps']['scrape_games']['message']}")

        except Exception as e:
            logger.error(f"❌ Error in Step 1 (Scraping Games): {str(e)}", exc_info=True)
            results["success"] = False
            results["steps"]["scrape_games"]["success"] = False
            results["steps"]["scrape_games"]["message"] = f"Failed: {str(e)}"
            return results  # Stop processing if games scraping fails critically

        # --- Step 2: Process matches --- 
        if results["steps"]["scrape_games"]["success"] and results["games_data"]:
            try:
                logger.info("--- Running Step 2: Processing Matches ---")
                
                # Call the async function directly using await
                processed_result = await self.match_processor.process_games_data_async(results["games_data"])

                if isinstance(processed_result, dict) and processed_result.get("status") == "success":
                    results["steps"]["process_matches"]["success"] = True
                    results["steps"]["process_matches"]["message"] = "Successfully processed matches."
                    logger.info(f"✅ Step 2 finished: {results['steps']['process_matches']['message']}")
                else:
                    error_msg = processed_result.get("error", "Unknown error during match processing")
                    logger.error(f"❌ Error in Step 2 (Processing Matches): {error_msg}")
                    results["steps"]["process_matches"]["success"] = False
                    results["steps"]["process_matches"]["message"] = f"Failed: {error_msg}"
                    results["success"] = False  # Mark overall as failed if match processing fails

            except Exception as e:
                logger.error(f"❌ Error in Step 2 (Processing Matches): {str(e)}", exc_info=True)
                results["success"] = False
                results["steps"]["process_matches"]["success"] = False
                results["steps"]["process_matches"]["message"] = f"Failed: {str(e)}"

        elif not results["games_data"]:
            logger.warning("⚠️ Skipping Step 2 (Processing Matches): No games data available.")
            results["steps"]["process_matches"]["message"] = "Skipped - No games data."
        else:
            logger.warning("⚠️ Skipping Step 2 (Processing Matches): Step 1 did not succeed.")
            results["steps"]["process_matches"]["message"] = "Skipped - Step 1 failed."

        # --- Step 3: Fetch odds --- 
        if results["steps"]["scrape_games"]["success"]:
            try:
                logger.info("--- Running Step 3: Fetching Odds ---")
                
                # Call the async function directly using await
                odds_result = await self.odds_fetcher.process_daily_report(date_str)

                successful_count = odds_result.get("successful", 0)
                failed_count = odds_result.get("failed", 0)

                # Success if some odds fetched or none needed/found
                if successful_count >= 0 and failed_count == 0:
                    results["steps"]["fetch_odds"]["success"] = True
                    results["steps"]["fetch_odds"]["message"] = f"Processed odds. Successful: {successful_count}, Skipped: {odds_result.get('skipped', 0)}. No failures."
                    logger.info(f"✅ Step 3 finished: {results['steps']['fetch_odds']['message']}")
                elif successful_count > 0 and failed_count > 0:
                    results["steps"]["fetch_odds"]["success"] = True  # Partial success
                    results["steps"]["fetch_odds"]["message"] = f"Processed odds. Successful: {successful_count}, Failed: {failed_count}, Skipped: {odds_result.get('skipped', 0)}."
                    logger.warning(f"⚠️ Step 3: Partial success. Failed to fetch/save odds for {failed_count} fixtures.")
                else:  # Only failures occurred or critical error
                    error_msg = odds_result.get("error", f"Failed for {failed_count} fixtures.")
                    logger.error(f"❌ Error in Step 3 (Fetching Odds): {error_msg}")
                    results["steps"]["fetch_odds"]["success"] = False
                    results["steps"]["fetch_odds"]["message"] = f"Failed: {error_msg}"
                    results["success"] = False  # Mark overall as failed

            except Exception as e:
                logger.error(f"❌ Critical Error in Step 3 (Fetching Odds): {str(e)}", exc_info=True)
                results["success"] = False
                results["steps"]["fetch_odds"]["success"] = False
                results["steps"]["fetch_odds"]["message"] = f"Failed Critically: {str(e)}"

        else:
            logger.warning("⚠️ Skipping Step 3 (Fetching Odds): Step 1 did not succeed.")
            results["steps"]["fetch_odds"]["message"] = "Skipped - Step 1 failed."

        # Final Summary
        logger.info("--- Data Fetching Summary ---")
        logger.info(f"Date: {results['date']}")
        logger.info(f"Overall Success: {results['success']}")
        logger.info(f"Step 1 (Games): Success={results['steps']['scrape_games']['success']}, Message='{results['steps']['scrape_games']['message']}'")
        logger.info(f"Step 2 (Matches): Success={results['steps']['process_matches']['success']}, Message='{results['steps']['process_matches']['message']}'")
        logger.info(f"Step 3 (Odds): Success={results['steps']['fetch_odds']['success']}, Message='{results['steps']['fetch_odds']['message']}'")

        return results

    # --- Backtesting Methods ---
    
    async def fetch_teams_data(self, team_ids: List[int] = None) -> List[int]:
        """Fetch team data if not already provided."""
        if not team_ids:
            logger.info("No team IDs provided, loading from database")
            team_ids = self.db.get_all_team_ids()
            
            if not team_ids:
                logger.warning("No teams found in database. Using default Premier League teams.")
                # Default to top Premier League teams if DB is empty
                team_ids = [
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
                logger.info(f"Using {len(team_ids)} default teams")
                
        return team_ids
    
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
    
    async def process_team(self, team_id: int, seasons: List[int]) -> int:
        """
        Process all fixtures for a specific team across all configured seasons.
        
        Args:
            team_id: Team ID to process
            seasons: List of seasons to process
            
        Returns:
            int: Number of fixtures processed
        """
        logger.info(f"Processing team {team_id}")
        fixtures_processed = 0
        
        for season in seasons:
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
    
    async def run_backtest(self, team_ids: List[int] = None, seasons: List[int] = None, force_refresh: bool = False) -> Dict:
        """
        Run the backtester to fetch and process historical data.
        
        Args:
            team_ids: List of team IDs to process. If None, uses all teams in the database or defaults.
            seasons: List of seasons to process. If None, uses current and previous season.
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
        
        # Set default seasons if none provided
        if not seasons:
            current_year = datetime.now().year
            seasons = [current_year, current_year - 1]
            
        # Ensure we have team IDs
        team_ids = await self.fetch_teams_data(team_ids)
        
        # Process each team
        total_fixtures = 0
        for i, team_id in enumerate(team_ids):
            fixtures = await self.process_team(team_id, seasons)
            total_fixtures += fixtures
            
            # Sleep between teams to avoid rate limiting (except for last team)
            if i + 1 < len(team_ids):
                logger.info(f"Sleeping for 3 seconds between teams ({i+1}/{len(team_ids)} completed)")
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
            "teams_processed": len(team_ids),
            "seasons_processed": seasons,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Data fetcher backtester completed in {duration:.2f} seconds")
        logger.info(f"API calls: {self.api_calls}")
        logger.info(f"Fixtures: {total_fixtures} total, {self.cached_fixtures_count} cached, {self.new_fixtures_found} new/updated")
        logger.info(f"Details: {self.new_details_processed} new, {missing_processed} missing processed")
        
        return result

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Unified data fetcher for football statistics')
    
    # Command mode: backtest or regular
    parser.add_argument('--mode', type=str, choices=['fetch', 'backtest'], default='fetch',
                       help='Operation mode: "fetch" for daily data or "backtest" for historical data')
    
    # Arguments for both modes
    parser.add_argument('--db-path', type=str, default='football_data.db',
                       help='Path to SQLite database file')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing of data even if it already exists')
    
    # Arguments for fetch mode
    parser.add_argument('--date', type=str, 
                       help='Date in YYYY-MM-DD format for fetch mode. Defaults to today\'s date.')
    
    # Arguments for backtest mode
    parser.add_argument('--teams', type=str,
                       help='Comma-separated list of team IDs for backtest mode')
    parser.add_argument('--seasons', type=str,
                       help='Comma-separated list of seasons for backtest mode')
    
    return parser.parse_args()

async def main():
    """Main async entry point with error handling."""
    args = parse_args()
    
    try:
        # Initialize the unified data fetcher
        fetcher = UnifiedDataFetcher(db_path=args.db_path)
        
        if args.mode == 'fetch':
            # Regular data fetching mode
            target_date = None
            if args.date:
                try:
                    target_date = datetime.strptime(args.date, "%Y-%m-%d")
                    target_date = target_date.replace(tzinfo=timezone.utc)
                except ValueError:
                    logger.error(f"Invalid date format: {args.date}. Using today's date.")
            
            results = await fetcher.fetch_daily_data(target_date=target_date, force_reprocess=args.force)
            
            # Save results to a log file
            log_filename = f"fetch_results_{results['date']}.json"
            with open(log_filename, "w") as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Results saved to {log_filename}")
            
        else:
            # Backtest mode
            team_ids = None
            if args.teams:
                try:
                    team_ids = [int(team_id.strip()) for team_id in args.teams.split(',')]
                except ValueError:
                    logger.error(f"Invalid team IDs: {args.teams}. Using default teams.")
            
            seasons = None
            if args.seasons:
                try:
                    seasons = [int(season.strip()) for season in args.seasons.split(',')]
                except ValueError:
                    logger.error(f"Invalid seasons: {args.seasons}. Using default seasons.")
            
            results = await fetcher.run_backtest(
                team_ids=team_ids,
                seasons=seasons,
                force_refresh=args.force
            )
            
            # Save results to a log file
            log_filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_filename, "w") as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Results saved to {log_filename}")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
    finally:
        # Close database connection
        if 'fetcher' in locals() and hasattr(fetcher, 'db'):
            fetcher.db.close()

if __name__ == "__main__":
    asyncio.run(main())