import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any
import asyncio
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, project_root)

from get_data.api_football.db_mongo import db_manager # Import the DB manager

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fetch odds for fixtures from daily report.')
    parser.add_argument('--date', type=str, 
                       help='Date in YYYY-MM-DD format. Defaults to today\'s date.',
                       default=datetime.now().strftime("%Y-%m-%d"))
    return parser.parse_args()

class RateLimiter:
    def __init__(self, calls_per_minute: int = 29):
        self.calls_per_minute = calls_per_minute
        self.interval = 60 / calls_per_minute
        self.last_call_time = 0

    async def wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.interval:
            await asyncio.sleep(self.interval - time_since_last)
        self.last_call_time = time.time()

class OddsFetcher:
    def __init__(self, api_manager=None):
        self.api_base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.bet365_id = "8"  # Bet365 bookmaker ID
        self.rate_limiter = RateLimiter(calls_per_minute=20)
        # Use provided api_manager or fall back to global instance
        from get_data.api_football.endpoints.api_manager import api_manager as global_api_manager
        self.api_manager = api_manager or global_api_manager
        logger.info("OddsFetcher initialized")
        
        # Markets we want to extract
        self.target_markets = [
            "Match Winner", "Home/Away", "Second Half Winner",
            "Goals Over/Under", "Goals Over/Under First Half",
            "HT/FT Double", "Both Teams Score",
            "Exact Score", "Highest Scoring Half",
            "Double Chance", "First Half Winner",
            "Team To Score First", "Team To Score Last",
            "Total - Home", "Total - Away",
            "Double Chance - First Half",
            "Odd/Even", "Odd/Even First Half",
            "Home Odd/Even", "Results/Both Teams Score",
            "Goals Over/Under Second Half",
            "Clean Sheet - Home", "Clean Sheet - Away",
            "Win to Nil - Home", "Win to Nil - Away",
            "Win Both Halves", "Double Chance - Second Half",
            "Both Teams Score - First Half",
            "Both Teams Score - Second Half",
            "Win Both Halves - Home",
            "Team To Score - Home", "Team To Score - Away"
        ]

    def _get_fixtures_from_games(self, date_str: str) -> List[Tuple[str, str, str, str]]:
        """
        Get fixture IDs from the daily_games collection in MongoDB using the new hierarchical structure.
        Returns a list of tuples: (fixture_id, home_team, away_team, league_name)
        """
        try:
            logger.info(f"Fetching games data for {date_str} from MongoDB...")
            games_data = db_manager.get_daily_games(date_str)
            
            if not games_data or not games_data.get("leagues"):
                logger.warning(f"No games data found in MongoDB for date {date_str}")
                return []
            
            fixtures = []
            
            # Iterate through leagues and matches in the retrieved data
            for league_id, league_info in games_data.get("leagues", {}).items():
                league_name = league_info.get("name", "Unknown League")
                for match in league_info.get("matches", []):
                    fixture_id = match.get("id")
                    home_team = match.get("home_team", {}).get("name", "Unknown Home")
                    away_team = match.get("away_team", {}).get("name", "Unknown Away")
                    
                    if fixture_id:
                        fixtures.append((str(fixture_id), home_team, away_team, league_name))
                        logger.debug(f"Found fixture from DB: {home_team} vs {away_team} (ID: {fixture_id})")
                    else:
                         logger.warning("Found match entry missing fixture ID in DB data")
            
            if not fixtures:
                logger.warning(f"No valid fixtures extracted from MongoDB data for {date_str}")
            else:
                logger.info(f"Found {len(fixtures)} unique fixtures from MongoDB for {date_str}")
            
            return fixtures
            
        except Exception as e:
            logger.error(f"Error getting fixtures from MongoDB: {str(e)}")
            return []

    async def _make_api_request(self, endpoint: str, params: Dict, retry_count: int = 3) -> Dict:
        """Make API request with rate limit handling and retries."""
        await self.rate_limiter.wait()
        url = f"{self.api_base_url}/{endpoint}"
        
        logger.info(f"Making API request to {url}")
        logger.debug(f"Params: {params}")
        
        for attempt in range(retry_count):
            try:
                # Get active API key and headers from API manager
                _, headers = self.api_manager.get_active_api_key()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 429:  # Rate limit hit
                            logger.warning("Rate limit hit, rotating API key...")
                            self.api_manager.handle_rate_limit(headers["x-rapidapi-key"])
                            # Retry with new key
                            _, new_headers = self.api_manager.get_active_api_key()
                            async with session.get(url, headers=new_headers, params=params) as retry_response:
                                if retry_response.status == 200:
                                    return await retry_response.json()
                        
                        elif response.status == 200:
                            return await response.json()
                        
                        response.raise_for_status()
                
            except aiohttp.ClientError as e:
                logger.error(f"API request error (attempt {attempt + 1}/{retry_count}): {str(e)}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{retry_count}): {str(e)}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                
        return {"errors": ["All retry attempts failed"]}

    async def fetch_odds(self, fixture_id: str) -> Dict:
        """Fetch odds for a specific fixture."""
        logger.info(f"Fetching odds for fixture {fixture_id}...")
        
        # Make sure fixture_id is a string
        params = {"fixture": str(fixture_id)}
        
        # Add bookmaker filter if needed
        if self.bet365_id:
            params["bookmaker"] = self.bet365_id
            
        return await self._make_api_request('odds', params)

    async def _save_odds_data(self, date_str: str, fixture_id: str, raw_odds_response: List[Dict]) -> bool:
        """
        Save odds data to the dedicated odds collection using the new database structure.
        Also adds a reference to the match document.
        """
        try:
            # Prepare the odds payload
            odds_payload = {
                "fixture_id": str(fixture_id),
                "match_date_str": date_str,
                "bookmakers": raw_odds_response,
                "retrieved_at_utc": datetime.now(timezone.utc)
            }
            
            # Save to the dedicated odds collection
            odds_saved = db_manager.save_odds_data(date_str, fixture_id, odds_payload)
            
            if not odds_saved:
                logger.error(f"Failed to save odds data for fixture {fixture_id} via DB Manager")
                return False
                
            # Update the match document with a reference flag/timestamp
            match_data = db_manager.get_match_data(str(fixture_id))
            if match_data:
                update_fields = {
                    "has_odds": True,
                    "odds_processed_at_utc": datetime.now(timezone.utc)
                 }
                # Save the updated match data using the main save method
                match_data.update(update_fields)
                match_save_success = db_manager.save_match_data(match_data)
                if not match_save_success:
                     logger.warning(f"Failed to update match document {fixture_id} with odds flag.")
                     # Continue anyway, as odds are saved in their collection
            else:
                 logger.warning(f"Match data not found for {fixture_id} when trying to update odds flag.")

            return True
        except Exception as e:
            logger.error(f"Error saving odds data or updating match for fixture {fixture_id}: {str(e)}", exc_info=True)
            return False

    async def process_fixtures_odds(self, fixture_ids: list[int], force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Processes a list of fixture IDs, fetching and saving odds to the 'odds' collection.

        Args:
            fixture_ids: A list of integer fixture IDs to process.
            force_reprocess: If True, re-fetches and updates odds even if they exist.

        Returns:
            A dictionary containing processing statistics.
        """
        processed_count = 0
        skipped_count = 0
        failed_fixtures = []

        logger.info(f"Starting odds processing for {len(fixture_ids)} fixtures.")

        for fixture_id_int in fixture_ids:
            fixture_id = str(fixture_id_int) # Use string ID internally
            try:
                # Check if odds already exist
                existing_odds = db_manager.get_odds_data(fixture_id)
                if existing_odds and not force_reprocess:
                    logger.info(f"Odds already exist for fixture {fixture_id} and force_reprocess=False. Skipping.")
                    skipped_count += 1
                    continue

                # Fetch the corresponding match data to get the date string
                # This is needed for saving odds with the correct date context
                match_data = db_manager.get_match_data(fixture_id)
                if not match_data:
                    logger.warning(f"Match data not found for fixture {fixture_id}. Cannot determine date for odds. Skipping.")
                    failed_fixtures.append(fixture_id)
                    continue
                date_str = match_data.get("date_str")
                if not date_str:
                    logger.warning(f"Match data for fixture {fixture_id} missing 'date_str'. Skipping odds fetch.")
                    failed_fixtures.append(fixture_id)
                    continue

                # Fetch odds from API
                logger.info(f"Fetching odds for fixture {fixture_id} (Date: {date_str})...")
                odds_response = await self.fetch_odds(fixture_id)

                if odds_response.get("errors") or not odds_response.get("response"):
                    # Handle API errors or empty responses
                    error_msg = odds_response.get('errors', ['No response data'])
                    # Check if the error indicates odds are not available (e.g., specific message or empty response)
                    # API might return empty response if odds aren't posted yet, which isn't strictly a failure
                    if not odds_response.get("response"):
                        logger.warning(f"No odds data returned by API for fixture {fixture_id}. Might be too early or not offered.")
                        # Decide whether to count this as skipped or failed. Let's skip.
                        skipped_count += 1
                    else:
                        logger.error(f"API error fetching odds for fixture {fixture_id}: {error_msg}")
                        failed_fixtures.append(fixture_id)
                    continue

                # Get the raw response list (contains bookmaker data)
                raw_odds_list = odds_response.get("response", [])

                # Save the odds data using the helper method
                # Pass the raw list, let the save method structure the payload
                if await self._save_odds_data(date_str, fixture_id, raw_odds_list):
                    logger.info(f"Successfully fetched and saved odds for fixture {fixture_id}")
                    processed_count += 1
                else:
                    logger.error(f"Failed during saving process for odds of fixture {fixture_id}")
                    failed_fixtures.append(fixture_id)

            except Exception as e:
                logger.error(f"Unexpected error processing odds for fixture ID {fixture_id_int}: {e}", exc_info=True)
                failed_fixtures.append(str(fixture_id_int)) # Add to failed list

        logger.info(f"Finished odds processing. Processed: {processed_count}, Skipped: {skipped_count}, Failed: {len(failed_fixtures)}")
        return {
            "processed_count": processed_count,
            "skipped_count": skipped_count,
            "failed_fixtures": failed_fixtures
        }

    async def process_daily_report(self, date_str: str) -> Dict[str, Any]:
        """
        Process fixtures from MongoDB for the given date and save the odds data to 
        the dedicated odds collection. Uses the new hierarchical database structure.

        Args:
            date_str: The date string in YYYY-MM-DD format

        Returns:
            A dictionary summarizing the results.
        """
        try:
            logger.info(f"Processing odds for fixtures on {date_str}...")
            
            # Get fixtures from the daily games data in MongoDB
            all_fixtures_from_db = self._get_fixtures_from_games(date_str)
            
            # Prepare stats for return
            stats = {
                "successful": 0,
                "failed": 0,
                "skipped": 0
            }

            if not all_fixtures_from_db:
                logger.warning("No fixtures found to process odds for.")
                return stats
            
            for fixture_id, home_team, away_team, league_name in all_fixtures_from_db:
                try:
                    # Check if match exists
                    if not db_manager.check_match_exists(date_str, fixture_id):
                        logger.warning(f"Match data not found for fixture {fixture_id}, skipping odds fetch")
                        stats["skipped"] += 1
                        continue
                    
                    # Check if odds already exist for this fixture
                    odds_data = db_manager.get_odds_data(date_str, fixture_id)
                    if odds_data:
                        logger.info(f"Odds data already exists for fixture {fixture_id}, skipping...")
                        stats["skipped"] += 1
                        continue
                    
                    # Fetch odds
                    logger.info(f"Processing {home_team} vs {away_team} (ID: {fixture_id})")
                    odds_response = await self.fetch_odds(fixture_id)
                    
                    if odds_response.get("errors"):
                        logger.error(f"API error fetching odds for fixture {fixture_id}: {odds_response.get('errors')}")
                        stats["failed"] += 1
                        continue
                    
                    # Get the raw response list
                    raw_odds_list = odds_response.get("response", [])
                    
                    if not raw_odds_list:
                        logger.warning(f"No odds response data found for fixture {fixture_id}")
                        # Consider if this is a failure or just no odds available
                        stats["failed"] += 1 # Count as failed if no odds are returned by API
                        continue
                        
                    # Save the odds data using the new method for dedicated odds collection
                    if await self._save_odds_data(date_str, fixture_id, raw_odds_list):
                        logger.info(f"Successfully saved odds API response for fixture {fixture_id}")
                        stats["successful"] += 1
                    else:
                        logger.error(f"Failed to save odds data for fixture {fixture_id}")
                        stats["failed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing odds for fixture {fixture_id}: {str(e)}")
                    stats["failed"] += 1
            
            logger.info(f"Odds processing complete for {date_str}")
            logger.info(f"Successful: {stats['successful']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")
            return stats
            
        except Exception as e:
            logger.error(f"Critical error in odds processing: {str(e)}")
            return {"error": str(e), "successful": 0, "failed": 0, "skipped": 0}

async def main():
    """Run as a standalone script."""
    args = parse_args()
    
    try:
        # Initialize required services if running standalone
        from get_data.api_football.endpoints.api_manager import api_manager
        api_manager.initialize()
        
        odds_fetcher = OddsFetcher()
        date_str = args.date
        
        logger.info(f"Starting odds fetching for date: {date_str}")
        results = await odds_fetcher.process_daily_report(date_str)
        
        logger.info("Odds fetching complete:")
        logger.info(f"Successful: {results.get('successful', 0)}")
        logger.info(f"Failed: {results.get('failed', 0)}")
        logger.info(f"Skipped: {results.get('skipped', 0)}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    finally:
        # Close DB connection
        db_manager.close_connection()

if __name__ == "__main__":
    asyncio.run(main())