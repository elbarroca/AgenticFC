import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any
from pathlib import Path
import asyncio
import aiohttp
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, project_root)

from get_data.api_football.endpoints.api_manager import api_manager  # Import the API manager
from get_data.api_football.db_mongo import db_manager # Import the DB manager

class RateLimiter:
    def __init__(self, calls_per_minute: int = 29):
        self.calls_per_minute = calls_per_minute
        self.interval = 60 / calls_per_minute  # Time between calls in seconds
        self.last_call_time = 0

    async def wait(self):
        """Wait appropriate time to maintain rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.interval:
            await asyncio.sleep(self.interval - time_since_last)
        self.last_call_time = time.time()

class MatchProcessor:
    def __init__(self):
        """Initialize MatchProcessor."""
        self.api_base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.rate_limiter = RateLimiter(calls_per_minute=25)
        logger.info(f"Initialized MatchProcessor")

    async def _make_api_request(self, endpoint: str, params: Dict, retry_count: int = 3) -> Dict:
        """Make API request with rate limit handling and retries."""
        await self.rate_limiter.wait()  # Wait appropriate time before making request
        url = f"{self.api_base_url}/{endpoint}"
        logger.info(f"Making API request to {url} with params {params}")
        
        for attempt in range(retry_count):
            try:
                # Get active API key and headers from API manager
                _, headers = api_manager.get_active_api_key()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 429:  # Rate limit hit
                            logger.warning("Rate limit hit, rotating API key...")
                            api_manager.handle_rate_limit(headers["x-rapidapi-key"])
                            # Retry with new key
                            _, new_headers = api_manager.get_active_api_key()
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

    async def _fetch_league_standings(self, league_id: str, season: int) -> Dict:
        """Fetch standings data for a league."""
        logger.info(f"Fetching standings for league {league_id}...")
        return await self._make_api_request('standings', {
            'league': league_id,
            'season': season
        })

    async def _fetch_team_statistics(self, team_id: str, league_id: str, season: int) -> Dict:
        """Fetch team statistics for the current season."""
        logger.info(f"Fetching team statistics for team {team_id}...")
        return await self._make_api_request('teams/statistics', {
            'team': team_id,
            'league': league_id,
            'season': season
        })

    async def _fetch_predictions(self, fixture_id: str) -> Dict:
        """Fetch predictions for a specific fixture."""
        logger.info(f"Fetching predictions for fixture {fixture_id}...")
        return await self._make_api_request('predictions', {
            'fixture': fixture_id
        })

    def _optimize_team_stats(self, raw_stats: Dict) -> Dict:
        """Extract and optimize relevant team statistics."""
        if not raw_stats or not raw_stats.get("response"):
            return {}
            
        stats = raw_stats.get("response", {})
        
        # Extract only needed stats to reduce size
        return {
            "form": stats.get("form", ""),
            "fixtures": {
                "played": stats.get("fixtures", {}).get("played", {}),
                "wins": stats.get("fixtures", {}).get("wins", {}),
                "draws": stats.get("fixtures", {}).get("draws", {}),
                "loses": stats.get("fixtures", {}).get("loses", {})
            },
            "goals": {
                "for": stats.get("goals", {}).get("for", {}),
                "against": stats.get("goals", {}).get("against", {})
            },
            "biggest": {
                "streak": stats.get("biggest", {}).get("streak", {}),
                "wins": stats.get("biggest", {}).get("wins", {}),
                "loses": stats.get("biggest", {}).get("loses", {})
            },
            "clean_sheet": stats.get("clean_sheet", {}),
            "failed_to_score": stats.get("failed_to_score", {})
        }
        
    def _optimize_predictions(self, raw_predictions: Dict) -> Dict:
        """Extract and optimize prediction data."""
        if not raw_predictions or not raw_predictions.get("response"):
            return {}
            
        predictions = raw_predictions.get("response", [])[0] if raw_predictions.get("response") else {}
        
        # Extract only relevant prediction data
        return {
            "comparison": predictions.get("comparison", {}),
            "predictions": {
                "winner": predictions.get("predictions", {}).get("winner", {}),
                "win_or_draw": predictions.get("predictions", {}).get("win_or_draw", False),
                "under_over": predictions.get("predictions", {}).get("under_over", ""),
                "goals": predictions.get("predictions", {}).get("goals", {}),
                "advice": predictions.get("predictions", {}).get("advice", "")
            },
            "h2h": [] # We'll exclude h2h to save space
        }

    async def process_fixtures(self, fixture_ids: list[int], force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Processes a list of fixture IDs, fetching details (predictions, stats, standings)
        and saving them to the dedicated 'match_processor' collection.

        Args:
            fixture_ids: A list of integer fixture IDs to process.
            force_reprocess: If True, re-fetches and updates data even if it exists in 'match_processor'.

        Returns:
            A dictionary containing processing statistics.
        """
        processed_count = 0
        skipped_count = 0
        failed_fixtures = []
        standings_cache = {} # Cache standings per league/season

        logger.info(f"Starting processing for {len(fixture_ids)} fixtures, saving to 'match_processor' collection.")

        for fixture_id_int in fixture_ids:
            fixture_id = str(fixture_id_int) # Use string ID internally and for DB _id
            try:
                # 1. Check if data already exists in the target collection
                if not force_reprocess and db_manager.check_match_processor_data_exists(fixture_id):
                    logger.info(f"Match processor data already exists for fixture {fixture_id} and force_reprocess=False. Skipping.")
                    skipped_count += 1
                    continue

                # 2. Get base match info from the 'matches' collection
                base_match_data = db_manager.get_match_data(fixture_id)

                if not base_match_data:
                    logger.warning(f"Base match data not found in 'matches' collection for fixture {fixture_id}. Skipping processing.")
                    # This fixture likely wasn't scraped correctly initially.
                    failed_fixtures.append(fixture_id)
                    continue

                # 3. Check if the match is finished -- REMOVED THIS CHECK
                # is_finished = base_match_data.get("match_info", {}).get("status", {}).get("short") in ["FT", "AET", "PEN"]
                # if not is_finished:
                #      logger.info(f"Skipping fixture {fixture_id}: Match not finished (Status: {base_match_data.get('match_info', {}).get('status', {}).get('short')}).")
                #      skipped_count += 1
                #      continue

                # logger.info(f"Processing details for finished fixture {fixture_id}...") # Adjusted log message
                logger.info(f"Processing details for fixture {fixture_id} (Status: {base_match_data.get('match_info', {}).get('status', {}).get('short', 'N/A')})...")

                # 4. Extract necessary info for API calls
                league_id = base_match_data.get("league_id")
                home_team_id = base_match_data.get("home_team", {}).get("id")
                away_team_id = base_match_data.get("away_team", {}).get("id")
                match_date_str = base_match_data.get("date_str")

                if not all([league_id, home_team_id, away_team_id, match_date_str]):
                     logger.warning(f"Skipping fixture {fixture_id}: Missing essential IDs or date in base match data.")
                     failed_fixtures.append(fixture_id)
                     continue

                # Determine season
                try:
                    match_dt = datetime.strptime(match_date_str, "%Y-%m-%d")
                    season = match_dt.year if match_dt.month >= 7 else match_dt.year - 1
                except ValueError:
                     logger.warning(f"Could not parse date {match_date_str} for fixture {fixture_id} to determine season. Skipping.")
                     failed_fixtures.append(fixture_id)
                     continue

                # 5. Fetch API data (Predictions, Stats, Standings)
                raw_predictions_response = None
                raw_home_stats_response = None
                raw_away_stats_response = None
                current_league_standings_response = None # This is the *API response* list for standings

                try:
                    # Fetch Predictions
                    predictions_data = await self._fetch_predictions(fixture_id)
                    raw_predictions_response = predictions_data.get("response") # List containing prediction dict

                    # Fetch Team Stats
                    home_stats_data = await self._fetch_team_statistics(home_team_id, league_id, season)
                    raw_home_stats_response = home_stats_data.get("response") # Dict containing home stats

                    away_stats_data = await self._fetch_team_statistics(away_team_id, league_id, season)
                    raw_away_stats_response = away_stats_data.get("response") # Dict containing away stats

                    # Fetch/Cache Standings (Still save raw standings to 'standings' collection for historical snapshots)
                    standings_key = f"{league_id}_{season}"
                    if standings_key not in standings_cache:
                        standings_data = await self._fetch_league_standings(league_id, season)
                        standings_cache[standings_key] = standings_data.get("response") # Cache raw API response list
                        # Save raw standings data to its dedicated collection
                        if standings_cache[standings_key]:
                             standings_payload = {
                                 "league_id": str(league_id),
                                 "season": season,
                                 "date_retrieved_str": match_date_str, # Use match date for context
                                 "standings_api_response": standings_cache[standings_key]
                             }
                             # Still save to the main standings collection
                             db_manager.save_standings_data(match_date_str, league_id, season, standings_payload)

                    current_league_standings_response = standings_cache.get(standings_key)

                    # 6. Prepare payload for the 'match_processor' collection
                    processor_payload = {
                        "fixture_id": fixture_id, # Keep fixture_id for easy lookup
                        "match_date_str": match_date_str,
                        "league_id": league_id,
                        "season": season,
                        "home_team_id": home_team_id,
                        "away_team_id": away_team_id,
                        # Store the relevant parts of the API responses
                        "predictions": raw_predictions_response[0] if raw_predictions_response else None,
                        "home_team_stats": raw_home_stats_response if raw_home_stats_response else None,
                        "away_team_stats": raw_away_stats_response if raw_away_stats_response else None,
                        # Store snapshot of league info from standings (optional, could be large)
                        "standings_snapshot": current_league_standings_response[0].get("league", {}) if current_league_standings_response else None,
                        "processed_at_utc": datetime.now(timezone.utc) # Add timestamp
                    }

                    # 7. Save to the 'match_processor' collection
                    success = db_manager.save_match_processor_data(processor_payload)

                    if success:
                        logger.info(f"Successfully saved processor data for fixture {fixture_id} to 'match_processor' collection.")
                        processed_count += 1
                    else:
                        logger.error(f"Failed to save processor data for fixture {fixture_id} to 'match_processor' collection.")
                        failed_fixtures.append(fixture_id)

                except Exception as fetch_err:
                    logger.error(f"Error fetching API details for fixture {fixture_id}: {fetch_err}", exc_info=True)
                    failed_fixtures.append(fixture_id)

            except Exception as outer_err:
                 logger.error(f"Unexpected error processing fixture ID {fixture_id_int}: {outer_err}", exc_info=True)
                 failed_fixtures.append(str(fixture_id_int)) # Add to failed list

        logger.info(f"Finished processing fixtures for 'match_processor' collection. Processed: {processed_count}, Skipped: {skipped_count}, Failed: {len(failed_fixtures)}")
        return {
            "processed_count": processed_count,
            "skipped_count": skipped_count,
            "failed_fixtures": failed_fixtures
        }

    async def process_games_data_async(self, games_data: Dict) -> Dict[str, Any]:
        """Process all matches from games data asynchronously using the new hierarchical DB structure.
           Saves FULL raw API responses for standings, stats, and predictions.
           DEPRECATED: Use process_fixtures instead, driven by fixture IDs from GameScraper/daily_games.
        """
        logger.warning("process_games_data_async is deprecated. Use process_fixtures.")
