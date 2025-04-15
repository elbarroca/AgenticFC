import sys
import json
import time
from datetime import datetime
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
from api_football.db_mongo import db_manager # Import the DB manager

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

    async def process_games_data_async(self, games_data: Dict) -> Dict[str, Any]:
        """Process all matches from games data asynchronously using the new hierarchical DB structure.
           Saves FULL raw API responses for standings, stats, and predictions.
        """
        try:
            date_str = games_data.get("date")
            if not date_str:
                error_msg = "No date found in games data"
                logger.error(f"❌ {error_msg}")
                return {"status": "error", "error": error_msg}
            
            # Get current season based on the processing date
            current_date = datetime.strptime(date_str, "%Y-%m-%d")
            season = current_date.year if current_date.month > 6 else current_date.year - 1
            
            # Track processing statistics
            stats = {
                "leagues_processed": 0,
                "matches_processed": 0,
                "matches_skipped": 0,
                "errors": 0
            }
            
            standings_cache = {} # Cache for raw standings response per league
            
            for league_id, league_data in games_data.get("leagues", {}).items():
                league_name = league_data.get("name", "Unknown_League")
                logger.info(f"Processing league: {league_name} (ID: {league_id})")
                
                try:
                    # --- Fetch and save league standings (RAW) --- 
                    if league_id not in standings_cache:
                        standings_data = await self._fetch_league_standings(league_id, season)
                        raw_standings_response = standings_data.get("response")
                        standings_cache[league_id] = raw_standings_response # Cache the raw response list
                        
                        if raw_standings_response:
                            # Prepare raw standings payload for the standings collection
                            standings_payload = {
                                # _id is handled by save_standings_data
                                "league_id": str(league_id),
                                "season": season,
                                "date": date_str, 
                                "standings_api_response": raw_standings_response # Store the raw array
                            }
                            # Save raw standings data to its dedicated collection
                            success = db_manager.save_standings_data(date_str, league_id, season, standings_payload)
                            if success:
                                logger.info(f"Saved raw standings for league {league_id} ({season}) to standings collection")
                            else:
                                logger.error(f"Failed to save raw standings for league {league_id} to standings collection")
                        else:
                            logger.warning(f"No standings data available or API error for league {league_id}")
                    
                    # Get cached raw standings for embedding snapshot
                    current_league_standings_response = standings_cache.get(league_id)

                    # --- Process each match in the league --- 
                    for match in league_data.get("matches", []):
                        fixture_id = str(match.get("id", ""))
                        if not fixture_id:
                            logger.warning("Skipping match - no match ID found")
                            stats["errors"] += 1
                            continue
                        
                        # Get team names for logging
                        home_team = match.get("home_team", {})
                        away_team = match.get("away_team", {})
                        home_team_name = home_team.get("name", "Unknown_Home")
                        away_team_name = away_team.get("name", "Unknown_Away")
                        logger.info(f"Processing match: {home_team_name} vs {away_team_name} (ID: {fixture_id})")
                        
                        # Check if match data already exists in the database
                        # NOTE: This check might prevent updates if you run the processor again.
                        # Consider if you want to overwrite or just skip.
                        existing_match = db_manager.get_match_data(date_str, fixture_id)
                        if existing_match and existing_match.get("predictions") and existing_match.get("home_team_stats") and existing_match.get("away_team_stats"):
                             logger.info(f"Stats & Predictions already exist for fixture {fixture_id}. Skipping fetch.")
                             stats["matches_skipped"] += 1
                             continue # Skip if we already have predictions and stats
                        elif existing_match:
                             logger.info(f"Match {fixture_id} exists, will update with stats/predictions.")
                        else:
                             logger.warning(f"Match {fixture_id} not found in DB, cannot add stats/predictions. Should have been created by GameScraper.")
                             stats["errors"] += 1
                             continue # Cannot update a non-existent match
                        
                        # Fetch additional data for the match (RAW)
                        try:
                            # Fetch predictions (RAW)
                            predictions_data = await self._fetch_predictions(fixture_id)
                            raw_predictions_response = predictions_data.get("response")
                            
                            # Fetch team statistics (RAW)
                            home_team_id = str(home_team.get("id", ""))
                            away_team_id = str(away_team.get("id", ""))
                            
                            raw_home_stats_response = None
                            raw_away_stats_response = None
                            
                            if home_team_id:
                                home_stats_data = await self._fetch_team_statistics(home_team_id, league_id, season)
                                raw_home_stats_response = home_stats_data.get("response")
                            
                            if away_team_id:
                                away_stats_data = await self._fetch_team_statistics(away_team_id, league_id, season)
                                raw_away_stats_response = away_stats_data.get("response")
                            
                            # Prepare update for the existing match document
                            update_payload = { 
                                "predictions": raw_predictions_response[0] if raw_predictions_response else None,
                                "home_team_stats": raw_home_stats_response, 
                                "away_team_stats": raw_away_stats_response,
                                # Embed snapshot of standings league info from the cached raw response
                                "standings_snapshot": current_league_standings_response[0].get("league", {}) if current_league_standings_response else {} 
                            }

                            # Merge and save
                            final_match_data = {**existing_match, **update_payload}
                            success = db_manager.save_match_data(date_str, fixture_id, final_match_data)
                            if success:
                                logger.info(f"Updated match data for fixture {fixture_id} with stats/predictions")
                                stats["matches_processed"] += 1
                            else:
                                logger.error(f"Failed to update match data for fixture {fixture_id}")
                                stats["errors"] += 1
                                
                        except Exception as match_e:
                            logger.error(f"Error fetching/updating match {fixture_id}: {str(match_e)}")
                            stats["errors"] += 1
                    
                    stats["leagues_processed"] += 1
                    
                except Exception as league_e:
                    logger.error(f"Error processing league {league_id}: {str(league_e)}")
                    stats["errors"] += 1
            
            # Return processing results
            success = stats["errors"] == 0
            return {
                "status": "success" if success else "partial_failure",
                "date": date_str,
                "stats": stats,
                "message": f"Processed {stats['leagues_processed']} leagues, updated {stats['matches_processed']} matches with {stats['errors']} errors."
            }

        except Exception as e:
            error_msg = f"Error processing games data: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
            return {"status": "error", "error": error_msg} 