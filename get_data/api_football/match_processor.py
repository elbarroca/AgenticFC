import os
import sys
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
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

from src.betting.utils.api_manager import api_manager  # Import the API manager

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
    def __init__(self, base_dir: str = "data"):
        """Initialize MatchProcessor with base directory for data storage."""
        self.base_dir = Path(base_dir)
        self._setup_directory_structure()
        self.api_base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.rate_limiter = RateLimiter(calls_per_minute=25)
        logger.info(f"Initialized MatchProcessor with base directory: {self.base_dir}")

    def _setup_directory_structure(self):
        """Create base directory structure if it doesn't exist."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory structure at {self.base_dir}")
        except Exception as e:
            logger.error(f"Error creating directory structure: {str(e)}")
            raise

    def _get_date_path(self, date_str: str) -> Path:
        """Get path for specific date."""
        date = datetime.strptime(date_str, "%Y-%m-%d")
        path = self.base_dir / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}"
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generated date path: {path}")
        return path

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize string to be used as a folder name."""
        sanitized = "".join(c if c.isalnum() or c.isspace() else "_" for c in name)
        sanitized = "_".join(filter(None, sanitized.split("_")))
        return sanitized

    def _get_league_path(self, base_path: Path, league_name: str) -> Path:
        """Get path for league."""
        sanitized_name = self._sanitize_filename(league_name)
        league_path = base_path / sanitized_name
        league_path.mkdir(exist_ok=True)
        return league_path

    def _get_match_filename(self, home_team: str, away_team: str) -> str:
        """Generate match filename from team names."""
        home = self._sanitize_filename(home_team)
        away = self._sanitize_filename(away_team)
        return f"{home}_vs_{away}.json"

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
        logger.info(f"\nFetching standings for league {league_id}...")
        return await self._make_api_request('standings', {
            'league': league_id,
            'season': season
        })

    async def _fetch_team_statistics(self, team_id: str, league_id: str, season: int) -> Dict:
        """Fetch team statistics for the current season."""
        logger.info(f"\nFetching team statistics for team {team_id}...")
        return await self._make_api_request('teams/statistics', {
            'team': team_id,
            'league': league_id,
            'season': season
        })

    async def _fetch_predictions(self, fixture_id: str) -> Dict:
        """Fetch predictions for a specific fixture."""
        logger.info(f"\nFetching predictions for fixture {fixture_id}...")
        return await self._make_api_request('predictions', {
            'fixture': fixture_id
        })

    async def process_games_file_async(self, games_file: str):
        """Process all matches from a games file asynchronously."""
        try:
            logger.info(f"\nLoading games file: {games_file}")
            with open(games_file, "r") as f:
                games_data = json.load(f)
            
            date = games_data.get("date")
            if not date:
                logger.error("Error: No date found in games data")
                return
                
            base_path = self._get_date_path(date)
            
            # Save original games data
            with open(base_path / "games.json", "w") as f:
                json.dump(games_data, f, indent=2)
            
            # Get current season
            current_date = datetime.strptime(date, "%Y-%m-%d")
            season = current_date.year if current_date.month > 6 else current_date.year - 1
            
            # Process each league
            for league_id, league_data in games_data.get("leagues", {}).items():
                league_name = league_data.get("name", "Unknown_League")
                logger.info(f"\nProcessing league: {league_name}")
                
                # Create league directory
                league_path = self._get_league_path(base_path, league_name)
                
                # Fetch and save league standings
                standings_data = await self._fetch_league_standings(league_id, season)
                if not standings_data.get("errors"):
                    with open(league_path / "standings.json", "w") as f:
                        json.dump(standings_data, f, indent=2)
                
                # Process each match in the league
                for match in league_data.get("matches", []):
                    match_id = match.get("id")
                    if not match_id:
                        logger.info("Skipping match - no match ID found")
                        continue
                    
                    # Get team names
                    home_team = match.get("home_team", {})
                    away_team = match.get("away_team", {})
                    home_team_name = home_team.get("name", "Unknown_Home")
                    away_team_name = away_team.get("name", "Unknown_Away")
                    logger.info(f"\nProcessing match: {home_team_name} vs {away_team_name}")
                    
                    # Create match data structure
                    match_data = {
                        "match_info": match,
                        "predictions": None,
                        "home_team_stats": None,
                        "away_team_stats": None
                    }
                    
                    # Only fetch predictions and stats if match hasn't started
                    if not match.get("status", {}).get("started", False):
                        # Fetch predictions
                        predictions_data = await self._fetch_predictions(match_id)
                        if not predictions_data.get("errors"):
                            match_data["predictions"] = predictions_data
                        
                        # Fetch team statistics
                        for team_type, team in [("home_team_stats", home_team), ("away_team_stats", away_team)]:
                            team_id = team.get("id")
                            if team_id:
                                team_stats = await self._fetch_team_statistics(team_id, league_id, season)
                                if not team_stats.get("errors"):
                                    match_data[team_type] = team_stats
                    
                    # Save match data with team names in filename
                    match_filename = self._get_match_filename(home_team_name, away_team_name)
                    with open(league_path / match_filename, "w") as f:
                        json.dump(match_data, f, indent=2)
                    
                    logger.info(f"Completed processing match {match_id}")
                    # No fixed sleep here - rate limiter handles delays
                
                logger.info(f"Completed processing league {league_name}")
                # No fixed sleep here - rate limiter handles delays

        except Exception as e:
            logger.error(f"Error processing games file: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # Re-raise the exception to be caught by the workflow

    async def process_games_data_async(self, games_data: Dict, base_path: Path) -> Dict[str, Any]:
        """Process all matches from games data asynchronously."""
        try:
            date = games_data.get("date")
            if not date:
                error_msg = "No date found in games data"
                logger.error(f"❌ {error_msg}")
                return {"status": "error", "error": error_msg}
            
            # Get current season
            current_date = datetime.strptime(date, "%Y-%m-%d")
            season = current_date.year if current_date.month > 6 else current_date.year - 1
            
            # Process each league
            for league_id, league_data in games_data.get("leagues", {}).items():
                league_name = league_data.get("name", "Unknown_League")
                logger.info(f"\nProcessing league: {league_name}")
                
                # Create league directory
                league_path = base_path / league_name
                league_path.mkdir(parents=True, exist_ok=True)
                
                # Fetch and save league standings
                standings_data = await self._fetch_league_standings(league_id, season)
                if not standings_data.get("errors"):
                    with open(league_path / "standings.json", "w") as f:
                        json.dump(standings_data, f, indent=2)
                
                # Process each match in the league
                for match in league_data.get("matches", []):
                    match_id = match.get("id")
                    if not match_id:
                        logger.info("Skipping match - no match ID found")
                        continue
                    
                    # Get team names
                    home_team = match.get("home_team", {})
                    away_team = match.get("away_team", {})
                    home_team_name = home_team.get("name", "Unknown_Home")
                    away_team_name = away_team.get("name", "Unknown_Away")
                    logger.info(f"\nProcessing match: {home_team_name} vs {away_team_name}")
                    
                    # Create match data structure
                    match_data = {
                        "match_info": match,
                        "predictions": None,
                        "home_team_stats": None,
                        "away_team_stats": None
                    }
                    
                    # Only fetch predictions and stats if match hasn't started
                    if not match.get("status", {}).get("started", False):
                        # Fetch predictions
                        predictions_data = await self._fetch_predictions(match_id)
                        if not predictions_data.get("errors"):
                            match_data["predictions"] = predictions_data
                        
                        # Fetch team statistics
                        for team_type, team in [("home_team_stats", home_team), ("away_team_stats", away_team)]:
                            team_id = team.get("id")
                            if team_id:
                                team_stats = await self._fetch_team_statistics(team_id, league_id, season)
                                if not team_stats.get("errors"):
                                    match_data[team_type] = team_stats
                    
                    # Save match data with team names in filename
                    match_filename = self._get_match_filename(home_team_name, away_team_name)
                    with open(league_path / match_filename, "w") as f:
                        json.dump(match_data, f, indent=2)
                    
                    logger.info(f"✅ Completed processing match {match_id}")
                    # No fixed sleep here - rate limiter handles delays
                
                logger.info(f"✅ Completed processing league {league_name}")
                # No fixed sleep here - rate limiter handles delays

            return {"status": "success"}

        except Exception as e:
            error_msg = f"Error processing games data: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
            return {"status": "error", "error": error_msg}

async def main():
    """Main function to process matches."""
    try:
        logger.info("Starting match processor...")
        processor = MatchProcessor()
        
        # Log initial API usage
        logger.info("Current API Usage Stats:")
        for key, stats in api_manager.get_usage_stats().items():
            logger.info(f"API Key {key}:")
            logger.info(f"  - Requests Made: {stats['requests_made']}")
            logger.info(f"  - Requests Remaining: {stats['requests_remaining']}")
            logger.info(f"  - Time until reset: {stats['time_until_reset']}")
        
        # Process the specific games file we have
        games_file = "data/2025/01/07/games.json"
        
        if not Path(games_file).exists():
            logger.error(f"Error: Games file not found at {games_file}")
            return
        
        await processor.process_games_file_async(str(games_file))
        
        # Log final API usage
        logger.info("\nFinal API Usage Stats:")
        for key, stats in api_manager.get_usage_stats().items():
            logger.info(f"API Key {key}:")
            logger.info(f"  - Requests Made: {stats['requests_made']}")
            logger.info(f"  - Requests Remaining: {stats['requests_remaining']}")
        
        logger.info("\nMatch processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 