import os
import sys
import json
import time
import requests
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, project_root)

from src.betting.utils.api_manager import api_manager  # Import the API manager

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
    def __init__(self, base_dir: str = "data", api_manager=None):
        self.base_dir = Path(base_dir)
        self.api_base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.bet365_id = "8"  # Bet365 bookmaker ID
        self.rate_limiter = RateLimiter(calls_per_minute=20)
        # Use provided api_manager or fall back to global instance
        from src.betting.utils.api_manager import api_manager as global_api_manager
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
        Get fixture IDs from games.json files in the daily folder.
        Returns a list of tuples: (fixture_id, home_team, away_team, league)
        """
        try:
            # Get date path
            date = datetime.strptime(date_str, "%Y-%m-%d")
            date_path = self.base_dir / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}"
            
            if not date_path.exists():
                print(f"No data directory found for date {date_str}")
                return []
            
            fixtures = set()  # Use set to prevent duplicates
            
            # Iterate through league directories
            for league_dir in date_path.iterdir():
                if league_dir.is_dir():
                    # Check for both .md and .json files
                    for md_file in league_dir.glob("*.md"):
                        try:
                            # Read the MD file to get match info
                            with open(md_file, 'r', encoding='utf-8') as f:
                                md_content = f.read()
                                
                            # Extract fixture ID from MD content
                            fixture_id_match = re.search(r'Fixture ID: (\d+)', md_content)
                            if not fixture_id_match:
                                logger.warning(f"No fixture ID found in {md_file}")
                                continue
                                
                            fixture_id = fixture_id_match.group(1)
                            
                            # Extract teams from filename
                            match_name = md_file.stem
                            if " vs " not in match_name:
                                logger.warning(f"Invalid match name format in {md_file}")
                                continue
                                
                            home_team, away_team = match_name.split(" vs ")
                            
                            fixtures.add((fixture_id, home_team, away_team, league_dir.name))
                            logger.info(f"Found fixture: {home_team} vs {away_team} (ID: {fixture_id})")
                            
                        except Exception as e:
                            logger.error(f"Error processing {md_file}: {str(e)}")
                            continue
            
            # Convert set back to list and sort by fixture_id
            fixtures_list = sorted(list(fixtures), key=lambda x: x[0])
            
            if not fixtures_list:
                logger.warning("No valid fixtures found in MD files")
            else:
                logger.info(f"\nFound {len(fixtures_list)} unique fixtures:")
                for fixture_id, home_team, away_team, league in fixtures_list:
                    logger.info(f"  - {home_team} vs {away_team} (ID: {fixture_id}) - {league}")
            
            return fixtures_list
            
        except Exception as e:
            logger.error(f"Error getting fixtures from games: {str(e)}")
            raise

    def _validate_md_file(self, md_path: Path) -> bool:
        """Validate that the MD file has the required structure."""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for required sections
            required_sections = [
                "# ⚽",  # Title
                "Fixture ID:",
                "Competition:",
                "Country:",
                "Date:",
                "Time:",
                "## 📊 Team Statistics",
                "## 🤝 Head-to-Head Analysis",
                "## 📈 Statistical Overview",
                "## ⚔️ Tactical Summary",
                "## 💰 Betting Recommendations"
            ]
            
            for section in required_sections:
                if section not in content:
                    logger.warning(f"Missing required section '{section}' in {md_path}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating MD file {md_path}: {str(e)}")
            return False

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

    def _filter_odds_data(self, odds_data: Dict) -> Dict:
        """Filter odds data to include only target markets."""
        filtered_data = {}
        
        if not odds_data or "response" not in odds_data:
            print("No response data found in odds_data")
            return filtered_data
            
        if not odds_data["response"]:
            print("Empty response array in odds_data")
            return filtered_data
            
        for response in odds_data["response"]:
            bookmakers = response.get("bookmakers", [])
            if not bookmakers:
                print("No bookmakers found in response")
                continue
                
            for bookmaker in bookmakers:
                if str(bookmaker["id"]) == str(self.bet365_id):
                    bets = bookmaker.get("bets", [])
                    if not bets:
                        print("No bets found for bookmaker")
                        continue
                        
                    for bet in bets:
                        if bet["name"] in self.target_markets:
                            filtered_data[bet["name"]] = {
                                "values": bet["values"],
                                "id": bet["id"]
                            }
        
        return filtered_data

    def _get_date_path(self, date_str: str) -> Path:
        """Get path for specific date."""
        date = datetime.strptime(date_str, "%Y-%m-%d")
        path = self.base_dir / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}"
        return path

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by replacing invalid characters."""
        return re.sub(r'[<>:"/\\|?*]', '_', filename)

    async def process_daily_report(self, date_str: str, specific_matches=None):
        """
        Process fixtures from MD files or specific matches and save odds data.
        
        Args:
            date_str: The date string in YYYY-MM-DD format
            specific_matches: Optional list of specific matches to process
                              Each match should be a dict with fixture.id, teams.home.name, teams.away.name
        """
        try:
            logger.info(f"Processing fixtures for {date_str}...")
            
            fixtures = []
            
            # If specific matches are provided, use them
            if specific_matches:
                logger.info(f"Processing {len(specific_matches)} specific matches")
                for match in specific_matches:
                    fixture_id = match.get("fixture", {}).get("id")
                    home_team = match.get("teams", {}).get("home", {}).get("name", "Unknown")
                    away_team = match.get("teams", {}).get("away", {}).get("name", "Unknown")
                    league = match.get("league", {}).get("name", "Unknown")
                    
                    if fixture_id:
                        fixtures.append((fixture_id, home_team, away_team, league))
            else:
                # Otherwise get fixtures from MD files
                fixtures = self._get_fixtures_from_games(date_str)
            
            if not fixtures:
                logger.warning("No valid fixtures found")
                return {
                    "successful": 0,
                    "failed": 0,
                    "successful_fixtures": [],
                    "failed_fixtures": []
                }
            
            # Create odds directory
            date = datetime.strptime(date_str, "%Y-%m-%d")
            date_path = self.base_dir / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}"
            odds_path = date_path / "odds"
            odds_path.mkdir(parents=True, exist_ok=True)
            
            # Process each fixture and save to individual files
            successful_fixtures = []
            failed_fixtures = []
            
            # Log initial API usage
            logger.info("\nInitial API Usage Stats:")
            for key, stats in self.api_manager.get_usage_stats().items():
                logger.info(f"API Key {key}:")
                logger.info(f"  - Requests Made: {stats['requests_made']}")
                logger.info(f"  - Requests Remaining: {stats['requests_remaining']}")
            
            for fixture_id, home_team, away_team, league in fixtures:
                try:
                    logger.info(f"Processing {home_team} vs {away_team} (ID: {fixture_id})")
                    
                    # Create filename based on team names
                    sanitized_home = self._sanitize_filename(home_team)
                    sanitized_away = self._sanitize_filename(away_team)
                    team_filename = f"{sanitized_home}_vs_{sanitized_away}_{fixture_id}.json"
                    
                    # Create output file path
                    output_file = odds_path / team_filename
                    
                    if output_file.exists():
                        logger.info(f"File already exists for fixture {fixture_id}, skipping...")
                        successful_fixtures.append((fixture_id, home_team, away_team))
                        continue
                    
                    # Fetch odds data
                    fixture_odds = await self.fetch_odds(fixture_id)
                    
                    if fixture_odds.get("errors"):
                        logger.error(f"Error fetching odds for fixture {fixture_id}: {fixture_odds['errors']}")
                        failed_fixtures.append((fixture_id, home_team, away_team, str(fixture_odds['errors'])))
                        continue
                    
                    filtered_odds = self._filter_odds_data(fixture_odds)
                    
                    if filtered_odds:
                        # Create fixture data structure
                        fixture_data = {
                            "fixture_id": fixture_id,
                            "date": date_str,
                            "match_info": {
                                "home_team": home_team,
                                "away_team": away_team,
                                "league": league
                            },
                            "odds": filtered_odds
                        }
                        
                        # Save fixture data
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(fixture_data, f, indent=2, ensure_ascii=False)
                        
                        logger.info(f"Saved odds data for fixture {fixture_id} to {output_file}")
                        successful_fixtures.append((fixture_id, home_team, away_team))
                    else:
                        logger.warning(f"No odds data found for fixture {fixture_id}")
                        failed_fixtures.append((fixture_id, home_team, away_team, "No odds data found"))
                    
                except Exception as e:
                    logger.error(f"Error processing fixture {fixture_id}: {str(e)}")
                    failed_fixtures.append((fixture_id, home_team, away_team, str(e)))
            # Log final API usage
            logger.info("\nFinal API Usage Stats:")
            for key, stats in self.api_manager.get_usage_stats().items():
                logger.info(f"API Key {key}:")
                logger.info(f"  - Requests Made: {stats['requests_made']}")
                logger.info(f"  - Requests Remaining: {stats['requests_remaining']}")
            
            return {
                "successful": len(successful_fixtures),
                "failed": len(failed_fixtures),
                "successful_fixtures": successful_fixtures,
                "failed_fixtures": failed_fixtures
            }
            
        except Exception as e:
            logger.error(f"Error processing fixtures: {str(e)}")
            raise

    def process_daily_report_sync(self, date_str: str):
        """Synchronous version of process_daily_report."""
        try:
            # Create event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async function in the loop
            results = loop.run_until_complete(self.process_daily_report(date_str))
            
            # Close the loop
            loop.close()
            
            return results
            
        except Exception as e:
            print(f"Error in process_daily_report_sync: {str(e)}")
            return None

async def main():
    """Main function to process daily report and fetch odds."""
    try:
        args = parse_args()
        date_str = args.date
        
        # Validate date format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.error("Error: Date must be in YYYY-MM-DD format")
            return
        
        logger.info(f"Starting odds fetcher for date: {date_str}")
        fetcher = OddsFetcher()
        
        # Process daily report
        await fetcher.process_daily_report(date_str)
        
        logger.info("Odds fetching completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Error: {str(e)}")
        logger.error("Please ensure the daily report exists for the specified date.")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 