import asyncio
import aiohttp
from bs4 import BeautifulSoup
import logging
from urllib.parse import quote
import json
from typing import Dict, List, Optional
import hashlib
import platform
from itertools import islice
import os
import sys
from datetime import datetime, timezone
from pymongo import errors as pymongo_errors
from tqdm.asyncio import tqdm_asyncio

# Add parent directory to sys.path to fix import issues when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from get_data.api_football.db_mongo import db_manager
    from get_data.api_football.db_ids.team_id_mappings import TEAM_ID_MAPPING
except ModuleNotFoundError:
    print("Error: This script should be run from the project root directory")
    print("Try running: python -m get_data.statarea.statarea_async_scraper")
    sys.exit(1)

# Configuration
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
MAX_CONCURRENT_REQUESTS = 8
BASE_DELAY = 1  # seconds between requests
MAX_RETRIES = 5
CACHE_EXPIRE_DAYS = 1
MATCH_HISTORY_LIMIT = 15  # Number of matches to scrape

# Convert the team mappings to the format expected by the scraper
TEAM_DATA = {
    team_name: {
        "api_id": mapping["statarea_id"],
        "country": mapping["country"]
    }
    for team_name, mapping in TEAM_ID_MAPPING.items()
}

progress = {
    'total': 0,
    'completed': 0,
    'successful': 0,
    'failed': 0,
    'skipped': 0,
    'last_team': None
}

# Helper Functions
def setup_logging(logs_dir: Optional[str] = None):
    """Configure logging with optional directory for log files"""
    handlers = [logging.StreamHandler()]
    
    if logs_dir:
        # Create logs directory if it doesn't exist
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, 'scraper_progress.log')
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.FileHandler('scraper_progress.log'))
    
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True  # Force reconfiguration if already configured
    )
    
    logging.info(f"Logging configured with {'external logs dir' if logs_dir else 'local log file'}")

def update_progress(success: bool, team: str):
    progress['completed'] += 1
    progress['last_team'] = team
    if success:
        progress['successful'] += 1
    else:
        progress['failed'] += 1
    
    current_total = progress['total']
    current_attempts = progress['completed'] + progress['skipped']

    log_interval = 10
    if current_attempts % log_interval == 0 or not success or current_attempts == current_total:
        logging.info(
            f"Progress: {current_attempts}/{current_total} | "
            f"Success: {progress['successful']} | "
            f"Failed: {progress['failed']} | "
            f"Skipped: {progress['skipped']} | "
            f"Last: {team}"
        )

def save_checkpoint(logs_dir: Optional[str] = None):
    checkpoint_path = os.path.join(logs_dir, 'progress_checkpoint.json') if logs_dir else 'progress_checkpoint.json'
    with open(checkpoint_path, 'w') as f:
        json.dump(progress, f)

def log_failed_url(url: str, logs_dir: Optional[str] = None):
    failed_urls_path = os.path.join(logs_dir, 'failed_urls.log') if logs_dir else 'failed_urls.log'
    with open(failed_urls_path, 'a') as f:
        f.write(f"{datetime.now().isoformat()},{url}\n")

def construct_url(team: str, game_type: str, country: str, period: int = 10) -> str:
    if period not in [5, 10, 15]:
        raise ValueError("Period must be 5, 10, or 15")
    
    base_url = "https://www.statarea.com/team/view"
    team_with_country = f"{team} ({country})"
    encoded_team = quote(team_with_country)
    return f"{base_url}/{encoded_team}/{game_type}/last{period}"

def extract_general_statistics(soup: BeautifulSoup) -> Dict:
    stats = {}
    container = soup.find('div', class_='teamstatistics')
    
    if container:
        for item in container.find_all('div', class_='factitem'):
            label = item.find('div', class_='label').text.strip()
            value = item.find('div', class_='value').text.strip()
            stats[label] = value
    
    return stats

def extract_team_bet_statistics(soup: BeautifulSoup) -> Dict:
    stats = {}
    container = soup.find('div', class_='teambetstatistics')
    
    if container:
        for chart in container.find_all('div', class_='barchart'):
            title = chart.find('div', class_='title').text.strip()
            stats[title] = {}
            for row in chart.find_all('div', class_='barrow'):
                name = row.find('div', class_='name').text.strip()
                value = row.find('div', class_='bar')['style'].split(':')[1].strip('%;')
                stats[title][name] = f"{value}%"
    
    return stats

def extract_match_history(soup: BeautifulSoup, team_name: str) -> List[Dict]:
    matches = []
    try:
        match_items = soup.select('div.matchitem')
        
        for match in match_items[:MATCH_HISTORY_LIMIT]:
            try:
                competition = match.select_one('div.competition').get_text(strip=True)
                date = match.select_one('div.date').get_text(strip=True)
                match_div = match.select_one('div.match')
                
                home_team = match_div.select_one('div.hostteam')
                home_team_name = home_team.select_one('a').get_text(strip=True)
                home_team_goals = home_team.select_one('div.goals').get_text(strip=True)
                
                away_team = match_div.select_one('div.guestteam')
                away_team_name = away_team.select_one('a').get_text(strip=True)
                away_team_goals = away_team.select_one('div.goals').get_text(strip=True)
                
                if team_name.lower() in home_team_name.lower():
                    venue = 'home'
                    opponent = away_team_name
                    team_goals = home_team_goals
                    opponent_goals = away_team_goals
                else:
                    venue = 'away'
                    opponent = home_team_name
                    team_goals = away_team_goals
                    opponent_goals = home_team_goals
                
                try:
                    team_goals_int = int(team_goals)
                    opponent_goals_int = int(opponent_goals)
                    result = 'win' if team_goals_int > opponent_goals_int else \
                             'loss' if team_goals_int < opponent_goals_int else 'draw'
                except ValueError:
                    result = None
                
                matches.append({
                    'date': date,
                    'competition': competition,
                    'opponent': opponent,
                    'team_goals': team_goals,
                    'opponent_goals': opponent_goals,
                    'result': result,
                    'venue': venue
                })
                
            except Exception as e:
                logging.warning(f"Failed to parse match item: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Error extracting match history: {str(e)}")
    
    logging.info(f"Extracted {len(matches)} matches for {team_name}")
    return matches

def check_needs_update(team: str, country: str, api_id: str, game_type: str, period: int) -> bool:
    """
    Check if team data needs to be updated using MongoDB.
    Returns True if data needs update, is missing, or if the check fails.
    Returns False only if data exists and is recent.
    """
    if not db_manager._initialized or db_manager._statarea_collection is None:
        logging.warning(f"MongoDB connection not initialized during check for {api_id} ({game_type}, {period}). Assuming update needed.")
        return True

    try:
        needs_update = db_manager.check_statarea_data_needs_update(
            api_id=api_id,
            game_type=game_type,
            period=period,
            cache_expire_days=CACHE_EXPIRE_DAYS
        )
        if not needs_update:
            logging.info(f"Skipping {team} ({game_type}, {period}) - recent data found.")
            progress['skipped'] += 1
        return needs_update

    except (pymongo_errors.OperationFailure, pymongo_errors.TimeoutError, pymongo_errors.ConnectionFailure) as op_err:
        logging.warning(f"MongoDB operation error checking update status for {api_id} ({game_type}, {period}): {type(op_err).__name__}. Assuming update needed.")
        return True
    except Exception as e:
        logging.error(f"Unexpected error checking update status for {api_id} ({game_type}, {period}): {str(e)}", exc_info=True)
        return True

def save_to_mongodb(stats: Dict) -> bool:
    """Save scraped data to MongoDB"""
    if not stats:
        return False
        
    if not db_manager._initialized or db_manager._statarea_collection is None:
        logging.error("MongoDB connection not initialized")
        return False
        
    try:
        return db_manager.save_statarea_data(stats)
    except Exception as e:
        logging.error(f"Error saving to MongoDB: {str(e)}")
        return False

async def scrape_team_stats_async(
    session: aiohttp.ClientSession,
    team: str,
    game_type: str,
    country: str,
    api_id: str,
    period: int = 10,
    logs_dir: Optional[str] = None
) -> Optional[Dict]:
    url = construct_url(team, game_type, country, period)
    logging.debug(f"Scraping URL: {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            await asyncio.sleep(BASE_DELAY)
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    current_hash = hashlib.md5(html.encode()).hexdigest()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Add UTC datetime for MongoDB
                    result = {
                        'team': team,
                        'country': country,
                        'api_id': api_id,
                        'game_type': game_type,
                        'period': period,
                        'scrape_date': datetime.now().isoformat(),
                        'scrape_date_utc': datetime.now(timezone.utc),
                        'content_hash': current_hash,
                        'general_statistics': extract_general_statistics(soup),
                        'team_bet_statistics': extract_team_bet_statistics(soup)
                    }
                    
                    if period == 15:
                        matches = extract_match_history(soup, team)
                        if not matches:
                            logging.warning(f"No matches found for {team} at {url}")
                        else:
                            logging.info(f"Found {len(matches)} matches for {team}")
                        result['match_history'] = matches
                    
                    return result
                else:
                    wait_time = min(2 ** (attempt + 1), 10)
                    logging.warning(
                        f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {team}: "
                        f"HTTP {response.status}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {team}: "
                f"{type(e).__name__}"
            )
            if attempt == MAX_RETRIES - 1:
                log_failed_url(url, logs_dir)
                return None
            await asyncio.sleep(BASE_DELAY)
    
    return None

async def scrape_with_progress(
    session: aiohttp.ClientSession,
    team: str,
    data: Dict,
    game_type: str,
    period: int,
    logs_dir: Optional[str] = None,
    force_update: bool = False
) -> Optional[Dict]:
    needs_update = force_update or check_needs_update(team, data['country'], data['api_id'], game_type, period)

    if not needs_update:
        return None

    result = await scrape_team_stats_async(session, team, game_type, data['country'], data['api_id'], period, logs_dir)

    update_progress(result is not None, f"{team} ({game_type}, {period})")

    save_checkpoint(logs_dir)
    return result

async def scrape_all_teams_async(
    teams: Dict,
    periods: List[int] = [10],
    logs_dir: Optional[str] = None,
    force_update: bool = False
):
    logging.info(f"Starting scrape for {len(teams)} teams using MongoDB")
    
    progress['total'] = len(teams) * len(periods) * 2
    progress['completed'] = 0
    progress['successful'] = 0
    progress['failed'] = 0
    progress['skipped'] = 0
    
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            scrape_with_progress(
                session, team, data, game_type, period, 
                logs_dir, force_update
            )
            for team, data in teams.items()
            for period in periods
            for game_type in ['host', 'guest']
        ]
        
        results = await tqdm_asyncio.gather(
            *tasks,
            desc="Scraping Progress",
            unit="team",
            total=progress['total']
        )
        
        saved_count = 0
        for result in results:
            if result:
                success = save_to_mongodb(result)
                if success:
                    saved_count += 1
        
        total_attempts = progress['completed'] + progress['skipped']
        logging.info(
            f"Completed! Total Attempts: {total_attempts} / {progress['total']} | "
            f"Success: {progress['successful']} | "
            f"Failed: {progress['failed']} | "
            f"Skipped: {progress['skipped']} | "
            f"Saved to MongoDB: {saved_count}"
        )
        
        return {
            "success": progress['failed'] == 0,
            "total_teams": len(teams),
            "total_tasks": progress['total'],
            "tasks_attempted": total_attempts,
            "successful_scrapes": progress['successful'],
            "failed_scrapes": progress['failed'],
            "skipped_tasks": progress['skipped'],
            "saved_to_mongodb": saved_count
        }

async def run_scraper_async(
    team_count: Optional[int] = None, 
    periods: List[int] = [10],
    logs_dir: Optional[str] = None,
    force_update: bool = False
):
    """Run scraper with MongoDB backend"""
    # Configure logging
    setup_logging(logs_dir)
    
    # Set Windows async policy if needed
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Initialize MongoDB connection
    if not db_manager._initialized:
        db_manager.__init__()
        if not db_manager._initialized:
            logging.error("Failed to initialize MongoDB connection")
            return {
                "success": False,
                "error": "MongoDB initialization failed"
            }
    
    # Check if StatArea collection is initialized
    if not hasattr(db_manager, '_statarea_collection') or db_manager._statarea_collection is None:
        logging.error("StatArea collection not initialized in db_manager")
        return {
            "success": False,
            "error": "StatArea collection not available"
        }
    
    # Select teams to scrape
    selected_teams = dict(islice(TEAM_DATA.items(), team_count)) if team_count else TEAM_DATA
    logging.info(f"Selected {len(selected_teams)} teams for scraping")
    
    # Run scraper
    return await scrape_all_teams_async(
        selected_teams, 
        periods, 
        logs_dir, 
        force_update
    )

# Main Execution
if __name__ == "__main__":
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Configure default logging
    setup_logging()
    
    # Run scraper with all periods
    asyncio.run(run_scraper_async(periods=[5, 10, 15]))