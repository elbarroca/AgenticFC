import asyncio
import aiohttp
from bs4 import BeautifulSoup
import logging
from urllib.parse import quote
import sqlite3
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm_asyncio
import json
from typing import Dict, List, Optional, Callable
import hashlib
import platform
from itertools import islice
import time
import os
from contextlib import contextmanager
from db_ids.team_data import TEAM_DATA

# ======================
# Configuration
# ======================
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
MAX_CONCURRENT_REQUESTS = 8
BASE_DELAY = 1  # seconds between requests
MAX_RETRIES = 5
CACHE_EXPIRE_DAYS = 1
MATCH_HISTORY_LIMIT = 15  # Number of matches to scrape
SQLITE_TIMEOUT = 30  # seconds
SQLITE_JOURNAL_MODE = "WAL"

progress = {
    'total': 0,
    'completed': 0,
    'successful': 0,
    'failed': 0,
    'skipped': 0,
    'last_team': None
}

# ======================
# Database Connection Manager
# ======================
@contextmanager
def get_db_connection(db_path: str = 'statarea_stats.db'):
    conn = None
    try:
        conn = sqlite3.connect(
            db_path,
            timeout=SQLITE_TIMEOUT,
            check_same_thread=False
        )
        conn.execute(f"PRAGMA journal_mode={SQLITE_JOURNAL_MODE}")
        yield conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

# ======================
# Helper Functions
# ======================
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
    
    if progress['completed'] % 10 == 0 or not success:
        logging.info(
            f"Progress: {progress['completed']}/{progress['total']} | "
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

# ======================
# Database Functions
# ======================
def initialize_database(db_path: str = 'statarea_stats.db'):
    """Initialize the database with required tables"""
    max_attempts = 3
    attempt = 0
    
    logging.info(f"Initializing database at path: {db_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    while attempt < max_attempts:
        try:
            with get_db_connection(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS teams (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    country TEXT NOT NULL,
                    last_scraped TEXT,
                    content_hash TEXT,
                    status TEXT DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    UNIQUE(name, country)
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS general_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT,
                    game_type TEXT CHECK(game_type IN ('host', 'guest')),
                    period INTEGER CHECK(period IN (5, 10, 15)),
                    scrape_date TEXT,
                    stat_name TEXT,
                    stat_value TEXT,
                    FOREIGN KEY (team_id) REFERENCES teams(id),
                    UNIQUE(team_id, game_type, period, stat_name)
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS bet_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT,
                    game_type TEXT CHECK(game_type IN ('host', 'guest')),
                    period INTEGER CHECK(period IN (5, 10, 15)),
                    scrape_date TEXT,
                    category TEXT,
                    stat_name TEXT,
                    stat_value TEXT,
                    FOREIGN KEY (team_id) REFERENCES teams(id),
                    UNIQUE(team_id, game_type, period, category, stat_name)
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT,
                    game_type TEXT CHECK(game_type IN ('host', 'guest')),
                    match_date TEXT,
                    competition TEXT,
                    opponent TEXT,
                    team_goals INTEGER,
                    opponent_goals INTEGER,
                    result TEXT CHECK(result IN ('win', 'loss', 'draw')),
                    venue TEXT CHECK(venue IN ('home', 'away')),
                    scrape_date TEXT,
                    FOREIGN KEY (team_id) REFERENCES teams(id),
                    UNIQUE(team_id, match_date, opponent)
                )
                ''')
                
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_match_history_team ON match_history(team_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_match_history_date ON match_history(match_date)')
                
                conn.commit()
                logging.info(f"Database initialized successfully at {db_path}")
                return True
                
        except sqlite3.Error as e:
            attempt += 1
            logging.error(f"Database initialization attempt {attempt} failed: {str(e)}")
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            continue
    
    logging.error("Failed to initialize database after multiple attempts")
    return False

def save_match_history(team_id: str, game_type: str, matches: List[Dict], scrape_date: str, db_path: str):
    if not matches:
        return
        
    logging.info(f"Attempting to save {len(matches)} matches for team {team_id}")
    
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            
            for match in matches:
                try:
                    cursor.execute('''
                    INSERT OR IGNORE INTO match_history (
                        team_id, game_type, match_date, competition,
                        opponent, team_goals, opponent_goals,
                        result, venue, scrape_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        team_id, game_type, match['date'],
                        match['competition'], match['opponent'],
                        match['team_goals'], match['opponent_goals'],
                        match['result'], match['venue'], scrape_date
                    ))
                except sqlite3.Error as e:
                    logging.warning(f"Failed to save match {match}: {str(e)}")
                    continue
            
            conn.commit()
            logging.info(f"Successfully saved {len(matches)} matches for team {team_id}")
            
    except Exception as e:
        logging.error(f"Failed to save match history: {str(e)}")
        raise

def save_to_database(stats: Dict, db_path: str):
    if not stats:
        return
        
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            
            cursor.execute('''
            INSERT OR REPLACE INTO teams (
                id, name, country, last_scraped, content_hash, status, retry_count
            ) VALUES (?, ?, ?, ?, ?, ?, COALESCE(
                (SELECT retry_count FROM teams WHERE id = ?), 0
            ))
            ''', (
                stats['api_id'], stats['team'], stats['country'],
                stats['scrape_date'], stats.get('content_hash'),
                'success' if stats.get('general_statistics') else 'failed',
                stats['api_id']
            ))
            
            if 'general_statistics' in stats:
                for stat_name, stat_value in stats['general_statistics'].items():
                    cursor.execute('''
                    INSERT OR REPLACE INTO general_stats (
                        team_id, game_type, period, scrape_date, stat_name, stat_value
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        stats['api_id'], stats['game_type'], stats['period'],
                        stats['scrape_date'], stat_name, stat_value
                    ))
            
            if 'team_bet_statistics' in stats:
                for category, category_stats in stats['team_bet_statistics'].items():
                    for stat_name, stat_value in category_stats.items():
                        cursor.execute('''
                        INSERT OR REPLACE INTO bet_stats (
                            team_id, game_type, period, scrape_date, category, stat_name, stat_value
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            stats['api_id'], stats['game_type'], stats['period'],
                            stats['scrape_date'], category, stat_name, stat_value
                        ))
            
            conn.commit()
            
            if stats['period'] == 15 and 'match_history' in stats and stats['match_history']:
                save_match_history(
                    stats['api_id'],
                    stats['game_type'],
                    stats['match_history'],
                    stats['scrape_date'],
                    db_path
                )
            
    except Exception as e:
        logging.error(f"Database error for {stats['team']}: {str(e)}")
        raise

def needs_scraping(team: str, country: str, api_id: str, db_path: str, 
                   check_exists_func: Optional[Callable] = None) -> bool:
    """
    Check if a team needs to be scraped, with optional custom check function
    """
    if check_exists_func:
        # Use provided custom function for checking
        return check_exists_func(team, country)
        
    # Default check logic
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT last_scraped, status FROM teams 
            WHERE id = ?
            ''', (api_id,))
            
            result = cursor.fetchone()
            
            if not result:
                return True
            
            last_scraped, status = result
            
            if status == 'failed':
                return True
            
            if last_scraped:
                last_scraped_date = datetime.fromisoformat(last_scraped)
                return datetime.now() - last_scraped_date > timedelta(days=CACHE_EXPIRE_DAYS)
            return True
            
    except sqlite3.Error as e:
        logging.error(f"Database error in needs_scraping: {str(e)}")
        return True

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
                    
                    result = {
                        'team': team,
                        'country': country,
                        'api_id': api_id,
                        'game_type': game_type,
                        'period': period,
                        'scrape_date': datetime.now().isoformat(),
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
    db_path: str,
    logs_dir: Optional[str] = None,
    force_update: bool = False,
    check_exists_func: Optional[Callable] = None
) -> Optional[Dict]:
    if not force_update and not needs_scraping(team, data['country'], data['api_id'], db_path, check_exists_func):
        progress['skipped'] += 1
        progress['completed'] += 1
        logging.debug(f"Skipping {team} as it was recently scraped")
        return None
        
    result = await scrape_team_stats_async(session, team, game_type, data['country'], data['api_id'], period, logs_dir)
    update_progress(result is not None, team)
    save_checkpoint(logs_dir)
    return result

async def scrape_all_teams_async(
    teams: Dict,
    periods: List[int] = [10],
    db_path: str = 'statarea_stats.db',
    logs_dir: Optional[str] = None,
    force_update: bool = False,
    check_exists_func: Optional[Callable] = None
):
    logging.info(f"Starting scrape for {len(teams)} teams with database: {db_path}")
    
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
                db_path, logs_dir, force_update, check_exists_func
            )
            for team, data in teams.items()
            for period in periods
            for game_type in ['host', 'guest']
        ]
        
        results = await tqdm_asyncio.gather(
            *tasks,
            desc="Scraping Progress",
            unit="team"
        )
        
        for result in results:
            if result:
                save_to_database(result, db_path)
        
        logging.info(
            f"Completed! Success: {progress['successful']} | "
            f"Failed: {progress['failed']} | "
            f"Skipped: {progress['skipped']}"
        )
        
        return {
            "success": progress['failed'] == 0,
            "total_teams": len(teams),
            "completed": progress['completed'],
            "successful": progress['successful'],
            "failed": progress['failed'],
            "skipped": progress['skipped']
        }

def run_scraper(
    team_count: Optional[int] = None, 
    periods: List[int] = [10],
    db_path: str = 'statarea_stats.db',
    force_update: bool = False
):
    """Run scraper synchronously"""
    selected_teams = dict(islice(TEAM_DATA.items(), team_count)) if team_count else TEAM_DATA
    return asyncio.run(scrape_all_teams_async(
        selected_teams, periods, db_path, force_update=force_update
    ))

async def run_scraper_async(
    team_count: Optional[int] = None, 
    periods: List[int] = [10],
    db_path: str = 'statarea_stats.db',
    logs_dir: Optional[str] = None,
    force_update: bool = False,
    check_exists_func: Optional[Callable] = None
):
    """
    Run scraper asynchronously with optional path for database and logs
    
    Args:
        team_count: Optional limit on number of teams to scrape
        periods: List of periods to scrape (5, 10, or 15)
        db_path: Path to SQLite database file
        logs_dir: Directory for log files
        force_update: Whether to update all teams regardless of last scrape time
        check_exists_func: Optional function to check if team needs scraping
        
    Returns:
        Dict with scraping results
    """
    # Configure logging
    setup_logging(logs_dir)
    
    # Set Windows async policy if needed
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Initialize database
    if not initialize_database(db_path):
        logging.error(f"Failed to initialize database at {db_path}")
        return {
            "success": False,
            "error": "Database initialization failed"
        }
    
    # Select teams to scrape
    selected_teams = dict(islice(TEAM_DATA.items(), team_count)) if team_count else TEAM_DATA
    logging.info(f"Selected {len(selected_teams)} teams for scraping")
    
    # Run scraper
    return await scrape_all_teams_async(
        selected_teams, 
        periods, 
        db_path, 
        logs_dir, 
        force_update,
        check_exists_func
    )

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Configure default logging
    setup_logging()
    
    # Initialize database first
    if not initialize_database():
        exit(1)
    
    # Once working, uncomment to run for all teams:
    run_scraper(periods=[5, 10, 15])