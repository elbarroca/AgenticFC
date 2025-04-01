import asyncio
import aiohttp
from bs4 import BeautifulSoup
import logging
from urllib.parse import quote
import sqlite3
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm_asyncio
import json
from typing import Dict, List, Optional
from itertools import islice
import hashlib
import platform
from get_data.db_ids.team_data import TEAM_DATA

# Fix for Windows event loop
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
MAX_CONCURRENT_REQUESTS = 8 # can go up to 10 but 8 doing it under 15 min avg 10~min
BASE_DELAY = 1  # seconds between requests
MAX_RETRIES = 5
CACHE_EXPIRE_DAYS = 1  # Only re-check teams scraped more than this many days ago

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler('scraper_progress.log'),
        logging.StreamHandler()
    ]
)

progress = {
    'total': 0,
    'completed': 0,
    'successful': 0,
    'failed': 0,
    'skipped': 0,
    'last_team': None
}

def update_progress(success: bool, team: str):
    """Update global progress counters"""
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

def save_checkpoint():
    """Save progress to resume later"""
    with open('progress_checkpoint.json', 'w') as f:
        json.dump(progress, f)

def log_failed_url(url: str):
    """Record failed URLs for later retry"""
    with open('failed_urls.log', 'a') as f:
        f.write(f"{datetime.now().isoformat()},{url}\n")

def construct_url(team: str, game_type: str, country: str, period: int = 10) -> str:
    """Build the StatArea URL for a team's stats"""
    if period not in [5, 10, 15]:
        raise ValueError("Period must be 5, 10, or 15")
    
    base_url = "https://www.statarea.com/team/view"
    team_with_country = f"{team} ({country})"
    encoded_team = quote(team_with_country)
    return f"{base_url}/{encoded_team}/{game_type}/last{period}"

def extract_general_statistics(soup: BeautifulSoup) -> Dict:
    """Parse general stats from the page"""
    stats = {}
    container = soup.find('div', class_='teamstatistics')
    
    if container:
        for item in container.find_all('div', class_='factitem'):
            label = item.find('div', class_='label').text.strip()
            value = item.find('div', class_='value').text.strip()
            stats[label] = value
    
    return stats

def extract_team_bet_statistics(soup: BeautifulSoup) -> Dict:
    """Parse betting stats from the page"""
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

def initialize_database(db_name: str = 'statarea_stats.db'):
    """Set up SQLite database with optimized schema"""
    conn = sqlite3.connect(db_name)
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
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_teams_last_scraped ON teams(last_scraped)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_teams_status ON teams(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_teams_retry ON teams(retry_count)')
    conn.commit()
    conn.close()

def cleanup_failed_entries():
    """Remove stats from previously failed attempts"""
    conn = sqlite3.connect('statarea_stats.db')
    cursor = conn.cursor()
    cursor.execute('''
    DELETE FROM general_stats WHERE team_id IN 
        (SELECT id FROM teams WHERE status = 'failed')
    ''')
    cursor.execute('''
    DELETE FROM bet_stats WHERE team_id IN 
        (SELECT id FROM teams WHERE status = 'failed')
    ''')
    conn.commit()
    conn.close()

def purge_chronic_failures(max_retries=5):
    """Remove teams that consistently fail"""
    conn = sqlite3.connect('statarea_stats.db')
    cursor = conn.cursor()
    cursor.execute('''
    DELETE FROM teams 
    WHERE status = 'failed' AND retry_count >= ?
    ''', (max_retries,))
    conn.commit()
    conn.close()

def needs_scraping(team: str, country: str, api_id: str, db_name: str = 'statarea_stats.db') -> bool:
    """Check if team needs to be scraped based on last_scraped time and status"""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT last_scraped, status FROM teams 
    WHERE id = ?
    ''', (api_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return True  # Never scraped before
    
    last_scraped, status = result
    
    # Always retry failed clubs
    if status == 'failed':
        return True
    
    # For successful scrapes, respect cache period
    last_scraped_date = datetime.fromisoformat(last_scraped) if last_scraped else datetime.min
    return datetime.now() - last_scraped_date > timedelta(days=CACHE_EXPIRE_DAYS)

async def scrape_team_stats_async(
    session: aiohttp.ClientSession,
    team: str,
    game_type: str,
    country: str,
    api_id: str,
    period: int = 10,
    max_retries: int = MAX_RETRIES
) -> Optional[Dict]:
    """Optimized scraping with caching headers and change detection"""
    url = construct_url(team, game_type, country, period)
    
    # Reset status to 'pending' before retry
    conn = sqlite3.connect('statarea_stats.db')
    cursor = conn.cursor()
    cursor.execute('''
    UPDATE teams SET status = 'pending'
    WHERE id = ?
    ''', (api_id,))
    conn.commit()
    conn.close()
    
    current_hash = None
    conn = sqlite3.connect('statarea_stats.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT content_hash FROM teams 
    WHERE id = ?
    ''', (api_id,))
    result = cursor.fetchone()
    previous_hash = result[0] if result else None
    conn.close()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    if previous_hash:
        headers['If-None-Match'] = previous_hash
    
    for attempt in range(max_retries):
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 304:
                    return None
                
                if response.status == 200:
                    html = await response.text()
                    current_hash = hashlib.md5(html.encode()).hexdigest()
                    
                    if current_hash == previous_hash:
                        return None
                    
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    return {
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
                else:
                    wait_time = min(2 ** (attempt + 1), 10)
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {team}: "
                        f"HTTP {response.status}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for {team}: "
                f"{type(e).__name__}"
            )
            if attempt == max_retries - 1:
                log_failed_url(url)
                return None
            await asyncio.sleep(BASE_DELAY)
    
    return None

def save_to_database(stats: Dict, db_name: str = 'statarea_stats.db'):
    """Save results to SQLite database"""
    if not stats:
        # Explicit failure case
        status = 'failed'
        stats = {'team': 'unknown', 'country': 'unknown', 'api_id': 'unknown', 'scrape_date': datetime.now().isoformat()}
    else:
        status = 'success' if stats.get('general_statistics') else 'failed'
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    try:
        # Get or create team record
        cursor.execute('''
        INSERT OR IGNORE INTO teams 
        (id, name, country, last_scraped, content_hash, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            stats['api_id'],
            stats['team'], 
            stats['country'],
            stats['scrape_date'],
            stats.get('content_hash'),
            status
        ))
        
        # Update team record
        update_params = {
            'last_scraped': stats['scrape_date'],
            'content_hash': stats.get('content_hash'),
            'status': status,
            'api_id': stats['api_id']
        }
        
        if status == 'failed':
            cursor.execute('''
            UPDATE teams SET 
                last_scraped = :last_scraped,
                content_hash = :content_hash,
                status = :status,
                retry_count = retry_count + 1
            WHERE id = :api_id
            ''', update_params)
        else:
            cursor.execute('''
            UPDATE teams SET 
                last_scraped = :last_scraped,
                content_hash = :content_hash,
                status = :status
            WHERE id = :api_id
            ''', update_params)
        
        # Only save stats if successful
        if status == 'success':
            # Clear old stats first
            cursor.execute('''
            DELETE FROM general_stats 
            WHERE team_id = ? AND game_type = ? AND period = ?
            ''', (stats['api_id'], stats['game_type'], stats['period']))
            
            cursor.execute('''
            DELETE FROM bet_stats 
            WHERE team_id = ? AND game_type = ? AND period = ?
            ''', (stats['api_id'], stats['game_type'], stats['period']))
            
            # Save general stats
            for stat_name, stat_value in stats['general_statistics'].items():
                cursor.execute('''
                INSERT INTO general_stats 
                (team_id, game_type, period, scrape_date, stat_name, stat_value)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    stats['api_id'], stats['game_type'], stats['period'],
                    stats['scrape_date'], stat_name, stat_value
                ))
            
            # Save bet stats
            for category, category_stats in stats['team_bet_statistics'].items():
                for stat_name, stat_value in category_stats.items():
                    cursor.execute('''
                    INSERT INTO bet_stats 
                    (team_id, game_type, period, scrape_date, category, stat_name, stat_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        stats['api_id'], stats['game_type'], stats['period'],
                        stats['scrape_date'], category, stat_name, stat_value
                    ))
        
        conn.commit()
    
    except Exception as e:
        logging.error(f"Database error for {stats['team']}: {e}")
        conn.rollback()
    finally:
        conn.close()

async def scrape_with_progress(
    session: aiohttp.ClientSession,
    team: str,
    country: str,
    api_id: str,
    game_type: str,
    period: int
) -> Optional[Dict]:
    """Wrapper to handle progress tracking"""
    if not needs_scraping(team, country, api_id):
        progress['skipped'] += 1
        progress['completed'] += 1
        return None
        
    result = await scrape_team_stats_async(session, team, game_type, country, api_id, period)
    update_progress(result is not None, team)
    save_checkpoint()
    return result

async def scrape_all_teams_async(
    teams: Dict[str, Dict],
    periods: List[int] = [10],
    max_concurrent: int = MAX_CONCURRENT_REQUESTS
):
    """Main scraping function with progress tracking"""
    progress['total'] = len(teams) * len(periods) * 2
    progress['completed'] = 0
    progress['successful'] = 0
    progress['failed'] = 0
    progress['skipped'] = 0
    
    initialize_database()
    cleanup_failed_entries()
    purge_chronic_failures()
    
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    
    async with aiohttp.ClientSession(
        connector=connector,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en-US,en;q=0.9"
        }
    ) as session:
        tasks = [
            scrape_with_progress(session, team, data['country'], data['api_id'], game_type, period)
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
                save_to_database(result)
        
        logging.info(
            f"Completed! Success: {progress['successful']} | "
            f"Failed: {progress['failed']} | "
            f"Skipped: {progress['skipped']}"
        )

def run_scraper(team_count: Optional[int] = None, periods: List[int] = [10]):
    """Run the scraper with progress tracking"""
    selected_teams = dict(islice(TEAM_DATA.items(), team_count)) if team_count else TEAM_DATA
    asyncio.run(scrape_all_teams_async(selected_teams, periods))

if __name__ == "__main__":
    # Run the scraper first to ensure we have fresh data
    run_scraper(periods=[5, 10, 15])
    