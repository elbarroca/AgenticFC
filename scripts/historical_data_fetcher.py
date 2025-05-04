import os
import asyncio
import aiohttp
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Set
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required modules
from get_data.api_football.db_mongo import db_manager
from get_data.api_football.db_ids.team_id_mappings import TEAM_ID_MAPPING

# API Configuration
API_KEY = "dca41d4edemshe469d9d1754cd7ap1c7e06jsn7c5425d89bef"
API_HOST = "api-football-v1.p.rapidapi.com"
BASE_URL = f"https://{API_HOST}/v3"

# Request limits
MAX_CONCURRENT_REQUESTS = 35
REQUEST_DELAY_SECONDS = 1.0

HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": API_HOST
}

async def _make_api_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    params: Dict,
    retry_count: int = 3
) -> Dict:
    """Makes an API request with rate limiting and retries."""
    url = f"{BASE_URL}/{endpoint}"
    
    async with semaphore:
        await asyncio.sleep(max(0.1, REQUEST_DELAY_SECONDS / MAX_CONCURRENT_REQUESTS))
        
        for attempt in range(retry_count):
            try:
                async with session.get(url, headers=HEADERS, params=params, timeout=30) as response:
                    if response.status == 429:
                        wait_time = (REQUEST_DELAY_SECONDS + 1) * (2 ** (attempt + 1))
                        logger.warning(f"Rate limit hit (429) on attempt {attempt+1}. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    if "errors" in data and data["errors"]:
                        logger.error(f"API errors for {url}: {data['errors']}")
                        return None
                    
                    return data
            except Exception as e:
                logger.error(f"Error on attempt {attempt+1} for {url}: {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(3 * (attempt + 1))
                else:
                    logger.error(f"All {retry_count} attempts failed for {url}")
                    return None
    
    return None

async def fetch_team_fixtures(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    team_id: int,
    season: int
) -> List[Dict]:
    """Fetches all fixtures for a team in a specific season."""
    logger.info(f"Fetching fixtures for Team ID: {team_id}, Season: {season}")
    params = {"team": str(team_id), "season": str(season)}
    
    response = await _make_api_request(session, semaphore, "fixtures", params)
    
    if not response or "response" not in response:
        logger.error(f"Failed to fetch fixtures for Team ID {team_id}, Season {season}")
        return []
    
    fixtures = response["response"]
    logger.info(f"Found {len(fixtures)} fixtures for Team ID {team_id}, Season {season}")
    return fixtures

async def fetch_fixture_details(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    fixture_id: int
) -> Dict:
    """Fetches complete details for a specific fixture."""
    logger.info(f"Fetching details for Fixture ID: {fixture_id}")
    params = {"id": str(fixture_id)}
    
    response = await _make_api_request(session, semaphore, "fixtures", params)
    
    if not response or "response" not in response or not response["response"]:
        logger.error(f"Failed to fetch details for Fixture ID {fixture_id}")
        return None
    
    # The API returns an array with a single fixture
    fixture_details = response["response"][0]
    return fixture_details

async def main_pipeline(seasons: List[int]):
    """Main processing pipeline to fetch team fixtures and details."""
    logger.info("=== STARTING DATA FETCHING PIPELINE ===")
    
    try:
        # Initialize DB Manager
        if not db_manager._initialized:
            db_manager.__init__()
        
        if not db_manager._initialized:
            logger.error("Failed to initialize DB Manager. Exiting.")
            return
        
        # Get team IDs from mapping
        team_ids = [int(team_info["mongodb_id"]) for team_info in TEAM_ID_MAPPING.values() 
                   if team_info.get("mongodb_id")]
        
        if not team_ids:
            logger.error("No valid team IDs found in TEAM_ID_MAPPING. Exiting.")
            return
        
        logger.info(f"Processing {len(team_ids)} teams for {len(seasons)} seasons")
        
        # Setup async resources
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS * 2)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # STEP 1: Fetch all fixtures for all teams and seasons
            logger.info("STEP 1: Fetching team fixtures")
            all_fixtures = []
            
            for season in seasons:
                for team_id in team_ids:
                    team_fixtures = await fetch_team_fixtures(session, semaphore, team_id, season)
                    all_fixtures.extend(team_fixtures)
            
            # Remove duplicates by fixture ID
            unique_fixtures = {}
            for fixture in all_fixtures:
                fixture_id = fixture["fixture"]["id"]
                unique_fixtures[fixture_id] = fixture
            
            logger.info(f"Found {len(unique_fixtures)} unique fixtures across all teams and seasons")
            
            # STEP 2: Save fixtures to database and identify those needing details
            logger.info("STEP 2: Saving team fixtures and identifying missing details")
            fixture_ids_needing_details = set()
            
            for fixture_id, fixture_data in unique_fixtures.items():
                # Save basic fixture data using the correct method
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    db_manager.save_match_data,
                    fixture_data
                )
                
                # Check if we need full details
                has_details = await asyncio.get_event_loop().run_in_executor(
                    None,
                    db_manager.check_match_exists,
                    str(fixture_id)
                )
                
                if not has_details:
                    fixture_ids_needing_details.add(fixture_id)
            
            logger.info(f"Need to fetch full details for {len(fixture_ids_needing_details)} fixtures")
            
            # STEP 3: Fetch and save full details for fixtures that need them
            logger.info("STEP 3: Fetching and saving full fixture details")
            
            detail_tasks = []
            for fixture_id in fixture_ids_needing_details:
                detail_tasks.append(fetch_fixture_details(session, semaphore, fixture_id))
            
            total_details = len(detail_tasks)
            processed = 0
            
            for i, detail_task in enumerate(asyncio.as_completed(detail_tasks)):
                fixture_details = await detail_task
                processed += 1
                
                if fixture_details:
                    fixture_id = fixture_details["fixture"]["id"]
                    
                    # Process and save the full fixture details
                    # Create a standardized document structure
                    match_data = {
                        "_id": str(fixture_id),
                        "fixture_id": str(fixture_id),
                        "date_utc": datetime.fromisoformat(fixture_details["fixture"]["date"].replace("Z", "+00:00")),
                        "date_str": datetime.fromisoformat(fixture_details["fixture"]["date"].replace("Z", "+00:00")).strftime("%Y-%m-%d"),
                        "season": fixture_details["league"]["season"],
                        "league_id": fixture_details["league"]["id"],
                        "league_name": fixture_details["league"]["name"],
                        "league_country": fixture_details["league"]["country"],
                        "status_short": fixture_details["fixture"]["status"]["short"],
                        "status_long": fixture_details["fixture"]["status"]["long"],
                        "home_team_id": fixture_details["teams"]["home"]["id"],
                        "home_team_name": fixture_details["teams"]["home"]["name"],
                        "away_team_id": fixture_details["teams"]["away"]["id"],
                        "away_team_name": fixture_details["teams"]["away"]["name"],
                        "home_goals": fixture_details["goals"]["home"],
                        "away_goals": fixture_details["goals"]["away"],
                        "score_halftime": fixture_details["score"]["halftime"],
                        "score_fulltime": fixture_details["score"]["fulltime"],
                        "fixture_details": fixture_details,
                        "fetch_timestamp_utc": datetime.now(timezone.utc)
                    }
                    
                    success = await asyncio.get_event_loop().run_in_executor(
                        None,
                        db_manager.save_match_data,
                        match_data
                    )
                    
                    if success:
                        logger.info(f"Saved full details for fixture {fixture_id} ({processed}/{total_details})")
                    else:
                        logger.error(f"Failed to save details for fixture {fixture_id}")
                
                if processed % 50 == 0 or processed == total_details:
                    logger.info(f"Progress: {processed}/{total_details} fixtures processed")
    
    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
    finally:
        # Close DB connection
        if db_manager._initialized and hasattr(db_manager, '_client'):
            db_manager.close_connection()
            logger.info("DB connection closed")
        
        logger.info("=== DATA FETCHING PIPELINE COMPLETED ===")

if __name__ == "__main__":
    # Define target seasons
    TARGET_SEASONS = [2024, 2023, 2022, 2021, 2020, 2019, 2018 , 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009]
    
    logger.info(f"Starting data fetching for seasons: {TARGET_SEASONS}")
    
    try:
        asyncio.run(main_pipeline(TARGET_SEASONS))
    except Exception as e:
        logger.error(f"Failed to run pipeline: {e}", exc_info=True)
        # Ensure DB connection is closed
        if 'db_manager' in locals() and hasattr(db_manager, '_initialized') and db_manager._initialized:
            db_manager.close_connection() 