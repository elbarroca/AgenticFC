import os
import asyncio
import aiohttp
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set
from pymongo.errors import ConnectionFailure
import traceback
import logging
# Add project root to sys.path
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)

# Insert project root first, then specific db_ids path if needed
sys.path.insert(0, project_root)
# sys.path.insert(1, str(db_ids_path.parent)) # Add parent of db_ids just in case
# sys.path.insert(1, str(project_root / "get_data/api_football")) # Another potential path needed depending on execution context
logger = logging.getLogger(__name__)

# Now we can import our local modules
from get_data.api_football.db_mongo import db_manager
try:
    from get_data.api_football.db_ids.team_id_mappings import TEAM_ID_MAPPING
    logger.info("Successfully imported TEAM_ID_MAPPING.")
except ModuleNotFoundError:
    logger.error(f"Could not find 'team_id_mappings' module. Checked sys.path: {sys.path}")
    # Fallback to an empty dict to prevent crashing, but log error prominently
    TEAM_ID_MAPPING = {}
    logger.error("TEAM_ID_MAPPING set to empty dict due to import error. No teams will be processed.")
except ImportError as ie:
     logger.error(f"ImportError encountered: {ie}. Checked sys.path: {sys.path}")
     TEAM_ID_MAPPING = {}
     logger.error("TEAM_ID_MAPPING set to empty dict due to import error. No teams will be processed.")


# --- Configuration ---
API_KEY = "dca41d4edemshe469d9d1754cd7ap1c7e06jsn7c5425d89bef"
API_HOST = "api-football-v1.p.rapidapi.com"
BASE_URL = f"https://{API_HOST}/v3"

if not API_KEY or API_KEY == "YOUR_PLACEHOLDER_API_KEY":
    logging.warning("API Key is not configured correctly. Set PAID_API_FOOTBALL_KEY environment variable or edit the script.")

HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": API_HOST
}

# Concurrency and Rate Limiting
MAX_CONCURRENT_REQUESTS = 15 # Adjusted concurrency slightly higher
REQUEST_DELAY_SECONDS = 2.0 # Adjusted base delay slightly lower

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Reduce aiohttp noise
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Async Helper Function for API Calls (Unchanged from previous version) ---
async def _make_async_api_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    params: Dict,
    retry_count: int = 3
) -> Optional[Dict]:
    """Makes an async API request using the hardcoded key, semaphore, and delay."""
    if not API_KEY or API_KEY == "YOUR_PLACEHOLDER_API_KEY":
        logger.error("Cannot make API request: API Key is not configured.")
        return None

    url = f"{BASE_URL}/{endpoint}"
    logger.debug(f"Attempting to acquire semaphore for {url} with params {params}")

    async with semaphore: # Control concurrency
        logger.debug(f"Semaphore acquired for {url}")
        # Apply delay *before* the request
        await asyncio.sleep(max(0.1, REQUEST_DELAY_SECONDS / MAX_CONCURRENT_REQUESTS)) # Ensure minimum 0.1s delay

        for attempt in range(retry_count):
            logger.debug(f"Making API request (attempt {attempt + 1}/{retry_count}) to {url}")
            try:
                async with session.get(url, headers=HEADERS, params=params, timeout=45) as response:
                    if response.status == 429:
                        wait_time = (REQUEST_DELAY_SECONDS + 1) * (2 ** (attempt + 1))
                        logger.warning(f"Rate limit hit (status 429) on attempt {attempt + 1} for {url}. Waiting {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    data = await response.json()

                    api_errors = data.get("errors")
                    if api_errors and isinstance(api_errors, (list, dict)) and len(api_errors) > 0:
                        if isinstance(api_errors, dict) and not api_errors: # Handle case where errors is an empty dict {}
                             pass # Ignore empty error dicts
                        else:
                             logger.error(f"API returned errors for {url} with params {params}: {api_errors}")
                             return None

                    if "results" in data and data.get("results") == 0:
                        logger.warning(f"API request successful, but no data found ('results: 0') for {url} with params {params}")
                        return data

                    if "response" in data and not data["response"]:
                        logger.warning(f"API request successful, but 'response' array is empty for {url} with params {params}")
                        return data

                    logger.debug(f"Successfully fetched data for {url} (attempt {attempt + 1})")
                    return data

            except asyncio.TimeoutError:
                 logger.warning(f"Request timed out for {url} (attempt {attempt + 1}/{retry_count}). Retrying...")
                 if attempt < retry_count - 1:
                     await asyncio.sleep(5 * (attempt + 1))
                 else:
                     logger.error(f"Request timed out after {retry_count} attempts for {url}.")
                     return None
            except aiohttp.ClientError as e:
                logger.error(f"Network/Client error for {url} (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    wait_time_network = 3 ** (attempt + 1)
                    logger.info(f"Retrying in {wait_time_network} seconds...")
                    await asyncio.sleep(wait_time_network)
                else:
                    logger.error(f"Network/Client error failed after {retry_count} attempts for {url}.")
                    return None
            except Exception as e:
                logger.error(f"An unexpected error occurred during API request to {url}: {e}", exc_info=True)
                return None

    logger.error(f"All retry attempts failed or semaphore timed out for API request to {url} with params {params}.")
    return None


# --- Async Core Data Fetching Logic ---

async def fetch_teams_for_league_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    league_id: int,
    season: int
) -> List[int]:
    """Fetches team IDs for a specific league and season."""
    logger.info(f"Fetching teams for League ID: {league_id}, Season: {season}")
    params = {"league": str(league_id), "season": str(season)}
    response_data = await _make_async_api_request(session, semaphore, "teams", params)

    if response_data and response_data.get("response"):
        team_ids = [item.get("team", {}).get("id") for item in response_data["response"]]
        team_ids = [tid for tid in team_ids if tid] # Filter out None or missing IDs
        logger.info(f"Found {len(team_ids)} teams for League {league_id}, Season {season}")
        return team_ids
    else:
        logger.error(f"Could not fetch or parse teams for League {league_id}, Season {season}.")
        return []

async def fetch_team_fixtures_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    team_id: int,
    season: int
) -> Optional[List[int]]:
    """
    Fetches all fixture IDs for a specific team and season.
    Returns the list of IDs or None if fetch fails.
    """
    logger.info(f"Fetching fixtures for Team ID: {team_id}, Season: {season}")
    params = {"team": str(team_id), "season": str(season)}
    all_fixture_ids = []

    response_data = await _make_async_api_request(session, semaphore, "fixtures", params)

    if response_data and response_data.get("response"):
        fixture_ids_list = [item.get("fixture", {}).get("id") for item in response_data["response"]]
        fixture_ids_list = [fid for fid in fixture_ids_list if fid] # Filter nulls

        if not fixture_ids_list:
            logger.debug(f"No fixtures found in the response for Team {team_id}, Season {season}.")
            return [] # Return empty list if none found
        else:
            logger.debug(f"Found {len(fixture_ids_list)} fixtures in the response for Team {team_id}, Season {season}")
            return fixture_ids_list
    else:
        logger.error(f"Could not fetch or parse fixtures response for Team {team_id}, Season {season}.")
        return None # Return None on fetch error


async def process_fixture_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    fixture_id: int,
    loop: asyncio.AbstractEventLoop
) -> str:
    """
    Fetches details (basic, stats, HALF-TIME stats, events, lineups, predictions)
    for a fixture ID asynchronously, checks DB, and saves if new using a flatter structure.
    Returns a status string like "SAVED_DETAILS", "SKIPPED", "ERROR".
    """
    fixture_id_str = str(fixture_id)
    logger.debug(f"Processing fixture ID: {fixture_id_str}")

    # 1. Check DB if match details already exist
    needs_details_fetch = False
    try:
        # Calls db_manager.check_match_exists which queries the 'matches' collection
        match_exists = await loop.run_in_executor(None, db_manager.check_match_exists, fixture_id_str)
        if not match_exists:
            needs_details_fetch = True
            logger.debug(f"Fixture {fixture_id_str} details need fetching.")
        else:
            logger.debug(f"Fixture {fixture_id_str} details already exist in DB.")
            # If match_exists is True, it returns "SKIPPED" later, preventing API calls

    except Exception as db_check_err:
         logger.error(f"Error checking DB for fixture {fixture_id_str}: {db_check_err}. Cannot proceed reliably.", exc_info=True)
         return "ERROR_DB_CHECK"

    if not needs_details_fetch:
        logger.info(f"Fixture {fixture_id_str} details fully populated in DB. Skipping fetch.")
        return "SKIPPED"

    logger.info(f"Fetching new details data for fixture {fixture_id_str}")

    # 2. Fetch required data concurrently
    fetch_tasks = {}
    params_id = {"id": fixture_id_str}
    params_fixture = {"fixture": fixture_id_str}
    params_fixture_half_stats = {"fixture": fixture_id_str, "half": "true"} # Params for half-time stats

    # Add all fetch tasks
    fetch_tasks["basic"] = _make_async_api_request(session, semaphore, "fixtures", params_id)
    fetch_tasks["stats_full"] = _make_async_api_request(session, semaphore, "fixtures/statistics", params_fixture) # Renamed key
    fetch_tasks["stats_half"] = _make_async_api_request(session, semaphore, "fixtures/statistics", params_fixture_half_stats) # Added half-time stats call
    fetch_tasks["events"] = _make_async_api_request(session, semaphore, "fixtures/events", params_fixture)
    fetch_tasks["lineups"] = _make_async_api_request(session, semaphore, "fixtures/lineups", params_fixture)
    fetch_tasks["preds"] = _make_async_api_request(session, semaphore, "predictions", params_fixture)

    # Execute tasks
    fetched_data = {}
    results = await asyncio.gather(*fetch_tasks.values(), return_exceptions=True)
    for key, result in zip(fetch_tasks.keys(), results):
        if isinstance(result, Exception):
            logger.error(f"Error fetching '{key}' data for fixture {fixture_id_str}: {result}")
            fetched_data[key] = None
        else:
            fetched_data[key] = result

    # 3. Process and Save Details
    saved_details = False
    basic_info_resp = fetched_data.get("basic")
    if not basic_info_resp or not basic_info_resp.get("response"):
        logger.error(f"Failed to fetch essential basic info for fixture {fixture_id_str}. Cannot save details.")
    elif len(basic_info_resp["response"]) != 1:
        logger.error(f"Expected 1 fixture in basic info response for {fixture_id_str}, got {len(basic_info_resp['response'])}.")
    else:
        basic_data = basic_info_resp["response"][0]
        fixture_info = basic_data.get("fixture", {})
        league_info = basic_data.get("league", {})
        teams_info = basic_data.get("teams", {})
        goals_info = basic_data.get("goals", {})
        score_info = basic_data.get("score", {})

        fixture_date_utc = None
        fixture_date_str_for_db = None
        fixture_date_str_iso = fixture_info.get("date")
        if fixture_date_str_iso:
            try:
                fixture_date_utc = datetime.fromisoformat(fixture_date_str_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
                fixture_date_str_for_db = fixture_date_utc.strftime("%Y-%m-%d")
            except ValueError:
                logger.error(f"Could not parse fixture date string '{fixture_date_str_iso}' for fixture {fixture_id_str}.")
        else:
            logger.error(f"Fixture {fixture_id_str} is missing the 'date' field. Cannot save essential field.")

        if fixture_date_utc and fixture_date_str_for_db:
            season = league_info.get("season")
            league_id = league_info.get("id")
            status_short = fixture_info.get("status", {}).get("short")
            home_team_id = teams_info.get("home", {}).get("id")
            away_team_id = teams_info.get("away", {}).get("id")

            if league_id is None or home_team_id is None or away_team_id is None or season is None or status_short is None:
                logger.error(f"Missing essential ID field (league, team, season, status) in basic info for {fixture_id_str}. Cannot save reliably.")
            else:
                # Get results for each fetched endpoint
                stats_full_resp = fetched_data.get("stats_full")
                stats_half_resp = fetched_data.get("stats_half") # Get half-time stats result
                events_resp = fetched_data.get("events")
                lineups_resp = fetched_data.get("lineups")
                predictions_resp = fetched_data.get("preds")

                processed_match_data = {
                    "_id": fixture_id_str,
                    "fixture_id": fixture_id_str,
                    "date_utc": fixture_date_utc,
                    "date_str": fixture_date_str_for_db,
                    "season": int(season),
                    "league_id": int(league_id),
                    "league_name": league_info.get("name"),
                    "league_country": league_info.get("country"),
                    "status_short": status_short,
                    "status_long": fixture_info.get("status", {}).get("long"),
                    "home_team_id": int(home_team_id),
                    "home_team_name": teams_info.get("home", {}).get("name"),
                    "away_team_id": int(away_team_id),
                    "away_team_name": teams_info.get("away", {}).get("name"),
                    "home_goals": goals_info.get("home"),
                    "away_goals": goals_info.get("away"),
                    "score_halftime": score_info.get("halftime"),
                    "score_fulltime": score_info.get("fulltime"),
                    "fixture_details": basic_data,
                    "statistics_full": stats_full_resp.get("response", []) if stats_full_resp else [], # Renamed field
                    "statistics_half": stats_half_resp.get("response", []) if stats_half_resp else [], # Added field for half-time stats
                    "events": events_resp.get("response", []) if events_resp else [],
                    "lineups": lineups_resp.get("response", []) if lineups_resp else [],
                    "predictions": (predictions_resp.get("response", [])[0]
                                    if predictions_resp and predictions_resp.get("response")
                                    else {}),
                    "fetch_timestamp_utc": datetime.now(timezone.utc)
                }

                try:
                    success = await loop.run_in_executor(None, db_manager.save_match_data, processed_match_data)
                    if success:
                        logger.info(f"Successfully saved details (incl. half stats) for fixture {fixture_id_str}")
                        saved_details = True
                    else:
                        logger.error(f"db_manager.save_match_data indicated failure for details fixture {fixture_id_str}.")
                except Exception as db_save_err:
                    logger.error(f"Exception during MongoDB details save for fixture {fixture_id_str}: {db_save_err}", exc_info=True)
        else:
             logger.error(f"Cannot save details for fixture {fixture_id_str} due to missing or invalid date.")

    # 5. Determine final status
    if saved_details:
        return "SAVED_DETAILS"
    elif not needs_details_fetch:
         return "SKIPPED"
    else:
         return "ERROR_PROCESS"


# --- Main Async Orchestration Logic ---
async def run_full_historical_fetch(target_seasons: List[int]):
    """
    Fetches historical fixture data based on DB lists.
    1. Reads all relevant fixture IDs from the 'team_season_fixtures' collection in DB.
    2. Checks which fixtures already exist in the 'matches' collection.
    3. Fetches and saves details ONLY for the missing fixtures from API.
    """
    # --- Initial Setup (API Key, Team Mapping, DB Check) ---
    if not API_KEY or API_KEY == "YOUR_PLACEHOLDER_API_KEY":
        logger.error("Cannot run: API Key is not configured.")
        return
    if not TEAM_ID_MAPPING:
        logger.error("Cannot run: TEAM_ID_MAPPING is empty or failed to import.")
        return
    if not db_manager._initialized:
        logger.error("DB Manager not initialized. Cannot run fetch.")
        return

    logger.info(f"Starting historical data fetch based on DB lists for {len(TEAM_ID_MAPPING)} teams across Seasons: {target_seasons}")
    start_time = time.time()
    loop = asyncio.get_running_loop()

    # --- Get Target Team IDs ---
    target_team_ids_int: List[int] = []
    for team_name, team_info in TEAM_ID_MAPPING.items():
         team_id_str = team_info.get("mongodb_id")
         if not team_id_str:
             logger.warning(f"Skipping team '{team_name}' due to missing 'mongodb_id'.")
             continue
         try:
             target_team_ids_int.append(int(team_id_str))
         except ValueError:
             logger.warning(f"Skipping team '{team_name}' due to non-integer mongodb_id: {team_id_str}.")
             continue
    logger.info(f"Identified {len(target_team_ids_int)} target team IDs from mapping.")

    # --- Shared Async Resources ---
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS * 2, limit_per_host=MAX_CONCURRENT_REQUESTS + 2)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=60)) as session:

        # --- Step 1: Read Fixture IDs from DB and Identify Missing Details ---
        logger.info("--- Step 1: Reading Fixture IDs from DB and Identifying Missing Details ---")

        # Get all fixture IDs stored in team_season_fixtures for the target teams/seasons
        all_db_fixture_ids_int: Set[int] = await loop.run_in_executor(
            None,
            db_manager.get_fixture_ids_for_teams_seasons,
            target_team_ids_int, target_seasons # Use the list derived from TEAM_ID_MAPPING
        )
        total_unique_fixtures_in_db = len(all_db_fixture_ids_int)
        logger.info(f"Found {total_unique_fixtures_in_db} unique fixture IDs across relevant team/season lists in the DB.")

        if not all_db_fixture_ids_int:
             logger.warning("No fixture IDs found in the database for the target teams/seasons. Exiting detail processing.")
             return

        # Get IDs of matches already present in the 'matches' collection
        existing_match_ids_str: Set[str] = await loop.run_in_executor(
            None,
            db_manager.get_existing_match_ids,
            all_db_fixture_ids_int # Pass the set to optimize the query
        )
        logger.info(f"Found {len(existing_match_ids_str)} fixture detail documents already in the 'matches' collection.")

        # Determine which fixtures need their details fetched
        existing_match_ids_int: Set[int] = {int(id_str) for id_str in existing_match_ids_str if id_str.isdigit()}
        fixtures_to_process_int: Set[int] = all_db_fixture_ids_int - existing_match_ids_int
        total_fixtures_to_process = len(fixtures_to_process_int)

        logger.info(f"Identified {total_fixtures_to_process} fixtures needing details to be fetched.")

        if not fixtures_to_process_int:
             logger.warning("All fixtures found in the DB lists already have details in the 'matches' collection. No fetching required.")
             # Log summary and exit cleanly
             end_time = time.time()
             logger.info("--- Historical Data Fetch Summary (No New Details Needed) ---")
             logger.info(f"Target Seasons: {target_seasons}")
             logger.info(f"Teams Considered: {len(TEAM_ID_MAPPING)}")
             logger.info(f"Total Unique Fixtures in DB Lists: {total_unique_fixtures_in_db}")
             logger.info(f"Total Fixture Details Already Present: {len(existing_match_ids_str)}")
             logger.info(f"Fixtures Requiring Detail Fetch: 0")
             logger.info(f"Total Execution Time: {end_time - start_time:.2f} seconds")
             logger.info("--- Historical Fetch Complete ---")
             return

        # --- Step 2: Process Details for Missing Fixtures ---
        sorted_fixtures_to_process = sorted(list(fixtures_to_process_int))
        logger.info(f"--- Step 2: Processing Details for {total_fixtures_to_process} Missing Fixtures ---")

        process_tasks = [
            process_fixture_async(session, semaphore, fixture_id, loop)
            for fixture_id in sorted_fixtures_to_process # This uses the set difference result
        ]

        # Run processing tasks and collect results
        results = []
        processed_count = 0
        process_start_time = time.time()
        fixture_detail_errors = 0 # Errors specifically from process_fixture_async
        for f in asyncio.as_completed(process_tasks):
            try:
                result = await f
                results.append(result)
                if result not in ["SAVED_DETAILS", "SKIPPED"]:
                     fixture_detail_errors += 1
                     logger.warning(f"process_fixture_async returned status '{result}' for a fixture expected to be missing.")
            except Exception as e:
                 logger.error(f"Critical error awaiting processing task result: {e}", exc_info=True)
                 results.append("ERROR_UNCAUGHT")
                 fixture_detail_errors += 1
            finally:
                 processed_count += 1
                 if processed_count % 100 == 0 or processed_count == total_fixtures_to_process:
                      elapsed_time = time.time() - process_start_time
                      rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                      logger.info(f"Processed {processed_count}/{total_fixtures_to_process} missing fixture details... ({rate:.2f} fix/sec)")

    end_time = time.time()

    # --- Step 3: Summarize results ---
    total_processed_attempted = len(results)
    summary = {
        "SAVED_DETAILS": results.count("SAVED_DETAILS"),
        "SKIPPED": results.count("SKIPPED"), # Should ideally be 0
        "ERROR_DB_CHECK": results.count("ERROR_DB_CHECK"), # Should be 0
        "ERROR_PROCESS": results.count("ERROR_PROCESS"),
        "ERROR_UNCAUGHT": results.count("ERROR_UNCAUGHT"),
    }
    total_errors = fixture_detail_errors # Only detail processing errors are relevant now

    logger.info("--- Full Historical Data Fetch Summary ---")
    logger.info(f"Target Seasons: {target_seasons}")
    logger.info(f"Teams Considered: {len(TEAM_ID_MAPPING)}")
    logger.info(f"Total Unique Fixtures Read from DB Lists: {total_unique_fixtures_in_db}")
    logger.info(f"Total Fixture Details Already Present in 'matches': {len(existing_match_ids_str)}")
    logger.info(f"Fixtures Identified for Detail Fetch: {total_fixtures_to_process}")
    logger.info(f"Total Fixture Details Processed (attempted this run): {total_processed_attempted}")
    logger.info(f"  -> Newly Saved Details: {summary['SAVED_DETAILS']}")
    if summary['SKIPPED'] > 0:
        logger.warning(f"  -> Skipped during detail processing (unexpected): {summary['SKIPPED']}")
    logger.info(f"  -> Errors Encountered (Fixture Detail Processing): {total_errors}")
    if total_errors > 0:
         logger.info(f"     - Detail Process Errors: {summary['ERROR_PROCESS']}")
         logger.info(f"     - Uncaught Errors: {summary['ERROR_UNCAUGHT']}")
    logger.info(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    logger.info("--- Historical Fetch Complete ---")


# Add to historical_data_fetcher.py - New function to get and save team fixture lists

async def fetch_and_save_team_fixture_lists(target_seasons: List[int]):
    """
    Specifically focuses on rebuilding the team_season_fixtures collection.
    1. Loops through all teams in TEAM_ID_MAPPING
    2. For each team, fetches fixture IDs for each season
    3. Saves the team-season fixture lists to MongoDB
    """
    logger.info(f"Starting team fixture list rebuild for {len(TEAM_ID_MAPPING)} teams across {len(target_seasons)} seasons")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS * 2)
    success_count = 0
    
    async with aiohttp.ClientSession(connector=connector) as session:
        for team_name, team_info in TEAM_ID_MAPPING.items():
            team_id_str = team_info.get("mongodb_id")
            if not team_id_str:
                logger.warning(f"Skipping team {team_name}: No mongodb_id")
                continue
                
            try:
                team_id = int(team_id_str)
                logger.info(f"Processing team: {team_name} (ID: {team_id})")
                
                for season in target_seasons:
                    # Skip if already exists in DB (unless forced)
                    if db_manager.check_team_season_fixture_list_exists(team_id, season):
                        logger.info(f"Team {team_name} season {season} fixture list already exists. Skipping.")
                        continue
                    
                    fixture_ids = await fetch_team_fixtures_async(session, semaphore, team_id, season)
                    if fixture_ids is not None:
                        # Save to MongoDB
                        saved = db_manager.save_team_season_fixture_list(team_id, season, fixture_ids)
                        if saved:
                            success_count += 1
                            logger.info(f"Successfully saved {len(fixture_ids)} fixtures for {team_name} season {season}")
                        else:
                            logger.error(f"Failed to save fixture list for {team_name} season {season}")
                    else:
                        logger.error(f"Failed to fetch fixtures for {team_name} season {season}")
            except ValueError:
                logger.warning(f"Skipping team {team_name}: Invalid mongodb_id format: {team_id_str}")
    
    logger.info(f"Team fixture list rebuild complete. Successfully saved {success_count} team-season fixture lists")
    return success_count

# New async main function to orchestrate the steps
async def main_fetch_pipeline(target_seasons: List[int]):
    """Runs the two-step historical data fetching pipeline."""
    logger.info("Attempting to initialize DB Manager...")
    db_initialized = False
    try:
        if not db_manager._initialized:
             db_manager.__init__() # Initialize synchronously if not already done
        db_initialized = db_manager._initialized

        if not db_initialized:
            logger.error("DB Manager failed to initialize. Cannot run fetch.")
            return # Exit if DB fails

        if not TEAM_ID_MAPPING:
            logger.error("TEAM_ID_MAPPING is empty. Cannot proceed with fetch.")
            return # Exit if no teams

        logger.info("DB Manager Initialized.")
        logger.info("--- STEP 1: Fetching and Saving Team Fixture Lists ---")
        await fetch_and_save_team_fixture_lists(target_seasons) # Await Step 1

        logger.info(f"--- STEP 2: Running Full Historical Fetch for Details ---")
        await run_full_historical_fetch(target_seasons) # Await Step 2

    except ConnectionFailure as cf:
         logger.error(f"MongoDB Connection Failure during pipeline: {cf}", exc_info=False)
         logger.error("Please ensure MONGO_URI is correct and the database is accessible.")
         traceback.print_exc()
    except Exception as e:
        logger.error(f"Critical error during fetch pipeline execution: {e}", exc_info=True)
    # Note: DB connection is closed in the finally block of the __main__ section


if __name__ == "__main__":
    # --- Configuration for the run ---
    TARGET_SEASONS = [2024, 2023, 2022, 2021, 2020, 2019] # Seasons from 2024 to 2019

    logger.info(f"Selected Target Seasons: {TARGET_SEASONS}")
    logger.info(f"Processing teams from TEAM_ID_MAPPING (ensure db_ids/team_id_mappings.py exists and is importable)")

    try:
        # Run the entire async pipeline
        asyncio.run(main_fetch_pipeline(TARGET_SEASONS))

    except Exception as e:
        # Catch any broad exceptions during asyncio.run() setup or teardown if any
        logger.error(f"Error running the main async pipeline: {e}", exc_info=True)
    finally:
        # Ensure connection closure happens even if main_fetch_pipeline fails early
        if 'db_manager' in locals() and hasattr(db_manager, '_initialized') and db_manager._initialized and hasattr(db_manager, '_client') and db_manager._client:
             db_manager.close_connection()
             logger.info("DB Connection closed.")
        elif 'db_manager' in locals() and hasattr(db_manager, '_initialized') and not db_manager._initialized:
             logger.info("DB Manager was not initialized, no connection to close.")
        else:
             logger.info("DB connection was likely already closed or not established.") 