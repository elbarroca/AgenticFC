from datetime import datetime, timezone
import logging
from typing import Optional, Dict, Any, List
import asyncio
import sys
from pathlib import Path

# Add project root to system path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent) # Assuming project root is 4 levels up
sys.path.insert(0, project_root)
# Use relative imports since the project root is already in sys.path
from api_football.endpoints.game_scraper import GameScraper
from api_football.endpoints.api_manager import api_manager
from api_football.endpoints.match_processor import MatchProcessor
from api_football.endpoints.odds_fetcher import OddsFetcher
from api_football.db_mongo import db_manager
from api_football.statarea_async_scraper import run_scraper_async

logger = logging.getLogger(__name__)

async def fetch_workflow_data(target_date: Optional[datetime] = None, force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Fetches and processes data for the Agentic FC Workflow for a specific date.
    Orchestrates fetching games of the day, processing match details, and fetching odds,
    storing results in the corresponding MongoDB collections within the Agentic FC Workflow DB.

    Workflow:
    1. Fetch games for the target_date -> save to 'games_of_the_day' collection.
    2. Extract fixture IDs from fetched games.
    3. Fetch match details for fixture IDs -> save to 'game_details' collection.
    4. Fetch odds for fixture IDs -> save to 'odds' collection.
    5. Fetch StatArea match history data for relevant teams (period=15).

    Args:
        target_date: The date to fetch data for (UTC). Defaults to today.
        force_reprocess: If True, components might bypass existence checks.

    Returns:
        Dict: Results summarizing the success status and actions taken.
    """
    if target_date is None:
        target_date = datetime.now(timezone.utc)

    date_str = target_date.strftime("%Y-%m-%d")
    logger.info(f"🚀 Starting Agentic FC Workflow data fetch for: {date_str}")

    # Initialize API manager (consider if this needs cluster-specific setup)
    try:
        api_manager.initialize() # Ensure this uses credentials/config for the correct API source
        logger.info("API Manager Initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize API Manager: {str(e)}")
        return {"success": False, "error": "API Manager initialization failed", "date": date_str}

    # Ensure DB Manager is connected to the 'Agentic FC Workflow' database
    logger.info(f"Ensuring DB connection to 'Agentic FC Workflow'...")

    results = {
        "success": True,
        "date": date_str,
        "steps": {
            "fetch_games_of_day": {"success": False, "message": "Not run", "fixture_ids": []},
            "process_match_details": {"success": False, "message": "Not run", "processed_count": 0, "failed_fixtures": []},
            "fetch_odds": {"success": False, "message": "Not run", "processed_count": 0, "failed_fixtures": []},
            "fetch_statarea_data": {"success": False, "message": "Not run", "processed_count": 0}
        }
    }
    fixture_ids: List[int] = []

    # --- Step 1: Fetch Games of the Day ---
    try:
        logger.info(f"--- Running Step 1: Fetching Games for {date_str} ---")
        scraper = GameScraper()
        naive_date = target_date.replace(tzinfo=None)

        # Run the synchronous get_games() in a threadpool
        organized_data = await asyncio.get_event_loop().run_in_executor(
            None, scraper.get_games, naive_date
        )

        if organized_data and organized_data.get("total_matches", 0) > 0:
            # 1a) Save the full daily summary
            db_manager.save_daily_games(date_str, organized_data)
            # 1b) Extract fixture IDs for steps 2 & 3
            for league_data in organized_data.get("leagues", {}).values():
                for match in league_data.get("matches", []):
                    try:
                        fixture_ids.append(int(match.get("id")))
                    except (TypeError, ValueError):
                        continue
            results["steps"]["fetch_games_of_day"].update({
                "success": True,
                "message": f"Fetched & saved {len(fixture_ids)} games to 'daily_games'.",
                "fixture_ids": fixture_ids
            })
            logger.info(f"✅ Step 1 finished: {len(fixture_ids)} fixtures.")
        else:
            # No new games — try loading existing summary
            logger.warning(f"⚠️ No games fetched for {date_str}. Loading from 'daily_games'...")
            existing = db_manager.get_daily_games(date_str)
            if existing:
                for league_data in existing.get("leagues", {}).values():
                    for match in league_data.get("matches", []):
                        try:
                            fixture_ids.append(int(match.get("id")))
                        except (TypeError, ValueError):
                            continue
                results["steps"]["fetch_games_of_day"].update({
                    "success": True,
                    "message": f"Loaded {len(fixture_ids)} games from 'daily_games'.",
                    "fixture_ids": fixture_ids
                })
                logger.info(f"✅ Step 1 loaded: {len(fixture_ids)} fixtures from cache.")
            else:
                results["steps"]["fetch_games_of_day"].update({
                    "success": True,
                    "message": "No games scheduled for this date.",
                    "fixture_ids": []
                })

    except Exception as e:
        logger.error(f"❌ Critical Error in Step 1 (Fetching Games): {str(e)}", exc_info=True)
        results["success"] = False
        results["steps"]["fetch_games_of_day"]["success"] = False
        results["steps"]["fetch_games_of_day"]["message"] = f"Failed Critically: {str(e)}"
        return results

    # --- Steps 2 & 3: Process Match Details & Fetch Odds ---
    # (keeping the existing code for these steps unchanged)
    
    # --- Step 2: Process Match Details (Now Updates 'matches' Collection) ---
    if fixture_ids:
        try:
            logger.info(f"--- Running Step 2: Processing Match Details in 'matches' for {len(fixture_ids)} fixtures ---")
            processor = MatchProcessor()
            # Call the new process_fixtures method
            process_result = await processor.process_fixtures(fixture_ids, force_reprocess=force_reprocess)

            processed_count = process_result.get("processed_count", 0)
            skipped_count = process_result.get("skipped_count", 0) # Added skipped count handling
            failed_fixtures = process_result.get("failed_fixtures", [])

            results["steps"]["process_match_details"]["processed_count"] = processed_count
            results["steps"]["process_match_details"]["failed_fixtures"] = failed_fixtures
            results["steps"]["process_match_details"]["skipped_count"] = skipped_count # Store skipped count

            if not failed_fixtures:
                results["steps"]["process_match_details"]["success"] = True
                results["steps"]["process_match_details"]["message"] = f"Successfully processed {processed_count} fixtures in 'matches'. Skipped {skipped_count}."
                logger.info(f"✅ Step 2 finished: {results['steps']['process_match_details']['message']}")
            else:
                results["steps"]["process_match_details"]["success"] = True # Partial success if some processed
                results["steps"]["process_match_details"]["message"] = f"Processed {processed_count}, skipped {skipped_count}, failed to process {len(failed_fixtures)} fixtures: {failed_fixtures}"
                logger.warning(f"⚠️ Step 2: Partial success. Failed fixtures: {failed_fixtures}")

        except Exception as e:
            logger.error(f"❌ Critical Error in Step 2 (Processing Matches): {str(e)}", exc_info=True)
            results["success"] = False
            results["steps"]["process_match_details"]["success"] = False
            results["steps"]["process_match_details"]["message"] = f"Failed Critically while processing 'matches': {str(e)}"

    else:
         logger.warning("⚠️ Skipping Step 2 (Processing Matches): No fixture IDs from Step 1.")
         results["steps"]["process_match_details"]["message"] = "Skipped - No fixture IDs."

    # --- Step 3: Fetch Odds ---
    if fixture_ids:
        try:
            logger.info(f"--- Running Step 3: Fetching Odds for {len(fixture_ids)} fixtures ---")
            odds_fetcher = OddsFetcher()
            # Call the new process_fixtures_odds method
            odds_result = await odds_fetcher.process_fixtures_odds(fixture_ids, force_reprocess=force_reprocess)

            processed_count = odds_result.get("processed_count", 0)
            skipped_count = odds_result.get("skipped_count", 0) # Added skipped count handling
            failed_fixtures = odds_result.get("failed_fixtures", [])

            results["steps"]["fetch_odds"]["processed_count"] = processed_count
            results["steps"]["fetch_odds"]["failed_fixtures"] = failed_fixtures
            results["steps"]["fetch_odds"]["skipped_count"] = skipped_count # Store skipped count

            if not failed_fixtures:
                results["steps"]["fetch_odds"]["success"] = True
                results["steps"]["fetch_odds"]["message"] = f"Successfully processed odds for {processed_count} fixtures. Skipped {skipped_count}."
                logger.info(f"✅ Step 3 finished: {results['steps']['fetch_odds']['message']}")
            else:
                results["steps"]["fetch_odds"]["success"] = True # Partial success if some processed
                results["steps"]["fetch_odds"]["message"] = f"Processed {processed_count}, skipped {skipped_count}, failed to fetch/save odds for {len(failed_fixtures)} fixtures: {failed_fixtures}"
                logger.warning(f"⚠️ Step 3: Partial success. Failed fixtures: {failed_fixtures}")

        except Exception as e:
            logger.error(f"❌ Critical Error in Step 3 (Fetching Odds): {str(e)}", exc_info=True)
            results["success"] = False
            results["steps"]["fetch_odds"]["success"] = False
            results["steps"]["fetch_odds"]["message"] = f"Failed Critically: {str(e)}"

    else:
        logger.warning("⚠️ Skipping Step 3 (Fetching Odds): No fixture IDs available.")
        results["steps"]["fetch_odds"]["message"] = "Skipped - No fixture IDs."

    # --- New Step 4: Fetch StatArea Match History Data ---
    try:
        logger.info(f"--- Running Step 4: Fetching StatArea Match History Data ---")
        
        # If we have fixtures, extract teams involved to prioritize them
        # This step is optional - we could also just run for all teams
        teams_to_scrape = None  # None means all teams
        
        # We only need period 15 for match history
        statarea_result = await run_scraper_async(
            team_count=teams_to_scrape,  # None means all teams
            periods=[15],  # We only need period 15 for match history
            force_update=force_reprocess
        )
        
        if statarea_result.get("success", False):
            results["steps"]["fetch_statarea_data"]["success"] = True
            processed_count = statarea_result.get("saved_to_mongodb", 0)
            results["steps"]["fetch_statarea_data"]["processed_count"] = processed_count
            results["steps"]["fetch_statarea_data"]["message"] = f"Successfully fetched and saved StatArea data for {processed_count} team entries."
            logger.info(f"✅ Step 4 finished: {results['steps']['fetch_statarea_data']['message']}")
        else:
            error_msg = statarea_result.get("error", "Unknown error")
            results["steps"]["fetch_statarea_data"]["success"] = False
            results["steps"]["fetch_statarea_data"]["message"] = f"Failed to fetch StatArea data: {error_msg}"
            logger.error(f"❌ Step 4 failed: {results['steps']['fetch_statarea_data']['message']}")
            # We don't consider this a critical failure for the whole workflow
            # results["success"] = False
            
    except Exception as e:
        logger.error(f"❌ Error in Step 4 (Fetching StatArea Data): {str(e)}", exc_info=True)
        results["steps"]["fetch_statarea_data"]["success"] = False
        results["steps"]["fetch_statarea_data"]["message"] = f"Failed with exception: {str(e)}"
        # We don't consider this a critical failure for the whole workflow
        # results["success"] = False

    # Final Summary
    logger.info("--- Agentic FC Workflow Data Fetch Summary ---")
    logger.info(f"Date: {results['date']}")
    logger.info(f"Overall Success: {results['success']}")
    logger.info(f"Step 1 (Games): Success={results['steps']['fetch_games_of_day']['success']}, Message='{results['steps']['fetch_games_of_day']['message']}'")
    logger.info(f"Step 2 (Update Matches): Success={results['steps']['process_match_details']['success']}, Message='{results['steps']['process_match_details']['message']}'")
    logger.info(f"Step 3 (Odds): Success={results['steps']['fetch_odds']['success']}, Message='{results['steps']['fetch_odds']['message']}'")
    logger.info(f"Step 4 (StatArea): Success={results['steps']['fetch_statarea_data']['success']}, Message='{results['steps']['fetch_statarea_data']['message']}'")

    # Standings data is not handled in this daily fetcher but the collection exists.
    logger.info("ℹ️ Note: Standings data processing is handled separately.")

    return results

# Keep __main__ block for testing, adapt for the new async function
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running Agentic FC Workflow data_fetcher directly (async)...")

    async def run_main():
        # Example: Fetch data for today
        # Set force_reprocess=True to potentially ignore existing data and refetch
        fetch_result = await fetch_workflow_data(force_reprocess=False)
        logger.info(f"Direct run completed. Overall Success: {fetch_result.get('success')}")

    # Run the async main function
    asyncio.run(run_main()) 