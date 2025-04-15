from datetime import datetime, timezone
import logging
from typing import Optional, Dict, Any
import asyncio

from get_data.api_football.endpoints.game_scraper import GameScraper
from get_data.api_football.endpoints.api_manager import api_manager
from get_data.api_football.endpoints.match_processor import MatchProcessor
from get_data.api_football.endpoints.odds_fetcher import OddsFetcher
from get_data.api_football.db_mongo import db_manager

logger = logging.getLogger(__name__)

# Make the main function async
async def fetch_all_data(target_date: Optional[datetime] = None, force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Fetch all necessary data sequentially: games, matches, and odds.
    Components now interact directly with MongoDB.
    Handles async operations correctly.
    
    Args:
        target_date: The date to fetch data for. If None, uses today's date.
        force_reprocess: If True, components might skip DB existence checks 
                         (NOTE: requires implementation within components if needed).
                         Currently, components check existence and skip if data is present.
        
    Returns:
        dict: Results containing success status and data processing summary.
    """
    if target_date is None:
        target_date = datetime.now(timezone.utc)

    date_str = target_date.strftime("%Y-%m-%d")
    logger.info(f"🗓️ Processing data for: {date_str}")

    # Initialize API manager once
    try:
        api_manager.initialize()
        logger.info("API Manager Initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize API Manager: {str(e)}")
        return {"success": False, "error": "API Manager initialization failed", "date": date_str}

    results = {
        "success": True,
        "date": date_str,
        "steps": {
            "scrape_games": {"success": False, "message": "Not run"},
            "process_matches": {"success": False, "message": "Not run"},
            "fetch_odds": {"success": False, "message": "Not run"}
        },
        "games_data": None, # Store fetched games data for subsequent steps
    }

    # --- Step 1: Get games data --- 
    try:
        logger.info("--- Running Step 1: Scraping Games ---")
        # Assuming GameScraper is refactored: no base_dir, saves to DB, returns data dict
        scraper = GameScraper() 

        # Convert target_date to naive datetime if scraper requires it
        date_obj = target_date.replace(tzinfo=None) if target_date.tzinfo else target_date
        games_data = scraper.get_games(date_obj) # Assuming it saves to DB and returns dict

        if not games_data or games_data.get("total_matches", 0) == 0:
            # Check if data exists in DB even if API returned nothing
            games_data_from_db = db_manager.get_daily_games(date_str)
            if games_data_from_db and games_data_from_db.get("total_matches", 0) > 0:
                 logger.warning(f"⚠️ No games returned by API for {date_str}, but data exists in DB. Using DB data.")
                 results["games_data"] = games_data_from_db
                 results["steps"]["scrape_games"]["success"] = True
                 results["steps"]["scrape_games"]["message"] = f"Used existing DB data ({games_data_from_db.get('total_matches')} matches)"
            else:
                logger.warning(f"⚠️ No games found or saved for date {date_str}")
                results["steps"]["scrape_games"]["success"] = True # Not a failure if no games exist
                results["steps"]["scrape_games"]["message"] = "No games found for this date."
                # Don't mark overall success as False, allow odds check etc. if needed
        else:
            # Data scraped and saved by scraper.get_games
            results["games_data"] = games_data
            results["steps"]["scrape_games"]["success"] = True
            results["steps"]["scrape_games"]["message"] = f"Successfully scraped and saved {games_data.get('total_matches', 0)} games to DB."
            logger.info(f"✅ Step 1 finished: {results['steps']['scrape_games']['message']}")

    except Exception as e:
        logger.error(f"❌ Error in Step 1 (Scraping Games): {str(e)}", exc_info=True)
        results["success"] = False
        results["steps"]["scrape_games"]["success"] = False
        results["steps"]["scrape_games"]["message"] = f"Failed: {str(e)}"
        return results # Stop processing if games scraping fails critically

    # --- Step 2: Process matches --- 
    # Proceed only if Step 1 was successful and produced games data
    if results["steps"]["scrape_games"]["success"] and results["games_data"]:
        try:
            logger.info("--- Running Step 2: Processing Matches ---")
            processor = MatchProcessor() # No base_dir needed
            
            # Call the async function directly using await
            processed_result = await processor.process_games_data_async(results["games_data"])

            if isinstance(processed_result, dict) and processed_result.get("status") == "success":
                results["steps"]["process_matches"]["success"] = True
                results["steps"]["process_matches"]["message"] = "Successfully processed matches and saved to DB."
                logger.info(f"✅ Step 2 finished: {results['steps']['process_matches']['message']}")
            else:
                error_msg = processed_result.get("error", "Unknown error during match processing")
                logger.error(f"❌ Error in Step 2 (Processing Matches): {error_msg}")
                results["steps"]["process_matches"]["success"] = False
                results["steps"]["process_matches"]["message"] = f"Failed: {error_msg}"
                results["success"] = False # Mark overall as failed if match processing fails

        except Exception as e:
            logger.error(f"❌ Error in Step 2 (Processing Matches): {str(e)}", exc_info=True)
            results["success"] = False
            results["steps"]["process_matches"]["success"] = False
            results["steps"]["process_matches"]["message"] = f"Failed: {str(e)}"
            # Optionally return results here if match processing failure is critical

    elif not results["games_data"]:
         logger.warning("⚠️ Skipping Step 2 (Processing Matches): No games data available.")
         results["steps"]["process_matches"]["message"] = "Skipped - No games data."
    else:
         logger.warning("⚠️ Skipping Step 2 (Processing Matches): Step 1 did not succeed.")
         results["steps"]["process_matches"]["message"] = "Skipped - Step 1 failed."


    # --- Step 3: Fetch odds --- 
    # Proceed only if Step 1 was successful (meaning we have a date and potentially games in DB)
    if results["steps"]["scrape_games"]["success"]:
        try:
            logger.info("--- Running Step 3: Fetching Odds ---")
            odds_fetcher = OddsFetcher() # No base_dir needed
            
            # Call the async function directly using await
            # Pass only the date string, as it fetches fixtures from DB
            odds_result = await odds_fetcher.process_daily_report(date_str)

            successful_count = odds_result.get("successful", 0)
            failed_count = odds_result.get("failed", 0)

            # Success if some odds fetched or none needed/found (0 successful, 0 failed)
            if successful_count >= 0 and failed_count == 0: 
                results["steps"]["fetch_odds"]["success"] = True
                results["steps"]["fetch_odds"]["message"] = f"Processed odds. Successful: {successful_count}, Skipped/Exist: {odds_result.get('skipped', 0)}. No failures."
                logger.info(f"✅ Step 3 finished: {results['steps']['fetch_odds']['message']}")
            elif successful_count > 0 and failed_count > 0:
                 results["steps"]["fetch_odds"]["success"] = True # Partial success
                 results["steps"]["fetch_odds"]["message"] = f"Processed odds. Successful: {successful_count}, Failed: {failed_count}, Skipped/Exist: {odds_result.get('skipped', 0)}."
                 logger.warning(f"⚠️ Step 3: Partial success. Failed to fetch/save odds for {failed_count} fixtures.")
                 # Decide if partial failure should mark overall success as False
                 # results["success"] = False
            else: # Only failures occurred or critical error
                error_msg = odds_result.get("error", f"Failed for {failed_count} fixtures.")
                logger.error(f"❌ Error in Step 3 (Fetching Odds): {error_msg}")
                results["steps"]["fetch_odds"]["success"] = False
                results["steps"]["fetch_odds"]["message"] = f"Failed: {error_msg}"
                results["success"] = False # Mark overall as failed

        except Exception as e:
            logger.error(f"❌ Critical Error in Step 3 (Fetching Odds): {str(e)}", exc_info=True)
            results["success"] = False
            results["steps"]["fetch_odds"]["success"] = False
            results["steps"]["fetch_odds"]["message"] = f"Failed Critically: {str(e)}"

    else:
        logger.warning("⚠️ Skipping Step 3 (Fetching Odds): Step 1 did not succeed.")
        results["steps"]["fetch_odds"]["message"] = "Skipped - Step 1 failed."

    # Final Summary
    logger.info("--- Data Fetching Summary ---")
    logger.info(f"Date: {results['date']}")
    logger.info(f"Overall Success: {results['success']}")
    logger.info(f"Step 1 (Games): Success={results['steps']['scrape_games']['success']}, Message='{results['steps']['scrape_games']['message']}'")
    logger.info(f"Step 2 (Matches): Success={results['steps']['process_matches']['success']}, Message='{results['steps']['process_matches']['message']}'")
    logger.info(f"Step 3 (Odds): Success={results['steps']['fetch_odds']['success']}, Message='{results['steps']['fetch_odds']['message']}'")

    return results

# Keep __main__ block for testing, but adapt for async
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running data_fetcher directly (async)...")

    async def run_main():
        # Example: Fetch data for today
        fetch_result = await fetch_all_data(force_reprocess=False)
        logger.info(f"Direct run completed. Overall Success: {fetch_result.get('success')}")

        # Remember to close DB connection if managed globally and needs explicit close
        try:
            pass
        except Exception as e:
            logger.warning(f"Could not close DB connection (if needed): {e}")

    # Run the async main function
    asyncio.run(run_main()) 