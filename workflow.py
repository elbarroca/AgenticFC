# workflow.py

import asyncio
import logging
import os
import sys
import argparse
from datetime import datetime, timezone, timedelta

# --- Workflow Command Line Arguments ---
# python workflow.py (uses defaults, runs for today)
# python workflow.py --date 2023-10-27 (runs for a specific date)
# python workflow.py --skip_fetch (skips fetching, assumes data exists)
# python workflow.py --debug (runs with verbose logging)
# python workflow.py --force_fetch --skip_odds (forces data fetching, processes, but skips odds matching)

# --- Path Setup ---
# Ensure the script can find modules in subdirectories
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add specific subdirectories if needed (might be redundant if project_root is sufficient)
get_data_dir = os.path.join(project_root, 'get_data')
score_data_dir = os.path.join(project_root, 'score_data')
if get_data_dir not in sys.path:
    sys.path.insert(1, get_data_dir)
if score_data_dir not in sys.path:
    sys.path.insert(1, score_data_dir)

# --- Imports from Project Modules ---
try:
    # From get_data/get_all_data.py
    from get_data.get_all_data import get_data, setup_database_directories
    # From score_data/combined.py
    from score_data.combined import (
        EnhancedSoccerMatchProcessor,
        add_matched_odds_info,
        INPUT_DIR_DEFAULT as COMBINED_INPUT_DIR_DEFAULT,       # Rename to avoid clash
        PROCESSED_DIR_DEFAULT as COMBINED_OUTPUT_DIR_DEFAULT, # Rename to avoid clash
        DEFAULT_BOOKMAKER_NAME as COMBINED_BOOKMAKER_DEFAULT # Rename to avoid clash
    )
    # From get_data/api_football/db_mongo.py (via combined)
    from get_data.api_football.db_mongo import db_manager
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Ensure 'get_data/get_all_data.py' and 'score_data/combined.py' exist and are importable.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Configure Logging ---
# Use a central logger for the workflow
logger = logging.getLogger("WorkflowCoordinator")
# Basic configuration initially, will be refined in main
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Main Workflow Function ---
async def main_workflow():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Orchestrates data fetching and processing.")

    # Date argument (optional, defaults to today UTC)
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Target date in YYYY-MM-DD format. Defaults to today (UTC).'
    )
    # Directory arguments (use defaults from combined script)
    parser.add_argument(
        '--raw_output_dir', # Renamed from combined's input_dir perspective
        type=str,
        default=COMBINED_INPUT_DIR_DEFAULT,
        help='Directory where raw daily game JSON files are stored (output of get_data/extract_daily_games, input for combined processor).'
    )
    parser.add_argument(
        '--processed_output_dir', # Renamed from combined's output_dir perspective
        type=str,
        default=COMBINED_OUTPUT_DIR_DEFAULT,
        help='Directory to save processed match JSON files.'
    )
    # Odds argument
    parser.add_argument(
        '--bookmaker',
        type=str,
        default=COMBINED_BOOKMAKER_DEFAULT,
        help='Name of the bookmaker for odds matching.'
    )
    # Control flags
    parser.add_argument(
        '--force_fetch',
        action='store_true',
        help='Force reprocessing/fetching of API-Football and Statarea data even if it seems up-to-date.'
    )
    parser.add_argument(
        '--skip_fetch',
        action='store_true',
        help='Skip the entire data fetching phase (get_data).'
    )
    parser.add_argument(
        '--skip_processing',
        action='store_true',
        help='Skip the initial game processing phase (combined step 1).'
    )
    parser.add_argument(
        '--skip_odds',
        action='store_true',
        help='Skip the odds matching and update phase (combined step 2).'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging for all components.'
    )

    args = parser.parse_args()

    # --- Configure Logging Level ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    # Configure the root logger
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    # Set level for specific loggers used by imported modules
    logging.getLogger("get_data").setLevel(log_level)
    logging.getLogger("api_football").setLevel(log_level)
    logging.getLogger("statarea").setLevel(log_level)
    logging.getLogger("extract_daily_games").setLevel(log_level)
    logging.getLogger("CombinedProcessor").setLevel(log_level) # From combined.py
    logging.getLogger("EnhancedSoccerMatchProcessor").setLevel(log_level) # From process_daily_games.py
    logging.getLogger("OddFinder").setLevel(log_level) # Assuming odd_finder uses this name
    logging.getLogger("WorkflowCoordinator").setLevel(log_level) # Our logger

    if args.debug:
         # Set debug for db_manager logger if it exists and you want its output
        try:
             logging.getLogger('get_data.api_football.db_mongo').setLevel(logging.DEBUG)
        except Exception:
             pass # Ignore if logger doesn't exist
        logger.info("--- Debug logging enabled ---")


    # --- Determine Target Date ---
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Please use YYYY-MM-DD.")
            sys.exit(1)
    else:
        # Default to today's date in UTC
        target_date = datetime.now(timezone.utc)
        # Optional: If you usually want yesterday's completed data, uncomment below
        # target_date = datetime.now(timezone.utc) - timedelta(days=1)

    date_str = target_date.strftime("%Y-%m-%d")

    logger.info("=================================================")
    logger.info(f"🚀 Starting Workflow for Date: {date_str}")
    logger.info("=================================================")
    logger.info(f"Configuration:")
    logger.info(f"  Target Date: {date_str}")
    logger.info(f"  Raw Data Output Dir (Combined Input): {args.raw_output_dir}")
    logger.info(f"  Processed Data Output Dir: {args.processed_output_dir}")
    logger.info(f"  Bookmaker for Odds: {args.bookmaker}")
    logger.info(f"  Force Fetch: {args.force_fetch}")
    logger.info(f"  Skip Fetch: {args.skip_fetch}")
    logger.info(f"  Skip Processing: {args.skip_processing}")
    logger.info(f"  Skip Odds: {args.skip_odds}")
    logger.info(f"  Debug Mode: {args.debug}")
    logger.info("-------------------------------------------------")


    # --- Phase 1: Data Fetching (get_data) ---
    fetch_successful = True
    if not args.skip_fetch:
        logger.info("--- Phase 1: Starting Data Fetching (get_data) ---")
        try:
            # Setup directories needed by get_data first
            if not setup_database_directories():
                logger.error("Failed to set up database directories. Aborting.")
                sys.exit(1)

            # Call the main function from get_all_data.py
            fetch_results = await get_data(target_date=target_date, force_reprocess=args.force_fetch)
            fetch_successful = fetch_results.get("success", False)
            if fetch_successful:
                 logger.info("--- Phase 1: Data Fetching Completed Successfully ---")
            else:
                 logger.error("--- Phase 1: Data Fetching Completed with Errors ---")
                 # Decide if we should stop the workflow
                 logger.error("Aborting workflow due to data fetching errors.")
                 sys.exit(1) # Exit if fetching failed

        except Exception as e:
            logger.error(f"Unhandled error during Data Fetching phase: {e}", exc_info=True)
            fetch_successful = False
            logger.error("Aborting workflow due to unhandled data fetching error.")
            sys.exit(1) # Exit on unhandled exception
    else:
        logger.info("--- Phase 1: Skipping Data Fetching (get_data) as requested ---")
        fetch_successful = True # Assume success if skipped

    # --- Phase 2: Initial Game Processing (combined step 1) ---
    processing_successful = True
    if not args.skip_processing and fetch_successful: # Only run if fetch was ok (or skipped)
        logger.info("\n--- Phase 2: Starting Initial Match Data Processing (Combined Step 1) ---")
        try:
            # Use the directory arguments passed to the workflow script
            processor = EnhancedSoccerMatchProcessor(
                input_dir=args.raw_output_dir,   # Where raw daily games are
                output_dir=args.processed_output_dir # Where processed files will go
            )
            # This method handles its own logging internally
            processor.process_all_matches_advanced()
            logger.info("--- Phase 2: Initial Match Data Processing Completed ---")
        except Exception as e:
            logger.error(f"Error during initial match processing phase: {e}", exc_info=True)
            processing_successful = False # Mark failure
            logger.error("Continuing to cleanup, but odds phase will be skipped.")
    elif args.skip_processing:
        logger.info("--- Phase 2: Skipping Initial Game Processing as requested ---")
        processing_successful = True # Assume success if skipped
    elif not fetch_successful:
         logger.warning("--- Phase 2: Skipping Initial Game Processing due to previous fetch errors ---")
         processing_successful = False


    # --- Phase 3: Odds Matching and Update (combined step 2) ---
    odds_successful = True
    if not args.skip_odds and processing_successful: # Only run if processing was ok (or skipped successfully)
        logger.info("\n--- Phase 3: Starting Odds Matching and Update (Combined Step 2) ---")
        try:
            # --- <<< MODIFIED SECTION START >>> ---
            # Attempt to trigger db_manager connection/initialization before proceeding
            logger.info("Ensuring MongoDB connection is available for odds lookup...")
            try:
                # Force reconnection if needed
                if not db_manager._initialized or db_manager._client is None:
                    # Re-initialize the connection
                    db_manager.__init__()
                
                # Verify connection by running a ping command
                db_manager._client.admin.command('ping')
                
                # Verify the odds database is available
                if 'odds' not in db_manager._dbs:
                    raise ConnectionError("'odds' database not found in db_manager._dbs")
                
                logger.info("MongoDB connection verified and ready for odds lookup.")
            except Exception as db_error:
                logger.error(f"Failed to ensure MongoDB connection: {db_error}", exc_info=True)
                raise ConnectionError(f"Cannot proceed with odds matching: {db_error}")
            # --- <<< MODIFIED SECTION END >>> ---


            # Call the odds function from combined.py
            odds_successful = add_matched_odds_info(
                processed_dir=args.processed_output_dir, # Directory with processed files
                bookmaker_name=args.bookmaker
            )
            if odds_successful:
                logger.info("--- Phase 3: Odds Matching and Update Phase Completed Successfully ---")
            else:
                # Note: add_matched_odds_info currently returns True even if DB errors occurred within get_odds_from_db.
                # It only returns False if its own explicit error count > 0.
                # We might need to refine error propagation.
                logger.error("--- Phase 3: Odds Matching and Update Phase potentially completed with underlying DB errors (check logs) ---")
                # Consider setting odds_successful = False here based on logged errors if needed for final status.

        except Exception as e:
             logger.error(f"Unhandled error during Odds Matching phase: {e}", exc_info=True)
             odds_successful = False
             logger.error("Continuing to cleanup.")

    elif args.skip_odds:
        logger.info("--- Phase 3: Skipping Odds Matching Phase as requested ---")
        odds_successful = True # Assume success if skipped
    elif not processing_successful:
        logger.warning("--- Phase 3: Skipping Odds Matching Phase due to errors in the processing phase ---")
        odds_successful = False

    # --- Phase 4: Cleanup ---
    logger.info("\n--- Phase 4: Starting Cleanup ---")
    try:
        # Ensure the MongoDB connection is closed if db_manager is initialized
        if db_manager:
            db_manager.close_connection()
            logger.info("MongoDB connection closed via db_manager.")
        else:
             logger.info("db_manager not initialized, skipping MongoDB connection close.")
    except Exception as e:
         logger.error(f"Error closing MongoDB connection: {e}", exc_info=True)


    # --- Final Summary ---
    logger.info("\n=================================================")
    logger.info("🏁 Workflow Finished")
    logger.info("=================================================")
    final_success = fetch_successful and processing_successful and odds_successful

    if final_success:
        logger.info("✅ Workflow completed successfully.")
        sys.exit(0)
    else:
        logger.warning("⚠️ Workflow finished with errors.")
        logger.warning(f"  Fetch Success: {fetch_successful}")
        logger.warning(f"  Processing Success: {processing_successful}")
        logger.warning(f"  Odds Success: {odds_successful}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_workflow())