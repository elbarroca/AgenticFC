# score_data/combined_processor.py

import json
import os
import logging
import glob
from decimal import Decimal, ROUND_HALF_UP
import sys
import argparse
import traceback

# --- EnhancedSoccerMatchProcessor Class (from process_daily_games.py) ---
# (Includes all necessary methods like clean_value, calculate_form_points, etc.)
# Due to the size, I'll import the class directly instead of pasting it here.
# Assuming process_daily_games.py is in the same directory (score_data)
try:
    from .process_daily_games import EnhancedSoccerMatchProcessor
except ImportError:
    # Fallback if running as a script directly and '.' doesn't work
    from process_daily_games import EnhancedSoccerMatchProcessor

# --- Functions and Constants from odd_finder.py ---
try:
    # Assume odd_finder.py is in the same directory
    from .odd_finder import (
        get_fixture_id_from_filename,
        load_processed_match_data as load_match_data_for_odds, # Rename to avoid potential future conflicts
        get_odds_from_db,
        find_matched_bets,
        convert_decimals_to_strings,
        BOOKMAKER_NAME as DEFAULT_BOOKMAKER_NAME # Use a default, allow override later if needed
    )
except ImportError:
     # Fallback if running as a script directly
     from odd_finder import (
        get_fixture_id_from_filename,
        load_processed_match_data as load_match_data_for_odds,
        get_odds_from_db,
        find_matched_bets,
        convert_decimals_to_strings,
        BOOKMAKER_NAME as DEFAULT_BOOKMAKER_NAME
    )

# --- Database Manager Import ---
# Adjust path to go up one level from score_data to the project root
script_dir_for_import = os.path.dirname(os.path.abspath(__file__))
project_root_for_import = os.path.abspath(os.path.join(script_dir_for_import, '..'))
get_data_dir_for_import = os.path.join(project_root_for_import, 'get_data', 'api_football')

# Add project root to sys.path if not already present
if project_root_for_import not in sys.path:
    sys.path.insert(0, project_root_for_import)

try:
    from get_data.api_football.db_mongo import db_manager
except ImportError as e:
    print(f"Error importing db_manager: {e}")
    print(f"Attempted to add '{project_root_for_import}' to sys.path")
    print(f"Ensure the 'get_data/api_football/db_mongo.py' structure exists relative to the project root.")
    sys.exit(1)


# --- Configure Logging ---
# Use a logger specific to this combined script
logger = logging.getLogger("CombinedProcessor")
logger.setLevel(logging.INFO)
# Prevent adding duplicate handlers
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Constants ---
# Assumes script is run from project root or score_data dir
# Base directory setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..')) # Go up one level

INPUT_DIR_DEFAULT = os.path.join(project_root, "daily_games")
PROCESSED_DIR_DEFAULT = os.path.join(project_root, "processed_matches")


# --- Odds Finding and Updating Logic (Adapted from odd_finder.py main loop) ---
def add_matched_odds_info(processed_dir: str, bookmaker_name: str):
    """
    Scans processed match files, finds odds, calculates matched bets,
    and updates the files.
    """
    logger.info(f"\n--- Starting Odds Matching and Update Phase ---")
    logger.info(f"Scanning for processed matches in: {processed_dir}")
    logger.info(f"Using Bookmaker: {bookmaker_name}")

    if not os.path.isdir(processed_dir):
         logger.error(f"Processed matches directory not found: {processed_dir}")
         # Attempt to use current working directory's processed_matches as fallback
         fallback_dir = os.path.join(os.getcwd(), "processed_matches")
         logger.warning(f"Attempting fallback directory: {fallback_dir}")
         if os.path.isdir(fallback_dir):
              processed_dir = fallback_dir
              logger.info(f"Using fallback directory: {processed_dir}")
         else:
              logger.error("Fallback directory also not found. Cannot proceed with odds matching.")
              return False # Indicate failure

    search_pattern = os.path.join(processed_dir, '**', '*.json')
    json_files = glob.glob(search_pattern, recursive=True)
    # Filter out non-match files if necessary (e.g., summaries)
    json_files = [f for f in json_files if '_vs_' in os.path.basename(f)]

    if not json_files:
        logger.warning(f"No processed match JSON files found recursively in {processed_dir}")
        return True # Not an error, just no files to process

    processed_odds_count = 0
    files_updated_count = 0
    total_matched_bets_count = 0
    error_count = 0
    skipped_date_count = 0

    for json_file_path in json_files:
        relative_path = os.path.relpath(json_file_path, processed_dir)
        logger.debug(f"\n--- Processing for Odds: {relative_path} ---")
        processed_odds_count += 1
        fixture_id = None
        try:
            fixture_id = get_fixture_id_from_filename(os.path.basename(json_file_path))
            if not fixture_id:
                logger.warning(f"Skipping odds for file (no fixture ID): {relative_path}")
                continue

            # Load data *once* - Use the renamed import
            processed_data, match_date_simple = load_match_data_for_odds(json_file_path)
            if not processed_data:
                logger.error(f"Failed to load processed data from {json_file_path} for odds matching. Skipping.")
                error_count += 1
                continue
            if not match_date_simple:
                 logger.warning(f"Skipping odds for fixture {fixture_id} due to missing/invalid date in {json_file_path}")
                 skipped_date_count += 1
                 continue

            # Get odds from DB using the imported function and db_manager
            odds_list = get_odds_from_db(fixture_id, match_date_simple, bookmaker_name)
            if not odds_list:
                logger.debug(f"Could not retrieve/find '{bookmaker_name}' odds for fixture {fixture_id} on {match_date_simple}. Skipping odds analysis for this file.")
                continue # Skip if no odds found for this bookmaker

            # Find matched bets using the imported function
            # Pass the logger instance to the imported function if it uses it globally
            # (Assuming odd_finder functions use the logger configured there, let's pass ours if needed)
            # Note: The odd_finder functions seem to configure their own logger or use db_manager's.
            # We will rely on that unless issues arise.
            matched_bets_for_file = find_matched_bets(processed_data, odds_list)

            update_required = False
            if matched_bets_for_file:
                # Sort by new weighted score, then probability
                matched_bets_for_file.sort(key=lambda x: (x.get('score', Decimal(0)), x.get('predicted_prob', Decimal(0))), reverse=True)
                total_matched_bets_count += len(matched_bets_for_file)

                # Convert all Decimals (and format floats) for JSON compatibility
                serializable_matches = convert_decimals_to_strings(matched_bets_for_file)

                # Add/Update the sorted, serializable list in the loaded data
                processed_data["matched_odds_info"] = serializable_matches
                update_required = True
                logger.info(f"Found {len(matched_bets_for_file)} matched bets meeting criteria for fixture {fixture_id} in: {relative_path}")

            else:
                # If no bets met threshold, remove old key if it exists
                if "matched_odds_info" in processed_data:
                    logger.debug(f"Removing previous 'matched_odds_info' as no bets met threshold for fixture {fixture_id} in {relative_path}")
                    del processed_data["matched_odds_info"]
                    update_required = True
                else:
                    logger.debug(f"No bets met >0.61 threshold for fixture {fixture_id} in {relative_path}. No update needed.")

            # Write the updated data back ONLY if changes were made
            if update_required:
                try:
                    with open(json_file_path, 'w') as f:
                        json.dump(processed_data, f, indent=4) # Use indent=4 like odd_finder
                    if matched_bets_for_file:
                        files_updated_count += 1
                    # logger.debug(f"Successfully updated file: {relative_path}") # Logged above or implicitly by removal log
                except Exception as write_error:
                    logger.error(f"Error writing updated data back to {relative_path}: {write_error}")
                    error_count += 1

        except Exception as e:
             logger.error(f"Unhandled error processing file for odds {relative_path} (Fixture: {fixture_id}): {e}")
             traceback.print_exc()
             error_count += 1

    # --- Odds Phase Summary ---
    logger.info("\n" + "="*50)
    logger.info("--- Odds Matching Summary ---")
    logger.info(f"Files scanned for odds processing: {processed_odds_count}")
    logger.info(f"Files skipped due to missing/invalid date: {skipped_date_count}")
    logger.info(f"Files successfully updated with matched odds info: {files_updated_count}")
    logger.info(f"Total matched bets found (>0.61 Prob) across all files: {total_matched_bets_count}")
    logger.info(f"Explicit error count during odds processing: {error_count}")
    logger.info("="*50)

    return error_count == 0 # Return True if successful (no errors)


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Processes daily game data and adds matched odds information.")
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR_DEFAULT, help='Directory containing raw daily game JSON files.')
    parser.add_argument('--output_dir', type=str, default=PROCESSED_DIR_DEFAULT, help='Directory to save processed match JSON files.')
    parser.add_argument('--bookmaker', type=str, default=DEFAULT_BOOKMAKER_NAME, help='Name of the bookmaker for odds matching.')
    parser.add_argument('--skip_processing', action='store_true', help='Skip the initial game processing phase.')
    parser.add_argument('--skip_odds', action='store_true', help='Skip the odds matching and update phase.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Optionally set debug for other loggers if needed
        logging.getLogger('process_daily_games').setLevel(logging.DEBUG) # Logger used by the class
        # Set debug for db_manager logger if it exists and you want its output
        try:
             logging.getLogger('get_data.api_football.db_mongo').setLevel(logging.DEBUG)
        except Exception:
             pass # Ignore if logger doesn't exist
        logger.info("--- Debug logging enabled ---")

    logger.info("Starting Combined Match Processing and Odds Analysis")
    logger.info(f"Input Directory: {args.input_dir}")
    logger.info(f"Output/Processed Directory: {args.output_dir}")

    processing_successful = True
    if not args.skip_processing:
        # --- Step 1: Process Daily Games ---
        logger.info("\n--- Starting Initial Match Data Processing Phase ---")
        try:
            processor = EnhancedSoccerMatchProcessor(
                input_dir=args.input_dir,
                output_dir=args.output_dir
            )
            # The process_all_matches_advanced method handles its own logging
            processor.process_all_matches_advanced()
            logger.info("--- Initial Match Data Processing Phase Completed ---")
        except Exception as e:
            logger.error(f"Error during initial match processing phase: {e}")
            traceback.print_exc()
            processing_successful = False # Mark failure
    else:
        logger.info("Skipping initial game processing phase as requested.")


    odds_successful = True
    if not args.skip_odds and processing_successful:
        # --- Step 2: Find and Add Matched Odds ---
        # Ensure DB connection is attempted (db_manager usually connects on first use)
        logger.info("Connecting to MongoDB via db_manager for odds lookup...")
        # Add a simple check if db_manager provides one, otherwise assume connection on demand.
        # if hasattr(db_manager, 'is_connected') and not db_manager.is_connected():
        #     logger.warning("db_manager reported not connected before odds phase.")

        odds_successful = add_matched_odds_info(args.output_dir, args.bookmaker)
        logger.info("--- Odds Matching and Update Phase Completed ---")

    elif not args.skip_odds and not processing_successful:
        logger.warning("Skipping odds matching phase due to errors in the initial processing phase.")
        odds_successful = False
    else:
        logger.info("Skipping odds matching phase as requested.")


    # --- Step 3: Cleanup ---
    try:
        db_manager.close_connection()
        logger.info("\nMongoDB connection closed via db_manager.")
    except Exception as e:
         logger.error(f"Error closing MongoDB connection: {e}")

    logger.info("\nCombined Processing Script Finished.")
    if not processing_successful or not odds_successful:
        logger.warning("Script finished with errors.")
        sys.exit(1)
    else:
        logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()