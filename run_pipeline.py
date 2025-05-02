# run_pipeline.py
from decimal import Decimal
import json
import logging
import asyncio
import math
import os
import sys
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO,
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
handlers=[logging.StreamHandler(sys.stdout)]) # Ensure logs go to console
logger = logging.getLogger(__name__)

# --- Add project root AND relevant subdirectories to sys.path for imports ---
project_root = Path(__file__).resolve().parent
get_data_dir = project_root / "get_data"
score_data_dir = project_root / "score_data" # Define score_data path

# Insert project root first
sys.path.insert(0, str(project_root))

# Insert get_data directory
if get_data_dir.is_dir():
    sys.path.insert(1, str(get_data_dir))
    logger.debug(f"Added {get_data_dir} to sys.path")
else:
    # If get_data isn't found, log a critical error as imports will fail
    logger.critical(f"FATAL: Could not find 'get_data' directory at: {get_data_dir}. Imports will fail.")
    sys.exit(1) # Exit early if essential directory is missing

# Insert score_data directory
if score_data_dir.is_dir():
    sys.path.insert(1, str(score_data_dir)) # Insert after root, before get_data might be safer
    logger.debug(f"Added {score_data_dir} to sys.path")
else:
    # This is critical as predict_games, odd_finder, etc., are here
    logger.critical(f"FATAL: Could not find 'score_data' directory at: {score_data_dir}. Imports will fail.")
    sys.exit(1)

# --- Import necessary execution functions/classes ---
try:
    # 1. Data Fetcher
    from get_data.data_fetcher import fetch_workflow_data

    # 2. Extract Daily Games
    #    We'll need to call its main logic more directly
    from score_data.extract_daily_games import DailyGameExtractor

    # 3. Predict Games
    #    Refactor predict_games.py to have a callable function if needed,
    #    or import its main processing logic. For now, let's assume we can import and run its core loop.
    #    (This might require slight modification of predict_games.py)
    from score_data.predict_games import process_fixture_json as predict_process_fixture
    from score_data.predict_games import UNIFIED_DATA_DIR as PREDICT_INPUT_DIR
    from score_data.predict_games import OUTPUT_DIR as PREDICT_OUTPUT_DIR
    import glob # For finding files for predict_games

    # 4. Odd Finder
    #    Refactor odd_finder.py if needed, or import its main logic.
    #    (This might require slight modification of odd_finder.py)
    from score_data.odd_finder import load_batch_prediction_data as odd_load_data
    from score_data.odd_finder import get_odds_from_db as odd_get_odds
    from score_data.odd_finder import process_combined_selections as odd_process_selections
    from score_data.odd_finder import convert_decimals_to_strings as odd_convert_decimals
    from score_data.odd_finder import get_match_date_simple as odd_get_date
    from score_data.odd_finder import BOOKMAKER_NAME as ODD_BOOKMAKER # Use the constant
    from score_data.odd_finder import INPUT_OUTPUT_FILE as ODD_INPUT_FILE # Use the constant
    from get_data.api_football.db_mongo import db_manager as odd_db_manager # For closing connection

    # 5. Paper Generator
    #    Refactor paper_generator.py if needed, or import its main logic.
    #    (This might require slight modification of paper_generator.py)
    from score_data.paper_generator import load_data as paper_load_data
    from score_data.paper_generator import filter_game_selections as paper_filter_selections
    from score_data.paper_generator import build_papers as paper_build
    from score_data.paper_generator import optimize_paper_stakes as paper_optimize
    from score_data.paper_generator import convert_for_json as paper_convert_json
    from score_data.paper_generator import find_fixture_id as paper_find_fixture_id
    # Import constants/defaults from paper_generator needed for the flow
    from score_data.paper_generator import DEFAULT_EDGE_THRESHOLD, DEFAULT_MIN_PROBABILITY, DEFAULT_MIN_ODDS, DEFAULT_MAX_ODDS
    from score_data.paper_generator import DEFAULT_PAPER_SIZES, DEFAULT_MAX_PAPERS_PER_SIZE, DEFAULT_PAPER_BUILD_STRATEGY
    from score_data.paper_generator import DEFAULT_KELLY_FRACTION, DEFAULT_RISK_AVERSION
    from score_data.paper_generator import DEFAULT_OUTPUT_FILE as PAPER_OUTPUT_FILE
    from collections import defaultdict

except ImportError as e:
    logger.critical(f"Failed to import necessary modules: {e}. Ensure all scripts are in the correct paths and sys.path is set correctly.")
    sys.exit(1)


async def run_full_workflow():
    """Orchestrates the execution of the entire AgenticFC pipeline."""
    logger.info("🚀 Starting AgenticFC Full Workflow Pipeline...")
    start_time = datetime.now()

    # === Step 1: Data Fetcher ===
    logger.info("--- Running Step 1: Data Fetcher ---")
    try:
        # Use today's date, or allow passing a specific date if needed
        target_date = datetime.now(timezone.utc)
        # Set force_reprocess based on needs, e.g., False for standard runs
        fetch_result = await fetch_workflow_data(target_date=target_date, force_reprocess=False)
        if not fetch_result.get("success"):
            logger.error("Data Fetcher step failed. Check data_fetcher logs.")
            # Decide if pipeline should stop or continue
            # return # Example: Stop pipeline on fetch failure
        logger.info("✅ Step 1: Data Fetcher Finished.")
    except Exception as e:
        logger.error(f"❌ Critical Error in Step 1 (Data Fetcher): {e}", exc_info=True)
        return # Stop pipeline

    # === Step 2: Extract Daily Games ===
    logger.info("--- Running Step 2: Extract Daily Games ---")
    try:
        # extractor = DailyGameExtractor(use_mongo=True) # Assumes MongoDB connection is handled internally
        extractor = DailyGameExtractor() # Use default use_mongo=True
        today_date_str = extractor.get_current_date_str()
        extraction_summary = extractor.extract_games_for_date(today_date_str) # Uses today's date by default
        if extraction_summary.get("error"):
             logger.error(f"Extraction failed: {extraction_summary['error']}")
             # return # Decide whether to stop
        processed_count = extraction_summary.get('total_games_processed', 0)
        found_count = extraction_summary.get('total_games_found_for_date', 0)
        logger.info(f"Extraction attempted: {found_count}, Successfully processed/saved: {processed_count}")
        logger.info("✅ Step 2: Extract Daily Games Finished.")
    except Exception as e:
        logger.error(f"❌ Critical Error in Step 2 (Extract Daily Games): {e}", exc_info=True)
        return # Stop pipeline


    # === Step 3: Predict Games ===
    logger.info("--- Running Step 3: Predict Games ---")
    predict_input_dir = os.path.join(project_root, PREDICT_INPUT_DIR)
    predict_output_dir = os.path.join(project_root, PREDICT_OUTPUT_DIR)
    all_prediction_results = []
    try:
        os.makedirs(predict_output_dir, exist_ok=True) # Ensure output dir exists
        # Find JSON files generated by Step 2
        unified_json_files = glob.glob(os.path.join(predict_input_dir, "*.json"))
        if not unified_json_files:
             logger.warning(f"No JSON files found in {predict_input_dir} for prediction step.")
        else:
             logger.info(f"Found {len(unified_json_files)} unified JSON files to predict.")
             for json_file in unified_json_files:
                 try:
                     fixture_results = predict_process_fixture(json_file)
                     if fixture_results:
                         all_prediction_results.append(fixture_results)
                 except Exception as e:
                     logger.error(f"Error processing fixture file {os.path.basename(json_file)} in prediction step: {e}", exc_info=True)

        # Save the batch prediction results (mimicking predict_games.py __main__)
        if all_prediction_results:
             output_filename = os.path.join(predict_output_dir, "batch_prediction_results.json")
             try:
                 # Use NpEncoder if needed, or a generic converter
                 class NpEncoder(json.JSONEncoder):
                     def default(self, obj):
                         if isinstance(obj, (np.integer, np.int_)): return int(obj)
                         if isinstance(obj, (np.floating, np.float_)): return float(obj)
                         if isinstance(obj, np.ndarray): return obj.tolist()
                         # Handle Decimal if predict_games uses it internally
                         if isinstance(obj, Decimal): return str(obj)
                         if isinstance(obj, (datetime, date)): return obj.isoformat()
                         if math.isnan(obj): return None # Handle NaN
                         return super(NpEncoder, self).default(obj)

                 with open(output_filename, 'w') as f:
                      json.dump(all_prediction_results, f, indent=4, cls=NpEncoder)
                 logger.info(f"Saved batch prediction results to {output_filename}")
             except Exception as e:
                 logger.error(f"Failed to save batch prediction results: {e}", exc_info=True)

        logger.info("✅ Step 3: Predict Games Finished.")
    except Exception as e:
        logger.error(f"❌ Critical Error in Step 3 (Predict Games): {e}", exc_info=True)
        return # Stop pipeline


    # === Step 4: Odd Finder ===
    logger.info("--- Running Step 4: Odd Finder ---")
    # Define input/output file relative to project root
    odd_finder_io_file = os.path.join(project_root, ODD_INPUT_FILE)
    odd_updates_applied = 0
    try:
        if not os.path.exists(odd_finder_io_file):
            logger.error(f"Odd Finder input file not found: {odd_finder_io_file}. Skipping Step 4.")
        else:
            # Load the batch prediction data generated by Step 3
            odd_batch_data, odd_data_format = odd_load_data(odd_finder_io_file)
            if odd_batch_data is None:
                logger.error("Failed to load batch prediction data for Odd Finder. Skipping Step 4.")
            else:
                processed_matches = 0
                error_count = 0
                if odd_data_format == "dict": updated_data = {} # Initialize if dict format

                match_iterator = None
                if odd_data_format == "list": match_iterator = enumerate(odd_batch_data)
                elif odd_data_format == "dict": match_iterator = odd_batch_data.items()

                if match_iterator:
                    for key_or_index, odd_match_data in match_iterator:
                        processed_matches += 1
                        # Extract fixture ID robustly
                        fixture_id = odd_match_data.get('fixture_id')
                        display_id = f"fixture {fixture_id}" if fixture_id else f"entry {key_or_index+1 if isinstance(key_or_index, int) else key_or_index}"

                        if not fixture_id:
                            logger.warning(f"Skipping {display_id} in Odd Finder: Could not determine fixture ID.")
                            if odd_data_format == "dict": updated_data[key_or_index] = odd_match_data
                            error_count += 1
                            continue

                        logger.debug(f"Odd Finder processing {display_id}")
                        match_date_simple = odd_get_date(odd_match_data)
                        if not match_date_simple:
                            logger.warning(f"Skipping {display_id} in Odd Finder: Could not determine date.")
                            if odd_data_format == "dict": updated_data[key_or_index] = odd_match_data
                            error_count += 1
                            continue

                        try:
                            odds_list = odd_get_odds(str(fixture_id), match_date_simple, ODD_BOOKMAKER)
                            selections_before = json.dumps(odd_match_data.get("top_n_combined_selections", []), default=str)
                            # odd_process_selections should modify odd_match_data in place
                            processed_odd_match_data = odd_process_selections(odd_match_data, odds_list)
                            selections_after = json.dumps(processed_odd_match_data.get("top_n_combined_selections", []), default=str)

                            if selections_before != selections_after:
                                 logger.info(f"Odd Finder updates applied to {display_id}.")
                                 odd_updates_applied += 1

                            if odd_data_format == "dict":
                                updated_data[key_or_index] = processed_odd_match_data

                        except Exception as e:
                            logger.error(f"Error during Odd Finder processing for {display_id}: {e}", exc_info=True)
                            if odd_data_format == "dict": updated_data[key_or_index] = odd_match_data
                            error_count += 1
                            continue

                # Determine final data to write
                final_odd_data_to_write = odd_batch_data if odd_data_format == "list" else updated_data

                # Serialize and Write Back
                try:
                    serializable_odd_data = odd_convert_decimals(final_odd_data_to_write)
                    with open(odd_finder_io_file, 'w') as f:
                        json.dump(serializable_odd_data, f, indent=4, ensure_ascii=False)
                    logger.info(f"Odd Finder successfully updated data written back to: {odd_finder_io_file}")
                except Exception as write_error:
                    logger.error(f"Error writing updated Odd Finder data back: {write_error}")

                # Close DB connection if Odd Finder uses it
                try:
                    odd_db_manager.close_connection()
                    logger.info("Odd Finder: MongoDB connection closed.")
                except Exception as e:
                    logger.error(f"Error closing Odd Finder MongoDB connection: {e}")


        logger.info("✅ Step 4: Odd Finder Finished.")
    except Exception as e:
        logger.error(f"❌ Critical Error in Step 4 (Odd Finder): {e}", exc_info=True)
        return # Stop pipeline


    # === Step 5: Paper Generator ===
    logger.info("--- Running Step 5: Paper Generator ---")
    paper_input_file = odd_finder_io_file # Input is the file updated by Odd Finder
    # --- RECALCULATE output path robustly ---
    # PAPER_OUTPUT_FILE constant is '../data/output/optimized_game_portfolios.json'
    # We want project_root / 'data' / 'output' / 'optimized_game_portfolios.json'
    paper_output_dir = project_root / "data" / "output"
    paper_output_filename = "optimized_game_portfolios.json" # Extract filename part
    paper_output_file_absolute = paper_output_dir / paper_output_filename
    # --- End recalculation ---

    try:
        if not os.path.exists(paper_input_file):
            logger.error(f"Paper Generator input file not found: {paper_input_file}. Skipping Step 5.")
        else:
            # Load data
            paper_batch_data = paper_load_data(paper_input_file)
            if paper_batch_data is None:
                logger.error("Failed to load data for Paper Generator. Skipping Step 5.")
            else:
                # Filter Selections per Game
                all_filtered_selections_by_game = defaultdict(list)
                processed_fixture_ids = set()
                for match_data in paper_batch_data:
                     fixture_id = paper_find_fixture_id(match_data)
                     if not fixture_id: continue
                     if fixture_id in processed_fixture_ids: continue
                     processed_fixture_ids.add(fixture_id)

                     game_selections = paper_filter_selections(
                         match_data, fixture_id,
                         DEFAULT_EDGE_THRESHOLD, DEFAULT_MIN_PROBABILITY, # Use defaults or load from config/args
                         DEFAULT_MIN_ODDS, DEFAULT_MAX_ODDS
                     )
                     if game_selections:
                          all_filtered_selections_by_game[fixture_id] = game_selections

                # Build Papers
                if not all_filtered_selections_by_game or len(all_filtered_selections_by_game) < min(DEFAULT_PAPER_SIZES):
                     logger.warning("Paper Generator: Not enough games with valid selections for minimum paper size. Skipping paper generation.")
                     papers_to_optimize = []
                else:
                     papers_to_optimize = paper_build(
                         all_filtered_selections_by_game,
                         paper_sizes=DEFAULT_PAPER_SIZES,
                         max_papers_per_size=DEFAULT_MAX_PAPERS_PER_SIZE,
                         strategy=DEFAULT_PAPER_BUILD_STRATEGY
                     )

                # Optimize Stakes
                optimized_papers = []
                if papers_to_optimize:
                    for i, paper in enumerate(papers_to_optimize):
                        paper_id = f"Paper_{i+1}"
                        optimized_result = paper_optimize(
                            paper_selections=paper,
                            kelly_fraction=DEFAULT_KELLY_FRACTION, # Use defaults or load
                            risk_aversion=DEFAULT_RISK_AVERSION, # Use defaults or load
                            paper_id=paper_id
                        )
                        if optimized_result:
                            optimized_papers.append(optimized_result)

                # Rank Papers (simple sort example)
                if optimized_papers:
                     # Example sort: by Sharpe Ratio descending
                     optimized_papers.sort(key=lambda p: p.get('paper_summary', {}).get('sharpe_ratio', Decimal('-Infinity')), reverse=True)
                     logger.info(f"Paper Generator ranked {len(optimized_papers)} papers.")


                # Prepare Final Output
                paper_output_data = {
                     "generation_info": {
                         "generated_at": datetime.utcnow().isoformat() + "Z",
                         "input_file": os.path.relpath(paper_input_file, project_root), # Use relative path for info
                         "output_file": os.path.relpath(paper_output_file_absolute, project_root) # Use relative path for info
                     },
                     "optimized_papers": optimized_papers
                }

                # Write Output (using the robust absolute path)
                serializable_paper_output = paper_convert_json(paper_output_data)
                try:
                     # Ensure the output directory exists just before writing
                     os.makedirs(paper_output_dir, exist_ok=True)
                     with open(paper_output_file_absolute, 'w', encoding='utf-8') as f: # Use absolute path
                         json.dump(serializable_paper_output, f, indent=4, ensure_ascii=False)
                     logger.info(f"Paper Generator successfully wrote optimized papers to: {paper_output_file_absolute}") # Log absolute path
                except Exception as write_error:
                     logger.error(f"Error writing Paper Generator output file to {paper_output_file_absolute}: {write_error}", exc_info=True) # Log absolute path


        logger.info("✅ Step 5: Paper Generator Finished.")
    except Exception as e:
        logger.error(f"❌ Critical Error in Step 5 (Paper Generator): {e}", exc_info=True)
        # No return here, let it finish

    # --- Pipeline Complete ---
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"🏁 AgenticFC Full Workflow Pipeline Finished.")
    logger.info(f"Total execution time: {duration}")


if __name__ == '__main__':
    # Ensure necessary directories exist before running
    required_dirs = [
        os.path.join(project_root, "data", "unified_data"),
        os.path.join(project_root, "data", "output"),
        os.path.join(project_root, "data", "output", "plots") # For predict_games plots
    ]
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)

    # Run the async workflow
    try:
        asyncio.run(run_full_workflow())
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user.")
    except Exception as main_e:
        logger.critical(f"Unhandled exception during pipeline execution: {main_e}", exc_info=True)
        sys.exit(1)
