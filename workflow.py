#wip

import subprocess
import sys
import logging
import os
from datetime import datetime

PYTHON_EXECUTABLE = sys.executable # Use the same python environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes workflow.py is in the project root

DATA_FETCHER_SCRIPT = os.path.join(BASE_DIR, "get_data", "data_fetcher.py")
EXTRACT_DAILY_GAMES_SCRIPT = os.path.join(BASE_DIR, "score_data", "extract_daily_games.py")
PREDICT_GAMES_SCRIPT = os.path.join(BASE_DIR, "score_data", "predict_games.py")
PAPER_GENERATOR_SCRIPT = os.path.join(BASE_DIR, "score_data", "paper_generator.py") # Added Paper Generator
# Future Odds Matcher Script:
ODDS_MATCHER_SCRIPT = os.path.join(BASE_DIR, "analyze_data", "match_odds.py") # Example path

LOG_FILE = os.path.join(BASE_DIR, "logs", "workflow.log")

# --- Logging Setup ---
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout) # Also print to console
    ]
)
logger = logging.getLogger("WorkflowManager")

def run_script(script_path: str, script_name: str) -> bool:
    """Runs a Python script using subprocess and logs the outcome."""
    logger.info(f"--- Starting: {script_name} ---")
    logger.info(f"Executing: {PYTHON_EXECUTABLE} {script_path}")

    # Ensure the script exists
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False

    try:

        script_dir = os.path.dirname(script_path)
        result = subprocess.run(
            [PYTHON_EXECUTABLE, os.path.basename(script_path)],
            capture_output=True,
            text=True,
            check=True, # Raise an exception if the script returns a non-zero exit code
            cwd=script_dir # Set working directory
        )
        logger.info(f"--- Finished: {script_name} ---")
        logger.debug(f"{script_name} STDOUT:\n{result.stdout}")
        if result.stderr:
             logger.warning(f"{script_name} STDERR:\n{result.stderr}") # Log stderr as warning
        return True
    except FileNotFoundError:
        logger.error(f"Error: Python executable not found at {PYTHON_EXECUTABLE}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"--- Failed: {script_name} ---")
        logger.error(f"Return Code: {e.returncode}")
        logger.error(f"STDOUT:\n{e.stdout}")
        logger.error(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while running {script_name}: {e}", exc_info=True)
        return False

def main_workflow():
    """Executes the main data processing and prediction workflow."""
    logger.info("=============================================")
    logger.info(f"Starting AgenticFC Workflow - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=============================================")

    success = True

    # Step 1: Fetch daily data
    logger.info("Step 1: Fetching Daily Data...")
    if not run_script(DATA_FETCHER_SCRIPT, "Data Fetcher"):
        success = False

    # Step 2: Extract and Unify Daily Games
    if success:
        logger.info("\nStep 2: Extracting & Unifying Daily Games...")
        if not run_script(EXTRACT_DAILY_GAMES_SCRIPT, "Extract Daily Games"):
            success = False

    # Step 3: Predict Games
    if success:
        logger.info("\nStep 3: Running Game Predictions...")
        if not run_script(PREDICT_GAMES_SCRIPT, "Predict Games"):
            success = False

    # Step 4: Generate Betting Papers
    if success:
        logger.info("\nStep 4: Generating Odds Matcher ...")
        if not run_script(ODDS_MATCHER_SCRIPT, "Odds Matcher"):
            success = False

    # Step 5: Run Game Predictions
    if success:
        logger.info("\nStep 5: Running Game Predictions...")
        if not run_script(PREDICT_GAMES_SCRIPT, "Predict Games"):
            success = False

    logger.info("\n=============================================")
    if success:
        logger.info("AgenticFC Workflow Completed Successfully")
    else:
        logger.error("AgenticFC Workflow Completed with ERRORS")
    logger.info("=============================================")

if __name__ == "__main__":
    main_workflow()