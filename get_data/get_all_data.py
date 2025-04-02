import asyncio
import logging
from datetime import datetime, timezone
import os
import sys
from typing import Dict, Any, Optional
import sqlite3

# Add parent directory to Python path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# API Football imports
from api_football.data_fetcher import fetch_all_data
from api_football.db_mongo import db_manager

# Statarea imports
from statarea.statarea_async_scraper import run_scraper_async, initialize_database

logger = logging.getLogger(__name__)

# Database directory setup
DATABASE_DIR = os.path.join(parent_dir, "database")
STATAREA_DB_PATH = os.path.join(DATABASE_DIR, "statarea_stats.db")
LOGS_DIR = os.path.join(DATABASE_DIR, "logs")

def setup_database_directories():
    """Create database and logs directories if they don't exist"""
    try:
        # Create main database directory
        if not os.path.exists(DATABASE_DIR):
            os.makedirs(DATABASE_DIR)
            logger.info(f"Created database directory: {DATABASE_DIR}")
            
        # Create logs directory
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)
            logger.info(f"Created logs directory: {LOGS_DIR}")
            
        return True
    except Exception as e:
        logger.error(f"Failed to create database directories: {e}")
        return False

def get_log_path(filename):
    """Get full path for a log file in the logs directory"""
    return os.path.join(LOGS_DIR, filename)

def check_statarea_needs_update(team: str, country: str, max_age_days: int = 1) -> bool:
    """Check if Statarea team data needs updating"""
    try:
        with sqlite3.connect(STATAREA_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT last_scraped FROM teams 
                WHERE name = ? AND country = ?
            ''', (team, country))
            
            result = cursor.fetchone()
            if not result:
                return True  # Team not in database
                
            last_scraped = datetime.fromisoformat(result[0])
            age = datetime.now() - last_scraped
            return age.days >= max_age_days
            
    except Exception as e:
        logger.error(f"Error checking Statarea team {team}: {e}")
        return True  # On error, assume we need to scrape

def check_api_football_needs_update(date_str: str) -> bool:
    """Check if API Football data needs updating for given date"""
    try:
        # Get summary of data for the date from MongoDB
        summary = db_manager.get_day_summary(date_str)
        
        if not summary:
            return True  # No data exists
            
        # Check if we have all types of data
        if (summary.get('has_games', False) and 
            summary.get('has_matches', False) and 
            summary.get('has_odds', False)):
            
            logger.info(f"API Football data already exists for {date_str}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error checking API Football data for {date_str}: {e}")
        return True  # On error, assume we need to fetch

async def get_data(target_date: Optional[datetime] = None, force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Fetch and store all data from API-Football and Statarea.
    Implements skipping logic for both data sources.
    
    Args:
        target_date: The date to fetch data for. If None, uses today's date.
        force_reprocess: If True, reprocess all data even if it exists.
        
    Returns:
        dict: Results containing success status and data processing summary.
    """
    # Setup database directories first
    if not setup_database_directories():
        return {
            "success": False,
            "error": "Failed to set up database directories"
        }
    
    if target_date is None:
        target_date = datetime.now(timezone.utc)
    
    date_str = target_date.strftime("%Y-%m-%d")
    logger.info(f"🗓️ Starting data collection for: {date_str}")
    
    results = {
        "success": True,
        "date": date_str,
        "api_football": {"success": False, "message": "Not run"},
        "statarea": {"success": False, "message": "Not run"}
    }
    
    # Step 1: Check and fetch API-Football data
    try:
        logger.info("--- Checking API-Football data ---")
        
        if force_reprocess or check_api_football_needs_update(date_str):
            logger.info("Fetching API-Football data...")
            api_results = await fetch_all_data(target_date, force_reprocess)
            
            results["api_football"] = {
                "success": api_results.get("success", False),
                "steps": api_results.get("steps", {}),
                "message": "Data fetched and processed"
            }
            
            if not api_results.get("success", False):
                logger.warning("⚠️ API-Football data collection completed with errors")
            else:
                logger.info("✅ API-Football data collection completed successfully")
        else:
            results["api_football"] = {
                "success": True,
                "message": "Skipped - Data already exists"
            }
            logger.info("✅ API-Football data skipped - already exists")
            
    except Exception as e:
        logger.error(f"❌ Critical error in API-Football data collection: {str(e)}", exc_info=True)
        results["api_football"] = {"success": False, "message": f"Critical error: {str(e)}"}
        results["success"] = False
    
    # Step 2: Initialize and check Statarea data
    try:
        logger.info("--- Initializing Statarea database ---")
        # Pass database path to initialize_database
        if not initialize_database(db_path=STATAREA_DB_PATH):
            logger.error("❌ Failed to initialize Statarea database")
            results["statarea"] = {"success": False, "message": "Database initialization failed"}
            results["success"] = False
        else:
            logger.info("✅ Statarea database initialized")
            
            # Step 3: Run Statarea scraper with skip logic
            logger.info("--- Running Statarea scraper ---")
            await run_scraper_async(
                periods=[5, 10, 15],
                force_update=force_reprocess,
                check_exists_func=check_statarea_needs_update,
                db_path=STATAREA_DB_PATH,
                logs_dir=LOGS_DIR
            )
            
            results["statarea"] = {"success": True, "message": "Processing completed"}
            logger.info("✅ Statarea processing completed")
            
    except Exception as e:
        logger.error(f"❌ Error in Statarea processing: {str(e)}", exc_info=True)
        results["statarea"] = {"success": False, "message": f"Error: {str(e)}"}
        results["success"] = False
    
    # Final summary
    logger.info("--- Data Collection Summary ---")
    logger.info(f"Date: {results['date']}")
    logger.info(f"Overall Success: {results['success']}")
    logger.info(f"API-Football: {results['api_football']['message']}")
    logger.info(f"Statarea: {results['statarea']['message']}")
    
    return results

if __name__ == "__main__":
    # Set up logging to both file and console
    log_file = get_log_path(f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create logs directory if it doesn't exist yet
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Running data collection...")
    
    # Run with force_reprocess=False to enable skipping
    asyncio.run(get_data(force_reprocess=False))