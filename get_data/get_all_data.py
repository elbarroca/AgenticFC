import asyncio
import logging
from datetime import datetime, timezone
import os
import sys
from typing import Dict, Any, Optional
import sqlite3
from contextlib import contextmanager

# Add parent directory to Python path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# API Football imports
from api_football.data_fetcher import fetch_all_data
from api_football.db_mongo import db_manager

# Statarea imports
from statarea.statarea_async_scraper import run_scraper_async, initialize_database

# Import DailyGameExtractor for game data extraction
from extract_daily_games import DailyGameExtractor, OUTPUT_DIR

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
            
        # Create output directory for daily games
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            logger.info(f"Created daily games output directory: {OUTPUT_DIR}")
            
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

def extract_daily_games(date_str: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract and process daily game data using DailyGameExtractor.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        output_path: Optional path for the summary file
        
    Returns:
        Dict with extraction results
    """
    logger.info(f"--- Extracting daily games data for {date_str} ---")
    
    try:
        # Create a custom StatAreaDBManager class that uses the correct database path
        from get_data.extract_daily_games import StatAreaDBManager, DailyGameExtractor
        
        class CustomStatAreaDBManager(StatAreaDBManager):
            """Override the StatAreaDBManager to use the correct database path"""
            
            def __init__(self):
                """Initialize with the correct database path."""
                logger.info(f"Initializing CustomStatAreaDBManager with database path: {STATAREA_DB_PATH}")
                super().__init__(db_path=STATAREA_DB_PATH)
            
            @contextmanager
            def get_db_connection(self):
                """Get a connection to the StatArea SQLite database with the correct path."""
                conn = None
                try:
                    # Log the database path for debugging
                    logger.info(f"Connecting to StatArea database at: {STATAREA_DB_PATH}")
                    conn = sqlite3.connect(
                        STATAREA_DB_PATH,  # Use the path from get_all_data.py
                        timeout=30,  # SQLITE_TIMEOUT
                        check_same_thread=False
                    )
                    yield conn
                except sqlite3.Error as e:
                    logger.error(f"StatArea database connection error: {str(e)}")
                    raise
                finally:
                    if conn:
                        conn.close()
        
        # Custom DailyGameExtractor that uses our custom StatAreaDBManager
        class CustomDailyGameExtractor(DailyGameExtractor):
            """Override the DailyGameExtractor to use our custom StatAreaDBManager"""
            
            def __init__(self, use_mongo=True):
                """Initialize with the custom database manager."""
                self.statarea_db = CustomStatAreaDBManager()
                self.mongo_db = None
                if use_mongo:
                    try:
                        from get_data.api_football.db_mongo import MongoDBManager
                        self.mongo_db = MongoDBManager()
                        logger.info("MongoDB connection successful")
                    except Exception as e:
                        logger.error(f"Failed to connect to MongoDB: {e}")
                        logger.warning("Continuing with StatArea DB only")
        
        # Initialize our custom extractor
        logger.info(f"Using database path: {STATAREA_DB_PATH}")
        
        # Check if the database file exists
        if not os.path.exists(STATAREA_DB_PATH):
            logger.error(f"Database file not found at path: {STATAREA_DB_PATH}")
            return {
                "success": False,
                "error": f"StatArea database file not found at {STATAREA_DB_PATH}"
            }
        else:
            logger.info(f"Database file found with size: {os.path.getsize(STATAREA_DB_PATH)} bytes")
        
        extractor = CustomDailyGameExtractor(use_mongo=True)
        
        # Check the database connection and list some teams for debugging
        try:
            teams = extractor.statarea_db.list_all_teams()
            if teams:
                logger.info(f"Successfully connected to database and found {len(teams)} teams")
                sample_teams = teams[:5]
                for team in sample_teams:
                    logger.info(f"Sample team: {team['name']} (ID: {team['id']}, Country: {team['country']})")
            else:
                logger.warning("Connected to database but found no teams")
        except Exception as e:
            logger.error(f"Error listing teams from database: {e}")
        
        # Extract games data
        games_data = extractor.extract_games_for_date(date_str)
        
        # Generate output filename if not provided
        if not output_path:
            output_path = os.path.join(OUTPUT_DIR, f"games_summary_{date_str}.json")
            
        # Save summary file if games were found
        if games_data['total_games'] > 0:
            extractor.save_summary_file(games_data, output_path)
            logger.info(f"✅ Successfully extracted {games_data['total_games']} games for {date_str}")
            logger.info(f"✅ Games summary saved to {output_path}")
            logger.info(f"✅ Individual game files saved to the '{OUTPUT_DIR}' directory")
            
            result = {
                "success": True,
                "total_games": games_data['total_games'],
                "summary_file": output_path
            }
        else:
            logger.warning(f"⚠️ No games found for {date_str}")
            result = {
                "success": True,
                "total_games": 0,
                "message": "No games found for this date"
            }
            
    except Exception as e:
        logger.error(f"❌ Error extracting daily games: {str(e)}", exc_info=True)
        result = {
            "success": False, 
            "error": str(e)
        }
    finally:
        # Close MongoDB connection if it exists
        if 'extractor' in locals() and extractor.mongo_db:
            try:
                extractor.mongo_db.close_connection()
            except:
                pass  # Ignore errors during cleanup
                
    return result

async def get_data(target_date: Optional[datetime] = None, force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Fetch and store all data from API-Football and Statarea.
    Implements skipping logic for both data sources.
    Also extracts and processes daily game data after fetching raw data.
    
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
        "statarea": {"success": False, "message": "Not run"},
        "daily_games": {"success": False, "message": "Not run"}
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
            logger.info(f"--- Running Statarea scraper with DB path: {STATAREA_DB_PATH} ---")
            
            # Check if database has teams
            has_teams = False
            try:
                with sqlite3.connect(STATAREA_DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM teams")
                    team_count = cursor.fetchone()[0]
                    has_teams = team_count > 0
                    logger.info(f"Database has {team_count} teams")
            except Exception as e:
                logger.error(f"Error checking team count: {e}")
            
            # Always force update if database is empty
            should_force_update = force_reprocess or not has_teams
            team_limit = None if has_teams else 20  # Limit to 20 teams if DB is empty
            
            if not has_teams:
                logger.warning("⚠️ Database is empty, forcing update of 20 teams to populate it")
            
            await run_scraper_async(
                team_count=team_limit,
                periods=[5, 10, 15],
                force_update=should_force_update,
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
    
    # Step 4: Extract and process daily games data
    # Only proceed if both data sources were successful or skipped successfully
    if results["api_football"]["success"] and results["statarea"]["success"]:
        try:
            logger.info("--- Extracting and processing daily games ---")
            daily_games_result = extract_daily_games(date_str)
            
            results["daily_games"] = daily_games_result
            if daily_games_result["success"]:
                logger.info("✅ Daily games extraction completed successfully")
            else:
                logger.warning("⚠️ Daily games extraction completed with errors")
                # Don't mark overall process as failed, it's non-critical
        except Exception as e:
            logger.error(f"❌ Error in daily games extraction: {str(e)}", exc_info=True)
            results["daily_games"] = {"success": False, "message": f"Error: {str(e)}"}
            # Don't mark overall process as failed, it's non-critical
    else:
        results["daily_games"] = {"success": False, "message": "Skipped - Data sources had errors"}
        logger.warning("⚠️ Daily games extraction skipped due to data source errors")
    
    # Final summary
    logger.info("--- Data Collection Summary ---")
    logger.info(f"Date: {results['date']}")
    logger.info(f"Overall Success: {results['success']}")
    logger.info(f"API-Football: {results['api_football']['message']}")
    logger.info(f"Statarea: {results['statarea']['message']}")
    logger.info(f"Daily Games: {results['daily_games'].get('message', results['daily_games'])}")
    
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