import os
import logging
import asyncio
# import json # No longer needed for file reading
from datetime import datetime
# from pathlib import Path # No longer needed for file paths
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fix the module import issue - Keep this if necessary
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import the necessary modules
from get_data.api_football.db_mongo import db_manager # Use db_manager singleton directly
from get_data.api_football.api_manager import api_manager

# Remove the import fixing block for odds_fetcher.py
# odds_fetcher_path = ...
# if os.path.exists(odds_fetcher_path): ...

# Import the refactored data_fetcher module
from get_data.api_football.data_fetcher import fetch_all_data

# Remove the MongoDBImporter class as its logic is now within fetch_all_data and components
# class MongoDBImporter:
#     ...
#     async def process_date(self, ...): ...

# Make main async
async def main():
    load_dotenv()
    try:
        # Initialize API Manager (already done inside fetch_all_data, but can be done here too if needed outside)
        # api_manager.initialize()

        logger.info("🚀 Starting data fetch and processing pipeline...")

        # Call the main async data fetching function using await
        # Specify a date if needed: target_date=datetime(2023, 5, 1)
        results = await fetch_all_data(force_reprocess=False)

        # Log the summary from fetch_all_data results
        if results.get("success", False):
            logger.info("✅ Pipeline completed successfully!")
            logger.info("📊 Summary:")
            for step, step_result in results.get("steps", {}).items():
                status = "Success" if step_result.get("success") else "Failed"
                message = step_result.get("message", "")
                logger.info(f"  - {step}: {status} ({message})")
        else:
            logger.error(f"❌ Pipeline failed on date {results.get('date', 'N/A')}")
            logger.info("📊 Summary:")
            for step, step_result in results.get("steps", {}).items():
                status = "Success" if step_result.get("success") else "Failed"
                message = step_result.get("message", "")
                logger.info(f"  - {step}: {status} ({message})")
            if "error" in results:
                 logger.error(f"  Error detail: {results['error']}")

    except Exception as e:
        logger.error(f"❌ Error in main execution: {str(e)}", exc_info=True)

    finally:
        # Ensure MongoDB connection is closed 
        # Use the close_connection method from the singleton instance
        try:
            db_manager.close_connection()
            logger.info("🔌 Database connection closed.")
        except Exception as e:
             logger.warning(f"Could not close DB connection: {e}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())