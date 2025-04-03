import asyncio
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

# Add the parent directory to sys.path to allow imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Add get_data directory to Python path
get_data_dir = os.path.join(parent_dir, "get_data")
sys.path.append(get_data_dir)

# Now we can import our modules
from get_data.get_all_data import get_data
from get_data.extract_daily_games import DailyGameExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def get_tomorrow_games():
    """
    Fetch and process data for tomorrow's games.
    """
    try:
        # Calculate tomorrow's date
        tomorrow = datetime.now() + timedelta(days=1)
        logger.info(f"Fetching data for: {tomorrow.strftime('%Y-%m-%d')}")

        # Get all required data
        data_result = await get_data(
            target_date=tomorrow,
            force_reprocess=True  # Force reprocess to ensure fresh data
        )

        if not data_result["success"]:
            logger.error("Failed to fetch required data")
            return

        # Create extractor instance
        extractor = DailyGameExtractor(use_mongo=True)

        # Extract games for tomorrow
        games_data = extractor.extract_games_for_date(tomorrow.strftime("%Y-%m-%d"))

        # Create output directory if it doesn't exist
        output_dir = Path("predictions")
        output_dir.mkdir(exist_ok=True)

        # Save the predictions
        output_file = output_dir / f"predictions_{tomorrow.strftime('%Y%m%d')}.json"
        extractor.save_summary_file(games_data, str(output_file))

        logger.info(f"✅ Successfully processed {games_data['total_games']} games for tomorrow")
        logger.info(f"📄 Predictions saved to: {output_file}")

    except Exception as e:
        logger.error(f"❌ Error processing tomorrow's games: {str(e)}", exc_info=True)
    finally:
        # Cleanup
        if 'extractor' in locals() and extractor.mongo_db:
            extractor.mongo_db.close_connection()

if __name__ == "__main__":
    logger.info("🚀 Starting tomorrow's games prediction process")
    asyncio.run(get_tomorrow_games())