import os
import requests
from datetime import datetime
from typing import Dict,Any, Optional
import logging
from get_data.api_football.endpoints.api_manager import api_manager
from get_data.api_football.db_mongo import db_manager, MongoDBManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixtureDetailsFetcher:
    """Class to fetch detailed information for specific fixtures from the API Football."""
    
    def __init__(self, db_manager_instance: MongoDBManager):
        """Initialize the fetcher with API configuration and DB manager."""
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
        
        # Get API key from environment variable
        api_key = os.getenv("API_FOOTBALL_KEY")
        if not api_key:
            logger.warning("API_FOOTBALL_KEY environment variable not set, using API manager")
            
        self.api_manager = api_manager
        self.db_manager = db_manager_instance
        
        logger.info("FixtureDetailsFetcher initialized successfully")

    def _make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limit handling."""
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"Making API request to {url}")
        
        try:
            # Get active API key and headers from API manager
            _, headers = self.api_manager.get_active_api_key()
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 429:  # Rate limit exceeded
                logger.warning("Rate limit exceeded, rotating API key...")
                self.api_manager.handle_rate_limit(headers["x-rapidapi-key"])
                # Retry with new key
                _, new_headers = self.api_manager.get_active_api_key()
                response = requests.get(url, headers=new_headers, params=params)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {str(e)}")
            return {"status": "failed", "message": str(e)}
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def get_fixture_details(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a specific fixture.
        
        Args:
            fixture_id (int): The ID of the fixture to fetch details for
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing fixture details or None if failed
        """
        try:
            # Prepare request parameters
            params = {"fixture": fixture_id}
            
            # Get basic fixture information
            logger.info(f"Fetching basic info for fixture {fixture_id}")
            basic_info = self._make_api_request("fixtures", params)
            
            # Get statistics
            logger.info(f"Fetching statistics for fixture {fixture_id}")
            statistics = self._make_api_request("fixtures/statistics", params)
            
            # Get events
            logger.info(f"Fetching events for fixture {fixture_id}")
            events = self._make_api_request("fixtures/events", params)
            
            # Get lineups
            logger.info(f"Fetching lineups for fixture {fixture_id}")
            lineups = self._make_api_request("fixtures/lineups", params)
            
            # Combine all data
            complete_data = {
                'basic_info': basic_info.get('response', []),
                'statistics': statistics.get('response', []),
                'events': events.get('response', []),
                'lineups': lineups.get('response', []),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Save to database using SQLite's save_fixture_detail method
            success = self.db_manager.save_match_data(complete_data)
            if success:
                logger.info(f"Successfully saved details for fixture {fixture_id}")
            else:
                logger.warning(f"Failed to save details for fixture {fixture_id}")
            
            return complete_data
            
        except Exception as e:
            logger.error(f"Error fetching fixture details for {fixture_id}: {str(e)}")
            return None

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    fetcher = FixtureDetailsFetcher()
    # Example usage
    # fetcher.get_fixture_details(fixture_id=710815)

if __name__ == "__main__":
    main()