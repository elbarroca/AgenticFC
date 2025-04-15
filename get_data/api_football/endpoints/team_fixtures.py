import os
import requests
from datetime import datetime
from typing import Dict, List, Any
import logging
from .api_manager import api_manager
from ..db_mongo import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeamFixturesFetcher:
    """Class to fetch all fixtures for a specific team from the API Football."""
    
    def __init__(self):
        """Initialize the fetcher with API configuration."""
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
        
        # Get API key from environment variable
        api_key = os.getenv("API_FOOTBALL_KEY")
        if not api_key:
            logger.warning("API_FOOTBALL_KEY environment variable not set, using API manager")
            
        logger.info("TeamFixturesFetcher initialized successfully")

    def _make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limit handling."""
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"Making API request to {url}")
        
        try:
            # Get active API key and headers from API manager
            _, headers = api_manager.get_active_api_key()
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 429:  # Rate limit exceeded
                logger.warning("Rate limit exceeded, rotating API key...")
                api_manager.handle_rate_limit(headers["x-rapidapi-key"])
                # Retry with new key
                _, new_headers = api_manager.get_active_api_key()
                response = requests.get(url, headers=new_headers, params=params)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {str(e)}")
            return {"status": "failed", "message": str(e)}
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def get_team_fixtures(self, team_id: int, season: int) -> Dict[str, Any]:
        """
        Fetch fixtures for a specific team and season.
        
        Args:
            team_id: Team ID to fetch fixtures for
            season: Season year
            
        Returns:
            Dict[str, Any]: API response containing fixtures data
        """
        logger.info(f"Fetching fixtures for team ID: {team_id}")
        
        # Prepare request parameters
        params = {
            "team": team_id,
            "season": season
        }
        
        # Make API request
        response = self._make_api_request("fixtures", params)
        return response

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    fetcher = TeamFixturesFetcher()
    # Example usage
    # fetcher.get_team_fixtures(team_id=33, season=2023)

if __name__ == "__main__":
    main()