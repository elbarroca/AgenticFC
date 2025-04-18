import os
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from get_data.api_football.endpoints.api_manager import api_manager
from get_data.api_football.db_mongo import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

    def _make_api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limit handling using api_manager."""
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"Making API request to {url} with params {params}")
        
        try:
            # Get active API key and headers from API manager
            active_key, headers = api_manager.get_active_api_key()
            if not active_key:
                 logger.error("No active API key available from manager.")
                 return None

            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 429:
                logger.warning(f"Rate limit hit (429) for key {active_key}. api_manager should handle rotation if configured.")
                return None

            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out for {url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error for {url}: {e.response.status_code} - {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request Exception for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error making API request to {url}: {e}", exc_info=True)
            return None

    def get_and_save_team_fixtures(self, team_id: int, season: int) -> Optional[List[int]]:
        """
        Fetch fixtures for a specific team and season, save the list to DB,
        and return the list of fixture IDs.

        Args:
            team_id: Team ID to fetch fixtures for
            season: Season year

        Returns:
            Optional[List[int]]: List of fixture IDs or None if fetch/save failed.
        """
        logger.info(f"Fetching and saving fixtures for team ID: {team_id}, Season: {season}")

        params = {"team": str(team_id), "season": str(season)}
        response_data = self._make_api_request("fixtures", params)

        if response_data and response_data.get("response"):
            fixture_ids = [item.get("fixture", {}).get("id") for item in response_data["response"]]
            fixture_ids = [fid for fid in fixture_ids if fid]

            if fixture_ids:
                 logger.info(f"Found {len(fixture_ids)} fixtures for Team {team_id}, Season {season}.")
                 success = db_manager.save_team_season_fixture_list(team_id, season, fixture_ids)
                 if not success:
                      logger.error(f"Failed to save fixture list to DB for Team {team_id}, Season {season}.")
                 return fixture_ids
            else:
                 logger.info(f"No fixtures found in response for Team {team_id}, Season {season}.")
                 db_manager.save_team_season_fixture_list(team_id, season, [])
                 return []
        else:
            logger.error(f"Failed to fetch or parse fixtures response for Team {team_id}, Season {season}.")
            return None

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Ensure DB is initialized (or handle potential init errors)
    try:
        if not db_manager._initialized:
            logger.info("Initializing DB Manager from team_fixtures main...")
            db_manager.__init__()
    except Exception as e:
        logger.error(f"Failed to initialize DB Manager: {e}", exc_info=True)
        return

    if not db_manager._initialized:
         logger.error("DB Manager failed to initialize. Exiting.")
         return

    fetcher = TeamFixturesFetcher()

    # --- Example Usage ---
    test_team_id = 33 # Manchester United
    test_season = 2023

    logger.info(f"\n--- Testing fetch and save for Team {test_team_id}, Season {test_season} ---")
    fixture_list = fetcher.get_and_save_team_fixtures(team_id=test_team_id, season=test_season)

    if fixture_list is not None:
        logger.info(f"Successfully fetched and attempted save for Team {test_team_id}, Season {test_season}. Fixture count: {len(fixture_list)}")
    else:
        logger.error(f"Failed to fetch/save fixtures for Team {test_team_id}, Season {test_season}.")

    # --- Clean up DB connection ---
    db_manager.close_connection()
    logger.info("DB connection closed.")

if __name__ == "__main__":
    main()