import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)

# Add project root if needed
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, project_root)

# Assuming api_manager and elo_fetcher are correctly set up and importable
try:
    from .api_manager import api_manager
    from .elo_fetcher import elo_fetcher # Import elo_fetcher instance
except ImportError:
     logger.critical("Failed to import api_manager or elo_fetcher. Ensure script is run as part of the package or paths are correct.")
     raise

class RateLimiter:
    def __init__(self, calls_per_minute: int = 28): # Slightly conservative rate
        assert calls_per_minute > 0, "Calls per minute must be positive"
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call_time = 0.0

    async def wait(self):
        """Wait appropriate time to maintain rate limit."""
        current_time = time.monotonic() # Use monotonic clock for interval measurement
        time_since_last = current_time - self.last_call_time
        wait_needed = self.interval - time_since_last
        if wait_needed > 0:
            await asyncio.sleep(wait_needed)
        self.last_call_time = time.monotonic() # Update time after waiting/call

class MatchProcessor:
    def __init__(self, api_base_url: str = "https://api-football-v1.p.rapidapi.com/v3"):
        """Initialize MatchProcessor for fetching API data."""
        assert api_base_url, "API base URL is required"
        self.api_base_url = api_base_url
        self.rate_limiter = RateLimiter(calls_per_minute=28) # Use the limiter
        logger.info(f"Initialized MatchProcessor for API: {api_base_url}")

    async def _make_api_request(self, endpoint: str, params: Dict, retry_count: int = 3) -> Optional[Dict[str, Any]]:
        """Make API request with rate limit handling and retries. Returns raw response dict or None on failure."""
        assert endpoint, "Endpoint is required"
        assert isinstance(params, dict), "Params must be a dict"
        assert retry_count >= 0, "Retry count cannot be negative"

        await self.rate_limiter.wait()
        url = f"{self.api_base_url}/{endpoint}"

        for attempt in range(retry_count + 1):
            session = None # Ensure session is defined for finally block
            response = None # Ensure response defined
            api_key = "N/A" # Default for logging if key fetch fails
            try:
                api_key, headers = api_manager.get_active_api_key() # Get fresh key/headers
                assert api_key and headers, "Failed to get active API key"

                session = aiohttp.ClientSession(headers=headers)
                async with session.get(url, params=params, timeout=20) as response: # Add timeout
                    if response.status == 429:
                        logger.warning(f"Rate limit hit (429) on attempt {attempt + 1}/{retry_count + 1} for {endpoint}. Key: ...{api_key[-5:]}. Rotating key.")
                        api_manager.handle_rate_limit(api_key)
                        if attempt < retry_count:
                             await asyncio.sleep(1.5) # Small delay before retry with new key
                             continue # Retry loop will get new key
                        else:
                             logger.error(f"Rate limit hit on final attempt for {endpoint}.")
                             return None # Failed after retries

                    response.raise_for_status() # Raises HTTPError for bad responses (4xx, 5xx) other than 429
                    response_data = await response.json()

                    assert isinstance(response_data, dict), f"API response is not a dictionary for {endpoint}"

                    errors_in_response = response_data.get("errors")
                    if errors_in_response and (isinstance(errors_in_response, list) and len(errors_in_response) > 0) or (isinstance(errors_in_response, dict) and len(errors_in_response) > 0) :
                         logger.warning(f"API returned errors for {endpoint} params {params}: {errors_in_response}")
                         # Treat API error field as failure for this data point
                         return None

                    assert "response" in response_data, f"API response missing 'response' key for {endpoint}"
                    return response_data # Return the full response dict

            except aiohttp.ClientResponseError as http_err:
                logger.error(f"HTTP error on attempt {attempt + 1}/{retry_count + 1} for {endpoint} params {params}: {http_err.status} {http_err.message}")
                if response and response.status == 404: # Specific handling for 404 Not Found
                     logger.warning(f"Endpoint {endpoint} with params {params} returned 404. Data likely unavailable.")
                     return None # Treat 404 as non-retryable failure for this data point
                # Other retryable Client errors (e.g., 5xx)
                if attempt >= retry_count: logger.error(f"HTTP error on final attempt for {endpoint}."); return None

            except aiohttp.ClientError as client_err: # Other client errors (timeout, connection issues)
                logger.error(f"Client error on attempt {attempt + 1}/{retry_count + 1} for {endpoint} params {params}: {client_err}")
                if attempt >= retry_count: logger.error(f"Client error on final attempt for {endpoint}."); return None

            except asyncio.TimeoutError:
                 logger.error(f"Request timed out on attempt {attempt + 1}/{retry_count + 1} for {endpoint} params {params}")
                 if attempt >= retry_count: logger.error(f"Timeout on final attempt for {endpoint}."); return None

            except AssertionError as ae:
                 logger.error(f"Assertion failed during API request processing for {endpoint} params {params}: {ae}")
                 return None # Assertion failure means unexpected state

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{retry_count + 1} for {endpoint} params {params}: {e}", exc_info=True)
                if attempt >= retry_count: logger.error(f"Unexpected error on final attempt for {endpoint}."); return None
            finally:
                 if session and not session.closed:
                      try:
                          await session.close()
                      except Exception as close_err:
                          logger.warning(f"Error closing aiohttp session: {close_err}")

            # Exponential backoff before retry (if not the last attempt)
            if attempt < retry_count:
                import random
                wait_time = (1.5 ** attempt) + (random.random() * 0.5)
                await asyncio.sleep(wait_time)

        logger.error(f"All {retry_count + 1} retry attempts failed for {endpoint} with params {params}.")
        return None

    async def fetch_predictions(self, fixture_id: int) -> Optional[List[Dict[str, Any]]]:
        """Fetch predictions API response for a specific fixture."""
        assert isinstance(fixture_id, int) and fixture_id > 0, "Fixture ID must be a positive integer"
        response_data = await self._make_api_request('predictions', {'fixture': str(fixture_id)})
        # Predictions response is a list
        return response_data.get("response") if response_data else None

    async def fetch_team_statistics(self, team_id: int, league_id: int, season: int) -> Optional[Dict[str, Any]]:
        """Fetch team statistics API response."""
        assert isinstance(team_id, int) and team_id > 0, "Team ID must be positive integer"
        assert isinstance(league_id, int) and league_id > 0, "League ID must be positive integer"
        assert isinstance(season, int) and season > 1990, "Season must be a valid year"
        response_data = await self._make_api_request('teams/statistics', {
            'team': str(team_id),
            'league': str(league_id),
            'season': str(season)
        })
        # Team stats response is a dict
        return response_data.get("response") if response_data else None

    async def fetch_standings(self, league_id: int, season: int) -> Optional[List[Dict[str, Any]]]:
        """Fetch standings API response for a league/season."""
        assert isinstance(league_id, int) and league_id > 0, "League ID must be positive integer"
        assert isinstance(season, int) and season > 1990, "Season must be a valid year"
        response_data = await self._make_api_request('standings', {
            'league': str(league_id),
            'season': str(season)
        })
        # Standings response is a list containing one dict usually
        return response_data.get("response") if response_data else None

    # Updated fetch_api_data_for_match signature and logic
    async def fetch_api_data_for_match(
        self,
        fixture_id: int,
        league_id: int,
        season: int,
        home_team_id: int,
        away_team_id: int,
        home_team_name: str, # Needed for ELO
        away_team_name: str, # Needed for ELO
        match_date: datetime   # Needed for ELO
    ) -> Dict[str, Optional[Any]]:
        """
        Fetches all required API data points (predictions, stats, standings) AND ELO
        for a single match.
        """
        assert isinstance(fixture_id, int) and fixture_id > 0
        assert isinstance(league_id, int) and league_id > 0
        assert isinstance(season, int) and season > 1990
        assert isinstance(home_team_id, int) and home_team_id > 0
        assert isinstance(away_team_id, int) and away_team_id > 0
        assert isinstance(home_team_name, str) and home_team_name
        assert isinstance(away_team_name, str) and away_team_name
        assert isinstance(match_date, datetime)

        logger.info(f"Fetching API data & ELO for Fixture: {fixture_id}, League: {league_id}, Season: {season}")

        # --- Fetch API Data Concurrently ---
        tasks = {
            "predictions": self.fetch_predictions(fixture_id),
            "home_stats": self.fetch_team_statistics(home_team_id, league_id, season),
            "away_stats": self.fetch_team_statistics(away_team_id, league_id, season),
            "standings": self.fetch_standings(league_id, season)
        }
        api_results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Combine API results
        api_data = {}
        keys = list(tasks.keys())
        for i, result in enumerate(api_results_list):
             key = keys[i]
             if isinstance(result, Exception):
                 logger.error(f"Error fetching '{key}' for fixture {fixture_id}: {result}", exc_info=False)
                 api_data[key] = None
             else:
                 api_data[key] = result

        # Log warnings for failed API calls
        if api_data.get("predictions") is None: logger.warning(f"API call for predictions failed or returned no data for fixture {fixture_id}")
        if api_data.get("home_stats") is None: logger.warning(f"API call for home stats failed or returned no data for fixture {fixture_id}, team {home_team_id}")
        if api_data.get("away_stats") is None: logger.warning(f"API call for away stats failed or returned no data for fixture {fixture_id}, team {away_team_id}")
        if api_data.get("standings") is None: logger.warning(f"API call for standings failed or returned no data for fixture {fixture_id}, league {league_id}")

        # --- Fetch ELO Data (Synchronous call in executor) ---
        home_elo, away_elo = None, None
        try:
            loop = asyncio.get_event_loop()
            logger.debug(f"Fetching ELO for fixture {fixture_id} ({home_team_name} vs {away_team_name} on {match_date.date()})")
            home_elo, away_elo = await loop.run_in_executor(
                None, # Use default thread pool executor
                elo_fetcher.get_elos_for_match, # The synchronous function to call
                home_team_name,                 # Argument 1 for the function
                away_team_name,                 # Argument 2
                match_date                      # Argument 3
            )
            logger.info(f"Fetched ELO for fixture {fixture_id}: Home={home_elo}, Away={away_elo}")
        except Exception as elo_err:
            logger.error(f"Error fetching ELO for fixture {fixture_id}: {elo_err}", exc_info=True)
            # Keep ELO as None if fetching fails

        # --- Combine all results ---
        final_results = {
            **api_data, # Unpack API data results
            "HomeTeamELO": home_elo,
            "AwayTeamELO": away_elo,
            "processed_at_utc": datetime.now(timezone.utc) # Add timestamp
        }

        return final_results
