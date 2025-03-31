import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import time
import logging
from src.betting.utils.api_manager import api_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameScraper:
    """Class to scrape football games from the API."""
    
    def __init__(self, base_dir: str = "data"):
        """Initialize the scraper with API configuration."""
        self.base_dir = Path(base_dir)
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
        
        # Get API key from environment variable
        api_key = os.getenv("API_FOOTBALL_KEY")
        if not api_key:
            logger.warning("API_FOOTBALL_KEY environment variable not set, using API manager")
            
        # All available leagues configuration
        self.all_leagues = {
           
            # Top 5 Leagues
             "39": {"name": "Premier League", "tier": 1, "country": "England"},          # England's top division
             "135": {"name": "Serie A", "tier": 1, "country": "Italy"},                 # Italy's top division
            "78": {"name": "Bundesliga", "tier": 1, "country": "Germany"},             # Germany's top division
             "61": {"name": "Ligue 1", "tier": 1, "country": "France"},                 # France's top division
             "140": {"name": "La Liga", "tier": 1, "country": "Spain"},                 # Spain's top division
            
            # Secondary Leagues
             "40": {"name": "Championship", "tier": 2, "country": "England"},           # England's second division
            #"136": {"name": "Serie B", "tier": 2, "country": "Italy"},                # Italy's second division
             "79": {"name": "2. Bundesliga", "tier": 2, "country": "Germany"},         # Germany's second division
             # "62": {"name": "Ligue 2", "tier": 2, "country": "France"},                # France's second division
             "141": {"name": "Segunda División", "tier": 2, "country": "Spain"},        # Spain's second division
            
            # Other Major European Leagues
             "88": {"name": "Eredivisie", "tier": 1, "country": "Netherlands"},        # Netherlands' top division
             "95": {"name": "Segunda Liga", "tier": 2, "country": "Portugal"},         # Portugal's second division
            "203": {"name": "Super Lig", "tier": 1, "country": "Turkey"},            # Turkey's top division
             #"179": {"name": "Premiership", "tier": 1, "country": "Scotland"},        # Scotland's top division

             "144": {"name": "Jupiler Pro League", "tier": 1, "country": "Belgium"},  # Belgium's top division
             "89": {"name": "Eredivisie 2", "tier": 2, "country": "Netherlands"},     # Netherlands' second division
             "94": {"name": "Primeira Liga", "tier": 1, "country": "Portugal"},       # Portugal's top division
             "106": {"name": "Ekstraklasa", "tier": 1, "country": "Poland"},         # Poland's Ekstraklasa league
            #"210": {"name": "HNL", "tier": 1, "country": "Croatia"},                # Croatia's HNL league
            
            # Nordic Leagues
            # "113": {"name": "Allsvenskan", "tier": 1, "country": "Sweden"},         # Sweden's top division
            # "103": {"name": "Eliteserien", "tier": 1, "country": "Norway"},         # Norway's top division
             "119": {"name": "Superliga", "tier": 1, "country": "Denmark"},          # Denmark's top division
            
            # Eastern European Leagues
             "283": {"name": "Liga 1", "tier": 1, "country": "Romania"},             # Romania's top division
            # "392": {"name": "First League", "tier": 1, "country": "Montenegro"},    # Montenegro's top division
            # "364": {"name": "A Lyga", "tier": 1, "country": "Lithuania"},           # Lithuania's top division

            #"289": {"name": "1st Division", "tier": 1, "country": "South-Africa"}, # South-Africa's top division
            
            # European Competitions
             "2": {"name": "UEFA Champions League", "tier": 1, "country": "Europe"},         # Europe's premier club competition
             "3": {"name": "UEFA Europa League", "tier": 1, "country": "Europe"},            # Europe's secondary club competition
             "848": {"name": "UEFA Europa Conference League", "tier": 1, "country": "Europe"} # Europe's tertiary club competition
            #"96": {"name": "Taca de Portugal", "tier": 1, "country": "Portugal"},      # Portugal's cup competition
        }
        
        # Use all_leagues as the meaningful leagues (active leagues)
        self.meaningful_leagues = self.all_leagues.copy()
        
        # List of strings that indicate non-main team matches
        self.excluded_terms = [
            "(w)", "women", "reserves", "u21", "u23", "u20", "u19", "u18", 
            "youth", "academy", "junior", "development"
        ]
        
        # Setup directory
        self._setup_directory_structure()
        logger.info("GameScraper initialized successfully")

    def _setup_directory_structure(self):
        """Create base directory structure if it doesn't exist."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory structure at {self.base_dir}")
        except Exception as e:
            print(f"Error creating directory structure: {str(e)}")
            raise

    def _get_date_path(self, date: datetime) -> Path:
        """Get path for specific date."""
        path = self.base_dir / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limit handling."""
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"Making API request to {url}")
        if params:
            logger.debug(f"Params: {params}")
        
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

    def get_active_leagues_for_date(self, date: datetime) -> Dict:
        """Get the active leagues for a specific date."""
        # Simply return all meaningful leagues since we're not using a schedule
        return self.meaningful_leagues

    def _is_valid_match(self, match: Dict) -> bool:
        """Check if a match should be included based on team names, competition, and league."""
        # Get team names and league info
        home_team = match.get("teams", {}).get("home", {}).get("name", "").lower()
        away_team = match.get("teams", {}).get("away", {}).get("name", "").lower()
        league = match.get("league", {})
        league_id = str(league.get("id", ""))
        
        # Get active leagues for the match date
        match_date = datetime.strptime(match["fixture"]["date"].split("T")[0], '%Y-%m-%d')
        active_leagues = self.get_active_leagues_for_date(match_date)
        
        # Check if league is in active leagues
        if league_id not in active_leagues:
            return False
        
        # Check for excluded terms in team names
        for term in self.excluded_terms:
            if term in home_team or term in away_team:
                return False
        
        return True

    def get_games(self, date: datetime) -> Dict:
        """Get games for a specific date."""
        try:
            api_date = date.strftime('%Y-%m-%d')
            print(f"\nFetching matches for {api_date}")
            
            # Initialize organized data structure
            organized_data = {
                "date": api_date,
                "total_matches": 0,
                "leagues": {}
            }
            
            # Fetch all matches for the date
            response = self._make_api_request(
                'fixtures',
                {
                    "date": api_date,
                    "season": 2024  # Use current active season
                }
            )
            
            if not response or "response" not in response:
                print("No matches found or API error")
                return organized_data
            
            matches = response["response"]
            print(f"Found {len(matches)} total matches")
            
            # Process matches
            filtered_count = 0
            for match in matches:
                if not self._is_valid_match(match):
                    continue
                
                league = match["league"]
                league_id = str(league["id"])
                league_info = self.get_active_leagues_for_date(date)[league_id]
                league_name = f"{league_info['name']} ({league_info['country']})"
                
                # Initialize league data if not exists
                if league_id not in organized_data["leagues"]:
                    organized_data["leagues"][league_id] = {
                        "name": league_name,
                        "country": league_info["country"],
                        "tier": league_info["tier"],
                        "matches": []
                    }
                
                # Create match data
                match_data = {
                    "id": str(match["fixture"]["id"]),
                    "time": match["fixture"]["date"],
                    "home_team": {
                        "id": str(match["teams"]["home"]["id"]),
                        "name": match["teams"]["home"]["name"],
                        "longName": match["teams"]["home"]["name"],
                        "score": match["goals"]["home"] if match["goals"]["home"] is not None else 0
                    },
                    "away_team": {
                        "id": str(match["teams"]["away"]["id"]),
                        "name": match["teams"]["away"]["name"],
                        "longName": match["teams"]["away"]["name"],
                        "score": match["goals"]["away"] if match["goals"]["away"] is not None else 0
                    },
                    "status": {
                        "started": match["fixture"]["status"]["short"] in ["1H", "2H", "HT", "ET", "P", "BT", "INT"],
                        "finished": match["fixture"]["status"]["short"] in ["FT", "AET", "PEN"],
                        "score": f"{match['goals']['home'] if match['goals']['home'] is not None else 0} - {match['goals']['away'] if match['goals']['away'] is not None else 0}",
                        "time": match["fixture"]["status"]["short"]
                    }
                }
                
                organized_data["leagues"][league_id]["matches"].append(match_data)
                organized_data["total_matches"] += 1
                filtered_count += 1
                
                # Print match info
                status = "Not Started"
                if match_data["status"]["started"]:
                    status = f"Live ({match_data['status']['time']})"
                if match_data["status"]["finished"]:
                    status = "Finished"
                
                print(f"  ✓ [{league_name}] {match_data['home_team']['name']} vs {match_data['away_team']['name']}")
                print(f"    Time: {match_data['time']}")
                print(f"    Status: {status}")
                if match_data["status"]["started"] or match_data["status"]["finished"]:
                    print(f"    Score: {match_data['status']['score']}")
            
            # Print summary
            if organized_data["total_matches"] > 0:
                print(f"\nFound {filtered_count} meaningful matches in {len(organized_data['leagues'])} leagues")
                print("\nLeagues with matches:")
                for league_id, league_data in organized_data["leagues"].items():
                    print(f"\n{league_data['name']} (Tier {league_data['tier']}) - {len(league_data['matches'])} matches")
            else:
                print("\nNo meaningful matches found for this date")
            
            return organized_data
            
        except Exception as e:
            print(f"\nError fetching games: {str(e)}")
            return {
                "date": date.strftime('%Y-%m-%d'),
                "total_matches": 0,
                "leagues": {}
            }

    def save_games(self, games_data: Dict):
        """Save games data to a JSON file."""
        if not games_data.get("total_matches", 0):
            print("No games to save")
            return

        try:
            date = datetime.strptime(games_data["date"], "%Y-%m-%d")
            date_path = self._get_date_path(date)
            games_file = date_path / "games.json"
            
            print(f"\nSaving matches data to {games_file}")
            with open(games_file, "w", encoding="utf-8") as f:
                json.dump(games_data, f, indent=2, ensure_ascii=False)
            print("Data saved successfully")
            
        except Exception as e:
            print(f"Error saving games data: {str(e)}")

def main():
    """Main function to scrape games for tomorrow."""
    try:
        logger.info("Starting game scraper...")
        scraper = GameScraper()
        
        # Log API usage before starting
        logger.info("Current API Usage Stats:")
        for key, stats in api_manager.get_usage_stats().items():
            logger.info(f"API Key {key}:")
            logger.info(f"  - Requests Made: {stats['requests_made']}")
            logger.info(f"  - Requests Remaining: {stats['requests_remaining']}")
            logger.info(f"  - Time until reset: {stats['time_until_reset']}")
        
        tomorrow = datetime.now() + timedelta(days=1)
        logger.info(f"\nProcessing games for tomorrow: {tomorrow.strftime('%Y-%m-%d')}")
        
        games_data = scraper.get_games(tomorrow)
        if games_data["total_matches"] > 0:
            scraper.save_games(games_data)
        
        # Log final API usage
        logger.info("\nFinal API Usage Stats:")
        for key, stats in api_manager.get_usage_stats().items():
            logger.info(f"API Key {key}:")
            logger.info(f"  - Requests Made: {stats['requests_made']}")
            logger.info(f"  - Requests Remaining: {stats['requests_remaining']}")
        
        logger.info("\nGame scraping completed successfully!")
        
    except Exception as e:
        logger.error(f"\nError in main function: {str(e)}")

if __name__ == "__main__":
    main() 