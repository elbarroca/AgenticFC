import os
import sys
import requests
from datetime import datetime
from typing import Dict
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, project_root)

from get_data.api_football.endpoints.api_manager import api_manager
from get_data.api_football.db_mongo import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameScraper:
    """Class to scrape football games from the API and store them in the optimized MongoDB structure."""
    
    def __init__(self):
        """Initialize the scraper with API configuration."""
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
             "136": {"name": "Serie B", "tier": 2, "country": "Italy"},                # Italy's second division
             "79": {"name": "2. Bundesliga", "tier": 2, "country": "Germany"},         # Germany's second division
             "62": {"name": "Ligue 2", "tier": 2, "country": "France"},                # France's second division
             "141": {"name": "Segunda División", "tier": 2, "country": "Spain"},        # Spain's second division
            
            # Other Major European Leagues
             "88": {"name": "Eredivisie", "tier": 1, "country": "Netherlands"},        # Netherlands' top division
             "95": {"name": "Segunda Liga", "tier": 2, "country": "Portugal"},         # Portugal's second division
             "203": {"name": "Super Lig", "tier": 1, "country": "Turkey"},            # Turkey's top division
             "179": {"name": "Premiership", "tier": 1, "country": "Scotland"},        # Scotland's top division
             "144": {"name": "Jupiler Pro League", "tier": 1, "country": "Belgium"},  # Belgium's top division
             "89": {"name": "Eredivisie 2", "tier": 2, "country": "Netherlands"},     # Netherlands' second division
             "94": {"name": "Primeira Liga", "tier": 1, "country": "Portugal"},       # Portugal's top division
             "106": {"name": "Ekstraklasa", "tier": 1, "country": "Poland"},         # Poland's Ekstraklasa league
             "210": {"name": "HNL", "tier": 1, "country": "Croatia"},                # Croatia's HNL league
             "218": {"name": "Austria Bundesliga", "tier": 1, "country": "Austria"}, # Austria's top division
             "207": {"name": "Super League", "tier": 1, "country": "Switzerland"},   # Switzerland's top division
            
            # Nordic Leagues
             "113": {"name": "Allsvenskan", "tier": 1, "country": "Sweden"},         # Sweden's top division
             "103": {"name": "Eliteserien", "tier": 1, "country": "Norway"},         # Norway's top division
             "119": {"name": "Superliga", "tier": 1, "country": "Denmark"},          # Denmark's top division
            
            # Eastern European Leagues
             "283": {"name": "Liga 1", "tier": 1, "country": "Romania"},             # Romania's top division
             #"392": {"name": "First League", "tier": 1, "country": "Montenegro"},    # Montenegro's top division
             #"364": {"name": "A Lyga", "tier": 1, "country": "Lithuania"},           # Lithuania's top division
             "333": {"name": "Premier League", "tier": 1, "country": "Ukraine"},     # Ukraine's top division
             "345": {"name": "Czech Liga", "tier": 1, "country": "Czech Republic"},  # Czech Republic's top division
             "197": {"name": "Grecian Football League", "tier": 1, "country": "Greece"},# Greece's top division
             "286": {"name": "SuperLiga", "tier": 1, "country": "Serbia"},          # Serbia's top division
             "318": {"name": "1. Division", "tier": 1, "country": "Cyprus"},        # Cyprus' top division
             "271": {"name": "NB I", "tier": 1, "country": "Hungary"},             # Hungary's top division
                         
            # European Competitions
             "2": {"name": "UEFA Champions League", "tier": 1, "country": "Europe"},         # Europe's premier club competition
             "3": {"name": "UEFA Europa League", "tier": 1, "country": "Europe"},            # Europe's secondary club competition
             "848": {"name": "UEFA Europa Conference League", "tier": 1, "country": "Europe"} # Europe's tertiary club competition
        }
        
        # Use all_leagues as the meaningful leagues (active leagues)
        self.meaningful_leagues = self.all_leagues.copy()
        
        # List of strings that indicate non-main team matches
        self.excluded_terms = [
            "(w)", "women", "reserves", "u21", "u23", "u20", "u19", "u18", 
            "youth", "academy", "junior", "development"
        ]
        
        logger.info("GameScraper initialized successfully")

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

    def _fetch_standings(self, league_id: str, season: int, date_str: str) -> Dict:
        """Fetch standings for a specific league and season."""
        logger.info(f"Fetching standings for league {league_id}, season {season}")
        
        response = self._make_api_request(
            'standings',
            {
                "league": league_id,
                "season": season
            }
        )
        
        if not response or "response" not in response or not response["response"]:
            logger.warning(f"No standings found for league {league_id}, season {season}")
            return {}
        
        standings_data = response["response"][0]
        
        # Optimize standings data structure
        optimized_standings = {
            "league": {
                "id": str(standings_data.get("league", {}).get("id", "")),
                "name": standings_data.get("league", {}).get("name", ""),
                "country": standings_data.get("league", {}).get("country", ""),
                "season": season
            },
            "standings": []
        }
        
        # Extract only necessary standings data
        for league_standings in standings_data.get("league", {}).get("standings", []):
            standings_group = []
            for team in league_standings:
                optimized_team = {
                    "rank": team.get("rank", 0),
                    "team": {
                        "id": str(team.get("team", {}).get("id", "")),
                        "name": team.get("team", {}).get("name", "")
                    },
                    "points": team.get("points", 0),
                    "goalsDiff": team.get("goalsDiff", 0),
                    "form": team.get("form", ""),
                    "all": {
                        "played": team.get("all", {}).get("played", 0),
                        "win": team.get("all", {}).get("win", 0),
                        "draw": team.get("all", {}).get("draw", 0),
                        "lose": team.get("all", {}).get("lose", 0),
                        "goals": team.get("all", {}).get("goals", {})
                    }
                }
                standings_group.append(optimized_team)
            optimized_standings["standings"].append(standings_group)
        
        # Save standings data to MongoDB
        db_manager.save_standings_data(date_str, league_id, season, optimized_standings)
        
        return optimized_standings
    def get_games(self, date: datetime) -> Dict:
        """Get games for a specific date and save in the optimized structure."""
        try:
            api_date = date.strftime('%Y-%m-%d')
            logger.info(f"Fetching matches for {api_date}")
            
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
                logger.warning(f"No matches found or API error for {api_date}")
                return organized_data
            
            matches = response["response"]
            logger.info(f"Found {len(matches)} total matches")
            
            # Process matches
            filtered_count = 0
            standings_cache = {}  # Cache standings to avoid duplicate API calls
            
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
                
                # Create match data with minimal required info
                match_data = {
                    "id": str(match["fixture"]["id"]),
                    "time": match["fixture"]["date"],
                    "home_team": {
                        "id": str(match["teams"]["home"]["id"]),
                        "name": match["teams"]["home"]["name"]
                    },
                    "away_team": {
                        "id": str(match["teams"]["away"]["id"]),
                        "name": match["teams"]["away"]["name"]
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
                
                # Fetch standings for this league if not already in cache
                if league_id not in standings_cache:
                    standings_cache[league_id] = self._fetch_standings(league_id, 2024, api_date)
                
                # Prepare detailed match data for the matches collection
                fixture_id = str(match["fixture"]["id"])
                # Extract date in YYYY-MM-DD format from fixture date
                match_date_str = match["fixture"]["date"].split("T")[0] if match["fixture"]["date"] else api_date
                
                detailed_match = {
                    "_id": fixture_id,
                    "fixture_id": fixture_id,  # Add this for consistency
                    "date_str": match_date_str,  # Add the date_str field
                    "league_id": league_id,
                    "league_name": league_name,
                    "home_team": {
                        "id": str(match["teams"]["home"]["id"]),
                        "name": match["teams"]["home"]["name"],
                        "logo": match["teams"]["home"].get("logo", "")
                    },
                    "away_team": {
                        "id": str(match["teams"]["away"]["id"]),
                        "name": match["teams"]["away"]["name"],
                        "logo": match["teams"]["away"].get("logo", "")
                    },
                    "match_info": {
                        "id": fixture_id,
                        "referee": match["fixture"].get("referee", ""),
                        "venue": {
                            "name": match["fixture"].get("venue", {}).get("name", ""),
                            "city": match["fixture"].get("venue", {}).get("city", "")
                        },
                        "status": match["fixture"]["status"],
                        "date": match["fixture"]["date"],
                        "timestamp": match["fixture"].get("timestamp", 0)
                    },
                    "goals": match["goals"],
                    "standings": standings_cache.get(league_id, {})
                }
                
                # Save detailed match data
                db_manager.save_match_data(detailed_match)
            
            # Print summary
            if organized_data["total_matches"] > 0:
                logger.info(f"Found {filtered_count} meaningful matches in {len(organized_data['leagues'])} leagues")
                for league_id, league_data in organized_data["leagues"].items():
                    logger.info(f"{league_data['name']} (Tier {league_data['tier']}) - {len(league_data['matches'])} matches")
            else:
                logger.warning(f"No meaningful matches found for date {api_date}")
            
            # Save daily games summary to MongoDB
            if organized_data["total_matches"] > 0:
                logger.info(f"Saving {api_date} games data to MongoDB...")
                try:
                    success = db_manager.save_daily_games(api_date, organized_data)
                    if success:
                        logger.info(f"Daily games data for {api_date} saved successfully to MongoDB")
                    else:
                        logger.error(f"Failed to save daily games data for {api_date} to MongoDB")
                except Exception as db_e:
                    logger.error(f"Error saving games data to MongoDB: {db_e}")
            
            return organized_data
            
        except Exception as e:
            logger.error(f"Error in get_games: {str(e)}", exc_info=True)
            return {"date": date.strftime('%Y-%m-%d'), "total_matches": 0, "leagues": {}, "error": str(e)}

# Main function for testing
def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create scraper
    scraper = GameScraper()
    
    # Get games for today
    today = datetime.now()
    games = scraper.get_games(today)
    
    print(f"Total matches: {games['total_matches']}")
    
if __name__ == "__main__":
    main() 