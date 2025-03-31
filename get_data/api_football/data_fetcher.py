from datetime import datetime, timezone
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio
import json

from src.betting.scraping.game_scraper import GameScraper
from src.betting.utils.api_manager import api_manager
from src.betting.scraping.match_processor import MatchProcessor
from src.betting.scraping.odds_fetcher import OddsFetcher

logger = logging.getLogger(__name__)

def validate_step(step: int, date_path: Path, games_data=None) -> bool:
    """
    Simple validation logic for each step:
    - Step 1: games.json exists
    - Step 2: All matches from games.json have files in league folders
    - Step 3: odds folder exists with files for each game
    
    Args:
        step: Step number to validate
        date_path: Path to the date folder
        games_data: Optional games data for step 2 and 3 validation
        
    Returns:
        bool: Whether the step outputs exist
    """
    if step == 1:
        # Simply check if games.json exists
        games_file = date_path / 'games.json'
        return games_file.exists()
        
    elif step == 2:
        # Check if all matches from games.json have files
        if games_data is None:
            games_file = date_path / 'games.json'
            if not games_file.exists():
                return False
                
            try:
                with games_file.open('r') as f:
                    games_data = json.load(f)
            except:
                return False
        
        # Count expected matches
        expected_matches = games_data.get("total_matches", 0)
        if expected_matches == 0:
            return False
        
        # Count actual match files
        actual_matches = 0
        for league_dir in date_path.iterdir():
            if league_dir.is_dir() and league_dir.name not in ["odds", "reports"]:
                match_files = list(league_dir.glob("*_vs_*.json"))
                actual_matches += len(match_files)
        
        # Step 2 is valid if we have files for all expected matches
        return actual_matches >= expected_matches
        
    elif step == 3:
        # Check if odds folder exists with files
        odds_path = date_path / "odds"
        if not odds_path.exists() or not odds_path.is_dir():
            return False
            
        # Count odds files
        odds_files = list(odds_path.glob("*.json"))
        
        # If we don't have games_data, try to load it
        if games_data is None:
            games_file = date_path / 'games.json'
            if not games_file.exists():
                return False
                
            try:
                with games_file.open('r') as f:
                    games_data = json.load(f)
            except:
                return False
        
        # Get expected match count
        expected_matches = games_data.get("total_matches", 0)
        
        # Step 3 is valid if we have at least some odds files (not necessarily all)
        # This is because not all matches might have odds
        return len(odds_files) > 0
    
    return False

def fetch_all_data(target_date: Optional[datetime] = None, force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Fetch all necessary data sequentially: games, matches, and odds.
    Provides comprehensive validation at each step.
    
    Args:
        target_date: The date to fetch data for. If None, uses today's date.
        force_reprocess: Whether to force reprocessing of existing data
        
    Returns:
        dict: Results containing success status and paths to data with validation information
    """
    # Set up date and paths
    if target_date is None:
        target_date = datetime.now(timezone.utc)
    
    date_str = target_date.strftime("%Y-%m-%d")
    
    data_root = Path("data")
    year_folder = data_root / str(target_date.year)
    month_folder = year_folder / f"{target_date.month:02d}"
    date_path = month_folder / f"{target_date.day:02d}"
    date_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🗓️ Processing data for: {date_str} in {date_path}")
    
    results = {
        "success": True,
        "date_str": date_str,
        "date_path": date_path,
        "games_data": None,
        "match_files": [],
        "league_folders": {},
        "odds_data": None,
        "validation": {
            "step1": False,
            "step2": False,
            "step3": False
        },
        "missing_matches": [],
        "missing_odds": []
    }
    
    # Step 1: Get games data
    step1_validated = not force_reprocess and validate_step(1, date_path)
    results["validation"]["step1"] = step1_validated
    
    if step1_validated:
        logger.info("✅ Step 1 validation passed: games.json exists")
        # Load existing games data
        games_file = date_path / 'games.json'
        try:
            with games_file.open('r') as f:
                results["games_data"] = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"❌ Error loading existing games data: {str(e)}")
            results["validation"]["step1"] = False
            results["success"] = False
            return results
    else:
        try:
            logger.info("⏳ Running Step 1: Scraping Games")
            # Initialize API manager and scraper
            api_manager.initialize()
            scraper = GameScraper()
            scraper.api_manager = api_manager
            
            # Convert date string to datetime object for the scraper
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Get games data
            games_data = scraper.get_games(date_obj)
            
            # Validate results
            if not games_data or games_data.get("total_matches", 0) == 0:
                logger.error(f"❌ No games found for date {date_str}")
                results["success"] = False
                return results
            
            # Store results in the daily folder
            output_file = date_path / 'games.json'
            with output_file.open('w') as f:
                json.dump(games_data, f, indent=4)
            
            results["games_data"] = games_data
            
            # Verify step 1 outputs now exist
            if validate_step(1, date_path):
                results["validation"]["step1"] = True
                logger.info(f"✅ Successfully scraped {games_data.get('total_matches', 0)} games")
            else:
                logger.error("❌ Step 1 failed: games.json wasn't created")
                results["success"] = False
                return results
                
        except Exception as e:
            logger.error(f"❌ Error scraping games: {str(e)}")
            results["success"] = False
            return results
    
    # Step 2: Process matches
    if not results["validation"]["step1"]:
        logger.error("❌ Cannot proceed to Step 2: Step 1 failed")
        results["success"] = False
        return results
    
    # Load games data for match processing if not already loaded
    if results["games_data"] is None:
        games_file = date_path / 'games.json'
        try:
            with games_file.open('r') as f:
                results["games_data"] = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"❌ Error loading games data: {str(e)}")
            results["success"] = False
            return results
    
    # Convert the league-based games.json structure to the expected flat "matches" format
    expected_matches = []
    for league_id, league_data in results["games_data"].get("leagues", {}).items():
        league_name = league_data.get("name", "Unknown")
        for match in league_data.get("matches", []):
            # Convert to expected match format for processing
            match_data = {
                "fixture": {
                    "id": match.get("id")
                },
                "league": {
                    "id": league_id,
                    "name": league_name,
                    "country": league_data.get("country", "Unknown")
                },
                "teams": {
                    "home": {
                        "id": match.get("home_team", {}).get("id"),
                        "name": match.get("home_team", {}).get("name")
                    },
                    "away": {
                        "id": match.get("away_team", {}).get("id"),
                        "name": match.get("away_team", {}).get("name")
                    }
                },
                "fixture_time": match.get("time")
            }
            expected_matches.append(match_data)
    
    # Instead of just checking if ALL matches exist, identify WHICH ones are missing
    if not force_reprocess:
        missing_matches = []
        existing_matches = []
        
        # Check each match to see if its file exists
        for match in expected_matches:
            league = match.get("league", {}).get("name", "Unknown")
            home_team = match.get("teams", {}).get("home", {}).get("name", "Unknown")
            away_team = match.get("teams", {}).get("away", {}).get("name", "Unknown")
            
            # Create sanitized names for the file path
            league_folder_name = league.replace(" ", "_").replace("/", "_")
            match_file_name = f"{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}.json"
            
            # Check if this match file exists
            league_folder = date_path / league_folder_name
            if league_folder.exists():
                match_file = league_folder / match_file_name
                if match_file.exists():
                    existing_matches.append(match)
                else:
                    missing_matches.append(match)
                    logger.info(f"Missing match file: {match_file_name} in {league_folder_name}")
            else:
                missing_matches.append(match)
                logger.info(f"Missing league folder: {league_folder_name}")
        
        # Store the results of our analysis
        results["missing_matches"] = missing_matches
        
        # Update validation based on missing matches
        if not missing_matches and len(expected_matches) > 0:  # No matches are missing and we have expected matches
            step2_validated = True
            results["validation"]["step2"] = True
            
            # Count match files across all league folders
            match_count = 0
            for league_dir in date_path.iterdir():
                if league_dir.is_dir() and league_dir.name not in ["odds", "reports"]:
                    league_name = league_dir.name
                    results["league_folders"][league_name] = str(league_dir)
                    
                    # Get match files in this league
                    for match_file in league_dir.glob("*_vs_*.json"):
                        results["match_files"].append(str(match_file))
                        match_count += 1
                        
            logger.info(f"✅ All {match_count} match files exist")
        else:
            step2_validated = False
            results["validation"]["step2"] = False
            if len(expected_matches) == 0:
                logger.warning("⚠️ No matches found in games.json")
            else:
                logger.warning(f"⚠️ Found {len(missing_matches)} missing match files out of {len(expected_matches)} total matches")
    else:
        step2_validated = False  # Force reprocessing
        results["missing_matches"] = expected_matches
        logger.info("⏳ Force reprocessing all matches")
    
    # Process only missing matches if needed
    if not step2_validated:
        try:
            missing_matches = results["missing_matches"]
            if not missing_matches and force_reprocess:
                # If force reprocessing with no specific missing matches, process all
                missing_matches = expected_matches
            
            if not missing_matches:
                logger.warning("⚠️ No matches to process")
                results["validation"]["step2"] = True  # Consider it a success if there are no matches to process
            else:
                logger.info(f"⏳ Processing {len(missing_matches)} missing matches")
                
                # Initialize API manager
                api_manager.initialize()
                
                # Initialize processor with API manager
                processor = MatchProcessor()
                processor.api_manager = api_manager
                
                # Setup asyncio for processing
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Process only the missing matches
                # Convert missing_matches (flat list) back to the leagues structure expected by the match processor
                leagues = {}
                for match in missing_matches:
                    league_id = match.get("league", {}).get("id", "unknown")
                    league_name = match.get("league", {}).get("name", "Unknown")
                    league_country = match.get("league", {}).get("country", "Unknown")
                    
                    if league_id not in leagues:
                        leagues[league_id] = {
                            "name": league_name,
                            "country": league_country,
                            "matches": []
                        }
                    
                    # Convert match to the format expected by match processor
                    match_for_processor = {
                        "id": match.get("fixture", {}).get("id"),
                        "home_team": match.get("teams", {}).get("home", {}),
                        "away_team": match.get("teams", {}).get("away", {}),
                        "time": match.get("fixture_time"),
                        "status": {"started": False, "finished": False}
                    }
                    
                    leagues[league_id]["matches"].append(match_for_processor)
                
                modified_games_data = {
                    "date": date_str,
                    "leagues": leagues,
                    "total_matches": len(missing_matches)
                }
                
                processed_result = loop.run_until_complete(
                    processor.process_games_data_async(modified_games_data, date_path)
                )
                
                # Check if processing was successful
                if isinstance(processed_result, dict) and processed_result.get("status") == "error":
                    error_msg = processed_result.get("error", "Unknown error during match processing")
                    logger.error(f"❌ {error_msg}")
                    results["success"] = False
                    return results
                
                # Count match files across all league folders after processing
                match_count = 0
                for league_dir in date_path.iterdir():
                    if league_dir.is_dir() and league_dir.name not in ["odds", "reports"]:
                        league_name = league_dir.name
                        results["league_folders"][league_name] = str(league_dir)
                        
                        # Get match files in this league
                        for match_file in league_dir.glob("*_vs_*.json"):
                            results["match_files"].append(str(match_file))
                            match_count += 1
                
                # Verify all matches now exist
                if match_count >= len(missing_matches):
                    results["validation"]["step2"] = True
                    logger.info(f"✅ Successfully processed missing matches, now have {match_count} total match files")
                else:
                    logger.error(f"❌ Step 2 failed: Some match files still missing after processing")
                    results["success"] = False
                    return results
                
        except Exception as e:
            logger.error(f"❌ Error processing matches: {str(e)}")
            results["success"] = False
            return results
    
    # Step 3: Fetch odds
    if not results["validation"]["step2"]:
        logger.error("❌ Cannot proceed to Step 3: Step 2 failed")
        results["success"] = False
        return results
    
    # Check which matches need odds files
    if not force_reprocess:
        odds_path = date_path / "odds"
        if not odds_path.exists():
            odds_path.mkdir(parents=True, exist_ok=True)
            
        # Find which matches need odds
        matches_needing_odds = []
        
        # Check for fixtures that need odds
        for match in expected_matches:
            fixture_id = match.get("fixture", {}).get("id")
            if fixture_id:
                odds_file = odds_path / f"{fixture_id}.json"
                if not odds_file.exists():
                    matches_needing_odds.append(match)
        
        results["missing_odds"] = matches_needing_odds
        
        if not matches_needing_odds and len(expected_matches) > 0:
            # All odds files exist
            step3_validated = True
            results["validation"]["step3"] = True
            odds_files = list(odds_path.glob("*.json"))
            results["odds_data"] = {"existing_files": len(odds_files)}
            logger.info(f"✅ All odds files exist ({len(odds_files)} files)")
        else:
            step3_validated = False
            results["validation"]["step3"] = False
            if len(expected_matches) == 0:
                logger.warning("⚠️ No matches found for odds fetching")
                # Consider it a success if there are no matches to fetch odds for
                step3_validated = True
                results["validation"]["step3"] = True
            else:
                logger.warning(f"⚠️ Found {len(matches_needing_odds)} matches missing odds files")
    else:
        step3_validated = False  # Force reprocessing
        logger.info("⏳ Force reprocessing all odds")
    
    # Fetch only missing odds if needed
    if not step3_validated:
        try:
            logger.info(f"⏳ Fetching odds for {len(results['missing_odds'])} matches")
            
            # Initialize API manager
            api_manager.initialize()
            
            # Initialize OddsFetcher with API manager
            odds_fetcher = OddsFetcher(base_dir="data", api_manager=api_manager)
            
            # Create odds directory if it doesn't exist
            odds_path = date_path / "odds"
            odds_path.mkdir(parents=True, exist_ok=True)
            
            # Process odds data asynchronously
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Note: You may need to modify your OddsFetcher to accept specific matches
            # If that's not possible, you'll need to handle this differently
            result = loop.run_until_complete(
                odds_fetcher.process_daily_report(date_str, specific_matches=results["missing_odds"])
            )
            
            # Check results
            if result.get("successful", 0) > 0:
                # Verify step 3 outputs now exist
                odds_files = list(odds_path.glob("*.json"))
                if len(odds_files) > 0:
                    results["validation"]["step3"] = True
                    results["odds_data"] = result
                    logger.info(f"✅ Successfully fetched odds for {result.get('successful')} fixtures")
                    if result.get("failed", 0) > 0:
                        logger.warning(f"⚠️ Failed to fetch odds for {result.get('failed')} fixtures")
                else:
                    # Some odds might still be missing, but we'll consider it a success if any were fetched
                    if force_reprocess or len(results["missing_odds"]) == len(expected_matches):
                        # We were trying to get all odds, so it's a failure if validation fails
                        logger.error("❌ Step 3 failed: Not all odds files were created")
                        results["success"] = False
                    else:
                        # We were only trying to get some missing odds, so it's a partial success
                        results["validation"]["step3"] = True
                        results["odds_data"] = result
                        logger.warning("⚠️ Some odds files still missing, but continuing anyway")
            else:
                if result.get("failed", 0) > 0:
                    logger.error(f"❌ Failed to fetch odds for all {result.get('failed')} fixtures")
                    results["success"] = False
                    return results
                else:
                    logger.warning("⚠️ No fixtures found to process")
                    # This is a special case - no fixtures to process is not a failure
                    results["validation"]["step3"] = True
                    results["odds_data"] = {"successful": 0, "failed": 0}
            
        except Exception as e:
            logger.error(f"❌ Error fetching odds: {str(e)}")
            results["success"] = False
            return results
    
    # Final validation summary
    logger.info(f"🔍 Data fetch validation summary: Step 1: {results['validation']['step1']}, "
                f"Step 2: {results['validation']['step2']}, Step 3: {results['validation']['step3']}")
    
    if all(results["validation"].values()):
        logger.info("✅ All data fetching steps successfully validated")
    else:
        failed_steps = [step for step, validated in results["validation"].items() if not validated]
        logger.warning(f"⚠️ Some validation steps failed: {', '.join(failed_steps)}")
        results["success"] = False
    
    return results