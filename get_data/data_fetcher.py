# data_fetcher.py (Complete with Odds and StatArea Integration)
from datetime import datetime, timezone , timedelta
import logging
from typing import Optional, Dict, Any, List, Set
import asyncio
import time
import pandas as pd
import numpy as np # Added for potential use in final summary

from api_football.endpoints.game_scraper import GameScraper
from api_football.endpoints.api_manager import api_manager
from api_football.endpoints.match_processor import MatchProcessor # Refactored MatchProcessor
from api_football.endpoints.elo_fetcher import elo_fetcher, TEAM_ELO_HISTORIES_CACHE, FAILED_ELO_FETCH_TEAMS # Import the EloFetcher instance and necessary items
from api_football.endpoints.feature_calculator import feature_calculator # Import the FeatureCalculator instance
from api_football.endpoints.odds_fetcher import OddsFetcher # Import OddsFetcher
from api_football.db_mongo import db_manager # Import the DBManager instance
from api_football.statarea_async_scraper import run_scraper_async # <<< Import StatArea scraper function
from get_data.api_football.endpoints.fixture_details import FixtureDetailsFetcher


logger = logging.getLogger(__name__)

# --- process_and_prepare_ml_data function (Keep as defined previously) ---
async def process_and_prepare_ml_data(fixture_id: int, force_reprocess: bool = False) -> Optional[Dict[str, Any]]:
    """
    Processes a single fixture: gets data from match_processor, calculates features using 
    historical matches from the 'matches' collection, and returns the final ML-ready document.
    
    Args:
        fixture_id: The ID of the fixture
        force_reprocess: If True, bypass existence check in 'ml_ready'
        
    Returns:
        Dictionary with ML-ready data or None if processing failed
    """
    assert isinstance(fixture_id, int) and fixture_id > 0, "Fixture ID must be a positive integer"
    fixture_id_str = str(fixture_id)

    # 1. Check if ML data already exists
    if not force_reprocess and db_manager.check_ml_ready_data_exists(fixture_id_str):
        logger.debug(f"ML-ready data already exists for fixture {fixture_id_str}. Skipping ML processing.")
        return {"status": "skipped_ml_exists", "fixture_id": fixture_id}

    # 2. Get data from 'match_processor' collection
    match_processor_data = db_manager.get_match_processor_data(fixture_id_str)
    if not match_processor_data:
        logger.warning(f"Match processor data not found for fixture {fixture_id_str}. Cannot prepare ML data.")
        return {"status": "failed_no_processor_data", "fixture_id": fixture_id}

    # 3. Extract Key Identifiers for feature calculation
    try:
        # Direct fields from match_processor collection
        home_team_id = match_processor_data.get('home_team_id')
        away_team_id = match_processor_data.get('away_team_id')
        league_id = match_processor_data.get('league_id')
        season = match_processor_data.get('season')
        match_date_str = match_processor_data.get('match_date')
        
        # If fields are missing, try extracting from base_match_info
        if not all([home_team_id, away_team_id, league_id, season, match_date_str]):
            base_info = match_processor_data.get('base_match_info', {})
            fd = base_info.get('fixture_details', {})
            fx = fd.get('fixture', {})
            lg = fd.get('league', {})
            tm = fd.get('teams', {})
            
            if not home_team_id: home_team_id = tm.get('home', {}).get('id')
            if not away_team_id: away_team_id = tm.get('away', {}).get('id')
            if not league_id: league_id = lg.get('id')
            if not season: season = lg.get('season')
            if not match_date_str: match_date_str = fx.get('date')

        assert all([home_team_id, away_team_id, league_id, season, match_date_str]), \
            f"Missing critical info for feature calculation: fixture {fixture_id_str}"

        # Convert to appropriate types
        home_team_id = int(home_team_id)
        away_team_id = int(away_team_id)
        league_id = int(league_id)
        season = int(season)
        match_date = pd.to_datetime(match_date_str, errors='coerce', utc=True)
        
        assert pd.notna(match_date), f"Invalid match date: {match_date_str} for fixture {fixture_id_str}"

    except (AssertionError, KeyError, ValueError, TypeError) as e:
        logger.error(f"Error extracting required fields for ML prep, fixture {fixture_id_str}: {e}", exc_info=True)
        return {"status": "failed_extraction", "fixture_id": fixture_id, "error": str(e)}

    # 4. Get Historical Matches from 'matches' collection
    team_features = {}
    league_features = {}
    try:
        lookback_limit = 25  # Increased from 15 to ensure enough history
        home_history_docs = db_manager.get_historical_matches(home_team_id, match_date, limit=lookback_limit)
        away_history_docs = db_manager.get_historical_matches(away_team_id, match_date, limit=lookback_limit)

        # Combine and deduplicate historical matches
        all_historical_docs_dict = {
            doc.get('fixture_details', {}).get('fixture', {}).get('id'): doc
            for doc in home_history_docs + away_history_docs
            if doc.get('fixture_details', {}).get('fixture', {}).get('id') is not None
        }
        all_historical_docs = list(all_historical_docs_dict.values())

        if not all_historical_docs:
            logger.warning(f"No historical match documents found for teams in fixture {fixture_id_str} before {match_date.isoformat()}. Rolling features will be NaN.")
        else:
            # Build canonical DataFrame for rolling feature calculation
            historical_canonical_df = feature_calculator._build_historical_canonical_df(all_historical_docs)

            if not historical_canonical_df.empty:
                # Calculate home team features
                home_features = feature_calculator.calculate_rolling_features_for_match(
                    home_team_id, match_date, historical_canonical_df
                )
                # Add 'Home_' prefix to all keys
                team_features.update({f"Home_{k}": v for k, v in home_features.items()})

                # Calculate away team features
                away_features = feature_calculator.calculate_rolling_features_for_match(
                    away_team_id, match_date, historical_canonical_df
                )
                # Add 'Away_' prefix to all keys
                team_features.update({f"Away_{k}": v for k, v in away_features.items()})

                # Prepare data for League Features
                league_hist_rows = []
                for doc in all_historical_docs:
                    try:
                        fd_l = doc.get('fixture_details', {})
                        fx_l = fd_l.get('fixture', {})
                        lg_l = fd_l.get('league', {})
                        gl_l = fd_l.get('goals', {})  # Check path here
                        dt_l = pd.to_datetime(fx_l.get('date'), errors='coerce', utc=True)
                        if pd.notna(dt_l):
                            league_id_l = int(lg_l.get('id'))
                            season_l = int(lg_l.get('season'))
                            fthg_l = pd.to_numeric(gl_l.get('home'), errors='coerce')
                            ftag_l = pd.to_numeric(gl_l.get('away'), errors='coerce')

                            league_hist_rows.append({
                                'fixture_id': fx_l.get('id'),
                                'Date': dt_l,
                                'LeagueID': league_id_l,
                                'Season': season_l,
                                'FTHG': fthg_l,
                                'FTAG': ftag_l
                            })
                    except (TypeError, ValueError, KeyError):
                        continue

                league_hist_df = pd.DataFrame(league_hist_rows).dropna(subset=['FTHG', 'FTAG', 'LeagueID', 'Season', 'Date'])
                if not league_hist_df.empty:
                    league_hist_df['LeagueID'] = league_hist_df['LeagueID'].astype(int)
                    league_hist_df['Season'] = league_hist_df['Season'].astype(int)

                    # Calculate league features
                    league_features = feature_calculator.calculate_league_rolling_features_for_match(
                        league_id, season, match_date, league_hist_df
                    )
            else:
                logger.warning(f"Historical canonical df was empty for fixture {fixture_id_str}. Rolling features will be NaN.")

    except Exception as feature_err:
        logger.error(f"Error calculating features for fixture {fixture_id_str}: {feature_err}", exc_info=True)
        # Continue anyway, features will be missing or NaN

    # 5. Assemble Final Document
    try:
        # Extract the necessary components from match_processor_data
        base_info_for_assembly = match_processor_data.get('base_match_info', {})
        elo_data_for_assembly = {
            'HomeTeamELO': match_processor_data.get('HomeTeamELO'),
            'AwayTeamELO': match_processor_data.get('AwayTeamELO')
        }
        # Pass the main match_processor_data as raw_api_data, 
        # final_data_assembly can pick specific keys if needed.
        raw_api_data_for_assembly = match_processor_data

        final_ml_doc = feature_calculator.final_data_assembly(
            base_match_info=base_info_for_assembly,
            raw_api_data=raw_api_data_for_assembly, # Contains other API results
            elo_data=elo_data_for_assembly,
            team_features=team_features,
            league_features=league_features
        )

        assert final_ml_doc.get("MatchID") or final_ml_doc.get("fixture_id"), "Final doc missing identifier"
        assert final_ml_doc.get("Date"), "Final doc missing Date"

        final_ml_doc["status"] = "success"  # Add status marker
        return final_ml_doc

    except Exception as assembly_err:
        logger.error(f"Error assembling final ML document for fixture {fixture_id_str}: {assembly_err}", exc_info=True)
        return {"status": "failed_assembly", "fixture_id": fixture_id, "error": str(assembly_err)}


async def process_base_match_data(fixture_id: int, force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Processes a single fixture's base data: fetches fixture details, API data (using MatchProcessor),
    and saves the combined data to the match_processor collection. ELO is now fetched within MatchProcessor.

    Args:
        fixture_id: The ID of the fixture to process
        force_reprocess: If True, reprocess even if already exists in match_processor

    Returns:
        Dict with status and results
    """
    assert isinstance(fixture_id, int) and fixture_id > 0, "Fixture ID must be a positive integer"
    fixture_id_str = str(fixture_id)

    # 1. Check if data already exists in match_processor collection
    if not force_reprocess and db_manager.check_match_processor_data_exists(fixture_id_str):
        logger.debug(f"Match processor data already exists for fixture {fixture_id_str}. Skipping base processing.")
        return {"status": "skipped_exists", "fixture_id": fixture_id}

    # 2. Get base match data from 'matches' collection (or fetch if needed)
    base_match_data = db_manager.get_match_data(fixture_id_str)
    if not base_match_data:
        logger.info(f"Base match data not found for fixture {fixture_id_str}. Fetching from API...")
        fetcher = FixtureDetailsFetcher(db_manager_instance=db_manager)
        fetched_data = fetcher.get_fixture_details(fixture_id)
        if not fetched_data:
            logger.error(f"Failed to fetch fixture details for {fixture_id_str}. Cannot process.")
            return {"status": "failed_fetch_details", "fixture_id": fixture_id}
        base_match_data = db_manager.get_match_data(fixture_id_str)
        if not base_match_data:
            logger.error(f"Still couldn't retrieve base match data for fixture {fixture_id_str} after fetching.")
            return {"status": "failed_retrieve_after_fetch", "fixture_id": fixture_id}

    # 3. Extract key identifiers needed for API calls
    league_id, season, home_team_id, away_team_id, home_team_name, away_team_name, match_date_str = None, None, None, None, None, None, None
    try:
        # --- Extraction Logic (Keep the multi-path logic from previous step) ---
        fd = {}
        if 'fixture_details' in base_match_data and isinstance(base_match_data['fixture_details'], dict):
            fd = base_match_data['fixture_details']
        elif 'basic_info' in base_match_data and isinstance(base_match_data['basic_info'], list) and base_match_data['basic_info']:
            basic_info = base_match_data['basic_info'][0]
            fd = {
                'fixture': basic_info.get('fixture', {}),
                'league': basic_info.get('league', {}),
                'teams': basic_info.get('teams', {}),
                'goals': basic_info.get('goals', {}),
                'score': basic_info.get('score', {})
            }
        elif 'fixture_details_raw' in base_match_data:
            raw_data = base_match_data.get('fixture_details_raw', {})
            if 'basic_info' in raw_data and isinstance(raw_data['basic_info'], list) and raw_data['basic_info']:
                basic_info = raw_data['basic_info'][0]
                fd = {
                    'fixture': basic_info.get('fixture', {}),
                    'league': basic_info.get('league', {}),
                    'teams': basic_info.get('teams', {}),
                    'goals': basic_info.get('goals', {}),
                    'score': basic_info.get('score', {})
                }
        else:
            logger.warning(f"Assuming flatter structure for {fixture_id_str}")
            fd = {
                'fixture': base_match_data.get('match_info', {}),
                'league': {
                    'id': base_match_data.get('league_id'),
                    'season': base_match_data.get('season'),
                    'name': base_match_data.get('league_name'),
                    'country': base_match_data.get('country'),
                    'round': base_match_data.get('round')
                },
                'teams': {
                    'home': base_match_data.get('home_team'),
                    'away': base_match_data.get('away_team')
                },
                'goals': {
                    'home': base_match_data.get('home_goals'),
                    'away': base_match_data.get('away_goals')
                },
                'score': {
                    'halftime': {
                        'home': base_match_data.get('ht_home_goals'),
                        'away': base_match_data.get('ht_away_goals')
                    }
                }
            }

        fx = fd.get('fixture', {})
        lg = fd.get('league', {})
        tm = fd.get('teams', {})

        league_id = lg.get('id')
        season = lg.get('season')
        home_team_id = tm.get('home', {}).get('id')
        away_team_id = tm.get('away', {}).get('id')
        home_team_name = tm.get('home', {}).get('name')
        away_team_name = tm.get('away', {}).get('name')
        match_date_str = fx.get('date')

        # Fallbacks
        if not league_id: league_id = base_match_data.get('league_id')
        if not season: season = base_match_data.get('standings', {}).get('league', {}).get('season')
        if not home_team_id: home_team_id = base_match_data.get('home_team', {}).get('id')
        if not away_team_id: away_team_id = base_match_data.get('away_team', {}).get('id')
        if not home_team_name: home_team_name = base_match_data.get('home_team', {}).get('name')
        if not away_team_name: away_team_name = base_match_data.get('away_team', {}).get('name')
        if not match_date_str: match_date_str = base_match_data.get('match_info', {}).get('date')
        if not match_date_str: match_date_str = base_match_data.get('date_str')

        assert all([league_id, season, home_team_id, away_team_id, home_team_name, away_team_name, match_date_str]), \
            f"Missing critical info in base_match_data for fixture {fixture_id_str} after attempting all extraction paths."

        try:
            match_date = pd.to_datetime(match_date_str, errors='raise', utc=True)
        except ValueError:
            match_date = pd.to_datetime(match_date_str, format='%Y-%m-%d', errors='raise', utc=True).replace(hour=12)
        assert pd.notna(match_date), f"Invalid or unparseable match date '{match_date_str}' for fixture {fixture_id_str}"

        league_id = int(league_id)
        season = int(season)
        home_team_id = int(home_team_id)
        away_team_id = int(away_team_id)

    except (AssertionError, KeyError, ValueError, TypeError) as e:
        logger.error(f"Error extracting required fields from base match data for fixture {fixture_id_str}: {e}", exc_info=True)
        return {"status": "failed_extraction", "fixture_id": fixture_id, "error": str(e)}

    # 4. Fetch API Data + ELO using the refactored MatchProcessor
    processor = MatchProcessor()
    logger.info(f"Calling MatchProcessor.fetch_api_data_for_match for fixture {fixture_id_str} with:")
    logger.info(f"  Home Team Name: '{home_team_name}' (Type: {type(home_team_name)})")
    logger.info(f"  Away Team Name: '{away_team_name}' (Type: {type(away_team_name)})")
    logger.info(f"  Match Date: {match_date.isoformat()} (Type: {type(match_date)})")
    combined_data = await processor.fetch_api_data_for_match(
        fixture_id, league_id, season, home_team_id, away_team_id,
        home_team_name, away_team_name, match_date
    )

    # 5. Create the document for match_processor collection
    processor_doc = {
        "fixture_id": fixture_id,
        "match_date": match_date,
        "match_date_str": match_date.strftime('%Y-%m-%d %H:%M:%S%z'),
        "league_id": league_id,
        "season": season,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,
        **combined_data,
        "base_match_info": base_match_data,
    }

    # 6. Save to match_processor collection
    success = db_manager.save_match_processor_data(processor_doc)
    if not success:
        logger.error(f"Failed to save match processor data for fixture {fixture_id_str}")
        return {"status": "failed_save", "fixture_id": fixture_id}

    logger.info(f"Successfully processed and saved base data, API data, and ELO for fixture {fixture_id_str} to match_processor")
    return {"status": "success", "fixture_id": fixture_id, "data": processor_doc}


async def ensure_historical_fixtures(lookback_days: int = 15) -> Dict[str, Any]:
    """
    Ensures that fixture details for the past 'lookback_days' exist in the matches collection.
    This is critical for rolling feature calculation which needs historical match data.
    
    Args:
        lookback_days: Number of days to look back from today for historical fixtures
        
    Returns:
        Dict with status and results summary
    """
    logger.info(f"Running historical fixture backfill for past {lookback_days} days...")
    
    today = datetime.now(timezone.utc)
    start_date = today - timedelta(days=lookback_days)
    
    # Get all fixture IDs from daily_games within date range
    fixture_ids = db_manager.get_fixture_ids_from_daily_games_range(start_date, today)
    logger.info(f"Found {len(fixture_ids)} fixtures in daily_games for past {lookback_days} days")
    
    if not fixture_ids:
        logger.warning(f"No fixtures found in daily_games for past {lookback_days} days")
        return {
            "success": True, 
            "fixtures_found": 0,
            "fixtures_missing": 0,
            "fixtures_fetched": 0,
            "fixtures_failed": 0,
            "message": "No historical fixtures found to backfill"
        }
    
    # Find which fixture IDs are missing from matches collection
    missing_ids = db_manager.find_missing_fixture_ids_in_matches(fixture_ids)
    logger.info(f"Found {len(missing_ids)} missing fixtures in matches collection (out of {len(fixture_ids)})")
    
    if not missing_ids:
        logger.info("No missing historical fixtures to fetch. Matches collection is complete.")
        return {
            "success": True, 
            "fixtures_found": len(fixture_ids),
            "fixtures_missing": 0,
            "fixtures_fetched": 0,
            "fixtures_failed": 0,
            "message": "All historical fixtures already exist in matches collection"
        }
    
    # Initialize FixtureDetailsFetcher for fetching missing fixtures
    fetcher = FixtureDetailsFetcher(db_manager_instance=db_manager)
    
    # Fetch missing fixtures
    fetch_results = {
        "success": True,
        "fixtures_found": len(fixture_ids),
        "fixtures_missing": len(missing_ids),
        "fixtures_fetched": 0,
        "fixtures_failed": 0,
        "failed_ids": []
    }
    
    for fixture_id in missing_ids:
        logger.info(f"Fetching missing historical fixture: {fixture_id}")
        try:
            fixture_data = fetcher.get_fixture_details(fixture_id)
            if fixture_data:
                fetch_results["fixtures_fetched"] += 1
                logger.info(f"Successfully fetched and saved historical fixture {fixture_id}")
            else:
                fetch_results["fixtures_failed"] += 1
                fetch_results["failed_ids"].append(fixture_id)
                logger.error(f"Failed to fetch historical fixture {fixture_id}")
        except Exception as e:
            fetch_results["fixtures_failed"] += 1
            fetch_results["failed_ids"].append(fixture_id)
            logger.error(f"Error fetching historical fixture {fixture_id}: {e}", exc_info=True)
    
    if fetch_results["fixtures_failed"] > 0:
        logger.warning(f"Failed to fetch {fetch_results['fixtures_failed']} historical fixtures.")
        fetch_results["message"] = f"Completed with {fetch_results['fixtures_failed']} failures out of {len(missing_ids)} missing fixtures"
    else:
        logger.info(f"Successfully fetched all {fetch_results['fixtures_fetched']} missing historical fixtures")
        fetch_results["message"] = "Successfully fetched all missing historical fixtures"
    
    return fetch_results


async def fetch_workflow_data_v2(target_date: Optional[datetime] = None, force_reprocess: bool = False, batch_size: int = 10) -> Dict[str, Any]:
    """
    Reordered workflow for ML-ready data preparation:
    1. Get Daily Games (Fixture IDs for target date)
    1.5. Pre-fetch ELO Histories (Optional Enhancement)
    2. Process Base Match Data + API Data + ELO (Save to match_processor)
    3. Fetch Odds for target fixtures
    4. Fetch StatArea data 
    5. Historical Backfill - Ensure fixtures from last 15 days exist in matches collection
    6. Prepare ML-ready data (using match_processor and historical data)
    
    Args:
        target_date: The date to fetch data for (UTC). Defaults to today.
        force_reprocess: If True, bypasses existence checks in all collections.
        batch_size: Number of fixtures to process concurrently.
        
    Returns:
        Dict: Results summarizing the success status and actions taken.
    """
    start_time = time.time()
    if target_date is None:
        target_date = datetime.now(timezone.utc)

    date_str = target_date.strftime("%Y-%m-%d")
    logger.info(f"🚀 Starting Agentic FC ML Data Prep Workflow for: {date_str}")

    # Initialize API manager
    try:
        api_manager.initialize()
        logger.info("API Manager Initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize API Manager: {str(e)}")
        return {"success": False, "error": "API Manager initialization failed", "date": date_str}

    # Ensure DB is connected
    try:
        assert db_manager._initialized, "DB Manager not initialized"
    except (AssertionError, Exception) as db_err:
        logger.error(f"❌ Database connection issue: {db_err}", exc_info=True)
        return {"success": False, "error": "Database connection failed", "date": date_str}

    # Initialize results dictionary
    results = {
        "success": True,  # Assume success until failure
        "date": date_str,
        "total_fixtures_found": 0,
        "base_fixtures_processed": 0,
        "base_fixtures_skipped": 0,
        "base_fixtures_failed": 0,
        "odds_fixtures_processed": 0,
        "odds_fixtures_skipped": 0,
        "odds_fixtures_failed": 0,
        "statarea_teams_attempted": 0,
        "statarea_successful_scrapes": 0,
        "statarea_failed_scrapes": 0,
        "statarea_skipped_tasks": 0,
        "statarea_saved_to_mongodb": 0,
        "historical_fixtures_found": 0,
        "historical_fixtures_missing": 0,
        "historical_fixtures_fetched": 0,
        "historical_fixtures_failed": 0,
        "ml_fixtures_processed": 0,
        "ml_fixtures_skipped": 0,
        "ml_fixtures_failed": 0,
        "ml_docs_saved": 0,
        "failed_base_fixture_ids": [],
        "failed_odds_fixture_ids": [],
        "failed_historical_fixture_ids": [],
        "failed_ml_fixture_ids": [],
        "duration_seconds": 0.0,
        "elo_prefetch_attempted": 0,
        "elo_prefetch_successful": 0,
        "elo_prefetch_failed": 0,
    }
    fixture_ids_to_process: List[int] = []
    team_names_to_prefetch: Set[str] = set() # Set to store unique team names for prefetching

    # --- Step 1: Fetch Games of the Day (Fixture IDs) ---
    try:
        logger.info(f"--- Running Step 1: Fetching Games for {date_str} ---")
        scraper = GameScraper()
        naive_date = target_date.replace(tzinfo=None)

        organized_data = await asyncio.get_event_loop().run_in_executor(
            None, scraper.get_games, naive_date
        )

        if organized_data and organized_data.get("total_matches", 0) > 0:
            db_manager.save_daily_games(date_str, organized_data)
            for league_data in organized_data.get("leagues", {}).values():
                for match in league_data.get("matches", []):
                    try:
                        fixture_ids_to_process.append(int(match.get("id")))
                        # Collect team names for prefetching
                        if match.get('home_team'): team_names_to_prefetch.add(match['home_team'])
                        if match.get('away_team'): team_names_to_prefetch.add(match['away_team'])
                    except (TypeError, ValueError, AttributeError, KeyError):
                        continue
            results["total_fixtures_found"] = len(fixture_ids_to_process)
            logger.info(f"✅ Step 1 finished: Found {results['total_fixtures_found']} fixtures and {len(team_names_to_prefetch)} unique teams for {date_str}.")
        else:
            # Try loading existing data if fetch failed or returned nothing
            existing = db_manager.get_daily_games(date_str)
            if existing:
                for league_data in existing.get("leagues", {}).values():
                    for match in league_data.get("matches", []):
                        try:
                            fixture_ids_to_process.append(int(match.get("id")))
                             # Collect team names for prefetching from existing data
                            if match.get('home_team'): team_names_to_prefetch.add(match['home_team'])
                            if match.get('away_team'): team_names_to_prefetch.add(match['away_team'])
                        except (TypeError, ValueError, AttributeError, KeyError):
                            continue
                results["total_fixtures_found"] = len(fixture_ids_to_process)
                logger.info(f"✅ Step 1 loaded: Found {results['total_fixtures_found']} fixtures and {len(team_names_to_prefetch)} unique teams for {date_str} from 'daily_games'.")
            else:
                logger.info(f"ℹ️ Step 1: No games found or loaded for {date_str}.")

    except Exception as e:
        logger.error(f"❌ Critical Error in Step 1 (Fetching Games): {str(e)}", exc_info=True)
        results["success"] = False
        results["duration_seconds"] = time.time() - start_time
        return results

    # --- Exit if no fixtures found ---
    if not fixture_ids_to_process:
        logger.info("No fixtures to process. Workflow finished.")
        results["duration_seconds"] = time.time() - start_time
        return results
        
    # --- Step 1.5: Pre-fetch ELO Histories ---
    if team_names_to_prefetch:
        logger.info(f"--- Running Step 1.5: Pre-fetching ELO Histories for {len(team_names_to_prefetch)} unique teams ---")
        prefetched_count = 0
        prefetch_failed_initially = len(FAILED_ELO_FETCH_TEAMS) # Count failures *before* this run

        # Define a helper async function to wrap the synchronous call
        async def prefetch_elo_task(team_name):
            loop = asyncio.get_event_loop()
            try:
                # Run the potentially blocking synchronous function in a thread pool executor
                history = await loop.run_in_executor(None, elo_fetcher._fetch_team_elo_history, team_name)
                return history is not None # Return True if successful, False otherwise
            except Exception as e:
                 logger.warning(f"Error during ELO prefetch task for {team_name}: {e}", exc_info=False)
                 return False # Mark as failed

        # Run prefetching tasks concurrently
        tasks = [prefetch_elo_task(name) for name in team_names_to_prefetch if name not in TEAM_ELO_HISTORIES_CACHE and name not in FAILED_ELO_FETCH_TEAMS]
        results["elo_prefetch_attempted"] = len(tasks)
        
        if tasks:
            prefetch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in prefetch_results:
                if isinstance(result, bool) and result:
                    prefetched_count += 1
        
        # Calculate final counts based on cache and failed set changes
        results["elo_prefetch_successful"] = prefetched_count
        results["elo_prefetch_failed"] = len(FAILED_ELO_FETCH_TEAMS) - prefetch_failed_initially # Count newly failed teams during this prefetch run
        
        logger.info(f"✅ Step 1.5 (ELO Prefetch) complete. Attempted: {results['elo_prefetch_attempted']}, Successful: {results['elo_prefetch_successful']}, Newly Failed: {results['elo_prefetch_failed']}")
    else:
         logger.info("--- Skipping Step 1.5: No team names found for ELO pre-fetching ---")


    # --- Step 2: Process Base Match Data + API + ELO (Save to match_processor) ---
    logger.info(f"--- Running Step 2: Processing {len(fixture_ids_to_process)} fixtures for Base Match Data + API + ELO ---")
    base_processed_count = 0
    base_skipped_count = 0
    base_failed_count = 0

    for i in range(0, len(fixture_ids_to_process), batch_size):
        batch_ids = fixture_ids_to_process[i:i + batch_size]
        logger.info(f"Processing Base Match batch {i//batch_size + 1}/{(len(fixture_ids_to_process) + batch_size - 1)//batch_size}: Fixtures {batch_ids[0]}...{batch_ids[-1]}")

        tasks = [process_base_match_data(fid, force_reprocess) for fid in batch_ids]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in enumerate(batch_results):
            fixture_id = batch_ids[idx]
            if isinstance(result, Exception):
                logger.error(f"Base Match Processing Error for fixture {fixture_id}: {result}")
                base_failed_count += 1
                results["failed_base_fixture_ids"].append(fixture_id)
            elif isinstance(result, dict):
                status = result.get("status")
                if status == "success":
                    base_processed_count += 1
                elif status == "skipped_exists":
                    base_skipped_count += 1
                else:
                    base_failed_count += 1
                    results["failed_base_fixture_ids"].append(fixture_id)
            else:
                logger.error(f"Unexpected Base Match result type for fixture {fixture_id}: {type(result)}")
                base_failed_count += 1
                results["failed_base_fixture_ids"].append(fixture_id)

    results["base_fixtures_processed"] = base_processed_count
    results["base_fixtures_skipped"] = base_skipped_count
    results["base_fixtures_failed"] = base_failed_count
    logger.info(f"✅ Step 2 (Base Match Processing) complete. Processed: {base_processed_count}, Skipped: {base_skipped_count}, Failed: {base_failed_count}")

    # --- Step 3: Fetch and Save Odds Data ---
    logger.info(f"--- Running Step 3: Fetching Odds for {len(fixture_ids_to_process)} fixtures ---")
    try:
        odds_fetcher = OddsFetcher()
        odds_result = await odds_fetcher.process_fixtures_odds(
            fixture_ids=fixture_ids_to_process,
            force_reprocess=force_reprocess
        )

        results["odds_fixtures_processed"] = odds_result.get("processed_count", 0)
        results["odds_fixtures_skipped"] = odds_result.get("skipped_count", 0)
        results["odds_fixtures_failed"] = len(odds_result.get("failed_fixtures", []))
        results["failed_odds_fixture_ids"] = odds_result.get("failed_fixtures", [])

        logger.info(f"✅ Step 3 (Odds Fetching) complete. Processed: {results['odds_fixtures_processed']}, Skipped: {results['odds_fixtures_skipped']}, Failed: {results['odds_fixtures_failed']}")
        if results["failed_odds_fixture_ids"]:
            logger.warning(f"Failed Odds Fixture IDs: {results['failed_odds_fixture_ids']}")

    except Exception as e:
        logger.error(f"❌ Critical Error in Step 3 (Fetching Odds): {str(e)}", exc_info=True)
        results["success"] = False
        results["odds_fixtures_failed"] = len(fixture_ids_to_process) - results["odds_fixtures_processed"] - results["odds_fixtures_skipped"]

    # --- Step 4: Fetch and Save StatArea Data ---
    logger.info(f"--- Running Step 4: Fetching StatArea Data ---")
    try:
        # Fetch for all periods (5, 10, 15)
        statarea_result = await run_scraper_async(
            periods=[5, 10, 15],
            force_update=force_reprocess
        )

        results["statarea_teams_attempted"] = statarea_result.get("total_tasks", 0)
        results["statarea_successful_scrapes"] = statarea_result.get("successful_scrapes", 0)
        results["statarea_failed_scrapes"] = statarea_result.get("failed_scrapes", 0)
        results["statarea_skipped_tasks"] = statarea_result.get("skipped_tasks", 0)
        results["statarea_saved_to_mongodb"] = statarea_result.get("saved_to_mongodb", 0)

        if statarea_result.get("success", False):
            logger.info(f"✅ Step 4 (StatArea) complete. Saved: {results['statarea_saved_to_mongodb']}, Skipped: {results['statarea_skipped_tasks']}, Failed: {results['statarea_failed_scrapes']}")
        else:
            logger.error(f"❌ Step 4 (StatArea) failed. Error: {statarea_result.get('error', 'Unknown')}")
            results["success"] = False

    except Exception as e:
        logger.error(f"❌ Critical Error in Step 4 (Fetching StatArea): {str(e)}", exc_info=True)
        results["success"] = False

    # --- Step 5: Ensure Historical Data Completeness ---
    logger.info(f"--- Running Step 5: Ensuring Historical Data Completeness (Past 15 days) ---")
    try:
        historical_result = await ensure_historical_fixtures(lookback_days=15)
        
        results["historical_fixtures_found"] = historical_result.get("fixtures_found", 0)
        results["historical_fixtures_missing"] = historical_result.get("fixtures_missing", 0)
        results["historical_fixtures_fetched"] = historical_result.get("fixtures_fetched", 0)
        results["historical_fixtures_failed"] = historical_result.get("fixtures_failed", 0)
        results["failed_historical_fixture_ids"] = historical_result.get("failed_ids", [])
        
        logger.info(f"✅ Step 5 (Historical Backfill) complete. Found: {results['historical_fixtures_found']}, Missing: {results['historical_fixtures_missing']}, Fetched: {results['historical_fixtures_fetched']}, Failed: {results['historical_fixtures_failed']}")
        
        if results["failed_historical_fixture_ids"]:
            logger.warning(f"Failed to fetch these historical fixtures: {results['failed_historical_fixture_ids']}")
    
    except Exception as e:
        logger.error(f"❌ Critical Error in Step 5 (Historical Backfill): {str(e)}", exc_info=True)
        results["success"] = False

    # --- Step 6: Process Fixtures and Prepare ML Data (Final Step) ---
    logger.info(f"--- Running Step 6: Processing {len(fixture_ids_to_process)} fixtures for ML data ---")
    ml_processed_count = 0
    ml_skipped_count = 0
    ml_failed_count = 0
    ml_docs_batch: List[Dict[str, Any]] = []

    for i in range(0, len(fixture_ids_to_process), batch_size):
        batch_ids = fixture_ids_to_process[i:i + batch_size]
        logger.info(f"Processing ML batch {i//batch_size + 1}/{(len(fixture_ids_to_process) + batch_size - 1)//batch_size}: Fixtures {batch_ids[0]}...{batch_ids[-1]}")

        tasks = [process_and_prepare_ml_data(fid, force_reprocess) for fid in batch_ids]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in enumerate(batch_results):
            fixture_id = batch_ids[idx]
            if isinstance(result, Exception):
                logger.error(f"ML Processing Error for fixture {fixture_id}: {result}")
                ml_failed_count += 1
                results["failed_ml_fixture_ids"].append(fixture_id)
            elif isinstance(result, dict):
                status = result.get("status")
                if status == "success":
                    ml_processed_count += 1
                    ml_docs_batch.append(result)
                elif status == "skipped_ml_exists":
                    ml_skipped_count += 1
                else:
                    ml_failed_count += 1
                    results["failed_ml_fixture_ids"].append(fixture_id)
            else:
                logger.error(f"Unexpected ML result type for fixture {fixture_id}: {type(result)}")
                ml_failed_count += 1
                results["failed_ml_fixture_ids"].append(fixture_id)

        # Save the completed ML batch to MongoDB
        if ml_docs_batch:
            logger.info(f"Saving {len(ml_docs_batch)} ML-ready documents from batch...")
            docs_to_save = [{k: v for k, v in doc.items() if k != 'status'} for doc in ml_docs_batch]
            save_success = db_manager.save_ml_ready_data_bulk(docs_to_save)
            if save_success:
                results["ml_docs_saved"] += len(docs_to_save)
            else:
                logger.error(f"Failed to save bulk ML data batch for fixtures starting {batch_ids[0]}.")
                ml_failed_count += len(docs_to_save)
                results["failed_ml_fixture_ids"].extend([d.get('fixture_id') for d in docs_to_save if d.get('fixture_id')])
            ml_docs_batch = []

    results["ml_fixtures_processed"] = ml_processed_count
    results["ml_fixtures_skipped"] = ml_skipped_count
    results["ml_fixtures_failed"] = ml_failed_count
    logger.info(f"✅ Step 6 (ML Data Prep) complete. Saved: {results['ml_docs_saved']}, Skipped: {ml_skipped_count}, Failed: {ml_failed_count}")

    # Final Summary
    end_time = time.time()
    results["duration_seconds"] = round(end_time - start_time, 2)
    logger.info("--- Agentic FC ML Data Prep Workflow Summary ---")
    logger.info(f"Date: {results['date']}")
    logger.info(f"Duration: {results['duration_seconds']} seconds")
    logger.info(f"Overall Success: {results['success']}")
    logger.info(f"Fixtures Found: {results['total_fixtures_found']}")
    
    # Base Data Summary
    logger.info(f"Base Match Processing:")
    logger.info(f"  - Processed: {results['base_fixtures_processed']}")
    logger.info(f"  - Skipped: {results['base_fixtures_skipped']}")
    logger.info(f"  - Failed: {results['base_fixtures_failed']}")
    
    # Odds Data Summary
    logger.info(f"Odds Fetching:")
    logger.info(f"  - Processed: {results['odds_fixtures_processed']}")
    logger.info(f"  - Skipped: {results['odds_fixtures_skipped']}")
    logger.info(f"  - Failed: {results['odds_fixtures_failed']}")
    
    # StatArea Summary
    logger.info(f"StatArea Processing:")
    logger.info(f"  - Successful Scrapes: {results['statarea_successful_scrapes']}")
    logger.info(f"  - Failed Scrapes: {results['statarea_failed_scrapes']}")
    logger.info(f"  - Skipped (Recent Cache): {results['statarea_skipped_tasks']}")
    logger.info(f"  - Saved to DB: {results['statarea_saved_to_mongodb']}")
    
    # Historical Backfill Summary
    logger.info(f"Historical Backfill:")
    logger.info(f"  - Fixtures Found: {results['historical_fixtures_found']}")
    logger.info(f"  - Fixtures Missing: {results['historical_fixtures_missing']}")
    logger.info(f"  - Fixtures Fetched: {results['historical_fixtures_fetched']}")
    logger.info(f"  - Fixtures Failed: {results['historical_fixtures_failed']}")
    
    # ML Data Summary
    logger.info(f"ML Processing:")
    logger.info(f"  - Processed: {results['ml_fixtures_processed']}")
    logger.info(f"  - Skipped: {results['ml_fixtures_skipped']}")
    logger.info(f"  - Failed: {results['ml_fixtures_failed']}")
    logger.info(f"  - Saved: {results['ml_docs_saved']}")
    
    # ELO Pre-fetching Summary
    logger.info(f"ELO Pre-fetching:")
    logger.info(f"  - Attempted: {results['elo_prefetch_attempted']}")
    logger.info(f"  - Successful: {results['elo_prefetch_successful']}")
    logger.info(f"  - Newly Failed: {results['elo_prefetch_failed']}")
    logger.info(f"  - Total Failed Teams (Cumulative): {len(FAILED_ELO_FETCH_TEAMS)}")
    
    # Failed IDs
    if results["failed_base_fixture_ids"]:
        logger.warning(f"Failed Base Fixture IDs: {results['failed_base_fixture_ids']}")
    if results["failed_odds_fixture_ids"]:
        logger.warning(f"Failed Odds Fixture IDs: {results['failed_odds_fixture_ids']}")
    if results["failed_historical_fixture_ids"]:
        logger.warning(f"Failed Historical Fixture IDs: {results['failed_historical_fixture_ids']}")
    if results["failed_ml_fixture_ids"]:
        logger.warning(f"Failed ML Fixture IDs: {results['failed_ml_fixture_ids']}")

    return results

# Keep __main__ block for testing
if __name__ == '__main__':
    # Setup logging for direct execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', force=True)
    logger.info("Running Agentic FC ML Data Prep Workflow directly (async)...")

    async def run_main():
        target_dt = None  # Defaults to today
        # force_reprocess = True # Set True to force all checks/updates
        fetch_result = await fetch_workflow_data_v2(target_date=target_dt, force_reprocess=False, batch_size=5)
        logger.info(f"Direct run completed. Overall Success: {fetch_result.get('success')}")
        logger.info(f"Summary: {fetch_result}")  # Log the full summary dict

    try:
        asyncio.run(run_main())
    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user.")
    except Exception as main_err:
        logger.critical(f"Workflow failed with unhandled exception: {main_err}", exc_info=True)