# path: ValidatePapers.py
import json
import datetime
import http.client # Added for API calls
import time # Added for potential retries/delays
from typing import Dict, Any, List, Tuple, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure

# --- Configuration ---
# API Config
API_KEY = "dca41d4edemshe469d9d1754cd7ap1c7e06jsn7c5425d89bef" # Consider moving to env variable
API_HOST = "api-football-v1.p.rapidapi.com"
# MongoDB Config (Replace with your actual connection details)
MONGO_URI = "mongodb://admin888:admin888@127.0.0.1:27017/?authSource=admin" # 
DATABASE_NAME = "agenticfc"
COLLECTION_NAME = "matches"
# --- End Configuration ---

# --- MongoDB Helper Functions ---
def get_db_collection() -> Optional[Collection]:
    """Establishes connection to MongoDB and returns the collection object."""
    try:
        # Increased timeout slightly for potentially slower auth checks
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000) 
        # The ismaster command is cheap and does not require auth, but ping is better for auth checks.
        # Let's try pinging the authSource database directly after connection.
        client.admin.command('ping') # Use ping on the admin database specified in authSource
        db = client[DATABASE_NAME]
        print(f"Successfully connected to MongoDB database '{DATABASE_NAME}' and authenticated.")
        return db[COLLECTION_NAME]
    except ConnectionFailure as e:
        print(f"Error: Failed to connect to MongoDB at {MONGO_URI}. Details: {e}")
        return None
    except OperationFailure as e: # Catch authentication errors specifically
         print(f"Error: MongoDB authentication failed. Check credentials and authSource in URI: {MONGO_URI}. Details: {e}")
         return None
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def check_db_for_results(fixture_id: str, db_collection: Collection) -> Optional[Dict[str, Any]]:
    """Checks MongoDB for finished match results, including league and teams."""
    try:
        result = db_collection.find_one({"fixture_id": fixture_id})
        if result:
            # Ensure it's stored in the format needed and is finished
            status = result.get("status")
            score = result.get("score_fulltime")
            league = result.get("league_name") # Added
            home_team = result.get("home_team") # Added
            away_team = result.get("away_team") # Added
            finished_statuses = ['FT', 'AET', 'PEN']
            # Check if essential data is present
            if status in finished_statuses and score and score.get("home") is not None and score.get("away") is not None and league and home_team and away_team:
                 print(f"Found finished result for fixture {fixture_id} in DB.")
                 # Ensure score values are integers if they aren't already
                 result["score_fulltime"]["home"] = int(result["score_fulltime"]["home"])
                 result["score_fulltime"]["away"] = int(result["score_fulltime"]["away"])
                 # Ensure team/league names are present
                 result["league_name"] = league
                 result["home_team"] = home_team
                 result["away_team"] = away_team
                 return result
            elif status:
                 print(f"Found result for fixture {fixture_id} in DB, but status is '{status}'. Will check API.")
                 return None # Found but not finished or missing essential data
            else:
                 # Found but data incomplete or status missing, treat as not found
                 print(f"Found result for fixture {fixture_id} in DB, but missing essential data (score, league, teams, or status). Will check API.")
                 return None
        else:
            return None # Not found in DB
    except Exception as e:
        print(f"Error querying MongoDB for fixture {fixture_id}: {e}")
        return None # Treat as not found on error

def store_results_in_db(results: Dict[str, Any], db_collection: Collection):
    """Stores or updates match results in MongoDB using fixture_id as the key."""
    # Ensure all required fields are present before storing
    required_keys = ["fixture_id", "status", "score_fulltime", "league_name", "home_team", "away_team", "retrieved_at"]
    if not results or not all(key in results for key in required_keys):
        print(f"Warning: Attempted to store incomplete results in DB for fixture {results.get('fixture_id', 'UNKNOWN')}. Missing keys.")
        return
    try:
        fixture_id = results["fixture_id"]
        # Use update_one with upsert=True to insert if not exists, or update if exists
        db_collection.update_one(
            {"fixture_id": fixture_id},
            {"$set": results},
            upsert=True
        )
        print(f"Stored/Updated results for fixture {fixture_id} in DB.")
    except Exception as e:
        print(f"Error storing results for fixture {fixture_id} in MongoDB: {e}")

# --- API Fetching Logic (Modified Original Function) ---
def _get_match_results_from_api(fixture_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the actual results for a given fixture ID using the API-Football endpoint.
    Includes league and team names.
    (Internal function, renamed from get_match_results)
    """
    conn = None
    try:
        conn = http.client.HTTPSConnection(API_HOST, timeout=10)
        headers = {
            'x-rapidapi-key': API_KEY,
            'x-rapidapi-host': API_HOST
        }
        endpoint = f"/v3/fixtures?id={fixture_id}"

        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        status = res.status
        raw_data = res.read()

        if status == 200:
            data = json.loads(raw_data.decode("utf-8"))
            if not data or not data.get("response") or len(data["response"]) == 0:
                print(f"Warning: Empty or invalid API response for fixture {fixture_id}.")
                return None

            fixture_data = data["response"][0]
            fixture_info = fixture_data.get("fixture", {})
            league_info = fixture_data.get("league", {})
            team_info = fixture_data.get("teams", {})
            score_data = fixture_data.get("score", {}).get("fulltime")

            fixture_status = fixture_info.get("status", {}).get("short")
            league_name = league_info.get("name")
            home_team_name = team_info.get("home", {}).get("name")
            away_team_name = team_info.get("away", {}).get("name")


            finished_statuses = ['FT', 'AET', 'PEN']
            # Check if all required info is present and match finished
            if fixture_status in finished_statuses and score_data and \
               score_data.get("home") is not None and score_data.get("away") is not None and \
               league_name and home_team_name and away_team_name:
                results = {
                    "fixture_id": fixture_id,
                    "status": fixture_status,
                    "score_fulltime": {
                        "home": int(score_data["home"]),
                        "away": int(score_data["away"])
                    },
                    "league_name": league_name, # Added
                    "home_team": home_team_name, # Added
                    "away_team": away_team_name, # Added
                    "retrieved_at": datetime.datetime.utcnow() # Add timestamp
                }
                return results
            else:
                missing_info = []
                if not fixture_status in finished_statuses: missing_info.append(f"status ({fixture_status})")
                if not score_data or score_data.get("home") is None or score_data.get("away") is None: missing_info.append("score")
                if not league_name: missing_info.append("league name")
                if not home_team_name: missing_info.append("home team")
                if not away_team_name: missing_info.append("away team")
                print(f"API Result: Match {fixture_id} missing data or not finished. Missing: {', '.join(missing_info) if missing_info else 'None'}. Status: {fixture_status}")
                return None # Return None if not finished or essential data missing
        else:
            print(f"Error: API request failed for fixture {fixture_id}. Status: {status}, Response: {raw_data.decode('utf-8')}")
            return None

    except http.client.HTTPException as e:
        print(f"Error: HTTP request exception for fixture {fixture_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON response for fixture {fixture_id}: {e}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while fetching results for {fixture_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

# --- Combined DB Check and API Fetch ---
def get_or_fetch_match_results(fixture_id: str, db_collection: Collection) -> Optional[Dict[str, Any]]:
    """
    Gets match results from DB cache if available and finished, otherwise fetches from API
    and updates the cache.
    """
    # 1. Check DB first
    db_result = check_db_for_results(fixture_id, db_collection)
    if db_result:
        return db_result

    # 2. If not in DB or not finished, fetch from API
    print(f"Fetching results for fixture {fixture_id} from API...")
    api_result = _get_match_results_from_api(fixture_id)

    # 3. If API fetch was successful and match is finished, store in DB
    if api_result:
        store_results_in_db(api_result, db_collection)
        return api_result
    else:
        # API fetch failed or match not finished
        return None

# --- Validation Logic (No changes needed from previous state) ---
def validate_selection(selection_data: Dict[str, Any], match_results: Dict[str, Any]) -> bool:
    """
    Validates if a betting selection was correct based on match results.
    Args:
        selection_data: The dictionary representing a single bet selection.
        match_results: The dictionary containing the actual match results from get_match_results.
    Returns:
        True if the selection was correct, False otherwise.
    """
    bet_type = selection_data.get("selection")
    # Ensure match_results and score are valid before proceeding
    if not bet_type or not match_results or "score_fulltime" not in match_results:
        print(f"Warning: Cannot validate selection - invalid bet type or missing results/score for fixture {selection_data.get('fixture_id')}.")
        return False

    score_ft = match_results["score_fulltime"]
    if score_ft.get("home") is None or score_ft.get("away") is None:
         print(f"Warning: Cannot validate selection - missing home/away score for fixture {selection_data.get('fixture_id')}.")
         return False

    home_goals_ft = int(score_ft["home"]) # Ensure integer comparison
    away_goals_ft = int(score_ft["away"]) # Ensure integer comparison
    total_goals_ft = home_goals_ft + away_goals_ft

    # --- Validation logic based on market_map_simple mappings ---
    try:
        # Match Winner bets
        if bet_type in ["1", "H", "Home Win"]:
            return home_goals_ft > away_goals_ft
        elif bet_type in ["X", "D", "Draw"]:
            return home_goals_ft == away_goals_ft
        elif bet_type in ["2", "A", "Away Win"]:
            return home_goals_ft < away_goals_ft

        # Double Chance bets
        elif bet_type in ["1X", "Home or Draw"]:
            return home_goals_ft >= away_goals_ft
        elif bet_type in ["X2", "Away or Draw"]:
            return home_goals_ft <= away_goals_ft
        elif bet_type in ["12", "No Draw (Home or Away Win)"]:
            return home_goals_ft != away_goals_ft

        # Goals Over/Under bets
        elif bet_type in ["O0.5", "Over 0.5 Goals"]:
            return total_goals_ft > 0.5
        elif bet_type in ["O1.5", "Over 1.5 Goals"]:
             return total_goals_ft > 1.5
        elif bet_type in ["O2.5", "Over 2.5 Goals"]:
            return total_goals_ft > 2.5
        elif bet_type in ["O3.5", "Over 3.5 Goals"]:
            return total_goals_ft > 3.5
        elif bet_type in ["O4.5", "Over 4.5 Goals"]:
            return total_goals_ft > 4.5
        elif bet_type in ["U0.5", "Under 0.5 Goals"]:
            return total_goals_ft < 0.5
        elif bet_type in ["U1.5", "Under 1.5 Goals"]:
            return total_goals_ft < 1.5
        elif bet_type in ["U2.5", "Under 2.5 Goals"]:
            return total_goals_ft < 2.5
        elif bet_type in ["U3.5", "Under 3.5 Goals"]:
            return total_goals_ft < 3.5
        elif bet_type in ["U4.5", "Under 4.5 Goals"]:
            return total_goals_ft < 4.5

        # Both Teams to Score bets
        elif bet_type in ["BTTS Yes"]:
            return home_goals_ft > 0 and away_goals_ft > 0
        elif bet_type in ["BTTS No"]:
            return not (home_goals_ft > 0 and away_goals_ft > 0)
        # Combined bets based on market_map_combined
        elif " and " in bet_type:
            parts = bet_type.split(" and ")
            if parts == ["1X", "U3.5"]:
                 return (home_goals_ft >= away_goals_ft) and (total_goals_ft < 3.5)
            elif parts == ["1X", "O2.5"]:
                return (home_goals_ft >= away_goals_ft) and (total_goals_ft > 2.5)
            elif parts == ["1X", "U2.5"]:
                return (home_goals_ft >= away_goals_ft) and (total_goals_ft < 2.5)
            elif parts == ["1X", "U1.5"]:
                return (home_goals_ft >= away_goals_ft) and (total_goals_ft < 1.5)
            elif parts == ["1X", "U4.5"]:
                return (home_goals_ft >= away_goals_ft) and (total_goals_ft < 4.5)
            elif parts == ["X2", "U3.5"]:
                return (home_goals_ft <= away_goals_ft) and (total_goals_ft < 3.5)
            elif parts == ["X2", "O2.5"]:
                return (home_goals_ft <= away_goals_ft) and (total_goals_ft > 2.5)
            elif parts == ["X2", "U2.5"]:
                return (home_goals_ft <= away_goals_ft) and (total_goals_ft < 2.5)
            elif parts == ["X2", "U1.5"]:
                return (home_goals_ft <= away_goals_ft) and (total_goals_ft < 1.5)
            elif parts == ["X2", "U4.5"]:
                return (home_goals_ft <= away_goals_ft) and (total_goals_ft < 4.5)
            elif parts == ["BTTS No", "U2.5"]:
                btts_no = not (home_goals_ft > 0 and away_goals_ft > 0)
                u2_5 = total_goals_ft < 2.5
                return btts_no and u2_5
            elif parts == ["BTTS Yes", "O2.5"]:
                btts_yes = home_goals_ft > 0 and away_goals_ft > 0
                o2_5 = total_goals_ft > 2.5
                return btts_yes and o2_5
            elif parts == ["BTTS No", "U3.5"]:
                btts_no = not (home_goals_ft > 0 and away_goals_ft > 0)
                u3_5 = total_goals_ft < 3.5
                return btts_no and u3_5
            elif parts == ["12", "O2.5"]:
                no_draw = home_goals_ft != away_goals_ft
                o2_5 = total_goals_ft > 2.5
                return no_draw and o2_5
            elif parts == ["12", "U2.5"]:
                no_draw = home_goals_ft != away_goals_ft
                u2_5 = total_goals_ft < 2.5
                return no_draw and u2_5
            else:
                print(f"Warning: Validation for combined bet '{bet_type}' not implemented.")
                return False

        else:
            print(f"Warning: Validation logic for bet type '{bet_type}' is not implemented.")
            return False

    except Exception as e:
        print(f"Error during validation logic for bet '{bet_type}' fixture {selection_data.get('fixture_id')}: {e}")
        return False


def calculate_stake_weighted_roi(papers_data: List[Dict[str, Any]], validation_results: Dict[str, Dict[str, bool]]) -> float:
    """
    Calculates the overall Return on Investment based on validated selections and OPTIMAL STAKES.
    (Renamed from calculate_roi for clarity)
    """
    total_staked = 0.0
    total_returned = 0.0

    for paper in papers_data:
        paper_id = paper.get("paper_id")
        selections = paper.get("staked_selections", [])
        paper_validation = validation_results.get(paper_id, {})

        for selection in selections:
            fixture_id = selection.get("fixture_id")
            # Ensure stake_fraction and odds are treated as floats
            try:
                stake_fraction = float(selection.get("optimal_stake_fraction", 0.0))
                odds = float(selection.get("odds", 1.0))
            except (ValueError, TypeError):
                print(f"Warning: Invalid stake or odds for selection in paper {paper_id}, fixture {fixture_id}. Skipping stake calculation.")
                continue

            is_correct = paper_validation.get(fixture_id, False)

            # Only consider stakes deemed meaningful by the optimization process
            if selection.get("has_meaningful_stake", False) and stake_fraction > 0: # Ensure stake is positive
                total_staked += stake_fraction
                if is_correct:
                    total_returned += stake_fraction * odds # Return includes original stake

    if total_staked == 0:
        return 0.0

    # ROI = (Total Returned - Total Staked) / Total Staked
    roi = (total_returned - total_staked) / total_staked
    return roi

# --- Paper-Level (All-or-Nothing) ROI Calculation (New) ---
def calculate_paper_level_roi(papers_data: List[Dict[str, Any]], validation_results: Dict[str, Dict[str, bool]]) -> Optional[float]:
    """
    Calculates ROI assuming each paper is a single bet:
    - Wins only if ALL selections in the paper are correct. Win amount = combined_odds - 1 unit.
    - Loses 1 unit if ANY selection is incorrect or cannot be validated.
    Returns ROI as a percentage (e.g., 0.1 for 10% ROI). Returns None if no papers are processed.
    """
    total_papers_betted = 0
    net_units = 0.0

    for paper in papers_data:
        paper_id = paper.get("paper_id")
        paper_summary = paper.get("paper_summary", {})
        combined_odds_str = paper_summary.get("combined_odds")
        paper_validation = validation_results.get(paper_id)

        # Check if paper has results and combined odds
        if not paper_id or not paper_validation or not combined_odds_str:
            # Cannot evaluate this paper (missing ID, validation results, or odds)
            continue

        # Check if the paper actually contained selections
        if not paper_validation: # Empty dict means no selections were validated
             continue

        try:
            combined_odds = float(combined_odds_str)
        except (ValueError, TypeError):
             print(f"Warning: Invalid combined_odds '{combined_odds_str}' for paper {paper_id}. Skipping paper ROI calculation.")
             continue # Skip if odds are invalid

        total_papers_betted += 1 # Count this paper as 'betted' (1 unit stake)

        # Check if all validated selections were correct
        all_correct = all(paper_validation.values())

        if all_correct:
            # Win: net gain is (combined_odds - 1) units
            net_units += (combined_odds - 1.0)
        else:
            # Loss: net loss is 1 unit
            net_units -= 1.0

    if total_papers_betted == 0:
        return None # Avoid division by zero if no papers could be evaluated

    # ROI = Total Net Units / Total Units Staked (which is total_papers_betted)
    roi = net_units / total_papers_betted
    return roi

# --- Combined Portfolio ROI Calculation ---
def calculate_combined_portfolio_roi(papers_data: List[Dict[str, Any]], validation_results: Dict[str, Dict[str, bool]]) -> Optional[Dict[str, Any]]:
    """
    Calculates a combined portfolio ROI across all papers, treating them as a single day's portfolio.
    Returns detailed metrics including total units staked, won, and net profit/loss.
    """
    # Tracking variables
    total_staked_units = 0.0
    total_returns = 0.0
    total_papers_evaluated = 0
    successful_papers = 0
    
    # Keep a log of paper outcomes for detailed analysis
    paper_outcomes = []
    
    for paper in papers_data:
        paper_id = paper.get("paper_id")
        paper_summary = paper.get("paper_summary", {})
        combined_odds_str = paper_summary.get("combined_odds")
        paper_validation = validation_results.get(paper_id)
        
        # Skip papers without validation results or odds
        if not paper_id or not paper_validation or not combined_odds_str:
            continue
        
        # Skip papers with no validated selections
        if not paper_validation:
            continue
            
        try:
            combined_odds = float(combined_odds_str)
            stake = 1.0  # Use standard 1 unit stake per paper
            
            # Count this paper in the portfolio
            total_papers_evaluated += 1
            total_staked_units += stake
            
            # Determine outcome
            all_correct = all(paper_validation.values())
            
            # Calculate returns and profit/loss for this paper
            paper_return = combined_odds * stake if all_correct else 0.0
            paper_profit = (combined_odds * stake - stake) if all_correct else -stake
            total_returns += paper_return
            
            # Calculate per-paper ROI percentage
            paper_roi_percentage = (paper_profit / stake) * 100.0 if stake > 0 else 0.0

            # Store detailed paper outcome
            paper_outcome = {
                "paper_id": paper_id,
                "combined_odds": combined_odds,
                "stake": stake,
                "all_correct": all_correct,
                "return": paper_return,
                "profit_loss_units": paper_profit, # Renamed for clarity
                "profit_loss_percentage": f"{paper_roi_percentage:.2f}%" # Added percentage format
            }
            
            paper_outcomes.append(paper_outcome)
            
            if all_correct:
                successful_papers += 1
            
        except (ValueError, TypeError):
            print(f"Warning: Invalid combined_odds '{combined_odds_str}' for paper {paper_id}. Skipping in portfolio ROI calculation.")
            continue
    
    # Calculate portfolio metrics
    if total_staked_units == 0:
        return None  # No papers were evaluated
        
    net_profit = total_returns - total_staked_units
    roi_decimal = net_profit / total_staked_units if total_staked_units > 0 else 0.0 # Keep raw decimal, handle division by zero
    # Use the correct percentage format specifier :.2%
    roi_percentage_str = f"{roi_decimal:.2%}" # Format as percentage string (e.g., "2.33%")
    success_rate = successful_papers / total_papers_evaluated if total_papers_evaluated > 0 else 0
    
    # Calculate absolute units won/lost for clarity
    total_units_won = total_returns
    total_units_lost = total_staked_units - successful_papers  # Lost 1 unit for each unsuccessful paper
    net_units = total_units_won - total_staked_units
    
    return {
        "total_papers_in_portfolio": total_papers_evaluated,
        "successful_papers": successful_papers,
        "unsuccessful_papers": total_papers_evaluated - successful_papers,
        "success_rate": success_rate,
        "success_rate_percentage": f"{success_rate:.2%}", # Correct format
        "total_units_staked": total_staked_units,
        "total_units_won": total_units_won,
        "total_units_lost": total_units_lost, # Units lost due to failed papers
        "net_profit_units": net_units,
        "combined_portfolio_roi_decimal": roi_decimal, # Keep decimal value
        "combined_portfolio_roi_percentage": roi_percentage_str, # Formatted string using :.2%
        "paper_outcomes": paper_outcomes, # Will now include profit_loss_percentage
        "final_balance": {
            "starting_balance": total_papers_evaluated,  # 1 unit per paper
            "ending_balance": total_returns,
            "absolute_change": net_units,
            "percentage_change": roi_percentage_str # Use consistent percentage format from :.2%
        }
    }

# --- Detailed Report Generation (Enhanced) ---
def generate_detailed_report(papers_data: List[Dict[str, Any]],
                             validation_results: Dict[str, Dict[str, bool]],
                             all_selection_details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates a detailed report dictionary with enhanced analysis."""
    report = {}
    total_papers = len(papers_data)
    # Filter selections that were actually validated (not skipped)
    validated_selections = [s for s in all_selection_details if s["validation_status"] == "Validated"]
    total_selections_validated = len(validated_selections)
    correct_selections_count = sum(1 for s in validated_selections if s["is_correct"])
    failed_selections = [s for s in validated_selections if not s["is_correct"]]
    total_failed_selections = len(failed_selections)

    # Paper win rate calculation (Only consider papers with at least one validated selection)
    evaluated_papers_count = 0
    correct_papers_count = 0
    for paper_id, results in validation_results.items():
        # Check if the paper had any validated selections associated with it
        paper_selections = [s for s in validated_selections if s["paper_id"] == paper_id]
        if paper_selections: # Only count papers that had selections attempted for validation
             evaluated_papers_count +=1
             # Check if all *validated* selections for this paper were correct
             all_validated_correct = all(s["is_correct"] for s in paper_selections)
             if all_validated_correct:
                 correct_papers_count += 1


    # --- ROI Calculations ---
    stake_weighted_roi = calculate_stake_weighted_roi(papers_data, validation_results)
    paper_level_roi_val = calculate_paper_level_roi(papers_data, validation_results)
    # Format Paper-Level ROI as percentage
    paper_level_roi_str = f"{paper_level_roi_val * 100.0:.2f}%" if paper_level_roi_val is not None else "N/A (No papers evaluated)"
    
    # Add new combined portfolio ROI calculation
    portfolio_roi_data = calculate_combined_portfolio_roi(papers_data, validation_results)
    # Use the pre-formatted percentage string from portfolio_roi_data
    combined_portfolio_roi_str = portfolio_roi_data.get("combined_portfolio_roi_percentage", "N/A (No portfolio data)") if portfolio_roi_data else "N/A (No portfolio data)"


    # First add detailed portfolio analysis BEFORE generating insights
    # This already includes the percentage fields calculated above
    report["portfolio_analysis"] = portfolio_roi_data if portfolio_roi_data else {"message": "No portfolio data available"}

    # --- Calculate League Performance BEFORE Insights ---
    league_stats = {}
    for selection in validated_selections: # Use only validated selections
        league = selection.get("league_name", "Unknown League")
        if league not in league_stats:
            league_stats[league] = {"total": 0, "correct": 0, "failed": 0}

        league_stats[league]["total"] += 1
        if selection["is_correct"]:
            league_stats[league]["correct"] += 1
        else:
            league_stats[league]["failed"] += 1

    league_performance = []
    for league, stats in league_stats.items():
        total = stats["total"]
        if total > 0:
            correct = stats["correct"]
            failed = stats["failed"]
            win_rate = correct / total
            league_performance.append({
                "league_name": league,
                "total_selections": total,
                "correct_selections": correct,
                "failed_selections": failed,
                "win_rate": f"{win_rate:.2%}"
            })
    # Sort leagues by win rate (descending), then by total selections (descending) for ties
    league_performance_sorted = sorted(league_performance, key=lambda x: (float(x["win_rate"][:-1]), x["total_selections"]), reverse=True)


    # --- Calculate Failure Analysis by Type BEFORE Insights ---
    failed_selection_types = {}
    for selection in failed_selections: # Use pre-filtered list
            bet_type = selection["bet_type"]
            failed_selection_types[bet_type] = failed_selection_types.get(bet_type, 0) + 1
    # Sort by count descending
    failed_selections_by_type_summary_sorted = dict(sorted(
        failed_selection_types.items(), key=lambda item: item[1], reverse=True
    ))

    # --- Calculate Combined Odds Analysis BEFORE Insights ---
    combined_odds_list = []
    successful_paper_combined_odds = []
    failed_paper_combined_odds = []

    for paper in papers_data:
        paper_id = paper.get("paper_id")
        paper_summary = paper.get("paper_summary", {})
        combined_odds_str = paper_summary.get("combined_odds")
        paper_validation = validation_results.get(paper_id)

        if not paper_id or not paper_validation or not combined_odds_str:
             continue # Skip papers without necessary info

        # Check if the paper had any validated selections
        paper_selections = [s for s in validated_selections if s["paper_id"] == paper_id]
        if not paper_selections:
            continue # Skip papers with no validated selections

        try:
            combined_odds = float(combined_odds_str)
            combined_odds_list.append(combined_odds)

            # Check if *all validated* selections were correct
            all_validated_correct = all(s["is_correct"] for s in paper_selections)
            if all_validated_correct:
                 successful_paper_combined_odds.append(combined_odds)
            else:
                 failed_paper_combined_odds.append(combined_odds)

        except (ValueError, TypeError):
            print(f"Warning: Invalid combined_odds format for paper {paper.get('paper_id')}. Skipping odds analysis for this paper.")
            continue

    if combined_odds_list:
        combined_odds_analysis_data = {
             "overall_average": sum(combined_odds_list) / len(combined_odds_list) if combined_odds_list else 0,
             "overall_min": min(combined_odds_list) if combined_odds_list else 0,
             "overall_max": max(combined_odds_list) if combined_odds_list else 0,
             "count_papers_with_odds": len(combined_odds_list),
             "successful_papers_average_odds": sum(successful_paper_combined_odds) / len(successful_paper_combined_odds) if successful_paper_combined_odds else None,
             "successful_papers_count": len(successful_paper_combined_odds),
             "failed_papers_average_odds": sum(failed_paper_combined_odds) / len(failed_paper_combined_odds) if failed_paper_combined_odds else None,
             "failed_papers_count": len(failed_paper_combined_odds),
        }
    else:
         combined_odds_analysis_data = {"message": "No valid combined odds found in papers with validated selections."}


    # --- Calculate Portfolio Performance Summary BEFORE Insights ---
    report["portfolio_performance_summary"] = {
        "total_papers_processed": total_papers, # Papers found in input file
        "total_papers_evaluated": evaluated_papers_count, # Papers with at least one validated selection
        "correct_papers": correct_papers_count,
        "paper_win_rate": f"{(correct_papers_count / evaluated_papers_count):.2%}" if evaluated_papers_count > 0 else "N/A",
        "total_selections_validated": total_selections_validated,
        "correct_selections": correct_selections_count,
        "failed_selections": total_failed_selections,
        "selection_win_rate": f"{(correct_selections_count / total_selections_validated):.2%}" if total_selections_validated > 0 else "N/A",
        # Format Stake-Weighted ROI as percentage
        "stake_weighted_roi": f"{stake_weighted_roi * 100.0:.2f}%",
        "paper_level_roi": paper_level_roi_str, # Already formatted
        "combined_portfolio_roi": combined_portfolio_roi_str,  # Use pre-formatted string
        "calculation_notes": {
            "paper_win_rate": "Based on papers where all validated selections were correct, out of papers with at least one validated selection.",
            "selection_win_rate": "Based on correctly validated selections out of total validated selections.",
            # Updated description for clarity
            "stake_weighted_roi": "Portfolio ROI calculated by weighting each selection's outcome by its optimal_stake_fraction.",
            "paper_level_roi": "Portfolio ROI calculated assuming 1 unit bet per paper. Paper wins if all selections are correct (ROI = combined_odds - 1), loses if any selection fails (ROI = -100%).",
            "combined_portfolio_roi": "Same as Paper-Level ROI, treating all evaluated papers as a single portfolio."
        }
    }

    # --- 7. Performance Insights (Now calculated AFTER prerequisite data) ---
    insights = []
    # Insight on ROI (Using formatted strings)
    insights.append(f"Overall Stake-Weighted ROI is {report['portfolio_performance_summary']['stake_weighted_roi']}.") # Use formatted value
    if paper_level_roi_val is not None:
        insights.append(f"Overall Paper-Level ROI (1 unit/paper) is {paper_level_roi_str}.") # Use formatted value
        if paper_level_roi_val > 0:
             insights.append("The paper-level ROI suggests profitability when treating each paper as a single accumulator bet.")
        else:
             insights.append("The paper-level ROI suggests unprofitability when treating each paper as a single accumulator bet.")
    else:
        insights.append("Paper-level ROI could not be calculated (likely no papers had valid odds and validated selections).")

    # Insight on Win Rates
    # Now uses the pre-calculated portfolio_performance_summary
    if evaluated_papers_count > 0 and total_selections_validated > 0:
        insights.append(f"Paper Win Rate ({report['portfolio_performance_summary']['paper_win_rate']}) vs Selection Win Rate ({report['portfolio_performance_summary']['selection_win_rate']}): Evaluate if combining selections into papers adds or detracts value.")
    elif evaluated_papers_count == 0:
         insights.append("Paper Win Rate could not be calculated (no papers evaluated).")
    else: # total_selections_validated == 0
         insights.append("Selection Win Rate could not be calculated (no selections validated).")

    # Insight on Leagues (Uses pre-calculated league_performance_sorted)
    if league_performance_sorted:
        best_league = league_performance_sorted[0]
        worst_league = league_performance_sorted[-1]
        insights.append(f"Highest performing league by win rate: {best_league['league_name']} ({best_league['win_rate']} from {best_league['total_selections']} selections).")
        if len(league_performance_sorted) > 1:
             insights.append(f"Lowest performing league by win rate: {worst_league['league_name']} ({worst_league['win_rate']} from {worst_league['total_selections']} selections). Consider reviewing strategy for this league.")
    else:
        insights.append("No league performance data available (no selections were successfully validated).")

    # Insight on Failed Bet Types (Uses pre-calculated failed_selections_by_type_summary_sorted)
    if failed_selections_by_type_summary_sorted:
        most_failed_type = list(failed_selections_by_type_summary_sorted.keys())[0]
        fail_count = failed_selections_by_type_summary_sorted[most_failed_type]
        insights.append(f"Most frequent failing bet type: '{most_failed_type}' ({fail_count} failures). Review model predictions or selection criteria for this market.")
        # Add more detail? e.g., top 3?
        if len(failed_selections_by_type_summary_sorted) > 2:
            top_3_fails = list(failed_selections_by_type_summary_sorted.items())[:3]
            insights.append(f"Top 3 failing bet types: {', '.join([f'{k} ({v})' for k, v in top_3_fails])}.")

    else:
        insights.append("No specific bet type failure analysis available (no validated selections failed).")

    # Insight on Combined Odds (Uses pre-calculated combined_odds_analysis_data)
    if combined_odds_analysis_data.get("successful_papers_average_odds") is not None and combined_odds_analysis_data.get("failed_papers_average_odds") is not None:
         avg_odds_success = combined_odds_analysis_data["successful_papers_average_odds"]
         avg_odds_fail = combined_odds_analysis_data["failed_papers_average_odds"]
         insights.append(f"Average combined odds for successful papers ({avg_odds_success:.2f}) vs failed papers ({avg_odds_fail:.2f}).")
         # Add comparison logic here if desired
         if avg_odds_success < avg_odds_fail:
              insights.append("Consider if higher-odds papers are disproportionately failing.")
         elif avg_odds_success > avg_odds_fail:
              insights.append("Successful papers tend to have higher combined odds on average.")

    # Add insight for the combined portfolio ROI (updated for clarity and formatting)
    if portfolio_roi_data:
        portfolio_data = report["portfolio_analysis"] # Already fetched
        net_units = portfolio_data.get('net_profit_units', 0)
        total_papers = portfolio_data.get('total_papers_in_portfolio', 0)
        # Use the formatted combined_portfolio_roi_str
        insights.append(f"Combined Portfolio Final Result: {net_units:+.2f} units net profit/loss from {total_papers} papers (ROI: {combined_portfolio_roi_str}).")

        # Add detailed breakdown of wins/losses
        total_staked = portfolio_data.get('total_units_staked', 0)
        total_won = portfolio_data.get('total_units_won', 0)
        insights.append(f"Portfolio summary: Staked {total_staked:.2f} units → Won {total_won:.2f} units → Net {net_units:+.2f} units") # Added sign for clarity

        # Show success vs failure count using formatted success rate
        successful_papers = portfolio_data.get('successful_papers', 0)
        unsuccessful_papers = portfolio_data.get('unsuccessful_papers', 0)
        success_rate_str = portfolio_data.get('success_rate_percentage', 'N/A')
        insights.append(f"Papers: {successful_papers} successful / {unsuccessful_papers} unsuccessful (Success rate: {success_rate_str})")

    report["performance_insights"] = insights

    # --- Add other report sections using pre-calculated data ---
    report["league_performance"] = league_performance_sorted
    report["failed_selections_by_type_summary"] = failed_selections_by_type_summary_sorted
    report["combined_odds_analysis"] = combined_odds_analysis_data


    # --- 4. Detailed Failed Selections ---
    # Include more context for each failed selection
    detailed_failures = []
    for failure in failed_selections: # Use pre-filtered list
         detail = {
              "paper_id": failure.get("paper_id"),
              "fixture_id": failure.get("fixture_id"),
              "league_name": failure.get("league_name"),
              "home_team": failure.get("home_team"),
              "away_team": failure.get("away_team"),
              "bet_type": failure.get("bet_type"),
              "odds": failure.get("odds"),
              "actual_score": failure.get("actual_score"),
              "optimal_stake_fraction": failure.get("optimal_stake_fraction"),
              "has_meaningful_stake": failure.get("has_meaningful_stake")
         }
         detailed_failures.append(detail)
    # Sort by paper_id, then fixture_id for consistency
    report["detailed_failed_selections"] = sorted(detailed_failures, key=lambda x: (x["paper_id"], x["fixture_id"]))

    # --- 6. Detailed Paper Breakdown (All Selections - Unchanged Logic but uses validated list) ---
    papers_detail = {}
    # Sort all validated selections first
    all_validated_details_sorted = sorted(validated_selections, key=lambda x: (x["paper_id"], x["fixture_id"]))

    for selection in all_validated_details_sorted:
        paper_id = selection["paper_id"]
        if paper_id not in papers_detail:
            # Find the corresponding paper data for summary info
            paper_info = next((p for p in papers_data if p.get("paper_id") == paper_id), {})
            paper_summary = paper_info.get("paper_summary", {})
            papers_detail[paper_id] = {
                "paper_summary": { # Include summary for context
                    "combined_odds": paper_summary.get("combined_odds"),
                    "expected_value": paper_summary.get("expected_value"),
                    "total_optimal_stake": paper_summary.get("total_optimal_stake_fraction")
                },
                "selections": [],
                "paper_overall_correct": None # Will be set based on validation_results for validated selections
            }

        # Simplify selection dict for report
        selection_report_item = {k: v for k, v in selection.items() if k != 'paper_id'} # Remove redundant paper_id
        papers_detail[paper_id]["selections"].append(selection_report_item)

    # Add overall paper correctness status based on *validated* selections
    for paper_id, detail_dict in papers_detail.items():
        paper_selections = detail_dict["selections"]
        if not paper_selections: # Should not happen if built from validated_selections, but safe check
             detail_dict["paper_overall_correct"] = "N/A (No Validated Selections)"
        else:
             # Check if all selections *listed* (which are the validated ones for this paper) are correct
             detail_dict["paper_overall_correct"] = all(s["is_correct"] for s in paper_selections)

    report["detailed_paper_results"] = papers_detail # Contains all validated selections grouped by paper

    return report

# --- Main Execution Logic (Enhanced Console Output) ---
def main(json_path: str = "data/output/optimized_game_portfolios.json",
         report_path: str = "data/output/validation_detailed_report.json"):
    """
    Main function to load data, validate papers/selections using DB cache
    (including league/teams), and generate enhanced detailed reports.
    """
    # --- 1. Connect to MongoDB ---
    db_collection = get_db_collection()
    if db_collection is None:
        print("Exiting due to MongoDB connection failure.")
        return

    # --- 2. Load JSON Data ---
    print(f"Attempting to load JSON from: {json_path}")
    try:
        with open(json_path, 'r') as f:
            loaded_json_data = json.load(f)
        print(f"Successfully loaded JSON. Type: {type(loaded_json_data)}")
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {json_path}: {e}")
        return
    except Exception as e:
        print(f"Error during file reading/loading: {e}")
        return

    if not isinstance(loaded_json_data, dict):
        print(f"Error: Expected JSON root to be a dictionary, but got {type(loaded_json_data)}.")
        return

    papers_key = 'optimized_papers'
    papers_data = loaded_json_data.get(papers_key, [])

    if not papers_data:
        print(f"Warning: Key '{papers_key}' not found or list is empty in JSON.")
        papers_data = [] # Ensure it's a list even if empty

    # --- 3. Validation Process ---
    validation_results: Dict[str, Dict[str, bool]] = {} # Stores correctness per fixture per paper
    all_selection_details: List[Dict[str, Any]] = [] # Stores detailed info for reporting

    print("Starting validation process...")
    processed_fixture_ids = set() # Track fixtures processed in this run

    for paper in papers_data:
        if not isinstance(paper, dict):
            print(f"Warning: Skipping item in papers_data because it's not a dictionary: {type(paper)}")
            continue

        paper_id = paper.get("paper_id")
        if not paper_id:
            print("Warning: Skipping paper with missing 'paper_id'.")
            continue

        selections = paper.get("staked_selections", [])
        validation_results[paper_id] = {} # Initialize results for this paper

        for selection in selections:
            if not isinstance(selection, dict):
                print(f"Warning: Skipping non-dictionary selection item in paper {paper_id}.")
                continue # Skip invalid selection format

            fixture_id_any = selection.get("fixture_id") # Could be int or str
            if not fixture_id_any:
                print(f"Warning: Skipping selection in paper {paper_id} due to missing 'fixture_id'.")
                continue # Skip processing this selection

            fixture_id = str(fixture_id_any) # Ensure fixture_id is a string for consistency

            # --- Prepare selection detail dict early ---
            selection_detail = {
                "paper_id": paper_id,
                "fixture_id": fixture_id, # Store as string
                "bet_type": selection.get("selection"),
                "odds": selection.get("odds"),
                "optimal_stake_fraction": selection.get("optimal_stake_fraction"),
                "has_meaningful_stake": selection.get("has_meaningful_stake", False),
                "league_name": "Unknown", # Default
                "home_team": "Unknown",   # Default
                "away_team": "Unknown",   # Default
                "is_correct": False,      # Default to False
                "validation_status": "Pending",
                "actual_score": None
            }

            # --- Get match results (from DB or API) ---
            # Only fetch if not already processed in this run (basic optimization)
            # Note: This assumes results don't change rapidly during the script run
            match_results = None
            # if fixture_id not in processed_fixture_ids: # Commenting out simple cache - get_or_fetch handles DB check
            match_results = get_or_fetch_match_results(fixture_id, db_collection)
                # if match_results:
                #     processed_fixture_ids.add(fixture_id) # Add to set if fetched successfully
            # else:
            #     # If already processed, check DB again (in case it finished between checks)
            #     match_results = check_db_for_results(fixture_id, db_collection)


            if match_results:
                # --- Populate details from results ---
                selection_detail["league_name"] = match_results.get("league_name", "Unknown")
                selection_detail["home_team"] = match_results.get("home_team", "Unknown")
                selection_detail["away_team"] = match_results.get("away_team", "Unknown")
                selection_detail["actual_score"] = match_results.get("score_fulltime")

                # --- Validate the selection ---
                is_selection_correct = validate_selection(selection, match_results)
                validation_results[paper_id][fixture_id] = is_selection_correct # Store result
                selection_detail["is_correct"] = is_selection_correct
                selection_detail["validation_status"] = "Validated"

            else:
                # Results not available (either not finished or API/DB error)
                print(f"Skipping validation for fixture {fixture_id} in paper {paper_id} due to missing/incomplete results.")
                # We don't store a result in validation_results if it couldn't be validated
                # validation_results[paper_id][fixture_id] = False # Removed: Only store actual validation outcomes
                selection_detail["validation_status"] = "Skipped - No Results"
                # Keep league/team as Unknown if results failed entirely

            all_selection_details.append(selection_detail) # Add details regardless of validation success

    print("Validation finished. Generating reports...")

    # --- 4. Generate Detailed JSON Report FIRST ---
    # Calculations are now done inside generate_detailed_report
    detailed_report_data = generate_detailed_report(papers_data, validation_results, all_selection_details)

    # --- 5. Generate Basic Console Report from the generated detailed data ---
    summary = detailed_report_data.get("portfolio_performance_summary", {})
    insights = detailed_report_data.get("performance_insights", [])
    portfolio_analysis = detailed_report_data.get("portfolio_analysis", {})
    
    # Safely get the net units with fallback
    net_units = "N/A"
    if isinstance(portfolio_analysis, dict) and "net_profit_units" in portfolio_analysis:
        net_units = f"{portfolio_analysis['net_profit_units']:+.2f}"
    
    print("\n--- Portfolio Performance Summary ---")
    print(f"Total Papers Processed: {summary.get('total_papers_processed', 'N/A')}")
    print(f"Total Papers Evaluated: {summary.get('total_papers_evaluated', 'N/A')}")
    print(f"Correct Papers: {summary.get('correct_papers', 'N/A')} ({summary.get('paper_win_rate', 'N/A')})")
    print(f"Total Selections Validated: {summary.get('total_selections_validated', 'N/A')}")
    print(f"Correct Selections: {summary.get('correct_selections', 'N/A')} ({summary.get('selection_win_rate', 'N/A')})")
    print(f"Failed Selections: {summary.get('failed_selections', 'N/A')}")
    print(f"Stake-Weighted ROI: {summary.get('stake_weighted_roi', 'N/A')}")
    print(f"Paper-Level ROI: {summary.get('paper_level_roi', 'N/A')}")
    print(f"Combined Portfolio ROI: {summary.get('combined_portfolio_roi', 'N/A')}")
    if net_units != "N/A":
        total_papers = portfolio_analysis.get('total_papers_in_portfolio', 0)
        print(f"TOTAL NET UNITS: {net_units} units from {total_papers} papers")
    print("------------------------------------")
    print("\n--- Key Insights ---")
    if insights:
        for insight in insights:
            print(f"- {insight}")
    else:
        print("No insights generated.")
    print("--------------------\n")


    # --- 6. Save Detailed JSON Report ---
    try:
        with open(report_path, 'w') as outfile:
            json.dump(detailed_report_data, outfile, indent=4, default=str) # Use default=str for safety (e.g., datetime)
        print(f"Detailed validation report saved to {report_path}")
    except Exception as e:
        print(f"Error saving detailed report to {report_path}: {e}")

    print("-------------------------\n")


if __name__ == "__main__":
    main()