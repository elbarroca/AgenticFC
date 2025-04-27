# path: ValidatePapers.py
import json
import datetime
from typing import Dict, Any, List, Tuple, Optional

# Placeholder for fetching actual match results.
# This needs to be implemented based on your data source (e.g., API client, database).
def get_match_results(fixture_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the actual results for a given fixture ID.

    Args:
        fixture_id: The unique identifier for the match.

    Returns:
        A dictionary containing match results (e.g., final score, half-time score)
        or None if results are not available.
    """
    # Placeholder implementation: Replace with actual data fetching logic.
    # Example structure of returned data:
    # return {
    #     "fixture_id": fixture_id,
    #     "status": "Match Finished",
    #     "score_fulltime": {"home": 2, "away": 1},
    #     "score_halftime": {"home": 1, "away": 0},
    #     # Add other relevant details like goals, cards, etc.
    # }
    print(f"Warning: Fetching results for fixture {fixture_id} is not implemented.")
    return None

# Placeholder for validating a single selection against results.
# This needs detailed logic for each bet type.
def validate_selection(selection_data: Dict[str, Any], match_results: Dict[str, Any]) -> bool:
    """
    Validates if a betting selection was correct based on match results.

    Args:
        selection_data: The dictionary representing a single bet selection.
        match_results: The dictionary containing the actual match results.

    Returns:
        True if the selection was correct, False otherwise.
    """
    bet_type = selection_data.get("selection")
    if not bet_type or not match_results:
        return False # Cannot validate without bet type or results

    score_ft = match_results.get("score_fulltime")
    if not score_ft:
        print(f"Warning: Fulltime score missing for fixture {match_results.get('fixture_id')}")
        return False # Cannot validate without score

    home_goals_ft = score_ft.get("home")
    away_goals_ft = score_ft.get("away")
    total_goals_ft = home_goals_ft + away_goals_ft

    # --- Implement validation logic for each bet type ---
    if bet_type == "1X":
        return home_goals_ft >= away_goals_ft
    elif bet_type == "BTTS Yes":
        return home_goals_ft > 0 and away_goals_ft > 0
    elif bet_type == "O0.5":
        return total_goals_ft > 0
    elif bet_type == "12":
        return home_goals_ft != away_goals_ft
    elif bet_type == "U3.5":
        return total_goals_ft < 3.5 # i.e., 0, 1, 2, or 3 goals
    # --- Add logic for other bet types (e.g., "1", "X", "2", "O1.5", "U2.5", combined bets) ---
    elif " and " in bet_type: # Handle combined bets (simple example)
        # Example: "1X and U3.5"
        parts = bet_type.split(" and ")
        # This requires a more robust parsing and validation logic
        # For now, just a placeholder:
        print(f"Warning: Validation for combined bet '{bet_type}' not fully implemented.")
        # Example for "1X and U3.5":
        if parts == ["1X", "U3.5"]:
             return (home_goals_ft >= away_goals_ft) and (total_goals_ft < 3.5)
        return False # Default for unimplemented combined bets
    else:
        print(f"Warning: Validation logic for bet type '{bet_type}' is not implemented.")
        return False # Default for unknown bet types

    return False # Should not be reached if logic is complete


def calculate_roi(papers_data: List[Dict[str, Any]], validation_results: Dict[str, Dict[str, bool]]) -> float:
    """
    Calculates the overall Return on Investment based on validated selections and stakes.

    Args:
        papers_data: The list of paper data from the JSON.
        validation_results: A dict mapping paper_id to selection validation results.
                            e.g., {"Paper_20": {"1361685": True, "1372293": False}}

    Returns:
        The calculated ROI (e.g., 0.05 for 5% ROI, -0.1 for -10% ROI).
    """
    total_staked = 0.0
    total_returned = 0.0

    for paper in papers_data:
        paper_id = paper.get("paper_id")
        selections = paper.get("staked_selections", [])
        paper_validation = validation_results.get(paper_id, {})

        for selection in selections:
            fixture_id = selection.get("fixture_id")
            stake_fraction = float(selection.get("optimal_stake_fraction", 0.0))
            odds = float(selection.get("odds", 1.0))
            is_correct = paper_validation.get(fixture_id, False)

            # Assuming stake_fraction is relative to a total bankroll (e.g., 1 unit)
            # Only consider stakes deemed meaningful
            if selection.get("has_meaningful_stake", False):
                 total_staked += stake_fraction
                 if is_correct:
                     total_returned += stake_fraction * odds # Return includes original stake

    if total_staked == 0:
        return 0.0

    # ROI = (Total Returned - Total Staked) / Total Staked
    roi = (total_returned - total_staked) / total_staked
    return roi


def main(json_path: str = "data/output/optimized_game_portfolios.json"):
    """
    Main function to load data, validate papers/selections, and generate a report.
    """
    try:
        with open(json_path, 'r') as f:
            papers_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return

    total_papers = 0
    total_selections = 0
    correct_selections = 0
    correct_papers = 0
    processed_fixtures = set() # Avoid fetching results for the same fixture multiple times

    # Store validation results: {paper_id: {fixture_id: bool}}
    validation_results: Dict[str, Dict[str, bool]] = {}
    # Store fetched results: {fixture_id: results_dict}
    match_results_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    print("Starting validation process...")

    for paper in papers_data:
        paper_id = paper.get("paper_id")
        if not paper_id:
            continue

        total_papers += 1
        selections = paper.get("staked_selections", [])
        validation_results[paper_id] = {}
        is_paper_correct = True # Assume correct until a selection fails

        for selection in selections:
            fixture_id = selection.get("fixture_id")
            if not fixture_id:
                continue

            total_selections += 1

            # Fetch results if not already cached
            if fixture_id not in processed_fixtures:
                 match_results = get_match_results(fixture_id)
                 match_results_cache[fixture_id] = match_results
                 processed_fixtures.add(fixture_id)
            else:
                 match_results = match_results_cache[fixture_id]

            if match_results:
                is_selection_correct = validate_selection(selection, match_results)
                validation_results[paper_id][fixture_id] = is_selection_correct
                if is_selection_correct:
                    correct_selections += 1
                else:
                    is_paper_correct = False # One wrong selection makes the paper wrong
            else:
                print(f"Skipping validation for fixture {fixture_id} in paper {paper_id} due to missing results.")
                is_paper_correct = False # Cannot validate paper if results are missing

        if is_paper_correct and selections: # Paper only correct if all selections validated and correct
             correct_papers += 1

    print("Validation finished. Generating report...")
    print("\n--- Validation Report ---")
    print(f"Total Papers Processed: {total_papers}")
    print(f"Total Selections Processed: {total_selections}")
    print(f"Correct Papers: {correct_papers} ({correct_papers / total_papers:.2%} win rate)")
    print(f"Correct Selections: {correct_selections} ({correct_selections / total_selections:.2%} win rate)")

    # Calculate ROI based on meaningful stakes
    calculated_roi = calculate_roi(papers_data, validation_results)
    print(f"Calculated ROI (based on optimal stakes): {calculated_roi:.2%}")
    print("-------------------------\n")

    # Optional: Save detailed results to a file
    # with open("validation_output.json", 'w') as outfile:
    #     json.dump(validation_results, outfile, indent=4)
    # print("Detailed validation results saved to validation_output.json")


if __name__ == "__main__":
    # You can change the path here if your JSON file is located elsewhere
    main()