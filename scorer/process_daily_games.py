import json
import os
import re
from typing import Dict, Any, Optional, Union

def clean_value(value: Any, target_type: str = 'float') -> Optional[Union[float, int, str]]:
    """
    Cleans raw stat values (often strings with '%') into numeric types.
    Handles potential None, empty strings, or empty dicts.
    """
    if value is None or value == "" or value == {}:
        return None

    if isinstance(value, (int, float)):
        return value # Already numeric

    if isinstance(value, str):
        # Remove percentage signs and whitespace
        cleaned_str = re.sub(r'\s*%|\s+', '', value)
        try:
            if target_type == 'float':
                # Convert percentages to decimals if needed
                if '%' in value:
                     # Handle cases where original string had '%' but was removed
                     # Or if the cleaned_str still implies percentage context
                     # This part might need adjustment based on how '%' was used
                     # Assuming '%' meant divide by 100 if it was present
                    return float(cleaned_str) / 100.0
                else:
                    return float(cleaned_str)
            elif target_type == 'int':
                # Use float conversion first for strings like "1.0" then int
                return int(float(cleaned_str))
            else: # Keep as string if specified, but cleaned
                return cleaned_str
        except (ValueError, TypeError):
            # If conversion fails, return the cleaned string or None
            return cleaned_str if cleaned_str else None
    
    # Handle other types if necessary, or return None by default
    return None


def extract_team_data(team_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and reorganizes relevant stats for a single team
    from the deeply nested JSON structure.
    """
    extracted = {}
    if not team_data:
        return extracted # Return empty if no team data provided

    extracted['id'] = team_data.get('id')
    extracted['name'] = team_data.get('name')
    extracted['logo'] = team_data.get('logo')

    # --- MongoDB Stats ---
    mongo_stats = team_data.get('mongodb_stats', {})
    if mongo_stats:
        extracted['form'] = mongo_stats.get('form') # Keep form as string

        # Fixtures
        fixtures = mongo_stats.get('fixtures', {})
        extracted['fixtures_played_home'] = fixtures.get('played', {}).get('home')
        extracted['fixtures_played_away'] = fixtures.get('played', {}).get('away')
        extracted['fixtures_played_total'] = fixtures.get('played', {}).get('total')
        extracted['fixtures_wins_home'] = fixtures.get('wins', {}).get('home')
        extracted['fixtures_wins_away'] = fixtures.get('wins', {}).get('away')
        extracted['fixtures_wins_total'] = fixtures.get('wins', {}).get('total')
        extracted['fixtures_draws_home'] = fixtures.get('draws', {}).get('home')
        extracted['fixtures_draws_away'] = fixtures.get('draws', {}).get('away')
        extracted['fixtures_draws_total'] = fixtures.get('draws', {}).get('total')
        extracted['fixtures_loses_home'] = fixtures.get('loses', {}).get('home')
        extracted['fixtures_loses_away'] = fixtures.get('loses', {}).get('away')
        extracted['fixtures_loses_total'] = fixtures.get('loses', {}).get('total')

        # Goals For
        goals_for = mongo_stats.get('goals', {}).get('for', {})
        extracted['goals_for_total_home'] = goals_for.get('total', {}).get('home')
        extracted['goals_for_total_away'] = goals_for.get('total', {}).get('away')
        extracted['goals_for_total'] = goals_for.get('total', {}).get('total')
        extracted['goals_for_avg_home'] = clean_value(goals_for.get('average', {}).get('home'))
        extracted['goals_for_avg_away'] = clean_value(goals_for.get('average', {}).get('away'))
        extracted['goals_for_avg_total'] = clean_value(goals_for.get('average', {}).get('total'))
        extracted['goals_for_minute'] = {
            minute: {
                'total': data.get('total'),
                'percentage': clean_value(data.get('percentage'))
            } for minute, data in goals_for.get('minute', {}).items() if data # Ensure data is not empty {}
        }

        # Goals Against
        goals_against = mongo_stats.get('goals', {}).get('against', {})
        extracted['goals_against_total_home'] = goals_against.get('total', {}).get('home')
        extracted['goals_against_total_away'] = goals_against.get('total', {}).get('away')
        extracted['goals_against_total'] = goals_against.get('total', {}).get('total')
        extracted['goals_against_avg_home'] = clean_value(goals_against.get('average', {}).get('home'))
        extracted['goals_against_avg_away'] = clean_value(goals_against.get('average', {}).get('away'))
        extracted['goals_against_avg_total'] = clean_value(goals_against.get('average', {}).get('total'))
        extracted['goals_against_minute'] = {
             minute: {
                'total': data.get('total'),
                'percentage': clean_value(data.get('percentage'))
            } for minute, data in goals_against.get('minute', {}).items() if data
        }

        # Biggest Streaks, Wins, Losses
        biggest = mongo_stats.get('biggest', {})
        extracted['biggest_streak_wins'] = biggest.get('streak', {}).get('wins')
        extracted['biggest_streak_draws'] = biggest.get('streak', {}).get('draws')
        extracted['biggest_streak_loses'] = biggest.get('streak', {}).get('loses')
        extracted['biggest_win_home'] = biggest.get('wins', {}).get('home')
        extracted['biggest_win_away'] = biggest.get('wins', {}).get('away')
        extracted['biggest_loss_home'] = biggest.get('loses', {}).get('home')
        extracted['biggest_loss_away'] = biggest.get('loses', {}).get('away')

        # Performance
        performance = mongo_stats.get('performance', {})
        extracted['clean_sheet_home'] = performance.get('clean_sheet', {}).get('home')
        extracted['clean_sheet_away'] = performance.get('clean_sheet', {}).get('away')
        extracted['clean_sheet_total'] = performance.get('clean_sheet', {}).get('total')
        extracted['failed_to_score_home'] = performance.get('failed_to_score', {}).get('home')
        extracted['failed_to_score_away'] = performance.get('failed_to_score', {}).get('away')
        extracted['failed_to_score_total'] = performance.get('failed_to_score', {}).get('total')
        # Lineups could be added if needed, but often less critical for top-level stats
        # extracted['lineups'] = performance.get('lineups')


    # --- StatArea Analysis ---
    # Extract key summarized metrics, especially the time-windowed ones
    statarea = team_data.get('statarea_analysis', {})
    if statarea:
        extracted['statarea_match_history'] = statarea.get('match_history', [])

        # Simplified structure for recent form analysis (5, 10, 15 games)
        extracted['statarea_analysis'] = {}
        for window in [5, 10, 15]:
            window_key = f'last_{window}_games'
            extracted['statarea_analysis'][window_key] = {
                'home': {}, # Host perspective for this team
                'away': {}  # Guest perspective for this team
            }
            # Home/Host Stats for this team
            host_stats = statarea.get('raw_stats', {}).get(f'host_{window}', {})
            if host_stats:
                 extracted['statarea_analysis'][window_key]['home'] = {
                     'avg_scored': clean_value(host_stats.get('Average scored goals per match')),
                     'avg_conceded': clean_value(host_stats.get('Average conceded goals per match')),
                     'chance_to_score': clean_value(host_stats.get('Chance to score goal next match')),
                     'chance_to_concede': clean_value(host_stats.get('Chance to conceded goal next match')),
                     'over_2_5_matches': host_stats.get('Matches over 2.5 goals in'),
                     'under_2_5_matches': host_stats.get('Matches under 2.5 goals in'),
                     'clean_sheets': host_stats.get('Number of clean sheet matches'),
                     'failed_to_score': host_stats.get('Failure to score matches'),
                     # Add more key stats from statarea if needed
                 }

            # Away/Guest Stats for this team
            guest_stats = statarea.get('raw_stats', {}).get(f'guest_{window}', {})
            if guest_stats:
                 extracted['statarea_analysis'][window_key]['away'] = {
                     'avg_scored': clean_value(guest_stats.get('Average scored goals per match')),
                     'avg_conceded': clean_value(guest_stats.get('Average conceded goals per match')),
                     'chance_to_score': clean_value(guest_stats.get('Chance to score goal next match')),
                     'chance_to_concede': clean_value(guest_stats.get('Chance to conceded goal next match')),
                     'over_2_5_matches': guest_stats.get('Matches over 2.5 goals in'),
                     'under_2_5_matches': guest_stats.get('Matches under 2.5 goals in'),
                     'clean_sheets': guest_stats.get('Number of clean sheet matches'),
                     'failed_to_score': guest_stats.get('Failure to score matches'),
                     # Add more key stats from statarea if needed
                 }
            
            # Add the summary analysis if present
            summary_analysis = statarea.get(f'analysis_{window}_games')
            if summary_analysis:
                 extracted['statarea_analysis'][window_key]['summary'] = {
                    'chance_to_score': clean_value(summary_analysis.get('chance_to_score')),
                    'chance_to_concede': clean_value(summary_analysis.get('chance_to_concede')),
                 }


    return extracted


def organize_fixture_data(home_json_path: str, away_json_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads home and away team JSON data, organizes it, and returns a combined dictionary.
    """
    try:
        with open(home_json_path, 'r') as f:
            home_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Home team JSON file not found at {home_json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {home_json_path}")
        return None

    try:
        with open(away_json_path, 'r') as f:
            away_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Away team JSON file not found at {away_json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {away_json_path}")
        return None

    # --- Basic Fixture Info (Assume same in both files, take from home) ---
    fixture_info = home_data.get('fixture_info', {})
    league_info = home_data.get('league', {})

    # --- Extract and Organize Team Data ---
    home_team_organized = extract_team_data(home_data.get('teams', {}).get('home', {}))
    away_team_organized = extract_team_data(away_data.get('teams', {}).get('away', {})) # Assumes away file has structure like teams.away

    # --- Combine into Final Structure ---
    organized_data = {
        "fixture_info": fixture_info,
        "league": league_info,
        "teams": {
            "home": home_team_organized,
            "away": away_team_organized
        }
    }

    return organized_data

def save_json_data(data: Dict[str, Any], output_filepath: str):
    """Saves the organized data dictionary to a JSON file."""
    try:
        with open(output_filepath, 'w') as f:
            json.dump(data, f, indent=4) # indent=4 makes the file human-readable
        print(f"Successfully organized data saved to: {output_filepath}")
    except IOError as e:
        print(f"Error saving data to {output_filepath}: {e}")
    except TypeError as e:
         print(f"Error: Data contains non-serializable types. {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Process all JSON files in daily_games directory ---
    daily_games_dir = "daily_games"
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(daily_games_dir):
        for file in files:
            # Skip standings.json files
            if file == "standings.json":
                continue
                
            # Only process JSON files
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                
                try:
                    with open(json_path, 'r') as f:
                        game_data = json.load(f)
                        
                    # Process the game data
                    organized_data = organize_fixture_data(game_data)
                    
                    # Create output filename based on original
                    output_file = os.path.join('processed', file)
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    
                    # Save processed data
                    if organized_data:
                        save_json_data(organized_data, output_file)
                    else:
                        print(f"Failed to organize data for {file}")
                        
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")