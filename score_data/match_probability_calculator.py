import json
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from collections import defaultdict
import warnings
from typing import Dict, List, Tuple, Optional
import math
from datetime import datetime
import re
import sys

# Suppress pandas warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Add parent directory to path so we can import get_data module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from get_data.db_ids.league_id_mappings import LEAGUE_ID_MAPPING, DIRECTORY_TO_LEAGUE_MAPPING
except ModuleNotFoundError:
    print("Warning: league_id_mappings module not found, creating fallback dictionaries")
    # Create fallback dictionaries based on directory structure
    LEAGUE_ID_MAPPING = {}
    DIRECTORY_TO_LEAGUE_MAPPING = {}
    
    # Populate from directory names in daily_output/daily_games
    daily_games_dir = "../daily_output/daily_games"
    if os.path.exists(daily_games_dir):
        for dir_name in os.listdir(daily_games_dir):
            if os.path.isdir(os.path.join(daily_games_dir, dir_name)):
                league_name = dir_name.replace("_", " ")
                DIRECTORY_TO_LEAGUE_MAPPING[dir_name] = {
                    "name": league_name,
                    "directory_name": dir_name,
                    "form_chars": ["W", "D", "L"]
                }
                LEAGUE_ID_MAPPING[league_name] = {
                    "directory_name": dir_name,
                    "form_chars": ["W", "D", "L"]
                }

# --- Configuration ---
NUM_GAMES = 15  # This is now explicitly defined and will be used consistently
NUM_GAMES_OPTIONS = [15, 10, 5]  # Options for different analysis windows

# Add documentation for important metrics
METRIC_DESCRIPTIONS = {
    "points_per_game": "Average number of points earned per match (3 for win, 1 for draw, 0 for loss)",
    "form_trend": "Direction and strength of recent performance trend (positive values indicate improvement)",
    "consistency_score": "How predictable a team's results are (0-1 scale, higher means more consistent)",
    "points_momentum": "Difference in points between recent matches and earlier matches",
    "points_volatility": "Variability in match results (higher values indicate inconsistency)",
    "xG_per_game": "Expected goals per game based on shot quality and historical data",
    "xG_vs_actual": "Difference between actual goals scored and expected goals (measure of finishing efficiency)",
    "correlation_with_opponent_rank": "Statistical relationship between opponent quality and team performance",
    "perf_vs_top_teams": "Points per game against teams in the top 6 positions",
    "perf_vs_mid_teams": "Points per game against teams in positions 7-12",
    "perf_vs_bottom_teams": "Points per game against teams in positions 13 and below"
}

STANDINGS_FILENAME = "standings.json"
RELEVANT_COMPETITION_KEYWORDS = ["Bundesliga", "League", "Cup", "Serie", "Primera", "Ligue", "Ekstraklasa", "Championship", "Division"]
DAILY_GAMES_DIR = "../daily_output/daily_games"
OUTPUT_DIR = "../daily_output/processed_matches"
DATE_PATTERN = r"(\d{4}-\d{2}-\d{2})_"

# --- Helper Functions ---

def load_json(file_path: str) -> Dict:
    """Loads JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def safe_get(data: Dict, keys, default=None):
    """Safely access nested dictionary keys."""
    if not isinstance(keys, list):
        keys = keys.split('.')
    temp = data
    try:
        for key in keys:
            if isinstance(temp, dict):
                temp = temp.get(key, None)
                if temp is None:
                    return default
            else:
                return default
        return temp
    except:
        return default

def parse_percentage(value) -> Optional[float]:
    """Converts percentage string ('75.00%') to float (0.75). Handles errors."""
    if isinstance(value, (int, float)):
        return float(value) / 100.0 if value > 1 else float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace('%', '')) / 100.0
        except ValueError:
            return None
    return None

def get_team_standing(standings_data: Dict, team_id=None, team_name=None) -> Dict:
    """Finds team standing info (rank, points, form) by ID or Name."""
    if not standings_data or 'standings' not in standings_data or not standings_data['standings']:
        return {'rank': None, 'points': None, 'name': None, 'form': None, 'goalsDiff': None}

    # Assuming the first list in 'standings' is the relevant one
    league_table = standings_data['standings'][0]

    for entry in league_table:
        s_team_id = safe_get(entry, ['team', 'id'])
        s_team_name = safe_get(entry, ['team', 'name'])
        s_rank = safe_get(entry, ['rank'])
        s_points = safe_get(entry, ['points'])
        s_form = safe_get(entry, ['form'])
        s_goalsDiff = safe_get(entry, ['goalsDiff'])
        s_goals_for = safe_get(entry, ['all', 'goals', 'for'])
        s_goals_against = safe_get(entry, ['all', 'goals', 'against'])

        match = False
        if team_id is not None and s_team_id == team_id:
            match = True
        elif team_name is not None and s_team_name and team_name.lower() == s_team_name.lower():
             match = True

        if match:
            return {
                'rank': s_rank, 
                'points': s_points, 
                'name': s_team_name,
                'form': s_form,
                'goalsDiff': s_goalsDiff,
                'goals_for': s_goals_for,
                'goals_against': s_goals_against
            }

    fallback_name = team_name if team_name else f"ID_{team_id}"
    return {'rank': None, 'points': None, 'name': fallback_name, 'form': None, 'goalsDiff': None}

def calculate_volatility(series: pd.Series) -> float:
    """Calculates standard deviation as a measure of volatility."""
    return series.std() if len(series) > 1 else 0.0

def calculate_momentum(series: pd.Series, window: int = 5) -> Optional[float]:
    """Calculates momentum based on recent performance vs. previous window."""
    if len(series) < window * 2:
        return None
    recent_mean = series.head(window).mean()
    prior_mean = series.iloc[window:window*2].mean()
    
    if pd.isna(recent_mean) or pd.isna(prior_mean):
        return None
    return recent_mean - prior_mean

def calculate_trend(series: pd.Series) -> Tuple[float, float, float]:
    """
    Calculate linear trend from time series data.
    Returns slope, intercept, and r-squared value.
    """
    if len(series) < 3:
        return 0, 0, 0
        
    y = series.values
    x = np.arange(len(y))
    
    # Simple linear regression: y = mx + b
    if np.all(np.isnan(y)):
        return 0, 0, 0
        
    # Remove NaN values for the calculation
    mask = ~np.isnan(y)
    if sum(mask) < 2:
        return 0, 0, 0
        
    x_clean = x[mask]
    y_clean = y[mask]
    
    try:
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        # Calculate R-squared
        y_pred = slope * x_clean + intercept
        ss_total = np.sum((y_clean - np.mean(y_clean))**2)
        ss_residual = np.sum((y_clean - y_pred)**2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        return slope, intercept, r_squared
    except:
        return 0, 0, 0

def extract_form_pattern(form_str: str) -> Dict:
    """
    Analyze form string (e.g., "WWLDW") to extract patterns and streaks.
    Returns metrics about current form.
    """
    if not form_str:
        return {
            'current_streak_type': None,
            'current_streak_length': 0,
            'win_percentage': 0,
            'form_score': 0,
            'form_volatility': 0
        }
    
    # Map form letters to points (extended for different notations)
    form_map = {
        'W': 3, 'D': 1, 'L': 0,
    }
    
    # Convert form string to points
    points = []
    for c in form_str.upper():
        if c in form_map:
            points.append(form_map[c])
    
    if not points:
        return {
            'current_streak_type': None,
            'current_streak_length': 0,
            'win_percentage': 0,
            'form_score': 0,
            'form_volatility': 0
        }
    
    # Map first character to standard form letter
    form_to_standard = {
        'W': 'W', 'V': 'W', 'G': 'W', 'Z': 'W', 'В': 'W',
        'D': 'D', 'N': 'D', 'E': 'D', 'R': 'D', 'Р': 'D', 
        'L': 'L', 'U': 'L', 'P': 'L', 'П': 'L'
    }
    
    # Calculate current streak
    current_result = form_to_standard.get(form_str[0].upper(), None) if form_str else None
    streak_length = 0
    
    if current_result:
        for c in form_str:
            c_upper = c.upper()
            if form_to_standard.get(c_upper) == current_result:
                streak_length += 1
            else:
                break
    
    # Calculate form metrics
    win_chars = [k for k, v in form_to_standard.items() if v == 'W']
    win_count = sum(form_str.upper().count(c) for c in win_chars)
    win_percentage = win_count / len(form_str) if form_str else 0
    
    # Calculate form score (weighted recent performances)
    form_score = sum(p * (i+1) for i, p in enumerate(reversed(points))) / sum(range(1, len(points)+1)) if points else 0
    
    # Calculate form volatility
    form_volatility = np.std(points) if len(points) > 1 else 0
    
    return {
        'current_streak_type': current_result,
        'current_streak_length': streak_length,
        'win_percentage': win_percentage,
        'form_score': form_score,
        'form_volatility': form_volatility
    }

def calculate_expected_goals(team_goals, opponent_quality, shot_data=None, league_id=None, league_name=None, venue=None):
    """
    Calculate expected goals (xG) based on dynamic league metrics.
    
    Args:
        team_goals: List of actual goals scored in previous matches
        opponent_quality: List of opponent strength metrics (ranks/points)
        shot_data: Optional shot quality data
        league_id: League ID for league-specific metrics
        league_name: League name for league-specific metrics
        venue: 'home' or 'away' to account for venue advantage
        
    Returns:
        xG value representing expected goals per game
    """
    # Load league-specific metrics
    league_metrics = load_league_metrics(league_id, league_name)
    
    # If no shot data, use enhanced model with dynamic parameters
    if not team_goals or not opponent_quality or len(team_goals) != len(opponent_quality):
        return None
    
    # Filter out missing data
    valid_idx = [i for i, q in enumerate(opponent_quality) if q is not None]
    if not valid_idx:
        return sum(team_goals) / len(team_goals) if team_goals else None
    
    filtered_goals = [team_goals[i] for i in valid_idx]
    filtered_quality = [opponent_quality[i] for i in valid_idx]
    
    # Get league average goals (dynamic)
    league_avg_goals = league_metrics['avg_goals_per_game_per_team']
    
    # Apply Bayesian adjustment toward league mean with dynamic weighting
    n_games = len(filtered_goals)
    max_weight = league_metrics['max_team_data_weight']
    weight = min(max_weight, n_games / 10)
    
    team_avg = sum(filtered_goals) / n_games if n_games > 0 else 0
    
    # Adjust based on opponent quality with dynamic parameters
    quality_factor = 1.0
    if filtered_quality and len(filtered_quality) > 0:
        # Convert ranks to strength (lower rank = higher strength)
        avg_rank = sum(filtered_quality) / len(filtered_quality)
        # Dynamically determine max rank based on data
        max_rank = 20  # Default
        unique_ranks = set(r for r in filtered_quality if r is not None)
        if unique_ranks:
            max_rank = max(unique_ranks)
        
        # Normalize ranks to 0-1 scale
        normalized_opp_strength = max(0, min(1, (max_rank - avg_rank) / (max_rank - 1))) if max_rank > 1 else 0.5
        
        # Adjust quality factor using league-specific parameters
        quality_adjustment_low = league_metrics['quality_adjustment_low']
        quality_adjustment_high = league_metrics['quality_adjustment_high']
        quality_factor = quality_adjustment_low + (quality_adjustment_high * (1 - normalized_opp_strength))
    
    # Apply home/away adjustment if specified
    venue_factor = 1.0
    if venue == 'home':
        venue_factor = 1.0 + league_metrics['home_advantage']
    elif venue == 'away':
        venue_factor = 1.0 - league_metrics['home_advantage']
    
    # Combine team data and league average with quality and venue adjustments
    xG = ((team_avg * weight * quality_factor * venue_factor) + 
          (league_avg_goals * (1 - weight)))
    
    return round(xG, 2)

def analyze_scoring_patterns(data: Dict, team_name: str) -> Dict:
    """
    Analyze a team's scoring patterns by time interval.
    Identifies when the team tends to score or concede goals.
    """
    result = {
        'scoring_by_interval': {},
        'conceding_by_interval': {},
        'key_patterns': []
    }
    
    # Extract goal timing data from statarea stats
    sa_raw = safe_get(data, ['teams', 'home', 'statarea_analysis', 'raw_stats'], {})
    
    # Start with 15-game sample for best data reliability
    period = 'host_15' if 'host_15' in sa_raw else 'guest_15'
    
    if period not in sa_raw:
        return result
    
    # Extract team's scoring pattern
    team_goals = safe_get(sa_raw, [period, f'{team_name} goals only'], {})
    all_goals = safe_get(sa_raw, [period, 'All goals in matches'], {})
    
    # Parse percentages for each time interval
    for interval, pct_str in team_goals.items():
        if isinstance(pct_str, str) and '%' in pct_str:
            result['scoring_by_interval'][interval] = parse_percentage(pct_str)
    
    # Infer conceding pattern by comparing all goals to team goals
    if team_goals and all_goals:
        for interval, all_pct_str in all_goals.items():
            all_pct = parse_percentage(all_pct_str)
            team_pct = result['scoring_by_interval'].get(interval, 0)
            
            # Estimate opponent goals percentage
            if all_pct is not None:
                # Calculate what portion of goals in this interval are from opponents
                # This is a rough estimation based on the total goals vs team goals
                opp_weight = 0.5  # Assuming roughly equal distribution as a starting point
                opp_pct = (all_pct - (team_pct * opp_weight)) / (1 - opp_weight) if team_pct > 0 else all_pct
                result['conceding_by_interval'][interval] = max(0, opp_pct)
    
    # Identify key patterns
    if result['scoring_by_interval']:
        max_scoring = max(result['scoring_by_interval'].items(), key=lambda x: x[1])
        if max_scoring[1] > 0.25:  # If more than 25% of goals come in one period
            result['key_patterns'].append(f"Team scores {max_scoring[1]*100:.0f}% of their goals in the {max_scoring[0]}")
    
    if result['conceding_by_interval']:
        max_conceding = max(result['conceding_by_interval'].items(), key=lambda x: x[1])
        if max_conceding[1] > 0.25:  # If more than 25% of conceded goals come in one period
            result['key_patterns'].append(f"Team concedes {max_conceding[1]*100:.0f}% of goals in the {max_conceding[0]}")
    
    # Calculate if team is a fast or slow starter
    early_scoring = sum(v for k, v in result['scoring_by_interval'].items() if '0-15' in k or '16-30' in k)
    late_scoring = sum(v for k, v in result['scoring_by_interval'].items() if '61-75' in k or '76-90' in k)
    
    if early_scoring > 0.4:
        result['key_patterns'].append("Fast starter: Team scores 40%+ of goals in the first 30 minutes")
    elif late_scoring > 0.4:
        result['key_patterns'].append("Strong finisher: Team scores 40%+ of goals in the final 30 minutes")
    
    return result

def calculate_consistency_score(results: List[str]) -> float:
    """
    Calculate consistency score based on how predictable a team's results are.
    Higher score means more consistent results pattern.
    """
    if not results:
        return 0.0
    
    # Count transitions between results
    transitions = defaultdict(int)
    for i in range(len(results) - 1):
        transitions[f"{results[i]}_to_{results[i+1]}"] += 1
    
    # Calculate entropy of transitions (lower entropy = more predictable)
    total_transitions = len(results) - 1
    if total_transitions == 0:
        return 0.0
    
    entropy = 0
    for count in transitions.values():
        p = count / total_transitions
        entropy -= p * math.log2(p) if p > 0 else 0
    
    # Normalize to 0-1 scale (1 = most consistent)
    # Max entropy would be when all possible transitions are equally likely
    max_possible_entropy = math.log2(min(9, total_transitions))  # 3 results (W/D/L) can have 9 transitions
    
    # Invert and normalize: 1 = perfectly consistent, 0 = completely random
    consistency = 1 - (entropy / max_possible_entropy) if max_possible_entropy > 0 else 0
    
    return consistency

def analyze_team_performance(team_data, standings_parent_path):
    """
    Analyzes team performance using dynamic league-specific parameters.
    """
    analysis_results = {}
    
    # Try to find the standings file
    # First check for date-specific standings
    date_match = None
    for file in os.listdir(standings_parent_path):
        if file.endswith("standings.json"):
            date_match = re.search(DATE_PATTERN, file)
            if date_match:
                standings_path = os.path.join(standings_parent_path, file)
                break
    
    # If no date-specific standings found, use the default standings filename
    if not date_match:
        standings_path = os.path.join(standings_parent_path, STANDINGS_FILENAME)
    
    if not os.path.exists(standings_path):
        print(f"Warning: Standings file not found at {standings_path}")
        # Try to find any standings file in the directory
        standings_files = [f for f in os.listdir(standings_parent_path) if f.endswith("standings.json")]
        if standings_files:
            standings_path = os.path.join(standings_parent_path, standings_files[0])
            print(f"Using alternative standings file: {standings_files[0]}")
        else:
            print(f"No standings files found in {standings_parent_path}")
    
    standings_data = load_json(standings_path)

    if not team_data:
        print("Error: No team data provided.")
        return None
        
    if not standings_data:
        print(f"Error: Could not load standings data from {standings_path}. Cannot perform opponent analysis.")
        analysis_results['warnings'] = ["Standings data unavailable."]

    # --- 1. Extract Basic Team Info & Schema ---
    team_id = safe_get(team_data, ['teams', 'home', 'id'])
    team_name = safe_get(team_data, ['teams', 'home', 'name'])
    league_id = safe_get(team_data, ['league', 'id'])
    league_name = safe_get(team_data, ['league', 'name'])
    
    # Get team's own standing information
    team_standing = get_team_standing(standings_data, team_id=team_id, team_name=team_name)

    analysis_results['team_info'] = {
        'id': team_id,
        'name': team_name,
        'league_id': league_id,
        'league_name': league_name,
        'current_rank': team_standing['rank'],
        'current_points': team_standing['points'],
        'form': team_standing['form'],
        'goals_diff': team_standing['goalsDiff']
    }

    # --- 2. Process Match History (Last N Relevant Games) ---
    raw_match_history = []

    # Try multiple paths for match history data
    match_history_paths = [
        ['teams', 'home', 'statarea_analysis', 'match_history'],
        ['teams', 'home', 'match_history'],
        ['teams', 'home', 'recent_matches'],
        ['match_history']
    ]

    for path in match_history_paths:
        history_data = safe_get(team_data, path, [])
        if history_data and isinstance(history_data, list) and len(history_data) > 0:
            raw_match_history = history_data
            break

    # Attempt to build match history from fixtures data if still empty
    if not raw_match_history:
        fixtures_data = safe_get(team_data, ['teams', 'home', 'fixtures', 'played'], [])
        if fixtures_data and isinstance(fixtures_data, list):
            for fixture in fixtures_data:
                # Extract match details
                fixture_id = fixture.get('id')
                opponent_id = fixture.get('opponent', {}).get('id')
                opponent_name = fixture.get('opponent', {}).get('name', 'Unknown')
                venue = fixture.get('venue', 'unknown')
                competition = fixture.get('competition', {}).get('name', 'Unknown')
                result = fixture.get('result', '')
                team_goals = fixture.get('goals', {}).get('for', 0)
                opponent_goals = fixture.get('goals', {}).get('against', 0)
                
                raw_match_history.append({
                    'date': fixture.get('date', ''),
                    'opponent': opponent_name,
                    'team_goals': team_goals,
                    'opponent_goals': opponent_goals,
                    'result': result.lower(),
                    'venue': venue,
                    'competition': competition,
                    'fixture_id': fixture_id,
                    'opponent_id': opponent_id
                })

    # Make competition filtering optional if too few matches are found
    processed_matches = []
    strict_filtering = True
    competition_keywords = list(RELEVANT_COMPETITION_KEYWORDS)  # Make a copy

    while len(processed_matches) < 5 and competition_keywords:
        processed_matches = []
        for match in raw_match_history:
            competition = str(match.get('competition', '')).lower()
            
            # Skip competition filtering if not in strict mode
            if strict_filtering and not any(keyword.lower() in competition for keyword in competition_keywords):
                continue
                
            opponent_name = match.get('opponent')
            if not opponent_name:
                continue

            # Get opponent standing information
            opponent_standing = {'rank': None, 'points': None}
            if standings_data:
                opponent_standing = get_team_standing(standings_data, team_name=opponent_name)

            # Add more comprehensive data extraction
            match_data = {
                'date': match.get('date'),
                'opponent': opponent_name,
                'opponent_rank': opponent_standing['rank'],
                'opponent_points': opponent_standing['points'],
                'team_goals': match.get('team_goals'),
                'opponent_goals': match.get('opponent_goals'),
                'result': match.get('result'),
                'venue': match.get('venue'),
                'competition': match.get('competition'),
                'goal_diff': safe_get(match, 'team_goals', 0) - safe_get(match, 'opponent_goals', 0),
                'points_earned': 3 if match.get('result') == 'win' else (1 if match.get('result') == 'draw' else 0),
                'opponent_strength': opponent_standing.get('points', 0),  # Use points as strength indicator
            }
            
            # Add goal timing data if available
            if 'goal_timing' in match:
                match_data['goal_timing'] = match['goal_timing']
            
            # Extract half-time goals if available
            if 'halftime' in match:
                match_data['halftime_team_goals'] = match.get('halftime', {}).get('team_goals', None)
                match_data['halftime_opponent_goals'] = match.get('halftime', {}).get('opponent_goals', None)
            
            processed_matches.append(match_data)
        
        # If we didn't get enough matches with current filtering, relax filtering
        if len(processed_matches) < 5:
            if strict_filtering:
                strict_filtering = False
            else:
                # Progressively remove competition keywords to allow more matches
                if competition_keywords:
                    competition_keywords.pop()  # Remove one keyword
                else:
                    break  # No more keywords to remove, exit the loop
        else:
            break  # We have enough matches, exit the loop

    if not processed_matches:
        print(f"Warning: No relevant matches found in the last {len(raw_match_history)} entries.")
        analysis_results.setdefault('warnings', []).append(f"No relevant matches found for last {NUM_GAMES} game analysis.")
        
        # Add fallback analysis using available data even without match history
        fallback_metrics = extract_fallback_metrics(team_data)
        if fallback_metrics:
            analysis_results['performance_metrics_last_n'] = fallback_metrics
            analysis_results.setdefault('warnings', []).append("Using limited statistics from available data.")
        
        return analysis_results

    matches_df = pd.DataFrame(processed_matches)
    analysis_results['last_n_games_details'] = matches_df.to_dict(orient='records')
    actual_games_analyzed = len(matches_df)
    analysis_results['actual_games_analyzed'] = actual_games_analyzed

    # --- 3. Advanced Performance Metrics Calculation ---
    perf_metrics = {}

    # Overall performance metrics
    victories = matches_df[matches_df['result'] == 'win'].shape[0]
    draws = matches_df[matches_df['result'] == 'draw'].shape[0] 
    defeats = matches_df[matches_df['result'] == 'loss'].shape[0]
    
    # Create results sequence for pattern analysis
    results_sequence = matches_df['result'].map({'win': 'W', 'draw': 'D', 'loss': 'L'}).tolist()
    results_sequence_str = ''.join(results_sequence)
    
    # Calculate Points Trends (regression analysis)
    points_slope, points_intercept, points_r2 = calculate_trend(matches_df['points_earned'])
    goal_diff_slope, goal_diff_intercept, goal_diff_r2 = calculate_trend(matches_df['goal_diff'])
    
    # Calculate Expected Goals metrics
    xG = calculate_expected_goals(
        matches_df['team_goals'].tolist(),
        matches_df['opponent_rank'].tolist()
    )
    
    # Consistency score based on predictability of results
    consistency_score = calculate_consistency_score(results_sequence)
    
    # Analyze form pattern from standings data
    form_analysis = extract_form_pattern(team_standing.get('form', ''))
    
    # Points per game ratio (last 5 vs last 15)
    ppg_recent = matches_df.head(5)['points_earned'].mean() if len(matches_df) >= 5 else None
    ppg_overall = matches_df['points_earned'].mean()
    ppg_ratio = ppg_recent / ppg_overall if ppg_overall and ppg_recent is not None else None
    
    # Performance against different levels of opposition
    top_teams_df = matches_df[matches_df['opponent_rank'] <= 6]
    mid_teams_df = matches_df[(matches_df['opponent_rank'] > 6) & (matches_df['opponent_rank'] <= 12)]
    bottom_teams_df = matches_df[matches_df['opponent_rank'] > 12]
    
    # Calculate performance vs different strength opponents
    perf_vs_top = top_teams_df['points_earned'].mean() if not top_teams_df.empty else None
    perf_vs_mid = mid_teams_df['points_earned'].mean() if not mid_teams_df.empty else None
    perf_vs_bottom = bottom_teams_df['points_earned'].mean() if not bottom_teams_df.empty else None
    
    # Overall metrics compilation
    perf_metrics['overall'] = {
        'wins': victories,
        'draws': draws, 
        'losses': defeats,
        'win_pct': victories / actual_games_analyzed,
        'draw_pct': draws / actual_games_analyzed,
        'loss_pct': defeats / actual_games_analyzed,
        'avg_goals_scored': matches_df['team_goals'].mean(),
        'avg_goals_conceded': matches_df['opponent_goals'].mean(),
        'avg_goal_diff': matches_df['goal_diff'].mean(),
        'total_points': int(matches_df['points_earned'].sum()),
        'points_per_game': matches_df['points_earned'].mean(),
        'clean_sheets': int(matches_df[matches_df['opponent_goals'] == 0].shape[0]),
        'failed_to_score': int(matches_df[matches_df['team_goals'] == 0].shape[0]),
        'clean_sheet_pct': int(matches_df[matches_df['opponent_goals'] == 0].shape[0]) / actual_games_analyzed,
        'failed_to_score_pct': int(matches_df[matches_df['team_goals'] == 0].shape[0]) / actual_games_analyzed,
        'avg_opponent_rank': matches_df['opponent_rank'].mean(),
        'avg_opponent_points': matches_df['opponent_points'].mean(),
        
        # Advanced Metrics
        'points_volatility': calculate_volatility(matches_df['points_earned']),
        'goal_diff_volatility': calculate_volatility(matches_df['goal_diff']),
        'points_momentum': calculate_momentum(matches_df['points_earned'], window=5),
        'goal_diff_momentum': calculate_momentum(matches_df['goal_diff'], window=5),
        'points_trend_slope': points_slope,
        'points_trend_strength': points_r2,
        'goal_diff_trend_slope': goal_diff_slope,
        'goal_diff_trend_strength': goal_diff_r2,
        'xG_per_game': xG,
        'results_pattern': results_sequence_str,
        'consistency_score': consistency_score,
        'form_score': form_analysis['form_score'],
        'form_volatility': form_analysis['form_volatility'],
        'current_streak': f"{form_analysis['current_streak_type']}{form_analysis['current_streak_length']}",
        'ppg_recent_vs_overall_ratio': ppg_ratio,
        'perf_vs_top_teams': perf_vs_top,
        'perf_vs_mid_teams': perf_vs_mid,
        'perf_vs_bottom_teams': perf_vs_bottom,
    }

    # Home/Away Split Analysis
    for venue in ['home', 'away']:
        venue_df = matches_df[matches_df['venue'] == venue]
        venue_games = len(venue_df)
        
        if venue_games > 0:
            # Calculate venue-specific trends
            venue_points_slope, _, venue_points_r2 = calculate_trend(venue_df['points_earned'])
            venue_xG = calculate_expected_goals(
                venue_df['team_goals'].tolist(),
                venue_df['opponent_rank'].tolist()
            )
            
            perf_metrics[venue] = {
                'games_played': venue_games,
                'wins': int(venue_df[venue_df['result'] == 'win'].shape[0]),
                'draws': int(venue_df[venue_df['result'] == 'draw'].shape[0]),
                'losses': int(venue_df[venue_df['result'] == 'loss'].shape[0]),
                'win_pct': venue_df[venue_df['result'] == 'win'].shape[0] / venue_games,
                'avg_goals_scored': venue_df['team_goals'].mean(),
                'avg_goals_conceded': venue_df['opponent_goals'].mean(),
                'avg_goal_diff': venue_df['goal_diff'].mean(),
                'points_per_game': venue_df['points_earned'].mean(),
                'clean_sheets': int(venue_df[venue_df['opponent_goals'] == 0].shape[0]),
                'failed_to_score': int(venue_df[venue_df['team_goals'] == 0].shape[0]),
                'avg_opponent_rank': venue_df['opponent_rank'].mean(),
                'avg_opponent_points': venue_df['opponent_points'].mean(),
                'points_volatility': calculate_volatility(venue_df['points_earned']),
                'points_trend': venue_points_slope,
                'xG_per_game': venue_xG,
            }
        else:
            perf_metrics[venue] = {'games_played': 0, 'message': 'Not enough games for specific analysis'}

    analysis_results['performance_metrics_last_n'] = perf_metrics

    # --- 4. Advanced Correlation Analysis ---
    correlation_results = {}
    
    # Only perform correlation if sufficient data points
    if not matches_df['opponent_rank'].isnull().all() and actual_games_analyzed > 5:
        # Filter out games where key data is missing
        corr_df = matches_df.dropna(subset=['opponent_rank', 'points_earned', 'goal_diff'])

        if len(corr_df) > 5:
            # Multiple correlation metrics
            metrics_to_correlate = [
                'points_earned', 'goal_diff', 'team_goals', 'opponent_goals',
            ]
            correlations = {}

            for metric in metrics_to_correlate:
                if metric in corr_df.columns and not corr_df[metric].isnull().all():
                    valid_ranks = corr_df['opponent_rank']
                    valid_metric = corr_df[metric]

                    # Pearson correlation (linear)
                    try:
                        pearson_corr, pearson_p = pearsonr(valid_ranks, valid_metric)
                        correlations[f'pearson_{metric}_vs_opponent_rank'] = {
                            'correlation': pearson_corr,
                            'p_value': pearson_p,
                            'significant': pearson_p < 0.05
                        }
                    except ValueError:
                        correlations[f'pearson_{metric}_vs_opponent_rank'] = {'error': 'Calculation failed'}

                    # Spearman correlation (monotonic, rank-based)
                    try:
                        spearman_corr, spearman_p = spearmanr(valid_ranks, valid_metric)
                        correlations[f'spearman_{metric}_vs_opponent_rank'] = {
                            'correlation': spearman_corr,
                            'p_value': spearman_p,
                            'significant': spearman_p < 0.05
                        }
                    except ValueError:
                        correlations[f'spearman_{metric}_vs_opponent_rank'] = {'error': 'Calculation failed'}
            
            # Compare performance against top vs bottom teams
            top_half_df = corr_df[corr_df['opponent_rank'] <= 9]  # Top half of league
            bottom_half_df = corr_df[corr_df['opponent_rank'] > 9]  # Bottom half of league
            
            if len(top_half_df) >= 3 and len(bottom_half_df) >= 3:
                try:
                    # Statistical test to compare performance against different opponent strengths
                    t_stat, p_val = ttest_ind(
                        top_half_df['points_earned'].values,
                        bottom_half_df['points_earned'].values,
                        equal_var=False
                    )
                    
                    # Mann-Whitney U test (non-parametric alternative)
                    u_stat, u_p = mannwhitneyu(
                        top_half_df['points_earned'].values,
                        bottom_half_df['points_earned'].values,
                        alternative='two-sided'
                    )
                    
                    top_bottom_comparison = {
                        't_test': {
                            'statistic': t_stat,
                            'p_value': p_val,
                            'significant': p_val < 0.05,
                        },
                        'mann_whitney': {
                            'statistic': u_stat,
                            'p_value': u_p,
                            'significant': u_p < 0.05,
                        },
                        'top_half_ppg': top_half_df['points_earned'].mean(),
                        'bottom_half_ppg': bottom_half_df['points_earned'].mean(),
                        'difference': top_half_df['points_earned'].mean() - bottom_half_df['points_earned'].mean(),
                    }
                    
                    correlations['top_vs_bottom_half_performance'] = top_bottom_comparison
                except:
                    correlations['top_vs_bottom_half_performance'] = {
                        'error': 'Comparison failed, insufficient or invalid data'
                    }
                
            correlation_results = correlations
            correlation_results['interpretation_notes'] = [
                "Correlation measures the relationship between opponent rank (lower is better team) and performance metrics.",
                "Positive correlation: team performs BETTER against HIGHER ranked (worse) opponents.",
                "Negative correlation: team performs BETTER against LOWER ranked (better) opponents.",
                "P-value < 0.05 suggests statistical significance (unlikely due to random chance).",
                f"Analysis based on {len(corr_df)} games with valid opponent rank data."
            ]
        else:
            correlation_results = {'message': f'Not enough games ({len(corr_df)}) with valid data for correlation analysis.'}
    else:
        correlation_results = {'message': 'Opponent rank data missing or insufficient for correlation analysis.'}

    analysis_results['correlation_analysis'] = correlation_results

    # --- 5. Goal Scoring Pattern Analysis ---
    scoring_patterns = analyze_scoring_patterns(team_data, team_name)
    analysis_results['scoring_patterns'] = scoring_patterns

    # --- 6. Extract StatArea Stats (for comparison) ---
    statarea_stats = {}
    sa_raw = safe_get(team_data, ['teams', 'home', 'statarea_analysis', 'raw_stats'], {})

    for period in ['host_15', 'guest_15', 'host_10', 'guest_10', 'host_5', 'guest_5']:
        if period in sa_raw:
            statarea_stats[period] = {
                'avg_scored': safe_get(sa_raw, [period, 'Average scored goals per match']),
                'avg_conceded': safe_get(sa_raw, [period, 'Average conceded goals per match']),
                'chance_score_%': parse_percentage(safe_get(sa_raw, [period, 'Chance to score goal next match'])),
                'chance_concede_%': parse_percentage(safe_get(sa_raw, [period, 'Chance to conceded goal next match'])),
                'over_2.5_matches': safe_get(sa_raw, [period, 'Matches over 2.5 goals in']),
                'clean_sheets': safe_get(sa_raw, [period, 'Number of clean sheet matches']),
                'failed_to_score': safe_get(sa_raw, [period, 'Failure to score matches']),
                'wins': safe_get(sa_raw, [period, f'Number of {team_name} wins']),
                'draws': safe_get(sa_raw, [period, f'Number of {team_name} draws']),
                'losses': safe_get(sa_raw, [period, f'Number of {team_name} loses']),
                'goal_timing_all': safe_get(sa_raw, [period, 'All goals in matches'], {}),
                'goal_timing_team': safe_get(sa_raw, [period, f'{team_name} goals only'], {}),
            }
            
            # Parse numeric values
            for k, v in statarea_stats[period].items():
                if isinstance(v, str):
                    try:
                        statarea_stats[period][k] = float(v)
                    except ValueError:
                        pass

    analysis_results['statarea_summary_stats'] = statarea_stats

    # --- 7. Enhanced Parametric Insights & Summary ---
    insights = []
    advanced_insights = []
    
    # Basic Performance Summary
    calc_ppg = safe_get(perf_metrics, ['overall', 'points_per_game'])
    insights.append(f"Over the last {actual_games_analyzed} relevant games, the team averaged {calc_ppg:.2f} points per game.")
    
    # Opponent Quality
    avg_opp_rank = safe_get(perf_metrics, ['overall', 'avg_opponent_rank'])
    if avg_opp_rank:
        insights.append(f"Average opponent rank was {avg_opp_rank:.1f} (lower is stronger).")
    
    # Performance Trend Analysis
    points_trend = safe_get(perf_metrics, ['overall', 'points_trend_slope'])
    if points_trend is not None:
        trend_desc = "improving" if points_trend > 0.05 else ("declining" if points_trend < -0.05 else "stable")
        advanced_insights.append(f"Team's form is {trend_desc} with a trend coefficient of {points_trend:.3f}.")
    
    # Consistency Analysis
    consistency = safe_get(perf_metrics, ['overall', 'consistency_score'])
    if consistency is not None:
        consistency_desc = "highly consistent" if consistency > 0.7 else (
            "moderately consistent" if consistency > 0.4 else "inconsistent")
        advanced_insights.append(f"Results pattern is {consistency_desc} (score: {consistency:.2f}/1.0).")
    
    # Correlation with Opponent Quality
    points_corr = safe_get(correlation_results, ['spearman_points_earned_vs_opponent_rank'])
    if points_corr and 'correlation' in points_corr:
        corr_val = points_corr['correlation']
        p_val = points_corr['p_value']
        sig = "significant" if p_val < 0.05 else "not significant"
        
        # Interpretation based on correlation strength
        if corr_val > 0.3:
            corr_insight = f"Strong positive correlation ({corr_val:.2f}, p={p_val:.3f}, {sig}) between opponent rank and points earned, indicating the team significantly outperforms against weaker opposition."
        elif corr_val > 0.1:
            corr_insight = f"Moderate positive correlation ({corr_val:.2f}, p={p_val:.3f}, {sig}) between opponent rank and points earned, suggesting better performance against weaker teams."
        elif corr_val < -0.3:
            corr_insight = f"Strong negative correlation ({corr_val:.2f}, p={p_val:.3f}, {sig}) between opponent rank and points earned, indicating the team significantly rises to the occasion against stronger opposition."
        elif corr_val < -0.1:
            corr_insight = f"Moderate negative correlation ({corr_val:.2f}, p={p_val:.3f}, {sig}) between opponent rank and points earned, suggesting better performance against stronger teams."
        else:
            corr_insight = f"Weak correlation ({corr_val:.2f}, p={p_val:.3f}, {sig}) between opponent rank and points earned, indicating consistent performance regardless of opposition quality."
        
        insights.append(corr_insight)
    
    # Top vs Bottom Comparison
    top_vs_bottom = safe_get(correlation_results, ['top_vs_bottom_half_performance'])
    if isinstance(top_vs_bottom, dict) and 'difference' in top_vs_bottom:
        diff = top_vs_bottom['difference']
        sig = "statistically significant" if top_vs_bottom.get('t_test', {}).get('significant', False) else "not statistically significant"
        
        if abs(diff) > 0.5:
            if diff > 0:
                insight = f"Team earns {abs(diff):.2f} more points per game against bottom-half teams than top-half teams ({sig})."
            else:
                insight = f"Team earns {abs(diff):.2f} more points per game against top-half teams than bottom-half teams ({sig})."
            advanced_insights.append(insight)
    
    # Momentum/Volatility Analysis
    points_mom = safe_get(perf_metrics, ['overall', 'points_momentum'])
    points_vol = safe_get(perf_metrics, ['overall', 'points_volatility'])
    
    if points_mom is not None:
        mom_desc = "positive (improving)" if points_mom > 0.3 else (
            "negative (declining)" if points_mom < -0.3 else "neutral")
        insights.append(f"Points momentum is {points_mom:.2f} ({mom_desc}).")
    
    if points_vol is not None:
        vol_desc = "highly volatile" if points_vol > 1.2 else (
            "moderately volatile" if points_vol > 0.8 else "consistent")
        advanced_insights.append(f"Performance volatility is {vol_desc} (σ={points_vol:.2f}).")
    
    # Current Form Assessment
    current_streak = safe_get(perf_metrics, ['overall', 'current_streak'])
    if current_streak:
        insights.append(f"Current form shows a streak of {current_streak}.")
    
    # Scoring Pattern Insights
    key_patterns = scoring_patterns.get('key_patterns', [])
    for pattern in key_patterns:
        advanced_insights.append(pattern)
    
    # Venue Performance Comparison
    home_ppg = safe_get(perf_metrics, ['home', 'points_per_game'])
    away_ppg = safe_get(perf_metrics, ['away', 'points_per_game'])
    
    if home_ppg is not None and away_ppg is not None:
        home_away_diff = home_ppg - away_ppg
        if abs(home_away_diff) > 0.5:
            venue_desc = f"significantly stronger at home" if home_away_diff > 0 else f"significantly stronger away"
            insights.append(f"Team is {venue_desc} (Δ={abs(home_away_diff):.2f} ppg).")
    
    # Expected Goals Analysis
    xg = safe_get(perf_metrics, ['overall', 'xG_per_game'])
    actual_goals = safe_get(perf_metrics, ['overall', 'avg_goals_scored'])
    
    if xg is not None and actual_goals is not None:
        xg_diff = actual_goals - xg
        if abs(xg_diff) > 0.3:
            xg_desc = "overperforming expected goals" if xg_diff > 0 else "underperforming expected goals"
            advanced_insights.append(f"Team is {xg_desc} by {abs(xg_diff):.2f} goals per game.")
    
    # Combine insights
    all_insights = insights + advanced_insights
    analysis_results['parametric_insights'] = all_insights
    
    # Compile key metrics for easy reference
    analysis_results['key_metrics'] = {
        'ppg': calc_ppg,
        'form_trend': points_trend,
        'consistency': consistency,
        'correlation_with_opponent_rank': points_corr['correlation'] if points_corr and 'correlation' in points_corr else None,
        'points_momentum': points_mom,
        'points_volatility': points_vol,
        'home_away_performance_gap': home_ppg - away_ppg if home_ppg is not None and away_ppg is not None else None,
        'xG_vs_actual': xg_diff if xg is not None and actual_goals is not None else None
    }

    # Add the new advanced metrics calculation
    advanced_metrics = calculate_advanced_metrics(matches_df)
    analysis_results['advanced_metrics'] = advanced_metrics

    # Add metadata and descriptions for metrics
    analysis_results['metric_explanations'] = {
        "points_per_game": "Average points earned per match - primary indicator of team performance",
        "win_pct": "Percentage of matches won - indicates team's ability to secure victories",
        "form_trend": "Slope of the linear regression line for recent points - positive values indicate improving form",
        "consistency_score": "Measure between 0-1 of how predictable a team's results are - higher values indicate more consistency",
        "correlation_with_opponent_rank": "Statistical relationship between opponent quality and team performance",
        "points_momentum": "Difference between points in most recent games vs previous games - indicates if team is gaining/losing momentum",
        "points_volatility": "Standard deviation of points earned - higher values indicate unpredictable performance",
        "home_away_performance_gap": "Difference between home and away PPG - indicates home field advantage",
        "xG_per_game": "Expected goals per game based on quality of chances created",
        "xG_vs_actual": "Difference between expected and actual goals - indicates finishing efficiency"
    }
    
    # Ensure xG is calculated and not null
    # Calculate xG using shot data if available, otherwise use enhanced model
    shot_data = safe_get(team_data, ['teams', 'home', 'shot_data'], None)
    
    home_xg = calculate_expected_goals(
        matches_df['team_goals'].tolist(),
        matches_df['opponent_rank'].tolist(),
        shot_data=safe_get(team_data, ['teams', 'home', 'shot_data']),
        league_id=league_id,
        league_name=league_name
    )
    
    # Calculate xG for home/away splits
    home_matches = matches_df[matches_df['venue'] == 'home']
    away_matches = matches_df[matches_df['venue'] == 'away']
    
    home_venue_xg = calculate_expected_goals(
        home_matches['team_goals'].tolist(),
        home_matches['opponent_rank'].tolist(),
        league_id=league_id,
        league_name=league_name,
        venue='home'
    ) if not home_matches.empty else None
    
    away_venue_xg = calculate_expected_goals(
        away_matches['team_goals'].tolist(),
        away_matches['opponent_rank'].tolist(),
        league_id=league_id,
        league_name=league_name,
        venue='away'
    ) if not away_matches.empty else None
    
    # Update the xG values in performance metrics
    perf_metrics['overall']['xG_per_game'] = home_xg
    if 'home' in perf_metrics and perf_metrics['home'].get('games_played', 0) > 0:
        perf_metrics['home']['xG_per_game'] = home_venue_xg
    if 'away' in perf_metrics and perf_metrics['away'].get('games_played', 0) > 0:
        perf_metrics['away']['xG_per_game'] = away_venue_xg
    
    # Calculate xG vs actual if xG is available
    if home_xg is not None:
        actual_goals = perf_metrics['overall'].get('avg_goals_scored', 0)
        perf_metrics['overall']['xG_vs_actual'] = actual_goals - home_xg
    
    # Handle mid-tier teams performance (fix null values)
    if perf_metrics['overall'].get('perf_vs_mid_teams') is None and mid_teams_df.empty:
        # Use weighted average of top and bottom if no mid-tier opponents
        top_perf = perf_metrics['overall'].get('perf_vs_top_teams', 0)
        bottom_perf = perf_metrics['overall'].get('perf_vs_bottom_teams', 0)
        
        if top_perf and bottom_perf:
            perf_metrics['overall']['perf_vs_mid_teams'] = (top_perf + bottom_perf) / 2
            perf_metrics['overall']['perf_vs_mid_teams_note'] = "Estimated from top/bottom performance (no mid-tier opponents faced)"
    
    # Enhance correlation analysis with better explanations
    if 'correlation_analysis' in analysis_results:
        corr_analysis = analysis_results['correlation_analysis']
        
        # Add correlation interpretation
        if 'spearman_points_earned_vs_opponent_rank' in corr_analysis:
            corr_val = corr_analysis['spearman_points_earned_vs_opponent_rank'].get('correlation', 0)
            p_val = corr_analysis['spearman_points_earned_vs_opponent_rank'].get('p_value', 1)
            
            # Add interpretation
            corr_analysis['correlation_interpretation'] = {
                "value": corr_val,
                "strength": get_correlation_strength(corr_val),
                "meaning": get_correlation_meaning(corr_val),
                "statistical_significance": p_val < 0.05,
                "explanation": generate_correlation_explanation(corr_val, p_val)
            }
    
    # Update key metrics with non-null values
    analysis_results['key_metrics'] = {
        'ppg': perf_metrics['overall'].get('points_per_game', 0),
        'form_trend': perf_metrics['overall'].get('points_trend_slope', 0),
        'consistency': perf_metrics['overall'].get('consistency_score', 0),
        'correlation_with_opponent_rank': corr_analysis.get('spearman_points_earned_vs_opponent_rank', {}).get('correlation', 0) 
                                           if 'correlation_analysis' in analysis_results else 0,
        'points_momentum': perf_metrics['overall'].get('points_momentum', 0),
        'points_volatility': perf_metrics['overall'].get('points_volatility', 0),
        'home_away_performance_gap': perf_metrics.get('home', {}).get('points_per_game', 0) - 
                                     perf_metrics.get('away', {}).get('points_per_game', 0),
        'xG_per_game': perf_metrics['overall'].get('xG_per_game', 0),
        'xG_vs_actual': perf_metrics['overall'].get('xG_vs_actual', 0)
    }
    
    # Add information about number of games analyzed
    analysis_results['last_n_games_info'] = {
        "games_analyzed": actual_games_analyzed,
        "target_games": NUM_GAMES,
        "explanation": f"Analysis based on the last {actual_games_analyzed} matches out of target {NUM_GAMES}"
    }
    
    # Add league metrics information to the output for transparency
    league_metrics_info = load_league_metrics(league_id, league_name)
    analysis_results['league_metrics_used'] = {
        'league_id': league_id,
        'league_name': league_name,
        'avg_goals_per_game': league_metrics_info.get('avg_goals_per_game_total', 0),
        'home_advantage_factor': league_metrics_info.get('home_advantage', 0.2),
        'data_source': 'dynamically calculated from league data'
    }
    
    return analysis_results

def get_correlation_strength(corr_val):
    """Return the strength description of a correlation value."""
    abs_corr = abs(corr_val)
    if abs_corr < 0.1:
        return "Negligible"
    elif abs_corr < 0.3:
        return "Weak"
    elif abs_corr < 0.5:
        return "Moderate"
    elif abs_corr < 0.7:
        return "Strong"
    else:
        return "Very strong"

def get_correlation_meaning(corr_val):
    """Return the meaning of the correlation in context of opponent rank."""
    if corr_val > 0.1:
        return "Team performs better against weaker opponents (higher ranks)"
    elif corr_val < -0.1:
        return "Team performs better against stronger opponents (lower ranks)"
    else:
        return "Team performs consistently regardless of opponent quality"

def generate_correlation_explanation(corr_val, p_val):
    """Generate a detailed explanation of correlation results."""
    strength = get_correlation_strength(corr_val)
    direction = "positive" if corr_val > 0 else "negative" if corr_val < 0 else "neutral"
    significance = "statistically significant" if p_val < 0.05 else "not statistically significant"
    
    explanation = f"The {strength.lower()} {direction} correlation (r={corr_val:.2f}) is {significance} (p={p_val:.3f})."
    
    # Detailed soccer-specific interpretation
    if corr_val > 0.1:
        if corr_val > 0.4:
            explanation += f" This team has a strong tendency to perform much better against weaker teams (teams with higher ranks)."
            explanation += f" They likely struggle to step up against top competition but can dominate against lower-ranked opponents."
        else:
            explanation += f" This team tends to perform better against weaker teams (teams with higher ranks)."
            explanation += f" They may find it challenging to secure points against top-tier opposition."
    elif corr_val < -0.1:
        if corr_val < -0.4:
            explanation += f" This team has a strong tendency to rise to the occasion against stronger teams (teams with lower ranks)."
            explanation += f" They often perform better when challenged by quality opposition but may underperform against weaker teams."
        else:
            explanation += f" This team tends to perform better against stronger teams (teams with lower ranks)."
            explanation += f" They may occasionally struggle to maintain focus against lower-ranked opposition."
    else:
        explanation += f" This team's performance is relatively consistent regardless of opponent strength."
        explanation += f" They maintain similar performance levels whether facing top teams or bottom teams."
    
    if not p_val < 0.05:
        explanation += " However, since the correlation is not statistically significant, this pattern could be due to random chance rather than a true relationship."
    
    return explanation

def ensure_output_directory():
    """Create the output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")
    
    # Create subdirectories for each league
    if os.path.exists(DAILY_GAMES_DIR):
        league_folders = [d for d in os.listdir(DAILY_GAMES_DIR) 
                if os.path.isdir(os.path.join(DAILY_GAMES_DIR, d))]
        
        for league_folder in league_folders:
            league_output_dir = os.path.join(OUTPUT_DIR, league_folder)
            os.makedirs(league_output_dir, exist_ok=True)
            
    print("Output directories ready")

def find_league_folders() -> List[str]:
    """Find all league folders in the daily_games directory."""
    if not os.path.exists(DAILY_GAMES_DIR):
        raise FileNotFoundError(f"Daily games directory not found: {DAILY_GAMES_DIR}")
    
    return [d for d in os.listdir(DAILY_GAMES_DIR) 
            if os.path.isdir(os.path.join(DAILY_GAMES_DIR, d))]

def get_match_files(league_folder: str) -> List[Tuple[str, str]]:
    """Find all match JSON files in a league folder and extract their dates."""
    league_path = os.path.join(DAILY_GAMES_DIR, league_folder)
    match_files = []
    
    for file in os.listdir(league_path):
        if file.endswith('.json') and 'standings' not in file and 'summary' not in file:
            file_path = os.path.join(league_path, file)
            date_match = re.search(DATE_PATTERN, file)
            if date_match:
                date = date_match.group(1)
                match_files.append((file_path, date))
    
    return match_files

def create_default_standings() -> Dict:
    """Create default standings data structure when real standings are unavailable."""
    return {
        "league_info": {
            "id": "0",
            "name": "Unknown League",
            "country": "Unknown",
            "season": datetime.now().year
        },
        "standings": [[]]  # Empty standings list
    }

def load_standings(league_folder: str, match_date: str) -> Dict:
    """
    Load the standings.json file for a given league and date.
    
    Args:
        league_folder: The name of the league folder (e.g., "Bundesliga_Germany")
        match_date: The date of the match in YYYY-MM-DD format
        
    Returns:
        Dictionary containing standings data or empty dict if not found
    """
    league_path = os.path.join(DAILY_GAMES_DIR, league_folder)
    if not os.path.exists(league_path):
        print(f"Warning: League folder not found at {league_path}")
        return create_default_standings()
    
    # Strategy 1: Look for exact date standings file (YYYY-MM-DD_standings.json)
    exact_standings_file = f"{match_date}_standings.json"
    exact_standings_path = os.path.join(league_path, exact_standings_file)
    
    if os.path.exists(exact_standings_path):
        print(f"Found exact date standings file: {exact_standings_file}")
        standings_path = exact_standings_path
    else:
        # Strategy 2: Look for any standings file in the directory
        standings_files = [f for f in os.listdir(league_path) if f.endswith("standings.json")]
        if standings_files:
            # Sort by date, newest first (assuming date prefix format)
            standings_files.sort(reverse=True)
            standings_path = os.path.join(league_path, standings_files[0])
            print(f"Using most recent standings file: {standings_files[0]}")
        else:
            print(f"No standings files found in {league_folder}")
            return create_default_standings()
    
    # Load and validate the standings file
    try:
        with open(standings_path, 'r', encoding='utf-8') as f:
            standings_data = json.load(f)
            
        # Validate basic structure
        if 'standings' not in standings_data:
            print(f"Warning: Invalid standings format in {standings_path} - missing 'standings' key")
            return create_default_standings()
            
        if not standings_data['standings'] or not isinstance(standings_data['standings'], list):
            print(f"Warning: Empty or invalid standings list in {standings_path}")
            return create_default_standings()
            
        # Check if we have team entries in the first standings group
        if not standings_data['standings'][0]:
            print(f"Warning: No teams found in standings in {standings_path}")
            return create_default_standings()
            
        print(f"Successfully loaded standings with {len(standings_data['standings'][0])} teams")
        return standings_data
            
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error decoding standings file {standings_path}: {e}")
    except Exception as e:
        print(f"Unexpected error loading standings from {standings_path}: {e}")
    
    return create_default_standings()

def process_match(match_file: str, standings_data: Dict = None) -> Dict:
    """
    Process a single match file with standings data to generate comprehensive analysis.
    
    Args:
        match_file: Path to the match JSON file
        standings_data: Optional pre-loaded standings data
        
    Returns:
        Dictionary containing complete match analysis for both teams
    """
    print(f"\nProcessing match: {os.path.basename(match_file)}")
    
    # Load match data
    try:
        with open(match_file, 'r', encoding='utf-8') as f:
            match_data = json.load(f)
    except Exception as e:
        print(f"Error loading match file {match_file}: {e}")
        return {"error": f"Failed to load match file: {str(e)}"}
    
    # Extract match date for standings lookup
    match_date = None
    date_match = re.search(DATE_PATTERN, os.path.basename(match_file))
    if date_match:
        match_date = date_match.group(1)
    else:
        # Try to extract from fixture info
        fixture_date = safe_get(match_data, ['fixture_info', 'date'])
        if fixture_date:
            try:
                match_date = fixture_date.split('T')[0]
            except:
                # Default to current date
                match_date = datetime.now().strftime('%Y-%m-%d')
    
    # Extract league folder path (parent directory of match file)
    league_folder_path = os.path.dirname(match_file)
    league_folder_name = os.path.basename(league_folder_path)
    
    # Load standings if not provided
    if not standings_data:
        standings_data = load_standings(league_folder_name, match_date)
    
    try:
        # Extract basic match info
        match_id = safe_get(match_data, ['fixture_info', 'id'], f"unknown_{int(datetime.now().timestamp())}")
        league_info = safe_get(match_data, ['league'], {})
        fixture_info = safe_get(match_data, ['fixture_info'], {})
        home_team_id = safe_get(match_data, ['teams', 'home', 'id'])
        home_team_name = safe_get(match_data, ['teams', 'home', 'name'])
        away_team_id = safe_get(match_data, ['teams', 'away', 'id'])
        away_team_name = safe_get(match_data, ['teams', 'away', 'name'])
        
        print(f"Match: {home_team_name} vs {away_team_name}")
        
        # Process home team
        print(f"Analyzing home team: {home_team_name}")
        home_team_analysis = analyze_team_performance(match_data, league_folder_path)
        
        # Create away team analysis by swapping home/away in match_data
        print(f"Analyzing away team: {away_team_name}")
        away_match_data = swap_home_away_teams(match_data)
        away_team_analysis = analyze_team_performance(away_match_data, league_folder_path)
        
        # Basic validation - if analysis failed completely, create dummy analysis
        if not home_team_analysis:
            print(f"Warning: Home team analysis failed for {home_team_name}")
            home_team_analysis = {"team_info": {"name": home_team_name}}
        
        if not away_team_analysis:
            print(f"Warning: Away team analysis failed for {away_team_name}")
            away_team_analysis = {"team_info": {"name": away_team_name}}
        
        # Enhanced match predictions with more detailed stats
        print("Generating match predictions")
        match_predictions = calculate_match_predictions(home_team_analysis, away_team_analysis, match_data)
        
        # Add head-to-head analysis
        h2h_analysis = analyze_head_to_head(match_data.get('h2h', []), home_team_id, away_team_id)
        
        # Generate parametric insights
        print("Generating parametric insights")
        parametric_insights = generate_parametric_insights(
            home_team_analysis, 
            away_team_analysis, 
            h2h_analysis, 
            match_predictions
        )
        
        # Combine analyses and add match overview
        result = {
            "match_id": match_id,
            "league": league_info,
            "fixture_info": fixture_info,
            "home_team": {
                "id": home_team_id,
                "name": home_team_name,
                "analysis": home_team_analysis
            },
            "away_team": {
                "id": away_team_id,
                "name": away_team_name,
                "analysis": away_team_analysis
            },
            "h2h": match_data.get('h2h', []),
            "h2h_analysis": h2h_analysis,
            "match_predictions": match_predictions,
            "parametric_insights": parametric_insights,
            "processed_at": datetime.now().isoformat()
        }
        
        # Add a documentation section to each output file
        result['documentation'] = {
            "metrics": METRIC_DESCRIPTIONS,
            "correlation_analysis": {
                "description": "Measures how team performance correlates with opponent quality",
                "interpretation": {
                    "positive_correlation": "Team performs better against weaker opponents (higher ranks)",
                    "negative_correlation": "Team performs better against stronger opponents (lower ranks)",
                    "p_value": "Statistical significance (p<0.05 indicates significance)"
                },
                "xG_explanation": "Expected Goals (xG) measures the quality of chances created based on historical probability of similar shots resulting in goals"
            },
            "last_n_games": f"Analysis based on up to {NUM_GAMES} recent relevant matches"
        }
        
        return result
        
    except Exception as e:
        print(f"Error in process_match for {match_file}: {e}")
        import traceback
        traceback.print_exc()
        
        # Create minimal response with error info
        return {
            "match_id": safe_get(match_data, ["fixture_info", "id"], f"error_{int(datetime.now().timestamp())}"),
            "error": str(e),
            "league": safe_get(match_data, ["league"], {}),
            "fixture_info": safe_get(match_data, ["fixture_info"], {}),
            "processed_at": datetime.now().isoformat(),
            "status": "error"
        }

def swap_home_away_teams(match_data: Dict) -> Dict:
    """Create a copy of match_data with home and away teams swapped."""
    result = match_data.copy()
    
    # Deep copy to avoid reference issues
    if "teams" in result:
        teams = result["teams"].copy()
        if "home" in teams and "away" in teams:
            result["teams"] = {
                "home": teams["away"].copy(),
                "away": teams["home"].copy()
            }
    
    return result

def calculate_match_predictions(home_analysis: Dict, away_analysis: Dict, match_data: Dict) -> Dict:
    """Calculate detailed match predictions based on both team analyses."""
    # Handle missing data with defaults
    if not home_analysis:
        home_analysis = {"performance_metrics_last_n": {"overall": {}, "home": {}}}
    if not away_analysis:
        away_analysis = {"performance_metrics_last_n": {"overall": {}, "away": {}}}
    
    # Ensure required sections exist
    home_analysis.setdefault("performance_metrics_last_n", {}).setdefault("overall", {})
    home_analysis.setdefault("performance_metrics_last_n", {}).setdefault("home", {})
    away_analysis.setdefault("performance_metrics_last_n", {}).setdefault("overall", {})
    away_analysis.setdefault("performance_metrics_last_n", {}).setdefault("away", {})
    
    predictions = {
        "outcome_probabilities": calculate_outcome_probabilities(home_analysis, away_analysis),
        "total_goals_distribution": calculate_total_goals_distribution(home_analysis, away_analysis),
        "goal_timing_analysis": analyze_goal_timing(home_analysis, away_analysis),
        "performance_metrics_comparison": compare_performance_metrics(home_analysis, away_analysis),
        "form_analysis": analyze_form_trends(home_analysis, away_analysis),
        "key_insights": generate_key_insights(home_analysis, away_analysis, match_data)
    }
    
    return predictions

def calculate_outcome_probabilities(home_analysis: Dict, away_analysis: Dict) -> Dict:
    """Calculate the probability of each match outcome (1X2)."""
    # Extract key metrics
    home_ppg_home = home_analysis.get("performance_metrics_last_n", {}).get("home", {}).get("points_per_game", 0)
    away_ppg_away = away_analysis.get("performance_metrics_last_n", {}).get("away", {}).get("points_per_game", 0)
    
    home_overall = home_analysis.get("performance_metrics_last_n", {}).get("overall", {})
    away_overall = away_analysis.get("performance_metrics_last_n", {}).get("overall", {})
    
    # Calculate baseline probabilities
    home_strength = (home_ppg_home if home_ppg_home else home_overall.get("points_per_game", 0)) * 1.2
    away_strength = (away_ppg_away if away_ppg_away else away_overall.get("points_per_game", 0)) * 0.8
    
    # Normalize to create probabilities
    total_strength = home_strength + away_strength + 1.0  # Adding 1.0 for draw likelihood
    home_win_prob = home_strength / total_strength
    away_win_prob = away_strength / total_strength
    draw_prob = 1.0 / total_strength
    
    # Adjust based on goal scoring/conceding patterns
    home_avg_scored = home_overall.get("avg_goals_scored", 0)
    home_avg_conceded = home_overall.get("avg_goals_conceded", 0)
    away_avg_scored = away_overall.get("avg_goals_scored", 0)
    away_avg_conceded = away_overall.get("avg_goals_conceded", 0)
    
    expected_home_goals = (home_avg_scored + away_avg_conceded) / 2
    expected_away_goals = (away_avg_scored + home_avg_conceded) / 2
    
    # Final probabilities with adjustments
    return {
        "home_win": round(home_win_prob, 3),
        "draw": round(draw_prob, 3),
        "away_win": round(away_win_prob, 3),
        "expected_goals": {
            "home": round(expected_home_goals, 2),
            "away": round(expected_away_goals, 2)
        },
        "btts_probability": calculate_btts_probability(home_analysis, away_analysis),
        "over_under_probabilities": calculate_over_under_probabilities(expected_home_goals, expected_away_goals)
    }

def calculate_btts_probability(home_analysis: Dict, away_analysis: Dict) -> float:
    """Calculate probability of both teams to score."""
    home_fail_to_score = home_analysis.get("performance_metrics_last_n", {}).get("overall", {}).get("failed_to_score_pct", 0)
    away_fail_to_score = away_analysis.get("performance_metrics_last_n", {}).get("overall", {}).get("failed_to_score_pct", 0)
    
    # Probability that both teams will score = 1 - P(at least one team fails to score)
    btts_prob = 1 - (home_fail_to_score + away_fail_to_score - (home_fail_to_score * away_fail_to_score))
    return round(max(0, min(1, btts_prob)), 3)

def calculate_over_under_probabilities(home_expected: float, away_expected: float) -> Dict:
    """Calculate over/under probabilities for common goal totals."""
    total_expected = home_expected + away_expected
    
    # Calculate probabilities using Poisson approximation
    result = {}
    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
        # Simplified Poisson cumulative probability approximation
        under_prob = calculate_poisson_under(total_expected, threshold)
        result[f"under_{threshold}"] = round(under_prob, 3)
        result[f"over_{threshold}"] = round(1 - under_prob, 3)
    
    return result

def calculate_poisson_under(lambda_val: float, threshold: float) -> float:
    """Calculate Poisson cumulative probability for being under threshold."""
    # Simple approximation for Poisson cumulative probability
    if threshold <= 0:
        return 0
    
    cumulative_prob = 0
    for k in range(int(threshold + 1)):
        cumulative_prob += (lambda_val ** k) * np.exp(-lambda_val) / math.factorial(k)
    
    return min(1, max(0, cumulative_prob))

def calculate_total_goals_distribution(home_analysis: Dict, away_analysis: Dict) -> Dict:
    """Calculate the distribution of total goals probabilities."""
    home_overall = home_analysis.get("performance_metrics_last_n", {}).get("overall", {})
    away_overall = away_analysis.get("performance_metrics_last_n", {}).get("overall", {})
    
    # Expected goals for each team
    home_expected = (home_overall.get("avg_goals_scored", 0) + 
                    away_overall.get("avg_goals_conceded", 0)) / 2
    
    away_expected = (away_overall.get("avg_goals_scored", 0) + 
                    home_overall.get("avg_goals_conceded", 0)) / 2
    
    # Calculate goal distribution using Poisson distribution
    distribution = {}
    max_goals = 6  # Cap at 6+ goals
    
    for home_goals in range(max_goals):
        for away_goals in range(max_goals):
            # Calculate Poisson probability for this scoreline
            home_prob = poisson_probability(home_goals, home_expected)
            away_prob = poisson_probability(away_goals, away_expected)
            scoreline_prob = home_prob * away_prob
            
            # Store probability
            total_goals = home_goals + away_goals
            if total_goals >= max_goals - 1:
                total_goals = f"{max_goals - 1}+"
            
            if str(total_goals) not in distribution:
                distribution[str(total_goals)] = 0
            
            distribution[str(total_goals)] += scoreline_prob
    
    # Round all probabilities
    for k in distribution:
        distribution[k] = round(distribution[k], 3)
    
    return {
        "distribution": distribution,
        "expected_total": round(home_expected + away_expected, 2),
        "most_likely": max(distribution.items(), key=lambda x: x[1])[0]
    }

def poisson_probability(k: int, lambda_val: float) -> float:
    """Calculate the Poisson probability for k events with lambda rate."""
    return (lambda_val ** k) * np.exp(-lambda_val) / math.factorial(k)

def analyze_goal_timing(home_analysis: Dict, away_analysis: Dict) -> Dict:
    """Analyze when goals are likely to be scored in the match."""
    # Extract timing patterns
    home_scoring = home_analysis.get("scoring_patterns", {}).get("scoring_by_interval", {})
    home_conceding = home_analysis.get("scoring_patterns", {}).get("conceding_by_interval", {})
    away_scoring = away_analysis.get("scoring_patterns", {}).get("scoring_by_interval", {})
    away_conceding = away_analysis.get("scoring_patterns", {}).get("conceding_by_interval", {})
    
    # Calculate likelihood of goals in each interval
    intervals = ["0-15 min.", "16-30 min.", "31-45 min.", "46-60 min.", "61-75 min.", "76-90 min."]
    home_goal_timing = {}
    away_goal_timing = {}
    
    for interval in intervals:
        # Home team scoring probability = (home scoring * away conceding) ^ 0.5
        home_goal_timing[interval] = round(((home_scoring.get(interval, 0) + away_conceding.get(interval, 0)) / 2), 3)
        
        # Away team scoring probability = (away scoring * home conceding) ^ 0.5
        away_goal_timing[interval] = round(((away_scoring.get(interval, 0) + home_conceding.get(interval, 0)) / 2), 3)
    
    # Find the most dangerous periods
    home_max_interval = max(home_goal_timing.items(), key=lambda x: x[1]) if home_goal_timing else (None, 0)
    away_max_interval = max(away_goal_timing.items(), key=lambda x: x[1]) if away_goal_timing else (None, 0)
    
    return {
        "home_team_goal_timing": home_goal_timing,
        "away_team_goal_timing": away_goal_timing,
        "home_team_most_dangerous_period": home_max_interval[0],
        "away_team_most_dangerous_period": away_max_interval[0],
        "first_goal_timing_prediction": predict_first_goal_timing(home_analysis, away_analysis)
    }

def predict_first_goal_timing(home_analysis: Dict, away_analysis: Dict) -> Dict:
    """Predict when the first goal is likely to be scored."""
    # Get first goal timing data
    home_statarea = home_analysis.get("statarea_summary_stats", {}).get("host_15", {})
    away_statarea = away_analysis.get("statarea_summary_stats", {}).get("guest_15", {})
    
    first_goal_home = home_statarea.get("Time of first goal in matches", {})
    first_goal_away = away_statarea.get("Time of first goal in matches", {})
    
    # Combine probabilities from both teams
    combined_timing = {}
    timing_intervals = ["0-10 min.", "11-20 min.", "21-30 min.", "31-40 min.", "41-50 min.", 
                       "51-60 min.", "61-70 min.", "71-80 min.", "81-90 min.", "without goal"]
    
    for interval in timing_intervals:
        home_prob = parse_percentage(first_goal_home.get(interval, "0%"))
        away_prob = parse_percentage(first_goal_away.get(interval, "0%"))
        
        if home_prob is not None and away_prob is not None:
            combined_timing[interval] = round((home_prob + away_prob) / 2, 3)
    
    # Find most likely timing
    most_likely = max(combined_timing.items(), key=lambda x: x[1]) if combined_timing else (None, 0)
    
    return {
        "timing_distribution": combined_timing,
        "most_likely_interval": most_likely[0],
        "no_goal_probability": combined_timing.get("without goal", 0)
    }

def compare_performance_metrics(home_analysis: Dict, away_analysis: Dict) -> Dict:
    """Compare key performance metrics between home and away teams."""
    home_metrics = home_analysis.get("performance_metrics_last_n", {}).get("overall", {})
    away_metrics = away_analysis.get("performance_metrics_last_n", {}).get("overall", {})
    
    # Key metrics to compare
    key_metrics = [
        "points_per_game", "win_pct", "avg_goals_scored", "avg_goals_conceded",
        "clean_sheet_pct", "failed_to_score_pct", "consistency_score"
    ]
    
    comparison = {}
    for metric in key_metrics:
        home_val = home_metrics.get(metric, 0)
        away_val = away_metrics.get(metric, 0)
        
        comparison[metric] = {
            "home": round(home_val, 3) if isinstance(home_val, (int, float)) else home_val,
            "away": round(away_val, 3) if isinstance(away_val, (int, float)) else away_val,
            "difference": round(home_val - away_val, 3) if isinstance(home_val, (int, float)) and isinstance(away_val, (int, float)) else None,
            "advantage": "home" if home_val > away_val else "away" if away_val > home_val else "equal"
        }
    
    # Count total advantages
    home_advantages = sum(1 for m in comparison.values() if m.get("advantage") == "home")
    away_advantages = sum(1 for m in comparison.values() if m.get("advantage") == "away")
    
    return {
        "metrics_comparison": comparison,
        "home_team_advantages": home_advantages,
        "away_team_advantages": away_advantages,
        "overall_advantage": "home" if home_advantages > away_advantages else "away" if away_advantages > home_advantages else "equal"
    }

def analyze_form_trends(home_analysis: Dict, away_analysis: Dict) -> Dict:
    """Analyze form trends and momentum for both teams."""
    home_metrics = home_analysis.get("performance_metrics_last_n", {}).get("overall", {})
    away_metrics = away_analysis.get("performance_metrics_last_n", {}).get("overall", {})
    
    # Extract form patterns
    home_form = home_analysis.get("team_info", {}).get("form", "")
    away_form = away_analysis.get("team_info", {}).get("form", "")
    
    # Calculate form momentum (recent vs earlier results)
    home_momentum = home_metrics.get("points_momentum", 0)
    away_momentum = away_metrics.get("points_momentum", 0)
    
    # Form streaks
    home_streak = home_metrics.get("current_streak", "")
    away_streak = away_metrics.get("current_streak", "")
    
    # Determine which team has stronger recent form
    home_form_score = home_metrics.get("form_score", 0)
    away_form_score = away_metrics.get("form_score", 0)
    form_advantage = "home" if home_form_score > away_form_score else "away" if away_form_score > home_form_score else "equal"
    
    # Form volatility comparison
    home_volatility = home_metrics.get("form_volatility", 0)
    away_volatility = away_metrics.get("form_volatility", 0)
    
    return {
        "home_team_form": home_form,
        "away_team_form": away_form,
        "home_team_momentum": round(home_momentum, 3) if home_momentum is not None else None,
        "away_team_momentum": round(away_momentum, 3) if away_momentum is not None else None,
        "home_team_streak": home_streak,
        "away_team_streak": away_streak,
        "form_advantage": form_advantage,
        "home_volatility": round(home_volatility, 3) if home_volatility is not None else None,
        "away_volatility": round(away_volatility, 3) if away_volatility is not None else None,
        "most_consistent_team": "home" if home_volatility < away_volatility else "away" if away_volatility < home_volatility else "equal"
    }

def generate_key_insights(home_analysis: Dict, away_analysis: Dict, match_data: Dict) -> List[str]:
    """Generate key insights and betting recommendations for the match."""
    insights = []
    
    # Home/Away strength insights
    home_home_ppg = home_analysis.get("performance_metrics_last_n", {}).get("home", {}).get("points_per_game")
    away_away_ppg = away_analysis.get("performance_metrics_last_n", {}).get("away", {}).get("points_per_game")
    
    if home_home_ppg and home_home_ppg > 2:
        insights.append(f"Home team is very strong at home, averaging {home_home_ppg:.2f} points per game")
    
    if away_away_ppg and away_away_ppg > 1.5:
        insights.append(f"Away team performs well on the road, averaging {away_away_ppg:.2f} points per game")
    
    # Goal patterns
    home_scored = home_analysis.get("performance_metrics_last_n", {}).get("overall", {}).get("avg_goals_scored", 0)
    away_scored = away_analysis.get("performance_metrics_last_n", {}).get("overall", {}).get("avg_goals_scored", 0)
    
    if home_scored + away_scored > 3:
        insights.append(f"High-scoring match expected, teams average {home_scored + away_scored:.2f} goals combined")
    
    if home_scored + away_scored < 2:
        insights.append(f"Low-scoring match likely, teams average only {home_scored + away_scored:.2f} goals combined")
    
    # Timing patterns
    home_patterns = home_analysis.get("scoring_patterns", {}).get("key_patterns", [])
    away_patterns = away_analysis.get("scoring_patterns", {}).get("key_patterns", [])
    
    for pattern in home_patterns:
        if "Strong finisher" in pattern:
            insights.append("Home team tends to score late in matches")
        elif "Fast starter" in pattern:
            insights.append("Home team often scores early")
    
    for pattern in away_patterns:
        if "Strong finisher" in pattern:
            insights.append("Away team tends to score late in matches")
        elif "Fast starter" in pattern:
            insights.append("Away team often scores early")
    
    # H2H insights
    h2h = match_data.get("h2h", [])
    if h2h and len(h2h) >= 3:
        home_wins = sum(1 for m in h2h if safe_get(m, ["home_team", "id"]) == safe_get(match_data, ["teams", "home", "id"]) and safe_get(m, ["home_team", "winner"]) == True)
        away_wins = sum(1 for m in h2h if safe_get(m, ["away_team", "id"]) == safe_get(match_data, ["teams", "away", "id"]) and safe_get(m, ["away_team", "winner"]) == True)
        draws = len(h2h) - home_wins - away_wins
        
        if home_wins > away_wins + draws:
            insights.append(f"Home team has dominated recent H2H with {home_wins} wins in last {len(h2h)} meetings")
        elif away_wins > home_wins + draws:
            insights.append(f"Away team has dominated recent H2H with {away_wins} wins in last {len(h2h)} meetings")
        elif draws > home_wins and draws > away_wins:
            insights.append(f"Teams have drawn {draws} of their last {len(h2h)} meetings")
    
    # Return limited set of most important insights
    return insights[:5]  # Limit to top 5 insights

def save_processed_match(match_id: str, league: str, date: str, data: Dict) -> None:
    """Save processed match data to output directory."""
    # Create league subdirectory if needed
    league_output_dir = os.path.join(OUTPUT_DIR, league)
    os.makedirs(league_output_dir, exist_ok=True)
    
    # Get team names for the filename
    home_team_name = safe_get(data, ['home_team', 'name'], '')
    away_team_name = safe_get(data, ['away_team', 'name'], '')
    
    # Clean team names for file system compatibility
    home_team_clean = re.sub(r'[^\w\s-]', '', home_team_name).replace(' ', '_')
    away_team_clean = re.sub(r'[^\w\s-]', '', away_team_name).replace(' ', '_')
    
    # Create team-based filename if team names are available
    if home_team_clean and away_team_clean:
        output_file = os.path.join(league_output_dir, f"{date}_{home_team_clean}_vs_{away_team_clean}_{match_id}_analysis.json")
    else:
        # Fallback to original naming if team names aren't available
        output_file = os.path.join(league_output_dir, f"{date}_{match_id}_analysis.json")
    
    # Make the data JSON serializable by converting non-serializable types
    serializable_data = convert_to_serializable(data)
    
    # Save data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"Saved analysis to {output_file}")
    except Exception as e:
        print(f"Error saving output file {output_file}: {e}")

def convert_to_serializable(obj):
    """Convert non-serializable objects to serializable types for JSON."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None  # Convert NaN/Inf to null
        return float(obj)
    elif isinstance(obj, bool):
        # Keep booleans as Python booleans
        return bool(obj)
    elif pd.isna(obj) or (hasattr(obj, '__float__') and math.isnan(float(obj))):
        return None  # Convert any NaN-like value to null
    elif isinstance(obj, (int, float, str)) or obj is None:
        # Check for inf/nan in float values
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    else:
        # Convert anything else to string
        return str(obj)

def process_all_matches():
    """Process all matches in the daily_games directory."""
    ensure_output_directory()
    
    if not os.path.exists(DAILY_GAMES_DIR):
        print(f"WARNING: Daily games directory not found at {DAILY_GAMES_DIR}")
        print("Creating directory structure for testing...")
        os.makedirs(DAILY_GAMES_DIR, exist_ok=True)
          
    league_folders = find_league_folders()
    
    if not league_folders:
        print(f"No league folders found in {DAILY_GAMES_DIR}")
        return
        
    total_processed = 0
    print(f"Found {len(league_folders)} league folders to process")
    print(f"Daily games directory: {os.path.abspath(DAILY_GAMES_DIR)}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    
    for league_folder in league_folders:
        match_files = get_match_files(league_folder)
        print(f"Processing {len(match_files)} matches in {league_folder}")
        
        if not match_files:
            print(f"No match files found in {league_folder}")
            continue
        
        # Group files by date for more efficient processing
        files_by_date = {}
        for match_file, date in match_files:
            if date not in files_by_date:
                files_by_date[date] = []
            files_by_date[date].append(match_file)
        
        # Process each date's matches
        for date, date_match_files in files_by_date.items():
            print(f"Processing {len(date_match_files)} matches for date {date}")
            
            # Load standings once per date to be efficient
            league_folder_path = os.path.join(DAILY_GAMES_DIR, league_folder)
            standings_data = load_standings(league_folder, date)
            
            for match_file in date_match_files:
                try:
                    # Process match with the standings data
                    match_data = process_match(match_file, standings_data)
                    
                    # Extract match ID and save
                    match_id = match_data.get("match_id", "unknown")
                    save_processed_match(match_id, league_folder, date, match_data)
                    total_processed += 1
                    
                except Exception as e:
                    print(f"Error processing {match_file}: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    print(f"Complete! Processed {total_processed} matches.")

def analyze_head_to_head(h2h_data: List[Dict], home_team_id: str, away_team_id: str) -> Dict:
    """
    Analyze head-to-head record between the two teams.
    
    Args:
        h2h_data: List of previous matches between the teams
        home_team_id: ID of the home team
        away_team_id: ID of the away team
        
    Returns:
        Dictionary with head-to-head analysis results
    """
    if not h2h_data or not isinstance(h2h_data, list):
        return {
            "num_matches": 0,
            "home_wins": 0,
            "away_wins": 0,
            "draws": 0,
            "summary": "No head-to-head data available"
        }
    
    # Initialize counters
    home_wins = 0
    away_wins = 0
    draws = 0
    home_goals = 0
    away_goals = 0
    
    # Recent form in last matches (most recent first)
    recent_results = []
    
    # Process each match
    for match in h2h_data:
        match_home_id = safe_get(match, ["home_team", "id"])
        match_away_id = safe_get(match, ["away_team", "id"])
        match_home_goals = safe_get(match, ["home_team", "goals"], 0)
        match_away_goals = safe_get(match, ["away_team", "goals"], 0)
        
        # Determine who won this match from perspective of current home team
        if match_home_id == home_team_id and match_away_id == away_team_id:
            # Same home/away setup as current match
            if match_home_goals > match_away_goals:
                home_wins += 1
                recent_results.append("H")
            elif match_home_goals < match_away_goals:
                away_wins += 1
                recent_results.append("A")
            else:
                draws += 1
                recent_results.append("D")
                
            home_goals += match_home_goals
            away_goals += match_away_goals
            
        elif match_home_id == away_team_id and match_away_id == home_team_id:
            # Reverse of current match setup
            if match_home_goals > match_away_goals:
                away_wins += 1
                recent_results.append("A")
            elif match_home_goals < match_away_goals:
                home_wins += 1
                recent_results.append("H")
            else:
                draws += 1
                recent_results.append("D")
                
            home_goals += match_away_goals
            away_goals += match_home_goals
    
    total_matches = home_wins + away_wins + draws
    
    if total_matches == 0:
        return {
            "num_matches": 0,
            "home_wins": 0,
            "away_wins": 0,
            "draws": 0,
            "summary": "No valid head-to-head matches found"
        }
    
    # Calculate averages and percentages
    avg_home_goals = home_goals / total_matches
    avg_away_goals = away_goals / total_matches
    home_win_pct = home_wins / total_matches * 100
    away_win_pct = away_wins / total_matches * 100
    draw_pct = draws / total_matches * 100
    
    # Generate recent form string (last 5 matches)
    recent_form = "".join(recent_results[:5])
    
    # Generate summary text
    if home_wins > away_wins:
        advantage = "home team"
        advantage_size = home_wins - away_wins
    elif away_wins > home_wins:
        advantage = "away team"
        advantage_size = away_wins - home_wins
    else:
        advantage = "neither team"
        advantage_size = 0
    
    summary = f"In {total_matches} previous meetings, the {advantage} has the advantage"
    if advantage_size > 0:
        summary += f" by {advantage_size} wins"
    
    # Determine if there's a strong trend
    strong_trend = None
    if home_wins >= total_matches * 0.7:
        strong_trend = "Strong home team dominance in H2H matches"
    elif away_wins >= total_matches * 0.7:
        strong_trend = "Strong away team dominance in H2H matches"
    elif draws >= total_matches * 0.5:
        strong_trend = "Teams tend to draw frequently in H2H matches"
    elif home_goals + away_goals >= total_matches * 3:
        strong_trend = "High-scoring H2H matches with an average of {:.1f} goals per game".format(
            (home_goals + away_goals) / total_matches
        )
    
    return {
        "num_matches": total_matches,
        "home_wins": home_wins,
        "away_wins": away_wins,
        "draws": draws,
        "home_win_pct": round(home_win_pct, 1),
        "away_win_pct": round(away_win_pct, 1),
        "draw_pct": round(draw_pct, 1),
        "total_goals": home_goals + away_goals,
        "avg_match_goals": round((home_goals + away_goals) / total_matches, 2),
        "avg_home_goals": round(avg_home_goals, 2),
        "avg_away_goals": round(avg_away_goals, 2),
        "recent_form": recent_form,
        "advantage": advantage,
        "strong_trend": strong_trend,
        "summary": summary
    }

def generate_parametric_insights(home_analysis: Dict, away_analysis: Dict, 
                                h2h_analysis: Dict, match_predictions: Dict) -> Dict:
    """
    Generate detailed parametric insights by combining team and head-to-head analyses.
    
    Args:
        home_analysis: Home team performance analysis
        away_analysis: Away team performance analysis
        h2h_analysis: Head-to-head analysis
        match_predictions: Match prediction data
        
    Returns:
        Dictionary of detailed insights and recommendations
    """
    insights = {
        "summary": [],
        "key_stats": {},
        "tactical_insights": [],
        "statistical_highlights": [],
        "betting_angles": []
    }
    
    # Extract home team metrics
    home_team_name = safe_get(home_analysis, ["team_info", "name"], "Home Team")
    home_metrics = safe_get(home_analysis, ["performance_metrics_last_n", "overall"], {})
    home_home_metrics = safe_get(home_analysis, ["performance_metrics_last_n", "home"], {})
    home_ppg = safe_get(home_metrics, ["points_per_game"], 0)
    home_home_ppg = safe_get(home_home_metrics, ["points_per_game"], 0)
    home_goals_scored = safe_get(home_metrics, ["avg_goals_scored"], 0)
    home_goals_conceded = safe_get(home_metrics, ["avg_goals_conceded"], 0)
    home_form = safe_get(home_analysis, ["team_info", "form"], "")
    
    # Extract away team metrics
    away_team_name = safe_get(away_analysis, ["team_info", "name"], "Away Team")
    away_metrics = safe_get(away_analysis, ["performance_metrics_last_n", "overall"], {})
    away_away_metrics = safe_get(away_analysis, ["performance_metrics_last_n", "away"], {})
    away_ppg = safe_get(away_metrics, ["points_per_game"], 0)
    away_away_ppg = safe_get(away_away_metrics, ["points_per_game"], 0)
    away_goals_scored = safe_get(away_metrics, ["avg_goals_scored"], 0)
    away_goals_conceded = safe_get(away_metrics, ["avg_goals_conceded"], 0)
    away_form = safe_get(away_analysis, ["team_info", "form"], "")
    
    # Summarize match probabilities
    home_win_prob = safe_get(match_predictions, ["outcome_probabilities", "home_win"], 0)
    draw_prob = safe_get(match_predictions, ["outcome_probabilities", "draw"], 0)
    away_win_prob = safe_get(match_predictions, ["outcome_probabilities", "away_win"], 0)
    expected_home_goals = safe_get(match_predictions, ["outcome_probabilities", "expected_goals", "home"], 0)
    expected_away_goals = safe_get(match_predictions, ["outcome_probabilities", "expected_goals", "away"], 0)
    
    # Compile key stats
    insights["key_stats"] = {
        "home_ppg_overall": round(home_ppg, 2),
        "home_ppg_at_home": round(home_home_ppg, 2),
        "away_ppg_overall": round(away_ppg, 2),
        "away_ppg_away": round(away_away_ppg, 2),
        "home_form": home_form,
        "away_form": away_form,
        "home_win_probability": round(home_win_prob * 100, 1),
        "draw_probability": round(draw_prob * 100, 1),
        "away_win_probability": round(away_win_prob * 100, 1),
        "expected_home_goals": round(expected_home_goals, 2),
        "expected_away_goals": round(expected_away_goals, 2),
        "expected_total_goals": round(expected_home_goals + expected_away_goals, 2),
        "h2h_advantage": h2h_analysis.get("advantage", "neither team")
    }
    
    # Generate overall summary
    most_likely_outcome = "home win" if home_win_prob > max(draw_prob, away_win_prob) else (
        "draw" if draw_prob > away_win_prob else "away win"
    )
    
    summary = f"Statistical model predicts a {most_likely_outcome} as the most likely outcome "
    summary += f"({round(max(home_win_prob, draw_prob, away_win_prob) * 100, 1)}% probability), "
    summary += f"with an expected score around {round(expected_home_goals, 1)}-{round(expected_away_goals, 1)}."
    
    insights["summary"].append(summary)
    
    # Home team form insight
    if home_form and len(home_form) >= 3:
        wins = home_form.count('W')
        draws = home_form.count('D')
        losses = home_form.count('L')
        if wins >= 3:
            insights["summary"].append(f"{home_team_name} is in good form with {wins} wins in their last {len(home_form)} matches.")
        elif losses >= 3:
            insights["summary"].append(f"{home_team_name} is struggling with {losses} losses in their last {len(home_form)} matches.")
    
    # Away team form insight
    if away_form and len(away_form) >= 3:
        wins = away_form.count('W')
        draws = away_form.count('D')
        losses = away_form.count('L')
        if wins >= 3:
            insights["summary"].append(f"{away_team_name} is in good form with {wins} wins in their last {len(away_form)} matches.")
        elif losses >= 3:
            insights["summary"].append(f"{away_team_name} is struggling with {losses} losses in their last {len(away_form)} matches.")
            
    # H2H insight
    if h2h_analysis["num_matches"] > 0:
        insights["summary"].append(f"Head-to-head history shows {h2h_analysis['summary']}.")
        
    # Add tactical insights based on scoring patterns
    home_scoring_patterns = safe_get(home_analysis, ["scoring_patterns", "key_patterns"], [])
    away_scoring_patterns = safe_get(away_analysis, ["scoring_patterns", "key_patterns"], [])
    
    for pattern in home_scoring_patterns:
        insights["tactical_insights"].append(f"{home_team_name}: {pattern}")
    
    for pattern in away_scoring_patterns:
        insights["tactical_insights"].append(f"{away_team_name}: {pattern}")
        
    # Home advantage vs Away strength insight
    if home_home_ppg > 1.8 and away_away_ppg < 1.0:
        insights["tactical_insights"].append(f"Strong home advantage expected: {home_team_name} averages {round(home_home_ppg, 2)} PPG at home while {away_team_name} struggles away with {round(away_away_ppg, 2)} PPG.")
    elif away_away_ppg > 1.8 and home_home_ppg < 1.0:
        insights["tactical_insights"].append(f"Away team advantage expected: {away_team_name} performs well away ({round(away_away_ppg, 2)} PPG) while {home_team_name} struggles at home ({round(home_home_ppg, 2)} PPG).")
        
    # Expected goals differential insight
    if abs(expected_home_goals - expected_away_goals) > 1.0:
        stronger_team = f"{home_team_name} significantly stronger" if expected_home_goals > expected_away_goals else f"{away_team_name} significantly stronger"
        insights["tactical_insights"].append(f"{stronger_team}: Expected goals differential of {abs(round(expected_home_goals - expected_away_goals, 2))} suggests a clear offensive advantage.")
    
    # Statistical highlights
    home_clean_sheet_pct = safe_get(home_metrics, ["clean_sheet_pct"], 0) * 100
    away_clean_sheet_pct = safe_get(away_metrics, ["clean_sheet_pct"], 0) * 100
    home_failed_to_score_pct = safe_get(home_metrics, ["failed_to_score_pct"], 0) * 100
    away_failed_to_score_pct = safe_get(away_metrics, ["failed_to_score_pct"], 0) * 100
    
    if home_clean_sheet_pct > 50:
        insights["statistical_highlights"].append(f"{home_team_name} keeps clean sheets in {round(home_clean_sheet_pct, 1)}% of their matches.")
    
    if away_clean_sheet_pct > 50:
        insights["statistical_highlights"].append(f"{away_team_name} keeps clean sheets in {round(away_clean_sheet_pct, 1)}% of their matches.")
    
    if home_failed_to_score_pct > 40:
        insights["statistical_highlights"].append(f"{home_team_name} fails to score in {round(home_failed_to_score_pct, 1)}% of their matches.")
    
    if away_failed_to_score_pct > 40:
        insights["statistical_highlights"].append(f"{away_team_name} fails to score in {round(away_failed_to_score_pct, 1)}% of their matches.")
    
    # Betting angles
    btts_prob = safe_get(match_predictions, ["outcome_probabilities", "btts_probability"], 0) * 100
    over_2_5_prob = safe_get(match_predictions, ["outcome_probabilities", "over_under_probabilities", "over_2.5"], 0) * 100
    
    if btts_prob > 65:
        insights["betting_angles"].append(f"Both teams to score looks promising ({round(btts_prob, 1)}% probability).")
    elif btts_prob < 40:
        insights["betting_angles"].append(f"Both teams to score looks unlikely ({round(btts_prob, 1)}% probability).")
    
    if over_2_5_prob > 65:
        insights["betting_angles"].append(f"Over 2.5 goals has high probability ({round(over_2_5_prob, 1)}%).")
    elif over_2_5_prob < 40:
        insights["betting_angles"].append(f"Under 2.5 goals looks promising ({round(100 - over_2_5_prob, 1)}% probability).")
    
    if home_win_prob > 0.6:
        insights["betting_angles"].append(f"Strong home win probability of {round(home_win_prob * 100, 1)}%.")
    elif away_win_prob > 0.5:
        insights["betting_angles"].append(f"Above average away win probability of {round(away_win_prob * 100, 1)}%.")
    elif draw_prob > 0.3:
        insights["betting_angles"].append(f"Draw has elevated probability of {round(draw_prob * 100, 1)}%.")
    
    return insights

def calculate_advanced_metrics(matches_df: pd.DataFrame) -> Dict:
    """Calculate advanced metrics from match history including:
    - Goal timing patterns
    - BTTS frequency
    - Over/Under statistics
    - Performance against different quality opponents
    """
    metrics = {}
    
    # BTTS (Both Teams To Score) analysis
    btts_matches = matches_df[(matches_df['team_goals'] > 0) & (matches_df['opponent_goals'] > 0)]
    metrics['btts_frequency'] = len(btts_matches) / len(matches_df) if len(matches_df) > 0 else 0
    
    # Over/Under analysis
    metrics['over_under'] = {
        'over_0.5': (matches_df['team_goals'] + matches_df['opponent_goals'] > 0.5).mean(),
        'over_1.5': (matches_df['team_goals'] + matches_df['opponent_goals'] > 1.5).mean(),
        'over_2.5': (matches_df['team_goals'] + matches_df['opponent_goals'] > 2.5).mean(),
        'over_3.5': (matches_df['team_goals'] + matches_df['opponent_goals'] > 3.5).mean(),
        'under_0.5': (matches_df['team_goals'] + matches_df['opponent_goals'] < 0.5).mean(),
        'under_1.5': (matches_df['team_goals'] + matches_df['opponent_goals'] < 1.5).mean(),
        'under_2.5': (matches_df['team_goals'] + matches_df['opponent_goals'] < 2.5).mean(),
        'under_3.5': (matches_df['team_goals'] + matches_df['opponent_goals'] < 3.5).mean(),
    }
    
    # Performance against different opponent tiers
    if 'opponent_rank' in matches_df.columns and not matches_df['opponent_rank'].isnull().all():
        # Top tier (ranks 1-6)
        top_tier = matches_df[matches_df['opponent_rank'] <= 6]
        metrics['vs_top_tier'] = {
            'matches_played': len(top_tier),
            'win_pct': (top_tier['result'] == 'win').mean() if len(top_tier) > 0 else None,
            'points_per_game': top_tier['points_earned'].mean() if len(top_tier) > 0 else None,
            'goals_scored_per_game': top_tier['team_goals'].mean() if len(top_tier) > 0 else None,
            'goals_conceded_per_game': top_tier['opponent_goals'].mean() if len(top_tier) > 0 else None
        }
        
        # Mid tier (ranks 7-12)
        mid_tier = matches_df[(matches_df['opponent_rank'] > 6) & (matches_df['opponent_rank'] <= 12)]
        metrics['vs_mid_tier'] = {
            'matches_played': len(mid_tier),
            'win_pct': (mid_tier['result'] == 'win').mean() if len(mid_tier) > 0 else None,
            'points_per_game': mid_tier['points_earned'].mean() if len(mid_tier) > 0 else None,
            'goals_scored_per_game': mid_tier['team_goals'].mean() if len(mid_tier) > 0 else None,
            'goals_conceded_per_game': mid_tier['opponent_goals'].mean() if len(mid_tier) > 0 else None
        }
        
        # Bottom tier (ranks 13+)
        bottom_tier = matches_df[matches_df['opponent_rank'] > 12]
        metrics['vs_bottom_tier'] = {
            'matches_played': len(bottom_tier),
            'win_pct': (bottom_tier['result'] == 'win').mean() if len(bottom_tier) > 0 else None,
            'points_per_game': bottom_tier['points_earned'].mean() if len(bottom_tier) > 0 else None,
            'goals_scored_per_game': bottom_tier['team_goals'].mean() if len(bottom_tier) > 0 else None,
            'goals_conceded_per_game': bottom_tier['opponent_goals'].mean() if len(bottom_tier) > 0 else None
        }
    
    # Extract goal timing data if available from statarea_analysis
    if 'goal_timing' in matches_df.columns:
        # Process goal timing data
        pass
        
    return metrics

def extract_fallback_metrics(team_data: Dict) -> Dict:
    """
    Extract and process fallback metrics when match history is not available.
    Uses statarea_analysis raw_stats directly to build basic performance metrics.
    
    Args:
        team_data: The team data dictionary
        
    Returns:
        Dictionary with performance metrics that can be used as fallback
    """
    fallback_metrics = {
        'overall': {},
        'home': {},
        'away': {}
    }
    
    # Extract statarea raw stats
    sa_raw = safe_get(team_data, ['teams', 'home', 'statarea_analysis', 'raw_stats'], {})
    team_name = safe_get(team_data, ['teams', 'home', 'name'], '')
    
    if not sa_raw:
        return fallback_metrics
    
    # Extract data from statarea periods
    for location, periods in [('overall', ['host_15', 'guest_15']), 
                             ('home', ['host_15', 'host_10', 'host_5']), 
                             ('away', ['guest_15', 'guest_10', 'guest_5'])]:
        
        # Use the first available period data
        for period in periods:
            if period in sa_raw:
                period_data = sa_raw[period]
                
                # Extract basic metrics
                avg_scored = safe_get(period_data, ['Average scored goals per match'])
                avg_conceded = safe_get(period_data, ['Average conceded goals per match'])
                clean_sheets = safe_get(period_data, ['Number of clean sheet matches'])
                failed_to_score = safe_get(period_data, ['Failure to score matches'])
                wins = safe_get(period_data, [f'Number of {team_name} wins'])
                draws = safe_get(period_data, [f'Number of {team_name} draws'])
                losses = safe_get(period_data, [f'Number of {team_name} loses'])
                
                # Try to parse these values as numbers
                metrics = {
                    'avg_goals_scored': _parse_numeric(avg_scored),
                    'avg_goals_conceded': _parse_numeric(avg_conceded),
                    'clean_sheets': _parse_numeric(clean_sheets),
                    'failed_to_score': _parse_numeric(failed_to_score),
                    'wins': _parse_numeric(wins),
                    'draws': _parse_numeric(draws),
                    'losses': _parse_numeric(losses),
                }
                
                # Calculate derived metrics
                total_matches = metrics['wins'] + metrics['draws'] + metrics['losses']
                if total_matches > 0:
                    metrics['win_pct'] = metrics['wins'] / total_matches
                    metrics['draw_pct'] = metrics['draws'] / total_matches
                    metrics['loss_pct'] = metrics['losses'] / total_matches
                    metrics['points_per_game'] = (metrics['wins'] * 3 + metrics['draws']) / total_matches
                    metrics['clean_sheet_pct'] = metrics['clean_sheets'] / total_matches
                    metrics['failed_to_score_pct'] = metrics['failed_to_score'] / total_matches
                
                # Calculate BTTS metrics
                btts_matches = safe_get(period_data, ['Both teams scored matches'])
                metrics['btts_matches'] = _parse_numeric(btts_matches)
                if total_matches > 0:
                    metrics['btts_pct'] = metrics['btts_matches'] / total_matches
                
                # Calculate over/under metrics
                over_matches = safe_get(period_data, ['Matches over 2.5 goals in'])
                under_matches = safe_get(period_data, ['Matches under 2.5 goals in'])
                metrics['over_2.5_matches'] = _parse_numeric(over_matches)
                metrics['under_2.5_matches'] = _parse_numeric(under_matches)
                if total_matches > 0:
                    metrics['over_2.5_pct'] = metrics['over_2.5_matches'] / total_matches
                    metrics['under_2.5_pct'] = metrics['under_2.5_matches'] / total_matches
                
                # Add scoring patterns if available
                goal_timing_team = safe_get(period_data, [f'{team_name} goals only'], {})
                if goal_timing_team:
                    timing_data = {}
                    for interval, pct_str in goal_timing_team.items():
                        timing_data[interval] = parse_percentage(pct_str)
                    metrics['scoring_patterns'] = timing_data
                
                fallback_metrics[location] = metrics
                break  # Use only the first available period for each location
    
    return fallback_metrics

def _parse_numeric(value) -> float:
    """Helper function to parse numeric values from strings."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return 0.0

def update_league_metrics():
    """
    Update all league metrics from latest data.
    Should be called periodically to refresh metrics.
    """
    global LEAGUE_METRICS
    LEAGUE_METRICS = {}  # Clear cache
    
    # Process each league directory
    for dir_name in os.listdir(DAILY_GAMES_DIR):
        league_path = os.path.join(DAILY_GAMES_DIR, dir_name)
        if not os.path.isdir(league_path):
            continue
            
        # Find latest standings file
        standings_files = [f for f in os.listdir(league_path) if f.endswith('standings.json')]
        if not standings_files:
            continue
            
        # Sort by date (newest first)
        standings_files.sort(reverse=True)
        standings_file = os.path.join(league_path, standings_files[0])
        
        # Load standings and calculate metrics
        standings_data = load_json(standings_file)
        if standings_data:
            metrics = calculate_league_metrics(standings_data, league_path)
            
            # Store in cache under various keys for easy lookup
            league_info = standings_data.get('league_info', {})
            league_id = str(league_info.get('id', ''))
            league_name = league_info.get('name', '')
            
            if league_id:
                LEAGUE_METRICS[league_id] = metrics
            if league_name:
                LEAGUE_METRICS[league_name] = metrics
            LEAGUE_METRICS[dir_name] = metrics
            
            print(f"Updated metrics for league: {league_name} (ID: {league_id}, Dir: {dir_name})")
    
    print(f"League metrics updated for {len(LEAGUE_METRICS)} leagues/identifiers")
    return LEAGUE_METRICS

# Replace hardcoded values with dynamic configuration

# This will store league-specific metrics
LEAGUE_METRICS = {}

def load_league_metrics(league_id=None, league_name=None):
    """
    Dynamically load or calculate league-specific metrics from standings and match data.
    
    Args:
        league_id: Optional league ID to load specific league metrics
        league_name: Optional league name to load specific league metrics
        
    Returns:
        Dictionary of league-specific parameters for calculations
    """
    global LEAGUE_METRICS
    
    # Create a cache key
    cache_key = str(league_id) if league_id else league_name
    
    # Return cached metrics if available
    if cache_key in LEAGUE_METRICS:
        return LEAGUE_METRICS[cache_key]
    
    # Define paths to search for league data
    league_data_paths = []
    if league_name:
        # Convert league name to directory format if needed
        league_dir = league_name.replace(' ', '_')
        league_data_paths.append(os.path.join(DAILY_GAMES_DIR, league_dir))
    
    # Find all league directories if no specific league requested
    if not league_data_paths:
        for dir_name in os.listdir(DAILY_GAMES_DIR):
            if os.path.isdir(os.path.join(DAILY_GAMES_DIR, dir_name)):
                league_data_paths.append(os.path.join(DAILY_GAMES_DIR, dir_name))
    
    # Process each potential league directory
    for league_path in league_data_paths:
        if not os.path.exists(league_path):
            continue
        
        # Find latest standings file
        standings_files = [f for f in os.listdir(league_path) if f.endswith('standings.json')]
        if not standings_files:
            continue
            
        # Sort by date (newest first)
        standings_files.sort(reverse=True)
        standings_file = os.path.join(league_path, standings_files[0])
        
        # Load standings data
        standings_data = load_json(standings_file)
        if not standings_data:
            continue
        
        # Extract league info
        league_info = standings_data.get('league_info', {})
        current_league_id = league_info.get('id')
        current_league_name = league_info.get('name')
        
        # Check if this is the league we're looking for
        if (league_id and str(current_league_id) != str(league_id)) or \
           (league_name and current_league_name != league_name and 
            league_name not in current_league_name and 
            os.path.basename(league_path) != league_name.replace(' ', '_')):
            continue
        
        # Calculate league metrics from standings data
        metrics = calculate_league_metrics(standings_data, league_path)
        
        # Store in cache
        LEAGUE_METRICS[str(current_league_id)] = metrics
        LEAGUE_METRICS[current_league_name] = metrics
        LEAGUE_METRICS[os.path.basename(league_path)] = metrics
        
        # Return if this is the specific league we wanted
        if league_id or league_name:
            return metrics
    
    # If no specific league found or no league specified, return default metrics
    default_metrics = {
        'avg_goals_per_game_per_team': 1.25,  # Fallback value if no data
        'avg_goals_per_game_total': 2.5,
        'home_advantage': 0.2,
        'shot_conversion_rate': 0.09,
        'xg_model_intercept': 0.85,
        'xg_model_distance_coef': -0.1,
        'xg_model_angle_coef': 0.01,
        'performance_weights': {
            'recent_form': 0.5,
            'historical': 0.3,
            'league_average': 0.2
        },
        'quality_adjustment_low': 0.7,
        'quality_adjustment_high': 0.6,
        'max_team_data_weight': 0.8
    }
    
    # Cache the default metrics
    if cache_key:
        LEAGUE_METRICS[cache_key] = default_metrics
    
    return default_metrics

def calculate_league_metrics(standings_data, league_path):
    """
    Calculate league-specific metrics from standings and match data.
    
    Args:
        standings_data: Loaded standings data for the league
        league_path: Path to league directory for additional data
        
    Returns:
        Dictionary of league-specific metrics
    """
    metrics = {}
    
    # Extract team stats from standings
    teams_stats = []
    if 'standings' in standings_data and standings_data['standings']:
        for team in standings_data['standings'][0]:
            if 'all' in team and 'goals' in team['all']:
                goals_for = team['all']['goals'].get('for', 0)
                goals_against = team['all']['goals'].get('against', 0)
                matches_played = team['all'].get('played', 0)
                
                if matches_played > 0:
                    teams_stats.append({
                        'goals_for': goals_for,
                        'goals_against': goals_against,
                        'matches_played': matches_played
                    })
    
    # Calculate league averages if we have team stats
    if teams_stats:
        total_goals = sum(team['goals_for'] for team in teams_stats)
        total_matches = sum(team['matches_played'] for team in teams_stats) / 2  # Each match counted twice
        
        if total_matches > 0:
            # Average goals per game (total)
            metrics['avg_goals_per_game_total'] = total_goals / total_matches
            
            # Average goals per game per team
            metrics['avg_goals_per_game_per_team'] = metrics['avg_goals_per_game_total'] / 2
    
    # If we couldn't calculate from standings, try to use match data
    if 'avg_goals_per_game_total' not in metrics:
        # Find all match files
        match_files = [f for f in os.listdir(league_path) 
                      if f.endswith('.json') and 'standings' not in f]
        
        goals_data = []
        for match_file in match_files[:100]:  # Limit to recent 100 matches for performance
            match_data = load_json(os.path.join(league_path, match_file))
            if not match_data:
                continue
                
            # Extract goals data
            if 'teams' in match_data:
                home_goals = safe_get(match_data, ['teams', 'home', 'goals'], 0)
                away_goals = safe_get(match_data, ['teams', 'away', 'goals'], 0)
                
                if isinstance(home_goals, (int, float)) and isinstance(away_goals, (int, float)):
                    goals_data.append({
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'total_goals': home_goals + away_goals
                    })
        
        # Calculate averages from match data
        if goals_data:
            avg_total = sum(m['total_goals'] for m in goals_data) / len(goals_data)
            metrics['avg_goals_per_game_total'] = avg_total
            metrics['avg_goals_per_game_per_team'] = avg_total / 2
    
    # Set defaults if still missing
    if 'avg_goals_per_game_total' not in metrics:
        metrics['avg_goals_per_game_total'] = 2.5
        metrics['avg_goals_per_game_per_team'] = 1.25
    
    # Calculate other metrics based on league data
    metrics['home_advantage'] = calculate_home_advantage(league_path)
    
    # Set model parameters based on league characteristics
    # These could be refined with machine learning for each league
    avg_goals = metrics['avg_goals_per_game_total']
    if avg_goals > 3.0:
        # High-scoring league
        metrics['xg_model_intercept'] = 0.9
        metrics['xg_model_distance_coef'] = -0.09
        metrics['shot_conversion_rate'] = 0.11
    elif avg_goals < 2.0:
        # Low-scoring league
        metrics['xg_model_intercept'] = 0.75
        metrics['xg_model_distance_coef'] = -0.12
        metrics['shot_conversion_rate'] = 0.07
    else:
        # Average league
        metrics['xg_model_intercept'] = 0.85
        metrics['xg_model_distance_coef'] = -0.1
        metrics['shot_conversion_rate'] = 0.09
    
    # Common parameters across leagues
    metrics['xg_model_angle_coef'] = 0.01
    
    # Set weighting parameters
    metrics['performance_weights'] = {
        'recent_form': 0.5,
        'historical': 0.3,
        'league_average': 0.2
    }
    
    # Quality adjustment parameters
    metrics['quality_adjustment_low'] = 0.7
    metrics['quality_adjustment_high'] = 0.6
    metrics['max_team_data_weight'] = 0.8
    
    return metrics

def calculate_home_advantage(league_path):
    """
    Calculate home advantage factor for a specific league
    
    Args:
        league_path: Path to league directory for data
        
    Returns:
        Home advantage factor (typically 0.1-0.3)
    """
    # Find all match files
    match_files = [f for f in os.listdir(league_path) 
                  if f.endswith('.json') and 'standings' not in f]
    
    # Collect home/away results
    home_points = []
    away_points = []
    
    for match_file in match_files[:100]:  # Limit to recent 100 matches
        match_data = load_json(os.path.join(league_path, match_file))
        if not match_data:
            continue
        
        # Extract result if available
        if 'teams' in match_data and 'home' in match_data['teams'] and 'away' in match_data['teams']:
            home_goals = safe_get(match_data, ['teams', 'home', 'goals'], None)
            away_goals = safe_get(match_data, ['teams', 'away', 'goals'], None)
            
            if home_goals is not None and away_goals is not None:
                # Calculate points
                if home_goals > away_goals:
                    home_points.append(3)
                    away_points.append(0)
                elif home_goals < away_goals:
                    home_points.append(0)
                    away_points.append(3)
                else:
                    home_points.append(1)
                    away_points.append(1)
    
    # Calculate home advantage
    if home_points and away_points:
        home_ppg = sum(home_points) / len(home_points)
        away_ppg = sum(away_points) / len(away_points)
        home_advantage = (home_ppg - away_ppg) / 6  # Normalize to typical range
        
        # Cap at reasonable values
        return max(0.05, min(0.4, home_advantage))
    
    # Default if no data
    return 0.2

if __name__ == "__main__":
    try:
        print("Starting match probability calculator...")
        print(f"Daily games directory: {os.path.abspath(DAILY_GAMES_DIR)}")
        print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
        
        if not os.path.exists(DAILY_GAMES_DIR):
            print(f"WARNING: Daily games directory not found at {DAILY_GAMES_DIR}")
            print("Creating directory structure for testing...")
            os.makedirs(DAILY_GAMES_DIR, exist_ok=True)
        
        process_all_matches()
    except Exception as e:
        print(f"Fatal error in match probability calculator: {e}")
        import traceback
        traceback.print_exc()

