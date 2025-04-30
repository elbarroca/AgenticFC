# utils/features.py

import pandas as pd
import numpy as np
from typing import List, Optional, Dict

from models.utils.elo import EloCalculator

# Import EloCalculator if needed directly, or assume Elo is pre-calculated
# from .elo import EloCalculator

# --- Configuration (Could be moved to config.py) ---
ROLLING_WINDOW_SIZE = 5 # Example: Use last 5 games for rolling stats/form

def calculate_rolling_stats(df: pd.DataFrame, window: int = ROLLING_WINDOW_SIZE) -> pd.DataFrame:
    """
    Calculates rolling average statistics for teams.

    **Important:** This basic implementation calculates rolling stats over the
    entire history provided. For **non-leaky features**, this function should be
    applied carefully, typically grouped by season and ensuring the rolling window
    only uses data *prior* to the current match date. A more robust implementation
    might involve iterating match by match or using time-based rolling windows.

    Args:
        df (pd.DataFrame): DataFrame with historical matches. Needs 'HomeTeam',
                           'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', etc.
                           and ideally a 'Date' column for sorting.
        window (int): The number of past matches to include in the rolling window.

    Returns:
        pd.DataFrame: DataFrame with new rolling average columns added (e.g.,
                      'HomeAvgGoalsScored_L{window}', 'AwayAvgShotsTarget_L{window}').
                      Indices/rows might change depending on implementation details.
                      **This example returns stats *for* the match based on past N games.**
    """
    print(f"Calculating rolling stats (window={window})... (Basic Implementation)")
    # Ensure sorted by date for meaningful rolling stats
    if 'Date' in df.columns:
        df = df.sort_values(by='Date').copy()
    else:
        print("Warning: No 'Date' column found for sorting. Rolling stats might be less meaningful.")

    # Create unique match identifier if needed (e.g., if index isn't unique)
    # df['match_id'] = df.index

    # --- Create team-specific stats per match ---
    # Melt/stack data to have one row per team per match
    home_stats = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']].rename(columns={
        'HomeTeam': 'Team', 'FTHG': 'GoalsScored', 'FTAG': 'GoalsConceded',
        'HS': 'Shots', 'AS': 'ShotsConceded', 'HST': 'ShotsTarget', 'AST': 'ShotsTargetConceded',
        'HC': 'Corners', 'AC': 'CornersConceded'
    })
    home_stats['Venue'] = 'H'

    away_stats = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'AS', 'HS', 'AST', 'HST', 'AC', 'HC']].rename(columns={
        'AwayTeam': 'Team', 'FTAG': 'GoalsScored', 'FTHG': 'GoalsConceded',
        'AS': 'Shots', 'HS': 'ShotsConceded', 'AST': 'ShotsTarget', 'HST': 'ShotsTargetConceded',
        'AC': 'Corners', 'HC': 'CornersConceded'
    })
    away_stats['Venue'] = 'A'

    team_stats_long = pd.concat([home_stats, away_stats], ignore_index=True).sort_values(by=['Team', 'Date'])

    # --- Calculate Rolling Averages ---
    # Group by team and apply rolling window. shift(1) ensures we use data *before* the current match.
    stats_to_roll = ['GoalsScored', 'GoalsConceded', 'Shots', 'ShotsTarget', 'Corners']
    rolling_features = {}

    grouped = team_stats_long.groupby('Team')
    for stat in stats_to_roll:
        # Calculate rolling mean, shifting to exclude the current row
        rolling_mean = grouped[stat].rolling(window=window, closed='left').mean() # closed='left' uses past N excluding current
        # Reset index to align properly after grouping
        rolling_mean = rolling_mean.reset_index(level=0, drop=True)
        rolling_features[f'Avg_{stat}_L{window}'] = rolling_mean


    # Convert dictionary of Series to DataFrame
    rolling_df = pd.DataFrame(rolling_features, index=team_stats_long.index)

    # Merge back with the original long format data
    team_stats_with_rolling = pd.concat([team_stats_long, rolling_df], axis=1)

    # --- Pivot back to wide format (one row per match) ---
    # Separate home and away rolling stats
    home_rolling = team_stats_with_rolling[team_stats_with_rolling['Venue'] == 'H'].set_index(df.index) # Use original index
    away_rolling = team_stats_with_rolling[team_stats_with_rolling['Venue'] == 'A'].set_index(df.index)

    # Add prefixes and merge back to the original df
    home_cols_map = {col: f'Home_{col}' for col in rolling_df.columns}
    away_cols_map = {col: f'Away_{col}' for col in rolling_df.columns}

    df_with_features = df.copy()
    df_with_features = df_with_features.merge(home_rolling[rolling_df.columns].rename(columns=home_cols_map),
                                              left_index=True, right_index=True, how='left')
    df_with_features = df_with_features.merge(away_rolling[rolling_df.columns].rename(columns=away_cols_map),
                                              left_index=True, right_index=True, how='left')

    print("Rolling stats calculation finished.")
    return df_with_features


def calculate_form(df: pd.DataFrame, window: int = ROLLING_WINDOW_SIZE) -> pd.DataFrame:
    """
    Calculates form based on points earned in the last N games.

    **Note:** Similar non-leakage considerations apply as in calculate_rolling_stats.
    This basic implementation calculates form based on the N games *before* the current one.
    """
    print(f"Calculating form points (window={window})... (Basic Implementation)")
    if 'Date' in df.columns:
        df = df.sort_values(by='Date').copy()

    def get_points(result):
        if result == 'W': return 3
        if result == 'D': return 1
        return 0

    all_points = []
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

    for team in teams:
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
        team_matches['Result'] = team_matches.apply(
            lambda row: 'W' if (row['HomeTeam'] == team and row['FTR'] == 'H') or \
                              (row['AwayTeam'] == team and row['FTR'] == 'A') else
                       'D' if row['FTR'] == 'D' else 'L', axis=1
        )
        team_matches['Points'] = team_matches['Result'].apply(get_points)
        # Calculate rolling sum of points, shifting to exclude current match
        team_matches[f'FormPts_L{window}'] = team_matches['Points'].rolling(window=window, closed='left').sum()
        all_points.append(team_matches[['HomeTeam', 'AwayTeam', f'FormPts_L{window}']].copy()) # Keep keys for merging

    if not all_points:
        df[f'HomeFormPts_L{window}'] = np.nan
        df[f'AwayFormPts_L{window}'] = np.nan
        return df

    form_df = pd.concat(all_points).drop_duplicates(subset=['HomeTeam', 'AwayTeam'], keep='last') # Keep last calculated form per match pair

    # This merging logic is tricky because form is team-specific, not match-specific *before* the match.
    # A better approach involves the long format similar to rolling stats.
    # Simplified merge (may need refinement):
    df_with_form = df.copy()
    home_form = form_df.rename(columns={'HomeTeam': 'Team', f'FormPts_L{window}': f'HomeFormPts_L{window}'})[['Team', f'HomeFormPts_L{window}']]
    away_form = form_df.rename(columns={'AwayTeam': 'Team', f'FormPts_L{window}': f'AwayFormPts_L{window}'})[['Team', f'AwayFormPts_L{window}']]

    # This simple merge won't work correctly without proper indexing/timing. Placeholder:
    # df_with_form = df_with_form.merge(home_form, left_on='HomeTeam', right_on='Team', how='left')
    # df_with_form = df_with_form.merge(away_form, left_on='AwayTeam', right_on='Team', how='left')
    # For now, just add NaN columns as a placeholder for the complex merge logic
    df_with_form[f'HomeFormPts_L{window}'] = np.nan
    df_with_form[f'AwayFormPts_L{window}'] = np.nan
    print("Warning: Form calculation merge is complex; returning NaN placeholders.")
    print("Form points calculation finished.")
    return df_with_form


def process_odds(df: pd.DataFrame, odds_cols: List[str] = ['B365H', 'B365D', 'B365A']) -> pd.DataFrame:
    """
    Converts betting odds to implied probabilities and calculates margins.

    Args:
        df (pd.DataFrame): DataFrame containing odds columns.
        odds_cols (List[str]): List of column names for Home, Draw, Away odds.

    Returns:
        pd.DataFrame: DataFrame with new columns for implied probabilities
                      (e.g., 'ImpliedProbH', 'ImpliedProbD', 'ImpliedProbA') and 'BookmakerMargin'.
    """
    print("Processing odds...")
    df_with_probs = df.copy()
    h_col, d_col, a_col = odds_cols

    # Calculate inverse odds (handle potential division by zero or invalid odds)
    inv_h = 1.0 / df_with_probs[h_col].replace(0, np.nan)
    inv_d = 1.0 / df_with_probs[d_col].replace(0, np.nan)
    inv_a = 1.0 / df_with_probs[a_col].replace(0, np.nan)

    # Calculate margin
    margin = inv_h + inv_d + inv_a
    df_with_probs['BookmakerMargin'] = (margin - 1.0).fillna(0) # Fill NaN margin with 0

    # Calculate implied probabilities (normalized to remove margin)
    df_with_probs['ImpliedProbH'] = (inv_h / margin).fillna(1/3) # Fill NaN probs with uniform
    df_with_probs['ImpliedProbD'] = (inv_d / margin).fillna(1/3)
    df_with_probs['ImpliedProbA'] = (inv_a / margin).fillna(1/3)

    print("Odds processing finished.")
    return df_with_probs

def add_elo_features(df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges pre-calculated Elo ratings and calculates Elo difference.

    Args:
        df (pd.DataFrame): The main feature DataFrame. Needs 'HomeTeam', 'AwayTeam'.
                           Should ideally have a unique index or date/team keys for merging.
        elo_df (pd.DataFrame): DataFrame containing 'HomeEloBefore', 'AwayEloBefore'
                               and keys to merge on (e.g., index, or Date/HomeTeam/AwayTeam).

    Returns:
        pd.DataFrame: DataFrame with 'HomeEloBefore', 'AwayEloBefore', and 'EloDiff' added.
    """
    print("Adding Elo features...")
    # Assuming both dfs can be merged on index after sorting/alignment
    # More robust merging might use multi-index (Date, HomeTeam, AwayTeam)
    if not all(col in elo_df.columns for col in ['HomeEloBefore', 'AwayEloBefore']):
         raise ValueError("elo_df must contain 'HomeEloBefore' and 'AwayEloBefore'.")

    df_with_elo = df.merge(elo_df[['HomeEloBefore', 'AwayEloBefore']],
                           left_index=True, right_index=True, how='left')

    # Calculate Elo difference
    df_with_elo['EloDiff'] = df_with_elo['HomeEloBefore'] - df_with_elo['AwayEloBefore']
    # Handle potential NaNs from merge if matches aren't in elo_df
    df_with_elo.fillna({'HomeEloBefore': 1500, 'AwayEloBefore': 1500, 'EloDiff': 0}, inplace=True)

    print("Elo features added.")
    return df_with_elo

def generate_features(df: pd.DataFrame, elo_df: Optional[pd.DataFrame] = None,
                      odds_cols: List[str] = ['B365H', 'B365D', 'B365A'],
                      rolling_window: int = ROLLING_WINDOW_SIZE) -> pd.DataFrame:
    """
    Main function to generate a feature set for modeling.

    Orchestrates calls to individual feature generation functions.

    Args:
        df (pd.DataFrame): Raw historical data.
        elo_df (Optional[pd.DataFrame]): DataFrame with pre-calculated historical Elo ratings.
                                         If None, Elo features won't be added.
        odds_cols (List[str]): Columns containing H/D/A odds.
        rolling_window (int): Window size for rolling stats and form.

    Returns:
        pd.DataFrame: DataFrame containing the generated numerical features.
                      May contain NaNs from rolling calculations on early matches.
    """
    print("--- Starting Feature Generation ---")
    features_df = df.copy()

    # 1. Process Odds
    if all(col in features_df.columns for col in odds_cols):
        features_df = process_odds(features_df, odds_cols)
    else:
        print(f"Warning: Odds columns {odds_cols} not found. Skipping odds features.")

    # 2. Calculate Rolling Stats (Basic Implementation - see note in function)
    # Requires stats columns like FTHG, FTAG, HS, AS, etc.
    stats_cols_present = all(c in features_df.columns for c in ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC'])
    if stats_cols_present:
         features_df = calculate_rolling_stats(features_df, window=rolling_window)
    else:
         print("Warning: Missing required columns for rolling stats (FTHG, FTAG, HS, AS, etc.). Skipping.")

    # 3. Calculate Form (Basic Implementation - see note in function)
    if 'FTR' in features_df.columns:
         features_df = calculate_form(features_df, window=rolling_window)
    else:
         print("Warning: Missing 'FTR' column. Skipping form calculation.")

    # 4. Add Elo Features (Requires pre-calculated Elo)
    if elo_df is not None:
        # Ensure indices align or use proper merge keys
        if df.index.equals(elo_df.index):
             features_df = add_elo_features(features_df, elo_df)
        else:
             print("Warning: Index mismatch between main df and elo_df. Skipping Elo features. Ensure alignment.")
    else:
        print("No elo_df provided. Skipping Elo features.")

    # 5. Add other features (H2H, League Pos Diff, etc.) - Placeholder
    # These often require more complex lookups or external data (standings)
    features_df['LeaguePosDiff'] = np.nan # Placeholder
    features_df['H2H_WinRatio'] = np.nan # Placeholder
    print("Placeholder features (LeaguePosDiff, H2H_WinRatio) added as NaN.")

    # 6. Select final numerical features (and handle NaNs)
    # Define columns intended for the model
    # This list depends heavily on which features were successfully generated
    potential_feature_cols = [
        'BookmakerMargin', 'ImpliedProbH', 'ImpliedProbD', 'ImpliedProbA',
        f'Home_Avg_GoalsScored_L{rolling_window}', f'Home_Avg_GoalsConceded_L{rolling_window}',
        f'Away_Avg_GoalsScored_L{rolling_window}', f'Away_Avg_GoalsConceded_L{rolling_window}',
        # Add other rolling stats columns here...
        f'HomeFormPts_L{rolling_window}', f'AwayFormPts_L{rolling_window}', # Currently NaNs
        'HomeEloBefore', 'AwayEloBefore', 'EloDiff', # Add if elo_df was provided
        'LeaguePosDiff', 'H2H_WinRatio' # Currently NaNs
    ]

    final_feature_cols = [col for col in potential_feature_cols if col in features_df.columns]
    print(f"\nSelected final feature columns ({len(final_feature_cols)}): {final_feature_cols}")

    final_df = features_df[final_feature_cols].copy()

    # Handle NaNs - Crucial step!
    # Simple strategy: fill with 0 or median/mean. More complex imputation might be better.
    # NaNs often occur at the start of the dataset due to rolling windows.
    initial_nan_count = final_df.isnull().sum().sum()
    if initial_nan_count > 0:
         print(f"Warning: Feature DataFrame has {initial_nan_count} NaN values. Filling with 0 for simplicity.")
         # Consider more sophisticated imputation based on feature distribution
         final_df.fillna(0, inplace=True)

    print("--- Feature Generation Complete ---")
    return final_df


# Example Usage (Optional)
if __name__ == '__main__':
    # Create more comprehensive dummy data
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-08', '2023-01-08', '2023-01-15', '2023-01-15', '2023-01-22', '2023-01-22']),
        'HomeTeam': ['Team A', 'Team C', 'Team B', 'Team D', 'Team A', 'Team C', 'Team B', 'Team D'],
        'AwayTeam': ['Team B', 'Team D', 'Team A', 'Team C', 'Team D', 'Team B', 'Team C', 'Team A'],
        'FTHG': [1, 0, 2, 1, 3, 0, 1, 1],
        'FTAG': [1, 0, 2, 1, 1, 0, 1, 1],
        'FTR': ['D', 'D', 'D', 'D', 'H', 'D', 'D', 'D'],
        'HS': [10, 5, 12, 8, 15, 7, 9, 11], 'AS': [8, 5, 12, 8, 10, 7, 9, 11],
        'HST': [4, 1, 5, 3, 7, 2, 4, 5], 'AST': [3, 1, 5, 3, 4, 2, 4, 5],
        'HC': [5, 2, 6, 4, 8, 3, 5, 6], 'AC': [3, 2, 6, 4, 5, 3, 5, 6],
        'B365H': [2.5, 3.0, 2.8, 2.6, 1.8, 3.5, 2.9, 2.7],
        'B365D': [3.2, 3.1, 3.3, 3.2, 3.8, 3.0, 3.1, 3.2],
        'B365A': [2.8, 2.4, 2.5, 2.7, 4.5, 2.1, 2.4, 2.6]
    }
    history = pd.DataFrame(data)
    history['FTR'] = np.select([history['FTHG'] > history['FTAG'], history['FTHG'] < history['FTAG']], ['H', 'A'], default='D')

    # --- Generate Elo (Optional) ---
    elo_calc = EloCalculator()
    history_with_elo = elo_calc.calculate_historical_elos(history)

    # --- Generate Features ---
    # Pass the df with Elo ratings if calculated
    feature_set = generate_features(history_with_elo, elo_df=history_with_elo, rolling_window=2)

    print("\nGenerated Features DataFrame (Sample):")
    print(feature_set.head())
    print("\nFeatures Info:")
    feature_set.info()