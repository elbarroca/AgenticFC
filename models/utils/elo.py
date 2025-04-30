# utils/elo.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import math

class EloCalculator:
    """
    Calculates and updates Elo ratings for teams based on match results.

    Attributes:
        k_factor (int): The maximum rating change per match. Higher values mean faster changes.
        home_advantage (int): Elo points added to the home team's rating for calculation purposes.
        default_rating (int): The initial rating assigned to teams not yet seen.
        team_ratings (Dict[str, int]): Dictionary storing the current Elo rating for each team.
    """

    def __init__(self, k_factor: int = 20, home_advantage: int = 65, default_rating: int = 1500):
        """
        Initializes the EloCalculator.

        Args:
            k_factor (int): The K-factor determining rating volatility.
            home_advantage (int): Elo points added to home team's effective rating.
            default_rating (int): Starting Elo rating for new teams.
        """
        if k_factor <= 0:
            raise ValueError("k_factor must be positive.")
        if home_advantage < 0:
            raise ValueError("home_advantage cannot be negative.")
        if default_rating <= 0:
             raise ValueError("default_rating must be positive.")

        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.default_rating = default_rating
        self.team_ratings: Dict[str, int] = {}
        print(f"EloCalculator initialized: K={k_factor}, HA={home_advantage}, Default={default_rating}")

    def get_rating(self, team: str) -> int:
        """Gets the current rating for a team, returning default if unknown."""
        return self.team_ratings.get(team, self.default_rating)

    def _expected_score(self, rating_a: int, rating_b: int) -> float:
        """
        Calculates the expected score (probability of winning) for team A against team B.
        Uses the standard Elo formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400)).

        Args:
            rating_a (int): Elo rating of team A.
            rating_b (int): Elo rating of team B.

        Returns:
            float: The expected score for team A (between 0 and 1).
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def update_ratings(self, home_team: str, away_team: str, ftr: str):
        """
        Updates the Elo ratings for home and away teams based on the match result (FTR).

        Args:
            home_team (str): Name of the home team.
            away_team (str): Name of the away team.
            ftr (str): Full Time Result ('H' for home win, 'D' for draw, 'A' for away win).
        """
        rating_home = self.get_rating(home_team)
        rating_away = self.get_rating(away_team)

        # Calculate expected scores considering home advantage
        expected_home = self._expected_score(rating_home + self.home_advantage, rating_away)
        expected_away = 1.0 - expected_home # E_A + E_B = 1

        # Determine actual scores based on FTR
        if ftr == 'H':
            actual_home, actual_away = 1.0, 0.0
        elif ftr == 'D':
            actual_home, actual_away = 0.5, 0.5
        elif ftr == 'A':
            actual_home, actual_away = 0.0, 1.0
        else:
            # Invalid FTR, no update
            print(f"Warning: Invalid FTR '{ftr}' for {home_team} vs {away_team}. No rating update.")
            return

        # Calculate new ratings
        new_rating_home = rating_home + self.k_factor * (actual_home - expected_home)
        new_rating_away = rating_away + self.k_factor * (actual_away - expected_away)

        # Update stored ratings (round to integer)
        self.team_ratings[home_team] = int(round(new_rating_home))
        self.team_ratings[away_team] = int(round(new_rating_away))

    def calculate_historical_elos(self, history_df: pd.DataFrame, date_col: str = 'Date',
                                  home_col: str = 'HomeTeam', away_col: str = 'AwayTeam',
                                  ftr_col: str = 'FTR') -> pd.DataFrame:
        """
        Calculates Elo ratings *before* each match in a historical DataFrame.

        Processes matches chronologically and adds 'HomeEloBefore' and 'AwayEloBefore' columns.

        Args:
            history_df (pd.DataFrame): DataFrame with historical matches. Must contain
                                       date_col, home_col, away_col, ftr_col.
            date_col (str): Name of the column containing match dates (for sorting).
            home_col (str): Name of the home team column.
            away_col (str): Name of the away team column.
            ftr_col (str): Name of the Full Time Result column ('H', 'D', 'A').

        Returns:
            pd.DataFrame: The original DataFrame with two new columns:
                          'HomeEloBefore' and 'AwayEloBefore'.

        Raises:
            KeyError: If required columns are missing in history_df.
        """
        print("Calculating historical Elo ratings...")
        required_cols = [date_col, home_col, away_col, ftr_col]
        if not all(col in history_df.columns for col in required_cols):
            raise KeyError(f"history_df must contain columns: {required_cols}")

        # Ensure data is sorted chronologically
        df_sorted = history_df.sort_values(by=date_col).copy()

        home_elos_before = []
        away_elos_before = []
        self.team_ratings = {} # Reset ratings for historical calculation

        # Iterate through matches row by row
        for index, row in df_sorted.iterrows():
            home_team = row[home_col]
            away_team = row[away_col]
            ftr = row[ftr_col]

            # Get ratings *before* the current match
            home_rating_before = self.get_rating(home_team)
            away_rating_before = self.get_rating(away_team)
            home_elos_before.append(home_rating_before)
            away_elos_before.append(away_rating_before)

            # Update ratings *after* recording the 'before' state
            self.update_ratings(home_team, away_team, ftr)

        # Add the calculated 'before' ratings to the DataFrame
        df_sorted['HomeEloBefore'] = home_elos_before
        df_sorted['AwayEloBefore'] = away_elos_before
        print("Historical Elo calculation complete.")
        return df_sorted

# Example Usage (Optional)
if __name__ == '__main__':
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-08', '2023-01-08']),
        'HomeTeam': ['Team A', 'Team C', 'Team B', 'Team D'],
        'AwayTeam': ['Team B', 'Team D', 'Team A', 'Team C'],
        'FTR': ['H', 'D', 'A', 'H']
    }
    history = pd.DataFrame(data)

    elo_calc = EloCalculator(k_factor=25, home_advantage=70)
    history_with_elo = elo_calc.calculate_historical_elos(history)

    print("\nHistorical Data with Elo Ratings (Before Match):")
    print(history_with_elo[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'HomeEloBefore', 'AwayEloBefore']])

    print("\nFinal Elo Ratings after processing:")
    print(elo_calc.team_ratings)