# models/poisson_model.py

import pandas as pd
import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple, Optional
import joblib
import warnings

from .base_model import BaseModel # Assuming you have a base_model.py

class PoissonModel(BaseModel):
    """
    Poisson-based model for predicting football match scorelines and outcomes.

    Estimates expected goals (lambda) for home and away teams based on
    historical performance (attack/defense strength relative to league average)
    and home advantage. Uses the independent Poisson distribution assumption
    to calculate probabilities for scorelines, 1X2 outcomes, O/U, and BTTS.
    """

    def __init__(self, max_goals_limit: int = 8):
        """
        Initializes the PoissonModel.

        Args:
            max_goals_limit (int): The maximum number of goals considered per team
                                   when building the probability matrix. Higher values
                                   increase precision slightly but also computation time.
        """
        if not isinstance(max_goals_limit, int) or max_goals_limit < 0:
            raise ValueError("max_goals_limit must be a non-negative integer.")

        self.max_goals_limit = max_goals_limit
        self.league_avg_goals: Dict[str, float] = {} # Stores {'home_scored', 'away_scored'}
        self.team_stats: Dict[str, Dict[str, float]] = {} # Stores stats per team
        self.is_fitted: bool = False

    def fit(self, history_df: pd.DataFrame):
        """
        Calculates league average goals and team-specific attack/defense strengths
        from historical match data.

        Args:
            history_df (pd.DataFrame): DataFrame containing historical match results.
                                       Required columns: 'HomeTeam', 'AwayTeam',
                                       'FTHG' (Full Time Home Goals),
                                       'FTAG' (Full Time Away Goals).
        """
        print("Fitting PoissonModel...")
        required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        if not all(col in history_df.columns for col in required_cols):
            raise ValueError(f"history_df must contain columns: {required_cols}")

        # Drop rows with missing essential data if any
        history_df = history_df[required_cols].dropna()
        if history_df.empty:
             raise ValueError("Provided history_df is empty or has missing values in required columns.")

        # 1. Calculate League Average Goals
        self.league_avg_goals['home_scored'] = history_df['FTHG'].mean()
        self.league_avg_goals['away_scored'] = history_df['FTAG'].mean()
        # Note: league avg home conceded = league avg away scored, and vice-versa
        self.league_avg_goals['home_conceded'] = self.league_avg_goals['away_scored']
        self.league_avg_goals['away_conceded'] = self.league_avg_goals['home_scored']

        if self.league_avg_goals['home_scored'] == 0 or self.league_avg_goals['away_scored'] == 0:
             warnings.warn("League average goals are zero. Attack/Defense strengths might be undefined or infinite.")
             # Handle division by zero later if necessary

        print(f"  League Avg Goals: Home Scored={self.league_avg_goals['home_scored']:.3f}, Away Scored={self.league_avg_goals['away_scored']:.3f}")

        # 2. Calculate Team-Specific Stats
        teams = pd.concat([history_df['HomeTeam'], history_df['AwayTeam']]).unique()
        self.team_stats = {}

        for team in teams:
            home_matches = history_df[history_df['HomeTeam'] == team]
            away_matches = history_df[history_df['AwayTeam'] == team]

            avg_home_scored = home_matches['FTHG'].mean() if not home_matches.empty else 0
            avg_home_conceded = home_matches['FTAG'].mean() if not home_matches.empty else 0
            avg_away_scored = away_matches['FTAG'].mean() if not away_matches.empty else 0
            avg_away_conceded = away_matches['FTHG'].mean() if not away_matches.empty else 0

            # Calculate Attack Strength (relative to league average for that venue)
            # Handle potential division by zero if league average is 0
            attack_strength_home = (avg_home_scored / self.league_avg_goals['home_scored']) \
                                   if self.league_avg_goals['home_scored'] > 0 else 1.0
            attack_strength_away = (avg_away_scored / self.league_avg_goals['away_scored']) \
                                   if self.league_avg_goals['away_scored'] > 0 else 1.0

            # Calculate Defense Strength (relative to league average for that venue)
            # Lower is better, so we use conceded goals.
            # We calculate "weakness" relative to league avg conceded (which equals league avg scored by opponent)
            defense_strength_home = (avg_home_conceded / self.league_avg_goals['home_conceded']) \
                                    if self.league_avg_goals['home_conceded'] > 0 else 1.0
            defense_strength_away = (avg_away_conceded / self.league_avg_goals['away_conceded']) \
                                    if self.league_avg_goals['away_conceded'] > 0 else 1.0

            self.team_stats[team] = {
                'avg_home_scored': avg_home_scored,
                'avg_home_conceded': avg_home_conceded,
                'avg_away_scored': avg_away_scored,
                'avg_away_conceded': avg_away_conceded,
                'attack_strength_home': attack_strength_home,
                'attack_strength_away': attack_strength_away,
                'defense_strength_home': defense_strength_home, # Lower is better defense
                'defense_strength_away': defense_strength_away # Lower is better defense
            }

        self.is_fitted = True
        print(f"Fitting complete. Calculated stats for {len(teams)} teams.")

    def _get_team_stats(self, team: str) -> Dict[str, float]:
        """Safely retrieves team stats, returning league average if team is unknown."""
        if team in self.team_stats:
            return self.team_stats[team]
        else:
            warnings.warn(f"Team '{team}' not found in fitted data. Using league average strengths (1.0).")
            # Return dictionary mimicking team_stats structure but with neutral values
            return {
                'avg_home_scored': self.league_avg_goals['home_scored'],
                'avg_home_conceded': self.league_avg_goals['home_conceded'],
                'avg_away_scored': self.league_avg_goals['away_scored'],
                'avg_away_conceded': self.league_avg_goals['away_conceded'],
                'attack_strength_home': 1.0,
                'attack_strength_away': 1.0,
                'defense_strength_home': 1.0,
                'defense_strength_away': 1.0
            }

    def estimate_lambdas(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """
        Estimates the expected goals (lambda) for the home and away teams.

        Formula (inspired by Dixon-Coles basic structure):
        lambda_home = home_attack * away_defense * league_avg_home_goals
        lambda_away = away_attack * home_defense * league_avg_away_goals

        Args:
            home_team (str): Name of the home team.
            away_team (str): Name of the away team.

        Returns:
            Tuple[float, float]: (lambda_home, lambda_away)

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        home_stats = self._get_team_stats(home_team)
        away_stats = self._get_team_stats(away_team)

        # Lambda for Home Team scoring against Away Team
        lambda_home = (home_stats['attack_strength_home'] *
                       away_stats['defense_strength_away'] * # Away team's defense when playing away
                       self.league_avg_goals['home_scored'])

        # Lambda for Away Team scoring against Home Team
        lambda_away = (away_stats['attack_strength_away'] *
                       home_stats['defense_strength_home'] * # Home team's defense when playing home
                       self.league_avg_goals['away_scored'])

        # Ensure lambdas are non-negative (can happen with extreme stats or zero averages)
        lambda_home = max(0, lambda_home)
        lambda_away = max(0, lambda_away)

        return lambda_home, lambda_away

    def predict_score_distribution(self, lambda_home: float, lambda_away: float) -> pd.DataFrame:
        """
        Calculates the probability distribution of scorelines using independent Poisson distributions.

        Args:
            lambda_home (float): Estimated expected goals for the home team.
            lambda_away (float): Estimated expected goals for the away team.

        Returns:
            pd.DataFrame: A matrix where index = home goals (0 to max_goals_limit),
                          columns = away goals (0 to max_goals_limit), and
                          values = probability of that exact scoreline.
                          The sum of the matrix might be slightly less than 1 if
                          max_goals_limit is low, representing the ignored probability
                          of higher scores.
        """
        if lambda_home < 0 or lambda_away < 0:
             raise ValueError("Lambda values cannot be negative.")

        max_g = self.max_goals_limit
        home_goals_probs = poisson.pmf(k=np.arange(max_g + 1), mu=lambda_home)
        away_goals_probs = poisson.pmf(k=np.arange(max_g + 1), mu=lambda_away)

        # Calculate the outer product to get the matrix of joint probabilities P(h, a) = P(h) * P(a)
        score_prob_matrix = np.outer(home_goals_probs, away_goals_probs)

        # Create DataFrame for better readability
        score_dist_df = pd.DataFrame(
            score_prob_matrix,
            index=pd.Index(range(max_g + 1), name='Home Goals'),
            columns=pd.Index(range(max_g + 1), name='Away Goals')
        )
        return score_dist_df

    def _calculate_1x2_probs(self, score_dist_df: pd.DataFrame) -> Dict[str, float]:
        """Calculates Home Win (H), Draw (D), Away Win (A) probabilities from score matrix."""
        prob_h = np.sum(np.tril(score_dist_df.values, k=-1)) # Sum lower triangle (excluding diagonal)
        prob_d = np.sum(np.diag(score_dist_df.values))      # Sum diagonal
        prob_a = np.sum(np.triu(score_dist_df.values, k=1))  # Sum upper triangle (excluding diagonal)

        # Normalize slightly in case matrix sum < 1 due to max_goals_limit
        total_prob = prob_h + prob_d + prob_a
        if total_prob > 0 and not np.isclose(total_prob, 1.0):
             prob_h /= total_prob
             prob_d /= total_prob
             prob_a /= total_prob

        return {'H': prob_h, 'D': prob_d, 'A': prob_a}

    def _calculate_over_under_probs(self, score_dist_df: pd.DataFrame, lines: list = [0.5, 1.5, 2.5, 3.5, 4.5]) -> Dict[str, float]:
        """Calculates Over/Under probabilities for given goal lines."""
        over_under_probs = {}
        total_goals_matrix = score_dist_df.index.values[:, None] + score_dist_df.columns.values[None, :]

        for line in lines:
            prob_over = score_dist_df.values[total_goals_matrix > line].sum()
            over_under_probs[f'Over {line}'] = prob_over
            over_under_probs[f'Under {line}'] = 1.0 - prob_over # Approximation if matrix sum < 1
            # More accurate under: score_dist_df.values[total_goals_matrix <= line].sum()

        return over_under_probs

    def _calculate_btts_prob(self, score_dist_df: pd.DataFrame) -> float:
        """Calculates Both Teams To Score (BTTS) probability."""
        # Sum probabilities where home_goals > 0 AND away_goals > 0
        prob_btts = score_dist_df.iloc[1:, 1:].values.sum()
        return prob_btts

    def predict(self, data: Dict[str, str]) -> Dict[str, any]:
        """
        Predicts probabilities for a match given home and away team names.

        Args:
            data (Dict[str, str]): Dictionary containing 'HomeTeam' and 'AwayTeam'.
                                   Example: {'HomeTeam': 'Arsenal', 'AwayTeam': 'Chelsea'}

        Returns:
            Dict[str, any]: A dictionary containing:
                - 'lambda_home': Estimated expected goals for home team.
                - 'lambda_away': Estimated expected goals for away team.
                - 'outcome_probs': {'H', 'D', 'A'} probabilities.
                - 'over_under_probs': Probabilities for standard O/U lines.
                - 'btts_prob': Both Teams To Score probability.
                - 'score_distribution': DataFrame of scoreline probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        if 'HomeTeam' not in data or 'AwayTeam' not in data:
            raise KeyError("Input data must contain 'HomeTeam' and 'AwayTeam'.")

        home_team = data['HomeTeam']
        away_team = data['AwayTeam']

        # 1. Estimate Lambdas
        lambda_home, lambda_away = self.estimate_lambdas(home_team, away_team)

        # 2. Get Score Distribution
        score_dist_df = self.predict_score_distribution(lambda_home, lambda_away)

        # 3. Calculate Aggregate Probabilities
        outcome_probs = self._calculate_1x2_probs(score_dist_df)
        over_under_probs = self._calculate_over_under_probs(score_dist_df)
        btts_prob = self._calculate_btts_prob(score_dist_df)

        return {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'outcome_probs': outcome_probs,
            'over_under_probs': over_under_probs,
            'btts_prob': btts_prob,
            'score_distribution': score_dist_df
        }

    def save(self, filepath: str):
        """Saves the fitted model state (league averages and team stats)."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        print(f"Saving Poisson model state to {filepath}...")
        state = {
            'league_avg_goals': self.league_avg_goals,
            'team_stats': self.team_stats,
            'max_goals_limit': self.max_goals_limit
        }
        joblib.dump(state, filepath)
        print("Model state saved successfully.")

    @classmethod
    def load(cls, filepath: str):
        """Loads a fitted model state."""
        print(f"Loading Poisson model state from {filepath}...")
        state = joblib.load(filepath)
        # Create instance with saved config
        model_instance = cls(max_goals_limit=state.get('max_goals_limit', 8)) # Use default if missing
        # Load the fitted attributes
        model_instance.league_avg_goals = state['league_avg_goals']
        model_instance.team_stats = state['team_stats']
        model_instance.is_fitted = True
        print("Model state loaded successfully.")
        return model_instance

# Example Usage
if __name__ == '__main__':
    # --- 1. Create Dummy Historical Data ---
    data = {
        'HomeTeam': ['Team A', 'Team C', 'Team A', 'Team B', 'Team D', 'Team B'],
        'AwayTeam': ['Team B', 'Team D', 'Team C', 'Team A', 'Team A', 'Team C'],
        'FTHG': [2, 0, 1, 3, 1, 2],
        'FTAG': [1, 0, 1, 1, 2, 2]
    }
    history = pd.DataFrame(data)
    print("Dummy Historical Data:")
    print(history)

    # --- 2. Fit the Model ---
    poisson_model = PoissonModel(max_goals_limit=6)
    poisson_model.fit(history)

    # --- 3. Predict a Match ---
    match_to_predict = {'HomeTeam': 'Team A', 'AwayTeam': 'Team B'}
    print(f"\nPredicting match: {match_to_predict['HomeTeam']} vs {match_to_predict['AwayTeam']}")
    try:
        predictions = poisson_model.predict(match_to_predict)

        # --- 4. Display Results ---
        print("\n--- Prediction Results ---")
        print(f"Estimated Lambdas: Home={predictions['lambda_home']:.3f}, Away={predictions['lambda_away']:.3f}")

        print("\nOutcome Probabilities (1X2):")
        for outcome, prob in predictions['outcome_probs'].items():
            implied_odds = 1 / prob if prob > 0 else float('inf')
            print(f"  P({outcome}): {prob:.4f} (Odds: {implied_odds:.2f})")

        print(f"\nBTTS Probability: {predictions['btts_prob']:.4f}")

        print("\nOver/Under Probabilities:")
        for line in [0.5, 1.5, 2.5, 3.5]:
            over_key = f'Over {line}'
            under_key = f'Under {line}'
            print(f"  Over {line}: {predictions['over_under_probs'][over_key]:.4f} | Under {line}: {predictions['over_under_probs'][under_key]:.4f}")

        print("\nScore Distribution Matrix (Top Left):")
        print(predictions['score_distribution'].iloc[:4, :4].round(4))

        # Find most likely score
        score_df = predictions['score_distribution']
        most_likely_idx = np.unravel_index(np.argmax(score_df.values, axis=None), score_df.shape)
        most_likely_score = (score_df.index[most_likely_idx[0]], score_df.columns[most_likely_idx[1]])
        most_likely_prob = score_df.values[most_likely_idx]
        print(f"\nMost Likely Score: {most_likely_score[0]}-{most_likely_score[1]} (Prob: {most_likely_prob:.4f})")

    except RuntimeError as e:
        print(f"Error during prediction: {e}")
    except KeyError as e:
        print(f"Error during prediction (likely missing team): {e}")

    # --- 5. Save and Load ---
    model_path = "temp_poisson_model.joblib"
    poisson_model.save(model_path)
    loaded_model = PoissonModel.load(model_path)

    # Verify loaded model prediction
    print("\nVerifying loaded model...")
    try:
        loaded_predictions = loaded_model.predict(match_to_predict)
        # Basic check (more rigorous checks needed for DataFrames/floats)
        assert abs(predictions['lambda_home'] - loaded_predictions['lambda_home']) < 1e-6
        assert abs(predictions['outcome_probs']['H'] - loaded_predictions['outcome_probs']['H']) < 1e-6
        print("Save/Load test passed (basic check).")
    except Exception as e:
        print(f"Error during loaded model verification: {e}")

    # Clean up
    import os
    if os.path.exists(model_path): os.remove(model_path)