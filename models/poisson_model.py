# models/poisson_model_enhanced.py

import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize # Added for potential future MLE fitting
from typing import Dict, Tuple, List, Optional
import joblib
import warnings
import datetime

# Assuming you have a base_model.py or just remove the inheritance if not
# from .base_model import BaseModel
# class PoissonModelEnhanced(BaseModel):
class PoissonModelEnhanced:
    """
    Enhanced Poisson-based model for predicting football match outcomes.

    Incorporates:
    - Time weighting (Dixon-Coles style) for recent form influence.
    - Explicit home advantage factor.
    - Unified team attack/defense strengths relative to overall league average.
    - Calculation of probabilities for scorelines, 1X2, O/U, and BTTS.
    """

    def __init__(self, max_goals_limit: int = 8, xi: float = 0.0019):
        """
        Initializes the PoissonModelEnhanced.

        Args:
            max_goals_limit (int): Max goals considered per team for probability matrix.
            xi (float): Time weighting coefficient (epsilon in Dixon-Coles).
                        Controls how quickly influence of older matches decays.
                        Default (0.0019) roughly halves influence every year (365 days).
                        Set to 0 for no time weighting.
        """
        if not isinstance(max_goals_limit, int) or max_goals_limit < 0:
            raise ValueError("max_goals_limit must be a non-negative integer.")
        if not isinstance(xi, (float, int)) or xi < 0:
             raise ValueError("xi (time_weighting_coeff) must be a non-negative number.")

        self.max_goals_limit = max_goals_limit
        self.xi = xi # Time weighting coefficient
        self.league_avg_goals: Dict[str, float] = {} # Stores {'home', 'away', 'total', 'overall_per_team'}
        self.home_advantage: Dict[str, float] = {} # Stores {'ratio', 'home_lambda_factor', 'away_lambda_factor'}
        self.team_params: Dict[str, Dict[str, float]] = {} # Stores {'attack', 'defense'} per team
        self.last_history_date: Optional[datetime.datetime] = None
        self.is_fitted: bool = False

    def _calculate_time_weights(self, dates: pd.Series) -> np.ndarray:
        """Calculates exponential time decay weights."""
        if self.xi == 0:
            # No weighting, return array of ones
            return np.ones(len(dates))

        if self.last_history_date is None:
             raise RuntimeError("Cannot calculate weights before setting last_history_date.")

        # Ensure dates are datetime objects
        dates = pd.to_datetime(dates)
        time_diff_days = (self.last_history_date - dates).dt.total_seconds() / (24 * 60 * 60)
        weights = np.exp(-self.xi * time_diff_days)
        # Normalize weights to sum to N (optional, but can help stabilize averages)
        # weights = weights * len(dates) / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(dates))
        return weights


    def fit(self, history_df: pd.DataFrame):
        """
        Calculates time-weighted league averages, home advantage, and team strengths.

        Args:
            history_df (pd.DataFrame): DataFrame containing historical match results.
                                       Required columns: 'HomeTeam', 'AwayTeam',
                                       'FTHG', 'FTAG', 'Date' (or 'Timestamp').
        """
        print("Fitting Enhanced PoissonModel...")
        date_col = 'Date' if 'Date' in history_df.columns else 'Timestamp'
        required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', date_col]
        if not all(col in history_df.columns for col in required_cols):
            raise ValueError(f"history_df must contain columns: {required_cols}")

        # Prepare data
        fit_df = history_df[required_cols].dropna().copy()
        if fit_df.empty:
             raise ValueError("Provided history_df is empty or has missing values in required columns.")
        fit_df[date_col] = pd.to_datetime(fit_df[date_col])
        self.last_history_date = fit_df[date_col].max() # Needed for weighting

        # --- Calculate Time Weights ---
        weights = self._calculate_time_weights(fit_df[date_col])
        if np.sum(weights) == 0: # Handle case where all weights might become zero (e.g., extreme xi)
            weights = np.ones(len(fit_df))
            warnings.warn("Sum of time weights is zero. Disabling time weighting for this fit.")

        # --- 1. Calculate Weighted League Average Goals ---
        avg_home_scored = np.average(fit_df['FTHG'], weights=weights)
        avg_away_scored = np.average(fit_df['FTAG'], weights=weights)
        self.league_avg_goals = {
            'home': avg_home_scored,
            'away': avg_away_scored,
            'total': avg_home_scored + avg_away_scored,
            # Overall average goals per team per match (used as baseline)
            'overall_per_team': (avg_home_scored + avg_away_scored) / 2.0
        }
        # Avoid division by zero later
        if self.league_avg_goals['overall_per_team'] == 0:
            warnings.warn("Weighted overall league average goals is zero. Strengths may be unreliable.")
            self.league_avg_goals['overall_per_team'] = 1e-6 # Set small floor

        print(f"  Weighted League Avgs: Home={avg_home_scored:.3f}, Away={avg_away_scored:.3f}, OverallPerTeam={self.league_avg_goals['overall_per_team']:.3f}")

        # --- 2. Calculate Explicit Home Advantage ---
        # Simple ratio method based on weighted averages
        if avg_away_scored > 0:
            ha_ratio = avg_home_scored / avg_away_scored
            # Convert ratio to multiplicative factors for home/away lambdas centered around 1
            # e.g. if ratio is 1.2, home factor > 1, away factor < 1
            home_factor = np.sqrt(ha_ratio)
            away_factor = 1.0 / home_factor
        else: # Handle case where away average is zero
            warnings.warn("Weighted average away goals is zero. Setting neutral home advantage (1.0).")
            ha_ratio = 1.0
            home_factor = 1.0
            away_factor = 1.0

        self.home_advantage = {
            'ratio': ha_ratio,
            'home_lambda_factor': home_factor,
            'away_lambda_factor': away_factor,
        }
        print(f"  Home Advantage Ratio: {ha_ratio:.3f} (Factors: Home={home_factor:.3f}, Away={away_factor:.3f})")

        # --- 3. Calculate Weighted Team-Specific Strengths (Unified) ---
        teams = pd.concat([fit_df['HomeTeam'], fit_df['AwayTeam']]).unique()
        self.team_params = {}
        baseline_avg = self.league_avg_goals['overall_per_team']

        for team in teams:
            team_home_df = fit_df[fit_df['HomeTeam'] == team]
            team_away_df = fit_df[fit_df['AwayTeam'] == team]
            home_weights = weights[fit_df['HomeTeam'] == team]
            away_weights = weights[fit_df['AwayTeam'] == team]

            # Weighted average goals scored BY the team (home and away)
            scored_home = np.average(team_home_df['FTHG'], weights=home_weights) if not team_home_df.empty else 0
            scored_away = np.average(team_away_df['FTAG'], weights=away_weights) if not team_away_df.empty else 0
            total_scored_matches = len(team_home_df) + len(team_away_df)
            # Combine home/away scored average (weighted by number of home/away games - approx)
            w_home = home_weights.sum() if not team_home_df.empty else 0
            w_away = away_weights.sum() if not team_away_df.empty else 0
            w_total = w_home + w_away
            avg_scored = ((scored_home * w_home + scored_away * w_away) / w_total) if w_total > 0 else 0

            # Weighted average goals conceded BY the team (home and away)
            conceded_home = np.average(team_home_df['FTAG'], weights=home_weights) if not team_home_df.empty else 0
            conceded_away = np.average(team_away_df['FTHG'], weights=away_weights) if not team_away_df.empty else 0
            avg_conceded = ((conceded_home * w_home + conceded_away * w_away) / w_total) if w_total > 0 else 0

            # Strengths relative to overall league average per team
            attack_strength = avg_scored / baseline_avg
            defense_strength = avg_conceded / baseline_avg # Lower is better

            self.team_params[team] = {
                'attack': attack_strength,
                'defense': defense_strength,
                # Store raw weighted averages for info if needed
                '_debug_avg_scored': avg_scored,
                '_debug_avg_conceded': avg_conceded
            }

        self.is_fitted = True
        print(f"Fitting complete. Calculated parameters for {len(teams)} teams.")


    def _get_team_params(self, team: str) -> Dict[str, float]:
        """Safely retrieves team parameters, returning neutral values (1.0) if unknown."""
        if team in self.team_params:
            return self.team_params[team]
        else:
            warnings.warn(f"Team '{team}' not found in fitted data. Using neutral strengths (1.0).")
            return {'attack': 1.0, 'defense': 1.0}


    def estimate_lambdas(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """
        Estimates the expected goals (lambda) incorporating unified strengths
        and explicit home advantage.

        Formula Idea:
        lambda_home = home_attack * away_defense * league_avg_overall * home_adv_factor
        lambda_away = away_attack * home_defense * league_avg_overall * away_adv_factor

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

        home_params = self._get_team_params(home_team)
        away_params = self._get_team_params(away_team)
        baseline_avg = self.league_avg_goals['overall_per_team']
        home_factor = self.home_advantage['home_lambda_factor']
        away_factor = self.home_advantage['away_lambda_factor']

        # Estimate lambda for Home Team
        lambda_home = (home_params['attack'] *
                       away_params['defense'] *
                       baseline_avg *
                       home_factor)

        # Estimate lambda for Away Team
        lambda_away = (away_params['attack'] *
                       home_params['defense'] *
                       baseline_avg *
                       away_factor)

        # Ensure lambdas are non-negative
        lambda_home = max(1e-6, lambda_home) # Use small floor > 0
        lambda_away = max(1e-6, lambda_away)

        return lambda_home, lambda_away


    def predict_score_distribution(self, lambda_home: float, lambda_away: float) -> pd.DataFrame:
        """ Calculates the probability distribution of scorelines (Identical to previous). """
        # (This function remains the same as in the original PoissonModel)
        if lambda_home <= 0 or lambda_away <= 0:
             # Lambda can technically be 0, Poisson PMF handles it, but <0 is invalid
             raise ValueError("Lambda values must be non-negative.")

        max_g = self.max_goals_limit
        home_goals_probs = poisson.pmf(k=np.arange(max_g + 1), mu=lambda_home)
        away_goals_probs = poisson.pmf(k=np.arange(max_g + 1), mu=lambda_away)
        score_prob_matrix = np.outer(home_goals_probs, away_goals_probs)
        score_dist_df = pd.DataFrame(
            score_prob_matrix,
            index=pd.Index(range(max_g + 1), name='Home Goals'),
            columns=pd.Index(range(max_g + 1), name='Away Goals')
        )
        return score_dist_df

    # --- Functions to calculate outcome probabilities from score distribution ---
    # --- (These _calculate_* methods are identical to the previous version) ---

    def _calculate_1x2_probs(self, score_dist_df: pd.DataFrame) -> Dict[str, float]:
        """Calculates Home Win (H), Draw (D), Away Win (A) probabilities."""
        prob_h = np.sum(np.tril(score_dist_df.values, k=-1))
        prob_d = np.sum(np.diag(score_dist_df.values))
        prob_a = np.sum(np.triu(score_dist_df.values, k=1))
        # Normalize slightly if matrix sum < 1 due to max_goals_limit
        total_prob = prob_h + prob_d + prob_a
        if total_prob > 1e-9 and not np.isclose(total_prob, 1.0):
             scale = 1.0 / total_prob
             prob_h *= scale
             prob_d *= scale
             prob_a *= scale
        return {'H': prob_h, 'D': prob_d, 'A': prob_a}

    def _calculate_over_under_probs(self, score_dist_df: pd.DataFrame, lines: list = [0.5, 1.5, 2.5, 3.5, 4.5]) -> Dict[str, float]:
        """Calculates Over/Under probabilities for given goal lines."""
        over_under_probs = {}
        total_goals_matrix = score_dist_df.index.values[:, None] + score_dist_df.columns.values[None, :]
        # Pre-calculate sum for normalization check
        total_prob_in_matrix = score_dist_df.values.sum()
        normalization_factor = 1.0 / total_prob_in_matrix if total_prob_in_matrix > 1e-9 and not np.isclose(total_prob_in_matrix, 1.0) else 1.0

        for line in lines:
            prob_over = score_dist_df.values[total_goals_matrix > line].sum() * normalization_factor
            # Calculate Under by summing relevant cells for better accuracy if matrix sum < 1
            prob_under = score_dist_df.values[total_goals_matrix <= line].sum() * normalization_factor
            # Ensure consistency
            prob_under = max(0.0, 1.0 - prob_over) if np.isclose(prob_over + prob_under, 1.0) else prob_under

            over_under_probs[f'Over {line}'] = prob_over
            over_under_probs[f'Under {line}'] = prob_under

        return over_under_probs

    def _calculate_btts_prob(self, score_dist_df: pd.DataFrame) -> Dict[str, float]:
        """Calculates Both Teams To Score (BTTS) Yes/No probability."""
        # Sum probabilities where home_goals > 0 AND away_goals > 0
        prob_btts_yes = score_dist_df.iloc[1:, 1:].values.sum()
        # Normalize
        total_prob_in_matrix = score_dist_df.values.sum()
        if total_prob_in_matrix > 1e-9 and not np.isclose(total_prob_in_matrix, 1.0):
            prob_btts_yes /= total_prob_in_matrix

        return {'BTTS Yes': prob_btts_yes, 'BTTS No': 1.0 - prob_btts_yes}

    def predict(self, data: Dict[str, str]) -> Dict[str, any]:
        """
        Predicts probabilities for a match given home and away team names.

        Args:
            data (Dict[str, str]): Dictionary containing 'HomeTeam' and 'AwayTeam'.

        Returns:
            Dict[str, any]: Dictionary with lambda estimates, 1X2 probs, O/U probs,
                          BTTS probs, and the score distribution DataFrame.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        if 'HomeTeam' not in data or 'AwayTeam' not in data:
            raise KeyError("Input data must contain 'HomeTeam' and 'AwayTeam'.")

        home_team = data['HomeTeam']
        away_team = data['AwayTeam']

        lambda_home, lambda_away = self.estimate_lambdas(home_team, away_team)
        score_dist_df = self.predict_score_distribution(lambda_home, lambda_away)

        outcome_probs = self._calculate_1x2_probs(score_dist_df)
        over_under_probs = self._calculate_over_under_probs(score_dist_df) # Uses default lines
        btts_probs = self._calculate_btts_prob(score_dist_df)

        # Combine O/U and BTTS results
        combined_probs = {}
        combined_probs.update(over_under_probs)
        combined_probs.update(btts_probs)

        return {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'outcome_probs_1X2': outcome_probs,
            'outcome_probs_OU_BTTS': combined_probs, # Merged O/U and BTTS
            'score_distribution': score_dist_df
        }

    def save(self, filepath: str):
        """Saves the fitted model state."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        print(f"Saving Enhanced Poisson model state to {filepath}...")
        state = {
            'league_avg_goals': self.league_avg_goals,
            'home_advantage': self.home_advantage,
            'team_params': self.team_params,
            'max_goals_limit': self.max_goals_limit,
            'xi': self.xi,
            'last_history_date': self.last_history_date
        }
        joblib.dump(state, filepath)
        print("Model state saved successfully.")

    @classmethod
    def load(cls, filepath: str):
        """Loads a fitted model state."""
        print(f"Loading Enhanced Poisson model state from {filepath}...")
        state = joblib.load(filepath)
        model_instance = cls(
            max_goals_limit=state.get('max_goals_limit', 8),
            xi=state.get('xi', 0.0019) # Load xi, use default if missing
        )
        model_instance.league_avg_goals = state['league_avg_goals']
        model_instance.home_advantage = state['home_advantage']
        model_instance.team_params = state['team_params']
        model_instance.last_history_date = state.get('last_history_date') # Load last date
        model_instance.is_fitted = True
        print("Model state loaded successfully.")
        return model_instance

# Example Usage
if __name__ == '__main__':
    # --- 1. Create Dummy Historical Data with Dates ---
    dates = pd.to_datetime([
        '2023-01-01', '2023-01-02', '2023-01-08',
        '2023-01-09', '2023-01-15', '2023-01-16',
        '2023-08-01', '2023-08-02', # More recent games
    ])
    data = {
        'Date': dates,
        'HomeTeam': ['Team A', 'Team C', 'Team A', 'Team B', 'Team D', 'Team B', 'Team A', 'Team B'],
        'AwayTeam': ['Team B', 'Team D', 'Team C', 'Team A', 'Team A', 'Team C', 'Team D', 'Team C'],
        'FTHG': [2, 0, 1, 3, 1, 2, 3, 1], # Team A scores more recently
        'FTAG': [1, 0, 1, 1, 2, 2, 0, 0]  # Team C concedes less recently
    }
    history = pd.DataFrame(data)
    print("Dummy Historical Data:")
    print(history)

    # --- 2. Fit the Enhanced Model ---
    # Use default xi for time weighting
    poisson_model = PoissonModelEnhanced(max_goals_limit=6, xi=0.002)
    poisson_model.fit(history)

    # --- 3. Predict a Match ---
    match_to_predict = {'HomeTeam': 'Team A', 'AwayTeam': 'Team C'}
    print(f"\nPredicting match: {match_to_predict['HomeTeam']} vs {match_to_predict['AwayTeam']}")
    try:
        predictions = poisson_model.predict(match_to_predict)

        # --- 4. Display Results ---
        print("\n--- Prediction Results ---")
        print(f"Estimated Lambdas: Home={predictions['lambda_home']:.3f}, Away={predictions['lambda_away']:.3f}")

        print("\nOutcome Probabilities (1X2):")
        for outcome, prob in predictions['outcome_probs_1X2'].items():
            implied_odds = 1 / prob if prob > 0 else float('inf')
            print(f"  P({outcome}): {prob:.4f} (Odds: {implied_odds:.2f})")

        print("\nOver/Under & BTTS Probabilities:")
        probs_ou_btts = predictions['outcome_probs_OU_BTTS']
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            over_key = f'Over {line}'
            under_key = f'Under {line}'
            if over_key in probs_ou_btts:
                 print(f"  Over {line}: {probs_ou_btts[over_key]:.4f} | Under {line}: {probs_ou_btts[under_key]:.4f}")
        if 'BTTS Yes' in probs_ou_btts:
            print(f"  BTTS Yes: {probs_ou_btts['BTTS Yes']:.4f} | BTTS No: {probs_ou_btts['BTTS No']:.4f}")


        print("\nScore Distribution Matrix (Top Left):")
        print(predictions['score_distribution'].iloc[:4, :4].round(4))

        # Find most likely score
        score_df = predictions['score_distribution']
        if not score_df.empty:
            most_likely_idx = np.unravel_index(np.argmax(score_df.values, axis=None), score_df.shape)
            most_likely_score = (score_df.index[most_likely_idx[0]], score_df.columns[most_likely_idx[1]])
            most_likely_prob = score_df.values[most_likely_idx]
            print(f"\nMost Likely Score: {most_likely_score[0]}-{most_likely_score[1]} (Prob: {most_likely_prob:.4f})")

    except RuntimeError as e:
        print(f"Error during prediction: {e}")
    except KeyError as e:
        print(f"Error during prediction (likely missing team): {e}")

    # --- 5. Save and Load ---
    model_path = "temp_poisson_enhanced_model.joblib"
    poisson_model.save(model_path)
    loaded_model = PoissonModelEnhanced.load(model_path)

    # Verify loaded model prediction
    print("\nVerifying loaded model...")
    try:
        loaded_predictions = loaded_model.predict(match_to_predict)
        assert abs(predictions['lambda_home'] - loaded_predictions['lambda_home']) < 1e-6
        assert abs(predictions['outcome_probs_1X2']['H'] - loaded_predictions['outcome_probs_1X2']['H']) < 1e-6
        assert abs(predictions['outcome_probs_OU_BTTS']['Over 2.5'] - loaded_predictions['outcome_probs_OU_BTTS']['Over 2.5']) < 1e-6
        print("Save/Load test passed (basic check).")
    except Exception as e:
        print(f"Error during loaded model verification: {e}")

    # Clean up
    import os
    if os.path.exists(model_path): os.remove(model_path)