# models/poisson_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson
from typing import Dict, Any, Tuple
import warnings

from models.utils.features import BaseFeatureConfig
from models.base_model import BaseModel

def _calculate_dual_conditions(
    hg: int, ag: int
) -> Dict[str, bool]:
    """Helper to evaluate conditions for various dual outcomes for a given scoreline."""
    total_goals = hg + ag
    home_win = hg > ag
    draw = hg == ag
    away_win = hg < ag
    over_05 = total_goals > 0.5
    under_05 = not over_05
    over_15 = total_goals > 1.5
    under_15 = not over_15
    over_25 = total_goals > 2.5
    under_25 = not over_25
    over_35 = total_goals > 3.5
    under_35 = not over_35
    over_45 = total_goals > 4.5
    under_45 = not over_45
    btts_yes = hg > 0 and ag > 0
    btts_no = not btts_yes

    return {
        # 1X2 & O/U 2.5
        'H_and_O25': home_win and over_25,
        'D_and_O25': draw and over_25,
        'A_and_O25': away_win and over_25,
        'H_and_U25': home_win and under_25,
        'D_and_U25': draw and under_25,
        'A_and_U25': away_win and under_25,
        # Double Chance & O/U 2.5
        '1X_and_O25': (home_win or draw) and over_25,
        '12_and_O25': (home_win or away_win) and over_25,
        'X2_and_O25': (draw or away_win) and over_25,
        '1X_and_U25': (home_win or draw) and under_25,
        '12_and_U25': (home_win or away_win) and under_25,
        'X2_and_U25': (draw or away_win) and under_25,
        # 1X2 & BTTS
        'H_and_BTTS_Y': home_win and btts_yes,
        'D_and_BTTS_Y': draw and btts_yes,
        'A_and_BTTS_Y': away_win and btts_yes,
        'H_and_BTTS_N': home_win and btts_no,
        'D_and_BTTS_N': draw and btts_no,
        'A_and_BTTS_N': away_win and btts_no,
         # Double Chance & BTTS
        '1X_and_BTTS_Y': (home_win or draw) and btts_yes,
        '12_and_BTTS_Y': (home_win or away_win) and btts_yes,
        'X2_and_BTTS_Y': (draw or away_win) and btts_yes,
        '1X_and_BTTS_N': (home_win or draw) and btts_no,
        '12_and_BTTS_N': (home_win or away_win) and btts_no,
        'X2_and_BTTS_N': (draw or away_win) and btts_no,
        # O/U 2.5 & BTTS
        'O25_and_BTTS_Y': over_25 and btts_yes,
        'O25_and_BTTS_N': over_25 and btts_no,
        'U25_and_BTTS_Y': under_25 and btts_yes,
        'U25_and_BTTS_N': under_25 and btts_no,
    }


def calculate_poisson_outcome_probs(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    max_goals: int = 10 # Number of goals to sum probabilities over
) -> Dict[str, np.ndarray]:
    """
    Calculates 1X2, multiple O/U lines, BTTS, specific goal band, AND
    accurate dual-outcome probabilities from Poisson lambdas via scoreline summation.

    Args:
        lambda_home: Predicted expected goals for the home team (1D array).
        lambda_away: Predicted expected goals for the away team (1D array).
        max_goals: Maximum number of goals considered for each team.

    Returns:
        Dictionary containing probability arrays for various single and dual outcomes.
    """
    # --- Input Assertions ---
    assert lambda_home.ndim == 1, "lambda_home must be a 1D array"
    assert lambda_away.ndim == 1, "lambda_away must be a 1D array"
    assert lambda_home.shape == lambda_away.shape, "Lambda arrays must have the same shape"
    assert np.all(lambda_home >= 0), "lambda_home contains negative values"
    assert np.all(lambda_away >= 0), "lambda_away contains negative values"

    n_matches = len(lambda_home)

    # --- Initialize Probability Accumulators ---
    # Singles
    prob_H = np.zeros(n_matches); prob_D = np.zeros(n_matches); prob_A = np.zeros(n_matches)
    prob_O05 = np.zeros(n_matches); prob_U05 = np.zeros(n_matches)
    prob_O15 = np.zeros(n_matches); prob_U15 = np.zeros(n_matches)
    prob_O25 = np.zeros(n_matches); prob_U25 = np.zeros(n_matches)
    prob_O35 = np.zeros(n_matches); prob_U35 = np.zeros(n_matches)
    prob_O45 = np.zeros(n_matches); prob_U45 = np.zeros(n_matches)
    prob_BTTS_Y = np.zeros(n_matches); prob_BTTS_N = np.zeros(n_matches)
    prob_goals_0_1 = np.zeros(n_matches)
    prob_goals_2_3 = np.zeros(n_matches)
    prob_goals_2_4 = np.zeros(n_matches)
    prob_goals_3_plus = np.zeros(n_matches) # == O2.5

    # Doubles (Initialize all potential combinations based on helper)
    # Get keys from one evaluation to initialize all dual accumulators
    _example_dual_keys = _calculate_dual_conditions(1, 1).keys()
    prob_duals = {f'prob_{key}': np.zeros(n_matches) for key in _example_dual_keys}

    # --- Pre-calculate PMF Matrix ---
    goal_range = np.arange(0, max_goals + 1)
    # Shape: (max_goals + 1, n_matches)
    home_goal_probs_pmf = poisson.pmf(goal_range[:, None], lambda_home)
    away_goal_probs_pmf = poisson.pmf(goal_range[:, None], lambda_away)

    # --- Sum Probabilities for Each Outcome ---
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            prob_scoreline = home_goal_probs_pmf[hg] * away_goal_probs_pmf[ag]
            total_goals = hg + ag

            # --- Singles ---
            if hg > ag: prob_H += prob_scoreline
            elif hg == ag: prob_D += prob_scoreline
            else: prob_A += prob_scoreline

            if total_goals > 0.5: prob_O05 += prob_scoreline
            if total_goals > 1.5: prob_O15 += prob_scoreline
            if total_goals > 2.5: prob_O25 += prob_scoreline
            if total_goals > 3.5: prob_O35 += prob_scoreline
            if total_goals > 4.5: prob_O45 += prob_scoreline

            if hg > 0 and ag > 0: prob_BTTS_Y += prob_scoreline

            if 0 <= total_goals <= 1: prob_goals_0_1 += prob_scoreline
            if 2 <= total_goals <= 3: prob_goals_2_3 += prob_scoreline
            if 2 <= total_goals <= 4: prob_goals_2_4 += prob_scoreline
            if total_goals >= 3:      prob_goals_3_plus += prob_scoreline

            # --- Doubles ---
            dual_conditions_met = _calculate_dual_conditions(hg, ag)
            for key, met in dual_conditions_met.items():
                if met:
                    prob_duals[f'prob_{key}'] += prob_scoreline


    # --- Finalize and Normalize/Clip ---
    # Normalize 1X2
    total_prob_1x2 = prob_H + prob_D + prob_A
    # Avoid division by zero or near-zero probabilities
    total_prob_1x2 = np.maximum(total_prob_1x2, 1e-12)
    prob_H /= total_prob_1x2
    prob_D /= total_prob_1x2
    prob_A /= total_prob_1x2
    prob_H = np.clip(prob_H, 0.0, 1.0)
    prob_D = np.clip(prob_D, 0.0, 1.0)
    prob_A = np.clip(prob_A, 0.0, 1.0)


    # Calculate Unders and BTTS No, clip for safety
    prob_O05 = np.clip(prob_O05, 0.0, 1.0); prob_U05 = 1.0 - prob_O05
    prob_O15 = np.clip(prob_O15, 0.0, 1.0); prob_U15 = 1.0 - prob_O15
    prob_O25 = np.clip(prob_O25, 0.0, 1.0); prob_U25 = 1.0 - prob_O25
    prob_O35 = np.clip(prob_O35, 0.0, 1.0); prob_U35 = 1.0 - prob_O35
    prob_O45 = np.clip(prob_O45, 0.0, 1.0); prob_U45 = 1.0 - prob_O45
    prob_BTTS_Y = np.clip(prob_BTTS_Y, 0.0, 1.0); prob_BTTS_N = 1.0 - prob_BTTS_Y

    # Clip goal band probabilities
    prob_goals_0_1 = np.clip(prob_goals_0_1, 0.0, 1.0)
    prob_goals_2_3 = np.clip(prob_goals_2_3, 0.0, 1.0)
    prob_goals_2_4 = np.clip(prob_goals_2_4, 0.0, 1.0)
    prob_goals_3_plus = np.clip(prob_goals_3_plus, 0.0, 1.0)

    # Clip dual probabilities
    for key in prob_duals:
        prob_duals[key] = np.clip(prob_duals[key], 0.0, 1.0)

    # Combine results
    results = {
        # 1X2
        'prob_H': prob_H, 'prob_D': prob_D, 'prob_A': prob_A,
        # Double Chance (derived from 1X2)
        'prob_1X': prob_H + prob_D, 'prob_12': prob_H + prob_A, 'prob_X2': prob_D + prob_A,
        # O/U Lines
        'prob_O05': prob_O05, 'prob_U05': prob_U05,
        'prob_O15': prob_O15, 'prob_U15': prob_U15,
        'prob_O25': prob_O25, 'prob_U25': prob_U25,
        'prob_O35': prob_O35, 'prob_U35': prob_U35,
        'prob_O45': prob_O45, 'prob_U45': prob_U45,
        # BTTS
        'prob_BTTS_Y': prob_BTTS_Y, 'prob_BTTS_N': prob_BTTS_N,
        # Goal Bands
        'prob_goals_0_1': prob_goals_0_1,
        'prob_goals_2_3': prob_goals_2_3,
        'prob_goals_2_4': prob_goals_2_4,
        'prob_goals_3_plus': prob_goals_3_plus,
    }
    # Add the calculated dual probabilities
    results.update(prob_duals)

    return results


# --- Poisson Model Class ---
class PoissonModel(BaseModel):
    """
    Poisson Regression model to predict expected goals (lambdas) based on features,
    and then derive outcome probabilities using the Poisson distribution.
    Uses scaled features provided by the BaseModel.
    """
    def __init__(self, model_params: Dict[str, Any], feature_config: BaseFeatureConfig):
        """Initializes the PoissonModel."""
        super().__init__(model_params)
        # Ensure feature_config is passed and stored
        assert isinstance(feature_config, BaseFeatureConfig), "feature_config must be provided and be a BaseFeatureConfig instance."
        self.feature_config = feature_config
        self._model_home = PoissonRegressor(**self.params)
        self._model_away = PoissonRegressor(**self.params)
        # Store models in a dictionary compatible with base save/load
        self._model = {'home': self._model_home, 'away': self._model_away}

    def _fit_model(self, X_scaled: pd.DataFrame, y: pd.DataFrame):
        """Fits two Poisson Regressors using SCALED features."""
        # --- Assertions specific to Poisson targets ---
        target_hg = self.feature_config.target_home_goals
        target_ag = self.feature_config.target_away_goals
        assert target_hg in y.columns, f"Target column '{target_hg}' not found in y."
        assert target_ag in y.columns, f"Target column '{target_ag}' not found in y."
        assert pd.api.types.is_numeric_dtype(y[target_hg]), f"Target '{target_hg}' is not numeric."
        assert pd.api.types.is_numeric_dtype(y[target_ag]), f"Target '{target_ag}' is not numeric."
        assert not y[[target_hg, target_ag]].isnull().any().any(), f"Target columns contain NaN values."
        assert np.all(y[target_hg] >= 0), f"Target '{target_hg}' contains negative values."
        assert np.all(y[target_ag] >= 0), f"Target '{target_ag}' contains negative values."

        assert X_scaled.columns.tolist() == self.features_in_, "Scaled features columns mismatch features_in_"

        print(f"Fitting Poisson Regressor for Home Goals ({target_hg}) using scaled features...")
        self._model['home'].fit(X_scaled, y[target_hg])

        print(f"Fitting Poisson Regressor for Away Goals ({target_ag}) using scaled features...")
        self._model['away'].fit(X_scaled, y[target_ag])

        print("Poisson models fitted successfully.")

    def _predict_proba_model(self, X_scaled: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predicts expected goals (lambdas) using SCALED features and calculates probabilities."""
        assert X_scaled.columns.tolist() == self.features_in_, "Scaled prediction features columns mismatch features_in_"

        print("Predicting expected goals (lambdas) using scaled features...")
        lambda_home = self._model['home'].predict(X_scaled)
        lambda_away = self._model['away'].predict(X_scaled)
        # Ensure non-negative and non-zero to avoid issues in poisson.pmf
        lambda_home = np.maximum(lambda_home, 1e-9)
        lambda_away = np.maximum(lambda_away, 1e-9)

        print("Calculating outcome probabilities (including duals) from lambdas via scoreline summation...")
        # Calls the UPDATED helper function which now includes duals
        outcome_probs = calculate_poisson_outcome_probs(lambda_home, lambda_away)

        # Add expected goals (lambdas) - potentially useful downstream
        outcome_probs['expected_HG'] = lambda_home
        outcome_probs['expected_AG'] = lambda_away

        # --- Updated Assertions on Output ---
        # Note: The exact set of keys depends on the _calculate_dual_conditions helper
        # We just check that expected singles and *some* duals are present.
        expected_single_keys = {
            'prob_H', 'prob_D', 'prob_A', 'prob_1X', 'prob_12', 'prob_X2',
            'prob_O05', 'prob_U05', 'prob_O15', 'prob_U15', 'prob_O25', 'prob_U25',
            'prob_O35', 'prob_U35', 'prob_O45', 'prob_U45',
            'prob_BTTS_Y', 'prob_BTTS_N',
            'prob_goals_0_1', 'prob_goals_2_3', 'prob_goals_2_4', 'prob_goals_3_plus',
            'expected_HG', 'expected_AG'
        }
        present_keys = set(outcome_probs.keys())
        assert expected_single_keys.issubset(present_keys), \
            f"Output keys missing expected singles.\nMissing: {expected_single_keys - present_keys}"
        # Check if at least one dual key is present (proof of concept)
        assert any(k.startswith('prob_') and '_and_' in k for k in present_keys), \
            "Output keys do not contain any dual probability keys (e.g., 'prob_H_and_O25')."


        num_rows = X_scaled.shape[0]
        for key, arr in outcome_probs.items():
            assert isinstance(arr, np.ndarray), f"Output '{key}' is not a numpy array."
            assert arr.shape == (num_rows,), f"Output '{key}' has incorrect shape {arr.shape}, expected ({num_rows},)."
            if key.startswith("prob_"):
                 assert np.all((arr >= 0) & (arr <= 1)), f"Probabilities in '{key}' are outside [0, 1]."
            elif key.startswith("expected_"):
                 assert np.all(arr >= 0), f"Expected goals in '{key}' are negative."

        # Check key probability sums remain valid
        assert np.allclose(outcome_probs['prob_H'] + outcome_probs['prob_D'] + outcome_probs['prob_A'], 1.0, atol=1e-6)
        assert np.allclose(outcome_probs['prob_1X'] + outcome_probs['prob_A'], 1.0, atol=1e-6) # Example derived check
        assert np.allclose(outcome_probs['prob_O25'] + outcome_probs['prob_U25'], 1.0, atol=1e-6)
        assert np.allclose(outcome_probs['prob_BTTS_Y'] + outcome_probs['prob_BTTS_N'], 1.0, atol=1e-6)
        assert np.allclose(outcome_probs['prob_goals_3_plus'], outcome_probs['prob_O25'], atol=1e-6), "P(Goals 3+) != P(Over 2.5)"
        # Example check for dual consistency: P(H and O2.5) + P(D and O2.5) + P(A and O2.5) == P(O2.5)
        assert np.allclose(outcome_probs['prob_H_and_O25'] + outcome_probs['prob_D_and_O25'] + outcome_probs['prob_A_and_O25'], outcome_probs['prob_O25'], atol=1e-6), "Sum P(Result and O2.5) != P(O2.5)"
        # Example check: P(O2.5 and BTTS_Y) + P(O2.5 and BTTS_N) == P(O2.5)
        assert np.allclose(outcome_probs['prob_O25_and_BTTS_Y'] + outcome_probs['prob_O25_and_BTTS_N'], outcome_probs['prob_O25'], atol=1e-6), "Sum P(O2.5 and BTTS) != P(O2.5)"


        print("Probabilities (including duals) calculated successfully via scoreline summation.")
        return outcome_probs

    # Override the base fit and predict_proba to ensure the correct signature is maintained
    # These just call the base methods which handle scaling and delegate to _fit_model/_predict_proba_model
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fits the model using the base class logic which includes scaling."""
        super().fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predicts probabilities using the base class logic which includes scaling."""
        return super().predict_proba(X)

    # save/load methods inherited from BaseModel should work correctly