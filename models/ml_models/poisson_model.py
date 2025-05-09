# models/poisson_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson
from typing import Dict, Any
from models.utils.features import BaseFeatureConfig
from models.base_model import BaseModel

def _calculate_dual_conditions(
    hg: int, ag: int,
    over_X5: bool, under_X5: bool, # Pass specific O/U condition
    line_X5_str: str # String like "O05", "U25" for key generation
) -> Dict[str, bool]:
    """
    Helper to evaluate conditions for various dual outcomes for a given scoreline
    against a specific Over/Under line.
    """
    # total_goals = hg + ag # Not directly needed here as over_X5/under_X5 are passed
    home_win = hg > ag
    draw = hg == ag
    away_win = hg < ag
    btts_yes = hg > 0 and ag > 0
    btts_no = not btts_yes

    # Use line_X5_str to make keys dynamic, e.g., H_and_O05, D_and_U15
    # The line_X5_str will be like "O05", "U15", "O25" etc.
    # We need to extract the O/U part for some keys
    # For example, if line_X5_str is "O05", we want "O05". If "U25", we want "U25".
    
    conditions = {
        # 1X2 & O/U X.5
        f'H_and_{line_X5_str}': home_win and over_X5,
        f'D_and_{line_X5_str}': draw and over_X5,
        f'A_and_{line_X5_str}': away_win and over_X5,
        # If line_X5_str was for "Over", the corresponding "Under" is also implicitly defined
        # but we'll generate keys based on the passed line_X5_str.
        # The calculate_poisson_outcome_probs will call this for both O and U lines.
        # So, if line_X5_str is "U25", then over_X5 would be false, and this is correct.

        # Double Chance & O/U X.5
        f'1X_and_{line_X5_str}': (home_win or draw) and over_X5,
        f'12_and_{line_X5_str}': (home_win or away_win) and over_X5,
        f'X2_and_{line_X5_str}': (draw or away_win) and over_X5,
        
        # O/U X.5 & BTTS (These are independent of H/D/A)
        # This part might be better handled in the main loop if we want all O/U with BTTS
        # For now, let's keep it focused on the passed line_X5_str
        f'{line_X5_str}_and_BTTS_Y': over_X5 and btts_yes,
        f'{line_X5_str}_and_BTTS_N': over_X5 and btts_no,
    }
    
    # 1X2 & BTTS (These are independent of O/U lines, so only generate once)
    # We'll add these separately in the main function to avoid redundancy if called multiple times
    # for different O/U lines.
    
    return conditions


def calculate_poisson_outcome_probs(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    max_goals: int = 8 # Number of goals to sum probabilities over
) -> Dict[str, np.ndarray]:
    """
    Calculates 1X2, multiple O/U lines, BTTS, specific goal band, AND
    accurate dual-outcome probabilities from Poisson lambdas via scoreline summation.
    Ensures all output probabilities are strictly within [0, 1].

    Args:
        lambda_home: Predicted expected goals for the home team (1D array).
        lambda_away: Predicted expected goals for the away team (1D array).
        max_goals: Maximum number of goals considered for each team.

    Returns:
        Dictionary containing probability arrays for various single and dual outcomes.
    """
    # --- Input Assertions ---
    if lambda_home.ndim > 1:
        lambda_home = lambda_home.flatten()
    if lambda_away.ndim > 1:
        lambda_away = lambda_away.flatten()
        
    assert lambda_home.ndim == 1, "lambda_home must be a 1D array"
    assert lambda_away.ndim == 1, "lambda_away must be a 1D array"
    assert lambda_home.shape == lambda_away.shape, "Lambda arrays must have the same shape"
    # Lambdas are now clipped in the trainable, but good to ensure non-negative here too for direct use.
    assert np.all(lambda_home >= 1e-9), "lambda_home contains non-positive values"
    assert np.all(lambda_away >= 1e-9), "lambda_away contains non-positive values"

    n_matches = len(lambda_home)

    # --- Initialize Probability Accumulators ---
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
    # This needs to be more dynamic now
    prob_duals = {} 
    # Initialize dual outcome keys that are standard (Result & BTTS, DC & BTTS)
    for res_key in ['H', 'D', 'A']:
        for btts_key in ['BTTS_Y', 'BTTS_N']:
            prob_duals[f'prob_{res_key}_and_{btts_key}'] = np.zeros(n_matches)
    for dc_key in ['1X', '12', 'X2']:
        for btts_key in ['BTTS_Y', 'BTTS_N']:
            prob_duals[f'prob_{dc_key}_and_{btts_key}'] = np.zeros(n_matches)

    # Initialize for specific O/U lines that will be combined
    # The `_calculate_dual_conditions` will be called for each O/U line string
    # O/U lines to consider for dual outcomes
    ou_lines_map = {
        "O05": 0.5, "U05": 0.5, 
        "O15": 1.5, "U15": 1.5,
        "O25": 2.5, "U25": 2.5,
        "O35": 3.5, "U35": 3.5,
        "O45": 4.5, "U45": 4.5,
    }
    # Initialize keys for duals involving these specific O/U lines
    for line_str, _ in ou_lines_map.items():
        # For Result & O/U Line
        for res_key in ['H', 'D', 'A']:
            prob_duals[f'prob_{res_key}_and_{line_str}'] = np.zeros(n_matches)
        # For DC & O/U Line
        for dc_key in ['1X', '12', 'X2']:
            prob_duals[f'prob_{dc_key}_and_{line_str}'] = np.zeros(n_matches)
        # For O/U Line & BTTS
        for btts_key in ['BTTS_Y', 'BTTS_N']:
             prob_duals[f'prob_{line_str}_and_{btts_key}'] = np.zeros(n_matches)

    goal_range = np.arange(0, max_goals + 1)
    home_goal_probs_pmf = poisson.pmf(goal_range[:, None], lambda_home)
    away_goal_probs_pmf = poisson.pmf(goal_range[:, None], lambda_away)

    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            prob_scoreline = home_goal_probs_pmf[hg] * away_goal_probs_pmf[ag]
            total_goals = hg + ag
            
            home_win = hg > ag
            draw = hg == ag
            away_win = hg < ag
            btts_yes = hg > 0 and ag > 0
            btts_no = not btts_yes

            # --- Singles ---
            if home_win: prob_H += prob_scoreline
            elif draw: prob_D += prob_scoreline
            else: prob_A += prob_scoreline

            if total_goals > 0.5: prob_O05 += prob_scoreline
            if total_goals > 1.5: prob_O15 += prob_scoreline
            if total_goals > 2.5: prob_O25 += prob_scoreline
            if total_goals > 3.5: prob_O35 += prob_scoreline
            if total_goals > 4.5: prob_O45 += prob_scoreline

            if btts_yes: prob_BTTS_Y += prob_scoreline

            if 0 <= total_goals <= 1: prob_goals_0_1 += prob_scoreline
            if 2 <= total_goals <= 3: prob_goals_2_3 += prob_scoreline
            if 2 <= total_goals <= 4: prob_goals_2_4 += prob_scoreline
            # prob_goals_3_plus will be assigned prob_O25 later

            # --- Doubles ---
            # Iterate through each O/U line for dual condition evaluation
            for line_str, line_val in ou_lines_map.items():
                is_over = total_goals > line_val
                is_under = total_goals <= line_val # Assuming O/U X.5 means > X.5 for Over

                current_over_X5 = False
                current_under_X5 = False

                if line_str.startswith("O"): # e.g. "O05"
                    current_over_X5 = is_over
                    current_under_X5 = not is_over # This is not 'is_under' from above, it's the opposite of current_over_X5
                elif line_str.startswith("U"): # e.g. "U05"
                    # For "Under" lines, the condition is met if total_goals IS NOT > line_val
                    # which is equivalent to total_goals <= line_val
                    current_over_X5 = is_under # The 'over_X5' arg in _calculate_dual_conditions means "condition for this line_str is met"
                    # current_under_X5 = not is_under # This is not used by _calculate_dual_conditions if we structure it well

                # Pass the boolean that corresponds to the line_str itself
                condition_for_line_str_met = current_over_X5

                # Call for Result & O/U Line and DC & O/U Line combinations
                # The _calculate_dual_conditions can be simplified if we only pass one O/U state
                dual_conditions_for_line = _calculate_dual_conditions(hg, ag, condition_for_line_str_met, not condition_for_line_str_met, line_str)
                for key, met in dual_conditions_for_line.items():
                    if met and f'prob_{key}' in prob_duals: # Ensure key exists
                        prob_duals[f'prob_{key}'] += prob_scoreline
            
            # Separately handle Result & BTTS and DC & BTTS (as they are not O/U dependent in their definition)
            if home_win and btts_yes: prob_duals['prob_H_and_BTTS_Y'] += prob_scoreline
            if home_win and btts_no:  prob_duals['prob_H_and_BTTS_N'] += prob_scoreline
            if draw and btts_yes:     prob_duals['prob_D_and_BTTS_Y'] += prob_scoreline
            if draw and btts_no:      prob_duals['prob_D_and_BTTS_N'] += prob_scoreline
            if away_win and btts_yes: prob_duals['prob_A_and_BTTS_Y'] += prob_scoreline
            if away_win and btts_no:  prob_duals['prob_A_and_BTTS_N'] += prob_scoreline

            if (home_win or draw) and btts_yes: prob_duals['prob_1X_and_BTTS_Y'] += prob_scoreline
            if (home_win or draw) and btts_no:  prob_duals['prob_1X_and_BTTS_N'] += prob_scoreline
            if (home_win or away_win) and btts_yes: prob_duals['prob_12_and_BTTS_Y'] += prob_scoreline
            if (home_win or away_win) and btts_no:  prob_duals['prob_12_and_BTTS_N'] += prob_scoreline
            if (draw or away_win) and btts_yes: prob_duals['prob_X2_and_BTTS_Y'] += prob_scoreline
            if (draw or away_win) and btts_no:  prob_duals['prob_X2_and_BTTS_N'] += prob_scoreline

    # --- Finalize and Normalize/Clip ---
    # Clip initial sums from scorelines before normalization
    prob_H = np.clip(prob_H, 0.0, 1.0)
    prob_D = np.clip(prob_D, 0.0, 1.0)
    prob_A = np.clip(prob_A, 0.0, 1.0)

    total_prob_1x2 = prob_H + prob_D + prob_A
    # Handle cases where sum might be zero (e.g., extreme lambdas leading to all zero PMFs)
    # or very far from 1.0 due to max_goals limit
    # If total_prob_1x2 is 0, probs remain 0. Otherwise, normalize.
    non_zero_sum_mask = total_prob_1x2 > 1e-12 
    
    if np.any(non_zero_sum_mask): # Proceed with normalization only if there's something to normalize
        prob_H[non_zero_sum_mask] = prob_H[non_zero_sum_mask] / total_prob_1x2[non_zero_sum_mask]
        prob_D[non_zero_sum_mask] = prob_D[non_zero_sum_mask] / total_prob_1x2[non_zero_sum_mask]
        prob_A[non_zero_sum_mask] = prob_A[non_zero_sum_mask] / total_prob_1x2[non_zero_sum_mask]

    # Final clip after normalization to ensure strict [0,1] bounds
    prob_H = np.clip(prob_H, 0.0, 1.0)
    prob_D = np.clip(prob_D, 0.0, 1.0)
    prob_A = np.clip(prob_A, 0.0, 1.0)

    # Ensure H+D+A sum to 1.0 by distributing any tiny error (mostly for the case where sum was slightly > 1)
    # This step is more robust after initial individual clips
    final_sum_1x2 = prob_H + prob_D + prob_A
    adjustment_mask = (np.abs(final_sum_1x2 - 1.0) > 1e-9) & (final_sum_1x2 > 1e-12)
    if np.any(adjustment_mask):
        prob_H[adjustment_mask] /= final_sum_1x2[adjustment_mask]
        prob_D[adjustment_mask] /= final_sum_1x2[adjustment_mask]
        prob_A[adjustment_mask] /= final_sum_1x2[adjustment_mask]
        # Re-clip after this final adjustment
        prob_H = np.clip(prob_H, 0.0, 1.0)
        prob_D = np.clip(prob_D, 0.0, 1.0)
        prob_A = np.clip(prob_A, 0.0, 1.0)


    # Calculate and clip other single outcomes
    prob_O05 = np.clip(prob_O05, 0.0, 1.0); prob_U05 = np.clip(1.0 - prob_O05, 0.0, 1.0)
    prob_O15 = np.clip(prob_O15, 0.0, 1.0); prob_U15 = np.clip(1.0 - prob_O15, 0.0, 1.0)
    prob_O25 = np.clip(prob_O25, 0.0, 1.0); prob_U25 = np.clip(1.0 - prob_O25, 0.0, 1.0)
    prob_O35 = np.clip(prob_O35, 0.0, 1.0); prob_U35 = np.clip(1.0 - prob_O35, 0.0, 1.0)
    prob_O45 = np.clip(prob_O45, 0.0, 1.0); prob_U45 = np.clip(1.0 - prob_O45, 0.0, 1.0)
    prob_BTTS_Y = np.clip(prob_BTTS_Y, 0.0, 1.0); prob_BTTS_N = np.clip(1.0 - prob_BTTS_Y, 0.0, 1.0)

    prob_goals_0_1 = np.clip(prob_goals_0_1, 0.0, 1.0)
    prob_goals_2_3 = np.clip(prob_goals_2_3, 0.0, 1.0)
    prob_goals_2_4 = np.clip(prob_goals_2_4, 0.0, 1.0)
    prob_goals_3_plus = prob_O25 # Already clipped

    # Clip dual probabilities (from scoreline summation)
    for key_dual in prob_duals:
        prob_duals[key_dual] = np.clip(prob_duals[key_dual], 0.0, 1.0)

    results = {
        'prob_H': prob_H, 'prob_D': prob_D, 'prob_A': prob_A,
        'prob_1X': np.clip(prob_H + prob_D, 0.0, 1.0),
        'prob_12': np.clip(prob_H + prob_A, 0.0, 1.0),
        'prob_X2': np.clip(prob_D + prob_A, 0.0, 1.0),
        'prob_O05': prob_O05, 'prob_U05': prob_U05,
        'prob_O15': prob_O15, 'prob_U15': prob_U15,
        'prob_O25': prob_O25, 'prob_U25': prob_U25,
        'prob_O35': prob_O35, 'prob_U35': prob_U35,
        'prob_O45': prob_O45, 'prob_U45': prob_U45,
        'prob_BTTS_Y': prob_BTTS_Y, 'prob_BTTS_N': prob_BTTS_N,
        'prob_goals_0_1': prob_goals_0_1,
        'prob_goals_2_3': prob_goals_2_3,
        'prob_goals_2_4': prob_goals_2_4,
        'prob_goals_3_plus': prob_goals_3_plus,
    }
    results.update(prob_duals)
    return results


# --- Poisson Model Class ---
class PoissonModel(BaseModel):
    """
    Poisson Regression model to predict expected goals (lambdas) based on features,
    and then derive outcome probabilities using the Poisson distribution.
    Uses scaled features provided by the BaseModel.
    """
    def __init__(self, model_params: Dict[str, Any], feature_config: BaseFeatureConfig, apply_scaling: bool = True):
        """Initializes the PoissonModel."""
        # Pass model_params, feature_config, and apply_scaling to the BaseModel constructor
        super().__init__(model_params, feature_config=feature_config, apply_scaling=apply_scaling)
        
        # self.feature_config = feature_config # This is now handled by BaseModel.__init__
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

        # --- Relax strict probability assertions ---
        # Check for extreme cases where probabilities don't sum exactly to 1.0
        h_d_a_sum = outcome_probs['prob_H'] + outcome_probs['prob_D'] + outcome_probs['prob_A']
        if not np.allclose(h_d_a_sum, 1.0, atol=1e-6):
            # Renormalize if sum is not close to 1.0
            print(f"Warning: 1X2 probs don't sum to 1.0 exactly. Mean sum: {np.mean(h_d_a_sum)}. Renormalizing...")
            sum_expanded = np.maximum(h_d_a_sum, 1e-12)  # Avoid division by zero
            outcome_probs['prob_H'] /= sum_expanded
            outcome_probs['prob_D'] /= sum_expanded
            outcome_probs['prob_A'] /= sum_expanded
        
        # --- Prefix all keys with model name ---
        prefixed_probs = {f"poisson_{key}": value for key, value in outcome_probs.items()}
        
        num_rows = X_scaled.shape[0]
        for key, arr in prefixed_probs.items():
            assert isinstance(arr, np.ndarray), f"Output '{key}' is not a numpy array."
            assert arr.shape == (num_rows,), f"Output '{key}' has incorrect shape {arr.shape}, expected ({num_rows},)."
            if "prob_" in key:
                assert np.all((arr >= 0) & (arr <= 1)), f"Probabilities in '{key}' are outside [0, 1]."
            elif "expected_" in key:
                assert np.all(arr >= 0), f"Expected goals in '{key}' are negative."

        print("Probabilities (including duals) calculated successfully via scoreline summation.")
        return prefixed_probs

    # Override the base fit and predict_proba to ensure the correct signature is maintained
    # These just call the base methods which handle scaling and delegate to _fit_model/_predict_proba_model
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fits the model using the base class logic which includes scaling."""
        super().fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predicts probabilities using the base class logic which includes scaling."""
        return super().predict_proba(X)