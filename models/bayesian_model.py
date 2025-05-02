# models/bayesian_model.py

import warnings
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import joblib
from typing import List, Optional, Dict, Any, Union, Tuple

from base_model import BaseModel

# Assuming you have these helpers or define them:
# from .base_model import BaseModel


def calculate_implied_probabilities(h: float, d: float, a: float, smooth: float = 1e-9) -> Dict[str, float]:
    """Converts odds to normalized implied probabilities."""
    if not all([h, d, a]) or any(o <= 0 for o in [h, d, a]):
        warnings.warn("Invalid odds input. Returning uniform priors.")
        return {'H': 1/3, 'D': 1/3, 'A': 1/3}
    inv_h, inv_d, inv_a = 1/h, 1/d, 1/a
    margin = inv_h + inv_d + inv_a
    if margin <= smooth: # Avoid division by zero or near-zero
        warnings.warn("Odds imply zero or near-zero margin. Returning uniform priors.")
        return {'H': 1/3, 'D': 1/3, 'A': 1/3}
    # Normalize to remove margin
    prob_h = inv_h / margin
    prob_d = inv_d / margin
    prob_a = inv_a / margin
    # Ensure sum is close to 1 after floating point ops
    norm_factor = 1.0 / (prob_h + prob_d + prob_a)
    return {'H': prob_h * norm_factor, 'D': prob_d * norm_factor, 'A': prob_a * norm_factor}
# --- End Mock ---


class BayesianUpdateModel(BaseModel):
    """
    A more structured Bayesian model for updating match outcome probabilities.

    Learns baseline priors and likelihood probabilities P(Evidence|Outcome) from
    historical data during the 'fit' step. Updates prior probabilities (e.g.,
    from current market odds) using new evidence via Bayesian proportional updates.

    Assumptions:
    - Evidence events are conditionally independent given the final outcome (simplification).
    - Likelihoods can be reasonably estimated from historical frequencies.
    """

    def __init__(self, default_likelihood_value: float = 1e-6, laplace_smoothing: float = 1.0):
        """
        Initializes the Bayesian model.

        Args:
            default_likelihood_value (float): Small probability assigned if an evidence/outcome
                                             combination was never observed during fitting.
                                             Helps avoid zero probabilities.
            laplace_smoothing (float): Add-k smoothing applied when calculating likelihoods
                                      from counts to handle unseen events. 1.0 is Laplace smoothing.
                                      Set to 0 to disable smoothing.
        """
        if not isinstance(default_likelihood_value, (float, int)) or not (0 < default_likelihood_value < 1):
            raise ValueError("default_likelihood_value must be a small positive float less than 1.")
        if not isinstance(laplace_smoothing, (float, int)) or laplace_smoothing < 0:
            raise ValueError("laplace_smoothing must be a non-negative number.")

        self.default_likelihood = default_likelihood_value
        self.smoothing = laplace_smoothing
        self.likelihoods: Dict[str, Dict[str, float]] = {} # Stores learned P(E|H), P(E|D), P(E|A) for each evidence type
        self.baseline_priors: Dict[str, float] = {'H': 1/3, 'D': 1/3, 'A': 1/3} # Overall H/D/A frequencies
        self.evidence_columns: Dict[str, str] = {} # Map evidence type key to DataFrame column name
        self.is_fitted: bool = False

    def _learn_likelihoods(self, history_df: pd.DataFrame, evidence_config: Dict[str, str], target_col: str = 'FTR'):
        """
        Learns P(Evidence|Outcome) from historical data.

        Args:
            history_df (pd.DataFrame): Historical data with outcomes and evidence flags.
            evidence_config (Dict[str, str]): Maps evidence keys (like 'red_card_home')
                                             to column names in history_df that indicate
                                             if the evidence occurred (e.g., 1 if yes, 0 if no).
            target_col (str): Column name for the outcome ('H', 'D', 'A').
        """
        print("Learning likelihoods P(Evidence|Outcome)...")
        learned_likelihoods = {}
        valid_outcomes = ['H', 'D', 'A']

        # Ensure target column contains valid outcomes
        history_df = history_df[history_df[target_col].isin(valid_outcomes)].copy()
        if history_df.empty:
            raise ValueError(f"No valid outcomes ('H', 'D', 'A') found in target column '{target_col}'.")

        # Calculate overall counts for each outcome (denominator for likelihood)
        outcome_counts = history_df[target_col].value_counts()
        total_H = outcome_counts.get('H', 0)
        total_D = outcome_counts.get('D', 0)
        total_A = outcome_counts.get('A', 0)

        # Check for zero counts which prevent likelihood calculation
        if total_H == 0 or total_D == 0 or total_A == 0:
            warnings.warn(f"One or more outcomes (H/D/A) have zero occurrences in the historical data. "
                          f"Likelihoods may be unreliable or use defaults. Counts: H={total_H}, D={total_D}, A={total_A}")

        for evidence_key, evidence_col in evidence_config.items():
            if evidence_col not in history_df.columns:
                warnings.warn(f"Evidence column '{evidence_col}' for key '{evidence_key}' not found in history_df. Skipping.")
                continue

            # Ensure evidence column is suitable (e.g., 0 or 1)
            if not pd.api.types.is_numeric_dtype(history_df[evidence_col]) or not history_df[evidence_col].isin([0, 1]).all():
                 warnings.warn(f"Evidence column '{evidence_col}' is not binary (0/1). Skipping likelihood calculation for '{evidence_key}'.")
                 continue

            likelihoods_for_evidence = {}
            # Calculate P(Evidence=1 | Outcome)
            for outcome in valid_outcomes:
                total_outcome = outcome_counts.get(outcome, 0)
                # Count how many times evidence occurred GIVEN the outcome
                evidence_given_outcome_count = history_df[
                    (history_df[target_col] == outcome) & (history_df[evidence_col] == 1)
                ].shape[0]

                # Apply Laplace smoothing
                # P(E|Outcome) = (Count(E and Outcome) + k) / (Count(Outcome) + k * NumberOfPossibleEvidenceValues)
                # Since evidence is binary (0/1), NumberOfPossibleEvidenceValues = 2
                numerator = evidence_given_outcome_count + self.smoothing
                denominator = total_outcome + self.smoothing * 2 # Smoothing for both E=1 and E=0 cases

                if denominator > 0:
                    likelihood = numerator / denominator
                    likelihoods_for_evidence[outcome] = max(self.default_likelihood, likelihood) # Ensure non-zero
                else:
                    # If Count(Outcome) is 0, can't reliably estimate
                    likelihoods_for_evidence[outcome] = self.default_likelihood
                    if total_outcome == 0: # Only warn if the outcome count was actually zero
                         print(f"  Warning: Zero instances of outcome '{outcome}'. Using default likelihood for evidence '{evidence_key}'.")


            learned_likelihoods[evidence_key] = likelihoods_for_evidence
            print(f"  Learned P({evidence_key}=1|Outcome): H={likelihoods_for_evidence.get('H', 0):.3f}, "
                  f"D={likelihoods_for_evidence.get('D', 0):.3f}, A={likelihoods_for_evidence.get('A', 0):.3f}")

        return learned_likelihoods

    def fit(self, history_df: pd.DataFrame, evidence_config: Dict[str, str], target_col: str = 'FTR'):
        """
        Fits the Bayesian model by calculating baseline priors and learning likelihoods.

        Args:
            history_df (pd.DataFrame): Historical data. Must contain the target column
                                       and all columns specified in `evidence_config`.
                                       Should ideally contain a large number of matches.
            evidence_config (Dict[str, str]): Maps evidence keys (e.g., 'red_card_home')
                                             to column names in history_df containing binary flags (0/1)
                                             indicating if the evidence occurred in that match.
            target_col (str): The column indicating the final result ('H', 'D', 'A').
        """
        print(f"Fitting BayesianUpdateModel...")
        required_cols = [target_col] + list(evidence_config.values())
        if not all(col in history_df.columns for col in required_cols):
            missing = set(required_cols) - set(history_df.columns)
            raise ValueError(f"history_df is missing required columns: {missing}")

        # --- 1. Calculate Baseline Priors (Overall H/D/A Frequencies) ---
        prior_counts = history_df[target_col].value_counts(normalize=True)
        self.baseline_priors['H'] = prior_counts.get('H', 0.0)
        self.baseline_priors['D'] = prior_counts.get('D', 0.0)
        self.baseline_priors['A'] = prior_counts.get('A', 0.0)
        # Ensure they sum to 1, handle potential floating point issues or missing outcomes
        total_prior = sum(self.baseline_priors.values())
        if total_prior > 0:
            self.baseline_priors = {k: v / total_prior for k, v in self.baseline_priors.items()}
        else:
            warnings.warn("Could not calculate valid baseline priors from data. Using uniform.")
            self.baseline_priors = {'H': 1/3, 'D': 1/3, 'A': 1/3}
        print(f"  Baseline Priors: H={self.baseline_priors['H']:.3f}, D={self.baseline_priors['D']:.3f}, A={self.baseline_priors['A']:.3f}")

        # --- 2. Learn Likelihoods P(Evidence|Outcome) ---
        self.evidence_columns = evidence_config # Store the mapping used for fitting
        self.likelihoods = self._learn_likelihoods(history_df, evidence_config, target_col)

        self.is_fitted = True
        print("BayesianUpdateModel fitting complete.")


    def update_prob_bayes_rule(self, prior_probs: Dict[str, float], evidence_key: str) -> Dict[str, float]:
        """
        Updates probabilities using Bayes' Theorem P(H|E) = P(E|H)P(H) / P(E).

        Args:
            prior_probs (dict): Current probabilities P(H), P(D), P(A).
            evidence_key (str): The key for the observed evidence (must be in self.likelihoods).

        Returns:
            dict: Updated (posterior) probabilities P(H|E), P(D|E), P(A|E).
        """
        if evidence_key not in self.likelihoods:
            warnings.warn(f"Evidence type '{evidence_key}' not found in learned likelihoods. Returning prior probabilities.")
            return prior_probs

        likelihoods_p_e_given_h = self.likelihoods[evidence_key] # Contains P(E=1|H), P(E=1|D), P(E=1|A)

        posterior_unnormalized = {}
        marginal_likelihood_p_e = 0 # P(E) = Sum[ P(E|Hi) * P(Hi) ] for Hi in {H, D, A}

        for outcome in ['H', 'D', 'A']:
            prior = prior_probs.get(outcome, 0)
            # Get P(E=1|Outcome) from stored likelihoods
            likelihood = likelihoods_p_e_given_h.get(outcome, self.default_likelihood)

            # Calculate P(E|Outcome) * P(Outcome)
            joint_prob = likelihood * prior
            posterior_unnormalized[outcome] = joint_prob
            marginal_likelihood_p_e += joint_prob

        # Normalize to get P(Outcome | E)
        posterior_normalized = {}
        if marginal_likelihood_p_e > 0:
            for outcome, unnorm_prob in posterior_unnormalized.items():
                posterior_normalized[outcome] = unnorm_prob / marginal_likelihood_p_e
        else:
            # Should not happen if priors are > 0 and likelihoods > default_likelihood > 0
            warnings.warn("Marginal likelihood P(E) calculated as zero. Returning prior probabilities.")
            posterior_normalized = prior_probs

        return posterior_normalized


    def predict(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Predicts the updated H/D/A probabilities for a given match scenario.

        It first establishes initial priors (e.g., from odds).
        Then, it identifies relevant evidence present in the 'data' dictionary
        based on the keys learned during 'fit' (self.evidence_columns).
        Finally, it sequentially updates the priors for each piece of evidence found.

        Args:
            data (Dict[str, Any]): A dictionary representing a single match scenario.
                                   Must contain keys for calculating initial priors
                                   (e.g., 'OddH', 'OddD', 'OddA') AND keys corresponding
                                   to the evidence columns used during fitting, indicating
                                   if that evidence is present (e.g., 'red_card_home': 1).
                         Example:
                         {
                             'OddH': 2.0, 'OddD': 3.5, 'OddA': 4.0, # Priors source
                             'red_card_home': 0,                  # Evidence flag from fit config
                             'key_player_injured_away': 1,        # Evidence flag from fit config
                             'significant_odds_drop_home': 0,     # Evidence flag from fit config
                             # ... other potential evidence flags ...
                         }

        Returns:
            Dict[str, float]: The final posterior probabilities {'H': P'(H), 'D': P'(D), 'A': P'(A)}.
        """
        check_is_fitted(self, ['is_fitted', 'likelihoods', 'baseline_priors', 'evidence_columns'])

        # --- 1. Determine Initial Priors ---
        # Prioritize odds if available, otherwise use baseline learned priors
        odds_keys = ['OddH', 'OddD', 'OddA'] # Example keys, adjust as needed
        if all(k in data for k in odds_keys):
            try:
                current_probs = calculate_implied_probabilities(
                    data[odds_keys[0]], data[odds_keys[1]], data[odds_keys[2]]
                )
                # Optional: Check if odds imply probabilities very different from baseline?
            except Exception as e:
                warnings.warn(f"Error calculating implied probabilities from odds: {e}. Using baseline priors.")
                current_probs = self.baseline_priors.copy()
        else:
            # print("Using baseline priors learned during fit.") # Optional info
            current_probs = self.baseline_priors.copy()

        # --- 2. Identify and Apply Evidence ---
        observed_evidence_keys = []
        for evidence_key, evidence_col in self.evidence_columns.items():
            if evidence_col in data and data[evidence_col] == 1: # Check if flag is set
                observed_evidence_keys.append(evidence_key)
            elif evidence_col not in data:
                warnings.warn(f"Evidence column '{evidence_col}' expected but not found in prediction data for key '{evidence_key}'.")

        if not observed_evidence_keys:
            # print("No specific evidence found in input data. Returning initial priors.") # Optional info
            return current_probs
        else:
             print(f"Applying evidence sequentially: {observed_evidence_keys}")

        # --- 3. Update Probabilities Sequentially ---
        for evidence_key in observed_evidence_keys:
            current_probs = self.update_prob_bayes_rule(current_probs, evidence_key)
            # Optional: Log intermediate probabilities after each update
            # print(f"  Probs after '{evidence_key}': H={current_probs['H']:.3f}, D={current_probs['D']:.3f}, A={current_probs['A']:.3f}")


        return current_probs

    def save(self, filepath: str):
        """Saves the fitted model state using joblib."""
        check_is_fitted(self, ['is_fitted', 'likelihoods', 'baseline_priors', 'evidence_columns'])
        print(f"Saving BayesianUpdateModel state to {filepath}...")
        state = {
            'likelihoods': self.likelihoods,
            'baseline_priors': self.baseline_priors,
            'evidence_columns': self.evidence_columns,
            'default_likelihood': self.default_likelihood,
            'smoothing': self.smoothing,
            '__class__': self.__class__.__name__ # Store class name for robustness
        }
        try:
            joblib.dump(state, filepath)
            print("Model state saved successfully.")
        except Exception as e:
            print(f"Error saving model state: {e}")
            raise

    @classmethod
    def load(cls, filepath: str):
        """Loads a fitted model state from a file."""
        print(f"Loading BayesianUpdateModel state from {filepath}...")
        try:
            state = joblib.load(filepath)
        except Exception as e:
            print(f"Error loading model state from {filepath}: {e}")
            raise

        # Basic validation
        required_keys = ['likelihoods', 'baseline_priors', 'evidence_columns', 'default_likelihood', 'smoothing']
        if not all(key in state for key in required_keys):
            missing = set(required_keys) - set(state.keys())
            raise ValueError(f"Loaded state is missing required keys: {missing}")
        if state.get('__class__') != cls.__name__:
            warnings.warn(f"Loading state saved from a different class ('{state.get('__class__')}') into '{cls.__name__}'.")

        # Create instance with saved config
        instance = cls(
            default_likelihood_value=state['default_likelihood'],
            laplace_smoothing=state['smoothing']
        )

        # Load the fitted attributes
        instance.likelihoods = state['likelihoods']
        instance.baseline_priors = state['baseline_priors']
        instance.evidence_columns = state['evidence_columns']
        instance.is_fitted = True # Mark as fitted

        print("Model state loaded successfully.")
        print(f"  Baseline Priors: H={instance.baseline_priors['H']:.3f}, D={instance.baseline_priors['D']:.3f}, A={instance.baseline_priors['A']:.3f}")
        print(f"  No. Evidence Types Loaded: {len(instance.likelihoods)}")
        return instance

# Example Usage
if __name__ == '__main__':
    # --- 1. Create Dummy Historical Data ---
    # Need columns for target and all evidence types defined in evidence_config
    data_size = 2000
    history_data = {
        'FTR': np.random.choice(['H', 'D', 'A'], size=data_size, p=[0.45, 0.25, 0.30]),
        # Simulate evidence occurrences (uncorrelated for simplicity here)
        # In reality, these should correlate with the outcome based on P(E|H) etc.
        'red_card_home': np.random.choice([0, 1], size=data_size, p=[0.97, 0.03]),
        'red_card_away': np.random.choice([0, 1], size=data_size, p=[0.96, 0.04]),
        'key_player_injured_home': np.random.choice([0, 1], size=data_size, p=[0.9, 0.1]),
        'key_player_injured_away': np.random.choice([0, 1], size=data_size, p=[0.88, 0.12]),
        'goal_scored_first_home': np.random.choice([0, 1], size=data_size, p=[0.55, 0.45]),
        # Add more evidence columns as needed...
    }
    history_df = pd.DataFrame(history_data)
    print("Dummy Historical Data Sample:")
    print(history_df.head())
    print("\nEvidence Frequencies:")
    print(history_df[[col for col in history_df.columns if col != 'FTR']].mean())

    # --- 2. Define Evidence Config ---
    # Map descriptive keys to column names in history_df
    evidence_config = {
        'home_red': 'red_card_home',
        'away_red': 'red_card_away',
        'home_key_injury': 'key_player_injured_home',
        'away_key_injury': 'key_player_injured_away',
        'home_scores_first': 'goal_scored_first_home' # Example, assumes goal_scored_first_home means E=1
    }

    # --- 3. Fit the Model ---
    bayes_model = BayesianUpdateModel(laplace_smoothing=1.0)
    bayes_model.fit(history_df, evidence_config=evidence_config, target_col='FTR')
    print("\nLearned Likelihoods (Sample):")
    print({k: v for k, v in list(bayes_model.likelihoods.items())[:2]}) # Show first couple

    # --- 4. Predict Scenarios ---
    print("\n--- Predicting Scenarios ---")

    # Scenario A: Use baseline priors, no evidence
    scenario_a_data = {} # No odds, no evidence flags
    probs_a = bayes_model.predict(scenario_a_data)
    print(f"\nScenario A (Baseline Priors, No Evidence): {probs_a}")

    # Scenario B: Use market odds, away key player injured
    scenario_b_data = {
        'OddH': 2.1, 'OddD': 3.3, 'OddA': 3.9, # Market odds
        'key_player_injured_away': 1 # This matches a column name from fit's evidence_config
        # 'home_red': 0, 'away_red': 0, 'home_key_injury': 0, 'home_scores_first': 0 # Explicitly set others to 0
    }
    # Ensure all expected evidence columns are present, defaulting to 0 if not provided
    for ev_col in bayes_model.evidence_columns.values():
        if ev_col not in scenario_b_data: scenario_b_data[ev_col] = 0
    probs_b = bayes_model.predict(scenario_b_data)
    print(f"\nScenario B (Odds Prior, Away Injury): {probs_b}")

    # Scenario C: Use market odds, Home red card AND Away key injury
    scenario_c_data = {
        'OddH': 2.5, 'OddD': 3.1, 'OddA': 3.0, # Market odds
        'red_card_home': 1,
        'key_player_injured_away': 1,
         # 'away_red': 0, 'home_key_injury': 0, 'home_scores_first': 0 # Explicitly set others to 0
    }
    # Ensure all expected evidence columns are present
    for ev_col in bayes_model.evidence_columns.values():
        if ev_col not in scenario_c_data: scenario_c_data[ev_col] = 0
    probs_c = bayes_model.predict(scenario_c_data)
    print(f"\nScenario C (Odds Prior, Home Red + Away Injury): {probs_c}")


    # --- 5. Save and Load ---
    model_path = "temp_bayes_model_proper.joblib"
    bayes_model.save(model_path)
    loaded_model = BayesianUpdateModel.load(model_path)

    # --- 6. Verify Loaded Model Prediction ---
    loaded_probs_c = loaded_model.predict(scenario_c_data)
    print("\nVerifying loaded model prediction (Scenario C)...")
    print(f"Original Prediction: {probs_c}")
    print(f"Loaded Prediction:   {loaded_probs_c}")
    assert all(abs(probs_c[k] - loaded_probs_c[k]) < 1e-6 for k in ['H', 'D', 'A'])
    assert loaded_model.evidence_columns == bayes_model.evidence_columns
    print("Save/Load test passed.")

    # Clean up
    import os
    if os.path.exists(model_path): os.remove(model_path)