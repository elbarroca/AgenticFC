# models/bayesian_model.py
import numpy as np
import pandas as pd
from .base_model import BaseModel
from .utils.features import calculate_implied_probabilities # Assuming you have this utility

class BayesianModel(BaseModel):
    """
    A Bayesian model for updating match outcome probabilities based on evidence.

    Starts with prior probabilities (e.g., derived from opening odds) and
    updates them using likelihoods associated with new evidence (e.g.,
    significant odds shifts, key player absences, red cards during the match).
    """

    def __init__(self, likelihood_config=None):
        """
        Initializes the Bayesian model.

        Args:
            likelihood_config (dict, optional): A configuration dictionary defining
                how different types of evidence affect the likelihoods for H/D/A outcomes.
                Example:
                {
                    'key_player_injured_home': {'H': 0.8, 'D': 1.0, 'A': 1.1}, # Likelihood factors P(E|H), P(E|D), P(E|A)
                    'significant_odds_drop_home': {'H': 1.2, 'D': 0.9, 'A': 0.9},
                    'red_card_home': {'H': 0.6, 'D': 1.1, 'A': 1.3}
                    # ... add more evidence types
                }
                Defaults to a basic configuration if None.
        """
        self.likelihood_config = likelihood_config if likelihood_config else self._get_default_likelihood_config()
        self.is_fitted = False # Track if any 'fitting' (e.g., learning likelihoods) has occurred

    def _get_default_likelihood_config(self):
        """Provides a basic, example likelihood configuration."""
        # --- IMPORTANT ---
        # These values are illustrative placeholders!
        # Real values should be derived from historical data analysis or expert knowledge.
        # P(Evidence | Hypothesis) - How likely is the evidence given the eventual outcome?
        # Example: If a home team gets a red card (Evidence), is that more likely if they
        # were going to lose anyway (Hypothesis=A) than if they were going to win (Hypothesis=H)?
        # We use factors here for simplicity in the proportional update.
        return {
            # Evidence: Key Home Player Injured (pre-match)
            'key_player_injured_home': {'H': 0.8, 'D': 1.0, 'A': 1.1}, # Less likely to win, more likely to lose
            # Evidence: Key Away Player Injured (pre-match)
            'key_player_injured_away': {'H': 1.1, 'D': 1.0, 'A': 0.8}, # More likely home wins, less likely away wins
            # Evidence: Home Team Odds Dropped Significantly (>15%?)
            'significant_odds_drop_home': {'H': 1.2, 'D': 0.95, 'A': 0.9}, # Market thinks home win is more likely
             # Evidence: Away Team Odds Dropped Significantly (>15%?)
            'significant_odds_drop_away': {'H': 0.9, 'D': 0.95, 'A': 1.2}, # Market thinks away win is more likely
            # Evidence: Home Team Red Card (in-play)
            'red_card_home': {'H': 0.6, 'D': 1.1, 'A': 1.3}, # Much less likely to win, draw/loss more likely
            # Evidence: Away Team Red Card (in-play)
            'red_card_away': {'H': 1.3, 'D': 1.1, 'A': 0.6}, # Much more likely home wins, away win less likely
             # Evidence: Home Team Scores First (in-play)
            'goal_scored_home': {'H': 1.5, 'D': 0.8, 'A': 0.5}, # Home win much more likely
             # Evidence: Away Team Scores First (in-play)
            'goal_scored_away': {'H': 0.5, 'D': 0.8, 'A': 1.5}, # Away win much more likely
        }

    def fit(self, data: pd.DataFrame, target_column: str = 'FTR'):
        """
        'Fits' the Bayesian model.

        For this type of model, fitting might involve:
        1. Learning the likelihood factors in `likelihood_config` from historical data.
           (e.g., How often did a home red card precede a home win, draw, or loss?).
           This is complex and requires careful analysis.
        2. Setting up default prior distributions based on overall league stats.
        3. For this example, we'll keep it simple and assume the likelihood_config
           is pre-defined or passed in. We'll just mark it as 'fitted'.

        Args:
            data (pd.DataFrame): Historical match data.
            target_column (str): The column indicating the final result (e.g., 'FTR').
        """
        print(f"Fitting Bayesian Model...")
        # --- Advanced Implementation ---
        # Here you could analyze 'data' to estimate the likelihood factors.
        # For example, calculate P(Home Red Card | FTR='H'), P(Home Red Card | FTR='D'), etc.
        # This requires identifying events (like red cards) in your historical data
        # and correlating them with outcomes. Your MongoDB data seems richer for this.
        # --- Simple Implementation ---
        print("Using pre-defined or default likelihood configuration.")
        # You could potentially calculate baseline priors here if needed.
        # e.g., overall H/D/A frequencies in the league.
        self.is_fitted = True
        print("Bayesian Model fitting complete (used configuration).")


    def update_prob_proportional(self, prior_probs: dict, likelihood_factors: dict) -> dict:
        """
        Updates prior probabilities using likelihood factors via proportional update.

        Posterior(H) ∝ Likelihood(E|H) * Prior(H)
        Then normalize across H, D, A.

        Args:
            prior_probs (dict): Current probabilities {'H': P(H), 'D': P(D), 'A': P(A)}.
            likelihood_factors (dict): Likelihood factors for the evidence
                                       {'H': L(E|H), 'D': L(E|D), 'A': L(E|A)}.
                                       These aren't strict probabilities but relative factors.

        Returns:
            dict: Updated (posterior) probabilities {'H': P'(H), 'D': P'(D), 'A': P'(A)}.
        """
        posterior_unnormalized = {}
        total_prob = 0

        for outcome in ['H', 'D', 'A']:
            prior = prior_probs.get(outcome, 0)
            # Use 1.0 as default factor if evidence doesn't apply to an outcome
            likelihood_factor = likelihood_factors.get(outcome, 1.0)
            unnormalized_prob = prior * likelihood_factor
            posterior_unnormalized[outcome] = unnormalized_prob
            total_prob += unnormalized_prob

        posterior_normalized = {}
        if total_prob > 0:
            for outcome, unnorm_prob in posterior_unnormalized.items():
                posterior_normalized[outcome] = unnorm_prob / total_prob
        else:
            # If total_prob is 0 (e.g., prior was 0 for all), return the prior
            # or handle as an error/edge case. Returning prior is safer.
            posterior_normalized = prior_probs

        return posterior_normalized

    def _get_likelihoods_for_evidence(self, evidence_type: str) -> dict:
        """
        Retrieves the likelihood factors for a given type of evidence from the config.

        Args:
            evidence_type (str): The key corresponding to the evidence in likelihood_config.

        Returns:
            dict: The likelihood factors {'H': L(E|H), 'D': L(E|D), 'A': L(E|A)},
                  or {'H': 1.0, 'D': 1.0, 'A': 1.0} if evidence type is unknown.
        """
        return self.likelihood_config.get(evidence_type, {'H': 1.0, 'D': 1.0, 'A': 1.0})

    def predict_with_evidence(self, initial_priors: dict, evidence_list: list) -> dict:
        """
        Sequentially updates initial priors based on a list of evidence.

        Args:
            initial_priors (dict): Starting probabilities {'H': P(H), 'D': P(D), 'A': P(A)}.
            evidence_list (list): A list of strings, where each string is a key
                                  in `self.likelihood_config` representing observed evidence.
                                  Example: ['key_player_injured_home', 'red_card_away']

        Returns:
            dict: The final posterior probabilities after considering all evidence.
        """
        if not isinstance(initial_priors, dict) or not all(k in initial_priors for k in ['H', 'D', 'A']):
             raise ValueError("initial_priors must be a dict with keys 'H', 'D', 'A'")
        if not isinstance(evidence_list, list):
             raise ValueError("evidence_list must be a list of evidence type strings")

        current_probs = initial_priors.copy()

        for evidence_type in evidence_list:
            likelihood_factors = self._get_likelihoods_for_evidence(evidence_type)
            current_probs = self.update_prob_proportional(current_probs, likelihood_factors)

        return current_probs

    def predict(self, data: dict) -> dict:
        """
        Predicts the updated H/D/A probabilities for a given match scenario.

        Args:
            data (dict): A dictionary representing a single match scenario. Must contain
                         at least initial odds and an evidence list.
                         Example:
                         {
                             'B365H': 2.0, 'B365D': 3.5, 'B365A': 4.0, # Or other odds source
                             'evidence': ['key_player_injured_away', 'significant_odds_drop_home']
                             # Add other relevant fields if needed for evidence detection
                             'HomeTeam': 'Team A', 'AwayTeam': 'Team B',
                             'HomeRedCards': 0, 'AwayRedCards': 1, # Example for in-play
                         }

        Returns:
            dict: The final posterior probabilities {'H': P'(H), 'D': P'(D), 'A': P'(A)}.
        """
        if not self.is_fitted:
            print("Warning: Model has not been explicitly fitted. Using default config.")
            # Or raise an error: raise RuntimeError("Model must be fitted before prediction.")

        # --- 1. Determine Initial Priors ---
        # Use provided odds. Need a utility to convert odds to probabilities.
        # Let's assume B365 odds are present. Handle cases where they might be missing.
        odds_keys = ['B365H', 'B365D', 'B365A'] # Or adapt to other bookies like 'PSH', 'PSD', 'PSA'
        if not all(k in data for k in odds_keys):
            # Fallback or error - using uniform priors if odds missing
            print(f"Warning: Odds {odds_keys} not found in data. Using uniform priors.")
            initial_priors = {'H': 1/3, 'D': 1/3, 'A': 1/3}
        else:
            try:
                # Assuming a utility function exists: models/utils/features.py
                # This function should handle potential zero or invalid odds and the overround.
                initial_priors = calculate_implied_probabilities(
                    data[odds_keys[0]], data[odds_keys[1]], data[odds_keys[2]]
                )
            except Exception as e:
                print(f"Error calculating implied probabilities: {e}. Using uniform priors.")
                initial_priors = {'H': 1/3, 'D': 1/3, 'A': 1/3}

        # --- 2. Identify Evidence ---
        # This part requires logic based on the input 'data' dictionary.
        # It might be passed directly, or you might need to infer it.
        evidence_list = data.get('evidence', []) # Get pre-defined list if available

        # --- Example: Infer evidence from other fields (if not passed directly) ---
        # This is where you'd check for injuries, red cards, significant odds shifts based on data fields.
        # Needs more context on how you represent this info in the 'data' dict for prediction.
        # Example:
        # if data.get('HomeKeyPlayerInjured', False): evidence_list.append('key_player_injured_home')
        # if data.get('AwayRedCards', 0) > 0: evidence_list.append('red_card_away')
        # Check for significant odds shifts compared to opening odds (if available)

        # --- 3. Update Probabilities ---
        final_probs = self.predict_with_evidence(initial_priors, evidence_list)

        return final_probs

# Example Usage (within bayesian_model.py)
if __name__ == '__main__':
    # Assume calculate_implied_probabilities is defined elsewhere or mocked here
    def calculate_implied_probabilities(h, d, a):
        if not all([h, d, a]) or any(o <= 0 for o in [h, d, a]): return {'H': 1/3, 'D': 1/3, 'A': 1/3}
        inv_h, inv_d, inv_a = 1/h, 1/d, 1/a
        margin = inv_h + inv_d + inv_a
        if margin == 0: return {'H': 1/3, 'D': 1/3, 'A': 1/3}
        return {'H': inv_h / margin, 'D': inv_d / margin, 'A': inv_a / margin}

    # --- Setup ---
    model = BayesianModel() # Uses default likelihood config
    model.fit(data=None) # Mark as fitted (in reality, might load historical data here)

    # --- Scenario 1: Pre-match, Away key player injured ---
    match_data_1 = {
        'B365H': 2.0, 'B365D': 3.4, 'B365A': 3.8,
        'evidence': ['key_player_injured_away']
    }
    print("\n--- Scenario 1 ---")
    initial_priors_1 = calculate_implied_probabilities(match_data_1['B365H'], match_data_1['B365D'], match_data_1['B365A'])
    print(f"Initial Priors: {initial_priors_1}")
    updated_probs_1 = model.predict(match_data_1)
    print(f"Evidence: {match_data_1['evidence']}")
    print(f"Updated Probs: {updated_probs_1}")

    # --- Scenario 2: In-play, Home team got red card after starting with even odds ---
    match_data_2 = {
        'B365H': 2.6, 'B365D': 3.3, 'B365A': 2.7, # Opening odds were roughly even
        'evidence': ['red_card_home'] # Event happened in-play
    }
    print("\n--- Scenario 2 ---")
    initial_priors_2 = calculate_implied_probabilities(match_data_2['B365H'], match_data_2['B365D'], match_data_2['B365A'])
    print(f"Initial Priors: {initial_priors_2}")
    updated_probs_2 = model.predict(match_data_2)
    print(f"Evidence: {match_data_2['evidence']}")
    print(f"Updated Probs: {updated_probs_2}")

    # --- Scenario 3: Pre-match, Home odds dropped significantly, Away player injured ---
    match_data_3 = {
        'B365H': 1.8, 'B365D': 3.6, 'B365A': 4.5,
        'evidence': ['significant_odds_drop_home', 'key_player_injured_away']
    }
    print("\n--- Scenario 3 ---")
    initial_priors_3 = calculate_implied_probabilities(match_data_3['B365H'], match_data_3['B365D'], match_data_3['B365A'])
    print(f"Initial Priors: {initial_priors_3}")
    updated_probs_3 = model.predict(match_data_3)
    print(f"Evidence: {match_data_3['evidence']}")
    print(f"Updated Probs: {updated_probs_3}")