# models/monte_carlo_model.py

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import time
from typing import Dict, Tuple, List, Optional

# This model simulates outcomes based on inputs; it doesn't fit data directly
# in the way ML models do, so inheriting from BaseModel isn't strictly necessary.

class MonteCarloModel:
    """
    Monte Carlo simulation model for football match outcomes using Poisson goal distributions.

    Simulates a match numerous times based on externally provided expected goal
    rates (lambdas) for each team. Estimates probabilities for various betting
    markets like 1X2, Over/Under, BTTS, and Correct Score.

    Accuracy is highly dependent on the quality of the input lambda values,
    which should be estimated using other models (e.g., PoissonModel, regression
    models trained on historical data, xG models).
    """

    def __init__(self, n_simulations: int = 10000, random_state: Optional[int] = None):
        """
        Initializes the MonteCarloModel simulator.

        Args:
            n_simulations (int): The number of simulation runs per prediction.
                                 Higher values yield more stable estimates but increase runtime.
            random_state (Optional[int]): Seed for the random number generator
                                          to ensure reproducibility. Defaults to None.
        """
        if not isinstance(n_simulations, int) or n_simulations <= 0:
            raise ValueError("n_simulations must be a positive integer.")
        self.n_simulations = n_simulations
        self.random_state = random_state
        # Initialize the random number generator
        self._rng = np.random.default_rng(seed=self.random_state)
        print(f"MonteCarloModel initialized with n_simulations={self.n_simulations}, random_state={self.random_state}")

    def _validate_lambdas(self, lambda_home: float, lambda_away: float):
        """Checks if lambda values are valid."""
        if not isinstance(lambda_home, (int, float)) or not isinstance(lambda_away, (int, float)):
             raise TypeError("lambda_home and lambda_away must be numeric.")
        if lambda_home < 0 or lambda_away < 0:
            raise ValueError("Lambda values cannot be negative.")

    def simulate_match(self, lambda_home: float, lambda_away: float) -> Dict[str, any]:
        """
        Performs the Monte Carlo simulation for a single match scenario.

        Args:
            lambda_home (float): Estimated expected goals for the home team.
            lambda_away (float): Estimated expected goals for the away team.

        Returns:
            Dict[str, any]: A dictionary containing simulation results:
                - 'lambda_inputs': {'home': lambda_home, 'away': lambda_away}
                - 'n_simulations': The number of simulations run.
                - 'outcome_probs': {'H': P(H), 'D': P(D), 'A': P(A)}
                - 'btts_prob': Probability of Both Teams To Score.
                - 'over_under_probs': Probabilities for standard O/U lines (0.5 to 5.5).
                - 'score_counts': Counter object with counts of each simulated scoreline (h, a).
                - 'score_probs': Dictionary mapping scoreline tuples (h, a) to probabilities.
                - 'most_likely_score': {'score': (h, a), 'prob': P(score)}
                - 'avg_goals': {'home': avg_h_goals, 'away': avg_a_goals, 'total': avg_total}
        """
        self._validate_lambdas(lambda_home, lambda_away)
        start_time = time.time()

        # --- Simulation Logic (Vectorized) ---
        # Draw n_simulations samples from Poisson distributions
        sim_home_goals = self._rng.poisson(lam=lambda_home, size=self.n_simulations)
        sim_away_goals = self._rng.poisson(lam=lambda_away, size=self.n_simulations)

        # --- Calculate Outcomes ---
        home_wins = sim_home_goals > sim_away_goals
        draws = sim_home_goals == sim_away_goals
        away_wins = sim_home_goals < sim_away_goals

        prob_h = np.mean(home_wins)
        prob_d = np.mean(draws)
        prob_a = np.mean(away_wins)

        # Ensure probabilities sum roughly to 1 (can have minor float issues)
        # assert np.isclose(prob_h + prob_d + prob_a, 1.0), "Probabilities do not sum to 1"

        outcome_probs = {'H': prob_h, 'D': prob_d, 'A': prob_a}

        # --- Calculate BTTS ---
        btts = (sim_home_goals > 0) & (sim_away_goals > 0)
        btts_prob = np.mean(btts)

        # --- Calculate Over/Under ---
        total_goals = sim_home_goals + sim_away_goals
        over_under_probs = {}
        ou_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        for line in ou_lines:
            over_prob = np.mean(total_goals > line)
            over_under_probs[f'Over {line}'] = over_prob
            over_under_probs[f'Under {line}'] = 1.0 - over_prob

        # --- Calculate Score Distributions ---
        # Combine goals into pairs and count occurrences
        simulated_scores = list(zip(sim_home_goals, sim_away_goals))
        score_counts = Counter(simulated_scores)

        # Calculate probabilities for each score
        score_probs = {score: count / self.n_simulations for score, count in score_counts.items()}

        # Find most likely scoreline
        if score_counts:
            most_likely_score_tuple, most_likely_count = score_counts.most_common(1)[0]
            most_likely_score_prob = most_likely_count / self.n_simulations
            most_likely_score = {'score': most_likely_score_tuple, 'prob': most_likely_score_prob}
        else: # Should not happen with n_simulations > 0
            most_likely_score = {'score': None, 'prob': 0.0}

        # --- Calculate Average Goals ---
        avg_goals = {
            'home': np.mean(sim_home_goals),
            'away': np.mean(sim_away_goals),
            'total': np.mean(total_goals)
        }

        end_time = time.time()
        # print(f"Simulation completed in {end_time - start_time:.4f} seconds.")

        return {
            'lambda_inputs': {'home': lambda_home, 'away': lambda_away},
            'n_simulations': self.n_simulations,
            'outcome_probs': outcome_probs,
            'btts_prob': btts_prob,
            'over_under_probs': over_under_probs,
            'score_counts': score_counts,
            'score_probs': score_probs,
            'most_likely_score': most_likely_score,
            'avg_goals': avg_goals
        }

    def predict(self, data: Dict[str, float]) -> Dict[str, any]:
        """
        Primary method to run simulation based on input lambda estimates.

        Args:
            data (Dict[str, float]): Dictionary containing the expected goals.
                                     Must have keys 'lambda_home' and 'lambda_away'.

        Returns:
            Dict[str, any]: The results dictionary from simulate_match.

        Raises:
            KeyError: If required lambda keys are missing.
            ValueError: If lambda values are invalid (negative).
            TypeError: If lambda values are not numeric.
        """
        if 'lambda_home' not in data or 'lambda_away' not in data:
            raise KeyError("Input data dictionary must contain 'lambda_home' and 'lambda_away'.")

        lambda_home = data['lambda_home']
        lambda_away = data['lambda_away']

        # Validation happens within simulate_match
        return self.simulate_match(lambda_home, lambda_away)

    # --- Advanced Considerations / Future Enhancements (Mentioned in Docstrings/Comments) ---
    # - Bivariate Poisson / Copulas: Model correlation between home/away goals. Requires more complex parameter estimation.
    # - Zero-Inflation (ZIP): Explicitly model excess zeros (0-0 draws). Requires estimating zero-inflation probability.
    # - Dixon-Coles Adjustments: Apply score-dependent adjustments. Requires estimating adjustment factors.
    # - Time-Varying Lambdas: Simulate goals minute-by-minute with changing lambdas (much more complex).
    # - Confidence Intervals: Calculate CIs for estimated probabilities based on n_simulations.


# Example Usage
if __name__ == '__main__':
    # --- 1. Get Lambda Estimates ---
    # **Crucial:** These values MUST come from a separate, well-calibrated model
    # trained on your data (e.g., PoissonModel, regression, xG model).
    # Example values:
    lambda_h_est = 1.75 # Example: Home team expected to score 1.75 goals
    lambda_a_est = 1.05 # Example: Away team expected to score 1.05 goals

    # --- 2. Initialize and Run Simulation ---
    mc_simulator = MonteCarloModel(n_simulations=20000, random_state=123)
    simulation_input = {'lambda_home': lambda_h_est, 'lambda_away': lambda_a_est}

    print(f"\nRunning simulation for: {simulation_input}")
    results = mc_simulator.predict(simulation_input)
    print("-" * 30)

    # --- 3. Display Key Results ---
    print("--- Simulation Results ---")
    print(f"Based on {results['n_simulations']} simulations.")
    print(f"Input Lambdas: Home={results['lambda_inputs']['home']:.3f}, Away={results['lambda_inputs']['away']:.3f}")
    print(f"Avg Goals: Home={results['avg_goals']['home']:.3f}, Away={results['avg_goals']['away']:.3f}, Total={results['avg_goals']['total']:.3f}")

    print("\nOutcome Probabilities (1X2):")
    for outcome, prob in results['outcome_probs'].items():
        implied_odds = 1 / prob if prob > 0 else float('inf')
        print(f"  P({outcome}): {prob:.4f} (Odds: {implied_odds:.2f})")

    print(f"\nBTTS Probability: {results['btts_prob']:.4f}")

    print("\nOver/Under Probabilities:")
    ou_lines_display = [0.5, 1.5, 2.5, 3.5, 4.5]
    for line in ou_lines_display:
        over_key = f'Over {line}'
        under_key = f'Under {line}'
        if over_key in results['over_under_probs']:
             print(f"  Over {line}: {results['over_under_probs'][over_key]:.4f} | Under {line}: {results['over_under_probs'][under_key]:.4f}")

    print(f"\nMost Likely Scoreline: {results['most_likely_score']['score'][0]}-{results['most_likely_score']['score'][1]} (Prob: {results['most_likely_score']['prob']:.4f})")

    print("\nTop 10 Scoreline Probabilities:")
    print(f"  Score (H-A) | Probability")
    print(f"  {'-'*25}")
    # Sort score_probs dictionary by probability (value)
    sorted_scores = sorted(results['score_probs'].items(), key=lambda item: item[1], reverse=True)
    for i, (score, prob) in enumerate(sorted_scores[:10]):
        print(f"  {score[0]:>5}-{score[1]:<5} | {prob:.4f}")

    # --- Example: Calculate probability of a specific market ---
    # Probability of Home win AND Over 2.5 goals
    h_and_o25_count = 0
    for (h, a), count in results['score_counts'].items():
        if h > a and (h + a) > 2.5:
            h_and_o25_count += count
    prob_h_and_o25 = h_and_o25_count / results['n_simulations']
    print(f"\nCalculated P(Home Win & Over 2.5): {prob_h_and_o25:.4f}")