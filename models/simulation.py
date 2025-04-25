# models/simulation.py
import logging
import pandas as pd
import numpy as np
from scipy.stats import poisson
from typing import Dict, Optional, List, Tuple, Any

logger = logging.getLogger(__name__)

def evaluate_simulation_scenarios(hg: int, ag: int) -> List[str]:
    """
    Evaluates a single simulation outcome (home goals, away goals)
    and returns a list of strings representing all scenarios met by this outcome.
    Includes simple and compound scenarios.
    """
    scenarios_met = set()
    tg = hg + ag # Total goals

    # --- Simple Scenarios ---
    # 1X2 Result
    if hg > ag: scenarios_met.add("H")
    elif hg == ag: scenarios_met.add("D")
    else: scenarios_met.add("A")

    # Double Chance
    if hg >= ag: scenarios_met.add("1X") # Home or Draw
    if hg <= ag: scenarios_met.add("X2") # Away or Draw
    if hg != ag: scenarios_met.add("12") # Home or Away

    # BTTS (Both Teams To Score)
    if hg > 0 and ag > 0: scenarios_met.add("BTTS Yes")
    else: scenarios_met.add("BTTS No")

    # Over/Under Goals (Common thresholds)
    if tg > 0.5:
        scenarios_met.add("O0.5")
    else:
        scenarios_met.add("U0.5")
        
    if tg > 1.5:
        scenarios_met.add("O1.5")
    else:
        scenarios_met.add("U1.5")
        
    if tg > 2.5:
        scenarios_met.add("O2.5")
    else:
        scenarios_met.add("U2.5")
        
    if tg > 3.5:
        scenarios_met.add("O3.5")
    else:
        scenarios_met.add("U3.5")
        
    if tg > 4.5:
        scenarios_met.add("O4.5")
    else:
        scenarios_met.add("U4.5")

    # --- Compound Scenarios ---
    # Result + O/U
    if "H" in scenarios_met and "O1.5" in scenarios_met: scenarios_met.add("H and O1.5")
    if "H" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("H and O2.5")
    if "D" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("D and U2.5")
    if "A" in scenarios_met and "O1.5" in scenarios_met: scenarios_met.add("A and O1.5")
    if "A" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("A and O2.5")

    # Double Chance + BTTS
    if "1X" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("1X and BTTS Yes")
    if "1X" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("1X and BTTS No")
    if "X2" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("X2 and BTTS Yes")
    if "X2" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("X2 and BTTS No")
    if "12" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("12 and BTTS Yes")

    # BTTS + O/U
    if "BTTS Yes" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("BTTS Yes and O2.5")
    if "BTTS No" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("BTTS No and U2.5")
    if "BTTS Yes" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("BTTS Yes and O3.5")

    # Result + BTTS
    if "H" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("H and BTTS Yes")
    if "H" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("H and BTTS No")
    if "A" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("A and BTTS Yes")
    if "A" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("A and BTTS No")
    if "D" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("D and BTTS Yes")
    if "D" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("D and BTTS No")

    return list(scenarios_met)

def run_monte_carlo_simulation(
    lambda_home: pd.Series,
    lambda_away: pd.Series,
    num_simulations: int = 10000,
    random_seed: Optional[int] = 42
) -> Optional[pd.DataFrame]:
    """
    Runs a Monte Carlo simulation based on predicted Poisson means for home/away goals.

    Args:
        lambda_home: Series of predicted expected goals for the home team (Poisson means).
                     Index should align with lambda_away and original data.
        lambda_away: Series of predicted expected goals for the away team (Poisson means).
                     Index should align with lambda_home and original data.
        num_simulations: Number of simulations to run per match.
        random_seed: Optional seed for reproducibility.

    Returns:
        DataFrame containing aggregated probabilities for various outcomes (1X2, BTTS, O/U 2.5,
        specific scorelines), indexed like the input Series. Returns None if inputs are invalid.
    """
    if not isinstance(lambda_home, pd.Series) or not isinstance(lambda_away, pd.Series):
        logger.error("lambda_home and lambda_away must be pandas Series.")
        return None
    if not lambda_home.index.equals(lambda_away.index):
        logger.error("Indices of lambda_home and lambda_away do not match.")
        return None
    if (lambda_home < 0).any() or (lambda_away < 0).any():
        logger.warning("Negative Poisson means detected. Clipping to 0.")
        lambda_home = lambda_home.clip(lower=0)
        lambda_away = lambda_away.clip(lower=0)

    num_matches = len(lambda_home)
    if num_matches == 0:
        logger.warning("Input lambda Series are empty. Returning empty DataFrame.")
        return pd.DataFrame()

    logger.info(f"Starting Monte Carlo simulation for {num_matches} matches with {num_simulations} simulations each...")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Create arrays for efficient simulation
    lambda_home_np = lambda_home.values
    lambda_away_np = lambda_away.values

    # Simulate home and away goals for all matches and all simulations at once
    # Shape: (num_matches, num_simulations)
    sim_home_goals = poisson.rvs(lambda_home_np[:, np.newaxis], size=(num_matches, num_simulations))
    sim_away_goals = poisson.rvs(lambda_away_np[:, np.newaxis], size=(num_matches, num_simulations))

    # Initialize results dictionary
    results_data = {}

    # Process each match
    for match_idx in range(num_matches):
        # Count scenarios for each simulation of this match
        scenario_counts = {}
        for sim_idx in range(num_simulations):
            hg = sim_home_goals[match_idx, sim_idx]
            ag = sim_away_goals[match_idx, sim_idx]
            scenarios = evaluate_simulation_scenarios(hg, ag)
            
            # Update counts
            for scenario in scenarios:
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        # Calculate probabilities for this match
        match_probs = {
            f"prob_{scenario}": count/num_simulations 
            for scenario, count in scenario_counts.items()
        }

        # Add to results
        for scenario, prob in match_probs.items():
            if scenario not in results_data:
                results_data[scenario] = np.zeros(num_matches)
            results_data[scenario][match_idx] = prob

    results_df = pd.DataFrame(results_data, index=lambda_home.index)

    logger.info(f"Monte Carlo simulation finished. Results shape: {results_df.shape}")
    return results_df
    return results_df
