# predict_fixture.py
import logging
import json
import pandas as pd
import numpy as np
import re # For parsing form string
import sys
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from scipy.stats import poisson
import glob # Import glob for file matching
from plotting_utils import create_combined_fixture_plot
import math # Added
from datetime import datetime, timezone # Added
from typing import Any, Dict, List, Optional, Tuple

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MONTE_CARLO_SIMULATIONS = 80000 # Increased number of simulations
TOP_N_SCENARIOS = 10 # Number of top scenarios to display
UNIFIED_DATA_DIR = "data/unified_data" # Directory containing the JSON files to process
OUTPUT_DIR = "data/output" # Directory for output results
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots") # Directory for saving plots
MC_MAX_SCORE_PLOT = 5 # Max goals for score matrix plot

# --- Helper Functions ---
def safe_get(data: Dict, keys: List[str], default: Any = None) -> Any:
    """Safely traverse nested dictionary keys."""
    if not isinstance(data, dict) or not isinstance(keys, list): return default
    current = data
    for key in keys:
        try:
            if isinstance(current, dict): current = current.get(key)
            elif isinstance(current, (list, tuple)) and isinstance(key, int):
                if 0 <= key < len(current): current = current[key]
                else: return default
            else: return default
            if current is None: return default
        except (TypeError, KeyError, IndexError): return default
    return current if current is not None else default

def parse_form_string(form_str: Optional[str], num_matches: int = 5) -> Tuple[int, int, float]:
    """Parses form string for streaks and form PPG."""
    if not form_str or len(form_str) == 0: return 0, 0, 0.0
    relevant_form = form_str[-num_matches:]
    win_streak = len(form_str) - len(form_str.rstrip('W')) if form_str.endswith('W') else 0
    loss_streak = len(form_str) - len(form_str.rstrip('L')) if form_str.endswith('L') else 0
    points = 0
    matches_counted = 0
    for char in relevant_form:
        if char == 'W': points += 3; matches_counted += 1
        elif char == 'D': points += 1; matches_counted += 1
        elif char == 'L': matches_counted += 1
    form_ppg = (points / matches_counted) if matches_counted > 0 else 0.0
    return win_streak, loss_streak, form_ppg

# --- Option 1: Analytical Poisson Calculations ---
def calculate_analytical_poisson_probs(lambda_home: float, lambda_away: float, max_goals: int = 5) -> Dict[str, Any]:
    """
    Calculates exact probabilities for outcomes based on independent Poisson distributions.
    Includes simple, DC, and compound outcomes by summing exact score probabilities.
    """
    probs = {}
    if lambda_home < 0 or lambda_away < 0:
        logger.warning("Invalid lambdas for analytical Poisson calculation.")
        return probs # Return empty dict

    # --- Exact Score Probabilities (up to max_goals) ---
    score_probs = {}
    total_prob_sum = 0.0 # To check normalization
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            score_probs[f"score_{h}-{a}"] = prob
            total_prob_sum += prob
    probs["poisson_score_probs"] = score_probs
    # logger.debug(f"Analytical Poisson total prob sum (up to {max_goals}-{max_goals}): {total_prob_sum:.4f}")

    # --- Calculate Probabilities by Summing Score Probs ---
    prob_H = 0.0; prob_D = 0.0; prob_A = 0.0
    prob_btts_yes = 0.0; prob_btts_no = 0.0
    prob_O = defaultdict(float); prob_U = defaultdict(float) # For O/U lines
    prob_compound = defaultdict(float) # For compound bets

    ou_lines = [0.5, 1.5, 2.5, 3.5, 4.5]

    for score, p in score_probs.items():
        try:
            h = int(score.split('_')[1].split('-')[0])
            a = int(score.split('_')[1].split('-')[1])
            tg = h + a
        except (ValueError, IndexError):
            continue # Skip malformed score keys

        # 1X2
        if h > a: prob_H += p
        elif h == a: prob_D += p
        else: prob_A += p

        # BTTS
        is_btts_yes = (h > 0 and a > 0)
        if is_btts_yes: prob_btts_yes += p
        else: prob_btts_no += p

        # O/U
        for n in ou_lines:
            if tg > n: prob_O[n] += p
            else: prob_U[n] += p

        # Compound Bets (add necessary conditions)
        is_h = (h > a); is_d = (h == a); is_a = (h < a)
        is_1X = (h >= a); is_X2 = (h <= a); is_12 = (h != a)

        # Result + O/U 2.5/3.5
        if is_h and tg > 2.5: prob_compound['H_and_O2.5'] += p
        if is_h and tg <= 2.5: prob_compound['H_and_U2.5'] += p
        if is_d and tg > 2.5: prob_compound['D_and_O2.5'] += p
        if is_d and tg <= 2.5: prob_compound['D_and_U2.5'] += p
        if is_a and tg > 2.5: prob_compound['A_and_O2.5'] += p
        if is_a and tg <= 2.5: prob_compound['A_and_U2.5'] += p
        if is_h and tg > 3.5: prob_compound['H_and_O3.5'] += p
        if is_h and tg <= 3.5: prob_compound['H_and_U3.5'] += p
        if is_d and tg > 3.5: prob_compound['D_and_O3.5'] += p
        if is_d and tg <= 3.5: prob_compound['D_and_U3.5'] += p
        if is_a and tg > 3.5: prob_compound['A_and_O3.5'] += p
        if is_a and tg <= 3.5: prob_compound['A_and_U3.5'] += p

        # Result + BTTS
        if is_h and is_btts_yes: prob_compound['H_and_BTTS_Yes'] += p
        if is_h and not is_btts_yes: prob_compound['H_and_BTTS_No'] += p
        if is_d and is_btts_yes: prob_compound['D_and_BTTS_Yes'] += p
        if is_d and not is_btts_yes: prob_compound['D_and_BTTS_No'] += p
        if is_a and is_btts_yes: prob_compound['A_and_BTTS_Yes'] += p
        if is_a and not is_btts_yes: prob_compound['A_and_BTTS_No'] += p

        # Double Chance + O/U 2.5/3.5
        if is_1X and tg > 2.5: prob_compound['1X_and_O2.5'] += p
        if is_1X and tg <= 2.5: prob_compound['1X_and_U2.5'] += p
        if is_X2 and tg > 2.5: prob_compound['X2_and_O2.5'] += p
        if is_X2 and tg <= 2.5: prob_compound['X2_and_U2.5'] += p
        if is_12 and tg > 2.5: prob_compound['12_and_O2.5'] += p
        if is_12 and tg <= 2.5: prob_compound['12_and_U2.5'] += p
        if is_1X and tg > 3.5: prob_compound['1X_and_O3.5'] += p
        if is_1X and tg <= 3.5: prob_compound['1X_and_U3.5'] += p
        if is_X2 and tg > 3.5: prob_compound['X2_and_O3.5'] += p
        if is_X2 and tg <= 3.5: prob_compound['X2_and_U3.5'] += p
        if is_12 and tg > 3.5: prob_compound['12_and_O3.5'] += p
        if is_12 and tg <= 3.5: prob_compound['12_and_U3.5'] += p

        # Double Chance + BTTS
        if is_1X and is_btts_yes: prob_compound['1X_and_BTTS_Yes'] += p
        if is_1X and not is_btts_yes: prob_compound['1X_and_BTTS_No'] += p
        if is_X2 and is_btts_yes: prob_compound['X2_and_BTTS_Yes'] += p
        if is_X2 and not is_btts_yes: prob_compound['X2_and_BTTS_No'] += p
        if is_12 and is_btts_yes: prob_compound['12_and_BTTS_Yes'] += p
        if is_12 and not is_btts_yes: prob_compound['12_and_BTTS_No'] += p

        # BTTS + O/U 2.5/3.5
        if is_btts_yes and tg > 2.5: prob_compound['BTTS_Yes_and_O2.5'] += p
        if is_btts_yes and tg <= 2.5: prob_compound['BTTS_Yes_and_U2.5'] += p
        if not is_btts_yes and tg > 2.5: prob_compound['BTTS_No_and_O2.5'] += p
        if not is_btts_yes and tg <= 2.5: prob_compound['BTTS_No_and_U2.5'] += p
        if is_btts_yes and tg > 3.5: prob_compound['BTTS_Yes_and_O3.5'] += p
        if is_btts_yes and tg <= 3.5: prob_compound['BTTS_Yes_and_U3.5'] += p
        if not is_btts_yes and tg > 3.5: prob_compound['BTTS_No_and_O3.5'] += p
        if not is_btts_yes and tg <= 3.5: prob_compound['BTTS_No_and_U3.5'] += p


    # --- Normalization (Optional, consider if needed based on truncation impact) ---
    prob_sum_1x2 = prob_H + prob_D + prob_A
    if prob_sum_1x2 > 1e-6 and abs(prob_sum_1x2 - total_prob_sum) > 1e-3 and total_prob_sum > 1e-6:
        logger.debug(f"Normalizing Bivariate 1X2 probs (Sum={prob_sum_1x2:.4f} vs TotalSum={total_prob_sum:.4f})")
        norm_factor = total_prob_sum / prob_sum_1x2
        prob_H *= norm_factor
        prob_D *= norm_factor
        prob_A *= norm_factor

    # Store calculated probabilities
    probs["poisson_prob_H"] = prob_H
    probs["poisson_prob_D"] = prob_D
    probs["poisson_prob_A"] = prob_A

    probs["poisson_prob_1X"] = prob_H + prob_D # Use potentially normalized values
    probs["poisson_prob_X2"] = prob_A + prob_D
    probs["poisson_prob_12"] = prob_H + prob_A

    probs["poisson_prob_BTTS_Yes"] = prob_btts_yes
    probs["poisson_prob_BTTS_No"] = prob_btts_no # Use calculated No probability

    for n in ou_lines:
        probs[f"poisson_prob_U{n}"] = prob_U[n]
        probs[f"poisson_prob_O{n}"] = prob_O[n] # Use calculated Over probability

    # Add compound probabilities (These are derived from the original score_probs, so consider if normalization should apply)
    for key, val in prob_compound.items():
        probs[f"poisson_prob_{key}"] = val

    # Note: Sum of probabilities for mutually exclusive sets (e.g., H, D, A or O2.5, U2.5)
    # might not exactly equal total_prob_sum due to floating point arithmetic or if max_goals is low.
    logger.info(f"Calculated analytical Poisson probabilities (Sum H+D+A={prob_H+prob_D+prob_A:.4f} vs Total={total_prob_sum:.4f}).")
    return probs

# --- Option 2: Elo Probability Calculation ---
def calculate_elo_probabilities(home_elo: Optional[float], away_elo: Optional[float], typical_draw_rate: float = 0.25) -> Dict[str, Optional[float]]:
    """
    Calculates H/D/A probabilities based on Elo ratings.
    Uses a simple adjustment for draw probability.
    """
    results = {"elo_prob_H": None, "elo_prob_D": None, "elo_prob_A": None}
    if home_elo is None or away_elo is None:
        logger.warning("Missing Elo rating(s), skipping Elo probability calculation.")
        return results

    try:
        home_elo = float(home_elo)
        away_elo = float(away_elo)

        elo_diff = away_elo - home_elo
        prob_h_win_exp = 1.0 / (1.0 + 10**(elo_diff / 400.0))
        prob_a_win_exp = 1.0 - prob_h_win_exp # = 1.0 / (1.0 + 10**(-elo_diff / 400.0))

        # Simple Draw Adjustment (Distribute non-draw probability according to win expectancies)
        # This is an approximation. More sophisticated methods exist (e.g., Dixon-Coles adjustment).
        prob_h = prob_h_win_exp * (1.0 - typical_draw_rate)
        prob_a = prob_a_win_exp * (1.0 - typical_draw_rate)
        prob_d = 1.0 - prob_h - prob_a # Ensure probabilities sum to 1

        # Clamp probabilities to [0, 1] range just in case of edge cases
        results["elo_prob_H"] = max(0.0, min(1.0, prob_h))
        results["elo_prob_D"] = max(0.0, min(1.0, prob_d))
        results["elo_prob_A"] = max(0.0, min(1.0, prob_a))

        logger.info(f"Calculated Elo probabilities (H/D/A): {results['elo_prob_H']:.3f}/{results['elo_prob_D']:.3f}/{results['elo_prob_A']:.3f}")

    except (ValueError, TypeError) as e:
        logger.error(f"Error converting Elo ratings to float: {e}")
    except Exception as e:
        logger.error(f"Error calculating Elo probabilities: {e}", exc_info=True)

    return results


# --- Option 3: Bivariate Poisson Calculations ---
def get_league_goal_covariance_lambda3(fixture_data: Dict) -> float:
    """Placeholder function to get estimated lambda3 (covariance term)."""
    # Example: Could look up based on league ID/name in a config file
    # league_id = safe_get(fixture_data, ['league', 'id']) # If available
    # For now, return a plausible default
    default_lambda3 = 0.10 # Adjust based on research/data
    # logger.info(f"Using default lambda3 (covariance) = {default_lambda3}")
    return default_lambda3

def bivariate_poisson_pmf(h: int, a: int, lambda1: float, lambda2: float, lambda3: float, max_k_sum: int = 15) -> float:
    """
    Calculates the probability mass function P(H=h, A=a) for a Bivariate Poisson distribution.
    Uses the parameterization lambda_home = lambda1+lambda3, lambda_away = lambda2+lambda3, cov = lambda3.
    Uses log calculations for better numerical stability.
    """
    if lambda1 < 0 or lambda2 < 0 or lambda3 < 0: return 0.0 # Lambdas must be non-negative
    if lambda1 == 0 and h > 0: return 0.0 # If lambda1 is 0, P(H>0) must be 0
    if lambda2 == 0 and a > 0: return 0.0 # If lambda2 is 0, P(A>0) must be 0

    try:
        log_term1 = -(lambda1 + lambda2 + lambda3)

        log_sum_term = -math.inf # Start with log(0)

        # Calculate the summation term in log space
        limit = min(h, a, max_k_sum) # Cap iterations
        for k in range(limit + 1):
            # Use math.lgamma for log factorials: lgamma(n+1) = log(n!)
            # Ensure arguments to lgamma are >= 0
            if h - k < 0 or a - k < 0 or k < 0: continue # Should not happen with range limit, but safety

            log_comb_h_k = math.lgamma(h + 1) - math.lgamma(k + 1) - math.lgamma(h - k + 1)
            log_comb_a_k = math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1)

            # Calculate log of lambda powers, handling log(0)
            log_lambda1_pow_hmk = (h - k) * math.log(lambda1) if lambda1 > 0 else (-math.inf if h - k > 0 else 0.0)
            log_lambda2_pow_amk = (a - k) * math.log(lambda2) if lambda2 > 0 else (-math.inf if a - k > 0 else 0.0)
            log_lambda3_pow_k = k * math.log(lambda3) if lambda3 > 0 else (-math.inf if k > 0 else 0.0)

            # Log of the k-th term in the sum (excluding factorials which cancel later)
            log_inner_term_k = log_comb_h_k + log_comb_a_k + log_lambda1_pow_hmk + log_lambda2_pow_amk + log_lambda3_pow_k

            # Combine terms using log-sum-exp trick for numerical stability
            # log(exp(a) + exp(b)) = max(a, b) + log(1 + exp(-abs(a-b)))
            if not math.isinf(log_inner_term_k): # Avoid adding -inf if term is zero
                if math.isinf(log_sum_term): # First valid term
                    log_sum_term = log_inner_term_k
                else:
                    # Manual logaddexp
                    if log_sum_term > log_inner_term_k:
                        log_sum_term = log_sum_term + math.log1p(math.exp(log_inner_term_k - log_sum_term)) # Use log1p for accuracy when exp() is small
                    else:
                        log_sum_term = log_inner_term_k + math.log1p(math.exp(log_sum_term - log_inner_term_k))


        # Full log probability: log(P(h,a)) = log_term1 + log_sum_term - log(h!) - log(a!)
        # Check if sum term remained -inf (meaning all terms were zero)
        if math.isinf(log_sum_term):
            return 0.0

        log_h_factorial = math.lgamma(h + 1)
        log_a_factorial = math.lgamma(a + 1)

        log_prob = log_term1 + log_sum_term - log_h_factorial - log_a_factorial

        # Convert back from log probability
        prob = math.exp(log_prob)
        return max(0.0, prob) # Ensure non-negative prob due to potential floating point inaccuracies

    except (ValueError, OverflowError) as e:
        # Fallback or warning if log calculations fail
        logger.debug(f"Numerical issue calculating Bivariate PMF log for h={h}, a={a}, l1={lambda1}, l2={lambda2}, l3={lambda3}: {e}. Trying direct calculation (less stable).")
        # Attempt direct calculation as a fallback (less numerically stable)
        try:
            term1 = math.exp(-(lambda1 + lambda2 + lambda3))
            sum_term = 0.0
            limit = min(h, a, max_k_sum)
            for k in range(limit + 1):
                 # Use math.comb and math.factorial carefully
                 comb_h_k = math.comb(h, k)
                 comb_a_k = math.comb(a, k)
                 fact_k = math.factorial(k)
                 pow_l1 = lambda1**(h-k) if lambda1 > 0 or h-k == 0 else 0.0
                 pow_l2 = lambda2**(a-k) if lambda2 > 0 or a-k == 0 else 0.0
                 pow_l3 = lambda3**k if lambda3 > 0 or k == 0 else 0.0
                 term_k = comb_h_k * comb_a_k * fact_k * pow_l1 * pow_l2 * pow_l3
                 sum_term += term_k

            fact_h = math.factorial(h)
            fact_a = math.factorial(a)
            if fact_h == 0 or fact_a == 0: return 0.0 # Avoid division by zero if h or a is large enough to overflow factorial

            prob = term1 * sum_term / (fact_h * fact_a)
            return max(0.0, prob)
        except (ValueError, OverflowError) as e_direct:
             logger.error(f"Direct Bivariate PMF calculation also failed for h={h}, a={a}: {e_direct}")
             return 0.0 # Return 0 probability on numerical errors

def calculate_bivariate_poisson_probs(lambda_home: float, lambda_away: float, lambda3: float, max_goals: int = 5) -> Dict[str, Any]:
    """
    Calculates outcome probabilities using the Bivariate Poisson model.
    Includes simple, DC, and compound outcomes by summing exact score probabilities.
    """
    probs = {}
    if lambda_home < 0 or lambda_away < 0 or lambda3 < 0 or lambda_home < lambda3 or lambda_away < lambda3:
        logger.warning(f"Invalid lambdas for Bivariate Poisson: H={lambda_home}, A={lambda_away}, L3={lambda3}. Skipping.")
        return probs

    lambda1 = max(0.0, lambda_home - lambda3) # Ensure non-negative
    lambda2 = max(0.0, lambda_away - lambda3) # Ensure non-negative

    # --- Exact Score Probabilities ---
    score_probs = {}
    total_prob_sum = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = bivariate_poisson_pmf(h, a, lambda1, lambda2, lambda3)
            score_probs[f"score_{h}-{a}"] = prob
            total_prob_sum += prob
    probs["biv_poisson_score_probs"] = score_probs
    logger.info(f"Calculated Bivariate Poisson probabilities (Sum up to {max_goals}-{max_goals}: {total_prob_sum:.4f}).") # Sum might be < 1

    # --- Calculate Probabilities by Summing Score Probs ---
    prob_H = 0.0; prob_D = 0.0; prob_A = 0.0
    prob_btts_yes = 0.0; prob_btts_no = 0.0
    prob_O = defaultdict(float); prob_U = defaultdict(float) # For O/U lines
    prob_compound = defaultdict(float) # For compound bets

    ou_lines = [0.5, 1.5, 2.5, 3.5, 4.5]

    for score, p in score_probs.items():
        try:
            h = int(score.split('_')[1].split('-')[0])
            a = int(score.split('_')[1].split('-')[1])
            tg = h + a
        except (ValueError, IndexError):
            continue # Skip malformed score keys

        # 1X2
        if h > a: prob_H += p
        elif h == a: prob_D += p
        else: prob_A += p

        # BTTS
        is_btts_yes = (h > 0 and a > 0)
        if is_btts_yes: prob_btts_yes += p
        else: prob_btts_no += p

        # O/U
        for n in ou_lines:
            if tg > n: prob_O[n] += p
            else: prob_U[n] += p

        # Compound Bets (add necessary conditions)
        is_h = (h > a); is_d = (h == a); is_a = (h < a)
        is_1X = (h >= a); is_X2 = (h <= a); is_12 = (h != a)

        # Result + O/U 2.5/3.5
        if is_h and tg > 2.5: prob_compound['H_and_O2.5'] += p
        if is_h and tg <= 2.5: prob_compound['H_and_U2.5'] += p
        if is_d and tg > 2.5: prob_compound['D_and_O2.5'] += p
        if is_d and tg <= 2.5: prob_compound['D_and_U2.5'] += p
        if is_a and tg > 2.5: prob_compound['A_and_O2.5'] += p
        if is_a and tg <= 2.5: prob_compound['A_and_U2.5'] += p
        if is_h and tg > 3.5: prob_compound['H_and_O3.5'] += p
        if is_h and tg <= 3.5: prob_compound['H_and_U3.5'] += p
        if is_d and tg > 3.5: prob_compound['D_and_O3.5'] += p
        if is_d and tg <= 3.5: prob_compound['D_and_U3.5'] += p
        if is_a and tg > 3.5: prob_compound['A_and_O3.5'] += p
        if is_a and tg <= 3.5: prob_compound['A_and_U3.5'] += p

        # Result + BTTS
        if is_h and is_btts_yes: prob_compound['H_and_BTTS_Yes'] += p
        if is_h and not is_btts_yes: prob_compound['H_and_BTTS_No'] += p
        if is_d and is_btts_yes: prob_compound['D_and_BTTS_Yes'] += p
        if is_d and not is_btts_yes: prob_compound['D_and_BTTS_No'] += p
        if is_a and is_btts_yes: prob_compound['A_and_BTTS_Yes'] += p
        if is_a and not is_btts_yes: prob_compound['A_and_BTTS_No'] += p

        # Double Chance + O/U 2.5/3.5
        if is_1X and tg > 2.5: prob_compound['1X_and_O2.5'] += p
        if is_1X and tg <= 2.5: prob_compound['1X_and_U2.5'] += p
        if is_X2 and tg > 2.5: prob_compound['X2_and_O2.5'] += p
        if is_X2 and tg <= 2.5: prob_compound['X2_and_U2.5'] += p
        if is_12 and tg > 2.5: prob_compound['12_and_O2.5'] += p
        if is_12 and tg <= 2.5: prob_compound['12_and_U2.5'] += p
        if is_1X and tg > 3.5: prob_compound['1X_and_O3.5'] += p
        if is_1X and tg <= 3.5: prob_compound['1X_and_U3.5'] += p
        if is_X2 and tg > 3.5: prob_compound['X2_and_O3.5'] += p
        if is_X2 and tg <= 3.5: prob_compound['X2_and_U3.5'] += p
        if is_12 and tg > 3.5: prob_compound['12_and_O3.5'] += p
        if is_12 and tg <= 3.5: prob_compound['12_and_U3.5'] += p

        # Double Chance + BTTS
        if is_1X and is_btts_yes: prob_compound['1X_and_BTTS_Yes'] += p
        if is_1X and not is_btts_yes: prob_compound['1X_and_BTTS_No'] += p
        if is_X2 and is_btts_yes: prob_compound['X2_and_BTTS_Yes'] += p
        if is_X2 and not is_btts_yes: prob_compound['X2_and_BTTS_No'] += p
        if is_12 and is_btts_yes: prob_compound['12_and_BTTS_Yes'] += p
        if is_12 and not is_btts_yes: prob_compound['12_and_BTTS_No'] += p

        # BTTS + O/U 2.5/3.5
        if is_btts_yes and tg > 2.5: prob_compound['BTTS_Yes_and_O2.5'] += p
        if is_btts_yes and tg <= 2.5: prob_compound['BTTS_Yes_and_U2.5'] += p
        if not is_btts_yes and tg > 2.5: prob_compound['BTTS_No_and_O2.5'] += p
        if not is_btts_yes and tg <= 2.5: prob_compound['BTTS_No_and_U2.5'] += p
        if is_btts_yes and tg > 3.5: prob_compound['BTTS_Yes_and_O3.5'] += p
        if is_btts_yes and tg <= 3.5: prob_compound['BTTS_Yes_and_U3.5'] += p
        if not is_btts_yes and tg > 3.5: prob_compound['BTTS_No_and_O3.5'] += p
        if not is_btts_yes and tg <= 3.5: prob_compound['BTTS_No_and_U3.5'] += p


    # --- Normalization (Optional, consider if needed based on truncation impact) ---
    prob_sum_1x2 = prob_H + prob_D + prob_A
    if prob_sum_1x2 > 1e-6 and abs(prob_sum_1x2 - total_prob_sum) > 1e-3 and total_prob_sum > 1e-6:
        logger.debug(f"Normalizing Bivariate 1X2 probs (Sum={prob_sum_1x2:.4f} vs TotalSum={total_prob_sum:.4f})")
        norm_factor = total_prob_sum / prob_sum_1x2
        prob_H *= norm_factor
        prob_D *= norm_factor
        prob_A *= norm_factor

    # Store calculated probabilities
    probs["biv_poisson_prob_H"] = prob_H
    probs["biv_poisson_prob_D"] = prob_D
    probs["biv_poisson_prob_A"] = prob_A

    probs["biv_poisson_prob_1X"] = prob_H + prob_D # Use potentially normalized values
    probs["biv_poisson_prob_X2"] = prob_A + prob_D
    probs["biv_poisson_prob_12"] = prob_H + prob_A

    probs["biv_poisson_prob_BTTS_Yes"] = prob_btts_yes
    probs["biv_poisson_prob_BTTS_No"] = prob_btts_no # Use calculated No probability

    for n in ou_lines:
        probs[f"biv_poisson_prob_U{n}"] = prob_U[n]
        probs[f"biv_poisson_prob_O{n}"] = prob_O[n] # Use calculated Over probability

    # Add compound probabilities (These are derived from the original score_probs, so consider if normalization should apply)
    for key, val in prob_compound.items():
        probs[f"biv_poisson_prob_{key}"] = val

    return probs


# --- Option 4: Refined Lambda Calculation (Example: Time-Weighted) ---
def calculate_weighted_strength_lambdas(fixture_data: Dict, decay_per_day: float = 0.005) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates lambdas using time-weighted historical averages from snapshot data.
    NOTE: Uses 'statarea_snapshot' history, which might differ from point-in-time DB query.
    Requires 'fixture_meta.date_utc' in the fixture data.
    """
    logger.info("Calculating time-weighted strength-adjusted lambdas...")

    home_snap = safe_get(fixture_data, ['raw_data', 'home', 'statarea_snapshot'], {})
    away_snap = safe_get(fixture_data, ['raw_data', 'away', 'statarea_snapshot'], {})
    fixture_date_str = safe_get(fixture_data, ['fixture_meta', 'date_utc'])

    if not home_snap or not away_snap or not fixture_date_str:
        logger.warning("Missing snapshot data or fixture date_utc for weighted lambda calculation.")
        return None, None

    try:
        # Use timezone-aware datetime object for comparison
        # Handle potential 'Z' suffix indicating UTC
        if fixture_date_str.endswith('Z'):
            fixture_date_str = fixture_date_str[:-1] + '+00:00'
        fixture_dt = datetime.fromisoformat(fixture_date_str).astimezone(timezone.utc)
    except (ValueError, TypeError) as e:
        logger.error(f"Could not parse fixture date_utc: {fixture_date_str}. Error: {e}")
        return None, None

    def get_weighted_averages(snapshot: Dict, perspective: str) -> Tuple[Optional[float], Optional[float]]:
        """Calculates weighted avg goals scored/conceded from match history."""
        history = snapshot.get("match_history", [])
        if not history:
            logger.warning(f"No match_history found in statarea_snapshot for {perspective} team.")
            return None, None

        total_weight_scored = 0.0
        total_weight_conceded = 0.0
        weighted_sum_scored = 0.0
        weighted_sum_conceded = 0.0
        matches_considered = 0

        for match in history:
            try:
                match_date_str = match.get("date")
                if not match_date_str: continue # Skip if no date

                # Assume match date is also UTC, handle potential formats
                if match_date_str.endswith('Z'):
                     match_date_str = match_date_str[:-1] + '+00:00'
                # Handle potential missing timezone info - assume UTC if naive
                parsed_dt = datetime.fromisoformat(match_date_str)
                if parsed_dt.tzinfo is None:
                    match_dt = parsed_dt.replace(tzinfo=timezone.utc)
                else:
                    match_dt = parsed_dt.astimezone(timezone.utc) # Make timezone aware

                days_diff = (fixture_dt - match_dt).days
                if days_diff < 0:
                    # logger.debug(f"Skipping future match date in history: {match_date_str}")
                    continue # Should not happen with snapshots, but safety check

                # Simple exponential decay weight: weight = (1 - decay_per_day)^days_diff
                weight = (1.0 - decay_per_day) ** days_diff
                if weight < 1e-6: continue # Ignore very old matches

                # Goal perspective depends on whether we are home or away IN THE SNAPSHOT
                # Assumes statarea snapshot history always contains 'team_goals' and 'opponent_goals'
                goals_scored = match.get("team_goals")
                goals_conceded = match.get("opponent_goals")

                # Ensure goals are numeric and not None
                if goals_scored is None or goals_conceded is None:
                    # logger.debug(f"Skipping match due to missing goals: {match}")
                    continue

                goals_scored = float(goals_scored)
                goals_conceded = float(goals_conceded)

                weighted_sum_scored += goals_scored * weight
                total_weight_scored += weight
                weighted_sum_conceded += goals_conceded * weight
                total_weight_conceded += weight
                matches_considered += 1

            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Skipping match in weighted calc due to error: {e} - Match: {match}")
                continue

        if matches_considered == 0:
            logger.warning(f"No valid matches found in history for weighted average calculation ({perspective}).")
            return None, None

        avg_scored = (weighted_sum_scored / total_weight_scored) if total_weight_scored > 1e-9 else 0.0
        avg_conceded = (weighted_sum_conceded / total_weight_conceded) if total_weight_conceded > 1e-9 else 0.0
        logger.debug(f"Weighted avg goals ({perspective}, {matches_considered} matches): Scored={avg_scored:.3f} (WeightSum={total_weight_scored:.3f}), Conceded={avg_conceded:.3f} (WeightSum={total_weight_conceded:.3f})")
        return avg_scored, avg_conceded

    # --- Calculate weighted averages for the specific fixture context ---
    h_avg_for_weighted, h_avg_conceded_weighted = get_weighted_averages(home_snap, "home")
    a_avg_for_weighted, a_avg_conceded_weighted = get_weighted_averages(away_snap, "away")

    # Validate weighted results
    if h_avg_for_weighted is None or h_avg_conceded_weighted is None or a_avg_for_weighted is None or a_avg_conceded_weighted is None:
        logger.warning("Failed to calculate one or more weighted averages, cannot compute weighted lambdas.")
        return None, None

    # --- Alternative Approach: Directly use weighted averages as expected goals ---
    # Expected Home Goals = Average of (Home Team's Weighted Attack) and (Away Team's Weighted Defense)
    # Expected Away Goals = Average of (Away Team's Weighted Attack) and (Home Team's Weighted Defense)
    try:
        lambda_home_w = (h_avg_for_weighted + a_avg_conceded_weighted) / 2.0
        lambda_away_w = (a_avg_for_weighted + h_avg_conceded_weighted) / 2.0

        lambda_home_w = max(0.0, lambda_home_w) # Ensure non-negative
        lambda_away_w = max(0.0, lambda_away_w)

        logger.info(f"Calculated Weighted Lambdas (Direct Avg): Home={lambda_home_w:.3f}, Away={lambda_away_w:.3f}")
        return lambda_home_w, lambda_away_w

    except (ValueError, TypeError, ZeroDivisionError) as e:
        logger.error(f"Error during weighted lambda direct average calculation: {e}")
        return None, None
    
def calculate_strength_adjusted_lambdas(fixture_data: Dict) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates attack/defense strengths based on home/away splits vs overall average
        and returns adjusted lambda values for Monte Carlo.
        """
        logger.info("Calculating strength-adjusted lambdas...")

        home_snap = safe_get(fixture_data, ['raw_data', 'home', 'match_processor_snapshot'], {})
        away_snap = safe_get(fixture_data, ['raw_data', 'away', 'match_processor_snapshot'], {})

        # --- Get Average Goals Scored and Conceded (Home/Away/Total) ---
        # Home Team
        h_avg_for_home = safe_get(home_snap, ['goals', 'for', 'average', 'home'])
        h_avg_conceded_home = safe_get(home_snap, ['goals', 'against', 'average', 'home'])
        h_avg_for_total = safe_get(home_snap, ['goals', 'for', 'average', 'total'])
        h_avg_conceded_total = safe_get(home_snap, ['goals', 'against', 'average', 'total'])

        # Away Team
        a_avg_for_away = safe_get(away_snap, ['goals', 'for', 'average', 'away'])
        a_avg_conceded_away = safe_get(away_snap, ['goals', 'against', 'average', 'away'])
        a_avg_for_total = safe_get(away_snap, ['goals', 'for', 'average', 'total'])
        a_avg_conceded_total = safe_get(away_snap, ['goals', 'against', 'average', 'total'])

        # --- Validate data ---
        required_vals = [
            h_avg_for_home, h_avg_conceded_home, h_avg_for_total, h_avg_conceded_total,
            a_avg_for_away, a_avg_conceded_away, a_avg_for_total, a_avg_conceded_total
        ]
        if None in required_vals:
            logger.error("Missing required average goal data in JSON snapshot for strength calculation. Cannot calculate adjusted lambdas.")
            # Return None for both lambdas if data is missing
            return None, None

        try:
            # Convert to float, handling potential non-numeric types gracefully
            h_avg_for_home = float(h_avg_for_home)
            h_avg_conceded_home = float(h_avg_conceded_home)
            # Prevent division by zero for total averages, use 1.0 as a neutral fallback
            h_avg_for_total = float(h_avg_for_total) if h_avg_for_total and h_avg_for_total > 0 else 1.0
            h_avg_conceded_total = float(h_avg_conceded_total) if h_avg_conceded_total and h_avg_conceded_total > 0 else 1.0

            a_avg_for_away = float(a_avg_for_away)
            a_avg_conceded_away = float(a_avg_conceded_away)
            a_avg_for_total = float(a_avg_for_total) if a_avg_for_total and a_avg_for_total > 0 else 1.0
            a_avg_conceded_total = float(a_avg_conceded_total) if a_avg_conceded_total and a_avg_conceded_total > 0 else 1.0

        except (ValueError, TypeError) as e:
            logger.error(f"Error converting goal average data to float: {e}. Cannot calculate adjusted lambdas.")
            return None, None

        # --- Calculate Strength Ratios ---
        # Attack Strength = Avg Goals For (Home/Away) / Avg Goals For (Total)
        home_attack_strength = h_avg_for_home / h_avg_for_total
        away_attack_strength = a_avg_for_away / a_avg_for_total

        # Defense Strength = Avg Goals Conceded (Home/Away) / Avg Goals Conceded (Total)
        home_defense_strength = h_avg_conceded_home / h_avg_conceded_total
        away_defense_strength = a_avg_conceded_away / a_avg_conceded_total

        # --- Calculate Expected Goals (Lambdas) ---
        # Lambda Home = Home Attack Strength * Away Defense Strength * Overall Avg Goals For (e.g., Home Team's overall avg)
        # Lambda Away = Away Attack Strength * Home Defense Strength * Overall Avg Goals For (e.g., Away Team's overall avg)
        # The multiplication by the team's *own* total average scales the strength interaction correctly.
        lambda_home = home_attack_strength * away_defense_strength * h_avg_for_total
        lambda_away = away_attack_strength * home_defense_strength * a_avg_for_total

        # Ensure non-negative
        lambda_home = max(0.0, lambda_home)
        lambda_away = max(0.0, lambda_away)

        logger.info(f"Strengths: H Att={home_attack_strength:.2f}, H Def={home_defense_strength:.2f}, A Att={away_attack_strength:.2f}, A Def={away_defense_strength:.2f}")
        logger.info(f"Calculated Strength-Adjusted Lambdas (Original): Home={lambda_home:.3f}, Away={lambda_away:.3f}")

        return lambda_home, lambda_away


# --- Monte Carlo Simulation ---
def evaluate_simulation_scenarios(hg: int, ag: int) -> Tuple[List[str], str]: # Return score string too
    """
    Evaluates a single simulation outcome (hg, ag) for various simple
    and compound scenarios. Includes more combinations like Result/DC + U/O 3.5.
    """
    scenarios_met = set()
    tg = hg + ag # Total goals
    score_string = f"{hg}-{ag}" # Create score string

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
    if tg > 0.5: scenarios_met.add("O0.5"); scenarios_met.add("U0.5 No") # Explicit No
    else: scenarios_met.add("U0.5"); scenarios_met.add("O0.5 No")

    if tg > 1.5: scenarios_met.add("O1.5"); scenarios_met.add("U1.5 No")
    else: scenarios_met.add("U1.5"); scenarios_met.add("O1.5 No")

    if tg > 2.5: scenarios_met.add("O2.5"); scenarios_met.add("U2.5 No")
    else: scenarios_met.add("U2.5"); scenarios_met.add("O2.5 No")

    if tg > 3.5: scenarios_met.add("O3.5"); scenarios_met.add("U3.5 No")
    else: scenarios_met.add("U3.5"); scenarios_met.add("O3.5 No")

    if tg > 4.5: scenarios_met.add("O4.5"); scenarios_met.add("U4.5 No")
    else: scenarios_met.add("U4.5"); scenarios_met.add("O4.5 No")


    # --- Compound Scenarios (Expanded) ---

    # Result + O/U 2.5
    if "H" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("H and O2.5")
    if "H" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("H and U2.5")
    if "D" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("D and O2.5")
    if "D" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("D and U2.5")
    if "A" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("A and O2.5")
    if "A" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("A and U2.5")

    # Result + O/U 3.5 (Added)
    if "H" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("H and O3.5")
    if "H" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("H and U3.5")
    if "D" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("D and O3.5")
    if "D" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("D and U3.5")
    if "A" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("A and O3.5")
    if "A" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("A and U3.5")

    # Double Chance + BTTS
    if "1X" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("1X and BTTS Yes")
    if "1X" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("1X and BTTS No")
    if "X2" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("X2 and BTTS Yes")
    if "X2" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("X2 and BTTS No")
    if "12" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("12 and BTTS Yes")
    if "12" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("12 and BTTS No") # Added 12 + BTTS No

    # Double Chance + O/U 2.5 (Added)
    if "1X" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("1X and O2.5")
    if "1X" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("1X and U2.5")
    if "X2" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("X2 and O2.5")
    if "X2" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("X2 and U2.5")
    if "12" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("12 and O2.5")
    if "12" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("12 and U2.5")

    # Double Chance + O/U 3.5 (Added)
    if "1X" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("1X and O3.5")
    if "1X" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("1X and U3.5")
    if "X2" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("X2 and O3.5")
    if "X2" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("X2 and U3.5")
    if "12" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("12 and O3.5")
    if "12" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("12 and U3.5")


    # BTTS + O/U
    if "BTTS Yes" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("BTTS Yes and O2.5")
    if "BTTS Yes" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("BTTS Yes and U2.5") # Added
    if "BTTS No" in scenarios_met and "O2.5" in scenarios_met: scenarios_met.add("BTTS No and O2.5")   # Added
    if "BTTS No" in scenarios_met and "U2.5" in scenarios_met: scenarios_met.add("BTTS No and U2.5")
    if "BTTS Yes" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("BTTS Yes and O3.5")
    if "BTTS Yes" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("BTTS Yes and U3.5") # Added
    if "BTTS No" in scenarios_met and "O3.5" in scenarios_met: scenarios_met.add("BTTS No and O3.5")   # Added
    if "BTTS No" in scenarios_met and "U3.5" in scenarios_met: scenarios_met.add("BTTS No and U3.5")   # Added


    # Result + BTTS
    if "H" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("H and BTTS Yes")
    if "H" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("H and BTTS No")
    if "A" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("A and BTTS Yes")
    if "A" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("A and BTTS No")
    if "D" in scenarios_met and "BTTS Yes" in scenarios_met: scenarios_met.add("D and BTTS Yes")
    if "D" in scenarios_met and "BTTS No" in scenarios_met: scenarios_met.add("D and BTTS No")

    return list(scenarios_met), score_string

def run_monte_carlo_simulation(
    lambda_home: float,
    lambda_away: float,
    num_simulations: int = MONTE_CARLO_SIMULATIONS,
    random_seed: Optional[int] = 42,
    max_score_plot: int = 5 # Max goals per side for score matrix
) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]: # Return two dicts
    """
    Runs MC simulation for a single fixture based on lambda values.
    Returns:
        - Dictionary of scenario probabilities (prob_X).
        - Dictionary of scoreline probabilities (score_X-Y).
    """
    logger.info(f"Starting Monte Carlo simulation with {num_simulations} runs...")
    if lambda_home < 0 or lambda_away < 0:
        logger.error("Negative lambda values provided for Monte Carlo.")
        return None, None

    if random_seed is not None:
        np.random.seed(random_seed)

    # Simulate goals for the single match
    sim_home_goals = poisson.rvs(lambda_home, size=num_simulations)
    sim_away_goals = poisson.rvs(lambda_away, size=num_simulations)

    # Count scenarios and scorelines across all simulations
    scenario_counts = Counter()
    scoreline_counts = Counter()
    for i in range(num_simulations):
        hg = sim_home_goals[i]
        ag = sim_away_goals[i]
        scenarios_this_run, score_string = evaluate_simulation_scenarios(hg, ag)
        scenario_counts.update(scenarios_this_run)
        # Only count scorelines within the desired plot range
        if hg <= max_score_plot and ag <= max_score_plot:
            scoreline_counts[score_string] += 1 # Use the score string as key

    if not scenario_counts:
        logger.warning("Monte Carlo simulation did not produce any scenarios.")
        return {}, {} # Return empty dicts

    # Calculate probabilities
    total_sims_float = float(num_simulations)
    scenario_probabilities = {
        f"prob_{scenario}": count / total_sims_float
        for scenario, count in scenario_counts.items()
    }
    scoreline_probabilities = {
         f"score_{score}": count / total_sims_float
         for score, count in scoreline_counts.items()
    }

    logger.info(f"Monte Carlo simulation finished. Found {len(scenario_probabilities)} unique scenarios and {len(scoreline_probabilities)} scorelines (up to {max_score_plot}-{max_score_plot}).")
    return scenario_probabilities, scoreline_probabilities

# --- Helper Function to Identify Unique Concepts ---
def get_selection_concept(selection_key: str) -> Optional[Tuple]:
    """
    Parses a selection key (e.g., 'prob_O2.5', 'poisson_prob_H and BTTS No', 'biv_poisson_score_1-1')
    and returns a canonical, comparable representation of the underlying concept.
    Returns None if the key format is unrecognized.
    Handles different prefixes (prob_, poisson_prob_, biv_poisson_prob_).
    """
    # Remove prefixes first
    key = selection_key
    prefixes_to_remove = ["prob_", "poisson_prob_", "biv_poisson_prob_"]
    for prefix in prefixes_to_remove:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break # Stop after removing one prefix

    # Simple outcomes
    if key in ["H", "D", "A", "1X", "X2", "12"]: return ("result", key)
    if key == "BTTS Yes": return ("btts", True)
    if key == "BTTS No": return ("btts", False)

    # Over/Under - Adjusted to correctly identify canonical concept
    ou_match = re.match(r"([OU])(\d+\.\d+)(\sNo)?", key)
    if ou_match:
        ou_type, value_str, is_no = ou_match.groups()
        value = float(value_str)
        # Canonical is Over: O2.5 == U2.5 No
        is_over_concept = (ou_type == "O" and not is_no) or (ou_type == "U" and is_no)
        # Canonical concept: (type, line, is_over)
        return ("total_goals", value, is_over_concept) # Concept always represents Over or Under

    # Compound bets (add more patterns as needed)
    # Result + O/U - Adjusted to correctly identify canonical concept
    res_ou_match = re.match(r"([HDA])\s+and\s+([OU])(\d+\.\d+)(\sNo)?", key) # Added (\sNo)?
    if res_ou_match:
        res, ou_type, value_str, is_no = res_ou_match.groups()
        value = float(value_str)
        is_over_concept = (ou_type == "O" and not is_no) or (ou_type == "U" and is_no)
        # Canonical: (type, result, line, is_over)
        return ("result_ou", res, value, is_over_concept)

    # Result + BTTS
    res_btts_match = re.match(r"([HDA])\s+and\s+BTTS\s+(Yes|No)", key)
    if res_btts_match:
        res, btts_status = res_btts_match.groups()
        # Canonical: (type, result, btts_is_yes)
        return ("result_btts", res, btts_status == "Yes")

    # Double Chance + O/U - Adjusted
    dc_ou_match = re.match(r"(1X|X2|12)\s+and\s+([OU])(\d+\.\d+)(\sNo)?", key) # Added (\sNo)?
    if dc_ou_match:
        dc, ou_type, value_str, is_no = dc_ou_match.groups()
        value = float(value_str)
        is_over_concept = (ou_type == "O" and not is_no) or (ou_type == "U" and is_no)
        # Canonical: (type, dc_result, line, is_over)
        return ("dc_ou", dc, value, is_over_concept)

    # Double Chance + BTTS
    dc_btts_match = re.match(r"(1X|X2|12)\s+and\s+BTTS\s+(Yes|No)", key)
    if dc_btts_match:
        dc, btts_status = dc_btts_match.groups()
        # Canonical: (type, dc_result, btts_is_yes)
        return ("dc_btts", dc, btts_status == "Yes")

    # BTTS + O/U - Adjusted
    btts_ou_match = re.match(r"BTTS\s+(Yes|No)\s+and\s+([OU])(\d+\.\d+)(\sNo)?", key) # Added (\sNo)?
    if btts_ou_match:
        btts_status, ou_type, value_str, is_no = btts_ou_match.groups()
        value = float(value_str)
        is_over_concept = (ou_type == "O" and not is_no) or (ou_type == "U" and is_no)
        # Canonical: (type, btts_is_yes, line, is_over)
        return ("btts_ou", btts_status == "Yes", value, is_over_concept)

    # Exact Score
    # Note: Prefixes for score keys differ ('score_X-Y', 'poisson_score_probs': {'score_X-Y': ...}, 'biv_poisson_score_probs': {'score_X-Y': ...})
    # We need to handle the structure difference here or in the calling function.
    # Let's assume the input key is already normalized to 'score_H-A' if it's a score.
    score_match = re.match(r"score_(\d+)-(\d+)", key)
    if score_match:
        h, a = map(int, score_match.groups())
        return ("score", h, a)

    logger.debug(f"Unrecognized selection key format for concept mapping: {selection_key}")
    return None

# --- NEW: Function to Reconstruct Display Key ---
def reconstruct_display_key(concept: Tuple) -> str:
    """ Reconstructs a user-friendly display key from a concept tuple. """
    concept_type = concept[0]
    try:
        if concept_type == "result": # ('result', key)
            return concept[1]
        elif concept_type == "btts": # ('btts', is_yes)
            return "BTTS Yes" if concept[1] else "BTTS No"
        elif concept_type == "total_goals": # ('total_goals', value, is_over)
            value, is_over = concept[1], concept[2]
            return f"O{value}" if is_over else f"U{value}"
        elif concept_type == "result_ou": # ('result_ou', res, value, is_over)
            res, value, is_over = concept[1], concept[2], concept[3]
            ou_str = f"O{value}" if is_over else f"U{value}"
            return f"{res} and {ou_str}"
        elif concept_type == "result_btts": # ('result_btts', res, btts_is_yes)
            res, is_yes = concept[1], concept[2]
            btts_str = "BTTS Yes" if is_yes else "BTTS No"
            return f"{res} and {btts_str}"
        elif concept_type == "dc_ou": # ('dc_ou', dc_result, value, is_over)
            dc, value, is_over = concept[1], concept[2], concept[3]
            ou_str = f"O{value}" if is_over else f"U{value}"
            return f"{dc} and {ou_str}"
        elif concept_type == "dc_btts": # ('dc_btts', dc_result, btts_is_yes)
            dc, is_yes = concept[1], concept[2]
            btts_str = "BTTS Yes" if is_yes else "BTTS No"
            return f"{dc} and {btts_str}"
        elif concept_type == "btts_ou": # ('btts_ou', btts_is_yes, value, is_over)
            is_yes, value, is_over = concept[1], concept[2], concept[3]
            btts_str = "BTTS Yes" if is_yes else "BTTS No"
            ou_str = f"O{value}" if is_over else f"U{value}"
            return f"{btts_str} and {ou_str}"
        elif concept_type == "score": # ('score', h, a)
             return f"Score {concept[1]}-{concept[2]}"
        else:
            logger.warning(f"Unknown concept type for display key reconstruction: {concept_type}")
            return "Unknown Concept"
    except IndexError:
         logger.error(f"Index error reconstructing display key for concept: {concept}")
         return "Error Key"

# --- NEW: Function to Calculate Combined Top Selections (v3 - Stricter Uniqueness) ---
def calculate_combined_top_selections(
    mc_probs: Optional[Dict],
    analytical_probs: Optional[Dict],
    bivariate_probs: Optional[Dict],
    top_n: int = TOP_N_SCENARIOS
) -> List[Dict]:
    """
    Aggregates probabilities, calculates average probability per unique concept,
    and returns the top N unique selections, robustly handling duplicates and opposites.
    """
    logger.info(f"Calculating top {top_n} combined unique selections (v3)...")

    # Step 1: Collect probabilities per canonical concept
    concept_to_probabilities = defaultdict(list)
    sources = {
        "mc": (mc_probs, "prob_", "mc_score_probs"),
        "analytical": (analytical_probs, "poisson_prob_", "poisson_score_probs"),
        "bivariate": (bivariate_probs, "biv_poisson_prob_", "biv_poisson_score_probs")
    }
    for source_name, (prob_dict, prefix, score_key) in sources.items():
        if not prob_dict: continue
        # Process non-score probabilities
        for key, prob in prob_dict.items():
            if key == score_key: continue
            if not isinstance(prob, (int, float)) or math.isnan(prob): continue
            concept = get_selection_concept(key)
            if concept: concept_to_probabilities[concept].append(prob)
        # Process score probabilities
        score_probs_dict = prob_dict.get(score_key)
        if isinstance(score_probs_dict, dict):
            for score_key_inner, score_prob in score_probs_dict.items():
                if not isinstance(score_prob, (int, float)) or math.isnan(score_prob): continue
                concept = get_selection_concept(score_key_inner)
                if concept: concept_to_probabilities[concept].append(score_prob)

    # Step 2: Calculate average probability and reconstruct display key for each unique concept
    aggregated_concept_data = []
    for concept, probs_list in concept_to_probabilities.items():
        if probs_list:
            avg_prob = sum(probs_list) / len(probs_list)
            display_key = reconstruct_display_key(concept)
            if display_key != "Unknown Concept" and display_key != "Error Key":
                 aggregated_concept_data.append({
                     'concept': concept,
                     'probability': avg_prob,
                     'selection': display_key
                 })
        else:
             logger.warning(f"Concept {concept} had no valid probability values collected.")

    # Step 3: Sort the aggregated data by probability descending
    try:
        aggregated_concept_data.sort(key=lambda item: item['probability'], reverse=True)
    except Exception as e:
        logger.error(f"Error sorting aggregated concept data: {e}", exc_info=True)
        return []

    # Step 4: Filter for Top N Unique Concepts, tracking seen concepts AND display keys
    top_n_unique_selections = []
    seen_concepts = set() # Track canonical concept tuples seen
    seen_display_keys = set() # Track display keys seen

    for item in aggregated_concept_data:
        concept = item['concept']
        display_key = item['selection']

        # *** Primary Uniqueness Check ***
        # Skip if this exact concept OR its display key has already been added
        if concept in seen_concepts or display_key in seen_display_keys:
            continue

        # Add the current selection
        top_n_unique_selections.append({
            "selection": display_key,
            "probability": item['probability']
        })

        # Mark this concept AND its display key as seen
        seen_concepts.add(concept)
        seen_display_keys.add(display_key)

        # --- Determine and Mark the Opposite Concept/Display Key as Seen ---
        opposite_concept = None
        opposite_display_key = None
        concept_type = concept[0]
        try:
            # Find opposite concept tuple
            if concept_type == 'btts':
                opposite_concept = ('btts', not concept[1])
            elif concept_type == 'total_goals':
                opposite_concept = ('total_goals', concept[1], not concept[2])
            elif concept_type == 'result_ou':
                 opposite_concept = ('result_ou', concept[1], concept[2], not concept[3])
            elif concept_type == 'dc_ou':
                 opposite_concept = ('dc_ou', concept[1], concept[2], not concept[3])
            elif concept_type == 'btts_ou':
                 opposite_concept = ('btts_ou', concept[1], concept[2], not concept[3])

            # If an opposite concept exists, mark it and its display key as seen
            if opposite_concept is not None:
                opposite_display_key = reconstruct_display_key(opposite_concept)
                if opposite_display_key != "Unknown Concept" and opposite_display_key != "Error Key":
                    seen_concepts.add(opposite_concept)
                    seen_display_keys.add(opposite_display_key)

        except IndexError as e:
             logger.error(f"Index error checking opposite concept for: {concept} - {e}")
        # --- End of Handling Opposite ---

        # Stop if we have enough unique selections
        if len(top_n_unique_selections) >= top_n:
            break

    logger.info(f"Found {len(top_n_unique_selections)} unique combined selections.")
    if len(top_n_unique_selections) < top_n:
         logger.warning(f"Could only find {len(top_n_unique_selections)} unique combined selections (less than requested {top_n}).")

    return top_n_unique_selections

# --- Processing Function for a Single File ---
def process_fixture_json(json_file_path: str) -> Optional[Dict[str, Any]]:
    """Loads, processes, predicts, ranks, and plots for a single fixture JSON file."""
    logger.info(f"--- Processing Fixture File: {os.path.basename(json_file_path)} ---")
    try:
        with open(json_file_path, 'r') as f:
            fixture_data = json.load(f)
        fixture_id = fixture_data.get("fixture_id", "N/A")
        home_team_name = safe_get(fixture_data, ['raw_data', 'home', 'basic_info', 'name'], 'Home')
        away_team_name = safe_get(fixture_data, ['raw_data', 'away', 'basic_info', 'name'], 'Away')
        logger.info(f"Loaded Fixture ID: {fixture_id} ({home_team_name} vs {away_team_name})")
    except (json.JSONDecodeError, IOError, KeyError) as e:
        logger.error(f"Error reading/parsing JSON file {json_file_path}: {e}")
        return None

    results = {
        "fixture_id": fixture_id,
        "home_team": home_team_name,
        "away_team": away_team_name,
        "file_path": json_file_path,
        # --- Add raw data for downstream use ---
        "fixture_meta": safe_get(fixture_data, ['fixture_meta'], {}), # Include fixture metadata (date, time, etc.)
        "raw_data": safe_get(fixture_data, ['raw_data'], {}),       # Include raw team/match data (logos, detailed stats)
        # --- Existing keys ---
        "mc_probs": None,
        "mc_score_probs": None, # Make sure MC function returns this
        "lambdas_original": (None, None),
        # "top_n_mc_selections": None, # Removed - replaced by combined
        "lambdas_weighted": (None, None),
        "analytical_poisson_probs": None,
        "elo_probs": None,
        "bivariate_poisson_probs": None,
        "top_n_combined_selections": None, # Added
    }

    # Original Monte Carlo Simulation
    lambdas_orig = calculate_strength_adjusted_lambdas(fixture_data)
    results["lambdas_original"] = lambdas_orig

    mc_scenario_results_dict, mc_score_results_dict = None, None
    if lambdas_orig[0] is not None and lambdas_orig[1] is not None:
         mc_scenario_results_dict, mc_score_results_dict = run_monte_carlo_simulation(
             lambdas_orig[0], lambdas_orig[1], max_score_plot=MC_MAX_SCORE_PLOT
         )
         if mc_scenario_results_dict is not None:
              results["mc_probs"] = mc_scenario_results_dict
              # --- REMOVE Old Top N MC Selections Calculation ---
              # Old code for top_n_mc_selections removed from here
         else:
              logger.error(f"Failed Monte Carlo scenario simulation for {fixture_id}")
              results["mc_probs"] = None # Ensure it's None

         if mc_score_results_dict is not None:
              results["mc_score_probs"] = mc_score_results_dict # Store MC scores
         else:
              logger.error(f"Failed Monte Carlo scoreline simulation for {fixture_id}")

    else:
         logger.error(f"Failed to calculate original strength-adjusted lambdas for MC simulation for {fixture_id}")
         # results["top_n_mc_selections"] = [] # Ensure it's an empty list - remove

    # Weighted Lambdas
    lambdas_w = calculate_weighted_strength_lambdas(fixture_data)
    results["lambdas_weighted"] = lambdas_w

    # Analytical Poisson
    if lambdas_orig[0] is not None and lambdas_orig[1] is not None:
        results["analytical_poisson_probs"] = calculate_analytical_poisson_probs(
            lambdas_orig[0], lambdas_orig[1], max_goals=MC_MAX_SCORE_PLOT
        )
    else:
        logger.warning("Skipping analytical Poisson due to missing original lambdas.")

    # Elo Probabilities
    home_elo = safe_get(fixture_data, ['engineered_features', 'home', 'elo_rating'])
    away_elo = safe_get(fixture_data, ['engineered_features', 'away', 'elo_rating'])
    results["elo_probs"] = calculate_elo_probabilities(home_elo, away_elo)

    # Bivariate Poisson
    if lambdas_orig[0] is not None and lambdas_orig[1] is not None:
        lambda3 = get_league_goal_covariance_lambda3(fixture_data)
        # Check if lambda3 is valid relative to lambdas_orig
        valid_lambda3 = False
        try:
             if lambda3 >= 0 and lambda3 <= lambdas_orig[0] and lambda3 <= lambdas_orig[1]:
                 valid_lambda3 = True
        except TypeError: # Handle None in lambdas_orig gracefully
             valid_lambda3 = False

        if valid_lambda3:
            results["bivariate_poisson_probs"] = calculate_bivariate_poisson_probs(
                lambdas_orig[0], lambdas_orig[1], lambda3, max_goals=MC_MAX_SCORE_PLOT
            )
        else:
             l0_disp = f"{lambdas_orig[0]:.3f}" if lambdas_orig[0] is not None else "N/A"
             l1_disp = f"{lambdas_orig[1]:.3f}" if lambdas_orig[1] is not None else "N/A"
             l3_disp = f"{lambda3:.3f}" if lambda3 is not None else "N/A"
             logger.warning(f"Skipping Bivariate Poisson: Invalid lambda combination (L0={l0_disp}, L1={l1_disp}, L3={l3_disp}).")
    else:
        logger.warning("Skipping Bivariate Poisson due to missing original lambdas.")

    # --- Calculate Combined Top Selections ---
    results["top_n_combined_selections"] = calculate_combined_top_selections(
        results["mc_probs"],
        results["analytical_poisson_probs"],
        results["bivariate_poisson_probs"],
        top_n=TOP_N_SCENARIOS
    )
    # Add mc_score_probs to mc_probs temporarily for combined calculation if needed
    # Let's adjust calculate_combined_top_selections to handle nested score dicts

    # Plotting
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if results:
        try:
            # Ensure plotting function can handle the potentially missing `top_n_mc_selections` if it expects it
            # Or update it to use `top_n_combined_selections` if relevant for the plot
            create_combined_fixture_plot(results, PLOTS_DIR, max_goals_matrix=MC_MAX_SCORE_PLOT)
        except Exception as plot_err:
            logger.error(f"Error during plotting for fixture {fixture_id}: {plot_err}", exc_info=True)

    logger.info(f"--- Finished Processing: {os.path.basename(json_file_path)} ---")
    return results

# --- Main Execution Logic ---
if __name__ == '__main__':
    logger.info("--- Starting Batch Fixture Processing ---")

    # Create the input directory if it doesn't exist
    if not os.path.isdir(UNIFIED_DATA_DIR):
        logger.info(f"Creating input directory: {UNIFIED_DATA_DIR}")
        try:
            os.makedirs(UNIFIED_DATA_DIR, exist_ok=True)
            logger.info(f"Successfully created input directory: {UNIFIED_DATA_DIR}")
        except Exception as e:
            logger.error(f"Failed to create input directory {UNIFIED_DATA_DIR}: {e}")
            sys.exit(1) # Exit if we can't create input dir

    # Create the output directory if it doesn't exist
    if not os.path.isdir(OUTPUT_DIR):
        logger.info(f"Creating output directory: {OUTPUT_DIR}")
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            logger.info(f"Successfully created output directory: {OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"Failed to create output directory {OUTPUT_DIR}: {e}")
            # sys.exit(1) # Decide if you want to exit here too

    # --- Find JSON Files ---
    json_files = glob.glob(os.path.join(UNIFIED_DATA_DIR, "*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in directory: {UNIFIED_DATA_DIR}")
        sys.exit(0)

    logger.info(f"Found {len(json_files)} JSON files to process.")

    # --- Process Each File ---
    all_results = []
    for json_file_path in json_files:
        try:
            fixture_results = process_fixture_json(json_file_path)
            if fixture_results:
                all_results.append(fixture_results)
        except Exception as e:
            logger.error(f"Unexpected error processing file {json_file_path}: {e}", exc_info=True)

    # --- Display Aggregated Results Summary (Updated Structure) ---
    logger.info("--- Batch Processing Summary ---")
    logger.info(f"Successfully processed {len(all_results)} out of {len(json_files)} files.")

    print("\n" + "="*70)
    print("                     BATCH PREDICTION RESULTS SUMMARY")
    print("="*70 + "\n")

    for result in all_results:
        print(f"--- Fixture: {result['fixture_id']} ({result['home_team']} vs {result['away_team']}) ---")
        print(f"File: {os.path.basename(result['file_path'])}")

        # --- 1. Monte Carlo Output ---
        lambda_h_orig, lambda_a_orig = result.get('lambdas_original', (None, None))
        lambda_h_str_orig = f"{lambda_h_orig:.3f}" if lambda_h_orig is not None else "N/A"
        lambda_a_str_orig = f"{lambda_a_orig:.3f}" if lambda_a_orig is not None else "N/A"
        print(f" Monte Carlo ({MONTE_CARLO_SIMULATIONS} sims, Lambdas Orig H/A: {lambda_h_str_orig}/{lambda_a_str_orig}):")
        if result.get('mc_probs'):
             mc_probs = result['mc_probs'] # Use get for safety
             mc_prob_h = mc_probs.get('prob_H', 0.0)
             mc_prob_d = mc_probs.get('prob_D', 0.0)
             mc_prob_a = mc_probs.get('prob_A', 0.0)
             o25_prob = mc_probs.get('prob_O2.5', 0.0)
             btts_prob = mc_probs.get('prob_BTTS Yes', 0.0)
             dc_1x_prob = mc_probs.get('prob_1X', 0.0)
             dc_x2_prob = mc_probs.get('prob_X2', 0.0)
             print(f"  P(1X2)   : {mc_prob_h:.3f} / {mc_prob_d:.3f} / {mc_prob_a:.3f}")
             print(f"  P(DC 1X/X2): {dc_1x_prob:.3f} / {dc_x2_prob:.3f}")
             print(f"  P(O2.5)  : {o25_prob:.3f}")
             print(f"  P(BTTS=Y): {btts_prob:.3f}")
             # Display top scorelines from MC? Optional
             # if result.get('mc_score_probs'):
             #     top_scores = sorted(result['mc_score_probs'].items(), key=lambda x: x[1], reverse=True)[:3]
             #     top_scores_str = ", ".join([f"{s.replace('score_','')}: {p:.3f}" for s, p in top_scores])
             #     print(f"  Top Scores: {top_scores_str}")
        else:
             print("  MC Results: Failed/Skipped")

        # --- 2. Analytical Poisson ---
        print(f" Analytical Poisson:")
        if result.get('analytical_poisson_probs'):
            ap_probs = result['analytical_poisson_probs']
            prob_h = ap_probs.get('poisson_prob_H', 0.0)
            prob_d = ap_probs.get('poisson_prob_D', 0.0)
            prob_a = ap_probs.get('poisson_prob_A', 0.0)
            prob_1x = ap_probs.get('poisson_prob_1X', 0.0)
            prob_x2 = ap_probs.get('poisson_prob_X2', 0.0)
            prob_btts_y = ap_probs.get('poisson_prob_BTTS_Yes', 0.0)
            prob_o25 = ap_probs.get('poisson_prob_O2.5', 0.0)
            print(f"  P(1X2)   : {prob_h:.3f} / {prob_d:.3f} / {prob_a:.3f}")
            print(f"  P(DC 1X/X2): {prob_1x:.3f} / {prob_x2:.3f}")
            print(f"  P(BTTS=Y): {prob_btts_y:.3f}")
            print(f"  P(O2.5)  : {prob_o25:.3f}")
        else:
            print("  Analytical Poisson: Failed/Skipped")

        # --- 3. Bivariate Poisson ---
        print(f" Bivariate Poisson:")
        if result.get('bivariate_poisson_probs'):
            bp_probs = result['bivariate_poisson_probs']
            prob_h = bp_probs.get('biv_poisson_prob_H', 0.0)
            prob_d = bp_probs.get('biv_poisson_prob_D', 0.0)
            prob_a = bp_probs.get('biv_poisson_prob_A', 0.0)
            prob_1x = bp_probs.get('biv_poisson_prob_1X', 0.0)
            prob_x2 = bp_probs.get('biv_poisson_prob_X2', 0.0)
            prob_btts_y = bp_probs.get('biv_poisson_prob_BTTS_Yes', 0.0)
            prob_o25 = bp_probs.get('biv_poisson_prob_O2.5', 0.0)
            print(f"  P(1X2)   : {prob_h:.3f} / {prob_d:.3f} / {prob_a:.3f}")
            print(f"  P(DC 1X/X2): {prob_1x:.3f} / {prob_x2:.3f}")
            print(f"  P(BTTS=Y): {prob_btts_y:.3f}")
            print(f"  P(O2.5)  : {prob_o25:.3f}")
        else:
            print("  Bivariate Poisson: Failed/Skipped")

        # --- 4. Other Models/Info (Elo, Weighted Lambdas) ---
        # Elo Probabilities
        if result.get('elo_probs') and result['elo_probs'].get('elo_prob_H') is not None:
            elo_p = result['elo_probs']
            prob_h = elo_p.get('elo_prob_H', 0.0)
            prob_d = elo_p.get('elo_prob_D', 0.0)
            prob_a = elo_p.get('elo_prob_A', 0.0)
            print(f" Elo Probs (H/D/A): {prob_h:.3f} / {prob_d:.3f} / {prob_a:.3f}")
        else:
            print(" Elo Probs: Failed/Skipped (Missing Elo?)")

        # Weighted Lambdas
        lambda_h_w, lambda_a_w = result.get('lambdas_weighted', (None, None))
        lambda_h_str_w = f"{lambda_h_w:.3f}" if lambda_h_w is not None else "N/A"
        lambda_a_str_w = f"{lambda_a_w:.3f}" if lambda_a_w is not None else "N/A"
        print(f" Weighted Lambdas (H/A): {lambda_h_str_w}/{lambda_a_str_w}")

        # --- 5. Top Combined Selections ---
        print(f" Top {TOP_N_SCENARIOS} Combined Selections (Avg Prob):")
        if result.get('top_n_combined_selections'):
            top_selections = result['top_n_combined_selections']
            if top_selections: # Check if the list is not empty
                for i, item in enumerate(top_selections):
                    # Ensure probability is formatted correctly, handle potential errors
                    try:
                        prob_str = f"{item['probability']:.4f}"
                    except (TypeError, ValueError):
                        prob_str = "N/A"
                    print(f"   {i+1: >2}. {item.get('selection', 'Error'):<30} | P = {prob_str}")
            else:
                 print(f"  Top {TOP_N_SCENARIOS} Combined Selections: No selections generated.")

        else:
            print(f"  Top {TOP_N_SCENARIOS} Combined Selections: Calculation failed or key missing.")


        print("-" * 70)

    # --- Save Results ---
    output_filename = os.path.join(OUTPUT_DIR, "batch_prediction_results.json")
    try:
        # Use a custom encoder to handle potential numpy types if necessary
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                # Handle tuples with None, converting None to None for JSON compatibility
                if isinstance(obj, tuple) and any(x is None for x in obj):
                    return [None if x is None else x for x in obj]
                # Handle NaN floats explicitly
                if isinstance(obj, float) and math.isnan(obj):
                    return None # Represent NaN as null in JSON
                return super(NpEncoder, self).default(obj)

        with open(output_filename, 'w') as f:
            json.dump(all_results, f, indent=4, cls=NpEncoder)
        logger.info(f"Saved detailed results to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save results to file: {e}", exc_info=True)


    logger.info("--- Script Finished ---")