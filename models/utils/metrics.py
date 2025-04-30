# utils/metrics.py

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

def accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculates classification accuracy."""
    return accuracy_score(y_true, y_pred)

def multi_logloss(y_true: pd.Series, y_prob: pd.DataFrame, labels: Optional[List] = None) -> float:
    """
    Calculates multi-class logarithmic loss.

    Args:
        y_true (pd.Series): True labels (e.g., 'H', 'D', 'A').
        y_prob (pd.DataFrame): DataFrame with predicted probabilities.
                               Columns should be named 'prob_H', 'prob_D', 'prob_A'
                               (or match the order in `labels`).
        labels (Optional[List]): Ordered list of class labels corresponding to y_prob columns.
                                 If None, inferred from y_true or y_prob columns.

    Returns:
        float: The calculated log loss. Returns np.inf if invalid inputs.
    """
    try:
        if labels is None:
            # Try to infer from y_prob columns if named like 'prob_X'
            labels = [col.split('_')[-1] for col in y_prob.columns if col.startswith('prob_')]
            if not labels or len(labels) != y_prob.shape[1]:
                 # Fallback to unique values in y_true
                 labels = sorted(y_true.unique())

        # Ensure probability columns exist and match labels
        prob_cols = [f"prob_{label}" for label in labels]
        if not all(col in y_prob.columns for col in prob_cols):
             # Try matching without 'prob_' prefix if needed
             prob_cols = labels
             if not all(col in y_prob.columns for col in prob_cols):
                 raise ValueError(f"y_prob DataFrame missing required probability columns for labels: {labels}")

        # Clip probabilities to avoid log(0)
        eps = 1e-15
        y_prob_clipped = y_prob[prob_cols].clip(lower=eps, upper=1 - eps)

        return log_loss(y_true, y_prob_clipped, labels=labels)
    except Exception as e:
        print(f"Error calculating log loss: {e}")
        return np.inf


def binary_logloss(y_true: pd.Series, y_prob_positive: pd.Series) -> float:
    """
    Calculates binary logarithmic loss.

    Args:
        y_true (pd.Series): True binary labels (0 or 1).
        y_prob_positive (pd.Series): Predicted probability of the positive class (class 1).

    Returns:
        float: The calculated log loss.
    """
    try:
        # Clip probabilities
        eps = 1e-15
        y_prob_clipped = y_prob_positive.clip(lower=eps, upper=1 - eps)
        return log_loss(y_true, y_prob_clipped)
    except Exception as e:
        print(f"Error calculating binary log loss: {e}")
        return np.inf

def calculate_roi(results_df: pd.DataFrame,
                  prob_col_h: str = 'prob_H', prob_col_d: str = 'prob_D', prob_col_a: str = 'prob_A',
                  odds_col_h: str = 'B365H', odds_col_d: str = 'B365D', odds_col_a: str = 'B365A',
                  true_result_col: str = 'FTR',
                  bet_threshold: float = 0.05, # Bet if predicted prob > implied prob + threshold
                  stake: float = 1.0) -> Dict[str, float]:
    """
    Calculates simple Return on Investment (ROI) based on a value betting strategy.

    **Note:** This is a simplified ROI calculation for illustration. Real-world
    ROI depends heavily on accurate odds, stake management, and strategy.

    Args:
        results_df (pd.DataFrame): DataFrame containing true results, predicted probabilities,
                                   and market odds.
        prob_col_h/d/a (str): Column names for predicted probabilities.
        odds_col_h/d/a (str): Column names for market odds.
        true_result_col (str): Column name for the true Full Time Result ('H', 'D', 'A').
        bet_threshold (float): Minimum value edge required to place a bet
                               (predicted_prob / implied_prob - 1 >= threshold).
        stake (float): The amount staked on each qualifying bet.

    Returns:
        Dict[str, float]: Dictionary containing 'total_staked', 'total_returned', 'roi'.
    """
    print("Calculating simple ROI...")
    df = results_df.copy()
    required_cols = [prob_col_h, prob_col_d, prob_col_a, odds_col_h, odds_col_d, odds_col_a, true_result_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"results_df missing one or more required columns for ROI: {required_cols}")

    total_staked = 0.0
    total_returned = 0.0
    bets_placed = 0

    for index, row in df.iterrows():
        placed_bet_this_match = False
        # Check Home bet value
        implied_h = 1.0 / row[odds_col_h] if row[odds_col_h] > 0 else 0
        if implied_h > 0 and (row[prob_col_h] / implied_h - 1) >= bet_threshold:
            total_staked += stake
            bets_placed += 1
            placed_bet_this_match = True
            if row[true_result_col] == 'H':
                total_returned += stake * row[odds_col_h]

        # Check Draw bet value (only if no Home bet placed)
        implied_d = 1.0 / row[odds_col_d] if row[odds_col_d] > 0 else 0
        if not placed_bet_this_match and implied_d > 0 and (row[prob_col_d] / implied_d - 1) >= bet_threshold:
            total_staked += stake
            bets_placed += 1
            placed_bet_this_match = True
            if row[true_result_col] == 'D':
                total_returned += stake * row[odds_col_d]

        # Check Away bet value (only if no Home or Draw bet placed)
        implied_a = 1.0 / row[odds_col_a] if row[odds_col_a] > 0 else 0
        if not placed_bet_this_match and implied_a > 0 and (row[prob_col_a] / implied_a - 1) >= bet_threshold:
            total_staked += stake
            bets_placed += 1
            # placed_bet_this_match = True # Not needed here
            if row[true_result_col] == 'A':
                total_returned += stake * row[odds_col_a]

    roi = ((total_returned - total_staked) / total_staked) * 100 if total_staked > 0 else 0.0
    print(f"ROI Calculation: {bets_placed} bets placed, Staked={total_staked:.2f}, Returned={total_returned:.2f}")
    return {'total_staked': total_staked, 'total_returned': total_returned, 'roi': roi, 'bets_placed': bets_placed}


def plot_calibration_curve(y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10, strategy: str = 'uniform'):
    """
    Plots a calibration curve for binary classification probabilities.

    Args:
        y_true (pd.Series): True binary labels (0 or 1).
        y_prob (pd.Series): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to group probabilities.
        strategy (str): Strategy used to define the widths of the bins ('uniform' or 'quantile').
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model Calibration")
    plt.xlabel("Mean Predicted Probability (Bin)")
    plt.ylabel("Fraction of Positives (Bin)")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example Usage (Optional)
if __name__ == '__main__':
    # --- Classification Example ---
    y_true_class = pd.Series(['H', 'D', 'A', 'H', 'H', 'D'])
    y_pred_class = pd.Series(['H', 'H', 'A', 'D', 'H', 'D']) # Example predictions
    y_prob_class = pd.DataFrame({
        'prob_H': [0.6, 0.5, 0.1, 0.4, 0.8, 0.3],
        'prob_D': [0.3, 0.3, 0.2, 0.4, 0.1, 0.5],
        'prob_A': [0.1, 0.2, 0.7, 0.2, 0.1, 0.2]
    })
    print("\n--- Classification Metrics ---")
    print(f"Accuracy: {accuracy(y_true_class, y_pred_class):.4f}")
    print(f"Log Loss: {multi_logloss(y_true_class, y_prob_class):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true_class, y_pred_class))

    # --- Binary Example ---
    y_true_bin = pd.Series([0, 1, 1, 0, 1, 0])
    y_prob_bin = pd.Series([0.2, 0.7, 0.9, 0.4, 0.6, 0.1]) # Prob of class 1
    print("\n--- Binary Metrics ---")
    print(f"Binary Log Loss: {binary_logloss(y_true_bin, y_prob_bin):.4f}")
    # plot_calibration_curve(y_true_bin, y_prob_bin) # Uncomment to plot

    # --- ROI Example ---
    roi_data = pd.DataFrame({
        'FTR': ['H', 'D', 'A', 'H'],
        'prob_H': [0.55, 0.30, 0.10, 0.60],
        'prob_D': [0.25, 0.45, 0.20, 0.25],
        'prob_A': [0.20, 0.25, 0.70, 0.15],
        'B365H': [1.9, 3.5, 8.0, 1.7],
        'B365D': [3.4, 2.5, 4.0, 3.6],
        'B365A': [4.5, 3.8, 1.5, 5.5]
    })
    print("\n--- ROI Calculation Example ---")
    roi_results = calculate_roi(roi_data, bet_threshold=0.05, stake=10)
    print(f"ROI Results: {roi_results}")