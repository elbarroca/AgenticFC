# models/markov_model.py
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
import joblib
from typing import Dict, List, Optional, Literal
from sklearn.exceptions import NotFittedError

from .base_model import BaseModel # Assuming base_model.py defines the interface

class MarkovModel(BaseModel):
    """
    Markov Chain model for analyzing football team form transitions (Win/Draw/Loss).

    Models the probability of transitioning from one match result state (W, D, L)
    to the next, based on aggregated historical sequences across teams.

    This model provides insights into form momentum and can generate features
    representing the expected outcome of the *next* match based *only* on the
    result of the *previous* match.
    """

    def __init__(self):
        """Initializes the MarkovModel for form transitions."""
        self.states: List[str] = ['W', 'D', 'L'] # Define the possible states (Win, Draw, Loss)
        self.state_to_idx: Dict[str, int] = {state: i for i, state in enumerate(self.states)}
        self.idx_to_state: Dict[int, str] = {i: state for state, i in self.state_to_idx.items()}
        self.num_states: int = len(self.states)

        self.transition_counts: Optional[np.ndarray] = None
        self.transition_matrix: Optional[np.ndarray] = None # Stores P(next_state | current_state)
        self.is_fitted: bool = False

    def _get_result_for_team(self, row: pd.Series, team_name: str) -> Optional[str]:
        """Determines if a team Won, Drew, or Lost a specific match."""
        if row['HomeTeam'] == team_name:
            if row['FTR'] == 'H': return 'W'
            if row['FTR'] == 'D': return 'D'
            if row['FTR'] == 'A': return 'L'
        elif row['AwayTeam'] == team_name:
            if row['FTR'] == 'A': return 'W'
            if row['FTR'] == 'D': return 'D'
            if row['FTR'] == 'H': return 'L'
        return None # Should not happen if team is in HomeTeam or AwayTeam

    def fit(self, history_df: pd.DataFrame):
        """
        Builds the state transition matrix from historical match results (FTR).

        Analyzes sequences of results for each team to count transitions
        between W, D, L states across the entire dataset.

        Args:
            history_df (pd.DataFrame): DataFrame containing historical match results.
                                       Required columns: 'HomeTeam', 'AwayTeam', 'FTR'.
                                       Should ideally be sorted by date if available,
                                       though the model aggregates transitions regardless of order within fit.
                                       'Date' column is recommended for clarity but not strictly used here.
        """
        print("Fitting MarkovModel for form transitions...")
        required_cols = ['HomeTeam', 'AwayTeam', 'FTR']
        if not all(col in history_df.columns for col in required_cols):
            raise ValueError(f"history_df must contain columns: {required_cols}")

        # Consider sorting by date if not already done, although aggregation smooths this
        # history_df = history_df.sort_values(by='Date') # If 'Date' column exists

        # Initialize transition counts matrix
        self.transition_counts = np.zeros((self.num_states, self.num_states), dtype=int)

        teams = pd.concat([history_df['HomeTeam'], history_df['AwayTeam']]).unique()
        print(f"Processing sequences for {len(teams)} teams...")

        total_transitions_counted = 0
        for team in teams:
            # Get all matches involving the team
            team_matches = history_df[(history_df['HomeTeam'] == team) | (history_df['AwayTeam'] == team)].copy()
            # Determine the result for the current team in each match
            team_matches['TeamResult'] = team_matches.apply(lambda row: self._get_result_for_team(row, team), axis=1)

            # Create the sequence of results
            result_sequence = team_matches['TeamResult'].dropna().tolist()

            # Count transitions within this team's sequence
            for i in range(len(result_sequence) - 1):
                from_state = result_sequence[i]
                to_state = result_sequence[i+1]

                if from_state in self.state_to_idx and to_state in self.state_to_idx:
                    from_idx = self.state_to_idx[from_state]
                    to_idx = self.state_to_idx[to_state]
                    self.transition_counts[from_idx, to_idx] += 1
                    total_transitions_counted += 1

        print(f"Total transitions counted across all teams: {total_transitions_counted}")
        if total_transitions_counted == 0:
             warnings.warn("No transitions were counted. Transition matrix will be empty or uniform.")
             # Handle case with no transitions - perhaps uniform probability?
             self.transition_matrix = np.full((self.num_states, self.num_states), 1.0 / self.num_states)
             self.is_fitted = True
             return

        # --- Calculate Transition Probabilities ---
        self.transition_matrix = np.zeros((self.num_states, self.num_states), dtype=float)
        row_sums = self.transition_counts.sum(axis=1)

        # Avoid division by zero for states with no outgoing transitions
        valid_rows = row_sums > 0
        self.transition_matrix[valid_rows, :] = self.transition_counts[valid_rows, :] / row_sums[valid_rows, np.newaxis]

        # Handle states with no observed outgoing transitions (e.g., a team always wins after winning)
        # Option 1: Leave probabilities as 0 (implies absorbing state or insufficient data)
        # Option 2: Assign uniform probability to next states (assumes equal chance if unseen)
        # Option 3: Assign probability 1 to staying in the same state (self-loop)
        zero_sum_rows = ~valid_rows
        if np.any(zero_sum_rows):
            print(f"Warning: States {[self.idx_to_state[i] for i, zero in enumerate(zero_sum_rows) if zero]} have no observed outgoing transitions.")
            # Applying Option 2: Uniform probability for next state from these rows
            self.transition_matrix[zero_sum_rows, :] = 1.0 / self.num_states

        self.is_fitted = True
        print("Fitting complete. Transition matrix built.")
        print("Transition Matrix (P(Next State | Current State)):")
        print(pd.DataFrame(self.transition_matrix, index=self.states, columns=self.states).round(4))


    def predict_next_state_probabilities(self, current_state: str) -> Dict[str, float]:
        """
        Predicts the probability distribution of the next state given the current state.

        Args:
            current_state (str): The current state ('W', 'D', or 'L').

        Returns:
            Dict[str, float]: A dictionary mapping each possible next state ('W', 'D', 'L')
                              to its probability.

        Raises:
            NotFittedError: If the model has not been fitted.
            ValueError: If current_state is not a valid state ('W', 'D', 'L').
        """
        if not self.is_fitted or self.transition_matrix is None:
            raise NotFittedError("MarkovModel has not been fitted yet. Call 'fit' first.")

        if current_state not in self.state_to_idx:
            raise ValueError(f"Invalid current_state '{current_state}'. Valid states are: {self.states}")

        current_idx = self.state_to_idx[current_state]
        probabilities = self.transition_matrix[current_idx, :]

        # Return as a dictionary mapping state names to probabilities
        return {self.idx_to_state[i]: prob for i, prob in enumerate(probabilities)}

    def predict(self, data: Dict[str, str]) -> Dict[str, float]:
        """
        Predicts the probability distribution for the next state based on the current state.

        This method adheres to the BaseModel interface but expects 'current_state'
        in the input dictionary instead of typical features X.

        Args:
            data (Dict[str, str]): A dictionary containing the current state.
                                   Expected key: 'current_state' (value: 'W', 'D', or 'L').

        Returns:
            Dict[str, float]: Probability distribution over the next possible states ('W', 'D', 'L').
        """
        if 'current_state' not in data:
            raise KeyError("Input data dictionary must contain 'current_state' key (e.g., {'current_state': 'W'}).")

        current_state = data['current_state']
        return self.predict_next_state_probabilities(current_state)

    def simulate_sequence(self, start_state: str, n_steps: int) -> List[str]:
        """
        Simulates a sequence of states starting from a given state.

        Args:
            start_state (str): The initial state ('W', 'D', or 'L').
            n_steps (int): The number of steps (transitions) to simulate.

        Returns:
            List[str]: The simulated sequence of states, including the start state.

        Raises:
            NotFittedError: If the model has not been fitted.
            ValueError: If start_state is invalid or n_steps is non-positive.
        """
        if not self.is_fitted or self.transition_matrix is None:
            raise NotFittedError("MarkovModel has not been fitted yet.")
        if start_state not in self.state_to_idx:
            raise ValueError(f"Invalid start_state '{start_state}'. Valid states are: {self.states}")
        if not isinstance(n_steps, int) or n_steps <= 0:
             raise ValueError("n_steps must be a positive integer.")

        sequence = [start_state]
        current_state = start_state
        rng = np.random.default_rng() # Use default RNG for simulation

        for _ in range(n_steps):
            current_idx = self.state_to_idx[current_state]
            next_state_probs = self.transition_matrix[current_idx, :]

            # Choose the next state based on the probabilities
            next_state_idx = rng.choice(self.num_states, p=next_state_probs)
            next_state = self.idx_to_state[next_state_idx]
            sequence.append(next_state)
            current_state = next_state

        return sequence

    def save(self, filepath: str):
        """Saves the fitted model state (transition matrix and state mappings)."""
        if not self.is_fitted:
            raise NotFittedError("Cannot save an unfitted model.")
        print(f"Saving Markov model state to {filepath}...")
        state = {
            'transition_matrix': self.transition_matrix,
            'states': self.states,
            'state_to_idx': self.state_to_idx,
            'idx_to_state': self.idx_to_state,
            # Optional: save transition_counts for inspection
            'transition_counts': self.transition_counts
        }
        joblib.dump(state, filepath)
        print("Model state saved successfully.")

    @classmethod
    def load(cls, filepath: str):
        """Loads a fitted model state."""
        print(f"Loading Markov model state from {filepath}...")
        state = joblib.load(filepath)
        # Create instance
        model_instance = cls()
        # Load the fitted attributes
        model_instance.transition_matrix = state['transition_matrix']
        model_instance.states = state['states']
        model_instance.state_to_idx = state['state_to_idx']
        model_instance.idx_to_state = state['idx_to_state']
        model_instance.num_states = len(model_instance.states)
        model_instance.transition_counts = state.get('transition_counts') # Load if saved
        model_instance.is_fitted = True
        print("Model state loaded successfully.")
        return model_instance

# Example Usage
if __name__ == '__main__':
    # --- 1. Create Dummy Historical Data ---
    print("\n--- MarkovModel Example (Form Transitions) ---")
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-08', '2023-01-08', '2023-01-15', '2023-01-15', '2023-01-22', '2023-01-22']),
        'HomeTeam': ['Team A', 'Team C', 'Team B', 'Team D', 'Team A', 'Team C', 'Team B', 'Team D'],
        'AwayTeam': ['Team B', 'Team D', 'Team A', 'Team C', 'Team D', 'Team B', 'Team C', 'Team A'],
        'FTHG': [1, 0, 2, 1, 3, 0, 1, 1],
        'FTAG': [1, 0, 2, 1, 1, 0, 1, 1],
        'FTR': ['D', 'D', 'D', 'D', 'H', 'D', 'D', 'D'] # Example results (A=Win, D=Draw, H=Loss for Away)
    }
    # Correcting FTR based on scores for clarity
    data['FTR'] = np.select(
        [data['FTHG'] > data['FTAG'], data['FTHG'] < data['FTAG']],
        ['H', 'A'], default='D'
    )
    history = pd.DataFrame(data).sort_values('Date')
    print("Dummy Historical Data:")
    print(history)

    # --- 2. Fit the Model ---
    markov_model = MarkovModel()
    markov_model.fit(history)

    # --- 3. Predict Next State Probabilities ---
    print("\nPredicting next state probabilities:")
    last_result = 'W' # Assume the team won their last match
    try:
        next_probs = markov_model.predict({'current_state': last_result})
        print(f"Probabilities after state '{last_result}':")
        for state, prob in next_probs.items():
             print(f"  P(Next={state} | Current={last_result}): {prob:.4f}")
    except (NotFittedError, KeyError, ValueError) as e:
        print(f"Error during prediction: {e}")

    last_result = 'D' # Assume the team drew their last match
    try:
        next_probs = markov_model.predict({'current_state': last_result})
        print(f"\nProbabilities after state '{last_result}':")
        for state, prob in next_probs.items():
             print(f"  P(Next={state} | Current={last_result}): {prob:.4f}")
    except (NotFittedError, KeyError, ValueError) as e:
        print(f"Error during prediction: {e}")

    # --- 4. Simulate a Sequence ---
    print("\nSimulating a sequence of 5 results starting from 'D':")
    try:
        simulated_seq = markov_model.simulate_sequence(start_state='D', n_steps=5)
        print(f"  Simulated sequence: {simulated_seq}")
    except (NotFittedError, ValueError) as e:
         print(f"Error during simulation: {e}")

    # --- 5. Save and Load ---
    model_path = "temp_markov_model_form.joblib"
    markov_model.save(model_path)
    loaded_model = MarkovModel.load(model_path)

    # Verify loaded model prediction
    print("\nVerifying loaded model...")
    try:
        loaded_probs = loaded_model.predict({'current_state': 'D'})
        original_probs = markov_model.predict({'current_state': 'D'})
        assert all(np.isclose(loaded_probs[s], original_probs[s]) for s in markov_model.states)
        print("Save/Load test passed.")
    except Exception as e:
        print(f"Error during loaded model verification: {e}")

    # Clean up
    import os
    if os.path.exists(model_path): os.remove(model_path)