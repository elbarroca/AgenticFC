# ml_models/random_forest_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any
import warnings

from models.base_model import BaseModel
from models.utils.features import BaseFeatureConfig
# Import the standardized probability calculation function
from models.ml_models.poisson_model import calculate_poisson_outcome_probs, _calculate_dual_conditions

class RandomForestModel(BaseModel):
    """
    Random Forest Regressor model to predict expected goals (lambdas) based on features,
    and then derive outcome probabilities using the calculate_poisson_outcome_probs function.
    Inherits scaling and save/load logic from BaseModel.
    """
    def __init__(self, model_params: Dict[str, Any], feature_config: BaseFeatureConfig, apply_scaling: bool = True):
        """Initializes the RandomForestModel."""
        # Pass model_params, feature_config, and apply_scaling to the BaseModel constructor
        super().__init__(model_params, feature_config=feature_config, apply_scaling=apply_scaling)

        # Default random state for reproducibility if not provided
        if 'random_state' not in self.params:
            self.params['random_state'] = 42 # Default state for home model

        # Instantiate the core model objects
        # Use a different random state for the away model for slight diversity
        away_params = self.params.copy()
        away_params['random_state'] = self.params.get('random_state', 42) + 1

        # Ensure n_jobs is set for performance, default to -1 (use all cores)
        if 'n_jobs' not in self.params:
            self.params['n_jobs'] = -1
            away_params['n_jobs'] = -1

        self._model_home = RandomForestRegressor(**self.params)
        self._model_away = RandomForestRegressor(**away_params)

        # Store models in a dictionary compatible with base save/load
        self._model = {'home': self._model_home, 'away': self._model_away}
        print(f"Initialized RandomForestModel with params: {self.params} (Home) and {away_params} (Away)")

    def _fit_model(self, X_scaled: pd.DataFrame, y: pd.DataFrame):
        """Fits two RandomForest Regressors using SCALED features."""
        # --- Assertions specific to goal targets ---
        target_hg = self.feature_config.target_home_goals
        target_ag = self.feature_config.target_away_goals
        assert target_hg in y.columns, f"Target column '{target_hg}' not found in y."
        assert target_ag in y.columns, f"Target column '{target_ag}' not found in y."
        assert pd.api.types.is_numeric_dtype(y[target_hg]), f"Target '{target_hg}' is not numeric."
        assert pd.api.types.is_numeric_dtype(y[target_ag]), f"Target '{target_ag}' is not numeric."
        assert not y[[target_hg, target_ag]].isnull().any().any(), f"Target columns contain NaN values."
        # RF doesn't strictly require non-negative targets, but our context (goals) does.
        assert np.all(y[target_hg] >= 0), f"Target '{target_hg}' contains negative values."
        assert np.all(y[target_ag] >= 0), f"Target '{target_ag}' contains negative values."

        assert X_scaled.columns.tolist() == self.features_in_, "Scaled features columns mismatch features_in_"

        print(f"Fitting RandomForest Regressor for Home Goals ({target_hg}) using scaled features...")
        self._model['home'].fit(X_scaled, y[target_hg])

        print(f"Fitting RandomForest Regressor for Away Goals ({target_ag}) using scaled features...")
        self._model['away'].fit(X_scaled, y[target_ag])

        print("RandomForest models fitted successfully.")

    def _predict_proba_model(self, X_scaled: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predicts expected goals (lambdas) using SCALED features and calculates probabilities."""
        assert X_scaled.columns.tolist() == self.features_in_, "Scaled prediction features columns mismatch features_in_"

        print("Predicting expected goals (lambdas) using RandomForest models...")
        lambda_home = self._model['home'].predict(X_scaled)
        lambda_away = self._model['away'].predict(X_scaled)

        lambda_home = np.maximum(lambda_home, 1e-9)
        lambda_away = np.maximum(lambda_away, 1e-9)

        print("Calculating outcome probabilities from RF-predicted lambdas using calculate_poisson_outcome_probs...")
        outcome_probs = calculate_poisson_outcome_probs(lambda_home, lambda_away)

        outcome_probs['expected_HG'] = lambda_home
        outcome_probs['expected_AG'] = lambda_away
        
        # --- Prefix all keys with model name ---
        prefixed_probs = {f"random_forest_{key}": value for key, value in outcome_probs.items()}

        # --- Standard Assertions on Output Dictionary (checking prefixed_probs) ---
        num_rows = X_scaled.shape[0]
        # Update expected_single_keys if calculate_poisson_outcome_probs changes its single key names/set
        # For now, assume it's consistent with the previous definition for a basic check
        expected_core_keys_subset = { # Check a subset of the expected prefixed keys
            'random_forest_prob_H', 'random_forest_prob_D', 'random_forest_prob_A', 
            'random_forest_prob_O25', 'random_forest_prob_U25', 
            'random_forest_prob_BTTS_Y', 'random_forest_prob_BTTS_N',
            'random_forest_expected_HG', 'random_forest_expected_AG',
            'random_forest_prob_H_and_O05', # Example of a new dual
            'random_forest_prob_1X_and_U45', # Example of a new dual
            'random_forest_prob_O15_and_BTTS_Y' # Example of a new dual
        }
        present_keys = set(prefixed_probs.keys())
        missing_keys = expected_core_keys_subset - present_keys
        assert not missing_keys, \
            f"Output keys missing expected prefixed singles/duals.\nMissing: {missing_keys}"
        
        # Check presence of a broader range of dual outcome key patterns
        assert any(k.startswith('random_forest_prob_') and '_and_O05' in k for k in present_keys), "Missing O0.5 duals"
        assert any(k.startswith('random_forest_prob_') and '_and_U45' in k for k in present_keys), "Missing U4.5 duals"
        assert any(k.startswith('random_forest_prob_') and '_and_BTTS_Y' in k for k in present_keys), "Missing BTTS_Y duals"


        for key, arr in prefixed_probs.items():
            assert isinstance(arr, np.ndarray), f"Output '{key}' is not a numpy array."
            assert arr.shape == (num_rows,), f"Output '{key}' has incorrect shape {arr.shape}, expected ({num_rows},)."
            if key.startswith("random_forest_prob_"):
                 assert np.all((arr >= 0) & (arr <= 1)), f"Probabilities in '{key}' are outside [0, 1]."
            elif key.startswith("random_forest_expected_"):
                 assert np.all(arr >= 0), f"Expected goals in '{key}' are negative."
        
        # Check basic probability consistency (checking prefixed_probs)
        # Ensure you access the correct prefixed keys for these sums
        assert np.allclose(prefixed_probs['random_forest_prob_H'] + prefixed_probs['random_forest_prob_D'] + prefixed_probs['random_forest_prob_A'], 1.0, atol=1e-5) # Adjusted tolerance
        assert np.allclose(prefixed_probs['random_forest_prob_O25'] + prefixed_probs['random_forest_prob_U25'], 1.0, atol=1e-5)
        assert np.allclose(prefixed_probs['random_forest_prob_BTTS_Y'] + prefixed_probs['random_forest_prob_BTTS_N'], 1.0, atol=1e-5)

        print("RandomForest-based probabilities calculated successfully.")
        return prefixed_probs

    # Inherit fit, predict_proba, save, load from BaseModel
    # No need to override them here unless adding RF-specific logic *outside* scaling/core prediction