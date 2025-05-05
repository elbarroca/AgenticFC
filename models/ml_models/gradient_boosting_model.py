# ml_models/gradient_boosting_model.py
import pandas as pd
import numpy as np
import lightgbm as lgb # Use LightGBM
from typing import Dict, Any
import warnings

from models.base_model import BaseModel
from models.utils.features import BaseFeatureConfig
# Import the standardized probability calculation function
from models.utils.poisson_model import calculate_poisson_outcome_probs, _calculate_dual_conditions

class GradientBoostingModel(BaseModel):
    """
    LightGBM Regressor model to predict expected goals (lambdas) based on features,
    using a Poisson objective. Derives outcome probabilities using the
    calculate_poisson_outcome_probs function.
    Inherits scaling and save/load logic from BaseModel.
    """
    def __init__(self, model_params: Dict[str, Any], feature_config: BaseFeatureConfig):
        """Initializes the GradientBoostingModel."""
        super().__init__(model_params)
        # Ensure feature_config is passed and stored
        assert isinstance(feature_config, BaseFeatureConfig), "feature_config must be provided and be a BaseFeatureConfig instance."
        self.feature_config = feature_config

        # Default random state for reproducibility if not provided
        if 'random_state' not in self.params:
            self.params['random_state'] = 42 # Default state for home model

        # Set objective to 'poisson' for goal count regression, if not specified
        if 'objective' not in self.params:
            self.params['objective'] = 'poisson'
            print("Defaulting LightGBM objective to 'poisson'.")
        elif self.params['objective'] != 'poisson':
             warnings.warn(f"LightGBM objective set to '{self.params['objective']}'. Consider 'poisson' for goal prediction.")

        # Instantiate the core model objects
        # Use a different random state for the away model for slight diversity
        away_params = self.params.copy()
        away_params['random_state'] = self.params.get('random_state', 42) + 1

        # Ensure n_jobs is set for performance, default to -1 (use all cores)
        if 'n_jobs' not in self.params:
            self.params['n_jobs'] = -1
            away_params['n_jobs'] = -1

        self._model_home = lgb.LGBMRegressor(**self.params)
        self._model_away = lgb.LGBMRegressor(**away_params)

        # Store models in a dictionary compatible with base save/load
        self._model = {'home': self._model_home, 'away': self._model_away}
        print(f"Initialized GradientBoostingModel (LightGBM) with params: {self.params} (Home) and {away_params} (Away)")

    def _fit_model(self, X_scaled: pd.DataFrame, y: pd.DataFrame):
        """Fits two LightGBM Regressors using SCALED features."""
        # --- Assertions specific to goal targets ---
        target_hg = self.feature_config.target_home_goals
        target_ag = self.feature_config.target_away_goals
        assert target_hg in y.columns, f"Target column '{target_hg}' not found in y."
        assert target_ag in y.columns, f"Target column '{target_ag}' not found in y."
        assert pd.api.types.is_numeric_dtype(y[target_hg]), f"Target '{target_hg}' is not numeric."
        assert pd.api.types.is_numeric_dtype(y[target_ag]), f"Target '{target_ag}' is not numeric."
        assert not y[[target_hg, target_ag]].isnull().any().any(), f"Target columns contain NaN values."
        # Poisson objective handles non-negative nature implicitly, but check anyway.
        assert np.all(y[target_hg] >= 0), f"Target '{target_hg}' contains negative values."
        assert np.all(y[target_ag] >= 0), f"Target '{target_ag}' contains negative values."

        assert X_scaled.columns.tolist() == self.features_in_, "Scaled features columns mismatch features_in_"

        print(f"Fitting LightGBM Regressor for Home Goals ({target_hg}) using scaled features...")
        # Note: Consider adding early stopping if you adapt the structure to include validation sets
        self._model['home'].fit(X_scaled, y[target_hg])

        print(f"Fitting LightGBM Regressor for Away Goals ({target_ag}) using scaled features...")
        self._model['away'].fit(X_scaled, y[target_ag])

        print("LightGBM models fitted successfully.")

    def _predict_proba_model(self, X_scaled: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predicts expected goals (lambdas) using SCALED features and calculates probabilities."""
        assert X_scaled.columns.tolist() == self.features_in_, "Scaled prediction features columns mismatch features_in_"

        print("Predicting expected goals (lambdas) using LightGBM models...")
        lambda_home = self._model['home'].predict(X_scaled)
        lambda_away = self._model['away'].predict(X_scaled)

        # Ensure non-negative lambdas, crucial for poisson calculations
        # LightGBM with Poisson objective should already output non-negative values, but clip for safety.
        lambda_home = np.maximum(lambda_home, 1e-9)
        lambda_away = np.maximum(lambda_away, 1e-9)

        print("Calculating outcome probabilities from LightGBM-predicted lambdas using calculate_poisson_outcome_probs...")
        # Reuse the standardized probability calculation logic
        outcome_probs = calculate_poisson_outcome_probs(lambda_home, lambda_away)

        # Add the model's predicted expected goals (lambdas)
        outcome_probs['expected_HG'] = lambda_home
        outcome_probs['expected_AG'] = lambda_away

        # --- Standard Assertions on Output Dictionary ---
        num_rows = X_scaled.shape[0]
        # Check presence of core single outcome keys
        expected_single_keys = {
            'prob_H', 'prob_D', 'prob_A', 'prob_O25', 'prob_U25', 'prob_BTTS_Y', 'prob_BTTS_N',
            'expected_HG', 'expected_AG'
        }
        present_keys = set(outcome_probs.keys())
        assert expected_single_keys.issubset(present_keys), \
            f"Output keys missing expected singles.\nMissing: {expected_single_keys - present_keys}"
        # Check presence of at least one dual outcome key
        assert any(k.startswith('prob_') and '_and_' in k for k in present_keys), \
            "Output keys do not contain any dual probability keys (e.g., 'prob_H_and_O25')."

        # Check shapes and value ranges
        for key, arr in outcome_probs.items():
            assert isinstance(arr, np.ndarray), f"Output '{key}' is not a numpy array."
            assert arr.shape == (num_rows,), f"Output '{key}' has incorrect shape {arr.shape}, expected ({num_rows},)."
            if key.startswith("prob_"):
                 assert np.all((arr >= 0) & (arr <= 1)), f"Probabilities in '{key}' are outside [0, 1]."
            elif key.startswith("expected_"):
                 assert np.all(arr >= 0), f"Expected goals in '{key}' are negative."

        # Check basic probability consistency
        assert np.allclose(outcome_probs['prob_H'] + outcome_probs['prob_D'] + outcome_probs['prob_A'], 1.0, atol=1e-6)
        assert np.allclose(outcome_probs['prob_O25'] + outcome_probs['prob_U25'], 1.0, atol=1e-6)
        assert np.allclose(outcome_probs['prob_BTTS_Y'] + outcome_probs['prob_BTTS_N'], 1.0, atol=1e-6)

        print("LightGBM-based probabilities calculated successfully.")
        return outcome_probs

    # Inherit fit, predict_proba, save, load from BaseModel