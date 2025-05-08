# ml_models/gradient_boosting_model.py
import pandas as pd
import numpy as np
import lightgbm as lgb # Use LightGBM
from typing import Dict, Any
import warnings

from models.base_model import BaseModel
from models.utils.features import BaseFeatureConfig
# Import the standardized probability calculation function
from models.ml_models.poisson_model import calculate_poisson_outcome_probs, _calculate_dual_conditions

class GradientBoostingModel(BaseModel):
    """
    LightGBM Regressor model to predict expected goals (lambdas) based on features,
    using a Poisson objective. Derives outcome probabilities using the
    calculate_poisson_outcome_probs function.
    Inherits scaling and save/load logic from BaseModel.
    """
    def __init__(self, model_params: Dict[str, Any], feature_config: BaseFeatureConfig, apply_scaling: bool = True):
        """Initializes the GradientBoostingModel."""
        # Pass model_params, feature_config, and apply_scaling to the BaseModel constructor
        super().__init__(model_params, feature_config=feature_config, apply_scaling=apply_scaling)
        
        # Ensure feature_config is passed and stored (This is now handled by BaseModel, but the assert is fine if kept for clarity here)
        assert isinstance(self.feature_config, BaseFeatureConfig), "feature_config must be provided and be a BaseFeatureConfig instance."
        # self.feature_config = feature_config # This is now handled by BaseModel.__init__

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

        # --- Relax strict probability assertions ---
        # Check probability sums and renormalize if needed
        h_d_a_sum = outcome_probs['prob_H'] + outcome_probs['prob_D'] + outcome_probs['prob_A']
        if not np.allclose(h_d_a_sum, 1.0, atol=1e-6):
            print(f"Warning: 1X2 probs don't sum to 1.0 exactly. Mean sum: {np.mean(h_d_a_sum)}. Renormalizing...")
            sum_expanded = np.maximum(h_d_a_sum, 1e-12)  # Avoid division by zero
            outcome_probs['prob_H'] /= sum_expanded
            outcome_probs['prob_D'] /= sum_expanded
            outcome_probs['prob_A'] /= sum_expanded
        
        # --- Prefix all keys with model name ---
        prefixed_probs = {f"gradient_boosting_{key}": value for key, value in outcome_probs.items()}

        # Check shapes and value ranges
        num_rows = X_scaled.shape[0]
        for key, arr in prefixed_probs.items():
            assert isinstance(arr, np.ndarray), f"Output '{key}' is not a numpy array."
            assert arr.shape == (num_rows,), f"Output '{key}' has incorrect shape {arr.shape}, expected ({num_rows},)."
            if "prob_" in key:
                assert np.all((arr >= 0) & (arr <= 1)), f"Probabilities in '{key}' are outside [0, 1]."
            elif "expected_" in key:
                assert np.all(arr >= 0), f"Expected goals in '{key}' are negative."

        print("LightGBM-based probabilities calculated successfully.")
        return prefixed_probs

    # Inherit fit, predict_proba, save, load from BaseModel