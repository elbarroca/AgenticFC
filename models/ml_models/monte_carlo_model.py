# ml_models/monte_carlo_model.py
import pandas as pd
import numpy as np
# from sklearn.linear_model import PoissonRegressor # Keep for fallback if external lambdas not provided
from typing import Dict, Any
from models.base_model import BaseModel
from models.utils.features import BaseFeatureConfig
# Import the standardized probability calculation function from poisson_model
from .poisson_model import calculate_poisson_outcome_probs


class MonteCarloModel(BaseModel):
    """
    Performs Monte Carlo simulations based on lambda estimates derived from
    an internal, simple PoissonRegressor model trained on input features.
    Estimates probabilistic outcomes by simulating numerous match scorelines.
    Inherits scaling and save/load logic from BaseModel.
    """
    def __init__(self, model_params: Dict[str, Any], feature_config: BaseFeatureConfig, apply_scaling: bool = True):
        """
        Initializes the MonteCarloModel.

        Args:
            model_params: Dictionary potentially containing:
                - 'n_simulations' (int): Number of simulations per match (default: 20000).
                - 'internal_estimator_alpha' (float): Regularization for internal lambda estimator (default: 1.0).
            feature_config: The feature configuration object.
            apply_scaling: Flag to indicate if internal scaling should be applied.
        """
        # Pass model_params, feature_config, and apply_scaling to the BaseModel constructor
        super().__init__(model_params, feature_config=feature_config, apply_scaling=apply_scaling)
        
        # The BaseClass now handles self.feature_config assignment.
        # The assertion below is good for this class's specific needs.
        assert isinstance(self.feature_config, BaseFeatureConfig), "feature_config is required."
        # self.feature_config = feature_config # This is now handled by BaseModel.__init__

        # --- Model Specific Parameters ---
        self.n_simulations: int = self.params.get('n_simulations', 20000)
        # internal_alpha = self.params.get('internal_estimator_alpha', 1.0) # No longer strictly needed if external lambdas are primary

        # If we want a fallback to internal estimation, we'd keep these.
        # For now, let's assume external lambdas will be provided for the "enhanced" version.
        # If not, this model would need its own lambda estimation logic if called directly.
        # self._lambda_estimator_home = PoissonRegressor(alpha=internal_alpha, max_iter=1000)
        # self._lambda_estimator_away = PoissonRegressor(alpha=internal_alpha, max_iter=1000)

        self._model = { # Store params that might be useful for saving/loading context
             'n_simulations': self.n_simulations,
             # 'internal_estimator_alpha': internal_alpha # If keeping internal estimators
        }
        print(f"Initialized MonteCarloModel with {self.n_simulations} simulations. Expects lambdas via X_scaled or internal estimation.")

    def _fit_model(self, X_scaled: pd.DataFrame, y: pd.DataFrame):
        """
        Fits internal lambda estimators IF external lambdas are not the primary mode.
        For an "enhanced" MC that *only* uses external lambdas, this might do nothing
        or fit very simple fallback estimators.
        """
        # If MC is *only* a calculator based on external lambdas, fit might be a no-op.
        # However, to keep it a complete BaseModel, it should have a fit.
        # Let's assume it can still fit its own basic lambda estimators as a fallback.
        # This part is only relevant if generate_oof_predictions.py calls fit on MC directly.
        # For the "enhanced" flow, we might bypass direct fitting of MC.

        # For now, let's make fit a no-op if we intend to always provide external lambdas
        # when this model is used in the "enhanced" way.
        # If you want it to still be trainable independently, then include the PoissonRegressor fitting.
        print("MonteCarloModel _fit_model: No explicit fitting if primarily using external lambdas for simulation.")
        print("If internal lambda estimation is desired, uncomment/implement estimator fitting here.")
        # target_hg = self.feature_config.target_home_goals
        # target_ag = self.feature_config.target_away_goals
        # if hasattr(self, '_lambda_estimator_home'): # Check if internal estimators exist
        #     print("Fitting internal lambda estimators for MonteCarloModel (fallback)...")
        #     self._model['estimator_home'].fit(X_scaled, y[target_hg])
        #     self._model['estimator_away'].fit(X_scaled, y[target_ag])
        #     print("Internal lambda estimators (fallback) fitted.")
        pass


    def _predict_proba_model(self, X_scaled: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Uses lambdas from X_scaled if provided (e.g., 'external_lambda_HG', 'external_lambda_AG'),
        otherwise would fall back to internal estimators (if implemented and fitted).
        Runs Monte Carlo simulations and calculates outcome probabilities.
        """
        assert isinstance(X_scaled, pd.DataFrame), "X_scaled must be a DataFrame"
        n_matches = X_scaled.shape[0]

        # --- 1. Get Lambdas ---
        # Check if external lambdas are provided directly in X_scaled
        # These column names would be set by generate_oof_predictions.py
        # when preparing X_val_fold for this "enhanced" MC call.
        external_lambda_hg_col = 'external_lambda_HG' # Define conventional names
        external_lambda_ag_col = 'external_lambda_AG'

        if external_lambda_hg_col in X_scaled.columns and external_lambda_ag_col in X_scaled.columns:
            print(f"MonteCarloModel: Using externally provided lambdas from columns: {external_lambda_hg_col}, {external_lambda_ag_col}")
            lambda_home_sim = X_scaled[external_lambda_hg_col].values
            lambda_away_sim = X_scaled[external_lambda_ag_col].values
        else:
            # Fallback: If you had internal estimators and wanted to use them
            # warnings.warn("External lambdas not found in X_scaled for MonteCarlo. Add internal lambda estimation if needed, or ensure X_scaled provides them.", RuntimeWarning)
            # For this "enhanced" version, we'll raise an error if external lambdas aren't provided,
            # as that's the primary intention.
            raise ValueError(f"MonteCarloModel in enhanced mode expects '{external_lambda_hg_col}' and "
                             f"'{external_lambda_ag_col}' in input X_scaled DataFrame.")
            # lambda_home_sim = self._model['estimator_home'].predict(X_scaled.drop(columns=[external_lambda_hg_col, external_lambda_ag_col], errors='ignore'))
            # lambda_away_sim = self._model['estimator_away'].predict(X_scaled.drop(columns=[external_lambda_hg_col, external_lambda_ag_col], errors='ignore'))

        lambda_home_sim = np.maximum(lambda_home_sim, 1e-9)
        lambda_away_sim = np.maximum(lambda_away_sim, 1e-9)

        print(f"Running {self.n_simulations} Monte Carlo simulations using provided lambdas...")
        # --- 2. Run Simulations (Vectorized) ---
        # This part is now identical to calculate_poisson_outcome_probs's simulation part,
        # but here it's explicitly Monte Carlo.
        # Alternatively, just call calculate_poisson_outcome_probs directly.
        
        # For consistency and to use the robust probability calculation logic:
        outcome_probs_raw = calculate_poisson_outcome_probs(lambda_home_sim, lambda_away_sim, max_goals=8) # Use existing function

        # Add the input lambdas to the output as well, clearly marked as "sim_input"
        outcome_probs_raw['sim_input_lambda_HG'] = lambda_home_sim
        outcome_probs_raw['sim_input_lambda_AG'] = lambda_away_sim
        
        # --- Prefix all keys with model name ---
        # Use a distinct prefix for this enhanced MC
        prefixed_probs = {f"monte_carlo_enhanced_{key}": value for key, value in outcome_probs_raw.items()}
        
        # Assertions (as in poisson_model)
        # ... (add assertions for shape, [0,1] range for probs, non-negative for lambdas) ...
        num_rows = X_scaled.shape[0]
        for key, arr in prefixed_probs.items():
            assert arr.shape == (num_rows,), f"MC Output '{key}' shape mismatch."
            if "prob_" in key: assert np.all((arr >= 0) & (arr <= 1.00001)), f"MC Probs '{key}' out of [0,1]." # Allow for tiny float issues before final clip
            elif "lambda_HG" in key or "lambda_AG" in key : assert np.all(arr >= 0), f"MC Lambdas '{key}' negative."


        print(f"Monte Carlo (enhanced) simulations complete. All outcome probabilities calculated.")
        return prefixed_probs

    # fit and predict_proba are inherited from BaseModel
    # The predict_proba in BaseModel will call self._predict_proba_model
    # We just need to ensure that when it's called for MC in generate_oof_predictions.py,
    # the X_val_fold DataFrame contains the 'external_lambda_HG' and 'external_lambda_AG' columns.