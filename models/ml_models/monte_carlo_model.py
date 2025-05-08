# ml_models/monte_carlo_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor # Internal model for lambda estimation
from typing import Dict, Any
from models.base_model import BaseModel
from models.utils.features import BaseFeatureConfig

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
        assert isinstance(self.feature_config, BaseFeatureConfig), "feature_config is required and must be a BaseFeatureConfig instance."
        # self.feature_config = feature_config # This is now handled by BaseModel.__init__

        # --- Model Specific Parameters ---
        self.n_simulations: int = self.params.get('n_simulations', 20000) # Use value from params if passed
        internal_alpha = self.params.get('internal_estimator_alpha', 1.0) # Allow configuring internal model
        assert isinstance(self.n_simulations, int) and self.n_simulations > 0, \
            "'n_simulations' must be a positive integer."
        assert isinstance(internal_alpha, (float, int)) and internal_alpha >= 0, \
            "'internal_estimator_alpha' must be non-negative."

        # --- Internal Model for Lambda Estimation ---
        self._lambda_estimator_home = PoissonRegressor(alpha=internal_alpha, max_iter=1000, tol=1e-3, warm_start=False)
        self._lambda_estimator_away = PoissonRegressor(alpha=internal_alpha, max_iter=1000, tol=1e-3, warm_start=False)

        self._model = {
             'n_simulations': self.n_simulations,
             'internal_estimator_alpha': internal_alpha,
             'estimator_home': self._lambda_estimator_home,
             'estimator_away': self._lambda_estimator_away
        }
        print(f"Initialized MonteCarloModel with {self.n_simulations} simulations.")
        print(f"Internal lambda estimators use alpha={internal_alpha}.")

    def _fit_model(self, X_scaled: pd.DataFrame, y: pd.DataFrame):
        """
        Fits the internal PoissonRegressors used for lambda estimation,
        using the scaled input features.
        """
        target_hg = self.feature_config.target_home_goals
        target_ag = self.feature_config.target_away_goals
        assert target_hg in y.columns and target_ag in y.columns
        assert pd.api.types.is_numeric_dtype(y[target_hg]) and pd.api.types.is_numeric_dtype(y[target_ag])
        assert not y[[target_hg, target_ag]].isnull().any().any()
        assert np.all(y[target_hg] >= 0) and np.all(y[target_ag] >= 0)
        assert X_scaled.columns.tolist() == self.features_in_, "Scaled features columns mismatch features_in_"

        print("Fitting internal lambda estimators for MonteCarloModel...")
        try:
            self._model['estimator_home'].fit(X_scaled, y[target_hg])
            self._model['estimator_away'].fit(X_scaled, y[target_ag])
        except Exception as e:
            print(f"ERROR fitting internal estimators for MonteCarloModel: {e}")
            raise
        print("Internal lambda estimators fitted successfully.")

    def _predict_proba_model(self, X_scaled: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Estimates lambdas using the internal fitted models, runs Monte Carlo simulations,
        and calculates outcome probabilities by aggregation for an expanded set of markets.
        """
        assert X_scaled.columns.tolist() == self.features_in_, "Scaled prediction features columns mismatch features_in_"
        assert 'estimator_home' in self._model and 'estimator_away' in self._model, "Internal estimators not found in model state."
        assert hasattr(self._model['estimator_home'], 'predict'), "Internal home estimator cannot predict."
        assert hasattr(self._model['estimator_away'], 'predict'), "Internal away estimator cannot predict."

        n_matches = X_scaled.shape[0]
        print(f"Estimating lambdas using internal estimators for {n_matches} matches...")

        # --- 1. Estimate Lambdas using internal model ---
        try:
            lambda_home_est = self._model['estimator_home'].predict(X_scaled)
            lambda_away_est = self._model['estimator_away'].predict(X_scaled)
        except Exception as e:
             print(f"ERROR predicting lambdas with internal estimators: {e}")
             raise
        lambda_home_est = np.maximum(lambda_home_est, 1e-9)
        lambda_away_est = np.maximum(lambda_away_est, 1e-9)

        print(f"Running {self.n_simulations} Monte Carlo simulations...")
        # --- 2. Run Simulations (Vectorized) ---
        sim_hg = np.random.poisson(lambda_home_est, size=(self.n_simulations, n_matches))
        sim_ag = np.random.poisson(lambda_away_est, size=(self.n_simulations, n_matches))

        # --- 3. Calculate Base Outcomes for Each Simulation ---
        sim_total_goals = sim_hg + sim_ag
        sim_home_win = sim_hg > sim_ag
        sim_draw = sim_hg == sim_ag
        sim_away_win = sim_hg < sim_ag
        sim_btts_yes = (sim_hg > 0) & (sim_ag > 0)
        sim_btts_no = ~sim_btts_yes

        # O/U Lines
        ou_lines_sim = {}
        for val_str, val_num in {"05":0.5, "15":1.5, "25":2.5, "35":3.5, "45":4.5}.items():
            ou_lines_sim[f'O{val_str}'] = sim_total_goals > val_num
            ou_lines_sim[f'U{val_str}'] = sim_total_goals <= val_num # Or `~ou_lines_sim[f'O{val_str}']` after O line is defined

        # Goal Bands
        sim_goals_0_1 = (sim_total_goals >= 0) & (sim_total_goals <= 1)
        sim_goals_2_3 = (sim_total_goals >= 2) & (sim_total_goals <= 3)
        sim_goals_2_4 = (sim_total_goals >= 2) & (sim_total_goals <= 4)
        sim_goals_3_plus = sim_total_goals >= 3 

        # Derived Double Chance
        sim_1X = sim_home_win | sim_draw
        sim_12 = sim_home_win | sim_away_win
        sim_X2 = sim_draw | sim_away_win

        # --- 4. Calculate ALL Dual Outcomes (Vectorized Evaluation) ---
        sim_conditions = {
            'H': sim_home_win, 'D': sim_draw, 'A': sim_away_win,
            '1X': sim_1X, '12': sim_12, 'X2': sim_X2,
            'BTTS_Y': sim_btts_yes, 'BTTS_N': sim_btts_no,
            **ou_lines_sim # Add all O/U lines to sim_conditions
        }

        dual_conditions_map = {}
        ou_line_keys = list(ou_lines_sim.keys()) # e.g. ['O05', 'U05', 'O15', 'U15', ...]

        # --- Result & O/U X.5 ---
        for result in ['H', 'D', 'A']:
            for ou_line_key in ou_line_keys: # Iterate through all O/U lines
                key = f"{result}_and_{ou_line_key}"
                dual_conditions_map[key] = sim_conditions[result] & sim_conditions[ou_line_key]

        # --- Double Chance & O/U X.5 ---
        for dc in ['1X', '12', 'X2']:
            for ou_line_key in ou_line_keys: # Iterate through all O/U lines
                key = f"{dc}_and_{ou_line_key}"
                dual_conditions_map[key] = sim_conditions[dc] & sim_conditions[ou_line_key]

        # --- Result & BTTS ---
        for result in ['H', 'D', 'A']:
            for btts_key in ['BTTS_Y', 'BTTS_N']:
                key = f"{result}_and_{btts_key}"
                dual_conditions_map[key] = sim_conditions[result] & sim_conditions[btts_key]
        
        # --- Double Chance & BTTS ---
        for dc in ['1X', '12', 'X2']:
            for btts_key in ['BTTS_Y', 'BTTS_N']:
                key = f"{dc}_and_{btts_key}"
                dual_conditions_map[key] = sim_conditions[dc] & sim_conditions[btts_key]

        # --- O/U X.5 & BTTS ---
        for ou_line_key in ou_line_keys: # Iterate through all O/U lines
             for btts_key in ['BTTS_Y', 'BTTS_N']:
                key = f"{ou_line_key}_and_{btts_key}"
                dual_conditions_map[key] = sim_conditions[ou_line_key] & sim_conditions[btts_key]
        
        print(f"Generated {len(dual_conditions_map)} dual outcome conditions for Monte Carlo.")

        # --- 5. Aggregate Probabilities ---
        outcome_probs_raw = {
            'prob_H': np.mean(sim_home_win, axis=0),
            'prob_D': np.mean(sim_draw, axis=0),
            'prob_A': np.mean(sim_away_win, axis=0),
            'prob_1X': np.mean(sim_1X, axis=0),
            'prob_12': np.mean(sim_12, axis=0),
            'prob_X2': np.mean(sim_X2, axis=0),
            'prob_BTTS_Y': np.mean(sim_btts_yes, axis=0),
            'prob_BTTS_N': np.mean(sim_btts_no, axis=0),
            'prob_goals_0_1': np.mean(sim_goals_0_1, axis=0),
            'prob_goals_2_3': np.mean(sim_goals_2_3, axis=0),
            'prob_goals_2_4': np.mean(sim_goals_2_4, axis=0),
            'prob_goals_3_plus': np.mean(sim_goals_3_plus, axis=0),
            # Add all O/U lines
            **{f'prob_{key}': np.mean(sim_array, axis=0) for key, sim_array in ou_lines_sim.items()},
            # Add all dual outcomes
            **{f'prob_{key}': np.mean(sim_array, axis=0) for key, sim_array in dual_conditions_map.items()}
        }
        outcome_probs_raw['expected_HG'] = lambda_home_est
        outcome_probs_raw['expected_AG'] = lambda_away_est

        # Clip all probabilities
        for key in outcome_probs_raw:
            if key.startswith("prob_"): # Clip only probabilities
                outcome_probs_raw[key] = np.clip(outcome_probs_raw[key], 0.0, 1.0)
        
        # --- Prefix all keys with model name ---
        outcome_probs = {f"monte_carlo_{key}": value for key, value in outcome_probs_raw.items()}


        # --- 6. Assertions (checking prefixed_probs) ---
        tol = 1e-5 # Looser tolerance for MC
        assert np.allclose(outcome_probs['monte_carlo_prob_H'] + outcome_probs['monte_carlo_prob_D'] + outcome_probs['monte_carlo_prob_A'], 1.0, atol=tol)
        # Add more checks for other O/U lines and duals if necessary

        print(f"Monte Carlo simulations complete. All outcome probabilities calculated.")
        return outcome_probs

    # Inherit fit, predict_proba, save, load from BaseModel