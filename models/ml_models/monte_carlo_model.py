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
    def __init__(self, model_params: Dict[str, Any], feature_config: BaseFeatureConfig):
        """
        Initializes the MonteCarloModel.

        Args:
            model_params: Dictionary potentially containing:
                - 'n_simulations' (int): Number of simulations per match (default: 20000).
                - 'internal_estimator_alpha' (float): Regularization for internal lambda estimator (default: 1.0).
            feature_config: The feature configuration object.
        """
        super().__init__(model_params) # Pass model_params up for potential base class use
        assert isinstance(feature_config, BaseFeatureConfig), "feature_config is required."
        self.feature_config = feature_config

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
        and calculates outcome probabilities by aggregation.
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
        sim_O05 = sim_total_goals > 0.5; sim_U05 = ~sim_O05
        sim_O15 = sim_total_goals > 1.5; sim_U15 = ~sim_O15
        sim_O25 = sim_total_goals > 2.5; sim_U25 = ~sim_O25
        sim_O35 = sim_total_goals > 3.5; sim_U35 = ~sim_O35
        sim_O45 = sim_total_goals > 4.5; sim_U45 = ~sim_O45

        # Goal Bands
        sim_goals_0_1 = (sim_total_goals >= 0) & (sim_total_goals <= 1)
        sim_goals_2_3 = (sim_total_goals >= 2) & (sim_total_goals <= 3)
        sim_goals_2_4 = (sim_total_goals >= 2) & (sim_total_goals <= 4)
        sim_goals_3_plus = sim_total_goals >= 3 # Equivalent to O2.5

        # Derived Double Chance
        sim_1X = sim_home_win | sim_draw
        sim_12 = sim_home_win | sim_away_win
        sim_X2 = sim_draw | sim_away_win

        # --- 4. Calculate ALL Dual Outcomes (Vectorized Evaluation) ---
        # Store base conditions for easier reference
        sim_conditions = {
            'H': sim_home_win, 'D': sim_draw, 'A': sim_away_win,
            '1X': sim_1X, '12': sim_12, 'X2': sim_X2,
            'O05': sim_O05, 'U05': sim_U05, 'O15': sim_O15, 'U15': sim_U15,
            'O25': sim_O25, 'U25': sim_U25, 'O35': sim_O35, 'U35': sim_U35,
            'O45': sim_O45, 'U45': sim_U45,
            'BTTS_Y': sim_btts_yes, 'BTTS_N': sim_btts_no,
        }

        dual_conditions_map = {}

        # --- 1X2 & O/U X.5 ---
        for result in ['H', 'D', 'A']:
            for total in ['O05', 'U05', 'O15', 'U15', 'O25', 'U25', 'O35', 'U35', 'O45', 'U45']:
                key = f"{result}_and_{total}"
                dual_conditions_map[key] = sim_conditions[result] & sim_conditions[total]

        # --- Double Chance & O/U X.5 ---
        for dc in ['1X', '12', 'X2']:
            for total in ['O05', 'U05', 'O15', 'U15', 'O25', 'U25', 'O35', 'U35', 'O45', 'U45']:
                key = f"{dc}_and_{total}"
                dual_conditions_map[key] = sim_conditions[dc] & sim_conditions[total]

        # --- 1X2 & BTTS ---
        for result in ['H', 'D', 'A']:
            for btts in ['BTTS_Y', 'BTTS_N']:
                key = f"{result}_and_{btts}"
                dual_conditions_map[key] = sim_conditions[result] & sim_conditions[btts]

        # --- Double Chance & BTTS ---
        for dc in ['1X', '12', 'X2']:
            for btts in ['BTTS_Y', 'BTTS_N']:
                key = f"{dc}_and_{btts}"
                dual_conditions_map[key] = sim_conditions[dc] & sim_conditions[btts]

        # --- O/U X.5 & BTTS ---
        for total in ['O05', 'U05', 'O15', 'U15', 'O25', 'U25', 'O35', 'U35', 'O45', 'U45']:
             for btts in ['BTTS_Y', 'BTTS_N']:
                key = f"{total}_and_{btts}"
                dual_conditions_map[key] = sim_conditions[total] & sim_conditions[btts]

        print(f"Generated {len(dual_conditions_map)} dual outcome conditions.")

        # --- 5. Aggregate Probabilities ---
        outcome_probs = {
            # Singles (Calculated directly or derived)
            'prob_H': np.mean(sim_home_win, axis=0),
            'prob_D': np.mean(sim_draw, axis=0),
            'prob_A': np.mean(sim_away_win, axis=0),
            'prob_1X': np.mean(sim_1X, axis=0),
            'prob_12': np.mean(sim_12, axis=0),
            'prob_X2': np.mean(sim_X2, axis=0),
            'prob_O05': np.mean(sim_O05, axis=0), 'prob_U05': np.mean(sim_U05, axis=0),
            'prob_O15': np.mean(sim_O15, axis=0), 'prob_U15': np.mean(sim_U15, axis=0),
            'prob_O25': np.mean(sim_O25, axis=0), 'prob_U25': np.mean(sim_U25, axis=0),
            'prob_O35': np.mean(sim_O35, axis=0), 'prob_U35': np.mean(sim_U35, axis=0),
            'prob_O45': np.mean(sim_O45, axis=0), 'prob_U45': np.mean(sim_U45, axis=0),
            'prob_BTTS_Y': np.mean(sim_btts_yes, axis=0), 'prob_BTTS_N': np.mean(sim_btts_no, axis=0),
            # Goal Bands
            'prob_goals_0_1': np.mean(sim_goals_0_1, axis=0),
            'prob_goals_2_3': np.mean(sim_goals_2_3, axis=0),
            'prob_goals_2_4': np.mean(sim_goals_2_4, axis=0),
            'prob_goals_3_plus': np.mean(sim_goals_3_plus, axis=0),
            # Doubles (Aggregated from the map)
            **{f'prob_{key}': np.mean(sim_array, axis=0) for key, sim_array in dual_conditions_map.items()}
        }

        # Add the estimated lambdas used for simulation
        outcome_probs['expected_HG'] = lambda_home_est
        outcome_probs['expected_AG'] = lambda_away_est

        # --- 6. Assertions and Clipping ---
        # Clip all probabilities
        for key in outcome_probs:
            if key.startswith("prob_"):
                outcome_probs[key] = np.clip(outcome_probs[key], 0.0, 1.0)

        # --- Standard Assertions (Check a few key ones) ---
        num_rows = X_scaled.shape[0]
        core_keys = { # Check core singles and a few representative duals
            'prob_H', 'prob_D', 'prob_A', 'prob_O25', 'prob_U25', 'prob_BTTS_Y', 'prob_BTTS_N',
            'prob_1X', 'prob_X2', 'prob_O45', 'prob_U05',
            'prob_H_and_O25', 'prob_1X_and_U15', 'prob_O35_and_BTTS_N', 'prob_A_and_BTTS_Y',
            'expected_HG', 'expected_AG'
        }
        present_keys = set(outcome_probs.keys())
        assert core_keys.issubset(present_keys), \
            f"Output keys missing expected core outcomes.\nMissing: {core_keys - present_keys}"
        # Check total number of probability keys (approximate check)
        # Expected singles (1X2, DC, O/U 0.5-4.5, BTTS, Bands) ~ 3+3+10+2+4 = 22
        # Expected duals = 30+30+6+6+10 = 82
        # Total prob keys ~ 104
        num_prob_keys = sum(1 for k in present_keys if k.startswith('prob_'))
        print(f"Generated {num_prob_keys} probability keys.")
        assert num_prob_keys > 100, f"Expected over 100 probability keys, found {num_prob_keys}"


        # Check shapes and value ranges
        for key, arr in outcome_probs.items():
            assert isinstance(arr, np.ndarray), f"Output '{key}' is not a numpy array."
            assert arr.shape == (num_rows,), f"Output '{key}' has incorrect shape {arr.shape}, expected ({num_rows},)."
            if key.startswith("prob_"):
                 assert np.all((arr >= 0) & (arr <= 1)), f"Probabilities in '{key}' are outside [0, 1]."
            elif key.startswith("expected_"):
                 assert np.all(arr >= 0), f"Expected goals in '{key}' are negative."

        # Check basic probability consistency (should hold due to simulation)
        tol = 1e-5 # Looser tolerance for MC
        assert np.allclose(outcome_probs['prob_H'] + outcome_probs['prob_D'] + outcome_probs['prob_A'], 1.0, atol=tol)
        assert np.allclose(outcome_probs['prob_O25'] + outcome_probs['prob_U25'], 1.0, atol=tol)
        assert np.allclose(outcome_probs['prob_BTTS_Y'] + outcome_probs['prob_BTTS_N'], 1.0, atol=tol)
        assert np.allclose(outcome_probs['prob_O45'] + outcome_probs['prob_U45'], 1.0, atol=tol)
        assert np.allclose(outcome_probs['prob_1X'] + outcome_probs['prob_A'], 1.0, atol=tol)

        # Check dual consistency examples
        assert np.allclose(outcome_probs['prob_H_and_O25'] + outcome_probs['prob_D_and_O25'] + outcome_probs['prob_A_and_O25'], outcome_probs['prob_O25'], atol=tol)
        assert np.allclose(outcome_probs['prob_1X_and_U35'] + outcome_probs['prob_A_and_U35'], outcome_probs['prob_U35'], atol=tol) # 1X + A = All outcomes
        assert np.allclose(outcome_probs['prob_O15_and_BTTS_Y'] + outcome_probs['prob_O15_and_BTTS_N'], outcome_probs['prob_O15'], atol=tol)


        print(f"Monte Carlo simulations complete. All outcome probabilities calculated.")
        return outcome_probs

    # Inherit fit, predict_proba, save, load from BaseModel