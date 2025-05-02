import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson
import gc
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseModel

class MonteCarloModel(BaseModel):
    """
    Monte Carlo simulation model for football match prediction.
    Uses XGBoost Poisson regressors to predict goals and runs simulations
    to calculate outcome probabilities.
    """
    def __init__(self, 
                 n_simulations=10000, 
                 batch_size=1000, 
                 use_float16=True,
                 random_seed=42, 
                 xgb_params=None,
                 **kwargs):
        """
        Initialize the Monte Carlo model.
        
        Args:
            n_simulations: Number of simulations to run per match
            batch_size: Batch size for processing simulations to manage memory
            use_float16: Use float16 to reduce memory consumption
            random_seed: Random seed for reproducibility
            xgb_params: Parameters for XGBoost models
        """
        super().__init__(**kwargs)
        self.n_simulations = n_simulations
        self.batch_size = batch_size
        self.use_float16 = use_float16
        self.random_seed = random_seed
        
        # Default XGBoost parameters
        self.xgb_params = xgb_params or {
            'objective': 'count:poisson', 
            'eval_metric': 'poisson-nloglik',
            'eta': 0.05, 
            'max_depth': 4, 
            'subsample': 0.8, 
            'colsample_bytree': 0.8,
            'min_child_weight': 1, 
            'gamma': 0.1, 
            'lambda': 1, 
            'alpha': 0,
            'enable_categorical': False
        }
        
        # Models will be initialized during fit
        self.model_hg = None
        self.model_ag = None
        self.features_hg = None
        self.features_ag = None
        
        # Defines which outcomes/scenarios we'll predict
        self.scenario_names = [
            'H', 'D', 'A', 'Over 1.5', 'Over 2.5', 'Under 2.5', 'Under 3.5', 
            'BTTS Yes', 'BTTS No', '1X', 'X2', '12',
            # Combined scenarios
            'H and O1.5', 'H and O2.5', 'H and U2.5', 'H and U3.5', 
            'H and BTTS Yes', 'H and BTTS No',
            'D and O1.5', 'D and U2.5', 'D and BTTS Yes', 'D and BTTS No',
            'A and O1.5', 'A and O2.5', 'A and U2.5', 'A and U3.5',
            'A and BTTS Yes', 'A and BTTS No',
            '1X and O1.5', '1X and O2.5', '1X and U2.5', '1X and U3.5',
            '1X and BTTS Yes', '1X and BTTS No',
            'X2 and O1.5', 'X2 and O2.5', 'X2 and U2.5', 'X2 and U3.5',
            'X2 and BTTS Yes', 'X2 and BTTS No',
            '12 and O1.5', '12 and O2.5', '12 and U2.5',
            '12 and BTTS Yes', '12 and BTTS No',
            'Over 2.5 and BTTS Yes', 'Under 2.5 and BTTS No'
        ]

    def _determine_features(self, data_df):
        """Determine feature sets for home goals and away goals models."""
        # Extract feature columns based on name patterns
        home_features = [col for col in data_df.columns if col.startswith('Home_')]
        away_features = [col for col in data_df.columns if col.startswith('Away_')]
        
        # Filter specific features for attacking and defense
        home_attack = [f for f in home_features if any(k in f for k in 
                      ['Scored', 'For', 'FormPoints', 'W_Count', 'BTTS', 'PossessionFor'])]
        home_defense = [f for f in home_features if any(k in f for k in 
                       ['Conceded', 'Against', 'L_Count', 'PossessionAgainst'])]
        away_attack = [f for f in away_features if any(k in f for k in 
                      ['Scored', 'For', 'FormPoints', 'W_Count', 'BTTS', 'PossessionFor'])]
        away_defense = [f for f in away_features if any(k in f for k in 
                       ['Conceded', 'Against', 'L_Count', 'PossessionAgainst'])]
        
        # Combine features
        features_hg = sorted(list(set(home_attack + away_defense)))
        features_ag = sorted(list(set(away_attack + home_defense)))
        
        return features_hg, features_ag

    def fit(self, X_train, y_train):
        """
        Fit the model to the training data.
        
        Args:
            X_train: DataFrame with features
            y_train: DataFrame with FTHG and FTAG columns for targets
        
        Returns:
            self: The fitted model
        """
        print(f"Fitting Monte Carlo model with {X_train.shape[0]} samples")
        
        # Check if y_train has the required target columns
        if not isinstance(y_train, pd.DataFrame) or 'FTHG' not in y_train.columns or 'FTAG' not in y_train.columns:
            raise ValueError("y_train must be a DataFrame with 'FTHG' and 'FTAG' columns")
        
        # Determine feature sets
        self.features_hg, self.features_ag = self._determine_features(X_train)
        
        # Ensure features exist
        if not self.features_hg or not self.features_ag:
            raise ValueError("Could not find sufficient features for HG/AG models")
        
        print(f"Using {len(self.features_hg)} features for HG model")
        print(f"Using {len(self.features_ag)} features for AG model")
        
        # Extract targets
        y_train_hg = y_train['FTHG']
        y_train_ag = y_train['FTAG']
        
        # Train Home Goals model
        print("Training Home Goals model...")
        self.model_hg = xgb.XGBRegressor(
            **self.xgb_params,
            n_estimators=500,
            early_stopping_rounds=20,
            random_state=self.random_seed
        )
        self.model_hg.fit(
            X_train[self.features_hg], 
            y_train_hg,
            eval_set=[(X_train[self.features_hg], y_train_hg)],
            verbose=False
        )
        
        # Train Away Goals model
        print("Training Away Goals model...")
        self.model_ag = xgb.XGBRegressor(
            **self.xgb_params,
            n_estimators=500,
            early_stopping_rounds=20,
            random_state=self.random_seed + 1
        )
        self.model_ag.fit(
            X_train[self.features_ag], 
            y_train_ag,
            eval_set=[(X_train[self.features_ag], y_train_ag)],
            verbose=False
        )
        
        print(f"Models trained successfully. HG best iteration: {self.model_hg.best_iteration}")
        print(f"AG best iteration: {self.model_ag.best_iteration}")
        
        self.is_fitted = True
        return self

    def predict(self, X_test, **kwargs):
        """
        Predict match outcomes using Monte Carlo simulation.
        
        Args:
            X_test: DataFrame with features for prediction
            
        Returns:
            DataFrame with prediction probabilities for various outcomes
        """
        super().predict(X_test)  # Call parent for checking if fitted
        
        num_matches = X_test.shape[0]
        print(f"Predicting for {num_matches} matches using Monte Carlo simulation")
        
        # Predict Poisson lambdas
        lambda_hg_pred = np.maximum(self.model_hg.predict(X_test[self.features_hg]), 0.01)
        lambda_ag_pred = np.maximum(self.model_ag.predict(X_test[self.features_ag]), 0.01)
        
        # Initialize scenario counters
        scenario_counts = {scenario: np.zeros(num_matches, dtype=np.int64) 
                           for scenario in self.scenario_names}
        
        # Run batched simulations
        print(f"Running {self.n_simulations} simulations in batches of {self.batch_size}")
        for batch_start in range(0, self.n_simulations, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.n_simulations)
            batch_size_actual = batch_end - batch_start
            
            # Generate simulations
            dtype = np.float16 if self.use_float16 else np.float64
            sim_hg_batch = poisson.rvs(mu=lambda_hg_pred[:, np.newaxis], size=(num_matches, batch_size_actual)).astype(dtype)
            sim_ag_batch = poisson.rvs(mu=lambda_ag_pred[:, np.newaxis], size=(num_matches, batch_size_actual)).astype(dtype)
            sim_total_goals = sim_hg_batch + sim_ag_batch
            
            # Calculate basic outcomes
            cond_H = sim_hg_batch > sim_ag_batch
            cond_D = sim_hg_batch == sim_ag_batch
            cond_A = sim_hg_batch < sim_ag_batch
            cond_O15 = sim_total_goals > 1.5
            cond_O25 = sim_total_goals > 2.5
            cond_U25 = sim_total_goals < 2.5
            cond_U35 = sim_total_goals < 3.5
            cond_BTTS_Yes = (sim_hg_batch > 0) & (sim_ag_batch > 0)
            cond_BTTS_No = ~cond_BTTS_Yes
            cond_1X = cond_H | cond_D
            cond_X2 = cond_A | cond_D
            cond_12 = cond_H | cond_A
            
            # Define scenario conditions
            scenario_conditions = {
                'H': cond_H,
                'D': cond_D,
                'A': cond_A,
                'Over 1.5': cond_O15,
                'Over 2.5': cond_O25,
                'Under 2.5': cond_U25,
                'Under 3.5': cond_U35,
                'BTTS Yes': cond_BTTS_Yes,
                'BTTS No': cond_BTTS_No,
                '1X': cond_1X,
                'X2': cond_X2,
                '12': cond_12,
                # Combined scenarios
                'H and O1.5': cond_H & cond_O15,
                'H and O2.5': cond_H & cond_O25,
                'H and U2.5': cond_H & cond_U25,
                'H and U3.5': cond_H & cond_U35,
                'H and BTTS Yes': cond_H & cond_BTTS_Yes,
                'H and BTTS No': cond_H & cond_BTTS_No,
                'D and O1.5': cond_D & cond_O15,
                'D and U2.5': cond_D & cond_U25,
                'D and BTTS Yes': cond_D & cond_BTTS_Yes,
                'D and BTTS No': cond_D & cond_BTTS_No,
                'A and O1.5': cond_A & cond_O15,
                'A and O2.5': cond_A & cond_O25,
                'A and U2.5': cond_A & cond_U25,
                'A and U3.5': cond_A & cond_U35,
                'A and BTTS Yes': cond_A & cond_BTTS_Yes,
                'A and BTTS No': cond_A & cond_BTTS_No,
                '1X and O1.5': cond_1X & cond_O15,
                '1X and O2.5': cond_1X & cond_O25,
                '1X and U2.5': cond_1X & cond_U25,
                '1X and U3.5': cond_1X & cond_U35,
                '1X and BTTS Yes': cond_1X & cond_BTTS_Yes,
                '1X and BTTS No': cond_1X & cond_BTTS_No,
                'X2 and O1.5': cond_X2 & cond_O15,
                'X2 and O2.5': cond_X2 & cond_O25,
                'X2 and U2.5': cond_X2 & cond_U25,
                'X2 and U3.5': cond_X2 & cond_U35,
                'X2 and BTTS Yes': cond_X2 & cond_BTTS_Yes,
                'X2 and BTTS No': cond_X2 & cond_BTTS_No,
                '12 and O1.5': cond_12 & cond_O15,
                '12 and O2.5': cond_12 & cond_O25,
                '12 and U2.5': cond_12 & cond_U25,
                '12 and BTTS Yes': cond_12 & cond_BTTS_Yes,
                '12 and BTTS No': cond_12 & cond_BTTS_No,
                'Over 2.5 and BTTS Yes': cond_O25 & cond_BTTS_Yes,
                'Under 2.5 and BTTS No': cond_U25 & cond_BTTS_No,
            }
            
            # Update counters
            for scenario, condition in scenario_conditions.items():
                scenario_counts[scenario] += np.sum(condition, axis=1)
            
            # Free memory
            del sim_hg_batch, sim_ag_batch, sim_total_goals
            del cond_H, cond_D, cond_A, cond_O15, cond_O25, cond_U25, cond_U35
            del cond_BTTS_Yes, cond_BTTS_No, cond_1X, cond_X2, cond_12
            del scenario_conditions
            gc.collect()
            
            # Progress indicator
            if (batch_start + self.batch_size) % (self.n_simulations // 5) == 0 or batch_end == self.n_simulations:
                print(f"  Processed {batch_end}/{self.n_simulations} simulations ({batch_end/self.n_simulations:.1%})")
        
        # Calculate probabilities
        scenario_probs = {scenario: counts / self.n_simulations for scenario, counts in scenario_counts.items()}
        
        # Create results DataFrame
        results_df = pd.DataFrame(index=X_test.index)
        
        # Add predicted lambdas
        results_df['lambda_hg'] = lambda_hg_pred
        results_df['lambda_ag'] = lambda_ag_pred
        
        # Add probabilities in format expected by backtest engine
        results_df['prob_H'] = scenario_probs['H'] 
        results_df['prob_D'] = scenario_probs['D']
        results_df['prob_A'] = scenario_probs['A']
        
        # Add additional probabilities
        for scenario, probs in scenario_probs.items():
            col_name = f"prob_{scenario.replace(' ', '_').replace('.', '')}"
            results_df[col_name] = probs
        
        # Add 'prediction' column (most likely outcome)
        results_df['prediction'] = results_df[['prob_H', 'prob_D', 'prob_A']].idxmax(axis=1).str.replace('prob_', '')
        
        return results_df
        
    def save(self, filepath):
        """Save the model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model components
        model_data = {
            'model_hg': self.model_hg,
            'model_ag': self.model_ag,
            'features_hg': self.features_hg,
            'features_ag': self.features_ag,
            'xgb_params': self.xgb_params,
            'n_simulations': self.n_simulations,
            'batch_size': self.batch_size,
            'use_float16': self.use_float16,
            'random_seed': self.random_seed,
            'scenario_names': self.scenario_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath):
        """Load a saved model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        # Load model data
        model_data = joblib.load(filepath)
        
        # Create instance with same parameters
        instance = cls(
            n_simulations=model_data['n_simulations'],
            batch_size=model_data['batch_size'],
            use_float16=model_data['use_float16'],
            random_seed=model_data['random_seed'],
            xgb_params=model_data['xgb_params']
        )
        
        # Restore model state
        instance.model_hg = model_data['model_hg']
        instance.model_ag = model_data['model_ag']
        instance.features_hg = model_data['features_hg']
        instance.features_ag = model_data['features_ag']
        instance.scenario_names = model_data['scenario_names']
        instance.is_fitted = True
        
        return instance
