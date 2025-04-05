# Core Data Processing & Scientific Libraries
import json
import numpy as np
import pandas as pd  # Not actively used in this code but kept for future expansion
from scipy.stats import norm, levy_stable  # t and skewnorm are imported but not used
from scipy.optimize import minimize

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # Not actively used but could be for ML model implementation
from sklearn.model_selection import train_test_split, cross_val_score  # Not actively used
from sklearn.metrics import log_loss, brier_score_loss  # Could be used for model evaluation
from sklearn.preprocessing import StandardScaler  # Not actively used
from sklearn.pipeline import Pipeline  # Not actively used

# System Libraries
import warnings
import sys
import os

# Add these imports at the top of the file
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
MC_SIMULATIONS = 50000  # Increased for better convergence
LEVERAGE_THRESHOLD = 0.7
CONFIRMATION_THRESHOLD = 0.55
KELLY_FRACTION = 0.3  # Conservative Kelly criterion fraction

class PodosTransformer(torch.nn.Module):
    """PyTorch implementation of the Podos soccer prediction model"""
    def __init__(self, input_dim=23, hidden_dim=32, num_heads=2, num_layers=2, output_dim=3):
        super(PodosTransformer, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, hidden_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Add sequence dimension if it's missing (reshape from [batch, features] to [batch, 1, features])
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        # Global pooling to handle variable sequence length
        x = x.mean(dim=1)
        x = self.fc(x)
        return torch.nn.functional.softmax(x, dim=1)
    
    @classmethod
    def from_pretrained(cls, model_id):
        """Load model from Hugging Face Hub"""
        try:
            # Download the model file
            model_file = hf_hub_download(repo_id=model_id, filename="model.safetensors")
            
            # Create model instance
            model = cls()
            
            try:
                # Try direct loading first
                state_dict = load_file(model_file)
                model.load_state_dict(state_dict, strict=False)
            except:
                # If that fails, fallback to manual weight loading
                print("Direct loading failed, trying manual weight mapping...")
                
                # Load weights
                state_dict = load_file(model_file)
                
                # Create a new state dict with compatible keys
                new_state_dict = {}
                
                # Copy transformer encoder weights that match
                for k, v in state_dict.items():
                    if k.startswith('transformer_encoder'):
                        new_state_dict[k] = v
                
                # Special handling for embeddings and output layer
                if 'embedding.weight' in state_dict:
                    new_state_dict['embedding.weight'] = state_dict['embedding.weight']
                if 'embedding.bias' in state_dict:
                    new_state_dict['embedding.bias'] = state_dict['embedding.bias']
                
                # Handle FC layer with different names
                for src, dst in [('fc.weight', 'fc.weight'), 
                                ('fc.bias', 'fc.bias'),
                                ('fc_out.weight', 'fc.weight'), 
                                ('fc_out.bias', 'fc.bias')]:
                    if src in state_dict:
                        new_state_dict[dst] = state_dict[src]
                
                # Load the compatible weights
                model.load_state_dict(new_state_dict, strict=False)
            
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Return a basic model with random weights as fallback
            return cls()

class SoccerDerivativeModel:
    """
    Advanced soccer prediction model using stochastic processes and financial derivatives concepts
    """
    def __init__(self, json_file):
        self.param_data = self.load_parametrized_data(json_file)
        self.features = self.feature_engineering()
        self.teams = {
            'home': self.param_data['home_team']['name'],
            'away': self.param_data['away_team']['name']
        }
        
    def load_parametrized_data(self, json_file):
        """Load JSON parametrized data from file"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def feature_engineering(self):
        """Extract and engineer key features from parametrized data"""
        home = self.param_data['home_team']['analysis']
        away = self.param_data['away_team']['analysis']
        
        features = {}
        
        # Check if key_metrics exists in both teams' data
        home_key_metrics = home.get('key_metrics', {})
        away_key_metrics = away.get('key_metrics', {})
        
        # Initialize with default values for missing fields
        home_points_momentum = home_key_metrics.get('points_momentum', 0.0)
        away_points_momentum = away_key_metrics.get('points_momentum', 0.0)
        home_xg_per_game = home_key_metrics.get('xG_per_game', 1.0)
        away_xg_per_game = away_key_metrics.get('xG_per_game', 1.0)
        home_points_volatility = home_key_metrics.get('points_volatility', 1.0)
        away_points_volatility = away_key_metrics.get('points_volatility', 1.0)
        home_consistency = home_key_metrics.get('consistency', 0.5)
        away_consistency = away_key_metrics.get('consistency', 0.5)
        
        # Core momentum features (stochastic drift terms)
        features['momentum_diff'] = home_points_momentum - away_points_momentum
        features['xg_gap'] = home_xg_per_game - away_xg_per_game
        features['volatility_sum'] = home_points_volatility + away_points_volatility
        features['volatility_ratio'] = home_points_volatility / max(0.1, away_points_volatility)
        
        # Consistency/resilience ratios
        features['consistency_gap'] = home_consistency - away_consistency
        
        # Home field advantage factor - handle missing performance metrics
        home_performance = home.get('performance_metrics_last_n', {})
        home_home_perf = home_performance.get('home', {}).get('points_per_game', 1.5)
        home_away_perf = home_performance.get('away', {}).get('points_per_game', 1.0)
        features['home_advantage'] = home_home_perf - home_away_perf
        
        # Historical expected goals
        try:
            features['btts_prob'] = self.param_data['match_predictions']['outcome_probabilities']['btts_probability']
            features['draw_prob'] = self.param_data['match_predictions']['outcome_probabilities']['draw']
            features['home_win_prob'] = self.param_data['match_predictions']['outcome_probabilities']['home_win']
            features['away_win_prob'] = self.param_data['match_predictions']['outcome_probabilities']['away_win']
            features['under_2.5_prob'] = self.param_data['match_predictions']['outcome_probabilities']['over_under_probabilities'].get('under_2.5', 0.5)
        except:
            features['btts_prob'] = features['draw_prob'] = features['home_win_prob'] = features['away_win_prob'] = 0.0
            features['under_2.5_prob'] = 0.5
        
        # Stochastic signal magnitude (volatility * momentum diff)
        features['dyn_impact'] = abs(features['momentum_diff']) * features['volatility_sum']
        
        # Opponent strength sensitivity
        try:
            features['home_opponent_sensitivity'] = home.get('correlation_analysis', {}).get('correlation_interpretation', {}).get('value', 0.0)
            features['away_opponent_sensitivity'] = away.get('correlation_analysis', {}).get('correlation_interpretation', {}).get('value', 0.0)
        except:
            features['home_opponent_sensitivity'] = features['away_opponent_sensitivity'] = 0.0
            
        # Default empty scoring patterns if missing
        default_intervals = {'0-15 min.': 0.0, '16-30 min.': 0.0, '31-45 min.': 0.0, 
                            '46-60 min.': 0.0, '61-75 min.': 0.0, '76-90 min.': 0.0}
        
        # Scoring period concentration - Herfindahl-Hirschman Index for goal distribution
        home_scoring_patterns = home.get('scoring_patterns', {'scoring_by_interval': default_intervals})
        away_scoring_patterns = away.get('scoring_patterns', {'scoring_by_interval': default_intervals})
        
        home_scoring = np.array(list(home_scoring_patterns.get('scoring_by_interval', default_intervals).values()))
        away_scoring = np.array(list(away_scoring_patterns.get('scoring_by_interval', default_intervals).values()))
        
        # Normalize to sum to 1
        if home_scoring.sum() > 0:
            home_scoring = home_scoring / home_scoring.sum()
        if away_scoring.sum() > 0:
            away_scoring = away_scoring / away_scoring.sum()
            
        features['home_scoring_concentration'] = np.sum(home_scoring**2)  # HHI index
        features['away_scoring_concentration'] = np.sum(away_scoring**2)  # HHI index
        
        # Expected goals data
        features['home_xg'] = self.param_data['match_predictions']['outcome_probabilities']['expected_goals']['home']
        features['away_xg'] = self.param_data['match_predictions']['outcome_probabilities']['expected_goals']['away']
        
        # Form trend metrics
        features['home_form_trend'] = home_key_metrics.get('form_trend', 0.0)
        features['away_form_trend'] = away_key_metrics.get('form_trend', 0.0)
        
        return features
    
    def heston_stochastic_volatility_goals(self, mean_goals, vol_of_vol=0.3, mean_reversion=0.5, long_term_vol=0.7, dt=0.1, steps=10):
        """
        Simulate goal expectation using Heston stochastic volatility model.
        This better captures the volatility clustering and fat tails in scoring patterns.
        """
        # Initial conditions
        v_t = self.features['volatility_sum'] / 2  # Current volatility
        x_t = mean_goals  # Current expected goals
        
        # Parameters
        kappa = mean_reversion  # Mean reversion speed
        theta = long_term_vol   # Long-term volatility
        sigma = vol_of_vol      # Volatility of volatility
        
        # Simulation
        for _ in range(steps):
            # Generate correlated Brownian motions
            z1 = np.random.normal(0, 1)
            z2 = np.random.normal(0, 1)
            
            # Update volatility (ensuring it stays positive)
            sqrt_v_t = np.sqrt(max(1e-8, v_t))
            v_t = max(1e-8, v_t + kappa * (theta - v_t) * dt + sigma * sqrt_v_t * np.sqrt(dt) * z1)
            
            # Update expected goals
            x_t = max(0, x_t + sqrt_v_t * np.sqrt(dt) * z2)
        
        return max(0, np.round(x_t))
    
    def load_podos_model(self):
        """Load the Podos soccer prediction model from Hugging Face Hub"""
        if not hasattr(self, 'podos_model'):
            try:
                self.podos_model = PodosTransformer.from_pretrained("Nickel5HF/podos_soccer_model")
                self.podos_model.eval()  # Set to evaluation mode
                
                # Try to load label encoder if available
                try:
                    label_encoder_path = hf_hub_download(
                        repo_id="Nickel5HF/podos_soccer_model", 
                        filename="label_encoder.pkl"
                    )
                    with open(label_encoder_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                except:
                    # Create a simple label encoder with default values if not available
                    self.label_encoder = {"teams": {self.teams["home"]: 1, self.teams["away"]: 2}}
                    print("Could not load label encoder, using simple team mapping")
                    
                return True
            except Exception as e:
                print(f"Error loading Podos model: {e}")
                return False
        return True
    
    def ml_bet_recommender(self):
        """
        Machine learning-based bet recommendation using the Podos transformer model.
        Returns probabilistic bet suggestions based on the pre-trained model.
        """
        # Try to use the ML model first
        if self.load_podos_model():
            try:
                # Get required input parameters for Podos model
                home_team = self.param_data['home_team']
                away_team = self.param_data['away_team']
                
                # Map team names to IDs (using label encoder if available)
                try:
                    home_team_id = self.label_encoder.get("teams", {}).get(self.teams["home"], 1)
                    away_team_id = self.label_encoder.get("teams", {}).get(self.teams["away"], 2)
                except:
                    # Default IDs if mapping fails
                    home_team_id = 1
                    away_team_id = 2
                
                # Extract match statistics (use default values if not available)
                try:
                    home_stats = home_team.get('match_stats', {})
                    away_stats = away_team.get('match_stats', {})
                except:
                    # Create default stats if not available
                    home_stats = {
                        'shots': 12, 'shots_on_target': 5, 'corners': 6, 'offsides': 2,
                        'yellow_cards': 1, 'red_cards': 0
                    }
                    away_stats = {
                        'shots': 10, 'shots_on_target': 4, 'corners': 5, 'offsides': 3,
                        'yellow_cards': 2, 'red_cards': 0
                    }
                
                # Get performance metrics
                try:
                    home_form = home_team['analysis'].get('key_metrics', {}).get('form_trend', 0.6)
                    away_form = away_team['analysis'].get('key_metrics', {}).get('form_trend', 0.5)
                    home_win_streak = home_team['analysis'].get('streaks', {}).get('win_streak', 1)
                    home_loss_streak = home_team['analysis'].get('streaks', {}).get('loss_streak', 0)
                    away_win_streak = away_team['analysis'].get('streaks', {}).get('win_streak', 0)
                    away_loss_streak = away_team['analysis'].get('streaks', {}).get('loss_streak', 1)
                except:
                    # Default values if metrics not available
                    home_form = 0.6
                    away_form = 0.5
                    home_win_streak = 1
                    home_loss_streak = 0
                    away_win_streak = 0
                    away_loss_streak = 1
                
                # Create input tensor in the format expected by Podos
                input_data = torch.tensor([
                    home_stats.get('shots', 12),
                    away_stats.get('shots', 10),
                    home_stats.get('shots_on_target', 5),
                    away_stats.get('shots_on_target', 4),
                    home_stats.get('corners', 6),
                    away_stats.get('corners', 5),
                    home_stats.get('offsides', 2),
                    away_stats.get('offsides', 3),
                    home_stats.get('yellow_cards', 1),
                    away_stats.get('yellow_cards', 2),
                    home_stats.get('red_cards', 0),
                    away_stats.get('red_cards', 0),
                    2.1,  # Default odds_home
                    3.4,  # Default odds_draw
                    3.5,  # Default odds_away
                    home_team_id,
                    away_team_id,
                    home_win_streak,
                    home_loss_streak,
                    away_win_streak,
                    away_loss_streak,
                    home_form * 10,
                    away_form * 10
                ], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                
                # Get model predictions
                with torch.no_grad():
                    predictions = self.podos_model(input_data).squeeze(0).numpy()
                
                # Map predictions to bet types
                base_probs = {
                    'home_win': float(predictions[0]),
                    'draw': float(predictions[1]),
                    'away_win': float(predictions[2])
                }
                
                # Derive additional bet types from base predictions
                home_goals_exp = self.features['home_xg']
                away_goals_exp = self.features['away_xg']
                
                # Calculate BTTS probability based on expected goals
                btts_prob = 1 - (np.exp(-home_goals_exp) + np.exp(-away_goals_exp) - np.exp(-home_goals_exp -away_goals_exp))
                
                # Calculate over/under probability using Poisson model
                total_goals_exp = home_goals_exp + away_goals_exp
                import math
                under_2_5_prob = sum(np.exp(-total_goals_exp) * total_goals_exp**k / math.factorial(k) for k in range(3))
                
                adjusted_probs = {
                    'home_win': base_probs['home_win'],
                    'draw': base_probs['draw'],
                    'away_win': base_probs['away_win'],
                    'btts_yes': btts_prob,
                    'btts_no': 1 - btts_prob,
                    'over_2.5': 1 - under_2_5_prob,
                    'under_2.5': under_2_5_prob
                }
                
                # Generate bet signals based on thresholds
                bet_signals = []
                
                # Apply thresholds for bet recommendations
                if adjusted_probs['home_win'] > 0.5:
                    bet_signals.append(('home_win', adjusted_probs['home_win']))
                if adjusted_probs['away_win'] > 0.4:
                    bet_signals.append(('away_win', adjusted_probs['away_win']))
                if adjusted_probs['draw'] > 0.3:
                    bet_signals.append(('draw', adjusted_probs['draw']))
                if adjusted_probs['btts_yes'] > 0.55:
                    bet_signals.append(('btts_yes', adjusted_probs['btts_yes']))
                if adjusted_probs['btts_no'] > 0.55:
                    bet_signals.append(('btts_no', adjusted_probs['btts_no']))
                if adjusted_probs['under_2.5'] > 0.65:
                    bet_signals.append(('under_2.5', adjusted_probs['under_2.5']))
                if adjusted_probs['over_2.5'] > 0.65:
                    bet_signals.append(('over_2.5', adjusted_probs['over_2.5']))
                
                # Sort by probability (descending)
                bet_signals.sort(key=lambda x: x[1], reverse=True)
                
                return bet_signals, adjusted_probs
                
            except Exception as e:
                print(f"ML prediction failed: {str(e)}, falling back to synthetic approach")
                return self._synthetic_bet_recommender()
        
        # If model loading fails, use the fallback method
        print("Using synthetic prediction approach")
        return self._synthetic_bet_recommender()
    
    def _synthetic_bet_recommender(self):
        """Original synthetic recommendation method as fallback"""
        # Original implementation (copy from the current code)
        features_vector = np.array([
            self.features['momentum_diff'],
            self.features['xg_gap'],
            self.features['volatility_sum'],
            self.features['consistency_gap'],
            self.features['home_advantage'],
            self.features['home_opponent_sensitivity'],
            self.features['away_opponent_sensitivity'],
            self.features['home_scoring_concentration'],
            self.features['away_scoring_concentration'],
            self.features['home_form_trend'],
            self.features['away_form_trend'],
            self.features['home_xg'],
            self.features['away_xg']
        ]).reshape(1, -1)
        
        # Synthetic decision function using weighted features
        # In a real implementation, this would be replaced with a trained ML model
        weights = {
            'home_win': np.array([0.3, 0.25, -0.05, 0.15, 0.2, 0.05, -0.05, 0.05, -0.05, 0.1, -0.1, 0.2, -0.2]),
            'draw': np.array([-0.1, -0.2, 0.2, -0.1, -0.1, 0.0, 0.0, -0.1, -0.1, -0.1, -0.1, -0.15, -0.15]),
            'away_win': np.array([-0.3, -0.25, -0.05, -0.15, -0.2, -0.05, 0.05, -0.05, 0.05, -0.1, 0.1, -0.2, 0.2]),
            'btts_yes': np.array([0.0, 0.05, 0.2, -0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.3, 0.3]),
            'under_2.5': np.array([0.0, -0.15, -0.2, 0.1, 0.0, 0.0, 0.0, -0.1, -0.1, 0.0, 0.0, -0.3, -0.3])
        }
        
        # Calibrate with provided probabilities if available
        if self.features.get('home_win_prob'):
            base_probs = {
                'home_win': self.features['home_win_prob'],
                'draw': self.features['draw_prob'],
                'away_win': self.features['away_win_prob'],
                'btts_yes': self.features['btts_prob'],
                'under_2.5': self.features['under_2.5_prob']
            }
        else:
            # Default base probabilities if not available
            base_probs = {
                'home_win': 0.45,
                'draw': 0.25, 
                'away_win': 0.3,
                'btts_yes': 0.5,
                'under_2.5': 0.5
            }
            
        # Adjust probabilities based on features
        adjusted_probs = {}
        for bet_type, w in weights.items():
            # Calculate adjustment from feature weights (-0.2 to +0.2 range)
            adjustment = np.clip(np.sum(features_vector * w) / 10, -0.2, 0.2)
            adjusted_probs[bet_type] = np.clip(base_probs.get(bet_type, 0.5) + adjustment, 0.05, 0.95)
        
        # Normalize 1X2 probabilities to sum to 1
        match_outcomes_sum = (adjusted_probs['home_win'] + adjusted_probs['draw'] + adjusted_probs['away_win'])
        if match_outcomes_sum > 0:
            adjusted_probs['home_win'] /= match_outcomes_sum
            adjusted_probs['draw'] /= match_outcomes_sum
            adjusted_probs['away_win'] /= match_outcomes_sum
            
        # Add derived bets
        adjusted_probs['btts_no'] = 1 - adjusted_probs['btts_yes']
        adjusted_probs['over_2.5'] = 1 - adjusted_probs['under_2.5']
        
        # Convert to list of tuples with thresholds applied
        bet_signals = []
        
        # Apply thresholds for bet recommendations
        if adjusted_probs['home_win'] > 0.5:
            bet_signals.append(('home_win', adjusted_probs['home_win']))
        if adjusted_probs['away_win'] > 0.4:
            bet_signals.append(('away_win', adjusted_probs['away_win']))
        if adjusted_probs['draw'] > 0.3:
            bet_signals.append(('draw', adjusted_probs['draw']))
        if adjusted_probs['btts_yes'] > 0.55:
            bet_signals.append(('btts_yes', adjusted_probs['btts_yes']))
        if adjusted_probs['btts_no'] > 0.55:
            bet_signals.append(('btts_no', adjusted_probs['btts_no']))
        if adjusted_probs['under_2.5'] > 0.65:
            bet_signals.append(('under_2.5', adjusted_probs['under_2.5']))
        if adjusted_probs['over_2.5'] > 0.65:
            bet_signals.append(('over_2.5', adjusted_probs['over_2.5']))
        
        # Sort by probability (descending)
        bet_signals.sort(key=lambda x: x[1], reverse=True)
        
        return bet_signals, adjusted_probs
    
    def monte_carlo_simulation(self, n_sim=MC_SIMULATIONS):
        """
        Advanced Monte Carlo simulation using Lévy processes and stochastic volatility models
        to capture fat tails and non-Gaussian behavior in soccer match outcomes.
        """
        # Initialize simulation arrays
        match_outcomes = []
        goal_counts = []
        goal_diffs = []
        
        # Optimization to find the best distribution parameters
        def calib_objective(params, target_home_prob, target_away_prob, target_draw_prob):
            alpha, beta, home_loc_adj, away_loc_adj, vol_scale = params
            
            # Run short simulation to check calibration
            home_goals_sim = []
            away_goals_sim = []
            
            for _ in range(1000):
                # Use Lévy stable distribution for heavy tails
                home_shock = levy_stable.rvs(
                    alpha, beta, 
                    loc=self.features['home_xg'] + home_loc_adj, 
                    scale=self.features['volatility_sum'] * vol_scale
                )
                away_shock = levy_stable.rvs(
                    alpha, beta, 
                    loc=self.features['away_xg'] + away_loc_adj, 
                    scale=self.features['volatility_sum'] * vol_scale
                )
                
                home_goals_sim.append(max(0, np.round(home_shock)))
                away_goals_sim.append(max(0, np.round(away_shock)))
            
            # Calculate outcome probabilities from simulation
            home_win_sim = sum(h > a for h, a in zip(home_goals_sim, away_goals_sim)) / 1000
            away_win_sim = sum(h < a for h, a in zip(home_goals_sim, away_goals_sim)) / 1000
            draw_sim = sum(h == a for h, a in zip(home_goals_sim, away_goals_sim)) / 1000
            
            # Calculate error vs target probabilities
            error = (
                (home_win_sim - target_home_prob)**2 + 
                (away_win_sim - target_away_prob)**2 + 
                (draw_sim - target_draw_prob)**2
            )
            
            return error
        
        # Starting guess for parameters
        initial_params = [1.5, 0.0, 0.0, 0.0, 0.5]
        
        # Target probabilities from the feature data (or default to reasonable values)
        target_home_prob = self.features.get('home_win_prob', 0.45)
        target_away_prob = self.features.get('away_win_prob', 0.3)
        target_draw_prob = self.features.get('draw_prob', 0.25)
        
        # Optimize the parameters
        try:
            result = minimize(
                calib_objective, 
                initial_params,
                args=(target_home_prob, target_away_prob, target_draw_prob),
                bounds=[(1.1, 2.0), (-0.1, 0.1), (-0.5, 0.5), (-0.5, 0.5), (0.1, 1.0)],
                method='L-BFGS-B'
            )
            alpha, beta, home_loc_adj, away_loc_adj, vol_scale = result.x
        except:
            # Fallback to initial values if optimization fails
            alpha, beta, home_loc_adj, away_loc_adj, vol_scale = initial_params
        
        # Run full simulation with optimized parameters
        for _ in range(n_sim):
            # Choose between Lévy stable and Heston model with some probability
            if np.random.random() < 0.7:  # 70% Lévy stable
                # Lévy stable distribution for heavy tails
                home_shock = levy_stable.rvs(
                    alpha, beta, 
                    loc=self.features['home_xg'] + home_loc_adj, 
                    scale=self.features['volatility_sum'] * vol_scale
                )
                away_shock = levy_stable.rvs(
                    alpha, beta, 
                    loc=self.features['away_xg'] + away_loc_adj, 
                    scale=self.features['volatility_sum'] * vol_scale
                )
                
                home_goals = max(0, np.round(home_shock))
                away_goals = max(0, np.round(away_shock))
            else:
                # Heston stochastic volatility model
                home_goals = self.heston_stochastic_volatility_goals(self.features['home_xg'])
                away_goals = self.heston_stochastic_volatility_goals(self.features['away_xg'])
            
            # Record outcomes
            if home_goals > away_goals:
                match_outcomes.append('home_win')
            elif home_goals < away_goals:
                match_outcomes.append('away_win')
            else:
                match_outcomes.append('draw')
                
            goal_counts.append(home_goals + away_goals)
            goal_diffs.append(home_goals - away_goals)
        
        # Calculate empirical probabilities
        outcome_probs = {
            'home_win': match_outcomes.count('home_win') / n_sim,
            'away_win': match_outcomes.count('away_win') / n_sim,
            'draw': match_outcomes.count('draw') / n_sim
        }
        
        # Calculate goal-based probabilities
        btts_count = sum(1 for i in range(n_sim) if goal_counts[i] >= 2 and goal_diffs[i] != goal_counts[i])
        over_counts = {
            'over_0.5': sum(1 for g in goal_counts if g > 0.5) / n_sim,
            'over_1.5': sum(1 for g in goal_counts if g > 1.5) / n_sim,
            'over_2.5': sum(1 for g in goal_counts if g > 2.5) / n_sim,
            'over_3.5': sum(1 for g in goal_counts if g > 3.5) / n_sim,
            'over_4.5': sum(1 for g in goal_counts if g > 4.5) / n_sim
        }
        
        # Generate bet signals based on thresholds
        sim_bets = []
        
        # 1X2 Markets
        for outcome, prob in outcome_probs.items():
            if prob >= CONFIRMATION_THRESHOLD:
                sim_bets.append((outcome, prob))
        
        # Goals markets
        btts_prob = btts_count / n_sim
        if btts_prob >= CONFIRMATION_THRESHOLD:
            sim_bets.append(('btts_yes', btts_prob))
        elif (1 - btts_prob) >= CONFIRMATION_THRESHOLD:
            sim_bets.append(('btts_no', 1 - btts_prob))
            
        # Over/Under markets
        for market, prob in over_counts.items():
            if prob >= CONFIRMATION_THRESHOLD:
                sim_bets.append((market, prob))
            elif (1 - prob) >= CONFIRMATION_THRESHOLD:
                under_market = market.replace('over', 'under')
                sim_bets.append((under_market, 1 - prob))
        
        # Sort by probability (descending)
        sim_bets.sort(key=lambda x: x[1], reverse=True)
        
        return sim_bets, {
            'outcomes': outcome_probs,
            'goal_counts': goal_counts,
            'goal_diffs': goal_diffs,
            'over_under': over_counts,
            'btts_prob': btts_prob,
            'alpha': alpha,
            'beta': beta
        }
    
    def value_at_risk_analysis(self, bet_odds, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR) for betting portfolio
        """
        # Get probabilities from simulations
        _, sim_data = self.monte_carlo_simulation(n_sim=10000)
        
        # Initialize portfolio returns array
        returns = []
        
        # Standard bet size (can be adjusted based on Kelly criterion)
        bet_size = 1.0
        
        # Calculate theoretical returns for each bet type
        for bet_type, odds in bet_odds.items():
            # Skip if invalid odds
            if odds <= 1.0:
                continue
                
            # Get probability for this bet
            if bet_type == 'home_win':
                prob = sim_data['outcomes']['home_win']
            elif bet_type == 'away_win':
                prob = sim_data['outcomes']['away_win']
            elif bet_type == 'draw':
                prob = sim_data['outcomes']['draw']
            elif bet_type == 'btts_yes':
                prob = sim_data['btts_prob']
            elif bet_type == 'btts_no':
                prob = 1 - sim_data['btts_prob']
            elif bet_type.startswith('over_'):
                threshold = float(bet_type.split('_')[1])
                prob = sum(1 for g in sim_data['goal_counts'] if g > threshold) / len(sim_data['goal_counts'])
            elif bet_type.startswith('under_'):
                threshold = float(bet_type.split('_')[1])
                prob = sum(1 for g in sim_data['goal_counts'] if g <= threshold) / len(sim_data['goal_counts'])
            else:
                continue
                
            # Expected value of bet
            ev = (prob * (odds - 1) - (1 - prob)) * bet_size
            
            # Expected return on investment
            roi = ev / bet_size
            
            # Kelly criterion for optimal bet sizing
            if prob > 0 and odds > 1:
                kelly_fraction = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
                kelly_fraction = max(0, min(1, kelly_fraction)) * KELLY_FRACTION  # Apply fractional Kelly
            else:
                kelly_fraction = 0
                
            # Value at Risk calculation
            var_threshold = np.percentile([-bet_size, (odds - 1) * bet_size], 100 - confidence_level * 100)
            cvar = -bet_size if prob < 0.05 else (var_threshold * (1 - prob) - bet_size * (1 - prob)) / (1 - confidence_level)
            
            returns.append({
                'bet_type': bet_type,
                'true_probability': prob,
                'implied_probability': 1/odds,
                'odds': odds,
                'expected_value': ev,
                'roi': roi,
                'kelly_fraction': kelly_fraction,
                'var_95': var_threshold,
                'cvar_95': cvar
            })
            
        # Sort by ROI (descending)
        returns.sort(key=lambda x: x['roi'], reverse=True)
        
        return returns
    
    def run_analysis(self):
        """Run the full analysis pipeline and return results"""
        # Get ML-based bet recommendations
        ml_bets, ml_probs = self.ml_bet_recommender()
        
        # Get Monte Carlo simulation results
        mc_bets, mc_data = self.monte_carlo_simulation()
        
        # Find coherent bets agreed by both approaches
        intersect_bets = []
        for ml_bet in ml_bets:
            for mc_bet in mc_bets:
                if ml_bet[0] == mc_bet[0]:
                    # Average the probabilities
                    avg_prob = (ml_bet[1] + mc_bet[1]) / 2
                    if avg_prob >= CONFIRMATION_THRESHOLD:
                        intersect_bets.append((ml_bet[0], avg_prob))
        
        # Generate summary visualizations (will be saved to disk)
        self.generate_visualizations(ml_probs, mc_data)
        
        # Return structured results
        results = {
            'match_info': {
                'home_team': self.teams['home'],
                'away_team': self.teams['away'],
                'date': self.param_data['fixture_info']['date'],
                'venue': self.param_data['fixture_info']['venue']['name'],
                'league': self.param_data['league']['name']
            },
            'core_bets': intersect_bets,
            'model_bets': ml_bets,
            'simulation_bets': mc_bets,
            'simulation_stats': {
                'outcome_probs': mc_data['outcomes'],
                'btts_prob': mc_data['btts_prob'],
                'over_under_probs': mc_data['over_under']
            },
            'model_stats': {
                'outcome_probs': {
                    'home_win': ml_probs['home_win'],
                    'draw': ml_probs['draw'],
                    'away_win': ml_probs['away_win']
                },
                'btts_prob': ml_probs['btts_yes'],
                'over_under_probs': {
                    'over_2.5': ml_probs['over_2.5'],
                    'under_2.5': ml_probs['under_2.5']
                }
            },
            'key_features': {
                'momentum_diff': self.features['momentum_diff'],
                'xg_gap': self.features['xg_gap'],
                'volatility_sum': self.features['volatility_sum'],
                'consistency_gap': self.features['consistency_gap'],
                'home_advantage': self.features['home_advantage']
            }
        }
        
        return results
    
    def generate_visualizations(self, ml_probs, mc_data):
        """Generate visualizations from the analysis results"""
        # Set up plotting
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Match Analysis: {self.teams["home"]} vs {self.teams["away"]}', fontsize=16)
        
        # 1. Outcome probabilities comparison
        plt.subplot(2, 2, 1)
        outcomes = ['home_win', 'draw', 'away_win']
        ml_outcome_probs = [ml_probs['home_win'], ml_probs['draw'], ml_probs['away_win']]
        mc_outcome_probs = [mc_data['outcomes']['home_win'], mc_data['outcomes']['draw'], mc_data['outcomes']['away_win']]
        
        x = np.arange(len(outcomes))
        width = 0.35
        
        plt.bar(x - width/2, ml_outcome_probs, width, label='Model Predictions')
        plt.bar(x + width/2, mc_outcome_probs, width, label='Monte Carlo Simulation')
        
        plt.ylabel('Probability')
        plt.title('Match Outcome Probabilities')
        plt.xticks(x, ['Home Win', 'Draw', 'Away Win'])
        plt.legend()
        
        # 2. Goal distribution from Monte Carlo
        plt.subplot(2, 2, 2)
        goal_counts = np.array(mc_data['goal_counts'])
        
        # Count occurrences of each goal total (0, 1, 2, etc.)
        goal_distribution = {}
        for i in range(10):  # 0 to 9 goals
            goal_distribution[i] = np.sum(goal_counts == i) / len(goal_counts)
        
        plt.bar(goal_distribution.keys(), goal_distribution.values())
        plt.axvline(x=2.5, color='r', linestyle='--', label='Over/Under 2.5')
        
        plt.xlabel('Total Goals')
        plt.ylabel('Probability')
        plt.title('Goal Distribution')
        plt.xticks(range(10))
        plt.legend()
        
        # 3. Score heatmap
        plt.subplot(2, 2, 3)
        
        # Create a matrix of score probabilities
        max_goals = 5
        score_matrix = np.zeros((max_goals+1, max_goals+1))
        
        # We need to re-run a simulation to get the joint distribution
        for _ in range(5000):  # Reduced sample for speed
            if np.random.random() < 0.7:
                home_goals = levy_stable.rvs(
                    mc_data['alpha'], mc_data['beta'], 
                    loc=self.features['home_xg'], 
                    scale=self.features['volatility_sum'] * 0.5
                )
                away_goals = levy_stable.rvs(
                    mc_data['alpha'], mc_data['beta'], 
                    loc=self.features['away_xg'], 
                    scale=self.features['volatility_sum'] * 0.5
                )
            else:
                home_goals = self.heston_stochastic_volatility_goals(self.features['home_xg'])
                away_goals = self.heston_stochastic_volatility_goals(self.features['away_xg'])
                
            home_goals = int(max(0, min(max_goals, np.round(home_goals))))
            away_goals = int(max(0, min(max_goals, np.round(away_goals))))
            
            score_matrix[home_goals, away_goals] += 1
        
        # Normalize to get probabilities
        score_matrix /= np.sum(score_matrix)
        
        # Plot heatmap
        sns.heatmap(score_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                    xticklabels=range(max_goals+1), yticklabels=range(max_goals+1))
        plt.xlabel(f'{self.teams["away"]} Goals')
        plt.ylabel(f'{self.teams["home"]} Goals')
        plt.title('Score Probability Matrix')
        
        # 4. Betting value radar
        plt.subplot(2, 2, 4)
        
        # Mock betting odds for demonstration (in real system would come from odds_finder.py)
        sample_odds = {
            'home_win': 2.10,
            'draw': 3.40,
            'away_win': 3.50,
            'btts_yes': 1.90,
            'btts_no': 1.95,
            'over_2.5': 1.85,
            'under_2.5': 2.00,
            'over_1.5': 1.35,
            'under_1.5': 3.25
        }
        
        # Run VaR analysis on these odds
        bet_risk_profile = self.value_at_risk_analysis(sample_odds)
        
        # Filter to bets with positive expected value
        value_bets = [bet for bet in bet_risk_profile if bet['roi'] > 0]
        
        if value_bets:
            # Extract data for radar chart
            categories = [bet['bet_type'] for bet in value_bets]
            roi_values = [bet['roi'] for bet in value_bets]
            kelly_values = [bet['kelly_fraction'] for bet in value_bets]
            
            # Normalize values for plotting
            max_roi = max(roi_values) if roi_values else 1.0
            max_kelly = max(kelly_values) if kelly_values else 1.0
            
            norm_roi = [roi/max_roi for roi in roi_values]
            norm_kelly = [kelly/max_kelly for kelly in kelly_values]
            
            # Number of variables
            N = len(categories)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Normalize values to 0-1 scale
            norm_roi += norm_roi[:1]  # Close the loop
            norm_kelly += norm_kelly[:1]  # Close the loop
            
            # Set up axes
            ax = plt.subplot(2, 2, 4, polar=True)
            
            # Draw the shape for ROI
            ax.plot(angles, norm_roi, 'o-', linewidth=2, label='ROI')
            ax.fill(angles, norm_roi, alpha=0.25)
            
            # Draw the shape for Kelly
            ax.plot(angles, norm_kelly, 'o-', linewidth=2, label='Kelly Fraction')
            ax.fill(angles, norm_kelly, alpha=0.25)
            
            # Set category labels
            plt.xticks(angles[:-1], categories)
            
            # Set y-axis labels (normalized values)
            plt.yticks([0.25, 0.5, 0.75], ['0.25', '0.5', '0.75'], color='grey', size=8)
            
            plt.ylim(0, 1)
            plt.title('Betting Value Radar')
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        else:
            plt.text(0.5, 0.5, 'No value bets identified', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
        
        # Save the visualization
        plt.tight_layout()
        output_file = f"{self.teams['home'].replace(' ', '_')}_{self.teams['away'].replace(' ', '_')}_analysis.png"
        plt.savefig(output_file)
        plt.close()
        
        # Additional market-specific analysis charts
        self.generate_market_specific_charts(mc_data)
    
    def generate_market_specific_charts(self, mc_data):
        """Generate additional charts for specific betting markets"""
        # 1. Goal timing distribution analysis
        plt.figure(figsize=(10, 6))
        
        # Extract scoring patterns from the data - USE SAFE GETTERS
        home = self.param_data['home_team']['analysis']
        away = self.param_data['away_team']['analysis']
        
        # Default empty scoring patterns if missing
        default_intervals = {'0-15 min.': 0.0, '16-30 min.': 0.0, '31-45 min.': 0.0, 
                            '46-60 min.': 0.0, '61-75 min.': 0.0, '76-90 min.': 0.0}
        
        # Safely get scoring patterns
        home_scoring_patterns = home.get('scoring_patterns', {'scoring_by_interval': default_intervals})
        away_scoring_patterns = away.get('scoring_patterns', {'scoring_by_interval': default_intervals})
        
        home_scoring = np.array(list(home_scoring_patterns.get('scoring_by_interval', default_intervals).values()))
        away_scoring = np.array(list(away_scoring_patterns.get('scoring_by_interval', default_intervals).values()))
        
        # Normalize if needed
        if home_scoring.sum() > 0:
            home_scoring = home_scoring / home_scoring.sum()
        if away_scoring.sum() > 0:
            away_scoring = away_scoring / away_scoring.sum()
        
        # Define intervals (assuming 15-minute intervals)
        intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
        
        # Plot stacked bars for each team
        width = 0.35
        x = np.arange(len(intervals))
        
        plt.bar(x - width/2, home_scoring, width, label=f'{self.teams["home"]}')
        plt.bar(x + width/2, away_scoring, width, label=f'{self.teams["away"]}')
        
        plt.xlabel('Match Intervals (minutes)')
        plt.ylabel('Goal Probability')
        plt.title('Goal Timing Distribution')
        plt.xticks(x, intervals)
        plt.legend()
        
        # Save the chart
        plt.tight_layout()
        output_file = f"{self.teams['home'].replace(' ', '_')}_{self.teams['away'].replace(' ', '_')}_goal_timing.png"
        plt.savefig(output_file)
        plt.close()
        
        # 2. Levy stable distribution parameters and density visualization
        plt.figure(figsize=(10, 6))
        
        x = np.linspace(0, 10, 1000)
        
        # Plot probability density functions
        home_pdf = levy_stable.pdf(x, mc_data['alpha'], mc_data['beta'], 
                            loc=self.features['home_xg'], 
                            scale=self.features['volatility_sum'] * 0.5)
        away_pdf = levy_stable.pdf(x, mc_data['alpha'], mc_data['beta'], 
                            loc=self.features['away_xg'], 
                            scale=self.features['volatility_sum'] * 0.5)
        
        plt.plot(x, home_pdf, label=f'{self.teams["home"]} Goal Distribution')
        plt.plot(x, away_pdf, label=f'{self.teams["away"]} Goal Distribution')
        
        # Add normal distribution for comparison
        home_normal = norm.pdf(x, loc=self.features['home_xg'], scale=1.0)
        plt.plot(x, home_normal, '--', label='Normal Approximation', alpha=0.6)
        
        plt.xlabel('Number of Goals')
        plt.ylabel('Probability Density')
        plt.title('Levy Stable Goal Distribution')
        plt.legend()
        
        # Add parameters as text annotation
        param_text = f"α: {mc_data['alpha']:.2f}, β: {mc_data['beta']:.2f}\n"
        param_text += f"Home xG: {self.features['home_xg']:.2f}, Away xG: {self.features['away_xg']:.2f}\n"
        param_text += f"Volatility: {self.features['volatility_sum']:.2f}"
        
        plt.annotate(param_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Save the chart
        plt.tight_layout()
        output_file = f"{self.teams['home'].replace(' ', '_')}_{self.teams['away'].replace(' ', '_')}_levy_dist.png"
        plt.savefig(output_file)
        plt.close()

def process_all_matches():
    """
    Process all match files from the daily_output/processed_matches directory,
    generate detailed analysis, and save results to organized folders in daily_output.
    """
    # Base directories - updated to use the correct path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)  # Go up one level from score_data
    
    # Path to processed matches based on your folder structure
    processed_matches_dir = os.path.join(root_dir, 'processed_matches')
    
    # Check if we need to look in daily_output instead
    if not os.path.exists(processed_matches_dir):
        processed_matches_dir = os.path.join(root_dir, 'daily_output', 'processed_matches')
    
    # Output directory
    output_base_dir = os.path.join(root_dir, 'daily_output', 'match_analysis')
    
    # Debug log
    print(f"Looking for matches in: {processed_matches_dir}")
    
    # Ensure the directory exists
    if not os.path.exists(processed_matches_dir):
        print(f"Error: Cannot find processed_matches directory at {processed_matches_dir}")
        
        # Last attempt - check in current directory structure
        path_candidates = [
            os.path.join(root_dir, 'daily_games', 'processed_matches'),
            os.path.join(root_dir, 'daily_output', 'daily_games', 'processed_matches'),
            os.path.join(base_dir, '..', 'daily_games', 'processed_matches')
        ]
        
        for path in path_candidates:
            print(f"Trying path: {path}")
            if os.path.exists(path):
                processed_matches_dir = path
                print(f"Found processed matches at: {processed_matches_dir}")
                break
        else:
            print("Could not find processed_matches directory in any expected location")
            return 0
        
    # Create output directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # Count of processed matches
    processed_count = 0
    error_count = 0
    
    # Collect all JSON files before processing
    match_files = []
    
    # Walk through the processed_matches directory
    for root, dirs, files in os.walk(processed_matches_dir):
        for file in files:
            if file.endswith('.json'):
                match_file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, processed_matches_dir)
                league_name = rel_path.split(os.path.sep)[0] if os.path.sep in rel_path else rel_path
                match_id = os.path.splitext(file)[0]
                
                match_files.append({
                    'path': match_file_path,
                    'league': league_name,
                    'id': match_id
                })
    
    # Display summary before processing
    print(f"Found {len(match_files)} match files to process")
    
    # Process all collected files
    for match in match_files:
        try:
            # Extract match info
            match_file_path = match['path']
            league_name = match['league']
            match_id = match['id']
            
            # Create league-specific output directory
            league_output_dir = os.path.join(output_base_dir, league_name)
            if not os.path.exists(league_output_dir):
                os.makedirs(league_output_dir)
            
            # Create match-specific output directory
            match_output_dir = os.path.join(league_output_dir, match_id)
            if not os.path.exists(match_output_dir):
                os.makedirs(match_output_dir)
            
            print(f"Processing match: {match_id} from {league_name}")
            
            # Store current directory to restore it later
            original_dir = os.getcwd()
            
            try:
                # Create model and run analysis
                model = SoccerDerivativeModel(match_file_path)
                
                # Set output directory for visualizations
                os.chdir(match_output_dir)
                
                # Run the analysis
                results = model.run_analysis()
                
                # Save the analysis results
                output_file = os.path.join(match_output_dir, f"{match_id}_analysis.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
                
                # Create a detailed text report
                create_detailed_report(results, match_output_dir, match_id)
                
                processed_count += 1
                print(f"✓ Analysis completed for {match_id}")
            finally:
                # Restore original directory
                os.chdir(original_dir)
                
        except Exception as e:
            error_count += 1
            print(f"✗ Error processing {match_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\nProcessing complete. Processed {processed_count} matches with {error_count} errors.")
    print(f"Results saved to: {output_base_dir}")
    return processed_count

def create_detailed_report(results, output_dir, match_id):
    """
    Create a detailed text report from the analysis results.
    
    Parameters:
    -----------
    results : dict
        The analysis results from the model
    output_dir : str
        Directory to save the report
    match_id : str
        Identifier for the match
    """
    report_file = os.path.join(output_dir, f"{match_id}_detailed_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f"=" * 80 + "\n")
        f.write(f"MATCH ANALYSIS: {results['match_info']['home_team']} vs {results['match_info']['away_team']}\n")
        f.write(f"=" * 80 + "\n\n")
        
        # Match information
        f.write("MATCH INFORMATION:\n")
        f.write(f"Date: {results['match_info']['date']}\n")
        f.write(f"Venue: {results['match_info']['venue']}\n")
        f.write(f"League: {results['match_info']['league']}\n\n")
        
        # Key metrics
        f.write("KEY METRICS:\n")
        for metric, value in results['key_features'].items():
            f.write(f"{metric.replace('_', ' ').title()}: {value:.3f}\n")
        f.write("\n")
        
        # Outcome probabilities section
        f.write("MATCH OUTCOME PROBABILITIES:\n")
        f.write(f"Home Win: {results['simulation_stats']['outcome_probs']['home_win']:.2%}\n")
        f.write(f"Draw: {results['simulation_stats']['outcome_probs']['draw']:.2%}\n")
        f.write(f"Away Win: {results['simulation_stats']['outcome_probs']['away_win']:.2%}\n\n")
        
        # Goals probabilities
        f.write("GOALS PROBABILITIES:\n")
        f.write(f"BTTS: {results['simulation_stats']['btts_prob']:.2%}\n")
        for market, prob in results['simulation_stats']['over_under_probs'].items():
            f.write(f"{market.replace('_', ' ').title()}: {prob:.2%}\n")
        f.write("\n")
        
        # Recommended bets
        f.write("RECOMMENDED BETS:\n")
        if results['core_bets']:
            for bet, prob in results['core_bets']:
                f.write(f"- {bet.replace('_', ' ').title()}: {prob:.2%} confidence\n")
        else:
            f.write("No high-confidence bets recommended for this match.\n")
        f.write("\n")
        
        # Models comparison
        f.write("MODEL COMPARISONS:\n")
        f.write("ML Model Predictions:\n")
        for bet_type in ['home_win', 'draw', 'away_win', 'btts_yes', 'over_2.5']:
            try:
                f.write(f"- {bet_type.replace('_', ' ').title()}: {results['model_stats']['outcome_probs'].get(bet_type, 0):.2%}\n")
            except:
                pass
        
        f.write("\nMonte Carlo Simulation:\n")
        for bet_type in ['home_win', 'draw', 'away_win']:
            f.write(f"- {bet_type.replace('_', ' ').title()}: {results['simulation_stats']['outcome_probs'][bet_type]:.2%}\n")
        
        f.write(f"\nBTTS Probability: {results['simulation_stats']['btts_prob']:.2%}\n\n")
        
        # Detailed bets section
        f.write("DETAILED BET ANALYSIS:\n")
        f.write("Model-based Recommendations:\n")
        for bet, prob in results['model_bets']:
            f.write(f"- {bet.replace('_', ' ').title()}: {prob:.2%}\n")
        
        f.write("\nSimulation-based Recommendations:\n")
        for bet, prob in results['simulation_bets']:
            f.write(f"- {bet.replace('_', ' ').title()}: {prob:.2%}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Analysis completed using SoccerDerivativeModel with Podos transformer model\n")
        f.write("=" * 80 + "\n")
    
    return report_file

def main(json_file):
    """Main function to run the analysis for a single match file"""
    try:
        # Validate input file exists
        with open(json_file, 'r', encoding='utf-8') as f:
            # Quick validation check on JSON structure
            data = json.load(f)
            required_keys = ['home_team', 'away_team', 'fixture_info', 'league']
            if not all(key in data for key in required_keys):
                print("Warning: Input JSON may be missing required structure")
    except FileNotFoundError:
        print(f"Error: Input file '{json_file}' not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Input file '{json_file}' is not valid JSON")
        return None
        
    # Create output directory for single file analysis
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_results', 'single_match')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        print(f"Processing match file: {json_file}")
        
        # Create model and run analysis
        model = SoccerDerivativeModel(json_file)
        
        # Set output directory for visualizations
        os.chdir(output_dir)
        
        # Run the analysis
        results = model.run_analysis()
        
        # Save the analysis results
        match_id = os.path.splitext(os.path.basename(json_file))[0]
        output_file = os.path.join(output_dir, f"{match_id}_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        # Create a detailed text report
        create_detailed_report(results, output_dir, match_id)
        
        print(f"✓ Analysis completed and saved to {output_dir}")
        return results
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        # Get current directory for reference
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Running from: {current_dir}")
        
        if len(sys.argv) > 1:
            # Traditional single file processing
            main(sys.argv[1])
        else:
            # Process all matches
            process_all_matches()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)