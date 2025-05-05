# models/base_model.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, List, TypeVar, Type
import joblib # Default saving mechanism
# Import BaseFeatureConfig to check type during load
from models.utils.features import BaseFeatureConfig
from sklearn.preprocessing import StandardScaler
import warnings

# Generic type for the model class itself
T = TypeVar('T', bound='BaseModel')

class BaseModel(ABC):
    """Abstract Base Class for all prediction models."""

    def __init__(self, model_params: Dict[str, Any]):
        """
        Initializes the base model. Subclasses MUST call this and then
        set self.feature_config appropriately.

        Args:
            model_params: Dictionary of hyperparameters for the specific model.
        """
        assert isinstance(model_params, dict), "model_params must be a dictionary."
        self.params = model_params
        self._model = None # Placeholder for the actual trained model object(s)
        self.features_in_: List[str] = [] # Stores feature names used during training
        # Feature config MUST be set by the subclass's __init__ method
        self.feature_config: BaseFeatureConfig | None = None
        self.scaler: StandardScaler | None = None

    @abstractmethod
    def _fit_model(self, X_scaled: pd.DataFrame, y: pd.DataFrame):
        """
        Subclass-specific fitting logic using scaled data.

        Args:
            X_scaled: Scaled DataFrame of input features.
            y: DataFrame of target variables.
        """
        pass # Subclass implements this

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Scales features, trains the model using the provided features and target variables.

        Args:
            X: DataFrame of input features.
            y: DataFrame of target variables (e.g., FTHG, FTAG).
               Subclasses should specify exact requirements.
        """
        # --- Base Assertions ---
        assert isinstance(X, pd.DataFrame), "Input X must be a pandas DataFrame."
        assert isinstance(y, pd.DataFrame), "Input y must be a pandas DataFrame."
        assert not X.empty, "Input feature DataFrame X is empty."
        assert not y.empty, "Input target DataFrame y is empty."
        assert X.shape[0] == y.shape[0], f"Row mismatch between X ({X.shape[0]}) and y ({y.shape[0]})."
        assert not X.isnull().any().any(), "Input features X contain NaN values. Handle them before fitting."

        # --- Check if subclass __init__ correctly set feature_config BEFORE storing features ---
        assert self.feature_config is not None, "Subclass __init__ must set self.feature_config before calling fit()."
        assert isinstance(self.feature_config, BaseFeatureConfig), "self.feature_config is not a BaseFeatureConfig instance."

        # Store feature names used for training - crucial for prediction consistency
        self.features_in_ = X.columns.tolist()
        assert self.features_in_, "Feature list is empty after assignment."
        print(f"Training model using {len(self.features_in_)} features.")

        # --- Feature Scaling ---
        print("Fitting StandardScaler and scaling features...")
        self.scaler = StandardScaler()
        X_scaled_np = self.scaler.fit_transform(X[self.features_in_])
        # Keep as DataFrame with correct columns for potential column-specific logic in subclasses
        X_scaled = pd.DataFrame(X_scaled_np, index=X.index, columns=self.features_in_)
        print("Scaling complete.")

        # --- Call Subclass Fitting Logic ---
        # Subclass implements the actual fitting logic using scaled data
        self._fit_model(X_scaled, y)

    @abstractmethod
    def _predict_proba_model(self, X_scaled: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Subclass-specific prediction logic using scaled data.

        Args:
            X_scaled: Scaled DataFrame of input features for prediction.

        Returns:
            Dictionary containing numpy arrays of probabilities/expectations.
        """
        pass # Subclass implements this

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Scales features and predicts base probabilities for new data.

        Args:
            X: DataFrame of input features for prediction. Must contain the same
               columns used during training, in any order.

        Returns:
            Dictionary containing numpy arrays of probabilities/expectations.
            Subclasses define the exact keys.
        """
        # --- Assertions ---
        assert self._model is not None, "Model must be fitted before calling predict_proba."
        assert self.scaler is not None, "Scaler must be fitted before calling predict_proba (model likely not fitted)."
        assert isinstance(X, pd.DataFrame), "Input X must be a pandas DataFrame."
        assert not X.empty, "Input feature DataFrame X for prediction is empty."
        assert not X.isnull().any().any(), "Input features X for prediction contain NaN values."
        # Assert feature_config exists before using features_in_
        assert self.feature_config is not None, "self.feature_config not set. Model likely not fitted/loaded correctly."
        assert self.features_in_, "self.features_in_ not set. Model likely not fitted/loaded correctly."

        # Check feature consistency
        input_features = set(X.columns)
        trained_features = set(self.features_in_)
        assert trained_features.issubset(input_features), \
            f"Feature mismatch: Input data is missing features the model was trained on.\n" \
            f"Missing: {trained_features - input_features}"

        # Select and reorder columns in prediction input to match training order
        try:
            X_ordered = X[self.features_in_]
        except KeyError as e:
             print(f"CRITICAL ERROR: Could not select all required features from prediction data. Missing: {e}")
             raise

        # --- Feature Scaling for Prediction ---
        print("Scaling prediction features using the fitted scaler...")
        X_scaled_np = self.scaler.transform(X_ordered)
        X_scaled = pd.DataFrame(X_scaled_np, index=X_ordered.index, columns=self.features_in_)
        print("Scaling complete.")

        # --- Call Subclass Prediction Logic ---
        # Subclass implements the actual prediction logic using scaled data
        return self._predict_proba_model(X_scaled)

    def save(self, filepath: str):
        """
        Saves the trained model object, parameters, feature list, scaler,
        and the full feature_config object.

        Args:
            filepath: Path to save the model file (e.g., 'saved_models/poisson_v1.joblib').
        """
        assert self._model is not None, "Attempting to save an unfitted model."
        assert self.features_in_, "Attempting to save a model with no features recorded."
        assert self.scaler is not None, "Attempting to save a model without a fitted scaler."
        # --- Assert feature_config exists and is correct type before saving ---
        assert self.feature_config is not None, "Attempting to save a model without feature_config set."
        assert isinstance(self.feature_config, BaseFeatureConfig), \
               f"self.feature_config is not a BaseFeatureConfig instance (Type: {type(self.feature_config)})."

        print(f"Saving model components to {filepath}...")
        try:
            # --- Bundle all necessary components, including the feature_config and scaler ---
            save_data = {
                'model_object': self._model,         # The trained model (e.g., sklearn estimator)
                'model_params': self.params,         # Hyperparameters used
                'features_in': self.features_in_,    # List of feature names
                'feature_config': self.feature_config, # The actual config object
                'scaler': self.scaler                # <-- Save the scaler object
            }
            joblib.dump(save_data, filepath)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")
            raise # Re-raise the exception to fail loudly

    @classmethod
    def load(cls: Type[T], filepath: str) -> T:
        """
        Loads a trained model, parameters, feature list, scaler, and the
        feature_config object from a file. It then uses the loaded parameters
        and feature_config to re-instantiate the model class.

        Args:
            filepath: Path to the saved model file.

        Returns:
            An instance of the specific model class (e.g., PoissonModel) with loaded state.
        """
        print(f"Loading model components from {filepath}...")
        try:
            loaded_data = joblib.load(filepath)

            # --- Assertions on loaded data structure and types ---
            assert isinstance(loaded_data, dict), "Loaded file did not contain a dictionary."
            required_keys = {'model_object', 'model_params', 'features_in', 'feature_config', 'scaler'}
            assert required_keys.issubset(loaded_data.keys()), \
                   f"Loaded file is missing required keys: {required_keys - set(loaded_data.keys())}"

            assert loaded_data['model_object'] is not None, "Loaded 'model_object' is None."
            assert isinstance(loaded_data['model_params'], dict), "'model_params' is not a dictionary."
            assert isinstance(loaded_data['features_in'], list), "'features_in' is not a list."
            assert isinstance(loaded_data['scaler'], StandardScaler), "'scaler' is not a StandardScaler instance."
            # --- Assert the loaded feature_config is the correct type ---
            assert isinstance(loaded_data['feature_config'], BaseFeatureConfig), \
                   f"Loaded 'feature_config' is not a BaseFeatureConfig instance (Type: {type(loaded_data['feature_config'])})."

            # Extract components
            model_object = loaded_data['model_object']
            model_params = loaded_data['model_params']
            features_in = loaded_data['features_in']
            feature_config = loaded_data['feature_config'] # The actual loaded config object
            scaler = loaded_data['scaler']                 # <-- Load the scaler object

            # --- Re-instantiate the correct model class (e.g., PoissonModel) ---
            # Pass the loaded params and the essential feature_config object
            instance = cls(model_params=model_params, feature_config=feature_config)

            # --- Restore the state ---
            instance._model = model_object
            instance.features_in_ = features_in
            instance.scaler = scaler # <-- Restore the scaler
            # instance.feature_config is already set correctly by the __init__ call above

            print(f"Model loaded successfully from {filepath}. Trained on {len(instance.features_in_)} features.")
            # Add info about the loaded feature config for verification
            print(f"Loaded with FeatureConfig: {type(instance.feature_config).__name__} (Include Odds: {getattr(instance.feature_config, 'include_odds', 'N/A')})")
            return instance

        except FileNotFoundError:
            print(f"Error: Model file not found at {filepath}")
            raise
        except Exception as e:
            # Add filepath to the error message for better debugging
            print(f"Error loading model from {filepath}: {e}")
            # Consider logging the traceback here if errors are hard to diagnose
            # import traceback
            # traceback.print_exc()
            raise # Re-raise to fail loudly