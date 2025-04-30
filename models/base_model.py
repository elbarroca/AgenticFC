# models/base_model.py

# Base model interface for all prediction models in the project.
# This file should be placed directly inside the models/ directory.
#
# The BaseModel class uses abc.ABC and abc.abstractmethod to define an interface
# that all model implementations must follow. This ensures consistency across
# different model types.
#
# Models that should inherit from BaseModel:
# - RandomForestModel
# - GradientBoostingModel  
# - PoissonModel
# - MarkovModel
# - LSTMModel
# - BayesianModel
#
# Example:
#   class RandomForestModel(BaseModel):
#       ...
#
# Note: MonteCarloModel may not need to inherit from BaseModel if it doesn't 
# require a meaningful fit() method, since it primarily simulates based on 
# external parameters. The model registry can still include it.
#
# Key features:
# - predict() method includes a check for self.is_fitted
# - Subclasses should either call super().predict() or implement their own check
# - load() is defined as an abstract @classmethod for consistent model loading

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """
    Abstract Base Class for all trainable prediction models.

    Defines a common interface for fitting, predicting, saving, and loading models,
    promoting consistency across different model implementations within the project.

    Subclasses must implement the abstract methods.
    """

    def __init__(self, **kwargs):
        """
        Base constructor. Can accept arbitrary keyword arguments if needed by subclasses,
        but doesn't enforce any specific parameters at this level.
        """
        self.is_fitted: bool = False # Track if the model has been fitted

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """
        Train or configure the model based on input data.

        The exact signature will vary depending on the model type.
        Examples:
        - fit(self, X_train: pd.DataFrame, y_train: pd.Series)
        - fit(self, history_df: pd.DataFrame)
        - fit(self, match_history: List[Dict])

        Implementations should set `self.is_fitted = True` upon successful completion.
        """
        pass

    @abstractmethod
    def predict(self, data: Any, **kwargs) -> Any:
        """
        Make predictions based on new input data.

        Args:
            data (Any): Input data for prediction. The format depends on the model
                        (e.g., pd.DataFrame, Dict, np.ndarray).
            **kwargs: Additional arguments specific to the prediction method.

        Returns:
            Any: The prediction results, typically a pd.DataFrame or np.ndarray.

        Raises:
            NotFittedError: If the model has not been fitted before calling predict.
        """
        if not self.is_fitted:
             # Consider using sklearn.exceptions.NotFittedError if scikit-learn is a dependency
             raise RuntimeError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate data before using 'predict'.")
        # The actual prediction logic is implemented in subclasses
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Saves the trained model state to a file.

        Args:
            filepath (str): The path (including filename) where the model state should be saved.
                            The exact format (e.g., .joblib, .h5, .json) depends on the model type.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        # The actual saving logic is implemented in subclasses
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        Loads a trained model state from a file.

        Args:
            filepath (str): The path to the saved model file.

        Returns:
            BaseModel: An instance of the specific model class with the loaded state.
                       The returned object should have `is_fitted = True`.
        """
        # The actual loading logic is implemented in subclasses
        pass

    def __repr__(self) -> str:
        """Provides a basic string representation of the model instance."""
        return f"{self.__class__.__name__}(is_fitted={self.is_fitted})"