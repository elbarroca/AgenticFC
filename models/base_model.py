# models/base_model.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """
    Abstract Base Class for all trainable prediction models.
    """

    def __init__(self, **kwargs):
        """Initialize base model."""
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """Train/configure the model."""
        pass

    @abstractmethod 
    def predict(self, data: Any, **kwargs) -> Any:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise RuntimeError(f"This {self.__class__.__name__} instance is not fitted yet.")
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save model state to file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """Load model state from file."""
        pass

    def __repr__(self) -> str:
        """String representation."""
        # Add more descriptive info for Bayesian model
        if self.__class__.__name__ == 'BayesianUpdateModel':
            if self.is_fitted:
                return (f"{self.__class__.__name__}("
                       f"is_fitted={self.is_fitted}, "
                       f"n_evidence_types={len(getattr(self, 'likelihoods', {}))}, "
                       f"baseline_priors={getattr(self, 'baseline_priors', None)})")
        return f"{self.__class__.__name__}(is_fitted={self.is_fitted})"