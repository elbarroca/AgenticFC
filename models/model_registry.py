# models/model_registry.py

from typing import Dict, Type, Optional, List
import logging

# --- Import all specific model classes ---
# Use relative imports assuming all model files are in the 'models' directory
try:
    from .base_model import BaseModel # Base class (optional dependency for type hinting)
    from .poisson_model import PoissonModel
    from .bayesian_model import BayesianModel
    from .markov_model import MarkovModel
    from .random_forest_model import RandomForestModel
    from .gradient_boosting_model import GradientBoostingModel # Assuming XGBoost implementation
    from .lstm_model import LSTMModel
    # MonteCarlo might not inherit from BaseModel if it doesn't 'fit' data
    from .monte_carlo_model import MonteCarloModel
    # EnsembleModel is usually used differently (wraps other models)
    # from .ensemble_model import EnsembleModel

    # --- Define the Model Registry ---
    # Maps string names to the actual model classes
    MODEL_REGISTRY: Dict[str, Type[BaseModel] | Type] = {
        # Models inheriting from BaseModel (or with compatible interface)
        "poisson": PoissonModel,
        "bayesian": BayesianModel,
        "markov": MarkovModel, # MarkovModel for form transitions was adapted to BaseModel
        "random_forest": RandomForestModel,
        "gradient_boosting": GradientBoostingModel, # Assumes the XGBoost one
        "lstm": LSTMModel,

        # Models that might have a different interface (like MonteCarlo)
        "monte_carlo": MonteCarloModel,

        # Ensemble model is typically constructed with other models, not loaded directly via registry often
        # "ensemble": EnsembleModel,
    }
    logging.info(f"Model registry initialized with keys: {list(MODEL_REGISTRY.keys())}")

except ImportError as e:
    logging.error(f"Error importing model classes for registry: {e}. Ensure all model files exist in the 'models' directory.")
    # Define an empty registry or raise error if imports fail critically
    MODEL_REGISTRY = {}
    # raise ImportError(f"Could not import model classes: {e}") from e


def get_model_class(model_name: str) -> Optional[Type[BaseModel] | Type]:
    """
    Retrieves a model class from the registry based on its string name.

    Args:
        model_name (str): The registered name of the model (e.g., 'random_forest').

    Returns:
        Optional[Type[BaseModel] | Type]: The corresponding model class, or None if not found.
    """
    model_name_lower = model_name.lower()
    model_class = MODEL_REGISTRY.get(model_name_lower)
    if model_class is None:
        logging.warning(f"Model '{model_name}' not found in the registry.")
    return model_class

def list_available_models() -> List[str]:
    """Returns a list of names of the models available in the registry."""
    return list(MODEL_REGISTRY.keys())

# Example Usage
if __name__ == '__main__':
    print("\n--- Model Registry Example ---")
    available = list_available_models()
    print(f"Available models: {available}")

    model_name_to_get = "random_forest"
    RFModelClass = get_model_class(model_name_to_get)

    if RFModelClass:
        print(f"\nRetrieved class for '{model_name_to_get}': {RFModelClass}")
        # You can now instantiate it (potentially using config)
        # try:
        #     # Example instantiation (requires config or default params)
        #     # import config
        #     # rf_instance = RFModelClass(**config.RF_PARAMS)
        #     rf_instance = RFModelClass() # Using defaults if defined
        #     print(f"Instantiated: {rf_instance}")
        # except Exception as e:
        #     print(f"Could not instantiate {RFModelClass.__name__}: {e}")
    else:
        print(f"\nCould not retrieve class for '{model_name_to_get}'.")

    model_name_to_get = "non_existent_model"
    NonExistentClass = get_model_class(model_name_to_get)
    if NonExistentClass is None:
        print(f"\nCorrectly handled non-existent model: '{model_name_to_get}'")