# models/model_registry.py
"""
Registry of available prediction models.
"""

from typing import Dict, Type, Optional, List, Any
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Define BaseModel type for reference (even if not imported)
class BaseModelType:
    """Placeholder for BaseModel type if import fails"""
    pass

# Import BaseModel for type checking
try:
    # Try absolute import first
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from base_model import BaseModel
    BaseModelType = BaseModel
    logging.info("Successfully imported BaseModel")
except ImportError as e:
    logging.warning(f"Could not import BaseModel: {e}")
    # BaseModelType remains the placeholder class

# Dictionary to store registered models
MODEL_REGISTRY = {}

# First, try to import the Monte Carlo model
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wrappers'))
    from wrappers.monte_carlo import MonteCarloModel
    MODEL_REGISTRY['monte_carlo'] = MonteCarloModel
    logging.info("Successfully registered MonteCarloModel")
except ImportError as e:
    logging.warning(f"Could not import MonteCarloModel: {e}")

# Try to import other models as available
# These are commented out since they likely don't exist yet
# try:
#     from poisson_model import PoissonModel
#     MODEL_REGISTRY['poisson'] = PoissonModel
# except ImportError:
#     pass

# try:
#     from random_forest_model import RandomForestModel
#     MODEL_REGISTRY['random_forest'] = RandomForestModel
# except ImportError:
#     pass

# Functions to access model registry

def get_model_class(model_name: str) -> Optional[Any]:
    """
    Get model class by name from registry.
    
    Args:
        model_name: Name of the model to retrieve
        
    Returns:
        Model class if found, None otherwise
    """
    model_name_lower = model_name.lower()
    model_class = MODEL_REGISTRY.get(model_name_lower)
    if model_class is None:
        logging.warning(f"Model '{model_name}' not found in the registry.")
    return model_class

def list_available_models() -> List[str]:
    """
    List all available model names.
    
    Returns:
        List of model names available in the registry
    """
    return list(MODEL_REGISTRY.keys())

# Print available models when module is imported
logging.info(f"Model registry initialized with models: {list_available_models()}")

# Example usage when run directly
if __name__ == '__main__':
    print("Available models:")
    for model_name in list_available_models():
        print(f"- {model_name}")