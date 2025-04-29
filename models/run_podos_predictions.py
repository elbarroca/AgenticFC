# run_podos_predictions.py
import logging
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pymongo import MongoClient
from typing import Optional, Any, Dict

# --- Dynamically add project root to sys.path ---
# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.dirname(current_dir)
# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End dynamic path addition ---


# --- Safetensors Import ---
try:
    from safetensors.torch import load_file
except ImportError:
    print("Error: 'safetensors' library not found. Please install it: pip install safetensors")
    sys.exit(1)

# --- Import from local modules ---
try:
    # Import the feature generation function and label encoder loader (relative to models dir now OK)
    # Assuming feature_engineering_podos is in the same 'models' directory
    from feature_engineering_podos import (
        generate_podos_features,
        load_label_encoder,
        create_mongodb_id_to_name_map,
        MONGO_URI,
        DB_NAME,
        MATCHES_COLLECTION,
        ODDS_COLLECTION,
        PODOS_EXPECTED_FEATURES
    )
    # Import the TEAM_ID_MAPPING dictionary directly (using project root path added above)
    from get_data.api_football.db_ids.team_id_mappings import TEAM_ID_MAPPING

except ImportError as e:
    # Adjust error message if paths are different
    print(f"Error: Could not import from 'models.feature_engineering_podos' or 'get_data.api_football.db_ids.team_id_mappings'.")
    print(f"Ensure correct paths and project structure.")
    print(f"Current sys.path: {sys.path}")
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Model-Specific Imports ---
from huggingface_hub import hf_hub_download
from joblib import load as joblib_load
from sklearn.preprocessing import LabelEncoder

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Podos Model Configuration
PODOS_MODEL_REPO = "Nickel5HF/podos_soccer_model"
PODOS_MODEL_FILENAME = "model.safetensors"


# --- Real PodosTransformer Definition (Based on model.safetensors structure) ---
class PodosTransformer(nn.Module):
    """
    PyTorch Module implementing the Podos Transformer architecture
    based on the tensor shapes found in Nickel5HF/podos_soccer_model/model.safetensors.
    """
    def __init__(self, input_dim=23, d_model=32, nhead=4, num_encoder_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        logger.info(f"Initializing PodosTransformer: input_dim={input_dim}, d_model={d_model}, nhead={nhead}, layers={num_encoder_layers}, d_ff={dim_feedforward}")

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection layer
        # Weight shape [32, 23] -> nn.Linear(23, 32)
        self.projection = nn.Linear(input_dim, d_model)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu', # Common default, adjust if known otherwise
            batch_first=True   # Assume batch_first based on typical usage
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # Final classification layer
        # Weight shape [3, 32] -> nn.Linear(32, 3)
        self.fc = nn.Linear(d_model, 3)

        self._is_fitted = False # Track if weights are loaded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Input shape: (batch_size, num_features=23)
        Output shape: (batch_size, num_classes=3) -> Raw logits
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Input tensor last dimension ({x.shape[-1]}) does not match model input_dim ({self.input_dim})")

        # 1. Project input features
        x_proj = self.projection(x) # Shape: (batch_size, d_model)

        # 2. Transformer Encoder expects shape (batch_size, seq_len, d_model)
        # Since we have only one "sequence element" per match, we add a dimension
        x_seq = x_proj.unsqueeze(1) # Shape: (batch_size, 1, d_model)

        # 3. Pass through transformer encoder
        transformer_output = self.transformer_encoder(x_seq) # Shape: (batch_size, 1, d_model)

        # 4. Remove sequence dimension
        transformer_output_flat = transformer_output.squeeze(1) # Shape: (batch_size, d_model)

        # 5. Final fully connected layer
        logits = self.fc(transformer_output_flat) # Shape: (batch_size, 3)

        return logits

    def predict_proba(self, x_input_df: pd.DataFrame) -> np.ndarray:
        """
        Takes a DataFrame of features, converts to tensor,
        runs inference, and returns probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Model weights not loaded. Call load_state_dict first.")
        if not isinstance(x_input_df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")
        if list(x_input_df.columns) != PODOS_EXPECTED_FEATURES:
             logger.warning("Input DataFrame columns order mismatch. Reordering.")
             x_input_df = x_input_df[PODOS_EXPECTED_FEATURES]

        self.eval() # Ensure model is in evaluation mode
        with torch.no_grad(): # Disable gradient calculations for inference
            try:
                X_tensor = torch.tensor(x_input_df.values, dtype=torch.float32)
                logits = self.forward(X_tensor)
                # Apply Softmax to convert logits to probabilities
                probabilities = torch.softmax(logits, dim=1)
                return probabilities.numpy() # Return as numpy array
            except Exception as e:
                logger.error(f"Error during model forward pass or tensor conversion: {e}", exc_info=True)
                # Returning None or raising might be appropriate depending on desired handling
                # For now, let's return None to indicate failure at this stage
                return None


# --- Model Loading Function (Updated) ---
def load_podos_model() -> Optional[PodosTransformer]:
    """Loads the Podos model and its weights from Hugging Face Hub."""
    logger.info(f"Attempting to load Podos model weights: {PODOS_MODEL_FILENAME} from {PODOS_MODEL_REPO}")
    model = None
    try:
        # 1. Instantiate the REAL model class
        # Use parameters inferred from safetensors structure
        model = PodosTransformer(
            input_dim=len(PODOS_EXPECTED_FEATURES), # Should be 23
            d_model=32,
            nhead=4, # Assumed, common choice for d_model=32
            num_encoder_layers=2,
            dim_feedforward=2048,
            dropout=0.1 # Default dropout
        )
        logger.info("PodosTransformer model structure initialized.")

        # 2. Download the safetensors file
        weights_path = hf_hub_download(
            repo_id=PODOS_MODEL_REPO,
            filename=PODOS_MODEL_FILENAME
        )
        logger.info(f"Downloaded model weights file to: {weights_path}")

        # 3. Load the state dictionary from the file
        state_dict = load_file(weights_path)
        logger.info(f"Successfully loaded state dictionary with {len(state_dict)} tensors.")

        # 4. Load the state dict into the model instance
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True) # Use strict=True for validation
        if missing_keys:
             logger.error(f"CRITICAL ERROR: Missing keys when loading state dict: {missing_keys}")
             return None
        if unexpected_keys:
             logger.error(f"CRITICAL ERROR: Unexpected keys found in state dict: {unexpected_keys}")
             return None

        logger.info("Successfully loaded weights into PodosTransformer model.")
        model._is_fitted = True # Mark weights as loaded

        # 5. Set model to evaluation mode
        model.eval()
        logger.info("Model set to evaluation mode (model.eval()).")

    except FileNotFoundError as e:
         logger.error(f"CRITICAL ERROR: Model weights file '{PODOS_MODEL_FILENAME}' not found in repo '{PODOS_MODEL_REPO}' or download failed. Error: {e}")
         return None
    except Exception as e:
        logger.error(f"CRITICAL ERROR: Failed to load Podos model weights. Error: {e}", exc_info=True)
        return None # Cannot proceed without the model

    return model


# --- Prediction Function (Updated to use model method) ---
def predict_podos_probabilities(
    model: PodosTransformer,
    features_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Generates 1X2 probabilities using the loaded Podos model's predict_proba method.
    Input: DataFrame with features for one or more matches.
    Output: DataFrame with 'prob_H', 'prob_D', 'prob_A' columns.
    """
    if not isinstance(model, PodosTransformer) or not model._is_fitted:
        logger.error("Invalid or unloaded model passed to predict_podos_probabilities.")
        return None
    if not isinstance(features_df, pd.DataFrame):
        logger.error("Input 'features_df' is not a Pandas DataFrame.")
        return None
    if features_df.empty:
        logger.error("Input 'features_df' is empty.")
        return None

    logger.info(f"Running prediction for {len(features_df)} sample(s)...")

    try:
        # Use the model's internal predict_proba method
        probabilities = model.predict_proba(features_df) # Returns numpy array

        if probabilities is None or not isinstance(probabilities, np.ndarray) or probabilities.ndim != 2 or probabilities.shape[1] != 3 or probabilities.shape[0] != len(features_df):
            # Log error if predict_proba failed or returned unexpected result
            logger.error(f"Model's predict_proba returned None or incorrect shape/type. Got: {type(probabilities)}, Shape: {getattr(probabilities, 'shape', 'None')}")
            return None


        # --- Post-prediction checks (good practice) ---
        if np.isnan(probabilities).any() or np.isinf(probabilities).any():
             logger.warning("NaN or Inf detected in predicted probabilities. Replacing with 1/3.")
             # Use np.nan_to_num which handles inf as well
             probabilities = np.nan_to_num(probabilities, nan=1/3, posinf=1.0, neginf=0.0)
             # Ensure re-normalization after fixing NaN/Inf
             prob_sums_after_fix = probabilities.sum(axis=1, keepdims=True)
             # Avoid division by zero if a row sums to zero after nan_to_num (highly unlikely but safe)
             prob_sums_after_fix[prob_sums_after_fix == 0] = 1.0
             probabilities = probabilities / prob_sums_after_fix


        prob_sums = probabilities.sum(axis=1)
        # Use a tolerance for floating point comparisons
        if not np.allclose(prob_sums, 1.0, atol=1e-4):
            logger.warning(f"Probabilities do not sum close to 1 after softmax. Min/Max sum: {prob_sums.min():.4f}/{prob_sums.max():.4f}. Renormalizing.")
            # Ensure probabilities sum to exactly 1, avoiding division by zero
            prob_sums_safe = np.maximum(prob_sums, 1e-9) # Prevent division by zero or very small numbers
            probabilities = probabilities / prob_sums_safe[:, np.newaxis]

        # --- End checks ---

        # Create DataFrame using the index from the input features_df to maintain alignment
        prob_df = pd.DataFrame(probabilities, columns=['prob_H', 'prob_D', 'prob_A'], index=features_df.index)
        logger.info("Prediction successful.")
        return prob_df

    except Exception as e:
        logger.error(f"Unexpected error during Podos prediction call: {e}", exc_info=True)
        return None

# --- Main Orchestration Logic (Updated) ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <fixture_id_1> [fixture_id_2] ...")
        print(f"Example: python {os.path.basename(__file__)} 1035144 1035145")
        sys.exit(1)

    fixture_ids_to_process = sys.argv[1:]
    logger.info(f"Received {len(fixture_ids_to_process)} fixture IDs to process: {fixture_ids_to_process}")

    # --- Initialize Connections and Components ---
    client = None
    podos_model = None
    label_encoder = None
    all_results = []
    mongodb_id_to_name_map = None # Initialize map

    try:
        # 1. Create the mongodb_id -> team_name map from the imported dictionary
        # Ensure TEAM_ID_MAPPING is the variable name used in team_id_mappings.py
        mongodb_id_to_name_map = create_mongodb_id_to_name_map(TEAM_ID_MAPPING)
        if mongodb_id_to_name_map is None:
            # create_mongodb_id_to_name_map logs the error
            logger.critical("Failed to create mongodb_id to name map from TEAM_ID_MAPPING. Exiting.")
            sys.exit(1)

        # 2. Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster') # Check connection
        db = client[DB_NAME]
        matches_collection = db[MATCHES_COLLECTION]
        odds_collection = db[ODDS_COLLECTION]
        logger.info(f"MongoDB connected. Using DB: '{DB_NAME}'")

        # 3. Load Label Encoder
        logger.info("Loading label encoder...")
        label_encoder = load_label_encoder() # Imported function
        if label_encoder is None:
            # load_label_encoder should log failure or return dummy
            raise RuntimeError("Failed to load or create a label encoder.")

        # 4. Load Podos Model and Weights
        logger.info("Loading Podos model and weights...")
        podos_model = load_podos_model() # Calls the updated function
        if podos_model is None:
            raise RuntimeError("Failed to load Podos model weights.")

        # 5. (Team ID mapping already loaded above)

        # --- Process Fixtures ---
        logger.info("--- Starting Fixture Processing Loop ---")
        for i, fixture_id in enumerate(fixture_ids_to_process):
            logger.info(f"Processing Fixture {i+1}/{len(fixture_ids_to_process)}: ID = {fixture_id}")
            features_df = None
            results_df = None

            # 6. Generate Features for the current fixture
            try:
                # Pass the map created earlier
                features_df = generate_podos_features(
                    fixture_id=str(fixture_id),
                    matches_coll=matches_collection,
                    odds_coll=odds_collection,
                    label_encoder=label_encoder,
                    mongodb_id_to_name_map=mongodb_id_to_name_map # Pass the processed map
                )
            except Exception as e:
                 logger.error(f"Error during feature generation for fixture {fixture_id}: {e}", exc_info=True)
                 continue # Skip this fixture

            if features_df is not None and not features_df.empty:
                # 7. Run Prediction with REAL model if features were generated
                try:
                    results_df = predict_podos_probabilities(podos_model, features_df)
                except Exception as e:
                     logger.error(f"Error during prediction for fixture {fixture_id}: {e}", exc_info=True)
                     continue # Skip this fixture

            # 8. Store and Log Results
            if results_df is not None and not results_df.empty:
                 logger.info(f"Successfully generated prediction for fixture {fixture_id}")
                 # Ensure index alignment if features_df had multiple rows (though unlikely here)
                 result_dict = results_df.iloc[0].to_dict() # Assuming one fixture at a time
                 result_dict['fixture_id'] = fixture_id
                 all_results.append(result_dict)
                 log_probs = f"H: {result_dict['prob_H']:.4f}, D: {result_dict['prob_D']:.4f}, A: {result_dict['prob_A']:.4f}"
                 logger.info(f"Result for {fixture_id}: {log_probs}")
            else:
                 logger.warning(f"Failed to generate prediction for fixture {fixture_id}. Skipping.")

            logger.info("-" * 20)

        logger.info("--- Finished Fixture Processing Loop ---")

    except Exception as e:
        logger.error(f"An unrecoverable error occurred during the main process: {e}", exc_info=True)
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed.")

    # --- Display Final Summary ---
    if all_results:
        logger.info(f"Successfully generated predictions for {len(all_results)} out of {len(fixture_ids_to_process)} requested fixtures.")
        results_summary_df = pd.DataFrame(all_results)
        # Reorder columns for clarity
        results_summary_df = results_summary_df[['fixture_id', 'prob_H', 'prob_D', 'prob_A']]
        print("\n--- Prediction Results Summary ---")
        pd.set_option('display.float_format', '{:.4f}'.format)
        # Print DataFrame using .to_string() for better console formatting
        print(results_summary_df.to_string(index=False))
    else:
        logger.warning("No results were generated.")

    # --- Final Warnings ---
    print("\n" + "="*60)
    print("** IMPORTANT WARNINGS & NOTES **")
    print("- Predictions generated using the PodosTransformer structure and loaded weights.")
    print("- Feature generation relies on historical data and odds from MongoDB.")
    print("  Accuracy depends heavily on data availability and correctness.")
    print("  Check logs for warnings about missing odds, statistics, or team name mismatches.")
    print("  Using default odds or incomplete historical stats significantly degrades quality.")
    # Add a note about the mapping source
    print("- Team ID mapping is now loaded directly from 'get_data/api_football/db_ids/team_id_mappings.py'.")
    print("  Ensure team names in that file match the label encoder classes EXACTLY.")
    print("="*60)

    logger.info("--- Script Finished ---")