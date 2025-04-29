# inspect_encoder.py
import logging
import sys
import os
import numpy as np
from joblib import load as joblib_load
from huggingface_hub import hf_hub_download
from typing import Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Match your main script) ---
PODOS_MODEL_REPO = "Nickel5HF/podos_soccer_model"
# List of potential filenames, starting with the one that worked
PODOS_LABEL_ENCODER_FILENAMES = [
    "label_encoder.pkl", # This one worked in your logs
    "label_encoder.joblib",
    "encoder.joblib",
    "team_encoder.joblib",
    "encoder.pkl",
    "team_encoder.pkl",
]

def load_actual_label_encoder() -> Optional[Any]:
    """Tries to load the label encoder from known filenames."""
    label_encoder = None
    logger.info(f"Attempting to load label encoder from {PODOS_MODEL_REPO}...")

    for filename in PODOS_LABEL_ENCODER_FILENAMES:
        logger.debug(f"Trying filename: {filename}")
        try:
            encoder_path = hf_hub_download(
                repo_id=PODOS_MODEL_REPO,
                filename=filename
            )
            label_encoder = joblib_load(encoder_path)
            logger.info(f"Successfully loaded label encoder from '{filename}'")
            return label_encoder # Success
        except Exception as e:
             if "404" in str(e) or "EntryNotFoundError" in str(e):
                  logger.debug(f"Encoder file '{filename}' not found.")
             else:
                  logger.warning(f"Could not load '{filename}'. Error: {e}")
    return None # Failed to load any

if __name__ == "__main__":
    logger.info("--- Inspecting Podos Label Encoder ---")

    encoder = load_actual_label_encoder()

    if encoder:
        logger.info(f"Encoder loaded successfully (Type: {type(encoder).__name__}).")

        if hasattr(encoder, 'classes_'):
            classes = encoder.classes_
            logger.info(f"Encoder contains {len(classes)} classes.")
            print("\n--- Label Encoder Classes ---")
            # Print all classes without truncation
            np.set_printoptions(threshold=sys.maxsize)
            print(classes)
            print("\n---------------------------\n")
            logger.info("These are the identifiers (likely team names) the model expects.")
            logger.info("Map your database IDs (e.g., '65') to these exact identifiers.")

            # Example: How to check if a name from your mapping exists
            try:
                test_name = "Aston Villa" # Example name from your mapping
                index = list(classes).index(test_name)
                logger.info(f"Example check: Found '{test_name}' at index {index} in encoder.classes_")
                transformed_id = encoder.transform([test_name])[0]
                logger.info(f"'{test_name}' transforms to numerical ID: {transformed_id}")
            except ValueError:
                logger.warning(f"Example check: Name '{test_name}' NOT FOUND in encoder.classes_")
            except Exception as e:
                 logger.error(f"Error during example check: {e}")


        else:
            logger.error("The loaded object does not have a 'classes_' attribute. Cannot inspect expected IDs.")
            print("\nERROR: Loaded object is not a standard LabelEncoder or is malformed.")

    else:
        logger.error("Failed to load any label encoder file from the repository.")
        print("\nERROR: Could not load the encoder file.")

    logger.info("--- Inspection Finished ---")