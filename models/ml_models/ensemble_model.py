# models/ensemble_model.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Literal
import warnings
import logging

from sklearn.exceptions import NotFittedError

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Assume base models have a .predict() method that returns a DataFrame
# with prediction and probability columns (e.g., 'prediction', 'prob_H', 'prob_D', 'prob_A')
# Or for regression, a 'prediction' column.
# Models like MonteCarlo/Poisson might need a wrapper or direct call to get probabilities.

class EnsembleModel:
    """
    Combines predictions from multiple base models to generate a final decision.

    Supports strategies like simple averaging, weighted averaging, or voting
    for classification tasks. Can also average predictions for regression tasks.
    """

    def __init__(self, models: Dict[str, Any], model_weights: Optional[Dict[str, float]] = None):
        """
        Initializes the EnsembleModel.

        Args:
            models (Dict[str, Any]): A dictionary where keys are model names (e.g., 'rf', 'xgb', 'poisson')
                                     and values are the instantiated and potentially fitted model objects.
                                     These objects must have a compatible `predict` method.
            model_weights (Optional[Dict[str, float]]): A dictionary mapping model names to their weights
                                                        for weighted averaging/voting. Weights should ideally
                                                        sum to 1. If None, equal weights are assumed.
        """
        if not models:
            raise ValueError("The 'models' dictionary cannot be empty.")

        self.models = models
        self.model_names = list(models.keys())
        self.weights = self._validate_and_normalize_weights(model_weights)

        logging.info(f"EnsembleModel initialized with models: {self.model_names}")
        logging.info(f"Using weights: {self.weights}")

    def _validate_and_normalize_weights(self, model_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Validates and normalizes weights, defaulting to equal weights if None."""
        if model_weights is None:
            # Default to equal weights
            num_models = len(self.models)
            return {name: 1.0 / num_models for name in self.model_names}
        else:
            # Check if all models have weights
            if set(model_weights.keys()) != set(self.model_names):
                missing = set(self.model_names) - set(model_weights.keys())
                extra = set(model_weights.keys()) - set(self.model_names)
                raise ValueError(f"Mismatch between models and weights. Missing: {missing}, Extra: {extra}")

            # Check if weights are non-negative
            if any(w < 0 for w in model_weights.values()):
                raise ValueError("Model weights cannot be negative.")

            # Normalize weights to sum to 1
            total_weight = sum(model_weights.values())
            if total_weight <= 0:
                raise ValueError("Total weight must be positive.")
            if not np.isclose(total_weight, 1.0):
                warnings.warn(f"Provided model weights do not sum to 1 (Sum={total_weight:.4f}). Normalizing weights.")
                return {name: w / total_weight for name, w in model_weights.items()}
            else:
                return model_weights # Already valid

    def set_weights(self, model_weights: Dict[str, float]):
        """Allows updating the model weights after initialization."""
        self.weights = self._validate_and_normalize_weights(model_weights)
        logging.info(f"Model weights updated to: {self.weights}")

    def gather_predictions(self, data: Any) -> Dict[str, pd.DataFrame]:
        """
        Collects predictions from all base models for the given input data.

        Args:
            data (Any): Input data suitable for the `predict` method of the base models.
                        This could be a pd.DataFrame, a dictionary, etc., depending on
                        what the base models expect. It's assumed this input contains
                        all necessary information (features, lambdas, etc.) for all models.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are model names and values
                                     are the prediction DataFrames returned by each model.

        Raises:
            RuntimeError: If a base model fails during prediction.
        """
        all_predictions: Dict[str, pd.DataFrame] = {}
        logging.info(f"Gathering predictions from {len(self.models)} models...")

        for name, model in self.models.items():
            try:
                # Check if the model has a 'predict' method
                if not hasattr(model, 'predict') or not callable(model.predict):
                     warnings.warn(f"Model '{name}' does not have a callable 'predict' method. Skipping.")
                     continue

                logging.debug(f"Predicting with model: {name}")
                # Pass the same input data to all models.
                # Assumes data contains everything needed (e.g., features for RF/XGB, lambdas for Poisson/MC)
                pred_df = model.predict(data)

                if not isinstance(pred_df, pd.DataFrame):
                     warnings.warn(f"Prediction from model '{name}' is not a DataFrame. Skipping.")
                     continue

                all_predictions[name] = pred_df
                logging.debug(f"Received predictions from model: {name}")

            except NotFittedError:
                 logging.error(f"Model '{name}' is not fitted. Cannot predict.")
                 raise RuntimeError(f"Model '{name}' is not fitted.") from NotFittedError
            except Exception as e:
                 logging.error(f"Error during prediction with model '{name}': {e}", exc_info=True)
                 # Option: Skip failing model or raise error? Raising is safer.
                 raise RuntimeError(f"Prediction failed for model '{name}'.") from e

        if not all_predictions:
             raise RuntimeError("No predictions were gathered from any base model.")

        logging.info("Finished gathering predictions.")
        return all_predictions

    def log_mismatches(self,
                       gathered_predictions: Dict[str, pd.DataFrame],
                       prob_threshold: float = 0.1, # Log if max prob diff > threshold
                       log_level: int = logging.WARNING):
        """
        Analyzes and logs significant disagreements between base model predictions.

        Args:
            gathered_predictions (Dict[str, pd.DataFrame]): The output from gather_predictions.
            prob_threshold (float): Log a warning if the standard deviation of probabilities
                                    for the predicted class exceeds this threshold.
            log_level (int): Logging level for mismatch messages (e.g., logging.INFO, logging.WARNING).
        """
        logging.info("Checking for prediction mismatches...")
        if not gathered_predictions:
            logging.warning("No predictions available to check for mismatches.")
            return

        # Assume all prediction DataFrames have the same index and number of rows
        # Use the first DataFrame to determine structure (columns, index)
        first_model_name = next(iter(gathered_predictions))
        ref_df = gathered_predictions[first_model_name]
        num_samples = len(ref_df)
        mismatch_count = 0

        # --- Identify common probability columns (e.g., prob_H, prob_D, prob_A) ---
        prob_cols_by_model = {}
        common_prob_cols = None
        for name, df in gathered_predictions.items():
            cols = [c for c in df.columns if c.startswith('prob_')]
            if not cols: continue # Skip models without probability columns
            prob_cols_by_model[name] = cols
            if common_prob_cols is None:
                common_prob_cols = set(cols)
            else:
                common_prob_cols.intersection_update(cols)

        if not common_prob_cols:
            logging.warning("No common probability columns found across models. Cannot compare probabilities.")
            common_prob_cols = []
        else:
            common_prob_cols = sorted(list(common_prob_cols))
            logging.info(f"Comparing common probability columns: {common_prob_cols}")


        for i in range(num_samples): # Iterate through each sample (match)
            sample_predictions = {}
            sample_probabilities = {}
            sample_final_preds = []

            # Collect predictions and probabilities for this sample
            for name, df in gathered_predictions.items():
                if i >= len(df): continue # Should not happen if indices align

                row = df.iloc[i]
                if 'prediction' in row:
                    sample_final_preds.append(row['prediction'])

                if common_prob_cols:
                    try:
                        sample_probabilities[name] = row[common_prob_cols].values
                    except KeyError:
                         logging.debug(f"Missing common prob cols for model {name}, sample {i}")
                         continue # Skip if model doesn't have the common cols for this row

            # Check for mismatch in final predicted class
            if len(set(sample_final_preds)) > 1:
                mismatch_count += 1
                logging.log(log_level, f"Mismatch found at index {ref_df.index[i]}: Predictions={dict(zip(gathered_predictions.keys(), sample_final_preds))}")

            # Check for high variance in probabilities (if available)
            if sample_probabilities and len(sample_probabilities) > 1:
                 # Stack probabilities for common columns across models
                 prob_array = np.array(list(sample_probabilities.values())) # Shape: (n_models, n_common_probs)
                 prob_std_dev = np.std(prob_array, axis=0) # Std dev for each common probability column

                 if np.any(prob_std_dev > prob_threshold):
                      logging.log(log_level, f"High probability variance at index {ref_df.index[i]}: StdDevs={dict(zip(common_prob_cols, prob_std_dev.round(3)))}")


        logging.info(f"Mismatch check complete. Found disagreements in final prediction for {mismatch_count}/{num_samples} samples.")


    def combine_predictions(self,
                            gathered_predictions: Dict[str, pd.DataFrame],
                            strategy: Literal['average', 'weighted_average'] = 'weighted_average',
                            target_type: Optional[str] = None
                            ) -> pd.DataFrame:
        """
        Combines predictions using the specified strategy.

        Args:
            gathered_predictions (Dict[str, pd.DataFrame]): Output from gather_predictions.
            strategy (Literal['average', 'weighted_average']): The combination strategy.
                                                               'voting' could be added but averaging is common for probs.
            target_type (Optional[str]): Specify 'classification', 'binary', or 'regression'.
                                         If None, attempts to infer from prediction columns.

        Returns:
            pd.DataFrame: DataFrame with the final ensemble prediction and probabilities.
        """
        logging.info(f"Combining predictions using strategy: '{strategy}'")
        if not gathered_predictions:
            raise ValueError("Cannot combine empty predictions.")

        # --- Determine Task Type and Probability Columns ---
        first_model_name = next(iter(gathered_predictions))
        ref_df = gathered_predictions[first_model_name]
        ref_index = ref_df.index

        # Infer target type if not specified
        if target_type is None:
            if any(c.startswith('prob_') for c in ref_df.columns):
                if len([c for c in ref_df.columns if c.startswith('prob_')]) > 2:
                    target_type = 'classification'
                else:
                    target_type = 'binary'
            elif 'prediction' in ref_df.columns and pd.api.types.is_numeric_dtype(ref_df['prediction']):
                 target_type = 'regression'
            else:
                 # Default or raise error if unable to infer
                 warnings.warn("Could not infer target_type, assuming classification.")
                 target_type = 'classification'
        logging.info(f"Determined target type: {target_type}")

        # --- Regression Averaging ---
        if target_type == 'regression':
            all_preds = []
            weights = []
            valid_models = []
            for name, df in gathered_predictions.items():
                if 'prediction' in df.columns and pd.api.types.is_numeric_dtype(df['prediction']):
                    all_preds.append(df['prediction'])
                    weights.append(self.weights[name])
                    valid_models.append(name)
                else:
                    warnings.warn(f"Model '{name}' prediction column missing or not numeric. Excluding from regression average.")

            if not valid_models:
                raise ValueError("No valid regression predictions found to combine.")

            pred_matrix = pd.concat(all_preds, axis=1)
            if strategy == 'average':
                final_prediction = pred_matrix.mean(axis=1)
            elif strategy == 'weighted_average':
                # Adjust weights in case some models were excluded
                valid_weights = np.array([self.weights[name] for name in valid_models])
                normalized_weights = valid_weights / valid_weights.sum()
                final_prediction = np.average(pred_matrix, axis=1, weights=normalized_weights)
            else:
                raise ValueError(f"Unsupported strategy '{strategy}' for regression.")

            return pd.DataFrame({'prediction': final_prediction}, index=ref_index)

        # --- Classification/Binary Averaging ---
        # Identify common probability columns across all models that provide them
        prob_cols_by_model = {}
        common_prob_cols = None
        valid_models_for_probs = []

        for name, df in gathered_predictions.items():
            cols = sorted([c for c in df.columns if c.startswith('prob_')])
            if cols:
                prob_cols_by_model[name] = cols
                valid_models_for_probs.append(name)
                if common_prob_cols is None:
                    common_prob_cols = set(cols)
                else:
                    # Ensure consistency - models should predict probs for the same classes
                    if set(cols) != common_prob_cols:
                         warnings.warn(f"Model '{name}' has different probability columns ({cols}) than expected ({common_prob_cols}). Excluding from probability average.")
                         valid_models_for_probs.remove(name)
                         # If exclusion happens, need to remove from common_prob_cols if it was the first one
                         if len(valid_models_for_probs) == 0: common_prob_cols = None

        if not valid_models_for_probs or not common_prob_cols:
            warnings.warn("No models found with consistent probability columns. Cannot average probabilities. Consider 'voting' strategy or check base model outputs.")
            # Fallback: Maybe try hard voting based on 'prediction' column?
            # For now, raise error or return empty
            raise ValueError("Cannot perform probability averaging due to inconsistent/missing probability columns.")

        common_prob_cols = sorted(list(common_prob_cols))
        logging.info(f"Averaging common probability columns: {common_prob_cols}")

        # Accumulate weighted probabilities
        weighted_probs_sum = pd.DataFrame(0.0, index=ref_index, columns=common_prob_cols)
        total_weight_used = 0.0

        for name in valid_models_for_probs:
            weight = self.weights[name]
            # Ensure the DataFrame has the common columns in the correct order
            model_probs = gathered_predictions[name][common_prob_cols]
            weighted_probs_sum += model_probs * weight
            total_weight_used += weight

        # Normalize by total weight used (in case some models were skipped or weights didn't sum to 1 initially)
        if total_weight_used > 0:
            final_probs_df = weighted_probs_sum / total_weight_used
        else: # Should not happen if valid_models_for_probs is not empty
             final_probs_df = weighted_probs_sum # Will be all zeros

        # Determine final prediction based on highest probability
        final_prediction_labels = final_probs_df.idxmax(axis=1).str.replace('prob_', '')

        # Combine prediction and probabilities
        final_df = pd.concat([pd.DataFrame({'prediction': final_prediction_labels}, index=ref_index), final_probs_df], axis=1)

        logging.info("Finished combining predictions.")
        return final_df


    def predict(self, data: Any, strategy: Literal['average', 'weighted_average'] = 'weighted_average') -> pd.DataFrame:
        """
        Generates the final ensemble prediction for the given input data.

        Args:
            data (Any): Input data suitable for the base models.
            strategy (Literal['average', 'weighted_average']): Combination strategy.

        Returns:
            pd.DataFrame: DataFrame with the final ensemble prediction and probabilities/values.
        """
        gathered_predictions = self.gather_predictions(data)
        self.log_mismatches(gathered_predictions) # Log potential issues
        final_prediction = self.combine_predictions(gathered_predictions, strategy=strategy)
        return final_prediction


# Example Usage
if __name__ == '__main__':
    print("\n--- EnsembleModel Example ---")

    # --- 1. Create Dummy Base Model Predictions ---
    # Imagine these came from previously run models
    dummy_index = pd.RangeIndex(start=0, stop=3, step=1)
    preds_rf = pd.DataFrame({
        'prediction': ['H', 'D', 'A'],
        'prob_H': [0.6, 0.3, 0.1],
        'prob_D': [0.3, 0.5, 0.2],
        'prob_A': [0.1, 0.2, 0.7]
    }, index=dummy_index)

    preds_xgb = pd.DataFrame({
        'prediction': ['H', 'H', 'A'],
        'prob_H': [0.7, 0.45, 0.15],
        'prob_D': [0.2, 0.35, 0.15],
        'prob_A': [0.1, 0.20, 0.70] # Note: XGB might order classes differently, but columns named consistently
    }, index=dummy_index)

    # Poisson/MC might only provide probabilities
    preds_poisson_probs = pd.DataFrame({
        'prob_H': [0.55, 0.35, 0.20],
        'prob_D': [0.25, 0.40, 0.30],
        'prob_A': [0.20, 0.25, 0.50]
    }, index=dummy_index)
    # Add a dummy 'predict' method to the DataFrame for compatibility with gather_predictions
    preds_poisson_probs.predict = lambda data: preds_poisson_probs # Simple mock

    # --- 2. Create Dummy Model Objects (or load real ones) ---
    # In reality, load fitted models here
    dummy_models = {
        'rf': preds_rf, # Using DataFrames directly as mocks
        'xgb': preds_xgb,
        'poisson': preds_poisson_probs
    }
    # Add dummy predict methods to the DataFrames for the example
    dummy_models['rf'].predict = lambda data: dummy_models['rf']
    dummy_models['xgb'].predict = lambda data: dummy_models['xgb']
    # Poisson already mocked above

    # --- 3. Initialize Ensemble ---
    # Example weights (must sum to 1 or will be normalized)
    weights = {'rf': 0.4, 'xgb': 0.4, 'poisson': 0.2}
    ensemble = EnsembleModel(models=dummy_models, model_weights=weights)

    # --- 4. Make Ensemble Prediction ---
    # Input 'dummy_data' is not actually used by the mocked predict methods here
    dummy_input_data = {'features': [1, 2, 3]}
    try:
        final_preds_weighted = ensemble.predict(dummy_input_data, strategy='weighted_average')
        print("\nFinal Ensemble Predictions (Weighted Average):")
        print(final_preds_weighted)

        final_preds_avg = ensemble.predict(dummy_input_data, strategy='average')
        print("\nFinal Ensemble Predictions (Simple Average):")
        print(final_preds_avg)

    except Exception as e:
        logging.error(f"Error during ensemble prediction example: {e}", exc_info=True)

    # --- Example: Regression ---
    print("\n--- Regression Example ---")
    preds_reg1 = pd.DataFrame({'prediction': [2.5, 1.8, 3.1]}, index=dummy_index)
    preds_reg2 = pd.DataFrame({'prediction': [2.8, 1.5, 2.9]}, index=dummy_index)
    preds_reg1.predict = lambda data: preds_reg1
    preds_reg2.predict = lambda data: preds_reg2

    reg_models = {'reg1': preds_reg1, 'reg2': preds_reg2}
    ensemble_reg = EnsembleModel(models=reg_models) # Equal weights

    try:
        final_preds_reg = ensemble_reg.predict(dummy_input_data, strategy='average')
        print("\nFinal Ensemble Predictions (Regression Average):")
        print(final_preds_reg)
    except Exception as e:
        logging.error(f"Error during regression ensemble example: {e}", exc_info=True)