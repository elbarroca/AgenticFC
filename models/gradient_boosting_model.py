# models/gradient_boosting_model.py
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import List, Optional, Dict, Any, Union, Tuple

from .base_model import BaseModel # Assuming base_model.py defines the interface

class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting Machine (XGBoost) model for predicting soccer outcomes
    (e.g., H/D/A classification, goal regression) using structured, tabular data.

    Leverages the XGBoost library for high performance and flexibility.
    Assumes input features (X) are preprocessed and numerical.
    """

    def __init__(self,
                 target_type: str = 'classification', # 'classification', 'binary', 'regression'
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 5,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 gamma: float = 0,
                 reg_alpha: float = 0,
                 reg_lambda: float = 1,
                 objective: Optional[str] = None, # Auto-set based on target_type if None
                 eval_metric: Optional[Union[str, List[str]]] = None, # Auto-set if None
                 early_stopping_rounds: Optional[int] = None, # Enable early stopping during fit
                 random_state: Optional[int] = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initializes the GradientBoostingModel (XGBoost).

        Args:
            target_type (str): Task type: 'classification', 'binary', 'regression'.
            n_estimators (int): Number of boosting rounds (trees).
            learning_rate (float): Step size shrinkage. Lower values require more estimators.
            max_depth (int): Maximum depth of a tree.
            subsample (float): Fraction of samples used per tree. Prevents overfitting.
            colsample_bytree (float): Fraction of features used per tree.
            gamma (float): Minimum loss reduction required to make a further partition (pruning).
            reg_alpha (float): L1 regularization term on weights.
            reg_lambda (float): L2 regularization term on weights (default=1).
            objective (Optional[str]): XGBoost objective function. If None, inferred from target_type.
                                       Examples: 'multi:softprob', 'binary:logistic', 'reg:squarederror'.
            eval_metric (Optional[Union[str, List[str]]]): Metric(s) for evaluation during training
                                       and early stopping. If None, inferred.
                                       Examples: 'mlogloss', 'logloss', 'rmse', 'mae'.
            early_stopping_rounds (Optional[int]): Activates early stopping if validation data is provided
                                                   during fit. Training stops if eval_metric doesn't improve.
            random_state (Optional[int]): Seed for reproducibility.
            n_jobs (int): Number of parallel threads. -1 uses all available cores.
            **kwargs: Additional keyword arguments passed directly to XGBClassifier/XGBRegressor.
        """
        if target_type not in ['classification', 'binary', 'regression']:
            raise ValueError("target_type must be 'classification', 'binary', or 'regression'")

        self.target_type = target_type
        self.early_stopping_rounds = early_stopping_rounds # Store separately for fit logic
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            **kwargs
        }

        # --- Auto-configure objective and eval_metric if not provided ---
        if objective is None:
            if target_type == 'classification':
                self.params['objective'] = 'multi:softprob' # Output probabilities
            elif target_type == 'binary':
                self.params['objective'] = 'binary:logistic'
            elif target_type == 'regression':
                self.params['objective'] = 'reg:squarederror'
        else:
            self.params['objective'] = objective

        if eval_metric is None:
            if target_type == 'classification':
                self.params['eval_metric'] = 'mlogloss'
            elif target_type == 'binary':
                self.params['eval_metric'] = 'logloss'
            elif target_type == 'regression':
                self.params['eval_metric'] = 'rmse' # Root Mean Squared Error
        else:
             self.params['eval_metric'] = eval_metric

        # Initialize internal state
        self.model: Optional[Union[xgb.XGBClassifier, xgb.XGBRegressor]] = None
        self.is_fitted: bool = False
        self.feature_names_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None # Original class labels (e.g., 'H', 'D', 'A')
        self.label_encoder_: Optional[LabelEncoder] = None # Used for classification targets
        self.num_class_: Optional[int] = None # Number of classes for multiclass

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
            verbose: bool = False):
        """
        Trains the XGBoost model.

        Args:
            X_train (pd.DataFrame): Features for training (numerical). Handle NaNs before fit.
            y_train (pd.Series): Target labels ('H', 'D', 'A'; 0, 1; or continuous values).
            eval_set (Optional[List[Tuple[pd.DataFrame, pd.Series]]]): List of (X, y) pairs for evaluation
                                       during training, enabling early stopping.
                                       Example: [(X_val, y_val)].
            verbose (bool): If True, prints evaluation metrics during training (if eval_set is provided).
        """
        print(f"Fitting GradientBoostingModel (XGBoost, target_type='{self.target_type}')...")

        # --- Input Validation ---
        # (Similar validation as RandomForestModel: check types, NaNs, lengths)
        if not isinstance(X_train, pd.DataFrame): raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
             if isinstance(y_train, np.ndarray): y_train = pd.Series(y_train)
             else: raise TypeError("y_train must be a pandas Series or numpy array.")
        if X_train.isnull().values.any(): raise ValueError("X_train contains NaN values.")
        if y_train.isnull().values.any(): raise ValueError("y_train contains NaN values.")
        if len(X_train) != len(y_train): raise ValueError("X_train and y_train lengths mismatch.")

        # --- Store Metadata & Prepare Target ---
        self.feature_names_ = list(X_train.columns)
        y_train_encoded = y_train.copy() # Default for regression/binary

        if self.target_type in ['classification', 'binary']:
            self.label_encoder_ = LabelEncoder()
            y_train_encoded = self.label_encoder_.fit_transform(y_train)
            self.classes_ = self.label_encoder_.classes_
            if self.target_type == 'classification':
                self.num_class_ = len(self.classes_)
                # Add num_class to params if objective is multi:* and not already set
                if 'multi' in self.params['objective'] and 'num_class' not in self.params:
                    self.params['num_class'] = self.num_class_
            print(f"  Target classes found and encoded: {dict(zip(self.classes_, range(len(self.classes_))))}")
        else: # Regression
             self.classes_ = None # Not applicable
             self.label_encoder_ = None
             self.num_class_ = None


        # --- Initialize Model ---
        if self.target_type == 'regression':
            self.model = xgb.XGBRegressor(**self.params)
        else: # classification or binary
            self.model = xgb.XGBClassifier(**self.params)

        # --- Prepare Fit Arguments (Early Stopping) ---
        fit_params = {}
        if self.early_stopping_rounds is not None and eval_set is not None:
            processed_eval_set = []
            for i, (X_val, y_val) in enumerate(eval_set):
                # Validate validation set features
                if list(X_val.columns) != self.feature_names_:
                     raise ValueError(f"Validation set {i} features mismatch training features.")
                if X_val.isnull().values.any():
                     raise ValueError(f"Validation set {i} contains NaN values.")
                # Encode validation target if needed
                y_val_encoded = y_val
                if self.label_encoder_:
                    try:
                        # Use transform only, assumes validation labels are subset of train labels
                        y_val_encoded = self.label_encoder_.transform(y_val)
                    except ValueError as e:
                        unseen = set(y_val) - set(self.label_encoder_.classes_)
                        raise ValueError(f"Validation set {i} contains unseen labels: {unseen}") from e
                processed_eval_set.append((X_val, y_val_encoded))

            fit_params['eval_set'] = processed_eval_set
            fit_params['early_stopping_rounds'] = self.early_stopping_rounds
            fit_params['verbose'] = verbose
        elif self.early_stopping_rounds is not None:
             warnings.warn("early_stopping_rounds is set, but no eval_set provided. Early stopping inactive.")

        # --- Train Model ---
        print(f"  Training with objective='{self.model.objective}', eval_metric='{self.model.eval_metric}'...")
        self.model.fit(X_train, y_train_encoded, **fit_params)
        self.is_fitted = True
        print("Fitting complete.")
        if self.early_stopping_rounds and eval_set and hasattr(self.model, 'best_iteration'):
             print(f"  Best iteration (early stopping): {self.model.best_iteration}")


    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities for classification/binary tasks.

        Args:
            X_test (pd.DataFrame): Features for prediction.

        Returns:
            pd.DataFrame: DataFrame with probability columns (e.g., 'prob_H', 'prob_D', 'prob_A').

        Raises:
            TypeError: If model target_type is 'regression'.
            NotFittedError: If model is not fitted.
        """
        check_is_fitted(self, ['model', 'feature_names_'])
        if self.target_type == 'regression':
            raise TypeError("predict_proba is not available for regression models.")
        if self.label_encoder_ is None or self.classes_ is None:
             raise RuntimeError("Class labels/encoder not available. Model might be regression type or fit failed.")

        # Validate X_test (features, NaNs) - simplified here, robust check needed
        if list(X_test.columns) != self.feature_names_:
             X_test = X_test[self.feature_names_] # Attempt reorder/select
        if X_test.isnull().values.any(): raise ValueError("X_test contains NaN values.")

        probabilities = self.model.predict_proba(X_test)
        prob_cols = [f"prob_{cls}" for cls in self.classes_]
        prob_df = pd.DataFrame(probabilities, columns=prob_cols, index=X_test.index)
        return prob_df

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions using the trained XGBoost model.

        Args:
            X_test (pd.DataFrame): Features for prediction. Handle NaNs beforehand.

        Returns:
            pd.DataFrame: DataFrame containing predictions.
                          - Classification/Binary: 'prediction' (original label), 'prob_...' columns.
                          - Regression: 'prediction' column with continuous values.
        """
        check_is_fitted(self, ['model', 'feature_names_'])

        # Validate X_test (features, NaNs) - simplified here
        if list(X_test.columns) != self.feature_names_:
             try:
                 X_test = X_test[self.feature_names_]
             except KeyError as e:
                 missing = set(self.feature_names_) - set(X_test.columns)
                 raise ValueError(f"Feature mismatch: Missing {missing}") from e
        if X_test.isnull().values.any(): raise ValueError("X_test contains NaN values.")

        print(f"Predicting on {len(X_test)} samples...")
        if self.target_type == 'regression':
            predictions = self.model.predict(X_test)
            results_df = pd.DataFrame({'prediction': predictions}, index=X_test.index)
        else: # Classification or Binary
            # Get probabilities first
            prob_df = self.predict_proba(X_test)
            # Get integer predictions from the model
            int_predictions = self.model.predict(X_test)
            # Decode integer predictions back to original labels
            decoded_predictions = self.label_encoder_.inverse_transform(int_predictions)
            pred_df = pd.DataFrame({'prediction': decoded_predictions}, index=X_test.index)
            results_df = pd.concat([pred_df, prob_df], axis=1)

            # Attempt standard H/D/A reordering for classification
            if self.target_type == 'classification':
                standard_order = ['H', 'D', 'A']
                ordered_cols = ['prediction']
                present_classes = [str(c) for c in self.classes_]
                if all(str(c) in present_classes for c in standard_order):
                    ordered_cols.extend([f"prob_{c}" for c in standard_order])
                else:
                    ordered_cols.extend(prob_df.columns) # Keep original prob columns
                try:
                    results_df = results_df[ordered_cols]
                except KeyError:
                    pass # Keep original if reorder fails

        print("Prediction complete.")
        return results_df

    def feature_importance(self, importance_type: str = 'gain') -> pd.Series:
        """
        Returns feature importances from the trained XGBoost model.

        Args:
            importance_type (str): Type of importance to return ('weight', 'gain', 'cover', 'total_gain', 'total_cover').
                                   'gain' is often a good default. Defaults to 'gain'.

        Returns:
            pd.Series: Feature importances, indexed by feature name, sorted descending.
        """
        check_is_fitted(self, ['model', 'feature_names_'])
        try:
            # Use the model's built-in method if available (newer XGBoost versions)
            # score = self.model.get_booster().get_score(importance_type=importance_type)
            # For compatibility, access the attribute directly:
            importances = self.model.feature_importances_ # Note: Default importance type might vary by XGB version/model type
            # If you need specific types like 'gain', you might need to access the booster:
            # booster = self.model.get_booster()
            # score_dict = booster.get_score(importance_type=importance_type)
            # # Need to map back to original feature names if booster renames them
            # importances = pd.Series(score_dict).reindex(self.feature_names_, fill_value=0)

        except AttributeError:
             raise NotFittedError("Could not retrieve feature importances. Model might not be fitted or type incompatible.")

        return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)

    def save_model(self, filepath: str):
        """
        Saves the trained model state (model, feature names, encoder, etc.) using joblib.

        Args:
            filepath (str): Path to save the model file (e.g., 'xgb_model.joblib').
        """
        if not self.is_fitted:
            raise NotFittedError("Cannot save an unfitted model.")
        print(f"Saving GradientBoostingModel (XGBoost) state to {filepath}...")
        state = {
            'model': self.model,
            'feature_names_': self.feature_names_,
            'classes_': self.classes_,
            'label_encoder_': self.label_encoder_,
            'target_type': self.target_type,
            'params': self.params, # Save init params
            'num_class_': self.num_class_
        }
        joblib.dump(state, filepath)
        print("Model state saved successfully.")

    @classmethod
    def load_model(cls, filepath: str):
        """
        Loads a trained model state from a file.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            GradientBoostingModel: An instance of the class with the loaded state.
        """
        print(f"Loading GradientBoostingModel (XGBoost) state from {filepath}...")
        state = joblib.load(filepath)
        # Create instance with saved config
        instance = cls(target_type=state.get('target_type', 'classification'), **state.get('params', {}))
        # Load the fitted attributes
        instance.model = state['model']
        instance.feature_names_ = state['feature_names_']
        instance.classes_ = state['classes_']
        instance.label_encoder_ = state['label_encoder_']
        instance.num_class_ = state.get('num_class_')
        instance.is_fitted = True
        print("Model state loaded successfully.")
        return instance

# Example Usage
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, log_loss

    # --- 1. Create Dummy Data (Using same setup as RF example) ---
    print("\n--- GradientBoostingModel (XGBoost) Example ---")
    np.random.seed(42)
    data_size = 500
    X = pd.DataFrame({
        'ImpliedProbH': np.random.uniform(0.2, 0.7, data_size), 'ImpliedProbD': np.random.uniform(0.2, 0.4, data_size),
        'ImpliedProbA': 1.0 - (np.random.uniform(0.2, 0.7, data_size) + np.random.uniform(0.2, 0.4, data_size)),
        'HomeFormPts_L5': np.random.randint(0, 16, data_size), 'AwayFormPts_L5': np.random.randint(0, 16, data_size),
        'HomeGoalDiff_L5': np.random.randint(-5, 6, data_size), 'AwayGoalDiff_L5': np.random.randint(-5, 6, data_size),
        'LeaguePosDiff': np.random.randint(-19, 20, data_size), 'H2H_HomeWinRatio': np.random.rand(data_size)
    })
    X['ImpliedProbA'] = X['ImpliedProbA'].clip(lower=0.05); X['ImpliedProbH'] = (1.0 - X['ImpliedProbD'] - X['ImpliedProbA']).clip(lower=0.05)
    prob_H_true = X['ImpliedProbH'] * 1.2 + (X['HomeFormPts_L5'] - X['AwayFormPts_L5']) / 30 - X['LeaguePosDiff'] / 40
    prob_A_true = X['ImpliedProbA'] * 1.2 + (X['AwayFormPts_L5'] - X['HomeFormPts_L5']) / 30 + X['LeaguePosDiff'] / 40
    prob_D_true = X['ImpliedProbD'] * 1.1 + (1 - abs(X['LeaguePosDiff'] / 20)) * 0.1
    total_prob = (prob_H_true + prob_D_true + prob_A_true).clip(lower=1e-6); p_H = (prob_H_true / total_prob).clip(0, 1)
    p_D = (prob_D_true / total_prob).clip(0, 1); p_A = 1.0 - p_H - p_D
    y_cat = [];
    for p_h, p_d, p_a in zip(p_H, p_D, p_A): probs = np.array([p_h, p_d, p_a]); probs /= probs.sum(); y_cat.append(np.random.choice(['H', 'D', 'A'], p=probs))
    y = pd.Series(y_cat)

    print("Dummy Data Generated (Features):"); print(X.head())
    print("\nTarget Distribution:"); print(y.value_counts(normalize=True))

    # --- 2. Split Data (Train, Validation, Test) ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
    print(f"\nTrain size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # --- 3. Initialize and Fit Model (with early stopping) ---
    xgb_model = GradientBoostingModel(
        target_type='classification',
        n_estimators=500, # Higher number for early stopping
        learning_rate=0.05,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.1,
        early_stopping_rounds=50, # Enable early stopping
        random_state=42
    )
    # Note: For early stopping, XGBoost needs the eval_set during fit
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # --- 4. Predict ---
    predictions_df = xgb_model.predict(X_test)
    print("\nPredictions on Test Set (Top 5):")
    print(predictions_df.head())

    # --- 5. Evaluate ---
    print("\nEvaluation:")
    accuracy = accuracy_score(y_test, predictions_df['prediction'])
    # Need probabilities for log loss - use predict_proba
    try:
        prob_df = xgb_model.predict_proba(X_test)
         # Ensure columns match the order expected by log_loss based on original labels
        logloss = log_loss(y_test, prob_df[[f"prob_{c}" for c in xgb_model.label_encoder_.classes_]])
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {logloss:.4f}")
    except Exception as e:
        print(f"Could not calculate Log Loss: {e}")
        print(f"Accuracy: {accuracy:.4f}")


    print("Classification Report:")
    report = classification_report(y_test, predictions_df['prediction'], labels=xgb_model.classes_)
    print(report)

    # --- 6. Feature Importance ---
    print("\nFeature Importances (default type):")
    try:
        importances = xgb_model.feature_importance()
        print(importances)
    except NotFittedError as e:
        print(e)

    # --- 7. Save and Load ---
    model_path = "temp_xgb_model_prod.joblib"
    xgb_model.save_model(model_path)
    loaded_model = GradientBoostingModel.load_model(model_path)

    # Verify loaded model makes same predictions
    loaded_predictions_df = loaded_model.predict(X_test)
    print("\nPredictions from Loaded Model (Top 5):")
    print(loaded_predictions_df.head())
    # Use appropriate comparison allowing for float tolerance
    pd.testing.assert_frame_equal(predictions_df, loaded_predictions_df, check_exact=False, atol=1e-6)
    print("\nSave/Load test passed.")

    # Clean up
    import os
    if os.path.exists(model_path): os.remove(model_path)