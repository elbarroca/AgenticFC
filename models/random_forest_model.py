# models/random_forest_model.py

import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import joblib
from typing import List, Optional, Dict, Any, Union

from .base_model import BaseModel # Assuming base_model.py defines the interface

class RandomForestModel(BaseModel):
    """
    Random Forest Classifier model for predicting soccer match outcomes (e.g., H/D/A)
    or binary targets (e.g., Over/Under 2.5, BTTS) using structured, tabular data.

    Leverages scikit-learn's RandomForestClassifier for robust and interpretable predictions.
    Assumes input features (X) are preprocessed and numerical.
    """

    def __init__(self,
                 target_type: str = 'classification', # 'classification' or 'binary'
                 n_estimators: int = 150,
                 max_depth: Optional[int] = 15,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 5,
                 max_features: Union[str, float] = 'sqrt',
                 class_weight: Optional[Union[str, Dict]] = 'balanced',
                 random_state: Optional[int] = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initializes the RandomForestModel.

        Args:
            target_type (str): Type of prediction task: 'classification' (multiclass, e.g., H/D/A)
                               or 'binary' (e.g., O/U, BTTS).
            n_estimators (int): The number of trees in the forest.
            max_depth (Optional[int]): Maximum depth of the trees. None means nodes expand until pure.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
            max_features (Union[str, float]): Number of features to consider when looking for the best split.
                                              'sqrt', 'log2', float (fraction), int (absolute number).
            class_weight (Optional[Union[str, Dict]]): Weights associated with classes. 'balanced' is useful
                                                       for imbalanced datasets. None gives equal weight.
            random_state (Optional[int]): Controls randomness for reproducibility.
            n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.
            **kwargs: Additional keyword arguments passed directly to RandomForestClassifier.
        """
        if target_type not in ['classification', 'binary']:
            raise ValueError("target_type must be 'classification' or 'binary'")

        self.target_type = target_type
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'oob_score': True, # Enable Out-of-Bag score for potential evaluation without test set
            **kwargs
        }

        # Initialize internal state
        self.model: Optional[RandomForestClassifier] = None
        self.is_fitted: bool = False
        self.feature_names_: Optional[List[str]] = None # Features seen during fit
        self.classes_: Optional[np.ndarray] = None      # Classes seen during fit (e.g., ['A', 'D', 'H'] or [0, 1])
        self.oob_score_: Optional[float] = None         # Out-of-Bag score after fitting

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the Random Forest model on the provided data.

        Args:
            X_train (pd.DataFrame): DataFrame of features for training. Must be numerical.
                                    NaN values should be handled before calling fit.
            y_train (pd.Series): Series of target labels ('H', 'D', 'A' for classification;
                                 0, 1 for binary).
        """
        print(f"Fitting RandomForestModel (target_type='{self.target_type}')...")

        # --- Input Validation ---
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            if isinstance(y_train, np.ndarray): y_train = pd.Series(y_train)
            else: raise TypeError("y_train must be a pandas Series or numpy array.")
        if X_train.isnull().values.any():
            # Consider adding an imputer here, but best practice is imputation upstream
            raise ValueError("X_train contains NaN values. Please handle missing data before fitting.")
        if y_train.isnull().values.any():
            raise ValueError("y_train contains NaN values.")
        if len(X_train) != len(y_train):
             raise ValueError("X_train and y_train must have the same number of samples.")

        # --- Store Metadata ---
        self.feature_names_ = list(X_train.columns)
        self.classes_ = np.unique(y_train)
        print(f"  Training data shape: {X_train.shape}")
        print(f"  Features used ({len(self.feature_names_)}): {self.feature_names_[:10]}...") # Show first few
        print(f"  Target classes found: {self.classes_}")
        if self.target_type == 'classification' and len(self.classes_) <= 2:
            warnings.warn(f"Target type is 'classification' but found only {len(self.classes_)} classes. Consider using 'binary'.")
        if self.target_type == 'binary' and len(self.classes_) > 2:
             raise ValueError(f"Target type is 'binary' but found {len(self.classes_)} classes: {self.classes_}")


        # --- Initialize and Train ---
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.oob_score_ = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None

        print("Fitting complete.")
        if self.oob_score_ is not None:
             print(f"  Out-of-Bag (OOB) Score Estimate: {self.oob_score_:.4f}")


    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions using the trained Random Forest model.

        Args:
            X_test (pd.DataFrame): DataFrame of features for prediction. Must have the
                                   same columns in the same order as X_train used for fitting.
                                   Handle NaNs beforehand.

        Returns:
            pd.DataFrame: A DataFrame containing predictions and probabilities.
                          Columns depend on `target_type`:
                          - 'classification': 'prediction', 'prob_H', 'prob_D', 'prob_A' (or similar based on classes_)
                          - 'binary': 'prediction', 'prob_0', 'prob_1'
        """
        check_is_fitted(self, ['model', 'feature_names_', 'classes_']) # Use scikit-learn's check

        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame.")

        # --- Feature Consistency Check ---
        if list(X_test.columns) != self.feature_names_:
            try:
                print("Warning: X_test columns order or names differ. Attempting to reorder/select...")
                X_test = X_test[self.feature_names_]
            except KeyError as e:
                missing = set(self.feature_names_) - set(X_test.columns)
                extra = set(X_test.columns) - set(self.feature_names_)
                raise ValueError(f"Feature mismatch: Missing {missing}, Extra {extra}") from e

        if X_test.isnull().values.any():
            # Again, imputation should ideally happen before calling predict
            raise ValueError("X_test contains NaN values. Please handle missing data before predicting.")

        # --- Predict ---
        print(f"Predicting on {len(X_test)} samples...")
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        # --- Format Output ---
        # Use self.classes_ which stores the order learned during fit
        prob_cols = [f"prob_{cls}" for cls in self.classes_]
        prob_df = pd.DataFrame(probabilities, columns=prob_cols, index=X_test.index)
        pred_df = pd.DataFrame(predictions, columns=['prediction'], index=X_test.index)

        results_df = pd.concat([pred_df, prob_df], axis=1)

        # Attempt to reorder probability columns to a standard if applicable
        if self.target_type == 'classification':
            standard_order = ['H', 'D', 'A']
            ordered_cols = ['prediction']
            present_classes = [str(c) for c in self.classes_] # Ensure string comparison works
            if all(str(c) in present_classes for c in standard_order):
                 ordered_cols.extend([f"prob_{c}" for c in standard_order])
            else: # Fallback to original order if standard classes aren't present
                 ordered_cols.extend(prob_cols)
            try:
                results_df = results_df[ordered_cols]
            except KeyError:
                 print("Warning: Could not reorder probability columns to H/D/A standard.")
                 # Keep original order if reordering fails

        print("Prediction complete.")
        return results_df

    def feature_importance(self) -> pd.Series:
        """
        Returns the feature importances (mean decrease in impurity) from the trained model.

        Returns:
            pd.Series: Feature importances, indexed by feature name, sorted descending.
        """
        check_is_fitted(self, ['model', 'feature_names_'])

        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)

    def save(self, filepath: str):
        """
        Saves the trained model state (model object, feature names, classes) using joblib.

        Args:
            filepath (str): The path to save the model file (e.g., 'rf_model.joblib').
        """
        if not self.is_fitted:
            raise NotFittedError("Cannot save an unfitted model.")
        print(f"Saving RandomForestModel state to {filepath}...")
        state = {
            'model': self.model,
            'feature_names_': self.feature_names_,
            'classes_': self.classes_,
            'target_type': self.target_type,
            'params': self.params # Save init params for potential inspection
        }
        joblib.dump(state, filepath)
        print("Model state saved successfully.")

    @classmethod
    def load(cls, filepath: str):
        """
        Loads a trained model state from a file.

        Args:
            filepath (str): The path to the saved model file.

        Returns:
            RandomForestModel: An instance of the class with the loaded state.
        """
        print(f"Loading RandomForestModel state from {filepath}...")
        state = joblib.load(filepath)
        # Create instance with saved config (or defaults if not saved)
        instance = cls(target_type=state.get('target_type', 'classification'), **state.get('params', {}))
        # Load the fitted attributes
        instance.model = state['model']
        instance.feature_names_ = state['feature_names_']
        instance.classes_ = state['classes_']
        instance.is_fitted = True
        instance.oob_score_ = instance.model.oob_score_ if hasattr(instance.model, 'oob_score_') else None
        print("Model state loaded successfully.")
        print(f"  Target type: {instance.target_type}")
        print(f"  Features expected ({len(instance.feature_names_)}): {instance.feature_names_[:10]}...")
        print(f"  Classes expected: {instance.classes_}")
        if instance.oob_score_ is not None:
             print(f"  Loaded OOB Score: {instance.oob_score_:.4f}")
        return instance

# Example Usage
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    # --- 1. Create Dummy Data (Numerical Features Only) ---
    print("\n--- RandomForestModel Example ---")
    np.random.seed(42)
    data_size = 500
    # Simulate features like odds, form, goal diff, maybe encoded team strength
    X = pd.DataFrame({
        'ImpliedProbH': np.random.uniform(0.2, 0.7, data_size),
        'ImpliedProbD': np.random.uniform(0.2, 0.4, data_size),
        'ImpliedProbA': 1.0 - (np.random.uniform(0.2, 0.7, data_size) + np.random.uniform(0.2, 0.4, data_size)), # Ensure sums roughly to 1
        'HomeFormPts_L5': np.random.randint(0, 16, data_size), # Points from last 5 games (0-15)
        'AwayFormPts_L5': np.random.randint(0, 16, data_size),
        'HomeGoalDiff_L5': np.random.randint(-5, 6, data_size),
        'AwayGoalDiff_L5': np.random.randint(-5, 6, data_size),
        'LeaguePosDiff': np.random.randint(-19, 20, data_size), # Diff in league position
        'H2H_HomeWinRatio': np.random.rand(data_size) # Historical win ratio in H2H
    })
    # Ensure ImpliedProbA is non-negative
    X['ImpliedProbA'] = X['ImpliedProbA'].clip(lower=0.05)
    X['ImpliedProbH'] = (1.0 - X['ImpliedProbD'] - X['ImpliedProbA']).clip(lower=0.05)


    # Simulate target variable (H/D/A) - loosely based on features
    prob_H_true = X['ImpliedProbH'] * 1.2 + (X['HomeFormPts_L5'] - X['AwayFormPts_L5']) / 30 - X['LeaguePosDiff'] / 40
    prob_A_true = X['ImpliedProbA'] * 1.2 + (X['AwayFormPts_L5'] - X['HomeFormPts_L5']) / 30 + X['LeaguePosDiff'] / 40
    prob_D_true = X['ImpliedProbD'] * 1.1 + (1 - abs(X['LeaguePosDiff'] / 20)) * 0.1

    total_prob = (prob_H_true + prob_D_true + prob_A_true).clip(lower=1e-6) # Avoid division by zero
    p_H = (prob_H_true / total_prob).clip(0, 1)
    p_D = (prob_D_true / total_prob).clip(0, 1)
    p_A = 1.0 - p_H - p_D # Ensure sum is 1

    y_cat = []
    for p_h, p_d, p_a in zip(p_H, p_D, p_A):
        # Ensure probabilities sum to 1 for choice
        probs = np.array([p_h, p_d, p_a])
        probs /= probs.sum()
        y_cat.append(np.random.choice(['H', 'D', 'A'], p=probs))
    y = pd.Series(y_cat)

    print("Dummy Data Generated (Features):")
    print(X.head())
    print("\nTarget Distribution:")
    print(y.value_counts(normalize=True))

    # --- 2. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # --- 3. Initialize and Fit Model ---
    rf_model = RandomForestModel(
        target_type='classification',
        n_estimators=200,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=8,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)

    # --- 4. Predict ---
    predictions_df = rf_model.predict(X_test)
    print("\nPredictions on Test Set (Top 5):")
    print(predictions_df.head())

    # --- 5. Evaluate ---
    print("\nEvaluation:")
    accuracy = accuracy_score(y_test, predictions_df['prediction'])
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    # Ensure labels match the order in the report if needed
    report = classification_report(y_test, predictions_df['prediction'], labels=rf_model.classes_)
    print(report)

    # --- 6. Feature Importance ---
    print("\nFeature Importances:")
    importances = rf_model.feature_importance()
    print(importances)

    # --- 7. Save and Load ---
    model_path = "temp_rf_model_prod.joblib"
    rf_model.save(model_path)
    loaded_model = RandomForestModel.load(model_path)

    # Verify loaded model makes same predictions
    loaded_predictions_df = loaded_model.predict(X_test)
    print("\nPredictions from Loaded Model (Top 5):")
    print(loaded_predictions_df.head())
    pd.testing.assert_frame_equal(predictions_df, loaded_predictions_df)
    print("\nSave/Load test passed.")

    # Clean up
    import os
    if os.path.exists(model_path): os.remove(model_path)