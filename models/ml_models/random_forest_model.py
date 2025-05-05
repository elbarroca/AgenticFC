# models/random_forest_model.py

import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted # Good practice for fitted checks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import LabelEncoder # To convert H/D/A to 0/1/2
import joblib
from typing import List, Optional, Dict, Any, Union

# Assuming you have a base_model.py defining the interface
# If not, you can remove the inheritance
from base_model import BaseModel

class RandomForestModel(BaseModel):
    """
    Random Forest Classifier model for predicting soccer match outcomes (e.g., H/D/A)
    or binary targets (e.g., Over/Under 2.5, BTTS) using structured, tabular data.

    Leverages scikit-learn's RandomForestClassifier for robust predictions.
    Assumes input features (X) are preprocessed (e.g., imputed) and numerical.
    Handles both binary and multiclass classification targets.
    """

    def __init__(self,
                 target_type: str = 'classification', # 'classification' (multiclass) or 'binary'
                 n_estimators: int = 150,
                 max_depth: Optional[int] = 15,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 5,
                 max_features: Union[str, float] = 'sqrt',
                 # 'balanced_subsample' often preferred for large datasets and performance
                 class_weight: Optional[Union[str, Dict]] = 'balanced_subsample',
                 random_state: Optional[int] = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initializes the RandomForestModel.

        Args:
            target_type (str): Type of prediction task: 'classification' (multiclass, e.g., H/D/A)
                               or 'binary' (e.g., O/U, BTTS). Primarily for validation.
            n_estimators (int): The number of trees in the forest.
            max_depth (Optional[int]): Maximum depth of the trees. None means nodes expand until pure.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
            max_features (Union[str, float]): Number of features to consider when looking for the best split.
            class_weight (Optional[Union[str, Dict]]): Weights associated with classes. 'balanced' or
                                                       'balanced_subsample' are useful for imbalanced datasets.
                                                       None gives equal weight.
            random_state (Optional[int]): Controls randomness for reproducibility.
            n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.
            **kwargs: Additional keyword arguments passed directly to RandomForestClassifier
                      (e.g., `criterion='entropy'`, `bootstrap=True`).
        """
        if target_type not in ['classification', 'binary']:
            raise ValueError("target_type must be 'classification' or 'binary'")

        self.target_type = target_type
        # Consolidate parameters for the underlying scikit-learn model
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': n_jobs,
            # Enable OOB score calculation during fit for model evaluation without a separate validation set
            'oob_score': True,
            'verbose': 0, # Keep console clean unless debugging
            **kwargs # Pass any extra valid RF parameters
        }

        # Initialize internal state attributes (following scikit-learn conventions)
        self.model: Optional[RandomForestClassifier] = None
        self.is_fitted: bool = False
        self.feature_names_: Optional[List[str]] = None # Features seen during fit
        self.classes_: Optional[np.ndarray] = None      # Classes seen during fit (e.g., [0, 1] or [0, 1, 2] mapping to ['A', 'D', 'H'])
        self.oob_score_: Optional[float] = None         # Out-of-Bag score after fitting

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the Random Forest model on the provided data.

        Args:
            X_train (pd.DataFrame): DataFrame of features for training. Must be numerical.
                                    NaN values should be handled *before* calling fit.
            y_train (pd.Series): Series of target labels (e.g., 0, 1 for binary;
                                 0, 1, 2 for multiclass mapped from H/D/A).
                                 Make sure labels are numerical for RF.
        """
        print(f"Fitting RandomForestModel (target_type='{self.target_type}')...")

        # --- Input Validation ---
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            # Allow numpy array but convert to Series internally for consistency
            if isinstance(y_train, np.ndarray): y_train = pd.Series(y_train, name="target")
            else: raise TypeError("y_train must be a pandas Series or numpy array.")
        if X_train.isnull().values.any():
            # Crucial check: RF cannot handle NaNs directly.
            warnings.warn("X_train contains NaN values. Ensure data is imputed before fitting for reliable results.")
            # Depending on policy, could raise ValueError instead.
            # raise ValueError("X_train contains NaN values. Please handle missing data before fitting.")
        if y_train.isnull().values.any():
            raise ValueError("y_train contains NaN values.")
        if len(X_train) != len(y_train):
             raise ValueError("X_train and y_train must have the same number of samples.")
        if not pd.api.types.is_numeric_dtype(y_train):
             raise ValueError("y_train must be numerical (e.g., 0, 1 or 0, 1, 2). Map categorical targets first.")

        # --- Store Metadata ---
        # Use list() to ensure it's a basic list, not a pandas Index object when saved
        self.feature_names_ = list(X_train.columns)
        self.classes_ = np.sort(y_train.unique()) # Store sorted unique classes seen
        print(f"  Training data shape: {X_train.shape}")
        print(f"  Features used ({len(self.feature_names_)}): {self.feature_names_[:min(10, len(self.feature_names_))]}...") # Show first few
        print(f"  Target classes found: {self.classes_}")

        # Validate target_type against actual classes found
        if self.target_type == 'classification' and len(self.classes_) <= 2:
            warnings.warn(f"Target type is 'classification' but found only {len(self.classes_)} classes: {self.classes_}. Consider using 'binary' type.")
        if self.target_type == 'binary' and len(self.classes_) > 2:
             raise ValueError(f"Target type is 'binary' but found {len(self.classes_)} classes: {self.classes_}")
        if self.target_type == 'binary' and not np.array_equal(self.classes_, [0, 1]):
             # Check if binary classes are exactly 0 and 1, often assumed by metrics/usage
             warnings.warn(f"Target type is 'binary' but classes are {self.classes_}. Ensure this matches expectations.")


        # --- Initialize and Train ---
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train.values, y_train.values) # Use .values for performance and compatibility
        self.is_fitted = True
        # Store OOB score if calculated
        self.oob_score_ = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None

        print("Fitting complete.")
        if self.oob_score_ is not None:
             # OOB score is an estimate of generalization accuracy on unseen data
             print(f"  Out-of-Bag (OOB) Score Estimate: {self.oob_score_:.4f}")


    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions (class labels and probabilities) using the trained model.

        Args:
            X_test (pd.DataFrame): DataFrame of features for prediction. Must have the
                                   same columns (or a superset) as X_train used for fitting.
                                   NaN values should be handled *before* calling predict.

        Returns:
            pd.DataFrame: A DataFrame containing predictions and probabilities.
                          Columns:
                          - 'prediction': The predicted class label (matching the fitted y_train labels).
                          - 'prob_{class}': Probability for each class seen during training
                                            (e.g., 'prob_0', 'prob_1', or 'prob_0', 'prob_1', 'prob_2').
        """
        # Ensure model is fitted before predicting
        check_is_fitted(self, ['model', 'feature_names_', 'classes_'])

        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame.")

        # --- Feature Consistency Check ---
        X_test_processed = X_test.copy()
        if list(X_test_processed.columns) != self.feature_names_:
            try:
                # Attempt to select and reorder columns to match training features
                print("Warning: X_test columns order or names differ. Attempting to reorder/select features.")
                X_test_processed = X_test_processed[self.feature_names_]
            except KeyError as e:
                missing = set(self.feature_names_) - set(X_test_processed.columns)
                extra = set(X_test_processed.columns) - set(self.feature_names_)
                err_msg = "Feature mismatch during predict:"
                if missing: err_msg += f" Missing columns in X_test: {missing}."
                if extra: err_msg += f" Extra columns in X_test: {extra}."
                raise ValueError(err_msg) from e

        if X_test_processed.isnull().values.any():
            warnings.warn("X_test contains NaN values. Ensure data is imputed before predicting for reliable results.")
            # Consider raising ValueError or implementing imputation within predict if policy allows
            # raise ValueError("X_test contains NaN values. Please handle missing data.")

        # --- Predict ---
        print(f"Predicting on {len(X_test_processed)} samples...")
        # Use .values for prediction consistency
        predictions_array = self.model.predict(X_test_processed.values)
        probabilities_array = self.model.predict_proba(X_test_processed.values)

        # --- Format Output ---
        # Use self.classes_ which stores the order learned during fit
        prob_cols = [f"prob_{cls}" for cls in self.classes_]
        prob_df = pd.DataFrame(probabilities_array, columns=prob_cols, index=X_test_processed.index)
        pred_df = pd.DataFrame(predictions_array, columns=['prediction'], index=X_test_processed.index)

        # Combine prediction and probabilities
        results_df = pd.concat([pred_df, prob_df], axis=1)

        print("Prediction complete.")
        return results_df

    def feature_importance(self) -> pd.Series:
        """
        Returns the feature importances (mean decrease in impurity) from the trained model.

        Returns:
            pd.Series: Feature importances, indexed by feature name, sorted descending.
                       Returns empty Series if model is not fitted.
        """
        if not self.is_fitted or self.model is None or self.feature_names_ is None:
            warnings.warn("Model not fitted yet. Cannot provide feature importances.")
            return pd.Series(dtype=float)

        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)

    def save(self, filepath: str):
        """
        Saves the trained model state (model object, feature names, classes, etc.) using joblib.

        Args:
            filepath (str): The path to save the model file (e.g., 'rf_model.joblib').
        """
        if not self.is_fitted:
            raise NotFittedError("Cannot save an unfitted model. Call fit() first.")
        print(f"Saving RandomForestModel state to {filepath}...")
        # Include all necessary attributes to reconstruct the state upon loading
        state = {
            'model': self.model,
            'feature_names_': self.feature_names_,
            'classes_': self.classes_,
            'target_type': self.target_type,
            'params': self.params, # Save init params for inspection/re-init
            'oob_score_': self.oob_score_
        }
        try:
            joblib.dump(state, filepath)
            print("Model state saved successfully.")
        except Exception as e:
            print(f"Error saving model state: {e}")
            raise

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
        try:
            state = joblib.load(filepath)
        except Exception as e:
            print(f"Error loading model state from {filepath}: {e}")
            raise

        # Basic validation of the loaded state
        required_keys = ['model', 'feature_names_', 'classes_', 'target_type', 'params']
        if not all(key in state for key in required_keys):
            missing = set(required_keys) - set(state.keys())
            raise ValueError(f"Loaded state is missing required keys: {missing}")

        # Create instance using saved parameters
        instance = cls(target_type=state['target_type'], **state['params'])

        # Load the fitted attributes directly
        instance.model = state['model']
        instance.feature_names_ = state['feature_names_']
        instance.classes_ = state['classes_']
        instance.oob_score_ = state.get('oob_score_') # Use .get for backward compatibility if oob was not saved
        instance.is_fitted = True # Mark as fitted

        print("Model state loaded successfully.")
        print(f"  Target type: {instance.target_type}")
        print(f"  Features expected ({len(instance.feature_names_)}): {instance.feature_names_[:min(10, len(instance.feature_names_))]}...")
        print(f"  Classes expected: {instance.classes_}")
        if instance.oob_score_ is not None:
             print(f"  Loaded OOB Score: {instance.oob_score_:.4f}")
        return instance

# Example Usage (similar to before, using this class structure)
if __name__ == '__main__':


    # --- 1. Create Dummy Data (Numerical Features Only) ---
    print("\n--- RandomForestModel Production Example ---")
    np.random.seed(42)
    data_size = 1000
    X = pd.DataFrame({
        'Stat_Home_Attack': np.random.normal(1.0, 0.2, data_size),
        'Stat_Away_Defense': np.random.normal(1.0, 0.2, data_size),
        'Stat_Away_Attack': np.random.normal(1.0, 0.2, data_size),
        'Stat_Home_Defense': np.random.normal(1.0, 0.2, data_size),
        'FormDiff_L5': np.random.randint(-10, 11, data_size),
        'GoalDiff_L10': np.random.randint(-15, 16, data_size),
        # Add a feature that might have NaNs to test handling (though fit expects no NaNs)
        # 'Maybe_Missing_Feature': np.random.choice([1.0, 2.0, np.nan], size=data_size, p=[0.5, 0.4, 0.1])
    })

    # Simulate target variable (H/D/A) - numerical 0, 1, 2
    lambda_h = (X['Stat_Home_Attack'] * X['Stat_Away_Defense'] * 1.4 + X['FormDiff_L5'] / 20).clip(0.1)
    lambda_a = (X['Stat_Away_Attack'] * X['Stat_Home_Defense'] * 1.1 - X['FormDiff_L5'] / 25).clip(0.1)
    # Simulate more draws when teams are closer in form
    p_adjust = np.exp(-abs(X['FormDiff_L5'] / 5)) * 0.3

    p_H = lambda_h / (lambda_h + lambda_a + p_adjust*2)
    p_A = lambda_a / (lambda_h + lambda_a + p_adjust*2)
    p_D = 1.0 - p_H - p_A

    y_cat = []
    for p_h, p_d, p_a in zip(p_H, p_D, p_A):
        probs = np.array([p_h, p_d, p_a]).clip(0) # Ensure non-negative
        probs /= probs.sum() # Ensure sums to 1
        y_cat.append(np.random.choice(['H', 'D', 'A'], p=probs))

    # Encode H/D/A to 0/1/2 for the model
    # Encode target labels using LabelEncoder
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_cat), name="FTR_Numeric") 
    print("Target mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    # --- 2. Handle NaNs before splitting (if added dummy NaNs) ---
    # Example: Impute before splitting (often better to do this within CV)
    # if 'Maybe_Missing_Feature' in X.columns:
    #     imputer = SimpleImputer(strategy='median')
    #     X['Maybe_Missing_Feature'] = imputer.fit_transform(X[['Maybe_Missing_Feature']])

    # --- 3. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    print(f"\nSplit: Train={len(X_train)}, Test={len(X_test)}")

    # --- 4. Initialize and Fit Model ---
    rf_model_hda = RandomForestModel(
        target_type='classification',
        n_estimators=200,       # More trees for better performance
        max_depth=10,           # Limit depth to prevent overfitting
        min_samples_split=20,   # Require more samples to split
        min_samples_leaf=10,    # Require more samples per leaf
        class_weight='balanced_subsample', # Handle imbalanced H/D/A classes
        random_state=42         # Use consistent random_state instead of RANDOM_SEED
    )
    
    # Fit the model
    rf_model_hda.fit(X_train, y_train)

    # --- 5. Make predictions ---
    predictions_df = rf_model_hda.predict(X_test)
    print("\nPredictions on Test Set (Top 5):")
    print(predictions_df.head())

    # --- 6. Evaluate ---
    print("\nEvaluation:")
    accuracy = accuracy_score(y_test, predictions_df['prediction'])
    print(f"Accuracy: {accuracy:.4f}")
    print(f"OOB Score: {rf_model_hda.oob_score_:.4f}") # Compare accuracy to OOB estimate

    # Classification Report needs original labels
    y_test_labels = label_encoder.inverse_transform(y_test)
    pred_labels = label_encoder.inverse_transform(predictions_df['prediction'])
    print("Classification Report:")
    print(classification_report(y_test_labels, pred_labels, labels=['H', 'D', 'A']))

    # Evaluate Probabilities
    # Ensure prob columns exist and match class order
    prob_cols_ordered = [f"prob_{c}" for c in rf_model_hda.classes_]
    if all(c in predictions_df.columns for c in prob_cols_ordered):
        # Log Loss
        logloss = log_loss(y_test, predictions_df[prob_cols_ordered], labels=rf_model_hda.classes_)
        print(f"Log Loss: {logloss:.4f}")

        # Brier Score (Multiclass version)
        # One-hot encode y_test
        y_test_one_hot = pd.get_dummies(y_test).values
        # Ensure predictions_df has probabilities for all classes present in y_test
        brier_multi = np.mean(np.sum((predictions_df[prob_cols_ordered].values - y_test_one_hot)**2, axis=1))
        print(f"Brier Score (Multiclass): {brier_multi:.4f}")
    else:
        print("Warning: Could not evaluate probability metrics due to missing columns.")

    # --- 7. Feature Importance ---
    print("\nFeature Importances:")
    print(rf_model_hda.feature_importance())

    # --- 8. Save and Load ---
    model_path = "temp_rf_model_prod_v2.joblib"
    rf_model_hda.save(model_path)
    loaded_model = RandomForestModel.load(model_path)

    # --- 9. Verify Loaded Model ---
    loaded_predictions_df = loaded_model.predict(X_test)
    print("\nPredictions from Loaded Model (Top 5):")
    print(loaded_predictions_df.head())
    # Assertions for verification
    pd.testing.assert_frame_equal(predictions_df, loaded_predictions_df)
    assert rf_model_hda.oob_score_ == loaded_model.oob_score_
    print("\nSave/Load test passed.")

    # Clean up
    import os
    if os.path.exists(model_path): os.remove(model_path)