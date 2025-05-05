# models/lstm_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import Model, load_model
from tensorflow import (
    Input, LSTM, Dense, Dropout, concatenate, TimeDistributed,
    BatchNormalization, Masking # Masking for variable length sequences if needed
)
from tensorflow import Adam
from tensorflow import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # For target encoding
# Assuming scaler objects are saved/loaded separately or handled in preprocessing
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Note: LSTMs don't fit the simple BaseModel interface well due to complex
# data shape and training procedures (epochs, batches, callbacks).
# We'll define fit/predict but acknowledge the different workflow.

class LSTMModel:
    """
    LSTM-based model for predicting football match outcomes or goals using
    sequences of past match data for both home and away teams.

    Assumes input data is preprocessed into 3D tensors:
    (batch_size, sequence_length, num_features)
    """

    def __init__(self, sequence_length: int, num_features: int,
                 lstm_units=64, dense_units=32, dropout_rate=0.3,
                 learning_rate=0.001, mode='classification', # 'classification' or 'regression'
                 num_classes=3): # Only used if mode='classification'
        """
        Initializes the LSTMModel configuration.

        Args:
            sequence_length (int): The number of past matches in each sequence (N).
            num_features (int): The number of features extracted from each past match.
            lstm_units (int): Number of units in the LSTM layers.
            dense_units (int): Number of units in the intermediate Dense layers.
            dropout_rate (float): Dropout rate for regularization.
            learning_rate (float): Learning rate for the Adam optimizer.
            mode (str): 'classification' (H/D/A) or 'regression' (e.g., goals).
            num_classes (int): Number of output classes for classification (e.g., 3 for H/D/A).
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.mode = mode
        self.num_classes = num_classes

        self.model = self._build_model()
        self._compile_model()
        self.is_compiled = True
        self.is_fitted = False
        self.target_encoder = None # To store encoder if used for classification

    def _build_model(self) -> Model:
        """Defines and builds the Keras LSTM model architecture."""
        print("Building LSTM model architecture...")

        # --- Input Layers ---
        # Input shape: (sequence_length, num_features)
        # We need two separate inputs: one sequence for the home team, one for the away team.
        home_input = Input(shape=(self.sequence_length, self.num_features), name='home_sequence_input')
        away_input = Input(shape=(self.sequence_length, self.num_features), name='away_sequence_input')

        # Optional: Masking layer if using variable length sequences (requires padding with a specific value)
        # home_masked = Masking(mask_value=0.0)(home_input) # Example if padding with 0
        # away_masked = Masking(mask_value=0.0)(away_input)

        # --- LSTM Layers ---
        # Option 1: Shared LSTM layer (learns general sequence patterns)
        shared_lstm = LSTM(self.lstm_units, return_sequences=False, name='shared_lstm') # return_sequences=False to get only the last output
        home_lstm = shared_lstm(home_input) # Use home_masked if using masking
        away_lstm = shared_lstm(away_input) # Use away_masked if using masking

        # Option 2: Separate LSTM layers (allows learning team-specific patterns)
        # home_lstm_layer = LSTM(self.lstm_units, return_sequences=False, name='home_lstm')
        # away_lstm_layer = LSTM(self.lstm_units, return_sequences=False, name='away_lstm')
        # home_lstm = home_lstm_layer(home_input)
        # away_lstm = away_lstm_layer(away_input)

        # --- Combine LSTM Outputs ---
        combined = concatenate([home_lstm, away_lstm], name='concatenate_lstm_outputs')

        # --- Dense Layers ---
        x = Dense(self.dense_units * 2, activation='relu', name='dense_1')(combined)
        x = Dropout(self.dropout_rate, name='dropout_1')(x)
        x = Dense(self.dense_units, activation='relu', name='dense_2')(x)
        x = Dropout(self.dropout_rate, name='dropout_2')(x)

        # --- Output Layer ---
        if self.mode == 'classification':
            output = Dense(self.num_classes, activation='softmax', name='output_classification')(x)
        elif self.mode == 'regression':
            # Output 1 unit (e.g., total goals) or 2 units (home_goals, away_goals)
            output = Dense(2, activation='linear', name='output_regression')(x) # Example: predict home and away goals
            # Consider ReLU if goals cannot be negative: Dense(2, activation='relu', ...)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose 'classification' or 'regression'.")

        # --- Create Model ---
        model = Model(inputs=[home_input, away_input], outputs=output, name='Football_LSTM_Model')
        print("Model built successfully.")
        model.summary() # Print model summary
        return model

    def _compile_model(self):
        """Compiles the Keras model with optimizer and loss function."""
        if self.model is None:
            raise RuntimeError("Model has not been built yet.")

        optimizer = Adam(learning_rate=self.learning_rate)

        if self.mode == 'classification':
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        elif self.mode == 'regression':
            loss = 'mean_squared_error' # or 'mean_absolute_error'
            metrics = ['mae', 'mse'] # Mean Absolute Error, Mean Squared Error
        else:
            # Should not happen due to check in _build_model
             raise ValueError(f"Invalid mode: {self.mode}")

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"Model compiled for {self.mode} with loss='{loss}', optimizer='Adam', metrics={metrics}")

    def fit(self, X_train_home_seq: np.ndarray, X_train_away_seq: np.ndarray, y_train: np.ndarray,
            epochs=50, batch_size=32, validation_split=0.2, callbacks=None):
        """
        Trains the LSTM model.

        Args:
            X_train_home_seq (np.ndarray): 3D array of home team sequences
                                           (samples, sequence_length, num_features).
            X_train_away_seq (np.ndarray): 3D array of away team sequences
                                           (samples, sequence_length, num_features).
            y_train (np.ndarray): Target values.
                                  - For classification: One-hot encoded array (samples, num_classes).
                                  - For regression: Array of target values (samples, num_outputs).
            epochs (int): Number of training epochs.
            batch_size (int): Number of samples per gradient update.
            validation_split (float): Fraction of training data to use as validation set.
            callbacks (list, optional): List of Keras callbacks (e.g., EarlyStopping).
                                        Defaults to None.
        """
        if not self.is_compiled:
            self._compile_model()
        if X_train_home_seq.shape[1:] != (self.sequence_length, self.num_features) or \
           X_train_away_seq.shape[1:] != (self.sequence_length, self.num_features):
            raise ValueError(f"Input sequence shapes must be (samples, {self.sequence_length}, {self.num_features})")

        # --- Target Encoding (Classification Only) ---
        if self.mode == 'classification' and y_train.ndim == 1:
             print("Detected 1D target for classification. Performing one-hot encoding.")
             # Use OneHotEncoder to handle potential unseen labels gracefully if needed,
             # or simple pandas get_dummies/keras to_categorical if labels are fixed ('H','D','A')
             # Example using Keras to_categorical assuming y_train contains integer labels 0, 1, 2
             # If y_train is 'H', 'D', 'A', need LabelEncoder first
             le = LabelEncoder()
             y_int = le.fit_transform(y_train)
             y_train_encoded = tf.keras.utils.to_categorical(y_int, num_classes=self.num_classes)
             self.target_encoder = le # Store the label encoder
             print(f"Target classes learned by LabelEncoder: {self.target_encoder.classes_}")
        else:
             y_train_encoded = y_train # Assume already correctly formatted

        # --- Default Callbacks ---
        if callbacks is None:
            callbacks = []
            # Add early stopping by default to prevent overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            callbacks.append(early_stopping)
            print("Added default EarlyStopping callback (monitor='val_loss', patience=10).")

        print(f"Starting training for {epochs} epochs with batch size {batch_size}...")
        history = self.model.fit(
            [X_train_home_seq, X_train_away_seq], # Input is a list of the two sequence arrays
            y_train_encoded,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1 # Show progress bar
        )
        self.is_fitted = True
        print("Training complete.")
        return history

    def predict(self, X_home_seq: np.ndarray, X_away_seq: np.ndarray) -> pd.DataFrame:
        """
        Makes predictions using the trained LSTM model.

        Args:
            X_home_seq (np.ndarray): 3D array of home team sequences for prediction.
            X_away_seq (np.ndarray): 3D array of away team sequences for prediction.

        Returns:
            pd.DataFrame: DataFrame containing predictions.
                          - Classification: Columns 'prediction' (H/D/A), 'prob_H', 'prob_D', 'prob_A'.
                          - Regression: Columns like 'pred_home_goals', 'pred_away_goals'.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        if X_home_seq.shape[1:] != (self.sequence_length, self.num_features) or \
           X_away_seq.shape[1:] != (self.sequence_length, self.num_features):
            raise ValueError(f"Input sequence shapes must be (samples, {self.sequence_length}, {self.num_features})")

        print(f"Making predictions on {len(X_home_seq)} samples...")
        raw_predictions = self.model.predict([X_home_seq, X_away_seq])

        if self.mode == 'classification':
            if self.target_encoder is None:
                 # Assume classes are 0, 1, ..., num_classes-1 if no encoder stored
                 predicted_indices = np.argmax(raw_predictions, axis=1)
                 predicted_labels = predicted_indices # Or map back if needed
                 class_labels = [f'class_{i}' for i in range(self.num_classes)]
                 print("Warning: target_encoder not found. Using class indices as labels.")
            else:
                 predicted_indices = np.argmax(raw_predictions, axis=1)
                 predicted_labels = self.target_encoder.inverse_transform(predicted_indices)
                 class_labels = self.target_encoder.classes_

            # Create DataFrame
            results_df = pd.DataFrame(raw_predictions, columns=[f"prob_{label}" for label in class_labels])
            results_df['prediction'] = predicted_labels

            # Reorder columns for consistency (prediction first, then probs H/D/A if possible)
            prob_cols_ordered = []
            standard_order = ['H', 'D', 'A'] # Desired order
            if all(c in class_labels for c in standard_order):
                 prob_cols_ordered = [f"prob_{label}" for label in standard_order]
            else:
                 prob_cols_ordered = list(results_df.columns)
                 if 'prediction' in prob_cols_ordered: prob_cols_ordered.remove('prediction')

            final_cols = ['prediction'] + prob_cols_ordered
            results_df = results_df[final_cols]


        elif self.mode == 'regression':
            # Assuming 2 outputs: home_goals, away_goals
            results_df = pd.DataFrame(raw_predictions, columns=['pred_home_goals', 'pred_away_goals'])
        else:
             # Should not happen
             raise ValueError(f"Invalid mode: {self.mode}")

        print("Predictions generated.")
        return results_df

    def save(self, filepath: str):
        """
        Saves the trained Keras model and target encoder (if applicable).

        Args:
            filepath (str): Base path/filename for saving. Model saved as filepath.h5,
                            encoder saved as filepath_encoder.joblib.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        model_path = f"{filepath}.h5"
        print(f"Saving Keras model to {model_path}...")
        self.model.save(model_path)
        print("Model saved.")

        if self.target_encoder:
            import joblib
            encoder_path = f"{filepath}_encoder.joblib"
            print(f"Saving target encoder to {encoder_path}...")
            joblib.dump(self.target_encoder, encoder_path)
            print("Encoder saved.")

    @classmethod
    def load(cls, filepath: str, compile_model=True):
        """
        Loads a trained Keras model and target encoder (if applicable).

        Args:
            filepath (str): Base path/filename used during saving.
            compile_model (bool): Whether to compile the loaded model. Set to False
                                  if you only want to use it for inference immediately.

        Returns:
            LSTMModel: An instance of the class with the loaded state.
        """
        model_path = f"{filepath}.h5"
        print(f"Loading Keras model from {model_path}...")
        loaded_keras_model = load_model(model_path, compile=compile_model)
        print("Model loaded.")

        # Infer parameters from the loaded model's structure
        try:
            # Input shape gives sequence length and features
            # Assumes both inputs have the same shape, which our _build_model ensures
            input_shape = loaded_keras_model.input_shape[0] # Shape of first input tensor
            sequence_length = input_shape[1]
            num_features = input_shape[2]
            # Output shape gives mode/num_classes
            output_shape = loaded_keras_model.output_shape
            if len(output_shape) == 2:
                if output_shape[1] > 1 and loaded_keras_model.layers[-1].activation.__name__ == 'softmax':
                    mode = 'classification'
                    num_classes = output_shape[1]
                else:
                    mode = 'regression'
                    num_classes = 3 # Default, not strictly used in regression mode init
            else: # Should not happen with dense output
                 raise ValueError("Cannot infer mode from model output shape")

            # Create an instance wrapper - hyperparameters like units/dropout aren't easily
            # retrieved, so we use the inferred structural ones.
            # The actual trained weights and architecture are in loaded_keras_model.
            instance = cls(sequence_length=sequence_length, num_features=num_features, mode=mode, num_classes=num_classes)
            instance.model = loaded_keras_model
            instance.is_fitted = True
            instance.is_compiled = compile_model

        except Exception as e:
            print(f"Error inferring parameters from loaded model: {e}")
            print("Please ensure the model architecture matches the expected structure.")
            raise

        # Load encoder if it exists
        encoder_path = f"{filepath}_encoder.joblib"
        try:
            import joblib
            import os
            if os.path.exists(encoder_path):
                print(f"Loading target encoder from {encoder_path}...")
                instance.target_encoder = joblib.load(encoder_path)
                print("Encoder loaded.")
            else:
                 print("No target encoder file found.")
                 instance.target_encoder = None
        except ImportError:
             print("joblib not installed, cannot load encoder.")
        except Exception as e:
             print(f"Error loading encoder: {e}")


        return instance

# --- Example Usage Placeholder ---
if __name__ == '__main__':
    print("\n--- LSTM Model Example Usage ---")

    # --- 1. Define Parameters ---
    SEQ_LENGTH = 10 # Look at last 10 games
    NUM_FEATURES = 5 # e.g., GoalsFor, GoalsAgainst, Result (0/1/2), HomeOdds, AwayOdds
    NUM_SAMPLES = 100 # Number of matches in our dataset
    MODE = 'classification' # 'classification' or 'regression'
    NUM_CLASSES = 3 # For H/D/A

    # --- 2. Create Dummy Data (CRITICAL: Requires careful preprocessing in reality) ---
    # Shape: (samples, sequence_length, num_features)
    # Need two input arrays: one for home team sequences, one for away team sequences
    print("Generating dummy data...")
    X_home_dummy = np.random.rand(NUM_SAMPLES, SEQ_LENGTH, NUM_FEATURES)
    X_away_dummy = np.random.rand(NUM_SAMPLES, SEQ_LENGTH, NUM_FEATURES)

    # Dummy Target Data
    if MODE == 'classification':
        # Integer labels 0, 1, 2 representing H, D, A
        y_dummy_int = np.random.randint(0, NUM_CLASSES, NUM_SAMPLES)
        # Convert to string labels for realistic scenario before fit handles encoding
        label_map = {0: 'H', 1: 'D', 2: 'A'}
        y_dummy = np.array([label_map[i] for i in y_dummy_int])
        print(f"Dummy target (classification): {y_dummy[:10]}...")
    else: # Regression
        # Predicting home goals and away goals
        y_dummy = np.random.rand(NUM_SAMPLES, 2) * 3 # Predict values between 0 and 3
        print(f"Dummy target (regression): {y_dummy[:5]}...")
    print("Dummy data generated.")


    # --- 3. Initialize Model ---
    lstm_model = LSTMModel(
        sequence_length=SEQ_LENGTH,
        num_features=NUM_FEATURES,
        lstm_units=32,
        dense_units=16,
        dropout_rate=0.2,
        mode=MODE,
        num_classes=NUM_CLASSES
    )

    # --- 4. Fit Model (using validation split) ---
    # In reality, use separate train/validation sets prepared beforehand
    print("\nStarting dummy training...")
    try:
        history = lstm_model.fit(
            X_home_dummy, X_away_dummy, y_dummy,
            epochs=5, # Use small epochs for quick example
            batch_size=16,
            validation_split=0.2 # Use part of the dummy data for validation
        )
        print("Dummy training finished.")

        # --- 5. Predict ---
        # Predict on the first few samples as a test
        num_pred_samples = 5
        predictions_df = lstm_model.predict(X_home_dummy[:num_pred_samples], X_away_dummy[:num_pred_samples])
        print(f"\nPredictions for first {num_pred_samples} samples:")
        print(predictions_df)

        # --- 6. Save and Load ---
        model_path = "temp_lstm_model"
        lstm_model.save(model_path)
        loaded_lstm_model = LSTMModel.load(model_path)

        # Verify loaded model prediction
        loaded_predictions_df = loaded_lstm_model.predict(X_home_dummy[:num_pred_samples], X_away_dummy[:num_pred_samples])
        print(f"\nPredictions from loaded model:")
        print(loaded_predictions_df)
        # Compare DataFrames (allow for small float differences)
        pd.testing.assert_frame_equal(predictions_df, loaded_predictions_df, check_exact=False, atol=1e-5)
        print("\nSave/Load test passed.")

        # Clean up
        import os
        if os.path.exists(f"{model_path}.h5"): os.remove(f"{model_path}.h5")
        if os.path.exists(f"{model_path}_encoder.joblib"): os.remove(f"{model_path}_encoder.joblib")

    except Exception as e:
        print(f"\nAn error occurred during example execution: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Deep learning models require specific library versions (TensorFlow/Keras).")
        print("Ensure TensorFlow is installed correctly (`pip install tensorflow`).")