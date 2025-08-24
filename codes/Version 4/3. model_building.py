# 3. model_building.py (Final Corrected Version)
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, GlobalMaxPooling1D, Dense, 
                                     concatenate, Dropout, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                        ReduceLROnPlateau, TensorBoard)
import datetime

# --- Configuration ---
DATA_PATH = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset\processed_for_dl"
MODEL_SAVE_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
BATCH_SIZE = 256

# --- Custom Data Generator for Large Datasets ---
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, input_keys, target_key, batch_size, indices, prefix):
        self.data_path = data_path
        self.input_keys = input_keys
        self.target_key = target_key
        self.batch_size = batch_size
        self.indices = indices
        self.num_samples = len(indices)
        self.prefix = prefix # 'X_train_' or 'X_test_'
        
        self.inputs = {key: np.load(os.path.join(data_path, f'{self.prefix}{key}.npy'), mmap_mode='r') for key in self.input_keys}
        self.targets = np.load(os.path.join(data_path, f'{target_key}.npy'), mmap_mode='r')

    def __len__(self):
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        batch_start = index * self.batch_size
        batch_end = (index + 1) * self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        
        # --- CHANGE: Create the dictionary with the clean keys the model expects ---
        X = {key: self.inputs[key][batch_indices] for key in self.input_keys}
        y = self.targets[batch_indices]
        
        return X, y

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- Step 1: Get Data Indices ---
    print("Step 1: Getting data indices and file paths...")
    try:
        train_indices = np.arange(len(np.load(os.path.join(DATA_PATH, 'y_train.npy'), mmap_mode='r')))
        test_indices = np.arange(len(np.load(os.path.join(DATA_PATH, 'y_test.npy'), mmap_mode='r')))
        np.random.shuffle(train_indices)
        print(f"  - Found {len(train_indices)} training samples and {len(test_indices)} test samples.")
    except FileNotFoundError as e:
        print(f"  - Error finding data files: {e}. Please run the data preparation script first.")
        exit()

    # --- Step 2: Create Data Generators ---
    print("\nStep 2: Creating data generators to feed the model from disk...")
    model_input_keys = ['mirna_sequence_input', 'rre_sequence_input', 'rev_sequence_input', 
                       'mirna_structure_input', 'numerical_features_input']
    
    train_generator = DataGenerator(DATA_PATH, model_input_keys, 'y_train', BATCH_SIZE, train_indices, 'X_train_')
    test_generator = DataGenerator(DATA_PATH, model_input_keys, 'y_test', BATCH_SIZE, test_indices, 'X_test_')

    # --- Step 3: Model Architecture ---
    print("\nStep 3: Building the regression model architecture...")
    # Get a sample batch to determine input shapes
    sample_X, _ = train_generator[0]
    
    # Create Keras Input layers with explicit names
    input_layers = {
        key: Input(shape=sample_X[key].shape[1:], name=key) for key in model_input_keys
    }

    # Define the model architecture using the named layers
    x_mirna = GlobalMaxPooling1D()(BatchNormalization()(Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(input_layers['mirna_sequence_input'])))
    x_rre = GlobalMaxPooling1D()(BatchNormalization()(Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(input_layers['rre_sequence_input'])))
    x_rev = GlobalMaxPooling1D()(BatchNormalization()(Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(input_layers['rev_sequence_input'])))
    x_structure = GlobalMaxPooling1D()(Conv1D(filters=32, kernel_size=5, activation='relu')(input_layers['mirna_structure_input']))
    x_numerical = Dense(16, activation='relu')(input_layers['numerical_features_input'])

    combined = concatenate([x_mirna, x_rre, x_rev, x_structure, x_numerical])
    combined = Dropout(0.5)(combined)

    x = Dense(128, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name='affinity_output')(x)

    model = Model(inputs=input_layers, outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mean_squared_error', 
                  metrics=['mean_absolute_error'])
    model.summary()

    # --- Step 4: Callbacks ---
    print("\nStep 4: Defining callbacks for training...")
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_DIR, 'best_regression_model.keras'), 
                        save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    print(f"  - TensorBoard logs will be saved to: {log_dir}")

    # --- Step 5: Train the Model ---
    print("\nStep 5: Starting model training using data generators...")
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=test_generator,
        callbacks=callbacks,
        verbose=1
    )

    # --- Step 6: Save History ---
    print("\nStep 6: Saving training history...")
    history_path = os.path.join(MODEL_SAVE_DIR, 'training_history_regression.json')
    history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f)

    print(f"\n--- Training Complete ---")