# 3b_incremental_training.py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                        ReduceLROnPlateau, TensorBoard)
import datetime

# --- IMPORTANT: This is the DataGenerator class copied from the original script ---
# It's needed here to load the new data.
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, input_keys, target_key, batch_size, indices, prefix):
        self.data_path = data_path
        self.input_keys = input_keys
        self.target_key = target_key
        self.batch_size = batch_size
        self.indices = indices
        self.num_samples = len(indices)
        self.prefix = prefix
        self.inputs = {key: np.load(os.path.join(data_path, f'{self.prefix}{key}.npy'), mmap_mode='r') for key in self.input_keys}
        self.targets = np.load(os.path.join(data_path, f'{target_key}.npy'), mmap_mode='r')
    def __len__(self):
        return int(np.floor(self.num_samples / self.batch_size))
    def __getitem__(self, index):
        batch_start, batch_end = index * self.batch_size, (index + 1) * self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        X = {key: self.inputs[key][batch_indices] for key in self.input_keys}
        y = self.targets[batch_indices]
        return X, y

# --- Configuration ---
# --- ACTION: Define the paths for your INCREMENTAL training session ---
# Path to the NEW data you prepared
NEW_DATA_PATH = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset\processed_for_dl_zika"
# Path to the model you ALREADY trained
EXISTING_MODEL_PATH = r"E:\1. miRNA-RNA-Deep-Learning-Model\models\best_regression_model.keras"
# Where to save the NEW, updated model
MODEL_SAVE_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\models"
BATCH_SIZE = 256

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Step 1: Load the PREVIOUSLY trained model ---
    print(f"--- Starting Incremental Training ---")
    print(f"Loading existing model from: {EXISTING_MODEL_PATH}")
    try:
        model = load_model(EXISTING_MODEL_PATH)
        print("  - Model loaded successfully.")
    except Exception as e:
        print(f"  - Error loading model: {e}")
        exit()

    # --- Step 2: Prepare the NEW data generators ---
    print(f"\nPreparing new data from: {NEW_DATA_PATH}")
    try:
        train_indices = np.arange(len(np.load(os.path.join(NEW_DATA_PATH, 'y_train.npy'), mmap_mode='r')))
        test_indices = np.arange(len(np.load(os.path.join(NEW_DATA_PATH, 'y_test.npy'), mmap_mode='r')))
        np.random.shuffle(train_indices)
    except FileNotFoundError as e:
        print(f"  - Error finding new data files: {e}. Please run data prep for the new dataset.")
        exit()

    model_input_keys = ['mirna_sequence_input', 'rre_sequence_input', 'rev_sequence_input', 
                       'mirna_structure_input', 'numerical_features_input']
    
    new_train_generator = DataGenerator(NEW_DATA_PATH, model_input_keys, 'y_train', BATCH_SIZE, train_indices, 'X_train_')
    new_test_generator = DataGenerator(NEW_DATA_PATH, model_input_keys, 'y_test', BATCH_SIZE, test_indices, 'X_test_')

    # --- Step 3: Re-compile the model with a VERY LOW learning rate ---
    # This is the key to fine-tuning without "catastrophic forgetting".
    print("\nRe-compiling model with a low learning rate for fine-tuning...")
    LOW_LEARNING_RATE = 0.00005
    model.compile(optimizer=Adam(learning_rate=LOW_LEARNING_RATE), 
                  loss='mean_squared_error', 
                  metrics=['mean_absolute_error'])
    model.summary()

    # --- Step 4: Set up new callbacks ---
    # We save the new model with a new name to avoid overwriting the original
    new_model_filepath = os.path.join(MODEL_SAVE_DIR, 'best_model_incrementally_trained.keras')
    log_dir = os.path.join("logs", "fit", f"incremental_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    callbacks = [
        ModelCheckpoint(filepath=new_model_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    # --- Step 5: Continue training on the NEW data ---
    print("\nStarting fine-tuning on the new dataset...")
    history = model.fit(
        new_train_generator,
        epochs=50, # Fine-tuning usually requires fewer epochs
        validation_data=new_test_generator,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n--- Incremental Training Complete ---")
    print(f"The newly fine-tuned model has been saved to: {new_model_filepath}")