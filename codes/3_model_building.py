# 3_model_building.py (Supreme Hybrid Architecture Version)
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, Dense, Dropout, BatchNormalization,
                                     concatenate, Bidirectional, LSTM,
                                     MultiHeadAttention, GlobalAveragePooling1D, LayerNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard)
import datetime

# --- Configuration Loader ---
def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir) 
        config_path = os.path.join(project_root, 'config.json')
    print(f"--- Loading configuration from: {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'.")
        exit()

# --- Custom Data Generator with Sample Weighting ---
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, input_keys, target_key, batch_size, indices, prefix, 
                 use_sample_weights=False, weight_alpha=10.0):
        self.data_path = data_path
        self.input_keys = input_keys
        self.target_key = target_key
        self.batch_size = batch_size
        self.indices = indices
        self.prefix = prefix
        self.use_sample_weights = use_sample_weights
        self.weight_alpha = weight_alpha
        
        # Use memory-mapping for large datasets
        self.inputs = {key: np.load(os.path.join(data_path, f'{self.prefix}{key}.npy'), mmap_mode='r') for key in self.input_keys}
        self.targets = np.load(os.path.join(data_path, f'{self.target_key}.npy'), mmap_mode='r')

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X = {key: self.inputs[key][batch_indices] for key in self.input_keys}
        y = self.targets[batch_indices]
        
        if self.use_sample_weights:
            # Create weights: higher affinity scores get exponentially more weight
            sample_weights = 1.0 + (y * self.weight_alpha)
            return X, y, sample_weights
        else:
            return X, y

# --- Custom Weighted Loss Function ---
def create_weighted_mse(pos_weight=5.0, threshold=0.1):
    def weighted_mse(y_true, y_pred):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        # Apply higher penalty for errors on high-affinity samples
        weights = tf.where(y_true >= threshold, pos_weight, 1.0)
        return mse * weights
    return weighted_mse

# --- "Supreme" Model Architecture ---
def build_supreme_model(input_shapes, params):
    """Builds the hybrid CNN-LSTM-Attention model."""
    
    # --- Define Input Layers ---
    input_layers = {key: Input(shape=shape, name=key) for key, shape in input_shapes.items()}

    # --- Reusable Sequence Processing Block (CNN -> Bi-LSTM) ---
    def create_seq_processor(input_tensor, conv_filters, kernel_size, lstm_units):
        x = Conv1D(filters=conv_filters, kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
        return x

    # --- Process Each Sequence Input ---
    primary_seq = create_seq_processor(
        input_layers['primary_sequence_input'], 
        params['cnn_filters'], params['cnn_kernel_size'], params['lstm_units']
    )
    target_seq = create_seq_processor(
        input_layers['target_sequence_input'], 
        params['cnn_filters'], params['cnn_kernel_size'], params['lstm_units']
    )
    competitor_seq_processed = create_seq_processor(
        input_layers['competitor_sequence_input'], 
        params['cnn_filters'] // 2, params['cnn_kernel_size'] - 2, params['lstm_units'] // 2
    )

    # --- Attention Mechanism (Transformer Block) ---
    # Primary sequence "queries" the target sequence to find important regions
    attention_output = MultiHeadAttention(
        num_heads=params['attention_heads'], key_dim=params['lstm_units']
    )(query=primary_seq, value=target_seq, key=target_seq)
    
    # Add & Norm (Standard Transformer practice)
    attention_output = LayerNormalization()(attention_output + primary_seq)
    
    # --- Pool the features from all branches ---
    primary_pooled = GlobalAveragePooling1D()(attention_output)
    target_pooled = GlobalAveragePooling1D()(target_seq)
    competitor_pooled = GlobalAveragePooling1D()(competitor_seq_processed)

    # --- Process Structure and Numerical Inputs ---
    struct_processed = GlobalAveragePooling1D()(Conv1D(32, 5, activation='relu')(input_layers['primary_structure_input']))
    numerical_processed = Dense(16, activation='relu')(input_layers['numerical_features_input'])

    # --- Combine All Processed Features ---
    combined = concatenate([
        primary_pooled, target_pooled, competitor_pooled, struct_processed, numerical_processed
    ])
    combined = Dropout(params['dropout_rate'])(combined)
    
    # --- Final Dense Layers for Prediction ---
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name='affinity_output')(x)

    model = Model(inputs=input_layers, outputs=output)
    return model

# --- Main Execution Block ---
if __name__ == "__main__":
    config = load_config()
    params = config['training_parameters']
    
    # --- Setup Paths from Config ---
    project_root = config['project_root']
    data_path = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    model_save_dir = os.path.join(project_root, config['output_folders']['main_models_folder'])
    logs_dir = os.path.join(project_root, config['output_folders']['logs_subfolder'])
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # --- Get Data Indices ---
    print("\nStep 1: Getting data indices...")
    try:
        train_indices = np.arange(len(np.load(os.path.join(data_path, 'y_train.npy'))))
        test_indices = np.arange(len(np.load(os.path.join(data_path, 'y_test.npy'))))
        np.random.shuffle(train_indices)
        print(f"  - Found {len(train_indices)} training and {len(test_indices)} test samples.")
    except FileNotFoundError as e:
        print(f"  - Error: {e}. Please run Stage 2 data preparation first.")
        exit()

    # --- Create Data Generators ---
    print("\nStep 2: Creating data generators...")
    input_keys = ['primary_sequence_input', 'target_sequence_input', 'competitor_sequence_input', 
                  'primary_structure_input', 'numerical_features_input']
    
    train_generator = DataGenerator(data_path, input_keys, 'y_train', params['batch_size'], train_indices, 'X_train_',
                                  use_sample_weights=params['use_sample_weighting'], 
                                  weight_alpha=params['sample_weight_alpha'])
    test_generator = DataGenerator(data_path, input_keys, 'y_test', params['batch_size'], test_indices, 'X_test_')

    # --- Build and Compile the Model ---
    print("\nStep 3: Building the 'Supreme' regression model...")
    sample_X, _ = train_generator[0]
    input_shapes = {key: val.shape[1:] for key, val in sample_X.items()}
    
    model = build_supreme_model(input_shapes, params)
    
    # --- Select Loss Function from Config ---
    if params['use_custom_loss']:
        loss_function = create_weighted_mse(params['custom_loss_pos_weight'])
        print("  - Using custom weighted MSE loss function.")
    else:
        loss_function = 'mean_squared_error'
        print("  - Using standard mean_squared_error loss function.")

    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), 
                  loss=loss_function, 
                  metrics=['mean_absolute_error'])
    model.summary()

    # --- Callbacks ---
    print("\nStep 4: Defining callbacks...")
    model_filepath = os.path.join(model_save_dir, 'best_supreme_model.keras')
    history_filepath = os.path.join(model_save_dir, 'history_supreme_model.json')
    log_dir = os.path.join(logs_dir, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    callbacks = [
        ModelCheckpoint(filepath=model_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        TensorBoard(log_dir=log_dir)
    ]
    print(f"  - Model checkpoints will be saved to: {model_filepath}")
    print(f"  - TensorBoard logs will be saved to: {log_dir}")

    # --- Train the Model ---
    print("\nStep 5: Starting model training...")
    history = model.fit(
        train_generator,
        epochs=params['epochs'],
        validation_data=test_generator,
        callbacks=callbacks,
        verbose=1
    )

    # --- Save Training History ---
    print(f"\nStep 6: Saving training history to {history_filepath}...")
    history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
    with open(history_filepath, 'w') as f:
        json.dump(history_dict, f)

    print("\n--- Supreme Model Training Complete ---")