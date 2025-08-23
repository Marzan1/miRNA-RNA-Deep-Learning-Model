# 3. model_building.py (Research-Grade Version)
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
import os
import json

# --- Configuration Constants ---
DATA_PATH = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset\processed_for_dl"
MODEL_SAVE_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 1. Load Processed Data ---
print("Loading processed data...")
X_train, X_test = {}, {}
try:
    for key in ['mirna_sequence_input', 'rre_sequence_input', 'rev_sequence_input', 'mirna_structure_input', 'numerical_features_input']:
        X_train[key] = np.load(os.path.join(DATA_PATH, f'X_train_{key}.npy'))
        X_test[key] = np.load(os.path.join(DATA_PATH, f'X_test_{key}.npy'))
    y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
    y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Please run the previous scripts.")
    exit()
print("Data loaded successfully.")

# --- 2. Calculate Class Weights for Imbalanced Data ---
print("\nCalculating class weights to handle data imbalance...")
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
print(f"  - Weights computed for classes: {class_weights}")

# --- 3. Model Architecture ---
print("\nBuilding the model...")
# Input layers
input_mirna_seq = Input(shape=X_train['mirna_sequence_input'].shape[1:], name='mirna_sequence_input')
input_rre_seq = Input(shape=X_train['rre_sequence_input'].shape[1:], name='rre_sequence_input')
input_rev_seq = Input(shape=X_train['rev_sequence_input'].shape[1:], name='rev_sequence_input')
input_mirna_structure = Input(shape=X_train['mirna_structure_input'].shape[1:], name='mirna_structure_input')
input_numerical = Input(shape=X_train['numerical_features_input'].shape[1:], name='numerical_features_input')

# Convolutional branches for sequences
x_mirna = Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(input_mirna_seq)
x_mirna = BatchNormalization()(x_mirna)
x_mirna = GlobalMaxPooling1D()(x_mirna)

x_rre = Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(input_rre_seq)
x_rre = BatchNormalization()(x_rre)
x_rre = GlobalMaxPooling1D()(x_rre)

x_rev = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(input_rev_seq)
x_rev = BatchNormalization()(x_rev)
x_rev = GlobalMaxPooling1D()(x_rev)

x_structure = Conv1D(filters=32, kernel_size=5, activation='relu')(input_mirna_structure)
x_structure = GlobalMaxPooling1D()(x_structure)

x_numerical = Dense(16, activation='relu')(input_numerical)

# Concatenate all branches
combined = concatenate([x_mirna, x_rre, x_rev, x_structure, x_numerical])
combined = Dropout(0.5)(combined)

# Fully connected layers
x = Dense(128, activation='relu')(combined)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid', name='output_label')(x)

model = Model(inputs=list(X_train.keys()), outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
model.summary()

# --- 4. Callbacks for Smarter Training ---
callbacks = [
    ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_DIR, 'best_model.keras'), save_best_only=True, monitor='val_auc', mode='max', verbose=1),
    EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
]

# --- 5. Train the Model ---
print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=256,
    validation_data=(X_test, y_test),
    class_weight=class_weights, # Use class weights here
    callbacks=callbacks,
    verbose=1
)

# --- 6. Save Final Model and History ---
history_path = os.path.join(MODEL_SAVE_DIR, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history.history, f)
print(f"\nTraining complete. Best model saved, and history logged to {history_path}")