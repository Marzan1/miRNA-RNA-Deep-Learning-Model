import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import os

# --- Configuration Constants ---
# Path where your processed .npy files are saved
DATA_PATH = r'E:\my_deep_learning_project\dataset\Processed_for_DL'
# Folder to save your trained model
MODEL_SAVE_PATH = r'E:\my_deep_learning_project\models' 

# Ensure the model save directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# --- Load Processed Data ---
print("Loading processed data...")

# Define filenames based on your data_preparation script's saving convention
input_files = {
    'mirna_sequence_input': 'X_train_mirna_seq.npy',
    'rre_sequence_input': 'X_train_rre_seq.npy',
    'rev_sequence_input': 'X_train_rev_seq.npy',
    'mirna_structure_input': 'X_train_mirna_struct.npy', # Note: 'mirna_struct'
    'numerical_features_input': 'X_train_numerical.npy' # Note: 'numerical'
}
target_file = 'y_train.npy'

# Load training data
X_train = {}
y_train = None
all_files_found_train = True
for key, filename in input_files.items():
    filepath = os.path.join(DATA_PATH, filename)
    if os.path.exists(filepath):
        X_train[key] = np.load(filepath)
    else:
        print(f"Error: Training file not found: {filepath}")
        all_files_found_train = False
        break

y_train_filepath = os.path.join(DATA_PATH, target_file)
if os.path.exists(y_train_filepath):
    y_train = np.load(y_train_filepath)
else:
    print(f"Error: Training target file not found: {y_train_filepath}")
    all_files_found_train = False

if not all_files_found_train:
    print("Aborting model building due to missing training data files.")
    exit()

# Load test data (adjusting filenames for test set)
X_test = {}
y_test = None
all_files_found_test = True
for key, filename in input_files.items():
    test_filename = filename.replace('X_train_', 'X_test_') # Convert train filename to test filename
    filepath = os.path.join(DATA_PATH, test_filename)
    if os.path.exists(filepath):
        X_test[key] = np.load(filepath)
    else:
        print(f"Error: Test file not found: {filepath}")
        all_files_found_test = False
        break

y_test_filepath = os.path.join(DATA_PATH, target_file.replace('y_train', 'y_test'))
if os.path.exists(y_test_filepath):
    y_test = np.load(y_test_filepath)
else:
    print(f"Error: Test target file not found: {y_test_filepath}")
    all_files_found_test = False

if not all_files_found_test:
    print("Aborting model building due to missing test data files.")
    exit()

print("Data loaded successfully.")
print("\n--- Training Data Shapes ---")
for key, arr in X_train.items():
    print(f"{key}: {arr.shape}")
print(f"y_train: {y_train.shape}")

print("\n--- Test Data Shapes ---")
for key, arr in X_test.items():
    print(f"{key}: {arr.shape}")
print(f"y_test: {y_test.shape}")

# --- Model Architecture ---
print("\nBuilding the deep learning model...")

# miRNA Sequence Branch (Input: (80, 5))
input_mirna_seq = Input(shape=X_train['mirna_sequence_input'].shape[1:], name='mirna_sequence_input')
x_mirna = Conv1D(filters=64, kernel_size=5, activation='relu')(input_mirna_seq)
x_mirna = GlobalMaxPooling1D()(x_mirna) # Reduces each feature map to a single value

# RRE Sequence Branch (Input: (150, 5))
input_rre_seq = Input(shape=X_train['rre_sequence_input'].shape[1:], name='rre_sequence_input')
x_rre = Conv1D(filters=64, kernel_size=5, activation='relu')(input_rre_seq)
x_rre = GlobalMaxPooling1D()(x_rre)

# REV Sequence Branch (Input: (200, 5)) - NEW
input_rev_seq = Input(shape=X_train['rev_sequence_input'].shape[1:], name='rev_sequence_input')
x_rev = Conv1D(filters=64, kernel_size=5, activation='relu')(input_rev_seq)
x_rev = GlobalMaxPooling1D()(x_rev)

# miRNA Structure Branch (Input: (80, 1))
input_mirna_structure = Input(shape=X_train['mirna_structure_input'].shape[1:], name='mirna_structure_input')
x_structure = Conv1D(filters=32, kernel_size=3, activation='relu')(input_mirna_structure)
x_structure = GlobalMaxPooling1D()(x_structure)

# Numerical Features Branch (Input: (4,))
input_numerical_features = Input(shape=X_train['numerical_features_input'].shape[1:], name='numerical_features_input')
x_numerical = Dense(32, activation='relu')(input_numerical_features)

# Concatenate all branches
combined = concatenate([x_mirna, x_rre, x_rev, x_structure, x_numerical])

# Fully connected layers
x = Dense(128, activation='relu')(combined)
x = Dropout(0.3)(x) # Adding dropout for regularization
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

# Output layer for binary classification (assuming 'label' is 0 or 1)
output = Dense(1, activation='sigmoid', name='output_label')(x) # Sigmoid for binary classification

# Create the model
model = Model(inputs=[input_mirna_seq, input_rre_seq, input_rev_seq, input_mirna_structure, input_numerical_features],
              outputs=output)

# --- Compile the Model ---
print("Compiling the model...")
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy', # Appropriate for binary classification
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='auc')])

model.summary()

# --- Train the Model ---
print("\nTraining the model...")
history = model.fit(X_train, y_train, # Pass the dictionary of inputs
                    epochs=20, # You can adjust the number of epochs
                    batch_size=32,
                    validation_data=(X_test, y_test), # Pass the dictionary of inputs
                    verbose=1)

print("\nModel training complete.")

# --- Evaluate the Model ---
print("\nEvaluating the model on the test set...")
loss, accuracy, precision, recall, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test AUC: {auc:.4f}")

# Generate classification report
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to binary predictions (0 or 1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save the Trained Model ---
model_name = "miRNA_RRE_REV_prediction_model.keras" # New name for the updated model
model_path = os.path.join(MODEL_SAVE_PATH, model_name)

model.save(model_path)
print(f"\nModel saved to: {model_path}")