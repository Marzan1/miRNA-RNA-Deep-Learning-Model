import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import json # To potentially load history if it exists, otherwise for dummy
import random # For generating dummy data

# --- Configuration Constants ---
PREPARED_DATASET_DIR = r"E:\my_deep_learning_project\dataset\Prepared Dataset"
MODEL_SAVE_DIR = r"E:\my_deep_learning_project\models"
HISTORY_SAVE_DIR = r"E:\my_deep_learning_project\training_history" # Directory where history.json might be saved

DATASET_FILE = "prepared_miRNA_RRE_dataset.csv"
MODEL_FILE = "miRNA_RRE_REV_prediction_model.keras"
HISTORY_FILE = "training_history.json" # Name of the history file

# Full paths
DATASET_PATH = os.path.join(PREPARED_DATASET_DIR, DATASET_FILE)
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILE)
HISTORY_PATH = os.path.join(HISTORY_SAVE_DIR, HISTORY_FILE)

# --- DEMO MODE FLAG ---
# Set to True to generate synthetic data for plots. These results are NOT scientifically valid.
# Set to False to use real data (plots may fail if only one class is present in real data).
DEMO_MODE = True 
# --------------------

print("Loading test data for analysis...")
try:
    df = pd.read_csv(DATASET_PATH)
    # Convert structure_vector from string representation of list to actual list/array
    df['structure_vector'] = df['structure_vector'].apply(eval)
    
    # Pad structure_vector to a consistent length (e.g., SEQUENCE_LENGTH_MAX used during prep)
    # Ensure this matches the expected input shape of your model's structure input
    MAX_STRUCTURE_LEN = 80 # Assuming SEQUENCE_LENGTH_MAX from data prep
    df['structure_vector'] = df['structure_vector'].apply(lambda x: x + [0] * (MAX_STRUCTURE_LEN - len(x)) if len(x) < MAX_STRUCTURE_LEN else x[:MAX_STRUCTURE_LEN])

    # Convert lists to numpy arrays for TensorFlow where needed
    # X_scalar and X_structure are already handled by .values and .tolist() .apply(np.array)
    
    # --- IMPORTANT: Extract all 5 model inputs ---
    X_scalar = df[['gc_content', 'dg', 'conservation', 'affinity']].values
    X_structure = np.array(df['structure_vector'].tolist())
    X_rre_sequence = df['rre_sequence'].values.astype(str) # Ensure sequences are string arrays
    X_rev_sequence = df['rev_sequence'].values.astype(str) # Ensure sequences are string arrays
    X_region = df['region'].values.astype(str) # Ensure regions are string arrays
    
    y = df['label'].values

    # Using a fixed random_state for reproducibility in splitting
    # This split should ideally match the one used during model training if possible.
    
    # Combined split for ALL 5 inputs and the label
    # This assumes `stratify` will work. If it fails due to single class, we'll re-split.
    try:
        _, X_test_scalar, \
        _, X_test_structure, \
        _, X_test_rre_sequence, \
        _, X_test_rev_sequence, \
        _, X_test_region, \
        _, y_test = train_test_split(
            X_scalar, X_structure, X_rre_sequence, X_rev_sequence, X_region, y,
            test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Warning: Stratification failed during train_test_split ({e}). Re-splitting without stratify.")
        # If stratification fails, try again without it (e.g., if only one class exists)
        _, X_test_scalar, \
        _, X_test_structure, \
        _, X_test_rre_sequence, \
        _, X_test_rev_sequence, \
        _, X_test_region, \
        _, y_test = train_test_split(
            X_scalar, X_structure, X_rre_sequence, X_rev_sequence, X_region, y,
            test_size=0.2, random_state=42
        )

    print("Test data loaded successfully.")
except Exception as e:
    print(f"Error loading test data or splitting: {e}")
    exit()

print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print("\nRe-evaluating the model on the test set...")
# --- Pass all 5 inputs to model.predict() ---
y_pred_proba_real = model.predict([
    X_test_scalar, 
    X_test_structure, 
    X_test_rre_sequence, 
    X_test_rev_sequence, 
    X_test_region
]).ravel()
# -------------------------------------------
y_pred_real = (y_pred_proba_real > 0.5).astype(int)

# --- DEMO MODE LOGIC ---
if DEMO_MODE:
    print("\n" + "="*80)
    print("!!! DEMO MODE ACTIVE !!!")
    print("!!! Generating synthetic data for plots. These results are NOT scientifically valid. !!!")
    print("!!! Remember to set DEMO_MODE = False and fix your dataset for real analysis. !!!")
    print("="*80 + "\n")

    # 1. Simulate y_test with both classes
    num_samples = len(y_test)
    num_zeros = num_samples // 2
    num_ones = num_samples - num_zeros
    y_test_demo = np.concatenate((np.zeros(num_zeros), np.ones(num_ones))).astype(int)
    np.random.shuffle(y_test_demo) # Shuffle to mix 0s and 1s

    # 2. Simulate y_pred_proba (probabilities for both classes)
    y_pred_proba_demo = np.zeros_like(y_test_demo, dtype=float)
    for i, label in enumerate(y_test_demo):
        if label == 0:
            # For true 0s, predict low probability for class 1
            y_pred_proba_demo[i] = random.uniform(0.05, 0.45) 
        else:
            # For true 1s, predict high probability for class 1
            y_pred_proba_demo[i] = random.uniform(0.55, 0.95)

    y_pred_demo = (y_pred_proba_demo > 0.5).astype(int)

    # For plotting functions and classification report, use the demo data
    y_test_for_plots = y_test_demo
    y_pred_proba_for_plots = y_pred_proba_demo
    y_pred_for_plots = y_pred_demo

    # 3. Create dummy history data
    dummy_history = {
        'loss': [0.6, 0.45, 0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05],
        'accuracy': [0.55, 0.7, 0.8, 0.88, 0.92, 0.95, 0.96, 0.97, 0.98, 0.99],
        'val_loss': [0.65, 0.5, 0.35, 0.25, 0.2, 0.15, 0.12, 0.1, 0.09, 0.08],
        'val_accuracy': [0.5, 0.65, 0.75, 0.82, 0.88, 0.91, 0.93, 0.94, 0.95, 0.96]
    }
    history_data = dummy_history # Use dummy history for plots

else: # REAL DATA MODE
    y_test_for_plots = y_test
    y_pred_proba_for_plots = y_pred_proba_real
    y_pred_for_plots = y_pred_real

    # Try loading real history
    history_data = None
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, 'r') as f:
                history_data = json.load(f)
            print(f"Training history loaded from {HISTORY_PATH}.")
        except Exception as e:
            print(f"Warning: Could not load training history from {HISTORY_PATH}: {e}. Skipping history plots.")
    else:
        print(f"Warning: Training history file not found at {HISTORY_PATH}. Skipping history plots.")

# --- End DEMO MODE LOGIC ---

# Evaluate model metrics using the REAL model's predictions on the REAL test set.
# This section remains true to your actual model's performance, even if single-class.
test_eval_results = model.evaluate([X_test_scalar, X_test_structure, X_test_rre_sequence, X_test_rev_sequence, X_test_region], y_test, verbose=0)
test_loss = test_eval_results[0] # First value is loss
# Keras metrics (like accuracy, precision, recall, AUC) can be computed if they were part of model.compile()
# Or re-calculate manually using y_test and y_pred_real / y_pred_proba_real
test_accuracy = (y_pred_real == y_test).mean() 

try:
    # Use sklearn metrics for precision, recall, AUC on real data
    # These might still be misleading or 0 if y_test is single-class (as per previous warnings)
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    test_precision = precision_score(y_test, y_pred_real, zero_division=0)
    test_recall = recall_score(y_test, y_pred_real, zero_division=0)
    # AUC requires both classes in y_test
    if len(np.unique(y_test)) > 1:
        test_auc = roc_auc_score(y_test, y_pred_proba_real)
    else:
        test_auc = 0.0 # Cannot compute AUC with single class
except Exception as e:
    test_precision = 0.0 
    test_recall = 0.0
    test_auc = 0.0
    print(f"Warning: Could not compute some sklearn metrics on real data: {e}")


print(f"Test Loss (Real Data): {test_loss:.4f}")
print(f"Test Accuracy (Real Data): {test_accuracy:.4f}")
print(f"Test Precision (Real Data): {test_precision:.4f}")
print(f"Test Recall (Real Data): {test_recall:.4f}")
print(f"Test AUC (Real Data): {test_auc:.4f}")

print(f"\nClass distribution in y_test (Real Data): {pd.Series(y_test).value_counts().to_dict()}")

# --- Classification Report (Uses potentially faked data for plotting if DEMO_MODE is True) ---
# This section is for generating plots that *require* multiple classes.
if len(np.unique(y_test_for_plots)) < 2:
    print("\nWARNING: Only one class present in y_test_for_plots (after demo mode adjustment or in real data).")
    print("Classification report and ROC/PRC plots require at least two classes to be meaningful.")
    print(f"Class 1 exists with {np.sum(y_test_for_plots == 1)} samples (in plotting data).")
    print("Skipping full classification report, ROC, and PR curves.")
    # Show basic metrics that are not class-dependent, or manually computed
    print("Basic metrics from real data evaluation:")
    print(f"Accuracy: {test_accuracy:.2f}, Precision: {test_precision:.2f}, Recall: {test_recall:.2f}")

else:
    print("\nClassification Report (Generated from plotting data, potentially synthetic):")
    print(classification_report(y_test_for_plots, y_pred_for_plots, target_names=['Class 0', 'Class 1'], zero_division=0))

    # --- Confusion Matrix ---
    print("\nConfusion Matrix (Generated from plotting data, potentially synthetic):")
    try:
        cm = confusion_matrix(y_test_for_plots, y_pred_for_plots)
        print(cm)
    except Exception as e:
        print(f"Could not generate confusion matrix due to: {e}")
    print(f"y_test_for_plots unique: {np.unique(y_test_for_plots).tolist()}, y_pred_for_plots unique: {np.unique(y_pred_for_plots).tolist()}")

    # --- ROC Curve ---
    fpr, tpr, thresholds = roc_curve(y_test_for_plots, y_pred_proba_for_plots)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (Potentially Synthetic)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_test_for_plots, y_pred_proba_for_plots)
    avg_precision = average_precision_score(y_test_for_plots, y_pred_proba_for_plots)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Potentially Synthetic)')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Training History Plots ---
if history_data:
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history_data['accuracy'])
    plt.plot(history_data['val_accuracy'])
    plt.title('Model Accuracy (Potentially Synthetic History)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history_data['loss'])
    plt.plot(history_data['val_loss'])
    plt.title('Model Loss (Potentially Synthetic History)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Warning: Training history data not available or not loaded. Skipping history plots.")

print("\nAnalysis and plotting complete.")
