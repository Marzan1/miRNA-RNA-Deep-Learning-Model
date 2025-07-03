# E:\my_deep_learning_project\codes\analysis_and_plotting.py

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import os
import json # For loading history

# --- Configuration Constants ---
DATA_PATH = r'E:\my_deep_learning_project\dataset\Processed_for_DL'
MODEL_SAVE_PATH = r'E:\my_deep_learning_project\models'
MODEL_NAME = "miRNA_RRE_REV_prediction_model.keras"
HISTORY_FILE_NAME = "training_history.json"

# --- Load Test Data ---
print("Loading test data for analysis...")
input_files = {
    'mirna_sequence_input': 'X_test_mirna_seq.npy',
    'rre_sequence_input': 'X_test_rre_seq.npy',
    'rev_sequence_input': 'X_test_rev_seq.npy',
    'mirna_structure_input': 'X_test_mirna_struct.npy',
    'numerical_features_input': 'X_test_numerical.npy'
}
target_file = 'y_test.npy'

X_test = {}
y_test = None
all_files_found = True
for key, filename in input_files.items():
    filepath = os.path.join(DATA_PATH, filename)
    if os.path.exists(filepath):
        X_test[key] = np.load(filepath)
    else:
        print(f"Error: Test file not found: {filepath}")
        all_files_found = False
        break

y_test_filepath = os.path.join(DATA_PATH, target_file)
if os.path.exists(y_test_filepath):
    y_test = np.load(y_test_filepath)
else:
    print(f"Error: Test target file not found: {y_test_filepath}")
    all_files_found = False

if not all_files_found:
    print("Aborting analysis due to missing test data files.")
    exit()

print("Test data loaded successfully.")

# --- Load the Trained Model ---
model_filepath = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
if not os.path.exists(model_filepath):
    print(f"Error: Model not found at {model_filepath}. Please ensure the model_building.py script ran successfully and saved the model.")
    exit()

print(f"Loading model from {model_filepath}...")
model = load_model(model_filepath)
print("Model loaded successfully.")

# --- Evaluate the Model (re-evaluate for consistency) ---
print("\nRe-evaluating the model on the test set...")
loss, accuracy, precision, recall, auc_score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test AUC: {auc_score:.4f}")

# Generate predictions for classification report and plots
y_pred_proba = model.predict(X_test, verbose=0) # Added verbose=0 for cleaner output
y_pred = (y_pred_proba > 0.5).astype(int)

# --- Class Distribution Check (important for report and plots) ---
unique_labels, counts = np.unique(y_test, return_counts=True)
print(f"\nClass distribution in y_test: {dict(zip(unique_labels, counts))}")

# --- Classification Report and Confusion Matrix ---
if len(unique_labels) < 2:
    print("\nWARNING: Only one class present in y_test. Classification report and AUC/PRC plots may be misleading or fail.")
    print("\nClassification Report (may be invalid due to single class in test set):")
    print(f"Class {unique_labels[0]} exists with {counts[0]} samples.")
    print("Cannot compute full classification report or ROC/PR curves as there's no other class to distinguish.")
    print(f"Metrics for class {unique_labels[0]}: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}")

    print("\nConfusion Matrix (may be invalid due to single class in test set):")
    # Try to generate confusion matrix with provided labels to suppress warning
    try:
        print(confusion_matrix(y_test, y_pred, labels=np.unique(np.concatenate((y_test, y_pred)))))
    except Exception as e:
        print(f"Could not generate confusion matrix due to: {e}")
        print(f"y_test unique: {np.unique(y_test)}, y_pred unique: {np.unique(y_pred)}")
    
else:
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --- Plotting Training History ---
history_file = os.path.join(MODEL_SAVE_PATH, HISTORY_FILE_NAME)
history_dict = None
if os.path.exists(history_file):
    with open(history_file, 'r') as f:
        history_dict = json.load(f)
    
    if history_dict:
        plt.figure(figsize=(12, 5))

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history_dict['accuracy'])
        plt.plot(history_dict['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid(True)

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history_dict['loss'])
        plt.plot(history_dict['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_history.png'))
        plt.close() # Close figure to avoid overlapping when showing next plot
        print(f"\nTraining history plot saved to {os.path.join(MODEL_SAVE_PATH, 'training_history.png')}")
    else:
        print("Warning: Training history loaded but is empty. Skipping history plots.")
else:
    print("Warning: Training history file not found. Skipping history plots.")

# --- ROC Curve and Precision-Recall Curve ---
if len(unique_labels) >= 2: # Only plot if both classes are present
    plt.figure(figsize=(12, 6))

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)

    plt.subplot(1, 2, 2)
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (area = {auprc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'model_performance_curves.png'))
    plt.show() # Show this plot after saving
    print(f"\nPerformance curves saved to {os.path.join(MODEL_SAVE_PATH, 'model_performance_curves.png')}")
else:
    print("\nSkipping ROC and Precision-Recall curve plots as only one class is present in the test set.")

print("\nAnalysis and plotting complete.")