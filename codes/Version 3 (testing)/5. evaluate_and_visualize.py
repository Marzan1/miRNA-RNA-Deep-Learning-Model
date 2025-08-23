# 5. evaluate_and_visualize.py (Research-Grade Analysis)
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# --- Configuration ---
DATA_PATH = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset\processed_for_dl"
MODEL_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\models"
MODEL_NAME = "best_model.keras"
HISTORY_NAME = "training_history.json"
PLOTS_DIR = os.path.join(MODEL_DIR, "evaluation_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 1. Load Data, Model, and History ---
print("Loading test data, model, and history...")
try:
    X_test = {}
    for key in ['mirna_sequence_input', 'rre_sequence_input', 'rev_sequence_input', 'mirna_structure_input', 'numerical_features_input']:
        X_test[key] = np.load(os.path.join(DATA_PATH, f'X_test_{key}.npy'))
    y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_NAME))
    with open(os.path.join(MODEL_DIR, HISTORY_NAME), 'r') as f:
        history = json.load(f)
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Ensure model training is complete.")
    exit()
print("All files loaded successfully.")

# --- 2. Model Evaluation ---
print("\nEvaluating model performance on the test set...")
y_pred_proba = model.predict(X_test, batch_size=512, verbose=0).ravel()
y_pred_class = (y_pred_proba > 0.5).astype(int)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_class, target_names=['Class 0 (Low Affinity)', 'Class 1 (High Affinity)']))

# --- 3. Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Training History
print("Generating training history plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Model Training History', fontsize=16)
ax1.plot(history['accuracy'], label='Train Accuracy', color='royalblue')
ax1.plot(history['val_accuracy'], label='Validation Accuracy', color='darkorange', linestyle='--')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend()
ax2.plot(history['loss'], label='Train Loss', color='royalblue')
ax2.plot(history['val_loss'], label='Validation Loss', color='darkorange', linestyle='--')
ax2.set_title('Model Loss (Binary Cross-Entropy)')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend()
history_plot_path = os.path.join(PLOTS_DIR, 'training_history.png')
plt.savefig(history_plot_path); plt.close()
print(f"Saved history plot to {history_plot_path}")

# Plot 2: Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix', fontsize=14)
plt.ylabel('Actual Class'); plt.xlabel('Predicted Class')
cm_plot_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_plot_path); plt.close()
print(f"Saved confusion matrix plot to {cm_plot_path}")

# Plot 3: ROC and Precision-Recall Curves
if len(np.unique(y_test)) > 1:
    print("Generating ROC and Precision-Recall curves...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Model Performance Curves', fontsize=16)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)'); ax1.legend(loc="lower right")
    ax2.plot(recall, precision, color='mediumblue', lw=2, label='Precision-Recall curve')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve'); ax2.legend(loc="lower left")
    curves_plot_path = os.path.join(PLOTS_DIR, 'performance_curves.png')
    plt.savefig(curves_plot_path); plt.close()
    print(f"Saved performance curves plot to {curves_plot_path}")

print(f"\nEvaluation and visualization complete. All plots saved in: '{PLOTS_DIR}'")