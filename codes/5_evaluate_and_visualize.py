# 5. evaluate_and_visualize.py (Research-Grade Regression Analysis)
import os
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Define paths to the data, saved model, and a new directory for all outputs.
DATA_PATH = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset\processed_for_dl"
MODEL_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\models"
MODEL_NAME = "best_regression_model.keras"
HISTORY_NAME = "training_history_regression.json"

# Create a unique, timestamped folder for this evaluation run's outputs
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
PLOTS_DIR = os.path.join(MODEL_DIR, f"evaluation_plots_{TIMESTAMP}")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Main Analysis Function ---
def analyze_model_performance():
    # --- 1. Load Data, Model, and History ---
    print("--- Starting Model Evaluation and Visualization ---")
    print("\nStep 1: Loading test data, model, and history...")
    X_test = {}
    try:
        model_input_keys = ['mirna_sequence_input', 'rre_sequence_input', 'rev_sequence_input', 
                            'mirna_structure_input', 'numerical_features_input']
        for key in model_input_keys:
            X_test[key] = np.load(os.path.join(DATA_PATH, f'X_test_{key}.npy'), mmap_mode='r')
        y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'), mmap_mode='r')
        
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_NAME))
        
        with open(os.path.join(MODEL_DIR, HISTORY_NAME), 'r') as f:
            history = json.load(f)
            
        print("  - All files loaded successfully.")
    except FileNotFoundError as e:
        print(f"  - Error loading files: {e}. Ensure model training is complete.")
        return

    # --- 2. Make Predictions and Calculate Metrics ---
    print("\nStep 2: Evaluating model performance on the test set...")
    # Using a larger batch size for prediction is more efficient
    y_pred = model.predict(X_test, batch_size=1024, verbose=1).ravel()

    # Calculate key regression metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    p_corr, _ = pearsonr(y_test, y_pred)

    metrics_summary = {
        "Mean Squared Error (MSE)": [mse],
        "Mean Absolute Error (MAE)": [mae],
        "R-squared (R2 Score)": [r2],
        "Pearson Correlation (r)": [p_corr]
    }
    metrics_df = pd.DataFrame(metrics_summary)
    
    print("\n--- Performance Metrics Summary ---")
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    
    # Save metrics table to a file
    metrics_path = os.path.join(PLOTS_DIR, 'performance_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, float_format='%.4f')
    print(f"\n  - Metrics table saved to: {metrics_path}")

    # --- 3. Generate Publication-Quality Plots ---
    print("\nStep 3: Generating and saving publication-quality plots...")
    # Set a professional plotting style and color palette
    plt.style.use('seaborn-v0_8-whitegrid')
    palette = sns.color_palette("viridis", as_cmap=True)

    # Plot 1: Training History
    plt.figure(figsize=(12, 7))
    plt.plot(history['loss'], label='Training Loss', color='royalblue', lw=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='darkorange', linestyle='--', lw=2)
    plt.title('Model Loss (Mean Squared Error) Over Epochs', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.legend(fontsize=12)
    plt.yscale('log') # Use a log scale for better visibility
    plt.tight_layout()
    history_plot_path = os.path.join(PLOTS_DIR, 'training_history.png')
    plt.savefig(history_plot_path, dpi=300)
    plt.close()
    print(f"  - Saved training history plot.")

    # Plot 2: Prediction Correlation Scatter Plot
    plt.figure(figsize=(10, 10))
    # Use a random sample for plotting to avoid a dense, unreadable blob
    sample_indices = np.random.choice(len(y_test), size=min(5000, len(y_test)), replace=False)
    sns.scatterplot(x=y_test[sample_indices], y=y_pred[sample_indices], alpha=0.6, s=50, color=palette(0.3))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Perfect Prediction')
    plt.title(f'Predicted vs. Actual Affinity\n$R^2$ Score: {r2:.3f} | Pearson r: {p_corr:.3f}', fontsize=16, pad=20)
    plt.xlabel('Actual Affinity Score', fontsize=12)
    plt.ylabel('Predicted Affinity Score', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    correlation_plot_path = os.path.join(PLOTS_DIR, 'prediction_correlation.png')
    plt.savefig(correlation_plot_path, dpi=300)
    plt.close()
    print(f"  - Saved prediction correlation plot.")

    # Plot 3: Residuals Plot (Error vs. Prediction)
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x=y_pred[sample_indices], y=residuals[sample_indices], alpha=0.5, s=50, color=palette(0.6))
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.title('Residuals (Actual - Predicted) vs. Predicted Value', fontsize=16, pad=20)
    plt.xlabel('Predicted Affinity Score', fontsize=12)
    plt.ylabel('Residual (Error)', fontsize=12)
    plt.tight_layout()
    residuals_plot_path = os.path.join(PLOTS_DIR, 'residuals_plot.png')
    plt.savefig(residuals_plot_path, dpi=300)
    plt.close()
    print(f"  - Saved residuals diagnostic plot.")

    # Plot 4: Error Distribution
    plt.figure(figsize=(12, 7))
    sns.histplot(residuals, kde=True, bins=50, color=palette(0.9))
    plt.title('Distribution of Prediction Errors (Residuals)', fontsize=16, pad=20)
    plt.xlabel('Error (Actual - Predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    error_dist_path = os.path.join(PLOTS_DIR, 'error_distribution.png')
    plt.savefig(error_dist_path, dpi=300)
    plt.close()
    print(f"  - Saved error distribution plot.")

    # --- 4. Generate Insight Tables ---
    print("\nStep 4: Generating tables with example predictions...")
    results_df = pd.DataFrame({'ActualAffinity': y_test, 'PredictedAffinity': y_pred, 'Error': residuals})
    
    # Get the 10 best predictions (smallest error)
    best_predictions = results_df.iloc[results_df['Error'].abs().nsmallest(10).index]
    
    # Get the 10 worst predictions (largest error)
    worst_predictions = results_df.iloc[results_df['Error'].abs().nlargest(10).index]

    print("\n--- 10 Best Predictions (Smallest Error) ---")
    print(best_predictions.to_string(float_format='%.4f'))
    
    print("\n--- 10 Worst Predictions (Largest Error) ---")
    print(worst_predictions.to_string(float_format='%.4f'))

    # Save these tables to files
    best_pred_path = os.path.join(PLOTS_DIR, 'best_10_predictions.csv')
    worst_pred_path = os.path.join(PLOTS_DIR, 'worst_10_predictions.csv')
    best_predictions.to_csv(best_pred_path, index=False, float_format='%.4f')
    worst_predictions.to_csv(worst_pred_path, index=False, float_format='%.4f')
    print(f"\n  - Insight tables saved.")

    print(f"\n--- Evaluation and Visualization Complete ---")
    print(f"All outputs have been saved to the folder: '{PLOTS_DIR}'")

if __name__ == "__main__":
    analyze_model_performance()