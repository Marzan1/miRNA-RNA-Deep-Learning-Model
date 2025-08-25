# s5_evaluate.py (Fully Config-Driven, Supreme Model Compatible)
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

# <<< CHANGE: Import our custom loss function to load the model correctly >>>
from s3_build_model import create_weighted_mse

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

# <<< CHANGE: New function to dynamically load all available test data >>>
def load_test_data(data_path):
    """Dynamically finds and loads all X_test_*.npy files."""
    X_test = {}
    print(f"  - Searching for test data in: {data_path}")
    for f in os.listdir(data_path):
        if f.startswith('X_test_') and f.endswith('.npy'):
            key = f.replace('X_test_', '').replace('.npy', '')
            print(f"    - Loading: {f}")
            X_test[key] = np.load(os.path.join(data_path, f), mmap_mode='r')
    
    y_test = np.load(os.path.join(data_path, 'y_test.npy'), mmap_mode='r')
    return X_test, y_test

# --- Main Analysis Function ---
def analyze_model_performance():
    print("--- Starting Model Evaluation and Visualization ---")
    
    # --- 1. Load Config, Data, Model, and History ---
    config = load_config()
    eval_params = config['evaluation_parameters']
    train_params = config['training_parameters']
    
    project_root = config['project_root']
    data_path = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    model_dir = os.path.join(project_root, config['output_folders']['main_models_folder'])
    
    # Create a unique, timestamped folder for this evaluation run's outputs
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plots_dir = os.path.join(model_dir, f"{eval_params['output_folder_prefix']}_{timestamp}")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\nStep 1: Loading test data, model, and history...")
    try:
        X_test, y_test = load_test_data(data_path)
        
        # Handle custom loss function for model loading
        custom_objects = {}
        if train_params['advanced_training']['use_custom_loss']:
            loss_instance = create_weighted_mse(train_params['advanced_training']['custom_loss_pos_weight'])
            custom_objects = {'weighted_mse': loss_instance}
            print("  - Loading model with custom weighted MSE loss.")
            
        model = tf.keras.models.load_model(os.path.join(model_dir, eval_params['model_to_evaluate']), custom_objects=custom_objects)
        
        with open(os.path.join(model_dir, eval_params['history_to_load']), 'r') as f:
            history = json.load(f)
            
        print("  - All files loaded successfully.")
    except (FileNotFoundError, IOError) as e:
        print(f"  - FATAL ERROR loading files: {e}. Ensure model training is complete and filenames in config.json are correct.")
        return

    # --- 2. Make Predictions and Calculate Metrics ---
    print("\nStep 2: Evaluating model performance on the test set...")
    y_pred = model.predict(X_test, batch_size=eval_params.get('prediction_batch_size', 1024), verbose=1).ravel()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    p_corr, _ = pearsonr(y_test, y_pred)

    metrics_df = pd.DataFrame({
        "Metric": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R-squared (R2 Score)", "Pearson Correlation (r)"],
        "Value": [mse, mae, r2, p_corr]
    })
    
    print("\n--- Performance Metrics Summary ---")
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    
    metrics_df.to_csv(os.path.join(plots_dir, 'performance_metrics.csv'), index=False, float_format='%.4f')
    print(f"\n  - Metrics table saved to: '{plots_dir}'")

    # --- 3. Generate Publication-Quality Plots ---
    print("\nStep 3: Generating and saving publication-quality plots...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Training History
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.yscale('log')
    plt.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=300)
    plt.close()

    # Plot 2: Prediction Correlation Scatter Plot
    plt.figure(figsize=(8, 8))
    sample_indices = np.random.choice(len(y_test), size=min(5000, len(y_test)), replace=False)
    sns.regplot(x=y_test[sample_indices], y=y_pred[sample_indices], scatter_kws={'alpha':0.3})
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Prediction')
    plt.title(f'Predicted vs. Actual Affinity\n$R^2$ Score: {r2:.3f} | Pearson r: {p_corr:.3f}')
    plt.xlabel('Actual Affinity Score')
    plt.ylabel('Predicted Affinity Score')
    plt.xlim(0, max(1.0, np.max(y_test), np.max(y_pred)) * 1.05)
    plt.ylim(0, max(1.0, np.max(y_test), np.max(y_pred)) * 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'prediction_correlation.png'), dpi=300)
    plt.close()
    
    print(f"  - All plots have been saved to the folder: '{plots_dir}'")

if __name__ == "__main__":
    analyze_model_performance()