# s4_predict.py (Fully Config-Driven, Supreme Model Compatible)
import os
import re
import json
import numpy as np
import pandas as pd
import joblib
from Bio import SeqIO
import tensorflow as tf
import warnings
import pyarrow as pa
import pyarrow.parquet as pq

# <<< CHANGE: Import our definitive processors and custom objects >>>
from molecule_processors import process_rna_universal
from s3_build_model import create_weighted_mse

# Suppress messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore', category=UserWarning)

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

def one_hot_encode_sequence(sequence, max_len):
    """Simple one-hot encoder for sequences."""
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    encoded_seq = np.zeros((max_len, len(nucleotide_map)), dtype=np.float32)
    for i, char in enumerate(sequence[:max_len]):
        encoded_seq[i, nucleotide_map.get(char.upper(), 4)] = 1
    return encoded_seq

def prepare_input_for_prediction(primary_data, target_data, competitor_data, scaler, pad_lengths):
    """
    Prepares a single data point for prediction using the Supreme model's input format.
    """
    max_primary_len, max_target_len, max_competitor_len = pad_lengths
    
    # Extract numerical features and apply scaling
    # Use dummy values if features are missing
    gc = primary_data.get('gc_content', 0.5)
    dg = primary_data.get('dg', 0.0)
    conservation = primary_data.get('conservation', 0.0) # Assuming no conservation data for new miRNAs
    
    num_features = [gc, dg, conservation]
    # Ensure the feature vector matches what the scaler expects
    if len(num_features) < scaler.n_features_in_:
        num_features += [0.0] * (scaler.n_features_in_ - len(num_features))
        
    scaled_numerical = scaler.transform([num_features])
    
    # One-hot encode sequences
    primary_seq_encoded = one_hot_encode_sequence(primary_data['sequence'], max_primary_len)
    target_seq_encoded = one_hot_encode_sequence(target_data.get('sequence', ''), max_target_len)
    competitor_seq_encoded = one_hot_encode_sequence(competitor_data.get('sequence', ''), max_competitor_len)
    
    # Prepare structure and graph data (use zeros if not available)
    structure_vector = json.loads(primary_data.get('structure_vector', '[]'))
    structure_padded = np.zeros((max_primary_len, 1), dtype=np.float32)
    structure_padded[:len(structure_vector), 0] = structure_vector
    
    inputs = {
        'primary_sequence_input': np.array([primary_seq_encoded]),
        'target_sequence_input': np.array([target_seq_encoded]),
        'competitor_sequence_input': np.array([competitor_seq_encoded]),
        'primary_structure_input': np.array([structure_padded]),
        'numerical_features_input': scaled_numerical
    }
    
    return inputs

def load_fasta_from_folder(folder_path):
    """Loads all sequences from all FASTA files in a directory."""
    records = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return records
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.fasta', '.fa', '.fna', '.txt'))]
    for filepath in file_paths:
        records.extend(list(SeqIO.parse(filepath, "fasta")))
    return records

# --- Main Execution ---
def main():
    print("--- Universal Molecule Ranking Tool ---")
    
    config = load_config()
    pred_params = config['prediction_parameters']
    train_params = config['training_parameters']
    
    # --- 1. Setup paths from config ---
    project_root = config['project_root']
    model_dir = os.path.join(project_root, config['output_folders']['main_models_folder'])
    scaler_dir = os.path.join(project_root, config['data_folders']['main_dataset_folder'], config['data_folders']['processed_for_dl_subfolder'])
    prediction_dir = os.path.join(project_root, config['output_folders']['prediction_subfolder'])
    os.makedirs(prediction_dir, exist_ok=True)
    
    # --- 2. Load model and scaler ---
    model_path = os.path.join(model_dir, pred_params['model_to_use'])
    scaler_path = os.path.join(scaler_dir, 'minmax_scaler.pkl')

    custom_objects = {}
    if train_params['advanced_training']['use_custom_loss']:
        loss_instance = create_weighted_mse(train_params['advanced_training']['custom_loss_pos_weight'])
        custom_objects = {'weighted_mse': loss_instance}
        print("  - Loading model with custom weighted MSE loss.")

    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        scaler = joblib.load(scaler_path)
        print(f"  - Successfully loaded model '{pred_params['model_to_use']}' and scaler.")
    except Exception as e:
        print(f"  - FATAL ERROR loading files: {e}")
        return

    # --- 3. Load sequences to predict ---
    input_folders = pred_params['input_folders']
    primary_folder = os.path.join(prediction_dir, input_folders['primary'])
    target_folder = os.path.join(prediction_dir, input_folders['target'])
    competitor_folder = os.path.join(prediction_dir, input_folders['competitor'])
    
    primary_records = load_fasta_from_folder(primary_folder)
    target_records = load_fasta_from_folder(target_folder)
    competitor_records = load_fasta_from_folder(competitor_folder)

    if not primary_records or not target_records:
        print("\nAborting: Missing primary sequences to rank or target sequences to predict against.")
        return

    # --- 4. Prepare for prediction ---
    target_record = target_records[0] # Use the first target found
    competitor_record = competitor_records[0] if competitor_records else None
    
    target_seq = str(target_record.seq).replace('T', 'U')
    if pred_params.get('use_prediction_region_slice', False):
        start, end = pred_params['prediction_target_region_slice']
        target_seq = target_seq[start:end]
        print(f"  - Slicing target sequence to region [{start}:{end}].")

    competitor_seq = str(competitor_record.seq).replace('T', 'U') if competitor_record else ""

    print(f"\nUsing Target: {target_record.id}")
    if competitor_record:
        print(f"Using Competitor: {competitor_record.id}")
    else:
        print("No competitor molecule provided.")
        
    print(f"Ranking {len(primary_records)} candidate molecules...")
    
    # --- 5. Run predictions in a memory-safe stream ---
    output_path = os.path.join(prediction_dir, pred_params['output_filename'])
    if os.path.exists(output_path): os.remove(output_path)
    
    results = []
    pad_lengths = (
        train_params['sequence_padding']['max_primary_len'],
        train_params['sequence_padding']['max_target_len'],
        train_params['sequence_padding']['max_competitor_len']
    )

    for i, primary_record in enumerate(primary_records):
        # We reuse the universal processor from Stage 1 to ensure consistency
        # This will generate GC, 1D structure, and even attempt to get a graph
        # This function is robust to failure and provides necessary features
        primary_processed = process_rna_universal(((primary_record.id, str(primary_record.seq)), config['training_parameters'], 'primary'))
        
        # We only need sequence for target and competitor here
        target_processed = {'sequence': target_seq}
        competitor_processed = {'sequence': competitor_seq}
        
        # Scenario 1: Predict affinity with competitor present
        inputs_with_competitor = prepare_input_for_prediction(primary_processed, target_processed, competitor_processed, scaler, pad_lengths)
        pred_with_competitor = model.predict(inputs_with_competitor, verbose=0)[0][0]

        # Scenario 2: Predict affinity without competitor (baseline)
        inputs_no_competitor = prepare_input_for_prediction(primary_processed, target_processed, {'sequence': ''}, scaler, pad_lengths)
        pred_no_competitor = model.predict(inputs_no_competitor, verbose=0)[0][0]

        results.append({
            'primary_molecule_id': primary_record.id,
            'predicted_affinity_baseline': float(pred_no_competitor),
            'predicted_affinity_with_competitor': float(pred_with_competitor),
            'competitive_effect (higher_is_better)': float(pred_no_competitor - pred_with_competitor),
        })
        print(f"  - Processed {i+1}/{len(primary_records)}...", end='\r')

    print("\n\n--- Prediction Complete ---")
    
    # Save results to a Parquet file for efficiency
    if results:
        results_df = pd.DataFrame(results)
        results_df.sort_values(by='competitive_effect (higher_is_better)', ascending=False, inplace=True)
        
        table = pa.Table.from_pandas(results_df)
        pq.write_table(table, output_path)
        print(f"Ranked results saved to '{output_path}'")
        
        print("\n--- Top 10 Candidates (Ranked by Competitive Effect) ---")
        print(results_df.head(10).to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    main()