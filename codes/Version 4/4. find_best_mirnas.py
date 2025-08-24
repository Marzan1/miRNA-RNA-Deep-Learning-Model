# 4. find_best_mirnas.py (Final Corrected Version)
import os
import re
import json
import subprocess
import numpy as np
import pandas as pd
import joblib
from Bio import SeqIO
from tensorflow.keras.models import load_model
import argparse
import warnings

# Suppress messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore', category=UserWarning)

# --- Configuration ---
MODEL_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\models"
DATA_PREP_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset\processed_for_dl"
PREDICTION_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\prediction"
os.makedirs(PREDICTION_DIR, exist_ok=True)

MODEL_FILE = 'best_regression_model.keras'
SCALER_FILE = 'minmax_scaler.pkl'

# --- Constants ---
PREDICTION_BATCH_SIZE = 1000
NUCLEOTIDE_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}

# --- Preprocessing Functions ---
def calculate_gc_content(sequence):
    if not sequence: return 0.0
    return (sequence.upper().count('G') + sequence.upper().count('C')) / len(sequence)

def predict_structure(sequence, max_len):
    try:
        result = subprocess.run(['RNAfold'], input=sequence[:max_len], capture_output=True, text=True, check=True, encoding='utf-8')
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            struct, dg_str = lines[1].split(' ', 1)
            match = re.search(r"[-+]?\d+\.\d+", dg_str)
            return struct, float(match.group(0)) if match else 0.0
    except Exception:
        return "." * len(sequence), 0.0
    return "." * len(sequence), 0.0

def one_hot_encode_sequence(sequence, max_len):
    encoded = np.zeros((max_len, len(NUCLEOTIDE_MAP)), dtype=np.float32)
    if not isinstance(sequence, str): sequence = ""
    for i, char in enumerate(sequence[:max_len]):
        encoded[i, NUCLEOTIDE_MAP.get(char.upper(), 4)] = 1
    return encoded

def prepare_input_for_prediction(mirna_seq, rre_seq, rev_seq, scaler, max_lens):
    max_mirna_len, max_rre_len, max_rev_len = max_lens
    gc = calculate_gc_content(mirna_seq)
    struct_str, dg = predict_structure(mirna_seq, max_mirna_len)
    mirna_encoded = one_hot_encode_sequence(mirna_seq, max_mirna_len)
    rre_encoded = one_hot_encode_sequence(rre_seq, max_rre_len)
    rev_encoded = one_hot_encode_sequence(rev_seq, max_rev_len)
    struct_encoded = np.zeros((max_mirna_len, 1), dtype=np.float32)
    struct_vec = [({'.': 0, '(': 1, ')': -1}).get(c, 0) for c in struct_str]
    struct_encoded[:len(struct_vec), 0] = struct_vec[:max_mirna_len]
    base_numerical_features = [gc, dg]
    num_scaler_features = scaler.n_features_in_
    final_numerical_features = base_numerical_features + [0.0] * (num_scaler_features - len(base_numerical_features))
    scaled_numerical = scaler.transform([final_numerical_features])
    return {
        'mirna_sequence_input': np.array([mirna_encoded]), 'rre_sequence_input': np.array([rre_encoded]),
        'rev_sequence_input': np.array([rev_encoded]), 'mirna_structure_input': np.array([struct_encoded]),
        'numerical_features_input': scaled_numerical
    }

def load_all_fastas_in_folder(folder_path):
    sequences = []
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.fasta', '.fa', '.txt'))]
    if not file_paths:
        print(f"Warning: No FASTA files found in folder: {folder_path}")
        return sequences
    for filepath in file_paths:
        try:
            sequences.extend(list(SeqIO.parse(filepath, "fasta")))
        except Exception as e:
            print(f"    - Error reading file {filepath}: {e}")
    return sequences

# --- Main Execution ---
def main(args):
    print("--- miRNA Ranking Prediction Tool (True Streaming Version) ---")
    
    # You may need to install/update this library: pip install pyarrow
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        model = load_model(os.path.join(MODEL_DIR, MODEL_FILE))
        scaler = joblib.load(os.path.join(DATA_PREP_DIR, SCALER_FILE))
        print("Model and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading files: {e}.")
        return

    # --- FIX: Initialize variables before the try block to ensure they exist ---
    MAX_MIRNA_LEN, MAX_RRE_LEN, MAX_REV_LEN = 80, 150, 200
    MIN_MIRNA_LEN_TRAINED = 20
    
    try:
        print("\nAuto-detecting model's trained sequence lengths...")
        # --- FIX: Access shape information directly from the model's input dictionary ---
        MAX_MIRNA_LEN = model.input['mirna_sequence_input'].shape[1]
        MAX_RRE_LEN = model.input['rre_sequence_input'].shape[1]
        MAX_REV_LEN = model.input['rev_sequence_input'].shape[1]
        
        print(f"  - Model expects miRNA length: {MIN_MIRNA_LEN_TRAINED}-{MAX_MIRNA_LEN}")
    except Exception as e:
        print(f"  - Warning: Could not auto-detect sequence lengths. Using defaults (80, 150, 200). Error: {e}")
    max_lens = (MAX_MIRNA_LEN, MAX_RRE_LEN, MAX_REV_LEN)

    # Load all necessary sequences
    target_rres = load_all_fastas_in_folder(args.rre_folder)
    target_revs = load_all_fastas_in_folder(args.rev_folder)
    candidate_mirnas = load_all_fastas_in_folder(args.mirna_folder)

    if not all([target_rres, target_revs, candidate_mirnas]):
        print("\nAborting: Missing input sequences.")
        return

    # --- CHANGE: Set up for streaming directly to a Parquet file ---
    output_filename = os.path.join(PREDICTION_DIR, 'ranked_mirna_predictions.parquet')
    if os.path.exists(output_filename):
        os.remove(output_filename) # Ensure a fresh start
    
    parquet_writer = None
    batch_results = []
    total_processed = 0

    rre_record, rev_record = target_rres[0], target_revs[0]
    rre_seq, rev_seq = str(rre_record.seq), str(rev_record.seq)
    print(f"\nUsing Target RRE: {rre_record.id} | Competitor: {rev_record.id}")
    print(f"Ranking {len(candidate_mirnas)} candidate miRNAs...")
    
    for i, mirna_record in enumerate(candidate_mirnas):
        mirna_seq = str(mirna_record.seq)
        
        warning_message = "OK"
        if not (MIN_MIRNA_LEN_TRAINED <= len(mirna_seq) <= MAX_MIRNA_LEN):
            warning_message = f"Length ({len(mirna_seq)}) outside trained range ({MIN_MIRNA_LEN_TRAINED}-{MAX_MIRNA_LEN}). Extrapolation."

        baseline_input = prepare_input_for_prediction(mirna_seq, rre_seq, "", scaler, max_lens)
        pred_baseline = model.predict(baseline_input, verbose=0)[0][0]
        
        competition_input = prepare_input_for_prediction(mirna_seq, rre_seq, rev_seq, scaler, max_lens)
        pred_competition = model.predict(competition_input, verbose=0)[0][0]
        
        batch_results.append({
            'mirna_id': mirna_record.id,
            'predicted_affinity_baseline': pred_baseline,
            'predicted_affinity_with_rev': pred_competition,
            'competitive_effect': pred_baseline - pred_competition,
            'Warning': warning_message
        })
        
        # --- CHANGE: Write to disk when the batch is full or at the end of the loop ---
        if len(batch_results) >= PREDICTION_BATCH_SIZE or (i + 1) == len(candidate_mirnas):
            if not batch_results: continue

            batch_df = pd.DataFrame(batch_results)
            table = pa.Table.from_pandas(batch_df)
            
            if parquet_writer is None:
                # For the first batch, create the writer with the correct data schema
                parquet_writer = pq.ParquetWriter(output_filename, table.schema)
            
            parquet_writer.write_table(table)
            
            total_processed += len(batch_results)
            print(f"  - Processed and saved {total_processed}/{len(candidate_mirnas)} miRNAs...")
            batch_results = [] # Reset the batch, freeing memory

    # --- CHANGE: Close the writer to finalize the file ---
    if parquet_writer:
        parquet_writer.close()

    print("\n--- Prediction Complete ---")
    print(f"Full, unsorted results have been saved to '{output_filename}'.")
    
    # Optional: Load and display the top results from the saved file for a quick preview
    print("\n--- Preview of Top 10 Predicted Interacting miRNAs ---")
    try:
        final_df = pd.read_parquet(output_filename)
        # Sort the DataFrame in memory for display purposes (only after all data is saved)
        final_df.sort_values(by='predicted_affinity_with_rev', ascending=False, inplace=True)
        print(final_df.head(10).to_string(float_format='%.4f'))
    except Exception as e:
        print(f"Could not read final Parquet file for preview. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict miRNA-RRE binding affinity.")
    parser.add_argument('--mirna_folder', type=str, default=os.path.join(PREDICTION_DIR, "mirnas_to_rank"))
    parser.add_argument('--rre_folder', type=str, default=os.path.join(PREDICTION_DIR, "targets_to_predict"))
    parser.add_argument('--rev_folder', type=str, default=os.path.join(PREDICTION_DIR, "competitor_to_compare"))
    args = parser.parse_args()
    main(args)