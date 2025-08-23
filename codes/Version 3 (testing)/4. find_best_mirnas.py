# 4. find_best_mirnas.py (Research-Grade Prediction Tool)
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import re
import subprocess
import joblib
from Bio import SeqIO

# --- Configuration ---
MODEL_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\models"
DATA_PREP_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset\processed_for_dl"
PREDICTION_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\prediction"
os.makedirs(PREDICTION_DIR, exist_ok=True)

MODEL_FILE = 'best_model.keras'
SCALER_FILE = 'minmax_scaler.pkl'

# --- INPUT FILES --- (Place your files in these locations)
TARGET_RRE_FILE = os.path.join(PREDICTION_DIR, "rre_target.fasta")
TARGET_REV_FILE = os.path.join(PREDICTION_DIR, "rev_competitor.fasta")
CANDIDATE_MIRNA_FILE = os.path.join(PREDICTION_DIR, "candidate_mirnas.fasta")

# Constants must match training script
MAX_MIRNA_LEN, MAX_RRE_LEN, MAX_REV_LEN = 80, 150, 200
NUCLEOTIDE_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
N_NUCLEOTIDES = len(NUCLEOTIDE_MAP)

# --- Preprocessing Functions (Identical to training pipeline) ---
def calculate_gc_content(sequence):
    if not sequence: return 0.0
    return (sequence.upper().count('G') + sequence.upper().count('C')) / len(sequence)

def predict_structure(sequence):
    try:
        result = subprocess.run(['RNAfold'], input=sequence, capture_output=True, text=True, check=True, encoding='utf-8')
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            struct, dg_str = lines[1].split(' ', 1)
            match = re.search(r"[-+]?\d+\.\d+", dg_str)
            return struct, float(match.group(0)) if match else 0.0
    except Exception:
        return "." * len(sequence), 0.0
    return "." * len(sequence), 0.0

def encode_structure_dot_bracket(structure):
    mapping = {'.': 0, '(': 1, ')': -1}
    return [mapping.get(c, 0) for c in structure]

def one_hot_encode_sequence(sequence, max_len):
    encoded = np.zeros((max_len, N_NUCLEOTIDES), dtype=np.float32)
    if not isinstance(sequence, str): sequence = ""
    for i, char in enumerate(sequence[:max_len]):
        encoded[i, NUCLEOTIDE_MAP.get(char.upper(), 4)] = 1
    return encoded

def prepare_input_for_prediction(mirna_seq, rre_seq, rev_seq, scaler):
    """Prepares a single sample's data into the model's expected format."""
    gc = calculate_gc_content(mirna_seq)
    struct_str, dg = predict_structure(mirna_seq)
    
    mirna_encoded = one_hot_encode_sequence(mirna_seq, MAX_MIRNA_LEN)
    rre_encoded = one_hot_encode_sequence(rre_seq, MAX_RRE_LEN)
    rev_encoded = one_hot_encode_sequence(rev_seq, MAX_REV_LEN)

    struct_encoded = np.zeros((MAX_MIRNA_LEN, 1), dtype=np.float32)
    struct_vec = encode_structure_dot_bracket(struct_str)
    struct_encoded[:len(struct_vec), 0] = struct_vec
    
    # We use a placeholder for conservation (0.0) as it's unknown for new miRNAs
    numerical_features = np.array([[gc, dg, 0.0]])
    scaled_numerical = scaler.transform(numerical_features)
    
    return {
        'mirna_sequence_input': np.array([mirna_encoded]), 'rre_sequence_input': np.array([rre_encoded]),
        'rev_sequence_input': np.array([rev_encoded]), 'mirna_structure_input': np.array([struct_encoded]),
        'numerical_features_input': scaled_numerical
    }

def load_first_fasta_from_file(filepath):
    """Loads the first sequence from a FASTA file."""
    try:
        with open(filepath, "r") as handle:
            return next(SeqIO.parse(handle, "fasta"))
    except Exception as e:
        print(f"Error: Could not load sequence from {filepath}. {e}")
        return None

# --- Main Execution ---
def main():
    print("--- miRNA Ranking Prediction Tool ---")
    
    try:
        model = load_model(os.path.join(MODEL_DIR, MODEL_FILE))
        scaler = joblib.load(os.path.join(DATA_PREP_DIR, SCALER_FILE))
        print("Model and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading files: {e}. Ensure model training is complete.")
        return

    target_rre = load_first_fasta_from_file(TARGET_RRE_FILE)
    target_rev = load_first_fasta_from_file(TARGET_REV_FILE)
    if not target_rre or not target_rev: return

    rre_seq, rev_seq = str(target_rre.seq), str(target_rev.seq)
    print(f"Target RRE: {target_rre.id}, Target REV: {target_rev.id}")

    try:
        candidate_mirnas = list(SeqIO.parse(CANDIDATE_MIRNA_FILE, "fasta"))
        print(f"Loaded {len(candidate_mirnas)} candidate miRNAs from {CANDIDATE_MIRNA_FILE}.")
    except Exception as e:
        print(f"Error loading candidate miRNAs: {e}")
        return

    results = []
    print("\nRanking miRNAs...")
    for i, mirna_record in enumerate(candidate_mirnas):
        mirna_seq = str(mirna_record.seq)
        
        # Prediction 1: Baseline (No competitor)
        baseline_input = prepare_input_for_prediction(mirna_seq, rre_seq, "", scaler)
        prob_baseline = model.predict(baseline_input, verbose=0)[0][0]
        
        # Prediction 2: With Competitor
        competition_input = prepare_input_for_prediction(mirna_seq, rre_seq, rev_seq, scaler)
        prob_competition = model.predict(competition_input, verbose=0)[0][0]
        
        results.append({
            'mirna_id': mirna_record.id,
            'prob_baseline': prob_baseline,
            'prob_with_competitor': prob_competition,
            'competitive_effect': prob_baseline - prob_competition
        })
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(candidate_mirnas)} miRNAs...")

    results_df = pd.DataFrame(results).sort_values(by='prob_with_competitor', ascending=False).reset_index(drop=True)
    
    output_filename = os.path.join(PREDICTION_DIR, 'ranked_mirna_predictions.csv')
    results_df.to_csv(output_filename, index=False)

    print("\n--- Top 10 Predicted Interacting miRNAs (in presence of competitor) ---")
    print(results_df.head(10))
    print(f"\nFull ranked results saved to '{output_filename}'.")

if __name__ == "__main__":
    main()