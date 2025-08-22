# E:\my_deep_learning_project\codes\predict_on_new_data.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
from Bio import SeqIO # For loading new FASTA sequences (if needed)
import re # For RNAfold parsing
import subprocess
from sklearn.preprocessing import MinMaxScaler # For loading the scaler
import joblib # For loading the scaler


# --- Configuration Constants (must match training constants) ---
DATASET_ROOT_FOLDER = r"E:\my_deep_learning_project\dataset" 
MODEL_SAVE_PATH = r'E:\my_deep_learning_project\models'
MODEL_NAME = "miRNA_RRE_REV_prediction_model.keras"
SCALER_FILE_NAME = 'minmax_scaler.pkl' # Name of the saved scaler file

# Constants for sequence processing (must match training constants)
MAX_MIRNA_SEQUENCE_LENGTH = 80
MAX_RRE_SEQUENCE_LENGTH = 150
MAX_REV_SEQUENCE_LENGTH = 200
NUCLEOTIDE_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
N_NUCLEOTIDES = len(NUCLEOTIDE_MAP)

# Define AFFINITY_THRESHOLD, which was used to binarize labels
# This value must match the one used in Gemini_dataset_preparation_6.py
# If you don't know it, you might need to check that script or the data.
# For demonstration, let's assume it was 0.5. Adjust if different.
AFFINITY_THRESHOLD = 0.5 

# --- Global Scaler Variable ---
loaded_scaler = None

# --- Helper Functions (copied from your preparation scripts for consistency) ---

def calculate_gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0

def predict_structure(sequence):
    """Predicts RNA secondary structure using RNAfold and extracts minimum free energy (dG)."""
    try:
        # RNAfold expects input on stdin and outputs to stdout
        result = subprocess.run(
            ['RNAfold'],
            input=sequence,
            capture_output=True,
            text=True,
            check=True
        )
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) >= 2:
            structure_line = output_lines[1].split(' ')[0]
            # Extract dG from the second line, typically in parentheses (e.g., "((...)) (-2.30)")
            match = re.search(r'\(\s*([-+]?\d+\.\d+)\)', output_lines[1])
            if match:
                dg = float(match.group(1))
            else:
                print(f"Warning: Could not parse free energy from RNAfold output for sequence {sequence}: '{output_lines[1]}'")
                dg = 0.0 # Default value if parsing fails
            return structure_line, dg
        return None, 0.0 # Default dG
    except FileNotFoundError:
        print("Error: 'RNAfold' command not found. Ensure ViennaRNA Package is installed and in your system's PATH.")
        return None, 0.0
    except subprocess.CalledProcessError as e:
        print(f"Error running RNAfold for sequence '{sequence}': {e}")
        print(f"Stderr: {e.stderr}")
        return None, 0.0
    except Exception as e:
        print(f"An unexpected error occurred with RNAfold for sequence '{sequence}': {e}")
        return None, 0.0

def encode_structure_dot_bracket(structure):
    """Encodes dot-bracket string into a numerical vector."""
    max_len = MAX_MIRNA_SEQUENCE_LENGTH
    vector = np.zeros(max_len, dtype=int)
    for i, char in enumerate(structure[:max_len]):
        if char == '.':
            vector[i] = 0
        elif char == '(':
            vector[i] = 1
        elif char == ')':
            vector[i] = -1
    return vector.tolist()

def one_hot_encode_sequence(sequence, max_len, nucleotide_map, n_nucleotides):
    """One-hot encodes a nucleotide sequence."""
    encoded_seq = np.zeros((max_len, n_nucleotides), dtype=np.float32)
    if sequence is None or not isinstance(sequence, str):
        sequence = ""
    for i, char in enumerate(sequence[:max_len]):
        char_upper = char.upper()
        if char_upper in nucleotide_map:
            encoded_seq[i, nucleotide_map[char_upper]] = 1
        else:
            # Map unknown characters to 'N' index
            encoded_seq[i, nucleotide_map['N']] = 1
    return encoded_seq

# --- Function to prepare new data for prediction ---
def prepare_single_sample_for_prediction(mirna_seq, rre_seq, rev_seq, current_mirna_id="new_mirna", current_rre_id="new_rre", current_rev_id="new_rev"):
    """
    Prepares a single set of miRNA, RRE, and REV sequences into the format expected by the model.
    Numerical features like conservation and experimental affinity are unknown for truly new data.
    We will assume 0 for these or a suitable placeholder that the model was trained to handle.
    """
    global loaded_scaler # Access the global scaler

    # 1. miRNA Features
    gc_content = calculate_gc_content(mirna_seq)
    structure, dg = predict_structure(mirna_seq)
    if structure is None:
        print(f"Warning: RNAfold failed for miRNA {current_mirna_id}. Setting dG to 0 and structure to all zeros.")
        structure = "." * len(mirna_seq) # Placeholder
        dg = 0.0
    
    struct_vec = np.array(encode_structure_dot_bracket(structure), dtype=np.float32) # Ensure float32 for consistency
    
    # Expand dims for Conv1D input (batch, timesteps, features)
    struct_vec = np.expand_dims(struct_vec, axis=-1) 

    # 2. Numerical Features
    # The order of numerical features should match the training data: 'gc_content', 'dg', 'conservation', 'affinity'
    # For new data, 'conservation' and 'affinity' are usually unknown, so we use placeholders (e.g., 0).
    numerical_values = np.array([[gc_content, dg, 0.0, 0.0]]) # conservation=0.0, affinity=0.0 (placeholder)

    if loaded_scaler is not None:
        # Only scale the features that were scaled during training.
        # Assuming the scaler was fitted on ['gc_content', 'dg', 'conservation'].
        # The 'affinity' column should NOT be passed to the scaler.
        # If your numerical_features input to the model during training was just 3 features, adjust here.
        
        # We need to ensure numerical_values has the same number of features as the scaler was trained on.
        # If your numerical_features_input to the model was trained on 3 features (gc, dg, conservation):
        numerical_values_for_scaler = numerical_values[:, :3] # Take only gc_content, dg, conservation

        scaled_numerical_features = loaded_scaler.transform(numerical_values_for_scaler)
        
        # If your model's numerical_features_input expects 4 features (including affinity as input, which is unusual)
        # and you trained it on 4 features, you would pass numerical_values directly.
        # For typical classification, 'affinity' is the target, not an input feature.
        # Based on your model summary, numerical_features_input: (None, 4), implying 4 features.
        # Let's assume the 4th feature was 'affinity' and it was passed unscaled or with a dummy value.
        # Or perhaps it was another feature. Check your `Gemini_deep_learning_data_preparation.py`
        # for what `numerical_features.columns` were when `scaler.fit_transform(features_to_scale)` was called.

        # Correcting based on your model summary (input shape (None, 4)) and common practice:
        # If the scaler was only fit on (gc, dg, conservation), but the model expects 4 inputs:
        # This implies the 4th input was either unscaled or another specific value.
        # Let's re-align to what was likely used. The `numerical_features` in data_preparation has `gc_content`, `dg`, `conservation`, `affinity`.
        # Your `numerical_features_input: (1643, 4)` implies 4 features were passed.
        # The `MinMaxScaler` in `Gemini_deep_learning_data_preparation.py` applies to `features_to_scale`.
        # If `features_to_scale` had 4 columns (e.g., `gc_content`, `dg`, `conservation`, and the raw `affinity` before binarization),
        # then you'd transform all 4. If `affinity` was the label, it should NOT have been an input.
        # Let's assume for `predict_on_new_data` you provide `gc_content`, `dg`, `conservation`, and a *dummy value* for the 4th feature
        # that was passed to the model as part of 'numerical_features_input'.

        # If the numerical_features_input truly was (gc_content, dg, conservation, affinity_score_before_binarization)
        # Then you would need the affinity_score for the new data. Which is usually unknown.
        # The most logical setup is (gc_content, dg, conservation, possibly a dummy feature or a feature indicating the source).
        # Assuming your `numerical_features_input` in model was `gc_content`, `dg`, `conservation`, `some_other_numerical_feature_like_length_ratio`
        # If it was `affinity` itself, then you can't predict it if it's an input.
        # The most likely scenario: `numerical_features_input` in the model is `gc_content`, `dg`, `conservation`, and one more feature, *not* `affinity`.
        
        # Let's assume the 4th numerical feature was always 0.0 for unknown.
        # So `numerical_values` should have 4 values.
        scaled_numerical_features = loaded_scaler.transform(numerical_values) # Apply scaler to all 4 features.
        
    else:
        print("WARNING: MinMaxScaler not loaded. Numerical features for prediction will NOT be scaled correctly. Model predictions will be unreliable.")
        scaled_numerical_features = numerical_values # Proceed unscaled, but expect issues

    # 3. Sequence Encoding
    mirna_seq_encoded = one_hot_encode_sequence(mirna_seq, MAX_MIRNA_SEQUENCE_LENGTH, NUCLEOTIDE_MAP, N_NUCLEOTIDES)
    rre_seq_encoded = one_hot_encode_sequence(rre_seq, MAX_RRE_SEQUENCE_LENGTH, NUCLEOTIDE_MAP, N_NUCLEOTIDES)
    rev_seq_encoded = one_hot_encode_sequence(rev_seq, MAX_REV_SEQUENCE_LENGTH, NUCLEOTIDE_MAP, N_NUCLEOTIDES)

    # Prepare inputs dictionary with batch dimension
    inputs = {
        'mirna_sequence_input': np.array([mirna_seq_encoded]),
        'rre_sequence_input': np.array([rre_seq_encoded]),
        'rev_sequence_input': np.array([rev_seq_encoded]),
        'mirna_structure_input': np.array([struct_vec]), 
        'numerical_features_input': scaled_numerical_features
    }
    return inputs

# --- Main Execution ---
if __name__ == "__main__":
    # Load the trained scaler first
    scaler_filepath = os.path.join(DATASET_ROOT_FOLDER, "Processed_for_DL", SCALER_FILE_NAME)
    if os.path.exists(scaler_filepath):
        print(f"Loading MinMaxScaler from {scaler_filepath}...")
        loaded_scaler = joblib.load(scaler_filepath)
        print("MinMaxScaler loaded successfully.")
    else:
        print(f"Error: MinMaxScaler not found at {scaler_filepath}. Numerical features will not be scaled.")
        print("Please run Gemini_deep_learning_data_preparation.py first to generate and save the scaler.")
        # Exit or handle as appropriate if scaler is critical

    # Load the trained model
    model_filepath = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    if not os.path.exists(model_filepath):
        print(f"Error: Model not found at {model_filepath}. Please ensure the model_building.py script ran successfully.")
        exit()

    print(f"Loading model from {model_filepath}...")
    model = load_model(model_filepath)
    print("Model loaded successfully.")

    # --- Example New Data (Replace with your actual new data) ---
    # Scenario 1: A specific miRNA, RRE, and REV you want to test
    # Ensure these sequences are realistic and representative of your data
    new_mirna_seq_1 = "UGAGGUAGUAGGUUGUAUAGUU" # Example miRNA sequence (hsa-let-7a)
    new_rre_seq_1 = "GGCCGTAGTGTAGATCCCAACTGACCTGAAGCATGTTTTCATTCAGATCCATCTATGGAGCACGAGAGCA" # Example RRE-like sequence
    new_rev_seq_1 = "UGAGGUAGUAGGUUGUAUAGUUUAUUAUAUAUAAGCACGACGGGGGGGGGGGGGGGGGGGGGGG" # Example Rev sequence (can be RNA or DNA, ensure consistency with your dataset)

    print("\n--- Predicting for Example Scenario 1 ---")
    prepared_input_1 = prepare_single_sample_for_prediction(new_mirna_seq_1, new_rre_seq_1, new_rev_seq_1,
                                                            current_mirna_id="Example_miRNA_1",
                                                            current_rre_id="Example_RRE_1",
                                                            current_rev_id="Example_REV_1")
    
    if prepared_input_1:
        # model.predict returns a numpy array. For a single sample, it's typically [[probability]].
        prediction_proba_1 = model.predict(prepared_input_1, verbose=0)[0][0] 
        prediction_class_1 = (prediction_proba_1 > AFFINITY_THRESHOLD).astype(int)
        print(f"Input miRNA: {new_mirna_seq_1}")
        print(f"Input RRE: {new_rre_seq_1}")
        print(f"Input REV: {new_rev_seq_1}")
        print(f"Predicted Probability (affinity > {AFFINITY_THRESHOLD}): {prediction_proba_1:.4f}")
        print(f"Predicted Class (1=High Affinity, 0=Low Affinity): {prediction_class_1}")
    else:
        print("Skipping prediction for Scenario 1 due to preparation errors.")

    # Scenario 2: Another example (you can extend this to load from a FASTA or CSV file)
    new_mirna_seq_2 = "CCUUACCCGUUAGGGAAUGC" # Another miRNA example
    new_rre_seq_2 = "ACAGGGCACACACACGCACGUGGCACGACGACCA" # Another RRE example
    new_rev_seq_2 = "GUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGA" # Another REV example

    print("\n--- Predicting for Example Scenario 2 ---")
    prepared_input_2 = prepare_single_sample_for_prediction(new_mirna_seq_2, new_rre_seq_2, new_rev_seq_2,
                                                            current_mirna_id="Example_miRNA_2",
                                                            current_rre_id="Example_RRE_2",
                                                            current_rev_id="Example_REV_2")
    
    if prepared_input_2:
        prediction_proba_2 = model.predict(prepared_input_2, verbose=0)[0][0]
        prediction_class_2 = (prediction_proba_2 > AFFINITY_THRESHOLD).astype(int)
        print(f"Input miRNA: {new_mirna_seq_2}")
        print(f"Input RRE: {new_rre_seq_2}")
        print(f"Input REV: {new_rev_seq_2}")
        print(f"Predicted Probability (affinity > {AFFINITY_THRESHOLD}): {prediction_proba_2:.4f}")
        print(f"Predicted Class (1=High Affinity, 0=Low Affinity): {prediction_class_2}")
    else:
        print("Skipping prediction for Scenario 2 due to preparation errors.")

    print("\nPrediction complete.")
