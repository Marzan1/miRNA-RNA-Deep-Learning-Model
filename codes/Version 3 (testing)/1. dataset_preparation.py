# 1. dataset_preparation.py (Extraordinary Version with All Enhancements)
import os
import pandas as pd
from Bio import SeqIO
import subprocess
import numpy as np
import re
import json
from itertools import product
from multiprocessing import Pool, cpu_count
import time

# --- Configuration Constants ---
DATASET_ROOT_FOLDER = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset"
MIRNA_FOLDER_NAME = "miRNA_dataset"
AFFINITY_FOLDER_NAME = "affinity_score"
CONSERVATION_FOLDER_NAME = "conservation_score"
RRE_FOLDER_NAME = "target" 
REV_FOLDER_NAME = "competitor"
PREPARED_DATASET_FOLDER_NAME = "prepared_dataset"

def get_data_path(base_path, folder_name):
    select_path = os.path.join(base_path, folder_name, 'select')
    return select_path if os.path.exists(select_path) else os.path.join(base_path, folder_name)

MIRNA_DATA_DIR = get_data_path(DATASET_ROOT_FOLDER, MIRNA_FOLDER_NAME)
AFFINITY_DATA_DIR = get_data_path(DATASET_ROOT_FOLDER, AFFINITY_FOLDER_NAME)
CONSERVATION_DATA_DIR = get_data_path(DATASET_ROOT_FOLDER, CONSERVATION_FOLDER_NAME)
RRE_DATA_DIR = get_data_path(DATASET_ROOT_FOLDER, RRE_FOLDER_NAME)
REV_DATA_DIR = get_data_path(DATASET_ROOT_FOLDER, REV_FOLDER_NAME)
PREPARED_DATASET_DIR = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME)

os.makedirs(PREPARED_DATASET_DIR, exist_ok=True)

# --- NEW: HYBRID CONFIGURATION ---
# Set to True to only use the Stem IIB region of RREs for training.
# Set to False to use the full RRE sequences.
FOCUS_ON_STEM_IIB_ONLY = True 

# Filtering Criteria
SEQUENCE_LENGTH_MIN = 20
SEQUENCE_LENGTH_MAX = 70
GC_CONTENT_MIN = 0.30
GC_CONTENT_MAX = 0.80
AFFINITY_THRESHOLD = 0.4
CSV_BATCH_SIZE = 10000 # Write to CSV every 10,000 rows to save memory

# --- Helper Functions ---
def calculate_gc_content(sequence):
    if not sequence: return 0.0
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    return gc_count / len(sequence)

def predict_structure(sequence):
    """Robustly runs RNAfold and extracts dot-bracket + dG."""
    try:
        result = subprocess.run(
            ['RNAfold'], input=sequence, capture_output=True,
            text=True, check=True, encoding='utf-8', timeout=30 # Add timeout
        )
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) >= 2:
            struct_line = output_lines[1]
            structure = struct_line.split(' ')[0]
            # ENHANCEMENT: More robust regex to find any floating point number
            match = re.search(r"[-+]?\d+\.\d+", struct_line)
            dg = float(match.group(0)) if match else 0.0
            return structure, dg
        return None, None
    except Exception:
        return None, None

def encode_structure_dot_bracket(structure):
    mapping = {'.': 0, '(': 1, ')': -1}
    return [mapping.get(char, 0) for char in structure]

def _get_files_in_folder(folder_path, extensions):
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}. Skipping.")
        return []
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in extensions)]

def process_mirna_entry(mirna_tuple):
    """Function to process a single miRNA, designed for parallel execution."""
    mirna_id, mirna_seq = mirna_tuple
    
    if not (SEQUENCE_LENGTH_MIN <= len(mirna_seq) <= SEQUENCE_LENGTH_MAX):
        return (mirna_id, "reject_length")
    
    gc = calculate_gc_content(mirna_seq)
    if not (GC_CONTENT_MIN <= gc <= GC_CONTENT_MAX):
        return (mirna_id, "reject_gc")
    
    structure, dg = predict_structure(mirna_seq)
    if structure is None:
        return (mirna_id, "reject_structure")
    
    return {
        'mirna_id': mirna_id, 'sequence': mirna_seq, 'gc_content': gc, 'dg': dg,
        'structure_vector': json.dumps(encode_structure_dot_bracket(structure))
    }

# --- Data Loading Functions ---
# (Using the memory-optimized versions from our previous discussion)
def load_data_from_fasta(folder_path):
    # ... (same as previous version)
    data_dict = {}
    file_paths = _get_files_in_folder(folder_path, ['.fasta', '.fa', '.txt'])
    if not file_paths: return data_dict
    for filepath in file_paths:
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                data_dict[record.id] = str(record.seq).replace('T', 'U')
        except Exception: pass
    return data_dict

def load_and_inspect_scores(folder_path, id_col, score_col, file_type_name):
    """
    Loads scores from all files in a directory, handles duplicates by taking the max,
    and prints a statistical summary of the scores found.
    """
    data_dict = {}
    all_scores = []
    file_paths = _get_files_in_folder(folder_path, ['.txt', '.tsv', '.csv'])
    if not file_paths: 
        print(f"  - No {file_type_name} files found in '{folder_path}'.")
        return data_dict
    
    print(f"  Scanning and inspecting {file_type_name} files in '{folder_path}'...")
    for filepath in file_paths:
        try:
            sep = '\t' if filepath.lower().endswith(('.txt', '.tsv')) else ','
            # Memory optimized for very large files
            df = pd.read_csv(
                filepath, sep=sep, comment='#',
                usecols=lambda column: column.lower().strip() in [id_col, score_col],
                dtype={id_col: str, score_col: float}, na_values=['NULL']
            )
            df.columns = [col.lower().strip() for col in df.columns]
            df.dropna(inplace=True)

            for _, row in df.iterrows():
                key = str(row[id_col])
                score = float(row[score_col])
                all_scores.append(score)
                data_dict[key] = max(data_dict.get(key, 0.0), score)
        except Exception as e:
            print(f"    - Error loading score file {filepath}: {e}")

    # NEW: Add the diagnostic summary
    if all_scores:
        scores_series = pd.Series(all_scores)
        print(f"    - Statistical Summary for {file_type_name} Scores:")
        # Indent the describe() output for better readability
        summary = scores_series.describe().to_string().replace('\n', '\n      ')
        print(f"      {summary}")
    else:
        print(f"    - No valid scores found in any of the files.")
    
    return data_dict

# --- Main Dataset Preparation Function ---
def prepare_dataset():
    start_time = time.time()
    print("--- Starting Extraordinary Dataset Preparation ---")
    if FOCUS_ON_STEM_IIB_ONLY:
        print("!!! MODE: Specialist model training. Focusing ONLY on RRE Stem IIB region. !!!")
    
    print("\nStep 1: Loading all source files...")
    all_mirnas = load_data_from_fasta(MIRNA_DATA_DIR)
    all_rre_full = load_data_from_fasta(RRE_DATA_DIR) # Load full RREs first
    all_rev = load_data_from_fasta(REV_DATA_DIR)
    affinity = load_and_inspect_scores(AFFINITY_DATA_DIR, 'mirna', 'interaction_score', 'Affinity')
    conservation = load_and_inspect_scores(CONSERVATION_DATA_DIR, 'mir family', 'pct', 'Conservation')

    # --- NEW: Filter and process RREs based on the hybrid switch ---
    all_rre_processed = {}
    if FOCUS_ON_STEM_IIB_ONLY:
        print("\nFiltering RREs for Stem IIB region (bases 90-150)...")
        for rre_id, full_seq in all_rre_full.items():
            if len(full_seq) >= 150:
                # Extract the subsequence for Stem IIB
                stem_iib_seq = full_seq[90:150]
                # Create a new ID to reflect this is a sub-region
                all_rre_processed[f"{rre_id}_StemIIB"] = stem_iib_seq
        print(f"  - Kept {len(all_rre_processed)} out of {len(all_rre_full)} RREs that were long enough.")
    else:
        print("\nUsing full RRE sequences for training.")
        all_rre_processed = all_rre_full

    if not all_rre_processed:
        print("\nCRITICAL ERROR: No RRE sequences remained after filtering. Cannot proceed.")
        return

    print(f"  Loaded {len(all_mirnas)} miRNAs, {len(all_rre_processed)} RREs (for processing), {len(all_rev)} REVs.")

    print("\nStep 2: Pre-processing miRNAs in parallel...")
    # ENHANCEMENT: Use multiprocessing to speed up RNAfold
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_mirna_entry, all_mirnas.items())

    processed_mirnas = []
    reject_log = {"length": 0, "gc": 0, "structure": 0}
    for result in results:
        if isinstance(result, dict):
            processed_mirnas.append(result)
        else:
            _, reason = result
            if reason == "reject_length": reject_log["length"] += 1
            elif reason == "reject_gc": reject_log["gc"] += 1
            elif reason == "reject_structure": reject_log["structure"] += 1

    print(f"  {len(processed_mirnas)} miRNAs passed filters.")
    print(f"  Rejection Log: Length={reject_log['length']}, GC%={reject_log['gc']}, Structure Fail={reject_log['structure']}")

    print("\nStep 3: Augmenting data with labels and scores...")
    for mirna_data in processed_mirnas:
        mirna_id = mirna_data['mirna_id']
        affinity_score = affinity.get(mirna_id, 0.0)
        mirna_data['label'] = 1 if affinity_score > AFFINITY_THRESHOLD else 0
        mirna_data['affinity'] = affinity_score
        
        # ENHANCEMENT: More robust miRNA family name parsing
        match = re.search(r"mir-\d+[a-z]?", mirna_id.lower())
        mirna_family = match.group(0) if match else mirna_id.lower()
        mirna_data['conservation'] = conservation.get(mirna_family, 0.0)
        
    print("\nStep 4: Creating augmented competitor list...")
    null_competitor = ('NO_COMPETITOR', '')
    all_competitors = list(all_rev.items()) + [null_competitor]
    print(f"  Augmented list created with {len(all_competitors)} entries.")

    print(f"\nStep 5: Generating and streaming all combinations to CSV (Batch Size: {CSV_BATCH_SIZE})...")
    output_path = os.path.join(PREPARED_DATASET_DIR, "Prepared_Dataset.csv")
    
    # ENHANCEMENT: Stream to CSV in batches to handle millions of rows without memory failure
    header_written = False
    batch = []
    total_rows = 0
    combinations = product(processed_mirnas, all_rre_processed.items(), all_competitors)

    for mirna_data, (rre_id, rre_seq), (rev_id, rev_seq) in combinations:
        row = mirna_data.copy()
        row.update({'rre_id': rre_id, 'rre_sequence': rre_seq, 'rev_id': rev_id, 'rev_sequence': rev_seq})
        batch.append(row)

        if len(batch) >= CSV_BATCH_SIZE:
            df_batch = pd.DataFrame(batch)
            df_batch.to_csv(output_path, mode='a', index=False, header=not header_written)
            header_written = True
            total_rows += len(batch)
            print(f"  ... {total_rows} rows written to CSV")
            batch = [] # Reset the batch

    # Write any remaining rows in the last batch
    if batch:
        df_batch = pd.DataFrame(batch)
        df_batch.to_csv(output_path, mode='a', index=False, header=not header_written)
        total_rows += len(batch)

    end_time = time.time()
    print("\n--- Dataset Preparation Summary ---")
    print(f"Total combinations generated (rows): {total_rows}")
    print(f"Dataset saved successfully to {output_path}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    prepare_dataset()