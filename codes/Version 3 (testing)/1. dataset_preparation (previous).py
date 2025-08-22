# 1. dataset_preparation.py (Revised for large, unbiased dataset generation)
import os
import pandas as pd
from Bio import SeqIO
import subprocess
import numpy as np
import re
import json
from itertools import product

# --- Configuration Constants ---
DATASET_ROOT_FOLDER = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset"
MIRNA_FOLDER_NAME = "Human miRNA dataset"
AFFINITY_FOLDER_NAME = "Affinity_Interaction Score file"
CONSERVATION_FOLDER_NAME = "Conservation Family Information file"
RRE_FOLDER_NAME = "RRE FASTA file"
REV_FOLDER_NAME = "REV dataset"
PREPARED_DATASET_FOLDER_NAME = "Prepared Dataset"

# Paths point to the 'Select' subfolders
MIRNA_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, MIRNA_FOLDER_NAME, 'Select')
AFFINITY_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, AFFINITY_FOLDER_NAME, 'Select')
CONSERVATION_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, CONSERVATION_FOLDER_NAME, 'Select')
RRE_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, RRE_FOLDER_NAME, 'Select')
REV_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, REV_FOLDER_NAME, 'Select')
PREPARED_DATASET_DIR = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME)

os.makedirs(PREPARED_DATASET_DIR, exist_ok=True)

# Filtering Criteria
SEQUENCE_LENGTH_MIN = 20
SEQUENCE_LENGTH_MAX = 70
GC_CONTENT_MIN = 0.30
GC_CONTENT_MAX = 0.80
AFFINITY_THRESHOLD = 0.6

# --- Helper Functions (Mostly unchanged) ---
def calculate_gc_content(sequence):
    if not sequence: return 0
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    return gc_count / len(sequence)

def predict_structure(sequence):
    try:
        result = subprocess.run(['RNAfold'], input=sequence, capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) >= 2:
            structure_line = output_lines[1].split(' ')[0]
            match = re.search(r'\(\s*([-+]?\d+\.\d+)\)', output_lines[1])
            dg = float(match.group(1)) if match else 0.0
            return structure_line, dg
        return None, None
    except Exception:
        return None, None

def encode_structure_dot_bracket(structure):
    mapping = {'.': 0, '(': 1, ')': -1}
    return [mapping.get(char, 0) for char in structure]

def validate_seed(mirna_seq, target_seq):
    if len(mirna_seq) < 8: return False
    seed_region = mirna_seq[1:8]
    complement = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
    rc_seed = "".join([complement.get(base.upper(), 'N') for base in reversed(seed_region)])
    return rc_seed in target_seq.upper()

def _get_files_in_folder(folder_path, extensions):
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}. Skipping.")
        return []
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in extensions)]

# --- Data Loading Functions ---
def load_data_from_fasta(folder_path):
    """Generic function to load ID and sequence from FASTA files."""
    data_dict = {}
    for filepath in _get_files_in_folder(folder_path, ['.fasta', '.fa']):
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                data_dict[record.id] = str(record.seq).replace('T', 'U')
        except Exception as e:
            print(f"Error loading FASTA file {filepath}: {e}")
    return data_dict

def load_affinity(folder_path):
    """Loads affinity scores, taking the max score for any duplicates."""
    affinity_data = {}
    for filepath in _get_files_in_folder(folder_path, ['.txt', '.tsv', '.csv']):
        try:
            sep = '\t' if filepath.lower().endswith(('.txt', '.tsv')) else ','
            df = pd.read_csv(filepath, sep=sep, comment='#')
            df.columns = [col.lower().strip() for col in df.columns]
            if 'mirna' not in df.columns or 'interaction_score' not in df.columns: continue
            df['interaction_score'] = pd.to_numeric(df['interaction_score'], errors='coerce').dropna()
            for _, row in df.iterrows():
                affinity_data[str(row['mirna'])] = max(affinity_data.get(str(row['mirna']), 0.0), float(row['interaction_score']))
        except Exception as e:
            print(f"Error loading affinity file {filepath}: {e}")
    return affinity_data

# --- Main Dataset Preparation Function ---
def prepare_dataset():
    print("--- Starting Dataset Preparation ---")
    
    # 1. Load all source data
    print("Step 1: Loading all source files...")
    all_mirnas = load_data_from_fasta(MIRNA_DATA_DIR)
    all_rre = load_data_from_fasta(RRE_DATA_DIR)
    all_rev = load_data_from_fasta(REV_DATA_DIR)
    affinity = load_affinity(AFFINITY_DATA_DIR)
    # Conservation loading can be added here if needed
    
    print(f"  Loaded {len(all_mirnas)} total miRNAs.")
    print(f"  Loaded {len(all_rre)} total RREs.")
    print(f"  Loaded {len(all_rev)} total REVs.")
    print(f"  Loaded {len(affinity)} affinity scores.")

    # 2. Pre-process and filter miRNAs
    print("\nStep 2: Pre-processing and filtering miRNAs...")
    processed_mirnas = []
    for mirna_id, mirna_seq in all_mirnas.items():
        # Filter by length and GC content
        if not (SEQUENCE_LENGTH_MIN <= len(mirna_seq) <= SEQUENCE_LENGTH_MAX): continue
        gc = calculate_gc_content(mirna_seq)
        if not (GC_CONTENT_MIN <= gc <= GC_CONTENT_MAX): continue
        
        # Predict structure
        structure, dg = predict_structure(mirna_seq)
        if structure is None: continue
        
        struct_vec = encode_structure_dot_bracket(structure)
        
        # Get affinity score, default to 0.0 if not found
        affinity_score = affinity.get(mirna_id, 0.0)
        label = 1 if affinity_score > AFFINITY_THRESHOLD else 0
        
        processed_mirnas.append({
            'mirna_id': mirna_id,
            'sequence': mirna_seq,
            'gc_content': gc,
            'dg': dg,
            'structure_vector': json.dumps(struct_vec),
            'affinity': affinity_score,
            'label': label,
            'conservation': 0.0 # Placeholder for conservation
        })
    
    print(f"  {len(processed_mirnas)} miRNAs passed pre-processing filters.")

    # 3. Create all possible combinations
    print("\nStep 3: Creating all possible (miRNA x RRE x REV) combinations...")
    
    final_data = []
    
    # Create a list of all RRE and REV items to combine
    rre_list = list(all_rre.items())
    rev_list = list(all_rev.items())

    # Use itertools.product to efficiently create all combinations
    # This is the core change that generates the large dataset
    combinations = product(processed_mirnas, rre_list, rev_list)

    for mirna_data, (rre_id, rre_seq), (rev_id, rev_seq) in combinations:
        
        # We can add a seed match check here if we still want to filter combinations
        # For now, we will include all combinations to create a large dataset
        # You can uncomment the following lines to re-enable seed match filtering
        
        # has_seed_match = False
        # target_regions = {"Stem_IIB": rre_seq[39:80], "Stem_IA": rre_seq[19:35]}
        # for region_name, region_seq in target_regions.items():
        #     if validate_seed(mirna_data['sequence'], region_seq):
        #         has_seed_match = True
        #         break
        # if not has_seed_match:
        #     continue

        row = mirna_data.copy()
        row.update({
            'rre_id': rre_id,
            'rre_sequence': rre_seq,
            'rev_id': rev_id,
            'rev_sequence': rev_seq,
        })
        final_data.append(row)

    df = pd.DataFrame(final_data)

    print("\n--- Dataset Preparation Summary ---")
    print(f"Total unique miRNAs after filtering: {len(processed_mirnas)}")
    print(f"Total combinations generated (rows in final dataset): {len(df)}")

    if not df.empty:
        print("\nClass Distribution in Final Dataset:")
        print(df['label'].value_counts())
        
        output_path = os.path.join(PREPARED_DATASET_DIR, "prepared_miRNA_RRE_dataset.csv")
        df.to_csv(output_path, index=False)
        print(f"\nDataset saved successfully to {output_path}")
    else:
        print("\nWARNING: Resulting dataset is empty. Check your input files and filters.")

    return df

# --- Main Execution ---
if __name__ == "__main__":
    prepare_dataset()