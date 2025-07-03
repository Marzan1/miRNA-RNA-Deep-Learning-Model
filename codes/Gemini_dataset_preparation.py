import os
import csv
import pandas as pd
import random
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Define global constants and paths
SEQUENCE_LENGTH_MIN = 21
SEQUENCE_LENGTH_MAX = 25
GC_CONTENT_MIN = 0.50
GC_CONTENT_MAX = 0.60

DATASET_FOLDER = r"E:\\my_deep_learning_project\\dataset"
MIRNA_FASTA = os.path.join(DATASET_FOLDER, "Human_miRNA.fasta")
AFFINITY_PATH = os.path.join(DATASET_FOLDER, "interactions_human.microT.mirbase.txt")
CONSERVATION_PATH = os.path.join(DATASET_FOLDER, "Conserved_Family_Info.txt") # Check column names in this file

RRE_SEQUENCE = "GGGUGUGGAAAUCUCUGGGUUAGACCAGAUCTGAGCUGGGUUCUCUGGGCAGCCAGAGGUGGUCUUAGCCUUCUUGAAUCCUGGCCUCCUCCAGGAUCCCAGGGUUCAAAUCCCACUGGCCUUGGCUGAAGGGGCAGUAGUCCUUCUGAUUGGCCAGGCUGCCUUCUGCUCCUGCUGGCCAGGCAGGUGCUGGCCACUAGCUGGUGACUAGUGACUUGCUGAUAGGGUGGGCUAUUUUCCUACU"
KNOWN_LOOP_SITES = [("AGGUGGU", "Stem IIB"), ("GAAGGGGCA", "Loop I"), ("CCUUCUGAUU", "Loop II")]

# Prepare target regions based on known loop sites
TARGET_REGIONS = {}
for motif, name in KNOWN_LOOP_SITES:
    idx = RRE_SEQUENCE.find(motif)
    if idx != -1:
        start = max(idx - 20, 0)
        end = min(idx + len(motif) + 20, len(RRE_SEQUENCE)) # Ensure 'end' doesn't exceed RRE_SEQUENCE length
        TARGET_REGIONS[name] = RRE_SEQUENCE[start:end]

# Define nucleotide complements
NUCLEOTIDES = ['A', 'U', 'G', 'C']
complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}

# Helper functions for calculations and validation
def calculate_gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

def validate_seed(mirna, region):
    # Implements the standard 7mer-m8 seed match: perfect complementarity from bases 2-8 of miRNA to target.
    # Assumes region is the mRNA target sequence.
    # miRNA seed is typically bases 2-7 or 2-8 (1-indexed or 0-indexed bases 1-6/7).
    # Here, we use 0-indexed bases [1:8] which are 7 bases.
    # For a stronger match, you might require a minimum number of matches or specific match types.
    
    mirna_seed = mirna[1:8] # Bases 2 through 8 (0-indexed 1 to 7)
    
    # Iterate through the region to find potential binding sites
    for i in range(len(region) - len(mirna_seed) + 1):
        target_site = region[i : i + len(mirna_seed)]
        
        # Check for perfect Watson-Crick complementarity (G:C, A:U)
        # Note: G:U wobble pairs are often allowed, but for simplicity, starting with perfect match
        is_perfect_match = True
        for j in range(len(mirna_seed)):
            if mirna_seed[j] != complement.get(target_site[j], ''):
                is_perfect_match = False
                break
        if is_perfect_match:
            return True
    return False

# Function to predict RNA structure using RNAfold
def predict_structure(seq):
    try:
        result = subprocess.run(
            ["RNAfold"],
            input=f"{seq}\n",
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.strip().split('\n')
        
        structure_line = None
        for line in lines:
            if line and ('(' in line or '.' in line or ')' in line) and line.strip().endswith(')'):
                if '(' in line and ')' in line and line.rfind('(') < line.rfind(')'):
                    structure_line = line
                    break
        
        if not structure_line:
            raise ValueError("Could not find structure and energy line in RNAfold output.")

        parts = structure_line.rsplit(' ', 1)
        if len(parts) < 2:
            raise ValueError(f"Could not parse structure and energy from line: '{structure_line}'")
        
        structure = parts[0].strip()
        energy_str = parts[1].strip('()')
        energy = float(energy_str)
        
        return structure, energy

    except subprocess.CalledProcessError as e:
        print(f"Error running RNAfold for sequence '{seq}' (exit code {e.returncode}): {e}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return None, None
    except FileNotFoundError:
        print("Error: 'RNAfold' command not found. Ensure ViennaRNA Package is installed and in PATH.")
        return None, None
    except ValueError as e:
        print(f"Error parsing RNAfold output for sequence '{seq}': {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred for sequence '{seq}': {e}")
        return None, None

# Functions to encode structure and load data
def encode_structure_dot_bracket(dot):
    # Pad with '.' if shorter than 25, truncate if longer.
    return [1 if x == '(' else -1 if x == ')' else 0 for x in dot.ljust(25, '.')[:25]]

def load_affinity():
    # Read the CSV. By default, header='infer' or header=0 is used,
    # which correctly treats the first row as headers.
    df = pd.read_csv(AFFINITY_PATH, sep='\t') 
    
    # Check if 'mirna' and 'interaction_score' columns exist
    if 'mirna' not in df.columns or 'interaction_score' not in df.columns:
        raise ValueError(
            f"Expected 'mirna' and 'interaction_score' columns in {AFFINITY_PATH}, "
            f"but found: {df.columns.tolist()}"
        )

    # Use .itertuples() without index=False to get named tuples.
    # This allows access by column name (e.g., row.mirna, row.interaction_score).
    # This is the most readable and robust way for files with headers.
    return {row.mirna: float(row.interaction_score) for row in df.itertuples()}

def load_conservation():
    # Assuming 'Conserved_Family_Info.txt' is tab-separated with no header as per TargetScan downloads
    # Adjust column indices [0] for miRNA and [9] for PCT score if that's what you mean by 'score'.
    # If it has headers, use names: df = pd.read_csv(CONSERVATION_PATH, sep='\t')
    # and return dict(zip(df['miRNA_column_name'], df['PCT_score_column_name']))
    df = pd.read_csv(CONSERVATION_PATH, sep='\t', header=None) # Assuming no header based on typical TargetScan
    # Example: TargetScan's 'Conserved Family Info' often has miR Family in col 0, PCT in col 9 (0-indexed)
    # You need to confirm the actual column indices from your specific file.
    # For now, let's assume column 0 is miRNA and a score column (e.g., column 9 for PCT if available)
    # THIS MIGHT NEED ADJUSTMENT based on actual file format
    
    # If the file is 'Conserved_Family_Info.txt' from default predictions, it might not have "score"
    # It has 'PCT' (Probability of Conserved Target). Let's use that if available.
    # If your file 'Conserved_Family_Info.txt' actually has a 'score' column as indicated in your initial description,
    # then you can use: df = pd.read_csv(CONSERVATION_PATH); return dict(zip(df['miRNA'], df['score']))
    
    # For a TargetScan 'Conserved Family Info' file, PCT is often the relevant score.
    # This might require parsing the file correctly to get the right miRNA identifier.
    # The miR Family column from 'Conserved Family Info.txt' might need careful handling to link back to individual mature miRNAs.
    # If it's the `Predicted Targets (default predictions)` file that has PCT, you'll need to parse that or map.
    # **Crucially, the file `Conserved_Family_Info.txt` from TargetScan contains miR Family, not individual miRNAs.**
    # You might need to map miRNAs to their family and then get the family conservation.
    # For now, let's use a dummy return if the file is not directly mappable.
    print(f"Warning: load_conservation is using dummy data or requires specific column index check for {CONSERVATION_PATH}")
    return {"dummy_mirna_seq": 0.9} # Placeholder - YOU MUST ADAPT THIS TO YOUR FILE'S ACTUAL STRUCTURE AND WHAT "score" means

def load_mirnas():
    return [str(r.seq).replace('T', 'U') for r in SeqIO.parse(MIRNA_FASTA, "fasta")
            if SEQUENCE_LENGTH_MIN <= len(r.seq) <= SEQUENCE_LENGTH_MAX]

# Main dataset preparation function
def prepare_dataset(output_csv):
    mirnas = load_mirnas()
    affinity = load_affinity()
    conservation = load_conservation() # This needs careful review based on your file
    data = []
    
    for mirna in mirnas:
        # Filter based on GC content as early as possible
        gc = calculate_gc_content(mirna)
        if not (GC_CONTENT_MIN <= gc <= GC_CONTENT_MAX):
            continue

        structure, dg = predict_structure(mirna)
        if structure is None or dg is None: # Skip if RNAfold prediction failed
            continue
        
        struct_vec = encode_structure_dot_bracket(structure)

        for region_name, region_seq in TARGET_REGIONS.items():
            if not validate_seed(mirna, region_seq):
                continue
            
            # Using specific default values (0.0 for unknown conservation/affinity)
            # This is a critical decision and affects model training.
            # Consider if unknown values should be excluded or imputed differently.
            mirna_conservation_score = conservation.get(mirna, 0.0) # Default to 0 if not found
            mirna_affinity_score = affinity.get(mirna, 0.0) # Default to 0 if not found

            data.append({
                'sequence': mirna,
                'region': region_name,
                'gc_content': gc,
                'dg': dg,
                'conservation': mirna_conservation_score,
                'affinity': mirna_affinity_score,
                'structure_vector': struct_vec,
                # Label based on affinity, using a threshold.
                # If affinity is 0.0 (default for unknown), it will be labeled 0.
                'label': 1 if mirna_affinity_score > 0.7 else 0 
            })
            
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    return df

# Main execution block
if __name__ == "__main__":
    # Set seeds for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Prepare the dataset
    print("Preparing dataset...")
    output_dataset_csv = os.path.join(DATASET_FOLDER, "prepared_miRNA_dataset.csv")
    dataset_df = prepare_dataset(output_dataset_csv)
    print(f"Dataset preparation complete. Saved to {output_dataset_csv}")
    print(f"Dataset shape: {dataset_df.shape}")
    print(dataset_df.head())

    # Further steps (e.g., model definition, training, evaluation) would follow here.
    # This example only covers data preparation.