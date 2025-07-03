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
GC_CONTENT_MIN = 0.30
GC_CONTENT_MAX = 0.80

DATASET_FOLDER = r"E:\\my_deep_learning_project\\dataset"
MIRNA_FASTA = os.path.join(DATASET_FOLDER, "Human_miRNA.fasta")
AFFINITY_PATH = os.path.join(DATASET_FOLDER, "interactions_human.microT.mirbase.txt")
CONSERVATION_PATH = os.path.join(DATASET_FOLDER, "Conserved_Family_Info.txt")

RRE_SEQUENCE = "GGGUGUGGAAAUCUCUGGGUUAGACCAGAUCTGAGCUGGGUUCUCUGGGCAGCCAGAGGUGGUCUUAGCCUUCUUGAAUCCUGGCCUCCUCCAGGAUCCCAGGGUUCAAAUCCCACUGGCCUUGGCUGAAGGGGCAGUAGUCCUUCUGAUUGGCCAGGCUGCCUUCUGCUCCUGCUGGCCAGGCAGGUGCUGGCCACUAGCUGGUGACUAGUGACUUGCUGAUAGGGUGGGCUAUUUUCCUACU"
KNOWN_LOOP_SITES = [("AGGUGGU", "Stem IIB"), ("GAAGGGGCA", "Loop I"), ("CCUUCUGAUU", "Loop II")]

# Prepare target regions based on known loop sites
TARGET_REGIONS = {}
for motif, name in KNOWN_LOOP_SITES:
    idx = RRE_SEQUENCE.find(motif)
    if idx != -1:
        start = max(idx - 20, 0)
        end = min(idx + len(motif) + 20, len(RRE_SEQUENCE))
        TARGET_REGIONS[name] = RRE_SEQUENCE[start:end]

# Define nucleotide complements
NUCLEOTIDES = ['A', 'U', 'G', 'C']
complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}

# Helper functions for calculations and validation
def calculate_gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

def validate_seed(mirna, region):
    mirna_seed = mirna[1:8] 
    
    for i in range(len(region) - len(mirna_seed) + 1):
        target_site = region[i : i + len(mirna_seed)]
        
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
            return None, None

        parts = structure_line.rsplit(' ', 1)
        if len(parts) < 2:
            return None, None
        
        structure = parts[0].strip()
        energy_str = parts[1].strip('()')
        energy = float(energy_str)
        
        return structure, energy

    except subprocess.CalledProcessError as e:
        return None, None
    except FileNotFoundError:
        print("Error: 'RNAfold' command not found. Ensure ViennaRNA Package is installed and in PATH.")
        return None, None
    except ValueError as e:
        return None, None
    except Exception as e:
        return None, None

# Functions to encode structure and load data
def encode_structure_dot_bracket(dot):
    encoded = [1 if x == '(' else -1 if x == ')' else 0 for x in dot]
    return (encoded + [0] * 25)[:25]

def load_affinity():
    df = pd.read_csv(AFFINITY_PATH, sep='\t') 
    
    if 'mirna' not in df.columns or 'interaction_score' not in df.columns:
        raise ValueError(
            f"Expected 'mirna' and 'interaction_score' columns in {AFFINITY_PATH}, "
            f"but found: {df.columns.tolist()}"
        )
    df['mirna'] = df['mirna'].astype(str)
    df['interaction_score'] = pd.to_numeric(df['interaction_score'], errors='coerce')

    df = df.dropna(subset=['interaction_score'])

    return {row.mirna: float(row.interaction_score) for row in df.itertuples()}

# Helper to generate potential miRNA family names from a mature miRNA ID
def get_potential_mirna_family_names(mirna_id):
    potential_names = set()
    original_id = mirna_id.strip().lower()

    if original_id.startswith('hsa-'):
        base_name = original_id[4:]
    else:
        base_name = original_id

    potential_names.add(base_name)

    if base_name.endswith(('-5p', '-3p')):
        family_no_arm = base_name[:-3]
        potential_names.add(family_no_arm)
        if family_no_arm.startswith('mir-'):
            potential_names.add('miR-' + family_no_arm[4:])
    
    if '.' in base_name and base_name.split('.')[-1].isdigit():
        family_no_variant = base_name.rsplit('.', 1)[0]
        potential_names.add(family_no_variant)
        if family_no_variant.startswith('mir-'):
            potential_names.add('miR-' + family_no_variant[4:])

    if base_name.endswith('*'):
        family_no_star = base_name[:-1]
        potential_names.add(family_no_star)
        if family_no_star.startswith('mir-'):
            potential_names.add('miR-' + family_no_star[4:])

    if not base_name.startswith('mir-') and not base_name.startswith('let-'):
        if len(base_name) > 0 and base_name[0].isdigit():
            potential_names.add('miR-' + base_name)
            if base_name.endswith(('-5p', '-3p')):
                potential_names.add('miR-' + base_name[:-3])

    final_potential_names = set()
    for name in potential_names:
        if name.startswith('mir-'):
            final_potential_names.add('miR-' + name[4:])
        elif name.startswith('let-'):
            final_potential_names.add(name)
        else:
            final_potential_names.add(name)

    return list(final_potential_names)

def load_conservation():
    df = pd.read_csv(CONSERVATION_PATH, sep='\t', na_values=['NULL'])
    
    # Rename the column to a valid attribute name to fix the original error
    df.rename(columns={'miR Family': 'miR_Family'}, inplace=True)
    
    if 'miR_Family' not in df.columns or 'PCT' not in df.columns:
        raise ValueError(
            f"Expected 'miR_Family' and 'PCT' columns in {CONSERVATION_PATH}, "
            f"but found: {df.columns.tolist()}"
        )
    
    df['miR_Family'] = df['miR_Family'].astype(str)
    df['PCT'] = pd.to_numeric(df['PCT'], errors='coerce')

    family_conservation = {}
    for row in df.itertuples():
        family_name = row.miR_Family 
        pct = row.PCT

        if pd.isna(pct):
            continue
        
        family_conservation[family_name.lower()] = max(family_conservation.get(family_name.lower(), 0.0), pct)
    
    return family_conservation

def load_mirnas():
    mirna_data = []
    
    for r in SeqIO.parse(MIRNA_FASTA, "fasta"):
        mirna_seq = str(r.seq).replace('T', 'U')
        # This length filter is correct for mature miRNAs
        if SEQUENCE_LENGTH_MIN <= len(mirna_seq) <= SEQUENCE_LENGTH_MAX:
            mirna_data.append((r.id, mirna_seq))
            
    print(f"Loaded {len(mirna_data)} miRNAs from {MIRNA_FASTA} meeting length criteria ({SEQUENCE_LENGTH_MIN}-{SEQUENCE_LENGTH_MAX}).")
    return mirna_data

# Main dataset preparation function
def prepare_dataset(output_csv):
    all_mirnas_with_ids = load_mirnas()
    affinity = load_affinity()
    conservation_by_family = load_conservation()
    data = []
    
    print(f"Loaded {len(all_mirnas_with_ids)} miRNAs with IDs.")
    print(f"Loaded {len(affinity)} affinity scores.")
    print(f"Loaded {len(conservation_by_family)} conservation families.")
    
    processed_count = 0
    skipped_count_length_gc = 0
    skipped_count_rnafold = 0
    skipped_count_seed = 0
    skipped_count_no_family_match = 0
    skipped_count_no_affinity = 0

    for i, (mirna_id, mirna_seq) in enumerate(all_mirnas_with_ids):
        processed_count += 1

        # Add a progress indicator
        if (i + 1) % 1000 == 0: # Print every 1000 miRNAs
            print(f"Processing miRNA {i + 1}/{len(all_mirnas_with_ids)}: {mirna_id}")
        
        # This GC content filter is correct for mature miRNAs
        gc = calculate_gc_content(mirna_seq)
        if not (GC_CONTENT_MIN <= gc <= GC_CONTENT_MAX):
            skipped_count_length_gc += 1
            continue

        structure, dg = predict_structure(mirna_seq)
        if structure is None or dg is None:
            skipped_count_rnafold += 1
            continue
        
        struct_vec = encode_structure_dot_bracket(structure)

        potential_family_names = get_potential_mirna_family_names(mirna_id)
        
        mirna_conservation_score = 0.0
        found_conservation = False
        for family_name_attempt in potential_family_names:
            if family_name_attempt in conservation_by_family:
                mirna_conservation_score = conservation_by_family[family_name_attempt]
                found_conservation = True
                break

        if not found_conservation:
            skipped_count_no_family_match += 1
        
        seed_match_found = False
        for region_name, region_seq in TARGET_REGIONS.items():
            if validate_seed(mirna_seq, region_seq):
                seed_match_found = True
                
                mirna_affinity_score = affinity.get(mirna_id, None)
                if mirna_affinity_score is None:
                    skipped_count_no_affinity += 1
                    continue

                label = 1 if mirna_affinity_score > 0.7 else 0

                data.append({
                    'mirna_id': mirna_id,
                    'sequence': mirna_seq,
                    'region': region_name,
                    'gc_content': gc,
                    'dg': dg,
                    'conservation': mirna_conservation_score,
                    'affinity': mirna_affinity_score,
                    'structure_vector': struct_vec,
                    'label': label
                })
        
        if not seed_match_found:
            skipped_count_seed += 1
            continue

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    print(f"\n--- Dataset Preparation Summary ---")
    print(f"Total miRNAs processed: {processed_count}")
    print(f"Skipped due to length/GC content: {skipped_count_length_gc}")
    print(f"Skipped due to RNAfold error: {skipped_count_rnafold}")
    print(f"Skipped due to no seed match in any target region: {skipped_count_seed}")
    print(f"Skipped due to no family match for conservation score (assigned 0.0): {skipped_count_no_family_match}")
    print(f"Skipped due to no affinity score: {skipped_count_no_affinity}")
    print(f"Final DataFrame size: {df.shape[0]} rows")

    return df

# Main execution block
if __name__ == "__main__":
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print("Preparing dataset...")
    output_dataset_csv = os.path.join(DATASET_FOLDER, "prepared_miRNA_dataset.csv")
    dataset_df = prepare_dataset(output_dataset_csv)
    print(f"Dataset preparation complete. Saved to {output_dataset_csv}")
    
    if not dataset_df.empty:
        print(f"Dataset shape: {dataset_df.shape}")
        print(dataset_df.head())
    else:
        print("Resulting dataset is empty. Please check input files and filtering criteria.")