import os
import pandas as pd
from Bio import SeqIO
import subprocess
import numpy as np

# --- Configuration Constants ---
# Base directory where all your subfolders are located
DATASET_ROOT_FOLDER = r"E:\my_deep_learning_project\dataset" # Use raw string for Windows paths

# Subfolder names within DATASET_ROOT_FOLDER
MIRNA_FOLDER_NAME = "Human miRNA dataset"
AFFINITY_FOLDER_NAME = "Affinity_Interaction Score file"
CONSERVATION_FOLDER_NAME = "Conservation Family Information file"
RRE_FOLDER_NAME = "RRE FASTA file"
REV_FOLDER_NAME = "REV dataset" # New folder for REV data
PREPARED_DATASET_FOLDER_NAME = "Prepared Dataset" # Folder to save output

# Full paths to your data folders
MIRNA_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, MIRNA_FOLDER_NAME)
AFFINITY_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, AFFINITY_FOLDER_NAME)
CONSERVATION_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, CONSERVATION_FOLDER_NAME)
RRE_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, RRE_FOLDER_NAME)
REV_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, REV_FOLDER_NAME)
PREPARED_DATASET_DIR = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME)

# Ensure the output directory exists
os.makedirs(PREPARED_DATASET_DIR, exist_ok=True)

# Filtering Criteria
SEQUENCE_LENGTH_MIN = 21
SEQUENCE_LENGTH_MAX = 25
GC_CONTENT_MIN = 0.40
GC_CONTENT_MAX = 0.70
AFFINITY_THRESHOLD = 0.7 # For binary labeling (label 1 if > 0.7, else 0)

# --- Helper Functions ---

def calculate_gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0

def predict_structure(sequence):
    try:
        result = subprocess.run(
            ['RNAfold'],
            input=sequence.encode('utf-8'),
            capture_output=True,
            text=True,
            check=True
        )
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) >= 2:
            structure_line = output_lines[1].split(' ')[0]
            dg_str = output_lines[1].split('(')[1].split(')')[0]
            dg = float(dg_str)
            return structure_line, dg
        return None, None
    except FileNotFoundError:
        print("Error: 'RNAfold' command not found. Ensure ViennaRNA Package is installed and in PATH.")
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"Error running RNAfold for sequence {sequence}: {e}")
        print(f"Stderr: {e.stderr}")
        return None, None
    except ValueError:
        print(f"Error parsing RNAfold output for sequence {sequence}: {output_lines}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred with RNAfold for sequence {sequence}: {e}")
        return None, None

def encode_structure_dot_bracket(structure):
    max_len = SEQUENCE_LENGTH_MAX
    vector = np.zeros(max_len, dtype=int)
    for i, char in enumerate(structure[:max_len]):
        if char == '.':
            vector[i] = 0
        elif char == '(':
            vector[i] = 1
        elif char == ')':
            vector[i] = -1
    return vector.tolist()

def validate_seed(mirna_seq, target_seq):
    if len(mirna_seq) < 8:
        return False

    seed_region = mirna_seq[1:8] # Bases 2 to 8 (0-indexed: 1 to 7)

    complement = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
    rc_seed = "".join([complement[base] for base in seed_region[::-1]])

    if len(target_seq) < len(rc_seed):
        return False

    return rc_seed in target_seq

def get_potential_mirna_family_names(mirna_id):
    parts = mirna_id.split('-')
    if len(parts) >= 3 and parts[0] == 'hsa' and parts[1] == 'miR':
        family_name = f"miR-{parts[2]}".lower()
        if len(parts) > 3 and (parts[3] == '3p' or parts[3] == '5p'):
            stem_family_name = f"miR-{parts[2]}".lower()
            return [family_name, stem_family_name]
        return [family_name]
    return []

# --- Data Loading Functions (Modified for automatic file discovery) ---

def _get_files_in_folder(folder_path, extensions):
    """Helper to get all files with specified extensions in a folder."""
    files = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}. Skipping.")
        return []
    for f in os.listdir(folder_path):
        if any(f.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(folder_path, f))
    return files

def load_mirnas():
    mirna_data = []
    seen_mirna_ids = set()
    mirna_fasta_files = _get_files_in_folder(MIRNA_DATA_DIR, ['.fasta', '.fa'])

    if not mirna_fasta_files:
        print(f"No miRNA FASTA files found in {MIRNA_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in mirna_fasta_files:
        try:
            current_file_mirnas = 0
            for r in SeqIO.parse(filepath, "fasta"):
                mirna_seq = str(r.seq).replace('T', 'U')
                if SEQUENCE_LENGTH_MIN <= len(mirna_seq) <= SEQUENCE_LENGTH_MAX:
                    if r.id not in seen_mirna_ids:
                        mirna_data.append((r.id, mirna_seq))
                        seen_mirna_ids.add(r.id)
                        current_file_mirnas += 1
            print(f"Loaded {current_file_mirnas} unique miRNAs from {filepath} meeting length criteria.")
        except Exception as e:
            print(f"Error loading miRNA FASTA file {filepath}: {e}. Skipping.")

    print(f"Total unique miRNAs loaded across all files: {len(mirna_data)}")
    return mirna_data

def load_affinity():
    all_affinity_data = {}
    affinity_files = _get_files_in_folder(AFFINITY_DATA_DIR, ['.txt', '.tsv', '.csv'])

    if not affinity_files:
        print(f"No affinity files found in {AFFINITY_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in affinity_files:
        try:
            sep = '\t' if filepath.endswith(('.txt', '.tsv')) else ','
            df = pd.read_csv(filepath, sep=sep, comment='#', na_values=['NULL'])

            df.columns = [col.lower().strip() for col in df.columns]
            if 'mirna' not in df.columns or 'interaction_score' not in df.columns:
                print(f"Warning: Expected 'mirna' and 'interaction_score' columns in {filepath}. Found: {df.columns.tolist()}. Skipping.")
                continue

            df['mirna'] = df['mirna'].astype(str)
            df['interaction_score'] = pd.to_numeric(df['interaction_score'], errors='coerce')
            df = df.dropna(subset=['interaction_score'])

            for _, row in df.iterrows():
                mirna_id = str(row['mirna'])
                score = float(row['interaction_score'])
                all_affinity_data[mirna_id] = max(all_affinity_data.get(mirna_id, 0.0), score)

            print(f"Loaded {len(df)} affinity entries from {filepath}.")

        except Exception as e:
            print(f"Error loading affinity file {filepath}: {e}. Skipping.")

    print(f"Total unique miRNAs with affinity scores loaded: {len(all_affinity_data)}")
    return all_affinity_data

def load_conservation():
    all_conservation_data = {}
    conservation_files = _get_files_in_folder(CONSERVATION_DATA_DIR, ['.txt', '.tsv', '.csv'])

    if not conservation_files:
        print(f"No conservation files found in {CONSERVATION_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in conservation_files:
        try:
            sep = '\t' if filepath.endswith(('.txt', '.tsv')) else ','
            df = pd.read_csv(filepath, sep=sep, na_values=['NULL'], comment='#')

            df.columns = [col.lower().strip() for col in df.columns]
            df.rename(columns={'mir family': 'mir_family'}, inplace=True)

            if 'mir_family' not in df.columns or 'pct' not in df.columns:
                print(f"Warning: Expected 'mir_family' and 'pct' columns in {filepath}. Found: {df.columns.tolist()}. Skipping.")
                continue

            df['mir_family'] = df['mir_family'].astype(str)
            df['pct'] = pd.to_numeric(df['pct'], errors='coerce')
            df = df.dropna(subset=['pct'])

            for _, row in df.iterrows():
                family_name = str(row['mir_family']).lower()
                pct = float(row['pct'])
                all_conservation_data[family_name] = max(all_conservation_data.get(family_name, 0.0), pct)

            print(f"Loaded {len(df)} conservation entries from {filepath}.")

        except Exception as e:
            print(f"Error loading conservation file {filepath}: {e}. Skipping.")

    print(f"Total unique conservation families loaded: {len(all_conservation_data)}")
    return all_conservation_data

def load_rre_sequences():
    rre_data = []
    rre_fasta_files = _get_files_in_folder(RRE_DATA_DIR, ['.fasta', '.fa'])

    if not rre_fasta_files:
        print(f"No RRE FASTA files found in {RRE_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in rre_fasta_files:
        try:
            for r in SeqIO.parse(filepath, "fasta"):
                rre_data.append((r.id, str(r.seq).replace('T', 'U')))
            print(f"Loaded {len(rre_data)} RRE sequences from {filepath}.")
        except Exception as e:
            print(f"Error loading RRE FASTA file {filepath}: {e}")

    print(f"Total RRE sequences loaded across all files: {len(rre_data)}")
    return rre_data

def load_rev_data():
    """
    Loads REV sequences or related data.
    This is a placeholder. The exact content and format of your REV data
    will determine how this function needs to be implemented.
    Assuming FASTA files for REV sequences for now.
    """
    rev_data = []
    rev_files = _get_files_in_folder(REV_DATA_DIR, ['.fasta', '.fa', '.txt', '.csv']) # Adjust extensions as needed

    if not rev_files:
        print(f"No REV files found in {REV_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in rev_files:
        try:
            if filepath.lower().endswith(('.fasta', '.fa')):
                for r in SeqIO.parse(filepath, "fasta"):
                    rev_data.append((r.id, str(r.seq))) # Keep protein sequence as is
            elif filepath.lower().endswith(('.txt', '.csv')):
                # Example for a simple text/CSV file: assumes ID, Sequence columns
                df = pd.read_csv(filepath, sep='\t' if filepath.endswith('.txt') else ',')
                df.columns = [col.lower().strip() for col in df.columns]
                if 'id' in df.columns and 'sequence' in df.columns:
                    for _, row in df.iterrows():
                        rev_data.append((str(row['id']), str(row['sequence'])))
                else:
                    print(f"Warning: Skipping {filepath}. Expected 'ID' and 'Sequence' columns for non-FASTA REV data.")

            print(f"Loaded {len(rev_data)} REV entries from {filepath}.")

        except Exception as e:
            print(f"Error loading REV file {filepath}: {e}. Skipping.")

    print(f"Total REV sequences/data loaded across all files: {len(rev_data)}")
    return rev_data


# --- Main Dataset Preparation Function ---

def prepare_dataset():
    print("Preparing dataset...")

    # Load all necessary data using the new folder-based loading functions
    all_mirnas_with_ids = load_mirnas()
    affinity = load_affinity()
    conservation_by_family = load_conservation()
    all_rre_sequences = load_rre_sequences()
    all_rev_data = load_rev_data() # Load REV data

    if not all_mirnas_with_ids:
        print("No miRNAs loaded. Aborting dataset preparation.")
        return pd.DataFrame()
    if not all_rre_sequences:
        print("No RRE sequences loaded. Aborting dataset preparation.")
        return pd.DataFrame()

    data = []
    
    # Statistics for the summary
    processed_miRNA_count = 0
    skipped_miRNA_pre_rre_loop_count = 0 # MiRNAs skipped due to GC or RNAfold BEFORE RRE loop
    
    # Initialize counts for filters AFTER GC/RNAfold, which are applied per (miRNA, RRE) pair
    skipped_no_seed_match_pair_count = 0
    skipped_no_affinity_pair_count = 0


    # Outer loop: Iterate through each miRNA
    for i, (mirna_id, mirna_seq) in enumerate(all_mirnas_with_ids):
        processed_miRNA_count += 1
        
        # --- miRNA-specific calculations and filters (before RRE loop) ---
        gc = calculate_gc_content(mirna_seq)
        if not (GC_CONTENT_MIN <= gc <= GC_CONTENT_MAX):
            skipped_miRNA_pre_rre_loop_count += 1
            continue # Skip this miRNA entirely if GC content is out of range

        structure, dg = predict_structure(mirna_seq)
        if structure is None:
            skipped_miRNA_pre_rre_loop_count += 1
            continue # Skip this miRNA entirely if RNAfold failed
        struct_vec = encode_structure_dot_bracket(structure)

        # Get conservation score for this miRNA
        mirna_conservation_score = 0.0
        family_names = get_potential_mirna_family_names(mirna_id)
        for family in family_names:
            if family in conservation_by_family:
                mirna_conservation_score = conservation_by_family[family]
                break # Take the first match

        # --- Loop through each RRE sequence ---
        for rre_idx, (rre_id, current_rre_seq) in enumerate(all_rre_sequences):
            
            # --- Define TARGET_REGIONS for the CURRENT RRE sequence ---
            # Using a large initial segment for broad coverage to get more data initially.
            # You can refine this to more precise, literature-backed loop regions later.
            current_target_regions = {
                "Primary_RRE_Binding_Region": current_rre_seq[0:min(len(current_rre_seq), 150)] # Take up to 150 bases
            }
            # Ensure the target region is not empty if RRE is too short or definition is bad
            for region_name, region_seq in list(current_target_regions.items()):
                if not region_seq or len(region_seq) < 7: # Must be at least 7 for seed match
                    print(f"Warning: Defined region '{region_name}' is too short or empty for RRE {rre_id}. Skipping this region.")
                    del current_target_regions[region_name]

            if not current_target_regions:
                continue # Skip this (miRNA, RRE) pair if no valid target regions

            # --- REV Competition Integration Placeholder ---
            # If you want to pair RREs with specific REV sequences (e.g., by year):
            # rev_for_this_rre = next((rev_seq for rev_id, rev_seq in all_rev_data if rev_id.startswith(rre_id.split('_')[0])), None)
            # You'd need a robust way to match Rev sequence IDs to RRE IDs (e.g., 'HXB2_1990_RRE' matches 'HXB2_1990_REV')
            
            current_rev_id = None
            current_rev_seq = None
            # Placeholder for finding the matching REV sequence for the current RRE
            # This logic needs to be tailored to how your RRE and REV IDs relate (e.g., year, strain)
            # Example: If RRE_ID is 'RRE_1990_HXB2' and REV_ID is 'REV_1990_HXB2'
            # You might need to parse IDs or use a direct mapping.
            # For now, let's just pick the first REV data for demonstration if no specific match
            if all_rev_data:
                 current_rev_id = all_rev_data[0][0] # Just take the first one for now
                 current_rev_seq = all_rev_data[0][1]

            # Calculate a placeholder Rev-RRE interaction score (e.g., from external model or literature)
            # rev_rre_interaction_score = calculate_rev_rre_affinity(current_rev_seq, current_rre_seq) # Requires specific function/model

            seed_match_found_for_rre_miRNA_pair = False
            for region_name, region_seq in current_target_regions.items():
                if validate_seed(mirna_seq, region_seq):
                    seed_match_found_for_rre_miRNA_pair = True

                    # --- Affinity check and Labeling ---
                    mirna_affinity_score = affinity.get(mirna_id, None)
                    if mirna_affinity_score is None:
                        skipped_no_affinity_pair_count += 1
                        continue # Skip this specific (miRNA, RRE, region) combination

                    # Label based on affinity
                    label = 1 if mirna_affinity_score > AFFINITY_THRESHOLD else 0

                    # --- If you were comparing with Rev-RRE interaction for label ---
                    # For example, if you want label=1 only if miRNA affinity > Rev-RRE affinity
                    # label = 1 if mirna_affinity_score > rev_rre_interaction_score else 0

                    data.append({
                        'mirna_id': mirna_id,
                        'sequence': mirna_seq,
                        'gc_content': gc,
                        'dg': dg,
                        'conservation': mirna_conservation_score, # Will be 0.0 if not found
                        'affinity': mirna_affinity_score,
                        'structure_vector': struct_vec,
                        'label': label,
                        'rre_id': rre_id,
                        'rre_sequence': current_rre_seq,
                        'region': region_name,
                        'rev_id': current_rev_id, # Add REV ID
                        'rev_sequence': current_rev_seq # Add REV sequence (or relevant part)
                        # 'rev_rre_interaction_score': rev_rre_interaction_score # Add if calculated
                    })
            
            if not seed_match_found_for_rre_miRNA_pair:
                skipped_no_seed_match_pair_count += 1 # This counts skips for (miRNA, RRE) pair if no region matches

        # Progress update
        if (i + 1) % 100 == 0 or (i + 1) == len(all_mirnas_with_ids):
            print(f"Processed {i + 1}/{len(all_mirnas_with_ids)} miRNAs. Current total rows: {len(data)}.")


    df = pd.DataFrame(data)

    # --- Dataset Preparation Summary ---
    print("\n--- Dataset Preparation Summary ---")
    print(f"Total unique miRNAs loaded: {len(all_mirnas_with_ids)}")
    print(f"Total RRE sequences loaded: {len(all_rre_sequences)}")
    print(f"Total REV sequences/data loaded: {len(all_rev_data)}")
    print(f"MiRNAs skipped due to length/GC/RNAfold errors (before RRE loop iteration): {skipped_miRNA_pre_rre_loop_count}")
    print(f"Potential (miRNA, RRE) pairs skipped due to no seed match: {skipped_no_seed_match_pair_count}")
    print(f"Individual (miRNA, RRE, region) entries skipped due to no affinity score: {skipped_no_affinity_pair_count}")
    print(f"Final DataFrame size: {len(df)} rows")
    print("Dataset preparation complete.")

    output_path = os.path.join(PREPARED_DATASET_DIR, "prepared_miRNA_RRE_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"Dataset shape: {df.shape}")

    if not df.empty:
        print("\nFirst 5 rows of the prepared dataset:")
        print(df.head())
    else:
        print("Resulting dataset is empty. Please check input files and filtering criteria.")

    return df

# --- Main Execution ---
if __name__ == "__main__":
    prepared_df = prepare_dataset()