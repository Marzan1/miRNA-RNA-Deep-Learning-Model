import os
import pandas as pd
from Bio import SeqIO
import subprocess
import numpy as np
import re # <--- IMPORTANT: This is for robust RNAfold output parsing

# --- Configuration Constants ---
# Base directory where all your subfolders are located
# Make sure this path is correct for your system!
DATASET_ROOT_FOLDER = r"E:\my_deep_learning_project\dataset" 

# Names of your subfolders within DATASET_ROOT_FOLDER
MIRNA_FOLDER_NAME = "Human miRNA dataset"
AFFINITY_FOLDER_NAME = "Affinity_Interaction Score file"
CONSERVATION_FOLDER_NAME = "Conservation Family Information file"
RRE_FOLDER_NAME = "RRE FASTA file"
REV_FOLDER_NAME = "REV dataset" # Folder for REV data
PREPARED_DATASET_FOLDER_NAME = "Prepared Dataset" # Folder to save output

# Full absolute paths to your data folders
MIRNA_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, MIRNA_FOLDER_NAME)
AFFINITY_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, AFFINITY_FOLDER_NAME)
CONSERVATION_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, CONSERVATION_FOLDER_NAME)
RRE_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, RRE_FOLDER_NAME)
REV_DATA_DIR = os.path.join(DATASET_ROOT_FOLDER, REV_FOLDER_NAME)
PREPARED_DATASET_DIR = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME)

# Ensure the output directory exists, create if not
os.makedirs(PREPARED_DATASET_DIR, exist_ok=True)

# Filtering Criteria
SEQUENCE_LENGTH_MIN = 20
# --- MODIFIED: miRNA length filter for positive class (Point 5) ---
SEQUENCE_LENGTH_MAX = 70 
# --- END MODIFICATION ---

GC_CONTENT_MIN = 0.30
GC_CONTENT_MAX = 0.80

# --- MODIFIED: Affinity Threshold (Point 1) ---
AFFINITY_THRESHOLD = 0.6 # For binary labeling (label 1 if score > 0.6, else 0)
# --- END MODIFICATION ---

# --- Helper Functions ---

def calculate_gc_content(sequence):
    """Calculates the GC content of an RNA sequence."""
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0

def predict_structure(sequence):
    """
    Predicts the secondary structure and minimum free energy (DG)
    of an RNA sequence using RNAfold from the ViennaRNA Package.
    """
    try:
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
            
            match = re.search(r'\(\s*([-+]?\d+\.\d+)\)', output_lines[1])
            if match:
                dg = float(match.group(1))
            else:
                print(f"Warning: Could not parse free energy from RNAfold output for sequence {sequence}: '{output_lines[1]}'")
                return None, None 

            return structure_line, dg
        return None, None
    except FileNotFoundError:
        print("Error: 'RNAfold' command not found. Ensure ViennaRNA Package is installed and in your system's PATH.")
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"Error running RNAfold for sequence {sequence}: {e}")
        print(f"Stderr: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred with RNAfold for sequence {sequence}: {e}")
        return None, None

def encode_structure_dot_bracket(structure):
    """
    Converts a dot-bracket structure string into a numerical vector.
    '.' -> 0, '(' -> 1, ')' -> -1. Pads with zeros to max_len.
    """
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
    """
    Checks for a 7-mer seed match (bases 2-8 of miRNA) with the reverse complement
    in the target sequence.
    """
    if len(mirna_seq) < 8:
        return False

    # miRNA seed region (bases 2 to 8, which is 0-indexed: 1 to 7)
    seed_region = mirna_seq[1:8] 

    complement = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
    # Calculate reverse complement of the seed
    rc_seed = "".join([complement[base] for base in seed_region[::-1]])

    if len(target_seq) < len(rc_seed):
        return False

    return rc_seed in target_seq

def get_potential_mirna_family_names(mirna_id):
    """
    Extracsts potential miRNA family names from a miRNA ID, handling common
    naming conventions (e.g., hsa-miR-21, hsa-miR-123-5p).
    """
    parts = mirna_id.split('-')
    if len(parts) >= 3 and parts[0] == 'hsa' and parts[1] == 'miR':
        family_name = f"miR-{parts[2]}".lower()
        # For miRNAs with 3p/5p, consider the stem family name as well
        if len(parts) > 3 and (parts[3] == '3p' or parts[3] == '5p'):
            stem_family_name = f"miR-{parts[2]}".lower()
            return [family_name, stem_family_name]
        return [family_name]
    return []

# --- Data Loading Functions (Modified for automatic file discovery) ---

def _get_files_in_folder(folder_path, extensions):
    """
    Helper function to get all files with specified extensions
    within a given folder.
    """
    files = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}. Skipping.")
        return []
    for f in os.listdir(folder_path):
        if any(f.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(folder_path, f))
    return files

def load_mirnas():
    """Loads miRNA IDs and sequences from all FASTA files in MIRNA_DATA_DIR."""
    mirna_data = []
    seen_mirna_ids = set() # To store unique miRNA IDs and avoid duplicates
    mirna_fasta_files = _get_files_in_folder(MIRNA_DATA_DIR, ['.fasta', '.fa'])

    if not mirna_fasta_files:
        print(f"No miRNA FASTA files found in {MIRNA_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in mirna_fasta_files:
        try:
            current_file_mirnas = 0
            for r in SeqIO.parse(filepath, "fasta"):
                mirna_seq = str(r.seq).replace('T', 'U') # Ensure all sequences are RNA (U instead of T)
                # --- MODIFIED: Apply length filter based on Point 5 ---
                if SEQUENCE_LENGTH_MIN <= len(mirna_seq) <= SEQUENCE_LENGTH_MAX:
                    if r.id not in seen_mirna_ids: # Add only if unique
                        mirna_data.append((r.id, mirna_seq))
                        seen_mirna_ids.add(r.id)
                        current_file_mirnas += 1
                # --- END MODIFICATION ---
            print(f"Loaded {current_file_mirnas} unique miRNAs from {filepath} meeting length criteria.")
        except Exception as e:
            print(f"Error loading miRNA FASTA file {filepath}: {e}. Skipping.")

    print(f"Total unique miRNAs loaded across all files: {len(mirna_data)}")
    return mirna_data

def load_affinity():
    """
    Loads miRNA interaction/affinity scores from all specified files
    in AFFINITY_DATA_DIR. Handles CSV, TSV, TXT.
    Takes the maximum score if a miRNA appears multiple times.
    """
    all_affinity_data = {}
    affinity_files = _get_files_in_folder(AFFINITY_DATA_DIR, ['.txt', '.tsv', '.csv'])

    if not affinity_files:
        print(f"No affinity files found in {AFFINITY_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in affinity_files:
        try:
            sep = '\t' if filepath.lower().endswith(('.txt', '.tsv')) else ','
            df = pd.read_csv(filepath, sep=sep, comment='#', na_values=['NULL'])

            df.columns = [col.lower().strip() for col in df.columns]
            if 'mirna' not in df.columns or 'interaction_score' not in df.columns:
                print(f"Warning: Expected 'mirna' and 'interaction_score' columns in {filepath}. Found: {df.columns.tolist()}. Skipping.")
                continue

            df['mirna'] = df['mirna'].astype(str)
            df['interaction_score'] = pd.to_numeric(df['interaction_score'], errors='coerce')
            df = df.dropna(subset=['interaction_score']) # Drop rows where score could not be parsed

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
    """
    Loads miRNA family conservation percentages from all specified files
    in CONSERVATION_DATA_DIR. Handles CSV, TSV, TXT.
    Takes the maximum percentage if a family appears multiple times.
    """
    all_conservation_data = {}
    conservation_files = _get_files_in_folder(CONSERVATION_DATA_DIR, ['.txt', '.tsv', '.csv'])

    if not conservation_files:
        print(f"No conservation files found in {CONSERVATION_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in conservation_files:
        try:
            sep = '\t' if filepath.lower().endswith(('.txt', '.tsv')) else ','
            df = pd.read_csv(filepath, sep=sep, na_values=['NULL'], comment='#')

            df.columns = [col.lower().strip() for col in df.columns]
            df.rename(columns={'mir family': 'mir_family'}, inplace=True)

            if 'mir_family' not in df.columns or 'pct' not in df.columns:
                print(f"Warning: Expected 'mir_family' and 'pct' columns in {filepath}. Found: {df.columns.tolist()}. Skipping.")
                continue

            df['mir_family'] = df['mir_family'].astype(str)
            df['pct'] = pd.to_numeric(df['pct'], errors='coerce')
            df = df.dropna(subset=['pct']) # Drop rows where percentage could not be parsed

            for _, row in df.iterrows():
                family_name = str(row['mir_family']).lower() # Ensure family name is lowercase for matching
                pct = float(row['pct'])
                all_conservation_data[family_name] = max(all_conservation_data.get(family_name, 0.0), pct)

            print(f"Loaded {len(df)} conservation entries from {filepath}.")

        except Exception as e:
            print(f"Error loading conservation file {filepath}: {e}. Skipping.")

    print(f"Total unique conservation families loaded: {len(all_conservation_data)}")
    return all_conservation_data

def load_rre_sequences():
    """Loads RRE IDs and sequences from all FASTA files in RRE_DATA_DIR."""
    rre_data = []
    rre_fasta_files = _get_files_in_folder(RRE_DATA_DIR, ['.fasta', '.fa'])

    if not rre_fasta_files:
        print(f"No RRE FASTA files found in {RRE_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in rre_fasta_files:
        try:
            for r in SeqIO.parse(filepath, "fasta"):
                rre_data.append((r.id, str(r.seq).replace('T', 'U'))) # Convert to RNA
            print(f"Loaded {len(rre_data)} RRE sequences from {filepath}.")
        except Exception as e:
            print(f"Error loading RRE FASTA file {filepath}: {e}")

    print(f"Total RRE sequences loaded across all files: {len(rre_data)}")
    return rre_data

def load_rev_data():
    """
    Loads REV sequences or related data from all files in REV_DATA_DIR.
    This function assumes FASTA files for protein/RNA sequences, or CSV/TSV
    with 'ID' and 'Sequence' columns. Adjust 'extensions' and parsing logic
    if your REV data is in a different format.
    """
    rev_data = []
    rev_files = _get_files_in_folder(REV_DATA_DIR, ['.fasta', '.fa', '.txt', '.csv', '.tsv'])

    if not rev_files:
        print(f"No REV files found in {REV_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in rev_files:
        try:
            if filepath.lower().endswith(('.fasta', '.fa')):
                for r in SeqIO.parse(filepath, "fasta"):
                    rev_data.append((r.id, str(r.seq))) 
            elif filepath.lower().endswith(('.txt', '.csv', '.tsv')):
                sep = '\t' if filepath.endswith(('.txt', '.tsv')) else ','
                df = pd.read_csv(filepath, sep=sep, comment='#')
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

    all_mirnas_with_ids = load_mirnas()
    affinity = load_affinity()
    conservation_by_family = load_conservation()
    all_rre_sequences = load_rre_sequences()
    all_rev_data = load_rev_data()

    if not all_mirnas_with_ids:
        print("No miRNAs loaded. Aborting dataset preparation.")
        return pd.DataFrame()
    if not all_rre_sequences:
        print("No RRE sequences loaded. Aborting dataset preparation.")
        return pd.DataFrame()

    mirnas_with_affinity_ids_set = set(affinity.keys()) 
    
    filtered_mirnas_for_processing = []
    for mirna_id, mirna_seq in all_mirnas_with_ids:
        if mirna_id in mirnas_with_affinity_ids_set:
            filtered_mirnas_for_processing.append((mirna_id, mirna_seq))

    print(f"Reduced processing pool to {len(filtered_mirnas_for_processing)} miRNAs with affinity data (from {len(all_mirnas_with_ids)} total).")

    data = []
    
    processed_miRNA_count = 0
    skipped_miRNA_pre_rre_loop_count = 0
    
    skipped_no_seed_match_pair_count = 0
    # No longer skipping if no affinity, will treat as 0
    # skipped_no_affinity_pair_count = 0 

    for i, (mirna_id, mirna_seq) in enumerate(filtered_mirnas_for_processing):
        processed_miRNA_count += 1
        
        gc = calculate_gc_content(mirna_seq)
        if not (GC_CONTENT_MIN <= gc <= GC_CONTENT_MAX):
            skipped_miRNA_pre_rre_loop_count += 1
            continue

        structure, dg = predict_structure(mirna_seq)
        if structure is None:
            skipped_miRNA_pre_rre_loop_count += 1
            continue
        struct_vec = encode_structure_dot_bracket(structure)

        mirna_conservation_score = 0.0
        family_names = get_potential_mirna_family_names(mirna_id)
        for family in family_names:
            if family in conservation_by_family:
                mirna_conservation_score = conservation_by_family[family]
                break

        for rre_idx, (rre_id, current_rre_seq) in enumerate(all_rre_sequences):
            
            # --- MODIFIED: RRE target regions (Point 3) ---
            # Define specific RRE target regions for seed matching
            current_target_regions = {}
            # Stem IIB (40 to 80 residues) -> Python slice [39:80]
            if len(current_rre_seq) >= 80: # Ensure sequence is long enough
                current_target_regions["Stem_IIB"] = current_rre_seq[39:80]
            else:
                # print(f"Warning: RRE {rre_id} too short for Stem_IIB region ({len(current_rre_seq)}bp).")
                pass # Can be ignored if common for shorter RREs
            
            # Stem IA (20-35 and 100-125) -> Python slices [19:35] and [99:125]
            if len(current_rre_seq) >= 35:
                current_target_regions["Stem_IA_Part1"] = current_rre_seq[19:35]
            if len(current_rre_seq) >= 125:
                current_target_regions["Stem_IA_Part2"] = current_rre_seq[99:125]
            # --- END MODIFICATION ---

            # Filter out any regions that might be too short after slicing
            valid_target_regions = {name: seq for name, seq in current_target_regions.items() if seq and len(seq) >= 7}

            if not valid_target_regions:
                # print(f"Skipping RRE {rre_id} as no valid target regions could be extracted.")
                continue
            
            current_rev_id = None
            current_rev_seq = None
            
            matched_rev = None
            for rev_id_val, rev_seq_val in all_rev_data:
                rre_parts = rre_id.split('_')
                rev_parts = rev_id_val.split('_')
                
                if len(rre_parts) >= 2 and len(rev_parts) >= 2 and \
                   rre_parts[0] == rev_parts[0] and rre_parts[1] == rev_parts[1]:
                    matched_rev = (rev_id_val, rev_seq_val)
                    break
            
            if matched_rev:
                current_rev_id, current_rev_seq = matched_rev
            elif all_rev_data:
                current_rev_id, current_rev_seq = all_rev_data[0] 
            
            seed_match_found_for_rre_miRNA_pair = False
            
            # --- MODIFIED: Handle unknown affinity (Point 2) ---
            mirna_affinity_score = affinity.get(mirna_id) 
            if mirna_affinity_score is None: 
                mirna_affinity_score = 0.0 # Treat as 0 if unidentified
            # --- END MODIFICATION ---

            # --- ADDED: rev_interaction_type based on Point 4' ---
            rev_interaction_type = "not_applicable"
            if current_rev_id and current_rev_seq: # Only evaluate if REV data exists
                # Default to negative
                rev_interaction_type = "negative"

                # Check for "super positive" - strong affinity, binds Stem IIB, Rev relevant
                if mirna_affinity_score > AFFINITY_THRESHOLD and "Stem_IIB" in valid_target_regions and \
                   validate_seed(mirna_seq, valid_target_regions["Stem_IIB"]):
                    rev_interaction_type = "super_positive"
                # Check for "medium" - binds Stem IA (secondary site), or moderate affinity
                elif ("Stem_IA_Part1" in valid_target_regions and validate_seed(mirna_seq, valid_target_regions["Stem_IA_Part1"])) or \
                     ("Stem_IA_Part2" in valid_target_regions and validate_seed(mirna_seq, valid_target_regions["Stem_IA_Part2"])) or \
                     (0 < mirna_affinity_score <= AFFINITY_THRESHOLD): # Moderate affinity not strictly negative
                    if rev_interaction_type != "super_positive": # Don't downgrade from super_positive
                        rev_interaction_type = "medium"

            # Iterate through only valid (non-empty, sufficient length) regions
            for region_name, region_seq in valid_target_regions.items():
                if validate_seed(mirna_seq, region_seq):
                    seed_match_found_for_rre_miRNA_pair = True

                    # --- MODIFIED: Label assignment based on new threshold (Point 1) ---
                    label = 1 if mirna_affinity_score > AFFINITY_THRESHOLD else 0
                    # --- END MODIFICATION ---

                    data.append({
                        'mirna_id': mirna_id,
                        'sequence': mirna_seq,
                        'gc_content': gc,
                        'dg': dg,
                        'conservation': mirna_conservation_score,
                        'affinity': mirna_affinity_score,
                        'structure_vector': struct_vec,
                        'label': label,
                        'rre_id': rre_id,
                        'rre_sequence': current_rre_seq,
                        'region': region_name,
                        'rev_id': current_rev_id,
                        'rev_sequence': current_rev_seq,
                        'rev_interaction_type': rev_interaction_type # ADDED new feature
                    })
            
            # --- IMPORTANT: The original logic skips if no seed match for the pair.
            # This implicitly treats non-seed-matching pairs as 'negative' by not including them.
            # If you want to explicitly include (miRNA, RRE) pairs with no seed match
            # but with an affinity of 0 (or below threshold) as labeled '0' data points,
            # you would need to modify this part to *always* add a row, even without a seed match.
            # For now, sticking to current structure which implicitly filters.
            if not seed_match_found_for_rre_miRNA_pair:
                skipped_no_seed_match_pair_count += 1
            # --- END IMPORTANT ---


        if (i + 1) % 100 == 0 or (i + 1) == len(filtered_mirnas_for_processing):
            print(f"Processed {i + 1}/{len(filtered_mirnas_for_processing)} relevant miRNAs. Current total rows: {len(data)}.")

    df = pd.DataFrame(data)

    print("\n--- Dataset Preparation Summary ---")
    print(f"Total unique miRNAs loaded: {len(all_mirnas_with_ids)}")
    print(f"Total unique miRNAs processed after affinity filter: {len(filtered_mirnas_for_processing)}") 
    print(f"Total RRE sequences loaded: {len(all_rre_sequences)}")
    print(f"Total REV sequences/data loaded: {len(all_rev_data)}")
    print(f"MiRNAs skipped due to length/GC/RNAfold errors (before RRE loop iteration): {skipped_miRNA_pre_rre_loop_count}")
    print(f"Potential (miRNA, RRE) pairs skipped due to no seed match (implies unwanted binding): {skipped_no_seed_match_pair_count}")
    # Removed skipped_no_affinity_pair_count as unidentified affinity is now handled as 0
    print(f"Final DataFrame size: {len(df)} rows")
    
    print("\n--- Class Distribution in Prepared Dataset ---")
    print(df['label'].value_counts())
    
    # --- ADDED: Distribution of new 'rev_interaction_type' feature ---
    if 'rev_interaction_type' in df.columns:
        print("\n--- Distribution of 'rev_interaction_type' in Prepared Dataset ---")
        print(df['rev_interaction_type'].value_counts())
    # --- END ADDED ---

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

