import os
import pandas as pd
from Bio import SeqIO
import subprocess
import numpy as np

# --- Configuration Constants (rest of your constants remain unchanged) ---
# Base directory where all your subfolders are located
DATASET_ROOT_FOLDER = r"E:\my_deep_learning_project\dataset" 

# Subfolder names within DATASET_ROOT_FOLDER
MIRNA_FOLDER_NAME = "Human miRNA dataset"
AFFINITY_FOLDER_NAME = "Affinity_Interaction Score file"
CONSERVATION_FOLDER_NAME = "Conservation Family Information file"
RRE_FOLDER_NAME = "RRE FASTA file"
REV_FOLDER_NAME = "REV dataset"
PREPARED_DATASET_FOLDER_NAME = "Prepared Dataset"

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
SEQUENCE_LENGTH_MAX = 21
GC_CONTENT_MIN = 0.40
GC_CONTENT_MAX = 0.70
AFFINITY_THRESHOLD = 0.7 

# --- Helper Functions ---

def calculate_gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0

def predict_structure(sequence):
    try:
        # --- FIX IS HERE: Remove .encode('utf-8') ---
        result = subprocess.run(
            ['RNAfold'],
            input=sequence, # 'sequence' is already a string, and text=True will handle encoding
            capture_output=True,
            text=True, # This tells subprocess to handle text decoding/encoding
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

# --- Data Loading Functions (Unchanged from previous code) ---

def _get_files_in_folder(folder_path, extensions):
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
                mirna_seq = str(r.seq).replace('T', 'U') # Ensure U for RNA
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
    rev_data = []
    rev_files = _get_files_in_folder(REV_DATA_DIR, ['.fasta', '.fa', '.txt', '.csv', '.tsv'])

    if not rev_files:
        print(f"No REV files found in {REV_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in rev_files:
        try:
            if filepath.lower().endswith(('.fasta', '.fa')):
                for r in SeqIO.parse(filepath, "fasta"):
                    rev_data.append((r.id, str(r.seq))) # Assuming REV sequences are strings, could be protein or RNA
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

    data = []
    
    processed_miRNA_count = 0
    skipped_miRNA_pre_rre_loop_count = 0
    
    skipped_no_seed_match_pair_count = 0
    skipped_no_affinity_pair_count = 0

    # Outer loop: Iterate through each miRNA
    for i, (mirna_id, mirna_seq) in enumerate(all_mirnas_with_ids):
        processed_miRNA_count += 1

        # Add a progress indicator
        if (i + 1) % 1000 == 0: # Print every 1000 miRNAs
            print(f"Processing miRNA {i + 1}/{len(all_mirnas_with_ids)}: {mirna_id}")
        
        # miRNA-specific calculations and filters (before RRE loop)
        gc = calculate_gc_content(mirna_seq)
        if not (GC_CONTENT_MIN <= gc <= GC_CONTENT_MAX):
            skipped_miRNA_pre_rre_loop_count += 1
            continue

        structure, dg = predict_structure(mirna_seq)
        if structure is None: # This should now work correctly
            skipped_miRNA_pre_rre_loop_count += 1
            continue
        struct_vec = encode_structure_dot_bracket(structure)

        mirna_conservation_score = 0.0
        family_names = get_potential_mirna_family_names(mirna_id)
        for family in family_names:
            if family in conservation_by_family:
                mirna_conservation_score = conservation_by_family[family]
                break

        # Loop through each RRE sequence
        for rre_idx, (rre_id, current_rre_seq) in enumerate(all_rre_sequences):
            
            current_target_regions = {
                "Primary_RRE_Binding_Region": current_rre_seq[0:min(len(current_rre_seq), 150)]
            }
            
            for region_name, region_seq in list(current_target_regions.items()):
                if not region_seq or len(region_seq) < 7:
                    # print(f"Warning: Defined region '{region_name}' is too short or empty for RRE {rre_id}. Skipping this region.")
                    del current_target_regions[region_name]

            if not current_target_regions:
                continue

            # --- REV Competition Integration Placeholder ---
            # This is where you would integrate REV data to calculate a Rev-RRE interaction score
            # or identify overlapping binding sites.
            
            current_rev_id = None
            current_rev_seq = None
            
            # This is a very basic way to try to match RRE and REV data (e.g., if their IDs share a common prefix like year/strain)
            # You would need to refine this logic based on your specific ID naming convention for RRE and REV sequences.
            # Example: RRE_ID "RRE_1990_HXB2", REV_ID "REV_1990_HXB2"
            matched_rev = None
            for rev_id_val, rev_seq_val in all_rev_data:
                # Basic string matching - adapt to your specific ID format
                if rre_id.split('_')[0] == rev_id_val.split('_')[0] and \
                   rre_id.split('_')[1] == rev_id_val.split('_')[1]: # Assuming "STRAIN_YEAR_..."
                   matched_rev = (rev_id_val, rev_seq_val)
                   break
            
            if matched_rev:
                current_rev_id, current_rev_seq = matched_rev
            elif all_rev_data: # If no specific match, just use the first one available as a fallback
                current_rev_id, current_rev_seq = all_rev_data[0]
            
            # --- Place where Rev-RRE binding affinity/complementarity would be estimated ---
            # This would likely involve:
            # 1. A function that takes a `current_rev_seq` (protein) and `current_rre_seq` (RNA)
            #    and predicts their binding strength (e.g., using a separate model, or known motifs).
            # 2. Or, if `current_rev_seq` is actually an RNA motif that Rev recognizes, then you might use
            #    `validate_seed` or a similar complementarity function between `mirna_seq` and this `rev_motif_seq`.
            # For now, it's just included as a column.
            
            # rev_rre_interaction_score = some_function_to_estimate_rev_rre_binding(current_rev_seq, current_rre_seq)
            # You would then potentially use this score to refine the 'label' or add it as a feature.

            seed_match_found_for_rre_miRNA_pair = False
            for region_name, region_seq in current_target_regions.items():
                if validate_seed(mirna_seq, region_seq):
                    seed_match_found_for_rre_miRNA_pair = True

                    mirna_affinity_score = affinity.get(mirna_id, None)
                    if mirna_affinity_score is None:
                        skipped_no_affinity_pair_count += 1
                        continue

                    label = 1 if mirna_affinity_score > AFFINITY_THRESHOLD else 0

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
                        'rev_id': current_rev_id,        # Now included
                        'rev_sequence': current_rev_seq  # Now included
                        # 'rev_rre_interaction_score': rev_rre_interaction_score # Add if calculated above
                    })
            
            if not seed_match_found_for_rre_miRNA_pair:
                skipped_no_seed_match_pair_count += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(all_mirnas_with_ids):
            print(f"Processed {i + 1}/{len(all_mirnas_with_ids)} miRNAs. Current total rows: {len(data)}.")

    df = pd.DataFrame(data)

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