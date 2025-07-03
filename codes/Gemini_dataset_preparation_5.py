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
SEQUENCE_LENGTH_MAX = 80
GC_CONTENT_MIN = 0.30
GC_CONTENT_MAX = 0.80
AFFINITY_THRESHOLD = 0.7 # For binary labeling (label 1 if score > 0.7, else 0)

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
        # Use subprocess.run with text=True and pass sequence directly (as str)
        # subprocess handles encoding/decoding automatically when text=True
        result = subprocess.run(
            ['RNAfold'],
            input=sequence, # <--- FIX: Pass string directly
            capture_output=True,
            text=True, # Ensure stdin/stdout are handled as text
            check=True # Raise CalledProcessError if RNAfold returns non-zero exit code
        )
        
        output_lines = result.stdout.strip().split('\n')
        
        if len(output_lines) >= 2:
            # The first part of the second line is the dot-bracket structure
            structure_line = output_lines[1].split(' ')[0]
            
            # --- FIX: Use regex to robustly extract DG value ---
            # This regex looks for ( optionally followed by spaces, then the number, then )
            match = re.search(r'\(\s*([-+]?\d+\.\d+)\)', output_lines[1])
            if match:
                dg = float(match.group(1)) # Extract the captured number
            else:
                # If regex fails to find the DG, print a warning and return None
                print(f"Warning: Could not parse free energy from RNAfold output for sequence {sequence}: '{output_lines[1]}'")
                return None, None 
            # --- END FIX ---

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
        # Catch any other unexpected errors during RNAfold processing
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
    Extracts potential miRNA family names from a miRNA ID, handling common
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
            # Use SeqIO.parse for efficient FASTA parsing
            for r in SeqIO.parse(filepath, "fasta"):
                mirna_seq = str(r.seq).replace('T', 'U') # Ensure all sequences are RNA (U instead of T)
                # Apply length filter
                if SEQUENCE_LENGTH_MIN <= len(mirna_seq) <= SEQUENCE_LENGTH_MAX:
                    if r.id not in seen_mirna_ids: # Add only if unique
                        mirna_data.append((r.id, mirna_seq))
                        seen_mirna_ids.add(r.id)
                        current_file_mirnas += 1
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
            # Read with pandas, ignoring comment lines starting with #, handling NULL values
            df = pd.read_csv(filepath, sep=sep, comment='#', na_values=['NULL'])

            # Standardize column names to lowercase for easier access
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
                # If a miRNA has multiple scores, take the max (strongest interaction)
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
            # Rename 'mir family' to 'mir_family' for consistency
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
                # If a family has multiple percentages, take the max
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
    # Add relevant extensions for your REV data files
    rev_files = _get_files_in_folder(REV_DATA_DIR, ['.fasta', '.fa', '.txt', '.csv', '.tsv'])

    if not rev_files:
        print(f"No REV files found in {REV_DATA_DIR}. Please check the folder and file extensions.")

    for filepath in rev_files:
        try:
            if filepath.lower().endswith(('.fasta', '.fa')):
                for r in SeqIO.parse(filepath, "fasta"):
                    # Assuming REV sequence here could be protein or RNA.
                    # Keep as is, or add .replace('T', 'U') if always RNA.
                    rev_data.append((r.id, str(r.seq))) 
            elif filepath.lower().endswith(('.txt', '.csv', '.tsv')):
                sep = '\t' if filepath.endswith(('.txt', '.tsv')) else ','
                df = pd.read_csv(filepath, sep=sep, comment='#')
                df.columns = [col.lower().strip() for col in df.columns]
                # Adjust 'id' and 'sequence' column names if yours are different
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
    """
    Orchestrates the loading, processing, and feature generation
    to create the final dataset.
    """
    print("Preparing dataset...")

    # Load all necessary data using the new folder-based loading functions
    all_mirnas_with_ids = load_mirnas()
    affinity = load_affinity()
    conservation_by_family = load_conservation()
    all_rre_sequences = load_rre_sequences()
    all_rev_data = load_rev_data() # Load REV data

    # Basic checks to ensure data was loaded
    if not all_mirnas_with_ids:
        print("No miRNAs loaded. Aborting dataset preparation.")
        return pd.DataFrame()
    if not all_rre_sequences:
        print("No RRE sequences loaded. Aborting dataset preparation.")
        return pd.DataFrame()

    data = [] # List to store dictionaries for DataFrame rows
    
    # Counters for logging and summary
    processed_miRNA_count = 0
    skipped_miRNA_pre_rre_loop_count = 0 # MiRNAs skipped due to GC or RNAfold BEFORE RRE loop
    skipped_no_seed_match_pair_count = 0 # (miRNA, RRE) pairs skipped if no seed match in *any* defined region
    skipped_no_affinity_pair_count = 0 # Specific (miRNA, RRE, region) entries skipped due to missing affinity

    # Outer loop: Iterate through each miRNA
    for i, (mirna_id, mirna_seq) in enumerate(all_mirnas_with_ids):
        processed_miRNA_count += 1
        
        # --- miRNA-specific calculations and filters (before RRE loop) ---
        # 1. GC Content Filter
        gc = calculate_gc_content(mirna_seq)
        if not (GC_CONTENT_MIN <= gc <= GC_CONTENT_MAX):
            skipped_miRNA_pre_rre_loop_count += 1
            continue # Skip this miRNA entirely if GC content is out of range

        # 2. RNA Secondary Structure Prediction
        structure, dg = predict_structure(mirna_seq)
        if structure is None:
            skipped_miRNA_pre_rre_loop_count += 1
            continue # Skip this miRNA entirely if RNAfold failed or parsing failed
        struct_vec = encode_structure_dot_bracket(structure)

        # 3. miRNA Conservation Score
        mirna_conservation_score = 0.0 # Default to 0 if no match found
        family_names = get_potential_mirna_family_names(mirna_id)
        for family in family_names:
            if family in conservation_by_family:
                mirna_conservation_score = conservation_by_family[family]
                break # Take the first matching family's score

        # --- Loop through each RRE sequence ---
        for rre_idx, (rre_id, current_rre_seq) in enumerate(all_rre_sequences):
            
            # --- Define TARGET_REGIONS for the CURRENT RRE sequence ---
            # For simplicity, initially taking the first 150 bases of RRE.
            # You can make this more sophisticated by defining specific loop regions
            # (e.g., Loop I, Stem IIB) if your RRE sequences have known, consistent
            # structures that map to specific indices.
            current_target_regions = {
                "Primary_RRE_Binding_Region": current_rre_seq[0:min(len(current_rre_seq), 150)] 
            }
            
            # Validate defined regions: ensure they are not empty and long enough for a seed match
            for region_name, region_seq in list(current_target_regions.items()):
                if not region_seq or len(region_seq) < 7: # Seed match requires at least 7 bases
                    print(f"Warning: Defined region '{region_name}' is too short or empty for RRE {rre_id}. Skipping this region.")
                    del current_target_regions[region_name]

            if not current_target_regions:
                continue # Skip this (miRNA, RRE) pair if no valid target regions could be defined

            # --- REV Competition Integration Placeholder ---
            # This section needs careful consideration based on how your REV data
            # relates to RRE and how you want to model competition.
            
            current_rev_id = None
            current_rev_seq = None
            
            # This is a very basic attempt to link RRE and REV data.
            # It assumes your RRE_ID and REV_ID might share common parts (e.g., strain and year).
            # Example: If RRE_ID is 'RRE_1990_HXB2' and REV_ID is 'REV_1990_HXB2'
            matched_rev = None
            for rev_id_val, rev_seq_val in all_rev_data:
                # You'll likely need to customize this matching logic based on your actual IDs
                rre_parts = rre_id.split('_')
                rev_parts = rev_id_val.split('_')
                
                # Example: If both IDs contain a strain and year at specific positions
                if len(rre_parts) >= 2 and len(rev_parts) >= 2 and \
                   rre_parts[0] == rev_parts[0] and rre_parts[1] == rev_parts[1]:
                   matched_rev = (rev_id_val, rev_seq_val)
                   break
                # Fallback: if no specific matching logic fits, or for a simpler pairing
                # You might just assign the first REV entry to all RREs if the Rev data is generic
                # or not meant to be RRE-specific.
            
            if matched_rev:
                current_rev_id, current_rev_seq = matched_rev
            elif all_rev_data: 
                # If no specific match, assign the first REV entry found (as a default/fallback)
                current_rev_id, current_rev_seq = all_rev_data[0]
            
            # --- Place where Rev-RRE binding affinity/complementarity would be estimated ---
            # This is where you would introduce a score for the Rev-RRE interaction.
            # This would typically come from:
            # 1. A separate computational model for protein-RNA binding.
            # 2. Experimental data (literature values for K_d or similar).
            # 3. If "rev_sequence" refers to an RNA motif that Rev binds, you might use
            #    sequence complementarity.
            # For now, it's just a placeholder; 'rev_rre_interaction_score' is not calculated.
            # rev_rre_interaction_score = some_function_to_estimate_rev_rre_binding(current_rev_seq, current_rre_seq)

            seed_match_found_for_rre_miRNA_pair = False
            for region_name, region_seq in current_target_regions.items():
                if validate_seed(mirna_seq, region_seq):
                    seed_match_found_for_rre_miRNA_pair = True

                    # --- Affinity check and Labeling ---
                    mirna_affinity_score = affinity.get(mirna_id, None)
                    if mirna_affinity_score is None:
                        skipped_no_affinity_pair_count += 1
                        continue # Skip this specific (miRNA, RRE, region) combination if no affinity score

                    # Label based on miRNA affinity threshold
                    label = 1 if mirna_affinity_score > AFFINITY_THRESHOLD else 0

                    # --- Potential for Rev-mediated labeling (Advanced) ---
                    # If you had `rev_rre_interaction_score`, you could make the label
                    # more complex, e.g., label=1 if miRNA affinity > Rev-RRE affinity,
                    # reflecting competition.
                    # Example: label = 1 if mirna_affinity_score > rev_rre_interaction_score else 0

                    # Append all collected features for this data point
                    data.append({
                        'mirna_id': mirna_id,
                        'sequence': mirna_seq,
                        'gc_content': gc,
                        'dg': dg, # Minimum Free Energy
                        'conservation': mirna_conservation_score,
                        'affinity': mirna_affinity_score,
                        'structure_vector': struct_vec, # Numerical representation of miRNA structure
                        'label': label, # Binary label for interaction
                        'rre_id': rre_id,
                        'rre_sequence': current_rre_seq,
                        'region': region_name, # The specific RRE region matched
                        'rev_id': current_rev_id, # Included REV ID
                        'rev_sequence': current_rev_seq # Included REV sequence (or relevant part)
                        # 'rev_rre_interaction_score': rev_rre_interaction_score # Add if you implement this calculation
                    })
            
            # If no seed match was found in any target region for this (miRNA, RRE) pair
            if not seed_match_found_for_rre_miRNA_pair:
                skipped_no_seed_match_pair_count += 1 

        # Progress update for long-running operations
        if (i + 1) % 100 == 0 or (i + 1) == len(all_mirnas_with_ids):
            print(f"Processed {i + 1}/{len(all_mirnas_with_ids)} miRNAs. Current total rows: {len(data)}.")

    # Create final DataFrame
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

    # Save the prepared dataset
    output_path = os.path.join(PREPARED_DATASET_DIR, "prepared_miRNA_RRE_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"Dataset shape: {df.shape}")

    # Display a preview of the dataset
    if not df.empty:
        print("\nFirst 5 rows of the prepared dataset:")
        print(df.head())
    else:
        print("Resulting dataset is empty. Please check input files and filtering criteria.")

    return df

# --- Main Execution Block ---
# This ensures that prepare_dataset() is called only when the script is run directly
if __name__ == "__main__":
    prepared_df = prepare_dataset()