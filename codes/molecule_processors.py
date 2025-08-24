# codes/processors.py
import subprocess
import re
import json

# --- Feature Calculation Functions ---
def calculate_gc_content(sequence):
    """Calculates the GC content of a sequence."""
    if not sequence: return 0.0
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    return gc_count / len(sequence)

def predict_rna_structure(sequence):
    """Calculates 2D structure and dG for an RNA sequence using RNAfold."""
    try:
        result = subprocess.run(['RNAfold'], input=sequence, capture_output=True, text=True, check=True, encoding='utf-8', timeout=30)
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) >= 2:
            struct_line = output_lines[1]
            structure = struct_line.split(' ')[0]
            match = re.search(r"[-+]?\d+\.\d+", struct_line)
            dg = float(match.group(0)) if match else 0.0
            encoded_structure = [({'.': 0, '(': 1, ')': -1}).get(c, 0) for c in structure]
            return {'structure_vector': json.dumps(encoded_structure), 'dg': dg}
        return None
    except Exception:
        return None

# --- Molecule Processors ---
# Each function takes a molecule tuple (id, seq) and a params dictionary.
# It returns a dictionary of processed features or a tuple indicating rejection.
def process_mirna(args):
    """Universal processor for miRNA-type molecules."""
    (mirna_id, mirna_seq), params = args
    
    if not (params['min_mirna_len'] <= len(mirna_seq) <= params['max_mirna_len']):
        return (mirna_id, "reject_length")
    
    gc = calculate_gc_content(mirna_seq)
    if not (params['min_gc_content'] <= gc <= params['max_gc_content']):
        return (mirna_id, "reject_gc")
    
    structural_features = predict_rna_structure(mirna_seq)
    if structural_features is None:
        return (mirna_id, "reject_structure")
    
    return {
        'id': mirna_id,
        'sequence': mirna_seq,
        'gc_content': gc,
        **structural_features
    }

# --- FUTURE EXPANSION: ADD NEW PROCESSORS HERE ---
# ACTION: In the future, a user can uncomment and implement this to add protein support.
# def process_protein(args):
#     """Universal processor for Protein-type molecules."""
#     (protein_id, protein_seq), params = args
#     # Add logic here for protein sequences (e.g., amino acid composition)
#     return {
#         'id': protein_id,
#         'sequence': protein_seq
#     }

# This dictionary is the "plugin manager". It maps names from config.json to the functions above.
PROCESSOR_MAP = {
    "miRNA": process_mirna,
    # "protein": process_protein, # A user would uncomment this line after creating the function
}