# codes/processors.py (With Development Switch for DSSR)
import os
import re
import json
import numpy as np
import subprocess

# --- Configuration Loader ---
def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir) 
        config_path = os.path.join(project_root, 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# --- Feature Calculation Functions ---
def calculate_gc_content(sequence):
    """Calculates the GC content of a sequence."""
    if not sequence: return 0.0
    return (sequence.upper().count('G') + sequence.upper().count('C')) / len(sequence)

def predict_rna_structure_1d(sequence):
    """Calculates 1D structure vector and dG for an RNA sequence using RNAfold."""
    try:
        result = subprocess.run(['RNAfold'], input=sequence, text=True, capture_output=True, check=True, encoding='utf-8', timeout=30)
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) >= 2:
            struct_line = output_lines[1]
            structure = struct_line.split(' ')[0]
            match = re.search(r"[-+]?\d+\.\d+", struct_line)
            dg = float(match.group(0)) if match else 0.0
            encoded_structure = [({'.': 0, '(': 1, ')': -1}).get(c, 0) for c in structure]
            return {'structure_vector': json.dumps(encoded_structure), 'dg': dg}
    except Exception:
        pass
    return None

def _parse_dot_bracket_to_adjacency(dbn_structure):
    """Converts a dot-bracket string to a binary adjacency matrix for GNNs."""
    seq_len = len(dbn_structure)
    adjacency_matrix = np.zeros((seq_len, seq_len), dtype=int)
    stack = []
    for i, char in enumerate(dbn_structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    # Add connections for the sequence backbone
    for i in range(seq_len - 1):
        adjacency_matrix[i, i + 1] = 1
        adjacency_matrix[i + 1, i] = 1
    return adjacency_matrix

def predict_graph_structure(molecule_id, sequence):
    """
    Generates a graph (adjacency matrix) for a molecule.
    1. If enabled, checks config for a PDB file and processes it with DSSR.
    2. Falls back to RNAfold prediction if PDB processing is disabled or fails.
    """
    config = load_config()
    
    # <<< CHANGE: Added a check for the new "development switch" >>>
    # Only try to process PDB files if the flag is set to true in config.json
    use_pdb = config.get('processing_parameters', {}).get('enable_pdb_processing', False)

    if use_pdb and 'structure_files' in config and molecule_id in config['structure_files']:
        pdb_path = config['structure_files'][molecule_id]
        if os.path.exists(pdb_path):
            try:
                # This block will only run if DSSR is installed AND the switch is on.
                result = subprocess.run(['x3dna-dssr', f'--input={pdb_path}'],
                                        capture_output=True, text=True, check=True, timeout=60)
                match = re.search(r'secondary structure in dot-bracket notation\s*\n\s*(\S+)', result.stdout)
                if match:
                    dot_bracket_string = match.group(1)
                    return _parse_dot_bracket_to_adjacency(dot_bracket_string)
            except Exception as e:
                print(f"  - WARNING: DSSR failed for {molecule_id}. Falling back to RNAfold. Error: {e}")
                pass 

    # Fallback to RNAfold prediction (this will be the default for the next 2 days)
    try:
        result = subprocess.run(['RNAfold'], input=sequence, text=True, capture_output=True, check=True, encoding='utf-8', timeout=30)
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) >= 2:
            structure = output_lines[1].split(' ')[0]
            return _parse_dot_bracket_to_adjacency(structure)
    except Exception:
        pass
        
    return None

# --- Universal Molecule Processor ---
def process_rna_universal(args):
    """
    Universal processor for any RNA-like molecule.
    Calculates GC, 1D structure, and 2D graph structure (for GNN).
    """
    (molecule_id, sequence), params = args
    # --- Feature Generation ---
    gc = calculate_gc_content(sequence)
    structural_features_1d = predict_rna_structure_1d(sequence)
    
    if structural_features_1d is None:
        return (molecule_id, "reject_structure_1d")

    # --- GNN Feature Generation ---
    adjacency_matrix = predict_graph_structure(molecule_id, sequence)
    
    if adjacency_matrix is None:
        seq_len = len(sequence)
        adjacency_matrix = np.zeros((seq_len, seq_len), dtype=int)
        
    serialized_adjacency = json.dumps(adjacency_matrix.tolist())

    return {
        'id': molecule_id,
        'sequence': sequence,
        'gc_content': gc,
        **structural_features_1d,
        'adjacency_matrix': serialized_adjacency
    }

PROCESSOR_MAP = {
    "miRNA": process_rna_universal,
    "RNA": process_rna_universal,
}