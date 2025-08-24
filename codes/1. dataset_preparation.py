# 1_dataset_preparation.py (Definitive, Final Version)
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

# --- Import the new processors library ---
from molecule_processors import PROCESSOR_MAP

# --- Configuration Loader ---
def load_config(config_path=None):
    """
    Loads the configuration from a JSON file.
    If no path is given, it automatically finds 'config.json' in the project root.
    """
    if config_path is None:
        # Build the path dynamically relative to this script's location
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # Assumes the script is in a /codes folder and config.json is one level up
        project_root = os.path.dirname(script_dir) 
        config_path = os.path.join(project_root, 'config.json')
    
    print(f"--- Loading configuration from: {config_path} ---")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'.")
        print("Please ensure 'config.json' is in the project's root directory.")
        exit()

# --- Helper Functions ---
def _get_files_in_folder(folder_path, extensions):
    """Gets all files with specified extensions from a folder."""
    if not os.path.exists(folder_path):
        return []
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in extensions)]

def load_data_from_fasta(folder_path):
    """Loads all sequences from all FASTA files in a directory."""
    data_dict = {}
    file_paths = _get_files_in_folder(folder_path, ['.fasta', '.fa', '.txt'])
    if not file_paths: return data_dict
    for filepath in file_paths:
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                data_dict[record.id] = str(record.seq).replace('T', 'U')
        except Exception: pass
    return data_dict

def load_scores(folder_path, id_col, score_col, file_type_name):
    """Loads and inspects score files from a directory."""
    data_dict, all_scores = {}, []
    print(f"  Scanning {file_type_name} files in '{folder_path}'...")
    for filepath in _get_files_in_folder(folder_path, ['.txt', '.tsv', '.csv']):
        try:
            sep = '\t' if filepath.lower().endswith(('.txt', '.tsv')) else ','
            df = pd.read_csv(
                filepath, sep=sep, comment='#',
                usecols=lambda column: column.lower().strip() in [id_col, score_col],
                dtype={id_col: str, score_col: float}, na_values=['NULL']
            )
            df.columns = [col.lower().strip() for col in df.columns]
            df.dropna(inplace=True)
            for _, row in df.iterrows():
                key, score = str(row[id_col]), float(row[score_col])
                all_scores.append(score)
                data_dict[key] = max(data_dict.get(key, 0.0), score)
        except Exception as e:
            print(f"    - Error loading score file {filepath}: {e}")
    if all_scores:
        scores_series = pd.Series(all_scores)
        print(f"    - Statistical Summary for {file_type_name} Scores:")
        summary = scores_series.describe().to_string().replace('\n', '\n      ')
        print(f"      {summary}")
    return data_dict

# --- Main Dataset Preparation Function ---
def prepare_dataset(config):
    start_time = time.time()
    
    # --- 1. Setup Paths and Parameters from Config ---
    PROJECT_ROOT = config['project_root']
    DATA_ROOT = os.path.join(PROJECT_ROOT, config['data_folders']['main_dataset_folder'])
    PREPARED_DATASET_DIR = os.path.join(DATA_ROOT, config['data_folders']['prepared_subfolder'])
    os.makedirs(PREPARED_DATASET_DIR, exist_ok=True)
    PARAMS = {**config['processing_parameters'], **config['training_parameters']}

    print("--- Starting Universal Dataset Preparation ---")
    
    # --- 2. Load All Data Sources Defined in Config ---
    print("\nStep 1: Loading all data sources from config...")
    data_sources = {}
    for name, source_info in config['data_sources'].items():
        # --- Skip any keys that start with an underscore (like '_comment') ---
        if name.startswith('_'):
            continue
        path = os.path.join(DATA_ROOT, source_info['folder'], 'select')
        if source_info['type'] == 'fasta':
            data_sources[name] = load_data_from_fasta(path)
        elif source_info['type'] == 'score':
            data_sources[name] = load_scores(path, source_info['id_col'], source_info['score_col'], name.capitalize())
    
    # --- 3. Process the Primary Molecule ---
    print("\nStep 2: Pre-processing the primary molecule...")
    primary_molecule_type = config['experiment_setup']['primary_molecule']
    primary_molecules_raw = data_sources[primary_molecule_type.lower()]
    processor_func = PROCESSOR_MAP.get(primary_molecule_type)
    
    processing_args = [(item, PARAMS) for item in primary_molecules_raw.items()]
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(processor_func, processing_args)

    processed_molecules, reject_log = [], {"length": 0, "gc": 0, "structure": 0}
    for result in results:
        if isinstance(result, dict): processed_molecules.append(result)
        else: _, reason = result; reject_log[reason.split('_')[1]] += 1
    print(f"  - {len(processed_molecules)} primary molecules passed filters. Rejects: {reject_log}")
    
    # --- 4. Augment with Affinity/Conservation Scores ---
    for molecule_data in processed_molecules:
        molecule_id = molecule_data['id']
        molecule_data['affinity'] = data_sources.get('affinity', {}).get(molecule_id, 0.0)
        match = re.search(r"mir-\d+[a-z]?", molecule_id.lower())
        mirna_family = match.group(0) if match else molecule_id.lower()
        molecule_data['conservation'] = data_sources.get('conservation', {}).get(mirna_family, 0.0)

    # --- 5. Prepare Target and Competitor Molecules ---
    target_type = config['experiment_setup']['target_molecule']
    competitor_type = config['experiment_setup']['competitor_molecule']
    all_targets_full = data_sources[f"{target_type.lower()}_target"]
    all_competitors = data_sources[f"{competitor_type.lower()}_competitor"]

    all_targets_processed = {}
    if PARAMS['focus_on_target_region']:
        region_name, (start, end) = PARAMS['target_region_name'], PARAMS['target_region_slice']
        for target_id, full_seq in all_targets_full.items():
            if len(full_seq) >= end:
                all_targets_processed[f"{target_id}_{region_name}"] = full_seq[start:end]
    else:
        all_targets_processed = all_targets_full

    if not all_targets_processed:
        print("\nCRITICAL ERROR: No target sequences remained after filtering.")
        return

    null_competitor = ('NO_COMPETITOR', '')
    all_competitors_augmented = list(all_competitors.items()) + [null_competitor]
    
    # --- 6. Generate and Stream Final Dataset to Parquet ---
    print("\nStep 3: Generating and streaming combinations to Parquet...")
    output_filename = f"Prepared_Dataset_{int(time.time())}.parquet"
    output_path = os.path.join(PREPARED_DATASET_DIR, output_filename)
    if os.path.exists(output_path): os.remove(output_path)
    
    import pyarrow as pa
    import pyarrow.parquet as pq

    batch, total_rows, parquet_writer = [], 0, None
    combinations = product(processed_molecules, all_targets_processed.items(), all_competitors_augmented)

    for primary_data, (target_id, target_seq), (competitor_id, competitor_seq) in combinations:
        row = {
            'primary_id': primary_data['id'], 'primary_sequence': primary_data['sequence'],
            'target_id': target_id, 'target_sequence': target_seq,
            'competitor_id': competitor_id, 'competitor_sequence': competitor_seq,
            **{k: v for k, v in primary_data.items() if k not in ['id', 'sequence']}
        }
        batch.append(row)

        if len(batch) >= PARAMS['batch_size']:
            df_batch = pd.DataFrame(batch)
            table = pa.Table.from_pandas(df_batch, preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_path, table.schema)
            parquet_writer.write_table(table)
            total_rows += len(batch)
            print(f"  ... {total_rows} rows written")
            batch = []

    if batch:
        df_batch = pd.DataFrame(batch)
        table = pa.Table.from_pandas(df_batch, preserve_index=False)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(output_path, table.schema)
        parquet_writer.write_table(table)
        total_rows += len(batch)

    if parquet_writer: parquet_writer.close()
    
    end_time = time.time()
    print("\n--- Dataset Preparation Summary ---")
    print(f"Total combinations generated (rows): {total_rows}")
    print(f"Dataset saved successfully to {output_path}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    config = load_config()
    prepare_dataset(config)