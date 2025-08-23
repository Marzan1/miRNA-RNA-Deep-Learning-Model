# 2. deep_learning_data_preparation.py (Enhanced Research-Grade Version)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import json
import time

# --- Configuration ---
DATASET_ROOT_FOLDER = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset"
PREPARED_DATASET_FOLDER_NAME = "prepared_dataset"
PREPARED_DATASET_FILENAME = "Prepared_Dataset.csv"
PREPARED_DATASET_PATH = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME, PREPARED_DATASET_FILENAME)
OUTPUT_DL_FOLDER = os.path.join(DATASET_ROOT_FOLDER, "processed_for_dl")
os.makedirs(OUTPUT_DL_FOLDER, exist_ok=True)

# --- Constants ---
CHUNK_SIZE = 50000
MAX_MIRNA_LEN, MAX_RRE_LEN, MAX_REV_LEN = 80, 150, 200
NUCLEOTIDE_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
N_NUCLEOTIDES = len(NUCLEOTIDE_MAP)
NUMERICAL_FEATURES = ['gc_content', 'dg', 'conservation']
SEQUENCE_FEATURES = ['sequence', 'rre_sequence', 'rev_sequence']
STRUCTURE_FEATURE = 'structure_vector'
LABEL_FEATURE = 'label'
EXPECTED_COLUMNS = NUMERICAL_FEATURES + SEQUENCE_FEATURES + [STRUCTURE_FEATURE, LABEL_FEATURE]

# --- Helper Functions ---
def one_hot_encode_sequence(sequence, max_len):
    encoded_seq = np.zeros((max_len, N_NUCLEOTIDES), dtype=np.float32)
    if not isinstance(sequence, str): sequence = ""
    for i, char in enumerate(sequence[:max_len]):
        encoded_seq[i, NUCLEOTIDE_MAP.get(char.upper(), 4)] = 1
    return encoded_seq

def validate_chunk_columns(chunk_columns):
    """Checks if all required columns are in the chunk."""
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in chunk_columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

# --- Main Processing Function ---
def main():
    start_time = time.time()
    print("--- Starting Enhanced Data Preparation for Deep Learning ---")
    
    if not os.path.exists(PREPARED_DATASET_PATH):
        print(f"\nError: Prepared dataset not found at '{PREPARED_DATASET_PATH}'.")
        return

    # --- Step 1: First Pass to Fit Scaler and Validate Schema ---
    print(f"\nStep 1: First pass on CSV to fit scaler, count rows, and validate schema...")
    total_rows = 0
    scaler = MinMaxScaler()
    
    with pd.read_csv(PREPARED_DATASET_PATH, chunksize=CHUNK_SIZE, low_memory=False, nrows=CHUNK_SIZE) as reader:
        # Validate schema on the first chunk
        first_chunk = next(reader)
        validate_chunk_columns(first_chunk.columns)
        print("  - CSV schema validated successfully.")
        total_rows += len(first_chunk)
        # Clean and fit scaler on the first chunk
        for col in NUMERICAL_FEATURES:
            first_chunk[col] = pd.to_numeric(first_chunk[col], errors='coerce').fillna(0)
        scaler.partial_fit(first_chunk[NUMERICAL_FEATURES])
        
    with pd.read_csv(PREPARED_DATASET_PATH, chunksize=CHUNK_SIZE, low_memory=False, skiprows=CHUNK_SIZE+1, header=None) as reader:
        # Continue with the rest of the file
        for chunk in reader:
            total_rows += len(chunk)
            chunk.columns = pd.read_csv(PREPARED_DATASET_PATH, nrows=0).columns
            for col in NUMERICAL_FEATURES:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
            scaler.partial_fit(chunk[NUMERICAL_FEATURES])
            
    print(f"  - Found {total_rows} total rows.")
    scaler_filepath = os.path.join(OUTPUT_DL_FOLDER, 'minmax_scaler.pkl')
    joblib.dump(scaler, scaler_filepath)
    print(f"  - Scaler fitted and saved to: {scaler_filepath}")

    # --- Step 2: Global Train/Test Split ---
    print("\nStep 2: Creating a global 80/20 train-test split...")
    all_indices = np.arange(total_rows)
    # Read only the label column for stratification to save memory
    all_labels = pd.read_csv(PREPARED_DATASET_PATH, usecols=[LABEL_FEATURE])[LABEL_FEATURE].values
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42, stratify=all_labels)
    train_indices_set = set(train_indices)
    test_indices_set = set(test_indices)
    print(f"  - Training indices: {len(train_indices)}, Testing indices: {len(test_indices)}")
    del all_labels, all_indices # Free up memory

    # --- Step 3: Second Pass to Process and Split Data ---
    print(f"\nStep 3: Second pass to process data in chunks of {CHUNK_SIZE}...")
    
    # Initialize lists to hold the processed data chunks
    train_data_chunks = []
    test_data_chunks = []

    processed_rows = 0
    with pd.read_csv(PREPARED_DATASET_PATH, chunksize=CHUNK_SIZE, low_memory=False) as reader:
        for chunk in reader:
            # Re-apply cleaning and transformations
            for col in NUMERICAL_FEATURES:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
            chunk[STRUCTURE_FEATURE] = chunk[STRUCTURE_FEATURE].apply(json.loads)

            # Split chunk based on global indices
            chunk_indices = chunk.index.values
            train_mask = np.isin(chunk_indices, list(train_indices_set))
            test_mask = np.isin(chunk_indices, list(test_indices_set))

            # Process train and test data for the chunk if they exist
            if np.any(train_mask):
                train_data_chunks.append({
                    'X_mirna': np.array([one_hot_encode_sequence(seq, MAX_MIRNA_LEN) for seq in chunk.loc[train_mask, 'sequence'].values]),
                    'X_rre': np.array([one_hot_encode_sequence(seq, MAX_RRE_LEN) for seq in chunk.loc[train_mask, 'rre_sequence'].values]),
                    'X_rev': np.array([one_hot_encode_sequence(seq, MAX_REV_LEN) for seq in chunk.loc[train_mask, 'rev_sequence'].values]),
                    'X_struct': np.expand_dims(pad_sequences(chunk.loc[train_mask, STRUCTURE_FEATURE].tolist(), maxlen=MAX_MIRNA_LEN, padding='post', dtype='float32'), axis=-1),
                    'X_num': scaler.transform(chunk.loc[train_mask, NUMERICAL_FEATURES].values),
                    'y': chunk.loc[train_mask, LABEL_FEATURE].values
                })
            
            if np.any(test_mask):
                test_data_chunks.append({
                    'X_mirna': np.array([one_hot_encode_sequence(seq, MAX_MIRNA_LEN) for seq in chunk.loc[test_mask, 'sequence'].values]),
                    'X_rre': np.array([one_hot_encode_sequence(seq, MAX_RRE_LEN) for seq in chunk.loc[test_mask, 'rre_sequence'].values]),
                    'X_rev': np.array([one_hot_encode_sequence(seq, MAX_REV_LEN) for seq in chunk.loc[test_mask, 'rev_sequence'].values]),
                    'X_struct': np.expand_dims(pad_sequences(chunk.loc[test_mask, STRUCTURE_FEATURE].tolist(), maxlen=MAX_MIRNA_LEN, padding='post', dtype='float32'), axis=-1),
                    'X_num': scaler.transform(chunk.loc[test_mask, NUMERICAL_FEATURES].values),
                    'y': chunk.loc[test_mask, LABEL_FEATURE].values
                })
                
            processed_rows += len(chunk)
            print(f"  - Processed {processed_rows}/{total_rows} rows...")

    # --- Step 4: Final Assembly and Saving ---
    print("\nStep 4: Assembling and saving final NumPy arrays...")
    
    # Define keys for clarity
    feature_keys = ['X_mirna', 'X_rre', 'X_rev', 'X_struct', 'X_num']
    model_input_keys = ['mirna_sequence_input', 'rre_sequence_input', 'rev_sequence_input', 'mirna_structure_input', 'numerical_features_input']

    # Assemble training data
    y_train = np.concatenate([d['y'] for d in train_data_chunks])
    X_train = {}
    for feature_key, model_input_key in zip(feature_keys, model_input_keys):
        X_train[model_input_key] = np.concatenate([d[feature_key] for d in train_data_chunks])
    
    # Assemble test data
    y_test = np.concatenate([d['y'] for d in test_data_chunks])
    X_test = {}
    for feature_key, model_input_key in zip(feature_keys, model_input_keys):
        X_test[model_input_key] = np.concatenate([d[feature_key] for d in test_data_chunks])
    
    # Save all files
    for key, data in X_train.items():
        np.save(os.path.join(OUTPUT_DL_FOLDER, f'X_train_{key}.npy'), data)
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'y_train.npy'), y_train)

    for key, data in X_test.items():
        np.save(os.path.join(OUTPUT_DL_FOLDER, f'X_test_{key}.npy'), data)
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'y_test.npy'), y_test)
    
    print("  - All files saved successfully.")
    
    end_time = time.time()
    print("\n--- Deep Learning Data Preparation Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()