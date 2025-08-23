# 2. deep_learning_data_preparation.py (Corrected Version)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import json
import time # <--- THIS LINE WAS ADDED TO FIX THE ERROR

# --- Configuration ---
DATASET_ROOT_FOLDER = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset"
PREPARED_DATASET_FOLDER_NAME = "prepared_dataset"
PREPARED_DATASET_FILENAME = "Prepared_Dataset.parquet"
PREPARED_DATASET_PATH = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME, PREPARED_DATASET_FILENAME)
OUTPUT_DL_FOLDER = os.path.join(DATASET_ROOT_FOLDER, "processed_for_dl")
os.makedirs(OUTPUT_DL_FOLDER, exist_ok=True)

# --- Constants ---
CHUNK_SIZE = 50000  # <--- ADD THIS LINE BACK
MAX_MIRNA_LEN, MAX_RRE_LEN, MAX_REV_LEN = 80, 150, 200
NUCLEOTIDE_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
N_NUCLEOTIDES = len(NUCLEOTIDE_MAP)
NUMERICAL_FEATURES = ['gc_content', 'dg', 'conservation']
TARGET_FEATURE = 'affinity'

# --- Helper Functions ---
def one_hot_encode_sequence(sequence, max_len):
    encoded_seq = np.zeros((max_len, N_NUCLEOTIDES), dtype=np.float32)
    if not isinstance(sequence, str): sequence = ""
    for i, char in enumerate(sequence[:max_len]):
        encoded_seq[i, NUCLEOTIDE_MAP.get(char.upper(), 4)] = 1
    return encoded_seq

# --- Main Processing Function ---
def main():
    start_time = time.time()
    print("--- Starting Enhanced Data Preparation for Deep Learning ---")
    
    if not os.path.exists(PREPARED_DATASET_PATH):
        print(f"\nError: Prepared dataset not found at '{PREPARED_DATASET_PATH}'.")
        return

    # --- Step 1: Preliminary Scan to get total rows and fit the scaler ---
    print(f"\nStep 1: First pass on '{PREPARED_DATASET_FILENAME}' to fit scaler and count rows...")
    
    # You may need this library: pip install pyarrow fastparquet
    import pyarrow.parquet as pq
    
    try:
        parquet_file = pq.ParquetFile(PREPARED_DATASET_PATH)
        total_rows = parquet_file.metadata.num_rows
        print(f"  - Found {total_rows} total rows in Parquet file.")
    except Exception as e:
        print(f"  - Error reading Parquet metadata: {e}")
        return

    scaler = MinMaxScaler()
    # Fit the scaler iteratively over batches to save memory
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=NUMERICAL_FEATURES):
        chunk_df = batch.to_pandas()
        scaler.partial_fit(chunk_df)

    scaler_filepath = os.path.join(OUTPUT_DL_FOLDER, 'minmax_scaler.pkl')
    joblib.dump(scaler, scaler_filepath)
    print(f"  - Scaler has been fitted and saved to: {scaler_filepath}")

    # --- Step 2: Create a global train/test split of indices ---
    print("\nStep 2: Creating a global 80/20 train-test split...")
    all_indices = np.arange(total_rows)
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
    train_indices_set = set(train_indices)
    test_indices_set = set(test_indices)
    print(f"  - Training indices: {len(train_indices)}, Testing indices: {len(test_indices)}")
    del all_indices

    # --- Step 3: Second pass to process data in chunks ---
    print(f"\nStep 3: Second pass to process data in chunks of {CHUNK_SIZE}...")
    
    train_chunks = {key: [] for key in ['y', 'mirna', 'rre', 'rev', 'struct', 'num']}
    test_chunks = {key: [] for key in ['y', 'mirna', 'rre', 'rev', 'struct', 'num']}
    processed_rows = 0

    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
        chunk = batch.to_pandas()
        chunk_indices = np.arange(processed_rows, processed_rows + len(chunk))
        
        # Determine which rows in this chunk belong to train vs test
        train_mask = np.isin(chunk_indices, train_indices_set)
        test_mask = np.isin(chunk_indices, test_indices_set)

        # Process and append train data
        if np.any(train_mask):
            train_subset = chunk[train_mask]
            train_chunks['y'].append(train_subset[TARGET_FEATURE].values.astype(np.float32))
            train_chunks['mirna'].append(np.array([one_hot_encode_sequence(seq, MAX_MIRNA_LEN) for seq in train_subset['sequence'].values]))
            train_chunks['rre'].append(np.array([one_hot_encode_sequence(seq, MAX_RRE_LEN) for seq in train_subset['rre_sequence'].values]))
            train_chunks['rev'].append(np.array([one_hot_encode_sequence(seq, MAX_REV_LEN) for seq in train_subset['rev_sequence'].values]))
            train_chunks['struct'].append(np.expand_dims(pad_sequences(train_subset['structure_vector'].apply(json.loads).tolist(), maxlen=MAX_MIRNA_LEN, padding='post', dtype='float32'), axis=-1))
            train_chunks['num'].append(scaler.transform(train_subset[NUMERICAL_FEATURES].values))
        
        # Process and append test data
        if np.any(test_mask):
            test_subset = chunk[test_mask]
            test_chunks['y'].append(test_subset[TARGET_FEATURE].values.astype(np.float32))
            test_chunks['mirna'].append(np.array([one_hot_encode_sequence(seq, MAX_MIRNA_LEN) for seq in test_subset['sequence'].values]))
            test_chunks['rre'].append(np.array([one_hot_encode_sequence(seq, MAX_RRE_LEN) for seq in test_subset['rre_sequence'].values]))
            test_chunks['rev'].append(np.array([one_hot_encode_sequence(seq, MAX_REV_LEN) for seq in test_subset['rev_sequence'].values]))
            test_chunks['struct'].append(np.expand_dims(pad_sequences(test_subset['structure_vector'].apply(json.loads).tolist(), maxlen=MAX_MIRNA_LEN, padding='post', dtype='float32'), axis=-1))
            test_chunks['num'].append(scaler.transform(test_subset[NUMERICAL_FEATURES].values))

        processed_rows += len(chunk)
        print(f"  - Processed {processed_rows}/{total_rows} rows...")

    # --- Step 4: Final assembly and saving ---
    print("\nStep 4: Assembling and saving final NumPy arrays...")
    
    y_train = np.concatenate(train_chunks['y'])
    y_test = np.concatenate(test_chunks['y'])

    X_train = {
        'mirna_sequence_input': np.concatenate(train_chunks['mirna']),
        'rre_sequence_input': np.concatenate(train_chunks['rre']),
        'rev_sequence_input': np.concatenate(train_chunks['rev']),
        'mirna_structure_input': np.concatenate(train_chunks['struct']),
        'numerical_features_input': np.concatenate(train_chunks['num'])
    }
    X_test = {
        'mirna_sequence_input': np.concatenate(test_chunks['mirna']),
        'rre_sequence_input': np.concatenate(test_chunks['rre']),
        'rev_sequence_input': np.concatenate(test_chunks['rev']),
        'mirna_structure_input': np.concatenate(test_chunks['struct']),
        'numerical_features_input': np.concatenate(test_chunks['num'])
    }
    
    for key, data in X_train.items(): np.save(os.path.join(OUTPUT_DL_FOLDER, f'X_train_{key}.npy'), data)
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'y_train.npy'), y_train)
    for key, data in X_test.items(): np.save(os.path.join(OUTPUT_DL_FOLDER, f'X_test_{key}.npy'), data)
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'y_test.npy'), y_test)
    
    print("  - All NumPy array files saved successfully.")
    
    end_time = time.time()
    print("\n--- Deep Learning Data Preparation Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()