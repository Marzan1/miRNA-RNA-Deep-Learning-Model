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

    # --- Step 1: Preliminary Scan ---
    print(f"\nStep 1: First pass on '{PREPARED_DATASET_FILENAME}' to fit scaler and count rows...")
    import pyarrow.parquet as pq
    try:
        parquet_file = pq.ParquetFile(PREPARED_DATASET_PATH)
        total_rows = parquet_file.metadata.num_rows
        print(f"  - Found {total_rows} total rows in Parquet file.")
    except Exception as e:
        print(f"  - Error reading Parquet metadata: {e}")
        return

    scaler = MinMaxScaler()
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=NUMERICAL_FEATURES):
        scaler.partial_fit(batch.to_pandas())

    scaler_filepath = os.path.join(OUTPUT_DL_FOLDER, 'minmax_scaler.pkl')
    joblib.dump(scaler, scaler_filepath)
    print(f"  - Scaler has been fitted and saved to: {scaler_filepath}")

    # --- Step 2: Global Train/Test Split ---
    print("\nStep 2: Creating a global 80/20 train-test split...")
    all_indices = np.arange(total_rows)
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
    train_indices_set = set(train_indices)
    test_indices_set = set(test_indices)
    print(f"  - Training indices: {len(train_indices)}, Testing indices: {len(test_indices)}")
    del all_indices

    # --- Step 3 & 4: Process Chunks and Save Directly to Temporary Files ---
    print(f"\nStep 3 & 4: Processing chunks and saving directly to disk...")

    # Define paths for our final output files
    output_paths = {
        'X_train_mirna': os.path.join(OUTPUT_DL_FOLDER, 'X_train_mirna_sequence_input.npy'),
        'X_train_rre': os.path.join(OUTPUT_DL_FOLDER, 'X_train_rre_sequence_input.npy'),
        'X_train_rev': os.path.join(OUTPUT_DL_FOLDER, 'X_train_rev_sequence_input.npy'),
        'X_train_struct': os.path.join(OUTPUT_DL_FOLDER, 'X_train_mirna_structure_input.npy'),
        'X_train_num': os.path.join(OUTPUT_DL_FOLDER, 'X_train_numerical_features_input.npy'),
        'y_train': os.path.join(OUTPUT_DL_FOLDER, 'y_train.npy'),
        'X_test_mirna': os.path.join(OUTPUT_DL_FOLDER, 'X_test_mirna_sequence_input.npy'),
        'X_test_rre': os.path.join(OUTPUT_DL_FOLDER, 'X_test_rre_sequence_input.npy'),
        'X_test_rev': os.path.join(OUTPUT_DL_FOLDER, 'X_test_rev_sequence_input.npy'),
        'X_test_struct': os.path.join(OUTPUT_DL_FOLDER, 'X_test_mirna_structure_input.npy'),
        'X_test_num': os.path.join(OUTPUT_DL_FOLDER, 'X_test_numerical_features_input.npy'),
        'y_test': os.path.join(OUTPUT_DL_FOLDER, 'y_test.npy'),
    }

    # Delete old files if they exist
    for path in output_paths.values():
        if os.path.exists(path):
            os.remove(path)

    processed_rows = 0
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
        chunk_df = batch.to_pandas()
        chunk_indices = pd.RangeIndex(start=processed_rows, stop=processed_rows + len(chunk_df))

        # Find intersections and create local indices
        train_chunk_indices = train_indices_set.intersection(chunk_indices)
        test_chunk_indices = test_indices_set.intersection(chunk_indices)
        local_train_indices = [idx - processed_rows for idx in train_chunk_indices]
        local_test_indices = [idx - processed_rows for idx in test_chunk_indices]

        # Process and append train data directly to files
        if local_train_indices:
            train_subset = chunk_df.iloc[local_train_indices]
            # --- This is an inline function to append to a .npy file ---
            def append_to_npy(filepath, data):
                with open(filepath, 'ab') as f:
                    np.save(f, data)
            
            append_to_npy(output_paths['y_train'], train_subset[TARGET_FEATURE].values.astype(np.float32))
            append_to_npy(output_paths['X_train_mirna'], np.array([one_hot_encode_sequence(seq, MAX_MIRNA_LEN) for seq in train_subset['sequence'].values]))
            append_to_npy(output_paths['X_train_rre'], np.array([one_hot_encode_sequence(seq, MAX_RRE_LEN) for seq in train_subset['rre_sequence'].values]))
            append_to_npy(output_paths['X_train_rev'], np.array([one_hot_encode_sequence(seq, MAX_REV_LEN) for seq in train_subset['rev_sequence'].values]))
            append_to_npy(output_paths['X_train_struct'], np.expand_dims(pad_sequences(train_subset['structure_vector'].apply(json.loads).tolist(), maxlen=MAX_MIRNA_LEN, padding='post', dtype='float32'), axis=-1))
            append_to_npy(output_paths['X_train_num'], scaler.transform(train_subset[NUMERICAL_FEATURES].values))

        # Process and append test data directly to files
        if local_test_indices:
            test_subset = chunk_df.iloc[local_test_indices]
            append_to_npy(output_paths['y_test'], test_subset[TARGET_FEATURE].values.astype(np.float32))
            append_to_npy(output_paths['X_test_mirna'], np.array([one_hot_encode_sequence(seq, MAX_MIRNA_LEN) for seq in test_subset['sequence'].values]))
            append_to_npy(output_paths['X_test_rre'], np.array([one_hot_encode_sequence(seq, MAX_RRE_LEN) for seq in test_subset['rre_sequence'].values]))
            append_to_npy(output_paths['X_test_rev'], np.array([one_hot_encode_sequence(seq, MAX_REV_LEN) for seq in test_subset['rev_sequence'].values]))
            append_to_npy(output_paths['X_test_struct'], np.expand_dims(pad_sequences(test_subset['structure_vector'].apply(json.loads).tolist(), maxlen=MAX_MIRNA_LEN, padding='post', dtype='float32'), axis=-1))
            append_to_npy(output_paths['X_test_num'], scaler.transform(test_subset[NUMERICAL_FEATURES].values))

        processed_rows += len(chunk_df)
        print(f"  - Processed and saved {processed_rows}/{total_rows} rows...")

    # --- Step 5: Finalize and clean up ---
    print("\nStep 5: Merging temporary chunk files into final arrays...")
    
    # This function will load all the small chunks and concatenate them
    def merge_npy_chunks(filepath):
        chunks = []
        with open(filepath, 'rb') as f:
            while True:
                try:
                    chunks.append(np.load(f, allow_pickle=True))
                except (EOFError, ValueError):
                    break
        return np.concatenate(chunks)

    for key, path in output_paths.items():
        print(f"  - Merging {key}...")
        final_array = merge_npy_chunks(path)
        np.save(path, final_array) # Overwrite the chunked file with the final merged array
    
    print("  - All NumPy array files finalized successfully.")
    
    end_time = time.time()
    print("\n--- Deep Learning Data Preparation Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()