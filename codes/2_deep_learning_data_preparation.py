# 2_deep_learning_data_preparation.py (Final, Corrected Version)
import os
import pandas as pd
import numpy as np
import json
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
DATASET_ROOT_FOLDER = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset"
PREPARED_DATASET_FOLDER_NAME = "prepared_dataset"
OUTPUT_DL_FOLDER = os.path.join(DATASET_ROOT_FOLDER, "processed_for_dl")
os.makedirs(OUTPUT_DL_FOLDER, exist_ok=True)

# --- Auto-detect the most recent Parquet file ---
prepared_folder_path = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME)
try:
    # Find all Parquet files in the directory
    prepared_files = [f for f in os.listdir(prepared_folder_path) if f.endswith('.parquet')]
    if not prepared_files:
        raise FileNotFoundError
    # Sort them (which works for timestamps) and get the latest one
    PREPARED_DATASET_FILENAME = sorted(prepared_files)[-1]
    PREPARED_DATASET_PATH = os.path.join(prepared_folder_path, PREPARED_DATASET_FILENAME)
except FileNotFoundError:
    print(f"FATAL ERROR: No .parquet files found in '{prepared_folder_path}'. Please run Stage 1 first.")
    exit()
# --- END of new auto-detection ---

# --- Constants ---
MAX_MIRNA_LEN, MAX_RRE_LEN, MAX_REV_LEN = 80, 150, 200 # These should match your model architecture plans
NUCLEOTIDE_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
NUMERICAL_FEATURES = ['gc_content', 'dg', 'conservation']
TARGET_FEATURE = 'affinity'

# --- Helper Functions ---
def one_hot_encode_sequence(sequence, max_len):
    encoded_seq = np.zeros((max_len, len(NUCLEOTIDE_MAP)), dtype=np.float32)
    if not isinstance(sequence, str): sequence = ""
    for i, char in enumerate(sequence[:max_len]):
        encoded_seq[i, NUCLEOTIDE_MAP.get(char.upper(), 4)] = 1
    return encoded_seq

# --- Main Processing Function ---
def main():
    start_time = time.time()
    print("--- Starting Data Preparation for Deep Learning ---")
    
    if not os.path.exists(PREPARED_DATASET_PATH):
        print(f"\nError: Prepared dataset not found at '{PREPARED_DATASET_PATH}'.")
        return

    print(f"Reading from Parquet file: {PREPARED_DATASET_PATH}")
    try:
        df = pd.read_parquet(PREPARED_DATASET_PATH)
        print(f"  - Successfully loaded {len(df)} rows.")
    except Exception as e:
        print(f"  - Error reading Parquet file: {e}")
        return

    print("\nStep 1: Fitting the scaler...")
    scaler = MinMaxScaler()
    scaler.fit(df[NUMERICAL_FEATURES])
    joblib.dump(scaler, os.path.join(OUTPUT_DL_FOLDER, 'minmax_scaler.pkl'))
    print("  - Scaler fitted and saved.")

    print("\nStep 2: Creating train-test split...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"  - Training set size: {len(train_df)}, Testing set size: {len(test_df)}")
    del df

    print("\nStep 3: Processing and saving datasets...")
    
    def process_dataframe(dataframe, scaler_obj):
        """Helper to process a dataframe into the final numpy format."""
        y = dataframe[TARGET_FEATURE].values.astype(np.float32)
        X = {
            'primary_sequence_input': np.array([one_hot_encode_sequence(seq, MAX_MIRNA_LEN) for seq in dataframe['primary_sequence'].values]),
            'target_sequence_input': np.array([one_hot_encode_sequence(seq, MAX_RRE_LEN) for seq in dataframe['target_sequence'].values]),
            'competitor_sequence_input': np.array([one_hot_encode_sequence(seq, MAX_REV_LEN) for seq in dataframe['competitor_sequence'].values]),
            'primary_structure_input': np.expand_dims(pad_sequences(dataframe['structure_vector'].apply(json.loads).tolist(), maxlen=MAX_MIRNA_LEN, padding='post', dtype='float32'), axis=-1),
            'numerical_features_input': scaler_obj.transform(dataframe[NUMERICAL_FEATURES].values)
        }
        return X, y

    # Process and save training data
    print("  - Processing and saving training data...")
    X_train, y_train = process_dataframe(train_df, scaler)
    for key, data in X_train.items(): np.save(os.path.join(OUTPUT_DL_FOLDER, f'X_train_{key}.npy'), data)
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'y_train.npy'), y_train)
    del train_df, X_train, y_train

    # Process and save test data
    print("  - Processing and saving test data...")
    X_test, y_test = process_dataframe(test_df, scaler)
    for key, data in X_test.items(): np.save(os.path.join(OUTPUT_DL_FOLDER, f'X_test_{key}.npy'), data)
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'y_test.npy'), y_test)
    del test_df, X_test, y_test

    print("  - All NumPy array files saved successfully.")
    
    end_time = time.time()
    print("\n--- Deep Learning Data Preparation Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()