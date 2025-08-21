# E:\my_deep_learning_project\codes\Gemini_deep_learning_data_preparation.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib # Import joblib for saving the scaler

# --- Configuration Constants (aligned with your updated preparation script) ---
DATASET_ROOT_FOLDER = r"E:\my_deep_learning_project\dataset"
PREPARED_DATASET_FOLDER_NAME = "Prepared Dataset"
PREPARED_DATASET_FILENAME = "prepared_miRNA_RRE_dataset.csv"
PREPARED_DATASET_PATH = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME, PREPARED_DATASET_FILENAME)

# Folder to save processed data for Deep Learning models
OUTPUT_DL_FOLDER = os.path.join(DATASET_ROOT_FOLDER, "Processed_for_DL")

# Ensure output directory exists
os.makedirs(OUTPUT_DL_FOLDER, exist_ok=True)


# Constants for sequence processing (aligned with your updated preparation script)
MAX_MIRNA_SEQUENCE_LENGTH = 80
MAX_RRE_SEQUENCE_LENGTH = 150
MAX_REV_SEQUENCE_LENGTH = 200

# Nucleotide mapping for one-hot encoding
NUCLEOTIDE_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
N_NUCLEOTIDES = len(NUCLEOTIDE_MAP) # 5 (A, U, G, C, N for padding/unknown)

# --- Helper Functions for Deep Learning Preparation ---

def one_hot_encode_sequence(sequence, max_len, nucleotide_map, n_nucleotides):
    """
    One-hot encodes a nucleotide sequence and pads/truncates it.
    'N' is used for padding or unknown nucleotides.
    """
    encoded_seq = np.zeros((max_len, n_nucleotides), dtype=np.float32)
    # Ensure sequence is treated as string; handle potential None/NaN
    if sequence is None or not isinstance(sequence, str):
        sequence = "" # Treat as empty sequence
    
    for i, char in enumerate(sequence[:max_len]):
        char_upper = char.upper() # Standardize case
        if char_upper in nucleotide_map:
            encoded_seq[i, nucleotide_map[char_upper]] = 1
        else: # Handle unexpected characters by treating them as 'N'
            encoded_seq[i, nucleotide_map['N']] = 1
    return encoded_seq

def process_and_save_for_dl(df):
    """
    Processes the prepared DataFrame for deep learning input,
    splits into train/test, and saves the data and scaler.
    """
    if df.empty:
        print("Input DataFrame is empty. Cannot proceed with deep learning data preparation.")
        return None, None, None, None

    # --- 1. Feature Selection and Type Conversion ---
    # Numerical features that need scaling and will be used as input
    # 'affinity' is included here assuming it's a continuous feature input,
    # and 'label' is the binary target derived from it.
    numerical_features_to_scale = [
        'gc_content',
        'dg',
        'conservation',
        'affinity' 
        # 'rev_rre_interaction_score' # Uncomment if you calculate and add this later
    ]

    # Convert structure_vector from string representation of list to actual list of ints
    df['structure_vector'] = df['structure_vector'].apply(
        lambda x: np.array(eval(x), dtype=np.int32) if isinstance(x, str) else np.array(x, dtype=np.int32)
    )

    # Handle potential NaNs in numerical columns (e.g., from RNAfold errors or missing affinity)
    for col in numerical_features_to_scale:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Ensure numeric, then fill NaNs

    # --- 2. Sequence One-Hot Encoding and Padding ---
    print("One-hot encoding miRNA sequences...")
    miRNA_sequences_encoded = np.array([
        one_hot_encode_sequence(seq, MAX_MIRNA_SEQUENCE_LENGTH, NUCLEOTIDE_MAP, N_NUCLEOTIDES)
        for seq in df['sequence']
    ])
    print(f"Shape of encoded miRNA sequences: {miRNA_sequences_encoded.shape}")

    print("One-hot encoding RRE sequences...")
    RRE_sequences_encoded = np.array([
        one_hot_encode_sequence(seq, MAX_RRE_SEQUENCE_LENGTH, NUCLEOTIDE_MAP, N_NUCLEOTIDES)
        for seq in df['rre_sequence']
    ])
    print(f"Shape of encoded RRE sequences: {RRE_sequences_encoded.shape}")

    print("One-hot encoding REV sequences...")
    REV_sequences_encoded = np.array([
        one_hot_encode_sequence(seq, MAX_REV_SEQUENCE_LENGTH, NUCLEOTIDE_MAP, N_NUCLEOTIDES)
        for seq in df['rev_sequence']
    ])
    print(f"Shape of encoded REV sequences: {REV_sequences_encoded.shape}")

    # --- 3. Structure Vector Padding ---
    print("Padding miRNA structure vectors...")
    structure_vectors_padded = pad_sequences(
        df['structure_vector'].tolist(), # .tolist() is important if elements are numpy arrays
        maxlen=MAX_MIRNA_SEQUENCE_LENGTH,
        dtype='int32',
        padding='post', # Pad with 0s at the end
        value=0
    )
    # Reshape to (num_samples, max_len, 1) for Conv1D input
    structure_vectors_padded = np.expand_dims(structure_vectors_padded, axis=-1)
    print(f"Shape of padded structure vectors: {structure_vectors_padded.shape}")

    # --- 4. Numerical Feature Scaling ---
    print("Scaling numerical features...")
    scaler = MinMaxScaler()
    scaled_numerical_features = scaler.fit_transform(df[numerical_features_to_scale])
    print(f"Shape of scaled numerical features: {scaled_numerical_features.shape}")
    
    # Save the fitted scaler for future use (e.g., for new predictions)
    scaler_filepath = os.path.join(OUTPUT_DL_FOLDER, 'minmax_scaler.pkl')
    joblib.dump(scaler, scaler_filepath)
    print(f"MinMaxScaler saved to: {scaler_filepath}")

    # --- 5. Prepare Labels (Target Variable) ---
    labels = df['label'].values
    print(f"Shape of labels: {labels.shape}")

    # --- 6. Split Data into Training and Test Sets ---
    print("Splitting data into training and testing sets...")
    
    # Create a dummy array for splitting that represents all samples
    dummy_x = np.arange(len(df)) 
    
    # Stratify by 'label' if your target variable is imbalanced (highly recommended for classification)
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        dummy_x, labels, test_size=0.2, random_state=42, stratify=labels 
    )

    # --- Class Distribution Check (CRUCIAL for debugging and understanding splits) ---
    print("\n--- Class Distribution after Splitting ---")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"Training set class distribution: {dict(zip(unique_train, counts_train))}")
    
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"Test set class distribution: {dict(zip(unique_test, counts_test))}")
    # If any class count is 0 in either set, especially test, it's a problem.

    # Extract respective portions for each input type using the generated indices
    X_train = {
        'mirna_sequence_input': miRNA_sequences_encoded[X_train_idx],
        'rre_sequence_input': RRE_sequences_encoded[X_train_idx],
        'mirna_structure_input': structure_vectors_padded[X_train_idx],
        'numerical_features_input': scaled_numerical_features[X_train_idx],
        'rev_sequence_input': REV_sequences_encoded[X_train_idx]
    }
    
    X_test = {
        'mirna_sequence_input': miRNA_sequences_encoded[X_test_idx],
        'rre_sequence_input': RRE_sequences_encoded[X_test_idx],
        'mirna_structure_input': structure_vectors_padded[X_test_idx],
        'numerical_features_input': scaled_numerical_features[X_test_idx],
        'rev_sequence_input': REV_sequences_encoded[X_test_idx]
    }
    
    print(f"Training set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    
    # --- 7. Save Processed Data for DL ---
    print(f"Saving processed data to {OUTPUT_DL_FOLDER}...")

    # Saving training data
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'X_train_mirna_seq.npy'), X_train['mirna_sequence_input'])
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'X_train_rre_seq.npy'), X_train['rre_sequence_input'])
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'X_train_mirna_struct.npy'), X_train['mirna_structure_input'])
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'X_train_numerical.npy'), X_train['numerical_features_input'])
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'X_train_rev_seq.npy'), X_train['rev_sequence_input'])
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'y_train.npy'), y_train)

    # Saving test data
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'X_test_mirna_seq.npy'), X_test['mirna_sequence_input'])
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'X_test_rre_seq.npy'), X_test['rre_sequence_input'])
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'X_test_mirna_struct.npy'), X_test['mirna_structure_input'])
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'X_test_numerical.npy'), X_test['numerical_features_input'])
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'X_test_rev_seq.npy'), X_test['rev_sequence_input'])
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'y_test.npy'), y_test)
    
    print(f"All processed data saved to {OUTPUT_DL_FOLDER}")
    
    return X_train, X_test, y_train, y_test

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading prepared dataset from: {PREPARED_DATASET_PATH}")
    if os.path.exists(PREPARED_DATASET_PATH):
        prepared_df = pd.read_csv(PREPARED_DATASET_PATH)
        print(f"Prepared dataset loaded. Shape: {prepared_df.shape}")
        
        # Display columns and a sample to verify
        print("Columns in prepared dataset:", prepared_df.columns.tolist())
        print("Sample of 'sequence' and 'rev_sequence' before processing:")
        print(prepared_df[['sequence', 'rev_sequence']].head())

        X_train, X_test, y_train, y_test = process_and_save_for_dl(prepared_df)
        
        if X_train is not None:
            print("\nDeep learning data preparation complete. Ready for model building!")
        else:
            print("\nDeep learning data preparation failed. Check earlier warnings/errors.")
    else:
        print(f"Error: Prepared dataset not found at {PREPARED_DATASET_PATH}.")
        print("Please run the dataset_preparation.py script first to generate the dataset.")