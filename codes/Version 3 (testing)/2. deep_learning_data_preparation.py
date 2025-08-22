# E:\my_deep_learning_project\codes\Gemini_deep_learning_data_preparation.py (Revised)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import json

# --- Configuration Constants ---
DATASET_ROOT_FOLDER = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset"
PREPARED_DATASET_FOLDER_NAME = "Prepared Dataset"
PREPARED_DATASET_FILENAME = "prepared_miRNA_RRE_dataset.csv"
PREPARED_DATASET_PATH = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME, PREPARED_DATASET_FILENAME)

# Folder to save processed data for Deep Learning models
OUTPUT_DL_FOLDER = os.path.join(DATASET_ROOT_FOLDER, "Processed_for_DL")
os.makedirs(OUTPUT_DL_FOLDER, exist_ok=True)

# Constants for sequence processing
MAX_MIRNA_LEN = 80
MAX_RRE_LEN = 150
MAX_REV_LEN = 200

# Nucleotide mapping for one-hot encoding
NUCLEOTIDE_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
N_NUCLEOTIDES = len(NUCLEOTIDE_MAP)

# --- Helper Functions ---
def one_hot_encode_sequence(sequence, max_len):
    """One-hot encodes a nucleotide sequence and pads/truncates it."""
    encoded_seq = np.zeros((max_len, N_NUCLEOTIDES), dtype=np.float32)
    
    # Handle missing sequence data gracefully
    if not isinstance(sequence, str):
        sequence = ""
        
    for i, char in enumerate(sequence[:max_len]):
        # Default to 'N' if character is not in the map
        nucleotide_index = NUCLEOTIDE_MAP.get(char.upper(), 4) 
        encoded_seq[i, nucleotide_index] = 1
        
    return encoded_seq

def process_and_save_for_dl(df):
    """
    Processes the DataFrame for deep learning input, splits it, and saves the data.
    """
    if df.empty:
        print("Input DataFrame is empty. Aborting data preparation.")
        return

    # --- 1. Feature Selection and Type Conversion ---
    
    # CRITICAL FIX: 'affinity' is directly related to the 'label'. Including it as a feature
    # is data leakage. We remove it from the model's inputs.
    numerical_features = ['gc_content', 'dg', 'conservation']
    
    print(f"Using numerical features for model input: {numerical_features}")

    # Convert structure_vector from JSON string to a list of numbers
    df['structure_vector'] = df['structure_vector'].apply(json.loads)

    # Clean numerical columns
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- 2. Sequence One-Hot Encoding ---
    print("\nStep 1: One-hot encoding all sequences...")
    X_mirna_seq = np.array([one_hot_encode_sequence(seq, MAX_MIRNA_LEN) for seq in df['sequence']])
    X_rre_seq = np.array([one_hot_encode_sequence(seq, MAX_RRE_LEN) for seq in df['rre_sequence']])
    X_rev_seq = np.array([one_hot_encode_sequence(seq, MAX_REV_LEN) for seq in df['rev_sequence']])
    print(f"  miRNA sequences shape: {X_mirna_seq.shape}")
    print(f"  RRE sequences shape: {X_rre_seq.shape}")
    print(f"  REV sequences shape: {X_rev_seq.shape}")

    # --- 3. Structure Vector Padding ---
    print("\nStep 2: Padding miRNA structure vectors...")
    X_structure = pad_sequences(
        df['structure_vector'].tolist(), 
        maxlen=MAX_MIRNA_LEN, 
        padding='post', 
        dtype='float32'
    )
    # Reshape for Conv1D input: (samples, timesteps, features)
    X_structure = np.expand_dims(X_structure, axis=-1)
    print(f"  Padded structure vectors shape: {X_structure.shape}")

    # --- 4. Numerical Feature Scaling ---
    print("\nStep 3: Scaling numerical features...")
    scaler = MinMaxScaler()
    X_numerical = scaler.fit_transform(df[numerical_features])
    print(f"  Scaled numerical features shape: {X_numerical.shape}")
    
    # Save the scaler for predicting on new data later
    scaler_filepath = os.path.join(OUTPUT_DL_FOLDER, 'minmax_scaler.pkl')
    joblib.dump(scaler, scaler_filepath)
    print(f"  Scaler saved to: {scaler_filepath}")

    # --- 5. Prepare Labels ---
    y = df['label'].values
    print(f"\nStep 4: Preparing labels (shape: {y.shape})...")

    # --- 6. Split Data into Training and Test Sets ---
    print("\nStep 5: Splitting data into training (80%) and testing (20%) sets...")
    
    # Create an array of indices to split all datasets consistently
    indices = np.arange(len(df))
    
    # Stratify by 'label' to ensure both train and test sets have a similar class distribution
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )

    # Use the indices to create the final dictionaries for the model
    X_train = {
        'mirna_sequence_input': X_mirna_seq[train_indices],
        'rre_sequence_input': X_rre_seq[train_indices],
        'rev_sequence_input': X_rev_seq[train_indices],
        'mirna_structure_input': X_structure[train_indices],
        'numerical_features_input': X_numerical[train_indices]
    }
    y_train = y[train_indices]

    X_test = {
        'mirna_sequence_input': X_mirna_seq[test_indices],
        'rre_sequence_input': X_rre_seq[test_indices],
        'rev_sequence_input': X_rev_seq[test_indices],
        'mirna_structure_input': X_structure[test_indices],
        'numerical_features_input': X_numerical[test_indices]
    }
    y_test = y[test_indices]
    
    print(f"  Training set size: {len(y_train)}")
    print(f"  Test set size: {len(y_test)}")
    print(f"  Training class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # --- 7. Save Processed Data for DL ---
    print(f"\nStep 6: Saving processed data to '{OUTPUT_DL_FOLDER}'...")
    
    # Save training data
    for key, data in X_train.items():
        np.save(os.path.join(OUTPUT_DL_FOLDER, f'X_train_{key}.npy'), data)
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'y_train.npy'), y_train)

    # Save test data
    for key, data in X_test.items():
        np.save(os.path.join(OUTPUT_DL_FOLDER, f'X_test_{key}.npy'), data)
    np.save(os.path.join(OUTPUT_DL_FOLDER, 'y_test.npy'), y_test)
    
    print("  All files saved successfully.")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading prepared dataset from: {PREPARED_DATASET_PATH}")
    if os.path.exists(PREPARED_DATASET_PATH):
        try:
            prepared_df = pd.read_csv(PREPARED_DATASET_PATH)
            print(f"Prepared dataset loaded. Shape: {prepared_df.shape}")
            
            if prepared_df.shape[0] > 0:
                process_and_save_for_dl(prepared_df)
                print("\nDeep learning data preparation complete. Ready for model building!")
            else:
                print("\nWarning: The prepared dataset is empty. Cannot proceed.")
                
        except Exception as e:
            print(f"\nAn error occurred while processing the CSV file: {e}")
            print("Please check if the CSV file is valid and not corrupted.")
            
    else:
        print(f"\nError: Prepared dataset not found at '{PREPARED_DATASET_PATH}'.")
        print("Please run the `1. dataset_preparation.py` script first to generate the dataset.")