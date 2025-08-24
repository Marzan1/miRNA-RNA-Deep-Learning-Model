# 6_split_dataset.py
import os
import pandas as pd
import numpy as np
import math

# --- USER CONFIGURATION ---

# The name of the large Parquet file you want to split.
# Ensure this matches the output from your Stage 1 script.
INPUT_PARQUET_FILE = "Prepared_Dataset_1756061187.parquet" 

# The number of smaller, roughly equal-sized files you want to create.
NUMBER_OF_PARTS = 100 

# The base directory where your prepared data is located.
DATASET_ROOT_FOLDER = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset"
PREPARED_DATASET_FOLDER_NAME = "prepared_dataset"

# --- END OF CONFIGURATION ---


def split_dataset():
    """
    Loads a large Parquet dataset and splits it into a specified
    number of smaller Parquet files.
    """
    print("--- Starting Dataset Dissection Script ---")

    # --- 1. Define Paths ---
    input_path = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME, INPUT_PARQUET_FILE)
    
    # Create a new subfolder to store the split parts
    output_folder = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME, "split_parts")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Input file: {input_path}")
    print(f"Output folder for split parts: {output_folder}")

    if not os.path.exists(input_path):
        print(f"\nFATAL ERROR: Input file not found at '{input_path}'.")
        print("Please make sure the INPUT_PARQUET_FILE name is correct.")
        return

    # --- 2. Load the Full Dataset ---
    print("\nLoading the full dataset into memory...")
    try:
        df = pd.read_parquet(input_path)
        total_rows = len(df)
        print(f"  - Successfully loaded {total_rows} rows.")
    except Exception as e:
        print(f"  - Error reading Parquet file: {e}")
        return

    # --- 3. Split the DataFrame ---
    print(f"\nSplitting the dataset into {NUMBER_OF_PARTS} parts...")
    
    # Use numpy's array_split for efficient splitting into nearly equal parts
    split_dfs = np.array_split(df, NUMBER_OF_PARTS)
    
    print("  - Data successfully split in memory.")

    # --- 4. Save Each Part to a New Parquet File ---
    print("\nSaving each part to a new Parquet file...")
    for i, part_df in enumerate(split_dfs):
        part_num = i + 1
        output_filename = f"prepared_dataset_part_{part_num}_of_{NUMBER_OF_PARTS}.parquet"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            part_df.to_parquet(output_path, index=False)
            print(f"  - Successfully saved '{output_filename}' with {len(part_df)} rows.")
        except Exception as e:
            print(f"  - Error saving part {part_num}: {e}")

    print("\n--- Dataset Dissection Complete ---")
    print(f"All parts have been saved to: '{output_folder}'")


if __name__ == "__main__":
    split_dataset()