# 1b_split_dataset.py (Corrected Memory-Safe Version)
import os
import pandas as pd
import pyarrow.parquet as pq
import math

# --- USER CONFIGURATION ---

# The name of the large Parquet file you want to split.
# Ensure this matches the output from your Stage 1 script.
INPUT_PARQUET_FILE = "Prepared_Dataset_1756063639.parquet" 

# The number of smaller, roughly equal-sized files you want to create.
NUMBER_OF_PARTS = 100

# The base directory where your prepared data is located.
DATASET_ROOT_FOLDER = r"E:\1. miRNA-RNA-Deep-Learning-Model\dataset"
PREPARED_DATASET_FOLDER_NAME = "prepared_dataset"

# --- END OF CONFIGURATION ---


def split_dataset_memory_safe():
    """
    Loads a large Parquet dataset in chunks (row groups) and splits it 
    into a specified number of smaller Parquet files without loading the
    entire file into RAM.
    """
    print("--- Starting Memory-Safe Dataset Dissection Script ---")

    # --- 1. Define Paths ---
    input_path = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME, INPUT_PARQUET_FILE)
    
    # Create a new subfolder to store the split parts
    output_folder = os.path.join(DATASET_ROOT_FOLDER, PREPARED_DATASET_FOLDER_NAME, "split_parts")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Input file: {input_path}")
    print(f"Output folder for split parts: {output_folder}")

    if not os.path.exists(input_path):
        print(f"\nFATAL ERROR: Input file not found at '{input_path}'.")
        return

    # --- 2. Open Parquet File and Plan the Split ---
    try:
        parquet_file = pq.ParquetFile(input_path)
        total_row_groups = parquet_file.num_row_groups
        
        if total_row_groups < NUMBER_OF_PARTS:
            print(f"\nWarning: The number of parts ({NUMBER_OF_PARTS}) is greater than the number of chunks in the file ({total_row_groups}).")
            print(f"         The script will create {total_row_groups} smaller files instead.")
            num_parts_to_create = total_row_groups
        else:
            num_parts_to_create = NUMBER_OF_PARTS
            
        # Calculate how many row groups go into each new file
        if num_parts_to_create > 0:
            groups_per_part = math.ceil(total_row_groups / num_parts_to_create)
        else:
            groups_per_part = 0 # Handle case with 0 row groups
        
        print(f"  - Input file has {total_row_groups} internal chunks (row groups).")
        print(f"  - Each of the {num_parts_to_create} output files will contain up to {groups_per_part} chunks.")

    except Exception as e:
        print(f"  - Error reading Parquet file: {e}")
        return

    # --- 3. Read Chunks and Write to New Files ---
    print("\nSplitting the dataset and saving parts...")
    
    current_group_index = 0
    for part_num in range(1, num_parts_to_create + 1):
        output_filename = f"prepared_dataset_part_{part_num}_of_{num_parts_to_create}.parquet"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            # Define the range of chunks for this part
            start_group = current_group_index
            end_group = min(current_group_index + groups_per_part, total_row_groups)

            # Open a writer for the new output file
            # <<< FIX: Changed parquet_file.schema to parquet_file.schema_arrow
            with pq.ParquetWriter(output_path, parquet_file.schema_arrow) as writer:
                # Read only the necessary chunks and write them to the new file
                for i in range(start_group, end_group):
                    writer.write_table(parquet_file.read_row_group(i))
            
            print(f"  - Successfully saved '{output_filename}' (contains chunks {start_group}-{end_group-1}).")
            current_group_index = end_group

        except Exception as e:
            print(f"  - Error saving part {part_num}: {e}")

    print("\n--- Dataset Dissection Complete ---")
    print(f"All parts have been saved to: '{output_folder}'")


if __name__ == "__main__":
    split_dataset_memory_safe()