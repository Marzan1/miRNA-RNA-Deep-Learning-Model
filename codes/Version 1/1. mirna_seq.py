#Creating the miRNA dataset

import random
import csv
import os

# Parameters for the synthetic miRNA generation
GC_CONTENT_MIN = 0.50  # Minimum GC content (50%)
GC_CONTENT_MAX = 0.60  # Maximum GC content (60%)
SEQUENCE_LENGTH_MIN = 21  # Minimum length of miRNA
SEQUENCE_LENGTH_MAX = 25  # Maximum length of miRNA
NUCLEOTIDES = ['A', 'U', 'G', 'C']  # RNA nucleotides

# RRE sequence you provided
RRE_SEQUENCE = "UAGCACCCACCAAGGCAAAGAGAAGAGUGGUGCAGAGAGAAAAAAGAGCAGUGGGAAUAGGAGCUUUGUUCCUUGGGUUCUUGGGAGCAGCAGGAAGCACUAUGGGCGCAGCCUCAAUGACGCUGACGGUACAGGCCAGACAAUUAUUGUCUGGUAUAGUGCAGCAGCAGAACAAUUUGCUGAGGGCUAUUGAGGCGCAACAGCAUCUGUUGCAACUCACAGUCUGGGGCAUCAAGCAGCUCCAGGCAAGAAUCCUGGCUGUGGAAAGAUACCUAAAGGAUCAACAGCUCCUGGGGAUUUGGGGUUGCUCUGGAAAACUCAUUUGCACCACUGCUGUGCCUUGGAAUGCUA"
STEM_IIB_SITE = "UAUGGGCGC"  # Example of Stem IIB binding region (can be adjusted)

# Precompute the complementary sequence for the Stem IIB site
complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
complemented_stem_iib = ''.join([complement[n] for n in STEM_IIB_SITE])

# Function to generate random sequences with desired GC content
def generate_sequence_with_gc_content(length, gc_min, gc_max):
    while True:
        # Randomly generate a sequence
        sequence = [random.choice(NUCLEOTIDES) for _ in range(length)]
        
        # Calculate GC content
        gc_content = (sequence.count('G') + sequence.count('C')) / length
        
        # Check if it falls within the desired GC content range
        if gc_min <= gc_content <= gc_max:
            return ''.join(sequence)

# Function to simulate secondary structure by ensuring certain regions are unpaired (e.g., hairpin)
def simulate_secondary_structure(miRNA, unpaired_region_length=6):
    # Simulating by ensuring part of the miRNA is "unpaired" (i.e., no self-complementary)
    unpaired_region = miRNA[-unpaired_region_length:]
    # Check if the unpaired region has low self-complementarity
    return not any(unpaired_region[i] == complement[unpaired_region[-i-1]] for i in range(unpaired_region_length // 2))

# Function to check if the seed region of the miRNA is complementary to the Stem IIB site
def check_seed_complementarity(miRNA, stem_iib_complement):
    # Check if the seed region (positions 2-8) is complementary to the Stem IIB site
    seed_region = miRNA[1:8]  # Positions 2-8 (0-based indexing)
    return seed_region == stem_iib_complement[:7]

# Function to check if the miRNA has complementary sequences to other regions of the RRE
def check_off_target_complementarity(miRNA, rre_sequence, stem_iib_site):
    # Exclude the Stem IIB site from the RRE sequence
    rre_without_stem_iib = rre_sequence.replace(stem_iib_site, '')
    
    # Check if the miRNA is complementary to any region of the RRE (excluding Stem IIB)
    for i in range(len(rre_without_stem_iib) - len(miRNA) + 1):
        substring = rre_without_stem_iib[i:i+len(miRNA)]
        complemented_substring = ''.join([complement[n] for n in substring])
        if miRNA == complemented_substring:
            return False  # miRNA is complementary to another region of the RRE
    return True  # No off-target complementarity

# Generate a list of synthetic miRNA sequences considering all constraints
def generate_mirnas(num_sequences, length_min, length_max, gc_min, gc_max):
    valid_sequences = []
    attempts = 0
    while len(valid_sequences) < num_sequences:
        attempts += 1
        # Randomly choose a length within the specified range
        length = random.randint(length_min, length_max)
        # Generate a random sequence with the desired GC content
        miRNA = generate_sequence_with_gc_content(length, gc_min, gc_max)
        
        # Check if the seed region of the miRNA is complementary to the Stem IIB site
        if not check_seed_complementarity(miRNA, complemented_stem_iib):
            continue  # Skip if not complementary
        
        # Check if the miRNA has complementary sequences to other regions of the RRE
        if not check_off_target_complementarity(miRNA, RRE_SEQUENCE, STEM_IIB_SITE):
            continue  # Skip if off-target complementarity is detected
        
        # Check if the miRNA has a valid secondary structure
        if not simulate_secondary_structure(miRNA):
            continue  # Skip if secondary structure is invalid
        
        # If all constraints are satisfied, add the miRNA to the list
        valid_sequences.append(miRNA)
        
        # Print progress
        print(f"Generated {len(valid_sequences)} valid sequences in {attempts} attempts.")
    
    return valid_sequences

# Generate miRNA sequences
synthetic_mirnas = generate_mirnas(100, SEQUENCE_LENGTH_MIN, SEQUENCE_LENGTH_MAX, GC_CONTENT_MIN, GC_CONTENT_MAX)

# Specify the directory and file name
save_directory = r"E:\my_deep_learning_project\dataset"  # Replace with your desired directory
file_name = "mirna_data.csv"
file_path = os.path.join(save_directory, file_name)

# Save the CSV file
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Sequence"])  # Header
    for seq in synthetic_mirnas:
        writer.writerow([seq])

print(f"CSV file saved at: {file_path}")
