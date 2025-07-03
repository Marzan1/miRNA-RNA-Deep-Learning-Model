import csv
import os
import random
from Bio import SeqIO

# Parameters
GC_CONTENT_MIN = 0.50  # Minimum GC content
GC_CONTENT_MAX = 0.60  # Maximum GC content
SEQUENCE_LENGTH_MIN = 21  # Minimum length of miRNA
SEQUENCE_LENGTH_MAX = 25  # Maximum length of miRNA
NUCLEOTIDES = ['A', 'U', 'G', 'C']  # RNA nucleotides

# Target region parameters
STEM_IIB_SITE = "AGGUGGU"
TARGET_REGION_START = RRE_SEQUENCE.find(STEM_IIB_SITE) - 30  # 30nt before
TARGET_REGION_END = RRE_SEQUENCE.find(STEM_IIB_SITE) + len(STEM_IIB_SITE) + 30  # 30nt after
TARGET_REGION = RRE_SEQUENCE[TARGET_REGION_START:TARGET_REGION_END]

# Biological factors to consider (replace with real data)
MIRNA_BINDING_AFFINITY = "x_data"  # Experimental KD values
MIRNA_EXPRESSION_LEVEL = "x_data"  # From RNA-seq
MIRNA_CONSERVATION = "x_data"     # PhyloP scores
MIRNA_SECONDARY_STRUCTURE = "x_data"  # From RNAfold

# Load real human miRNAs
def load_human_mirnas(file_path):
    return [str(record.seq).replace('T', 'U') 
            for record in SeqIO.parse(file_path, "fasta")
            if SEQUENCE_LENGTH_MIN <= len(record.seq) <= SEQUENCE_LENGTH_MAX]

# Enhanced sequence validation
def validate_mirna(mirna):
    # GC content check
    gc_content = (mirna.count('G') + mirna.count('C')) / len(mirna)
    if not GC_CONTENT_MIN <= gc_content <= GC_CONTENT_MAX:
        return False
    
    # Seed complementarity (positions 2-8)
    seed = mirna[1:8]
    if not any(seed[i] == complement[TARGET_REGION[i]] for i in range(7)):
        return False
    
    # Secondary structure check
    if not is_valid_structure(mirna):
        return False
        
    return True

# Generate dataset with real miRNAs
def create_dataset(input_file, output_file):
    mirnas = load_human_mirnas(input_file)
    validated = []
    
    for mirna in mirnas:
        if validate_mirna(mirna):
            # Calculate additional features
            features = {
                'sequence': mirna,
                'binding_affinity': predict_affinity(mirna),  # x_data
                'conservation_score': get_conservation(mirna), # x_data
                'structure': predict_structure(mirna)  # x_data
            }
            validated.append(features)
    
    # Save with all features
    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=validated[0].keys())
        writer.writeheader()
        writer.writerows(validated)

# Data splitting function
def split_dataset(csv_path):
    df = pd.read_csv(csv_path)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.25, random_state=42)  # 60/20/20 split
    return train, val, test

if __name__ == "__main__":
    # Path setup
    input_path = r"E:\my_deep_learning_project\dataset\Human_miRNA.fasta" 
    output_path = r"E:\my_deep_learning_project\dataset\mirna_features.csv"
    
    # Create enhanced dataset
    create_dataset(input_path, output_path)
    
    # Split data
    train, val, test = split_dataset(output_path)
    print(f"Dataset split complete - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")