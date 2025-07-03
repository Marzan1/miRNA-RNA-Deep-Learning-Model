#Preparing data for Deep learning

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    # Load miRNA sequences from CSV
    csv_path = r"E:\my_deep_learning_project\dataset\mirna_data.csv"
    data = pd.read_csv(csv_path)
    sequences = data['Sequence'].tolist()

    # Generate synthetic labels
    binding_affinity = np.random.uniform(0, 1, len(sequences))
    specificity = np.random.uniform(0, 1, len(sequences))
    stability = np.random.uniform(0, 1, len(sequences))

    # Combine labels
    labels = np.array([binding_affinity, specificity, stability]).T

    # One-hot encode and pad sequences
    def one_hot_encode(sequence, length=25):
        mapping = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
        encoded = [mapping[nucleotide] for nucleotide in sequence]
        if len(encoded) < length:
            encoded.extend([[0, 0, 0, 0]] * (length - len(encoded)))
        return encoded

    X = np.array([one_hot_encode(seq) for seq in sequences])

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
