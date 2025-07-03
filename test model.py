# test_model.py
import torch
import numpy as np

# Define the model architectures (must match the training file)
class miRNAModelCNN(torch.nn.Module):
    def __init__(self):
        super(miRNAModelCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 32, kernel_size=3)
        self.pool = torch.nn.MaxPool1d(2)
        self.fc1 = torch.nn.Linear(32 * 11, 64)
        self.fc2 = torch.nn.Linear(64, 3)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class miRNAModelLSTM(torch.nn.Module):
    def __init__(self):
        super(miRNAModelLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=4, hidden_size=64, batch_first=True)
        self.fc = torch.nn.Linear(64, 3)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Function to preprocess new sequences (must match the training file)
def one_hot_encode(sequence, length=25):
    mapping = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    encoded = [mapping[nucleotide] for nucleotide in sequence]
    return pad_sequence(encoded, length)

def pad_sequence(sequence, length, pad_value=[0, 0, 0, 0]):
    if len(sequence) < length:
        sequence.extend([pad_value] * (length - len(sequence)))
    return sequence

# Load the saved models
loaded_cnn_model = miRNAModelCNN()
loaded_cnn_model.load_state_dict(torch.load('cnn_model.pth'))
loaded_cnn_model.eval()

loaded_lstm_model = miRNAModelLSTM()
loaded_lstm_model.load_state_dict(torch.load('lstm_model.pth'))
loaded_lstm_model.eval()

# Example new miRNA sequences
new_sequences = [
    "AUGCAUGAUCGUA",  # Example sequence 1
    "UACGUACGUACGU",  # Example sequence 2
    "CGUACGUACGUAC",  # Example sequence 3
    # Add more sequences here...
]

# Preprocess the new sequences (one-hot encode and pad)
X_new = np.array([one_hot_encode(seq) for seq in new_sequences])

# Convert to PyTorch tensor
X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

# Make predictions using the loaded CNN model
with torch.no_grad():
    predictions_cnn = loaded_cnn_model(X_new_tensor).numpy()

# Make predictions using the loaded LSTM model
with torch.no_grad():
    predictions_lstm = loaded_lstm_model(X_new_tensor).numpy()

# Function to rank miRNAs and select the top N
def rank_mirnas(sequences, predictions, top_n=10, property_index=0):
    """
    Rank miRNAs based on a specific property and select the top N.

    Args:
        sequences (list): List of miRNA sequences.
        predictions (np.array): Predicted properties for the sequences.
        top_n (int): Number of top miRNAs to select.
        property_index (int): Index of the property to rank by (0: binding affinity, 1: specificity, 2: stability).

    Returns:
        list: Top N miRNA sequences and their predicted properties.
    """
    # Combine sequences with their predictions
    ranked_data = list(zip(sequences, predictions))
    
    # Sort by the specified property (descending order)
    ranked_data.sort(key=lambda x: x[1][property_index], reverse=True)
    
    # Select the top N
    return ranked_data[:top_n]

# Rank miRNAs based on binding affinity (property_index=0)
top_n = 10  # Number of top miRNAs to select
top_cnn_mirnas = rank_mirnas(new_sequences, predictions_cnn, top_n=top_n, property_index=0)
top_lstm_mirnas = rank_mirnas(new_sequences, predictions_lstm, top_n=top_n, property_index=0)

# Print the top miRNAs
print("Top miRNAs (CNN Model):")
for seq, pred in top_cnn_mirnas:
    print(f"Sequence: {seq}, Binding Affinity: {pred[0]:.4f}, Specificity: {pred[1]:.4f}, Stability: {pred[2]:.4f}")

print("\nTop miRNAs (LSTM Model):")
for seq, pred in top_lstm_mirnas:
    print(f"Sequence: {seq}, Binding Affinity: {pred[0]:.4f}, Specificity: {pred[1]:.4f}, Stability: {pred[2]:.4f}")