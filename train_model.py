import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Data Preparation
# Specify the full path to the CSV file
csv_path = r"E:\my_deep_learning_project\dataset\mirna_data.csv"

# Load miRNA sequences from CSV
data = pd.read_csv(csv_path)
sequences = data['Sequence'].tolist()  # Ensure this is a list of strings

# Generate synthetic labels (replace with real data if available)
binding_affinity = np.random.uniform(0, 1, len(sequences))  # Random binding affinity (0 to 1)
specificity = np.random.uniform(0, 1, len(sequences))       # Random specificity (0 to 1)
stability = np.random.uniform(0, 1, len(sequences))         # Random stability (0 to 1)

# Combine labels
labels = np.array([binding_affinity, specificity, stability]).T

# Define the fixed length for all sequences
FIXED_LENGTH = 25  # Adjust based on your maximum sequence length

# Function to pad sequences
def pad_sequence(sequence, length, pad_value=[0, 0, 0, 0]):
    if len(sequence) < length:
        # Pad the sequence with the pad_value
        sequence.extend([pad_value] * (length - len(sequence)))
    return sequence

# One-hot encode sequences and pad them to the same length
def one_hot_encode(sequence, length=FIXED_LENGTH):
    mapping = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    encoded = [mapping[nucleotide] for nucleotide in sequence]
    return pad_sequence(encoded, length)

# One-hot encode and pad all sequences
X = np.array([one_hot_encode(seq) for seq in sequences])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Step 2: Define the Model
class miRNAModel(nn.Module):
    def __init__(self):
        super(miRNAModel, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3)  # Input channels: 4 (A, U, G, C)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * 11, 64)  # Adjust based on sequence length after pooling
        self.fc2 = nn.Linear(64, 3)       # Output: binding affinity, specificity, stability
        self.dropout = nn.Dropout(0.5)    # Add dropout to prevent overfitting

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for Conv1d
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = miRNAModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # Add L2 regularization

# Step 3: Train the Model
train_losses = []
val_losses = []

for epoch in range(50):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Calculate average training loss for the epoch
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    # Evaluate on the test set (validation)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
        val_losses.append(val_loss)

    print(f"Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

# Step 4: Evaluate the Model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).detach().numpy()

# Calculate MAE and R²
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, R²: {r2}")

# Step 5: Visualize Results
plt.plot(range(1, 51), train_losses, label='Training Loss')
plt.plot(range(1, 51), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()