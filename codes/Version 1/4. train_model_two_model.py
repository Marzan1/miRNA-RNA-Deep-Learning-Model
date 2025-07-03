import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
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

# Step 2: Define the Models
class miRNAModelCNN(nn.Module):
    def __init__(self):
        super(miRNAModelCNN, self).__init__()
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

class miRNAModelLSTM(nn.Module):
    def __init__(self):
        super(miRNAModelLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 3)  # Output: binding affinity, specificity, stability

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM layer
        x = x[:, -1, :]       # Take the last time step's output
        x = self.fc(x)
        return x

# Step 3: Define the train_model function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50, patience=5):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
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

        # Evaluate on the validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

    return train_losses, val_losses

# Step 4: Hyperparameter Tuning
def hyperparameter_tuning(model_class, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor):
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    best_loss = float('inf')
    best_params = {}

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Testing lr={lr}, batch_size={batch_size}")
            
            # Create DataLoader with current batch size
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model, optimizer, and scheduler
            model = model_class()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
            
            # Train the model using the train_model function
            train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50)
            
            # Check if this is the best model
            if min(val_losses) < best_loss:
                best_loss = min(val_losses)
                best_params = {'lr': lr, 'batch_size': batch_size}

    print(f"Best Parameters: {best_params}, Best Validation Loss: {best_loss}")
    return best_params

# Perform hyperparameter tuning for CNN
print("Hyperparameter Tuning for CNN")
best_cnn_params = hyperparameter_tuning(miRNAModelCNN, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

# Perform hyperparameter tuning for LSTM
print("Hyperparameter Tuning for LSTM")
best_lstm_params = hyperparameter_tuning(miRNAModelLSTM, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

# Step 5: Train and Evaluate the Final Model
def train_final_model(model_class, best_params, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    # Create DataLoader with the best batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
    
    # Initialize model, optimizer, and scheduler
    model = model_class()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Train the model using the train_model function
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=50)
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).detach().numpy()
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Final Model - MAE: {mae}, R²: {r2}")
    
    return model, train_losses, val_losses

# Train and evaluate the final CNN model
print("Training Final CNN Model")
final_cnn_model, cnn_train_losses, cnn_val_losses = train_final_model(miRNAModelCNN, best_cnn_params, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

# Train and evaluate the final LSTM model
print("Training Final LSTM Model")
final_lstm_model, lstm_train_losses, lstm_val_losses = train_final_model(miRNAModelLSTM, best_lstm_params, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

# Step 4: Cross-Validation (Basic Example)
def cross_validate(model_class, X, y, n_splits=5, epochs=50, lr=0.001):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Initialize model, loss, optimizer, and scheduler
        model = model_class()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        # Train the model
        train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val_tensor).detach().numpy()
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            fold_results.append((mae, r2))

        print(f"Fold {fold+1} - MAE: {mae}, R²: {r2}")

    return fold_results

# Step 5: Run Cross-Validation for CNN and LSTM
print("Cross-Validating CNN Model")
cnn_results = cross_validate(miRNAModelCNN, X, labels)

print("Cross-Validating LSTM Model")
lstm_results = cross_validate(miRNAModelLSTM, X, labels)

# Step 6: Visualize Results
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    
    # Plot each label separately
    for i, label in enumerate(['Binding Affinity', 'Specificity', 'Stability']):
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, label=label)
        plt.plot([min(y_true[:, i]), max(y_true[:, i])], 
                 [min(y_true[:, i]), max(y_true[:, i])], 
                 color='red', linestyle='--')
    
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.legend()
    plt.show()

# Example visualization for the last fold of CNN
model = miRNAModelCNN()
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).detach().numpy()
plot_predictions(y_test, y_pred, "CNN Predictions vs True Values")

# Example visualization for the last fold of LSTM
model = miRNAModelLSTM()
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).detach().numpy()
plot_predictions(y_test, y_pred, "LSTM Predictions vs True Values")

# Predict properties for all sequences
model.eval()
with torch.no_grad():
    X_all_tensor = torch.tensor(X, dtype=torch.float32)  # All sequences
    y_pred_all = model(X_all_tensor).detach().numpy()    # Predicted properties

# Add predictions to the original dataframe
data['Predicted_Binding_Affinity'] = y_pred_all[:, 0]
data['Predicted_Specificity'] = y_pred_all[:, 1]
data['Predicted_Stability'] = y_pred_all[:, 2]

# Sort by a specific property (e.g., binding affinity)
best_mirnas = data.sort_values(by='Predicted_Binding_Affinity', ascending=False)
print(best_mirnas.head())  # Show top 5 miRNAs with highest binding affinity

# Save the trained models
torch.save(final_cnn_model.state_dict(), 'cnn_model.pth')
torch.save(final_lstm_model.state_dict(), 'lstm_model.pth')
