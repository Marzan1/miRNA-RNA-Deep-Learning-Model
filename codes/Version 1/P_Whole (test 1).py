# =================================== Script: Best miRNA Predictor ===================================
# Description: World's best miRNA predictor targeting HIV RRE Stem IIB and other functional loops.
# Author: Marjan
# Folder Path: E:\my_deep_learning_project\dataset
# -----------------------------------------------------------------------------------------------------

# =================================== 1. INITIAL DATA PREPARATION ===================================
import os
import csv
import pandas as pd
import random
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SEQUENCE_LENGTH_MIN = 21
SEQUENCE_LENGTH_MAX = 25
GC_CONTENT_MIN = 0.50
GC_CONTENT_MAX = 0.60

DATASET_FOLDER = r"E:\\my_deep_learning_project\\dataset"
MIRNA_FASTA = os.path.join(DATASET_FOLDER, "Human_miRNA.fasta")
AFFINITY_PATH = os.path.join(DATASET_FOLDER, "interactions_human.microT.mirbase.txt")
CONSERVATION_PATH = os.path.join(DATASET_FOLDER, "Conserved_Family_Info.txt")

RRE_SEQUENCE = "GGGUGUGGAAAUCUCUGGGUUAGACCAGAUCTGAGCUGGGUUCUCUGGGCAGCCAGAGGUGGUCUUAGCCUUCUUGAAUCCUGGCCUCCUCCAGGAUCCCAGGGUUCAAAUCCCACUGGCCUUGGCUGAAGGGGCAGUAGUCCUUCUGAUUGGCCAGGCUGCCUUCUGCUCCUGCUGGCCAGGCAGGUGCUGGCCACUAGCUGGUGACUAGUGACUUGCUGAUAGGGUGGGCUAUUUUCCUACU"
KNOWN_LOOP_SITES = [("AGGUGGU", "Stem IIB"), ("GAAGGGGCA", "Loop I"), ("CCUUCUGAUU", "Loop II")]

TARGET_REGIONS = {}
for motif, name in KNOWN_LOOP_SITES:
    idx = RRE_SEQUENCE.find(motif)
    if idx != -1:
        start = max(idx - 20, 0)
        end = idx + len(motif) + 20
        TARGET_REGIONS[name] = RRE_SEQUENCE[start:end]

NUCLEOTIDES = ['A', 'U', 'G', 'C']
complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}

def calculate_gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

def validate_seed(mirna, region):
    seed = mirna[1:8]
    return any(seed[i] == complement.get(region[i], '') for i in range(7))

def predict_structure(seq):
    with open("temp_seq.fa", "w") as f:
        f.write(f">temp\n{seq}\n")
    result = subprocess.run(["RNAfold"], input=f">temp\n{seq}\n", 
                       capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    structure = lines[1].split(' ')[0]
    energy = float(lines[1].split('(')[-1].replace(")", ""))
    return structure, energy

def encode_structure_dot_bracket(dot):
    return [1 if x == '(' else -1 if x == ')' else 0 for x in dot.ljust(25, '.')[:25]]

def load_affinity():
    df = pd.read_csv(AFFINITY_PATH, sep='\t', header=None)
    return {row[0]: float(row[4]) for row in df.itertuples(index=False)}

def load_conservation():
    df = pd.read_csv(CONSERVATION_PATH)
    return dict(zip(df['miRNA'], df['score']))

def load_mirnas():
    return [str(r.seq).replace('T', 'U') for r in SeqIO.parse(MIRNA_FASTA, "fasta")
            if SEQUENCE_LENGTH_MIN <= len(r.seq) <= SEQUENCE_LENGTH_MAX]

def prepare_dataset(output_csv):
    mirnas = load_mirnas()
    affinity = load_affinity()
    conservation = load_conservation()
    data = []
    for mirna in mirnas:
        for region_name, region_seq in TARGET_REGIONS.items():
            if not validate_seed(mirna, region_seq): continue
            gc = calculate_gc_content(mirna)
            structure, dg = predict_structure(mirna)
            struct_vec = encode_structure_dot_bracket(structure)
            data.append({
                'sequence': mirna,
                'region': region_name,
                'gc_content': gc,
                'dg': dg,
                'conservation': conservation.get(mirna, 0.7),
                'affinity': affinity.get(mirna, 0.5),
                'structure_vector': struct_vec,
                'label': 1 if affinity.get(mirna, 0.5) > 0.7 else 0
            })
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    return df

# =================================== 2. DATASET TRANSFORMATION ===================================
class miRNADataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = {'A':0, 'U':1, 'G':2, 'C':3, 'N':4}
        seq_encoded = [encoding.get(nt, 4) for nt in row['sequence'].ljust(25, 'N')[:25]]
        structure_vec = row['structure_vector']
        features = [row['gc_content'], row['dg'], row['conservation'], row['affinity']]
        x = torch.tensor(seq_encoded + structure_vec + features, dtype=torch.float32)
        y = torch.tensor(row['label'], dtype=torch.float32)
        return x, y

# =================================== 3. TRAINING, TESTING, VALIDATION ===================================
class AttentionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 25, 1)
        out, _ = self.attn(x, x, x)
        pooled = out.mean(dim=1)
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(pooled)))).squeeze(1)

def train_model(train_loader, val_loader):
    model = AttentionModel(input_dim=33)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        print(f"Epoch {epoch+1} complete")
    return model

def evaluate_model(model, loader):
    all_y, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x)
            all_y.extend(y.tolist())
            all_pred.extend(preds.tolist())
    return np.array(all_y), np.array(all_pred)

# =================================== 4. GRAPH PREPARATION ===================================
def generate_graphs(y_true, y_pred):
    pred_bin = (y_pred > 0.5).astype(int)
    print(classification_report(y_true, pred_bin))
    plt.figure(); sns.histplot(y_pred); plt.title("Prediction Distribution"); plt.savefig("pred_dist.png")
    plt.figure(); sns.boxplot(x=pred_bin, y=y_pred); plt.title("Boxplot by Class"); plt.savefig("boxplot.png")
    plt.figure(); sns.heatmap(confusion_matrix(y_true, pred_bin), annot=True); plt.title("Confusion Matrix"); plt.savefig("conf_matrix.png")
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(); plt.plot(fpr, tpr); plt.title("ROC Curve"); plt.savefig("roc.png")
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(); plt.plot(recall, precision); plt.title("PR Curve"); plt.savefig("pr.png")
    plt.figure(); sns.violinplot(x=pred_bin, y=y_pred); plt.title("Violin Plot"); plt.savefig("violin.png")
    plt.figure(); sns.kdeplot(y_pred); plt.title("KDE Plot"); plt.savefig("kde.png")
    plt.figure(); plt.scatter(range(len(y_pred)), y_pred); plt.title("Scatter Plot"); plt.savefig("scatter.png")
    plt.figure(); sns.ecdfplot(y_pred); plt.title("ECDF Plot"); plt.savefig("ecdf.png")
    plt.figure(); plt.plot(sorted(y_pred)); plt.title("Sorted Prediction Plot"); plt.savefig("sorted.png")

# =================================== 5. MAIN EXECUTION ===================================
if __name__ == '__main__':
    output_csv = os.path.join(DATASET_FOLDER, "mirna_final_features.csv")
    df = prepare_dataset(output_csv)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

    train_loader = DataLoader(miRNADataset(train_df), batch_size=16, shuffle=True)
    val_loader = DataLoader(miRNADataset(val_df), batch_size=16)
    test_loader = DataLoader(miRNADataset(test_df), batch_size=16)

    model = train_model(train_loader, val_loader)
    y_true, y_pred = evaluate_model(model, test_loader)
    generate_graphs(y_true, y_pred)
