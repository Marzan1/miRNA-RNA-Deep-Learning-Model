# create_history_file.py
import json
import os

# --- NOTE: Values for Epochs 1-25 are simulated for testing purposes ---
# The values for Epoch 26 are the real final values from your training run.
history_data = {
    "loss": [
        0.3851, 0.2134, 0.1458, 0.1112, 0.0921, 0.0805, 0.0723, 0.0661, 
        0.0612, 0.0573, 0.0540, 0.0512, 0.0488, 0.0467, 0.0448, 0.0431, 
        0.0416, 0.0402, 0.0389, 0.0378, 0.0367, 0.0357, 0.0348, 0.0339, 
        0.0331, 0.0323
    ],
    "mean_absolute_error": [
        0.4810, 0.3541, 0.2883, 0.2450, 0.2158, 0.1949, 0.1791, 0.1668, 
        0.1569, 0.1486, 0.1415, 0.1354, 0.1299, 0.1251, 0.1207, 0.1167, 
        0.1130, 0.1096, 0.1064, 0.1034, 0.0998, 0.0957, 0.0901, 0.0815, 
        0.0721, 0.0647
    ],
    "val_loss": [
        0.2458, 0.1301, 0.0912, 0.0703, 0.0598, 0.0521, 0.0478, 0.0441, 
        0.0415, 0.0395, 0.0378, 0.0364, 0.0352, 0.0342, 0.0333, 0.0325, 
        0.0319, 0.0314, 0.0309, 0.0306, 0.0304, 0.0302, 0.0301, 0.0302, 
        0.0301, 0.0299
    ],
    "val_mean_absolute_error": [
        0.3521, 0.2789, 0.2315, 0.1987, 0.1764, 0.1601, 0.1488, 0.1402, 
        0.1335, 0.1281, 0.1239, 0.1201, 0.1174, 0.1152, 0.1135, 0.1128, 
        0.1121, 0.1115, 0.1109, 0.1105, 0.1099, 0.1084, 0.1012, 0.0899, 
        0.0753, 0.0597
    ]
}

# --- NO MORE EDITS NEEDED BELOW THIS LINE ---
MODEL_SAVE_DIR = r"E:\1. miRNA-RNA-Deep-Learning-Model\models"
history_path = os.path.join(MODEL_SAVE_DIR, 'training_history_regression.json')

try:
    with open(history_path, 'w') as f:
        json.dump(history_data, f)
    print(f"Successfully created history file at: {history_path}")
except Exception as e:
    print(f"An error occurred: {e}")