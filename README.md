# miRNA-RRE Binding Prediction using Deep Learning

## Overview
This project provides a comprehensive deep learning pipeline designed to predict the binding interactions between microRNAs (miRNAs) and Rev Response Element (RRE) sequences, incorporating additional biological features including the REV protein sequence. Understanding these interactions is vital for research in gene regulation, viral mechanisms (e.g., HIV-1 Rev protein's role in viral replication), and therapeutic development.

The pipeline covers the entire machine learning workflow: from meticulous data preparation and feature engineering using raw biological data, to the construction and training of a specialized deep learning model, and finally, thorough evaluation and visualization of the model's performance.

## Features
* **Automated Data Preparation:** Scripts to process diverse raw biological data (miRNA and RRE sequences, conservation scores, binding affinity data, structural information, and REV protein sequences) into a structured, unified dataset ready for deep learning. This includes handling various data formats and feature encoding.
* **Multi-Input Deep Learning Model (TensorFlow/Keras):** A custom-designed deep learning architecture built with **TensorFlow/Keras**, capable of simultaneously processing heterogeneous input types:
    * Scalar numerical features (e.g., GC content, DeltaG, conservation, affinity).
    * Numerical sequence structure representations (e.g., dot-bracket notation encoded numerically).
    * Raw sequence strings (miRNA, RRE, REV) that are dynamically vectorized and embedded within the model for feature extraction (e.g., using Conv1D layers).
* **Model Training and Evaluation:** Functionality to train the deep learning model on the prepared dataset, monitor performance (accuracy, precision, recall, AUC), and save the trained model. Includes comprehensive evaluation on a held-out test set.
* **Prediction on New Data:** Dedicated script to load a trained model and make predictions on new, unseen miRNA, RRE, and REV sequences.
* **Comprehensive Evaluation & Visualization:** Tools to analyze the trained model's performance on a held-out test set, including:
    * Classification Reports
    * Confusion Matrices
    * Receiver Operating Characteristic (ROC) Curves
    * Precision-Recall (PR) Curves
    * Visualization of Training and Validation History (Accuracy and Loss over epochs)

## Getting Started
Follow these steps to set up the project locally and run the pipeline.

### Prerequisites
* Python 3.8 or higher
* pip (Python package installer)
* git (for cloning the repository)
* **ViennaRNA Package:** Required for the `RNAfold` command used in structure prediction. Ensure it's installed and `RNAfold` is accessible from your system's PATH. You can usually find installation instructions on the ViennaRNA website.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Marzan1/miRNA-RNA-Deep-Learning-Model.git](https://github.com/Marzan1/miRNA-RNA-Deep-Learning-Model.git)
    cd miRNA-RNA-Deep-Learning-Model
    ```
2.  **Create and activate a virtual environment (highly recommended):**
    ```bash
    python -m venv venv_dl # Or choose a name like venv_python313
    ```
    * **On Windows:**
        ```bash
        .\venv_dl\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv_dl/bin/activate
        ```
3.  **Install the required Python packages:**
    This project uses TensorFlow/Keras and other common data science libraries.
    ```bash
    pip install tensorflow numpy pandas scikit-learn matplotlib biopython joblib
    ```
    * **Note:** It's highly recommended to create a `requirements.txt` file (e.g., using `pip freeze > requirements.txt` after successful installation) to manage exact dependencies.

## Project Structure (Focus on Version 2)

miRNA-RNA-Deep-Learning-Model/
├── codes/
│   ├── Version 1/                    # Older or alternative code versions (not the primary focus for current usage)
│   └── Version 2/                    # Primary and most up-to-date code version
│       ├── 1. dataset_preparation.py # comprehensive raw data preprocessing and numerical feature generation.
│       ├── 2. deep_learning_data_preparation.py # Likely involved in preparing data specifically for DL model input (e.g., X_train_*.npy).
│       ├── 3. model_building.py      # Main script for defining, training, and evaluating the deep learning model.
│       ├── 4. predict_on_new_data.py # Script for loading a trained model and making predictions on new, unseen data.
│       ├── 5. analysis_and_plotting.py # For generating analysis reports (e.g., classification report, confusion matrix).
│       └── 6. present graph.py       # Script for generating and displaying various plots (ROC, PR, training history).
├── Prepared Dataset/                 # NEW FOLDER: Contains raw or initially prepared datasets from '1. dataset_preparation.py'
│   ├── mirna_data.csv                # Example: raw miRNA data CSV
│   ├── prepared_miRNA_RRE_dataset.csv # Example: combined and prepared dataset
│   └── prepared_miRNA_dataset.csv    # Example: prepared miRNA-only dataset
├── Processed_for_DL/                 # NEW FOLDER: Contains numerically processed data (Numpy arrays) ready for deep learning models, output from '2. deep_learning_data_preparation.py'
│   ├── X_test_mirna_seq.npy          # Example: processed test data for miRNA sequences
│   ├── X_test_mirna_struct.npy       # Example: processed test data for miRNA structures
│   ├── X_test_numerical.npy          # Example: processed test data for numerical features
│   ├── X_test_rev_seq.npy            # Example: processed test data for reverse sequences
│   ├── X_test_rre_seq.npy            # Example: processed test data for RRE sequences
│   ├── X_train_mirna_seq.npy         # Example: processed training data for miRNA sequences
│   ├── X_train_mirna_struct.npy      # Example: processed training data for miRNA structures
│   ├── X_train_numerical.npy         # Example: processed training data for numerical features
│   ├── X_train_rev_seq.npy           # Example: processed training data for reverse sequences
│   ├── X_train_rre_seq.npy           # Example: processed training data for RRE sequences
│   ├── minmax_scaler.pkl             # Example: saved scaler object
│   ├── y_test.npy                    # Example: test labels
│   └── y_train.npy                   # Example: training labels
├── models/                           # Directory intended for saving trained Keras models (.keras files).
│   ├── cnn_model.pth                 # Older PyTorch CNN model (from Version 1, moved here for organization).
│   ├── lstm_model.pth                # Older PyTorch LSTM model (from Version 1, moved here for organization).
│   └── miRNA_RRE_REV_prediction_model.keras # Main trained TensorFlow/Keras model.
├── Notes/                            # Contains miscellaneous project notes and research findings.
│
├── .git/                             # Git repository files (hidden).
├── .gitignore                        # Specifies files/directories ignored by Git (e.g., dataset/, virtual environments).
└── README.md                         # This README file.


* **Note on `.pth` files:** `cnn_model.pth` and `lstm_model.pth` are older PyTorch models from previous development versions (likely `Version 1`). They have been moved into the `models/` directory for better organization but are not used by the current `Version 2` TensorFlow/Keras workflow. Models saved by `3. model_building.py` will have a `.keras` extension.

## Usage (Using `codes/Version 2`)
Follow these steps in the specified order to run the full pipeline using the latest scripts in `codes/Version 2`.

**Important Note on Data Storage:** The `dataset/` folder is listed in `.gitignore` and is not tracked by this repository. The raw and processed datasets are too large to be included directly in this Git repository. **Please download them from the following OneDrive link and place them in your local `dataset/` directory:**

**[Download the Dataset from OneDrive here](https://1drv.ms/f/c/61c8f613658b59de/Enm_X_4FpYlPlOaPvTg6OhIBKfSV2rwEhxXTRPcO7TIC4Q?e=xPSVqw)**.
*This OneDrive folder contains both the `Raw Data/` and `Processed_for_DL/` subdirectories. Please download its contents and place them directly into your local `dataset/` folder.*

### 1. Obtain and Structure Raw Data

Once you have downloaded the dataset from OneDrive, ensure your local directory structure matches what the scripts expect:

* **Create the Data Directories:**
    * Create a `dataset` folder in your project root (it will be ignored by Git).
    * Inside `dataset/`, you should have `Raw Data/` (if the raw data is provided separately) and `Processed_for_DL/` (where intermediate NumPy arrays will be placed, or if the processed data is already on OneDrive).
    ```bash
    mkdir -p dataset/Raw\ Data # Create if raw data will be processed by scripts
    mkdir -p dataset/Processed_for_DL # Create if not already downloaded
    ```
* **Place the downloaded data:** Extract the contents of the OneDrive download into your `dataset/` folder, ensuring `Raw Data/` and `Processed_for_DL/` are populated as needed.
* **Optional: Convert FASTA files:** If your raw sequences are not in standard FASTA format, you might use a utility script if you have one available, or convert them manually. (Note: The `1. fa to fasta converstion.py` script is no longer explicitly listed in your `Version 2` structure).

### 2. Prepare the Dataset

This step processes your raw data into the necessary NumPy arrays (`.npy` files) in `dataset/Processed_for_DL/`, which are then used as input for model training. It also saves the `MinMaxScaler` for numerical features.

* **Addressing Data Bias (Crucial for Model Performance):**
    A common challenge in biological interaction datasets is class imbalance, where positive interactions (label = 1) might significantly outnumber negative interactions (label = 0). To train a robust and meaningful classification model, it is essential to balance the dataset by ensuring a sufficient number of label = 0 instances.
    **Before running `1. dataset_preparation.py`, you must review and potentially modify it to implement a strategy for generating negative samples.** This could involve:
    * **Generating Random Non-Target Sequences:** For existing miRNAs, create random RNA sequences (of appropriate length and base composition) that are highly unlikely to be true RRE targets. Pair these with miRNAs and assign label = 0.
    * **Shuffling Known Targets:** Take existing RRE target sequences and randomly shuffle their bases to destroy binding motifs, then pair them with miRNAs and assign label = 0.
    * **Incorporating Known Non-Interactions:** If you have access to databases or literature that explicitly identify non-interacting miRNA-RRE pairs, integrate them into your dataset.

* **Run the data preparation script:**
    ```bash
    python codes/Version\ 2/1.\ dataset_preparation.py
    ```
    This script will generate `X_train_*.npy`, `y_train.npy`, `X_test_*.npy`, `y_test.npy`, and `minmax_scaler.pkl` within the `dataset/Processed_for_DL/` directory.

### 3. Build Model, Train, and Evaluate

This step defines the deep learning model architecture, trains it using the prepared dataset, evaluates its performance, and saves the trained model.

* **Before running:**
    * Review `codes/Version 2/3. model_building.py` to understand the multi-input model architecture, training parameters (epochs, batch size, learning rate), and evaluation metrics. Adjust as needed.
* **Run the model building and training script:**
    ```bash
    python codes/Version\ 2/3.\ model_building.py
    ```
    This script will:
    * Load the processed `.npy` data from `dataset/Processed_for_DL/`.
    * Define and compile the TensorFlow/Keras model.
    * Train the model using the loaded training data.
    * Evaluate the trained model on the test data and print metrics (Loss, Accuracy, Precision, Recall, AUC).
    * Print a Classification Report and Confusion Matrix.
    * Save the trained model (e.g., `miRNA_RRE_REV_prediction_model.keras`) in the `models/` directory.

### 4. Make Predictions on New Data

After a model has been trained and saved, you can use `4. predict_on_new_data.py` to make predictions on individual new sequences.

* **Before running:**
    * Ensure a trained model (`miRNA_RRE_REV_prediction_model.keras`) exists in your `models/` directory.
    * Review `codes/Version 2/4. predict_on_new_data.py`. This script includes example new sequences. You will need to modify the `new_mirna_seq_X`, `new_rre_seq_X`, and `new_rev_seq_X` variables within this script to provide your actual sequences for prediction.
* **Run the prediction script:**
    ```bash
    python codes/Version\ 2/4.\ predict_on_new_data.py
    ```
    This script will:
    * Load the saved `MinMaxScaler` and the trained `miRNA_RRE_REV_prediction_model.keras`.
    * Preprocess the new input sequences (calculating GC content, predicting structure, encoding sequences, and scaling numerical features).
    * Output the predicted probability and binary class (High Affinity/Low Affinity) for the provided new sequences.

### 5. Analyze Results and Generate Plots

The `6. present graph.py` script can be used to generate various diagnostic plots for model performance visualization.

* **Important Note on Plots (DEMO_MODE):**
    The `codes/Version 2/6. present graph.py` script includes a `DEMO_MODE` flag.
    * If `DEMO_MODE = True` (default in the script): The script will generate synthetic `y_test` labels and `y_pred_proba` values that represent both 0 and 1 classes. This allows for the generation of ROC curves, Precision-Recall curves, and training history plots even if your actual dataset is highly imbalanced. These plots are for demonstration purposes ONLY and do NOT reflect the true performance of your model on an imbalanced dataset. They are useful for presentation but should not be used for scientific conclusions until your data bias is fully addressed.
    * If `DEMO_MODE = False`: The script will attempt to use your actual test data and real training history. If your `prepared_miRNA_RRE_dataset.csv` (or the underlying data used to create `y_test.npy`) still contains only a single class (e.g., all label = 1), the ROC and Precision-Recall plots will likely fail or produce misleading results due to the absence of both classes.
* **Run the analysis script:**
    ```bash
    python codes/Version\ 2/6.\ present\ graph.py
    ```
    This script will:
    * If `DEMO_MODE` is True (or if your real data is balanced), it will generate and display:
        * A Classification Report
        * A Confusion Matrix
        * An ROC Curve
        * A Precision-Recall Curve
        * Training & Validation Accuracy/Loss Plots
    * All generated plots (ROC, PR, History) will be automatically saved as PNG files within the `dataset/Processed_for_DL/` directory (based on the `DATA_PATH` constant in the script). They will also appear in separate pop-up display windows.

## Configuration
Key configurable parameters, such as directory paths, file names, maximum sequence lengths, and affinity thresholds, are defined as constants at the top of each Python script in `codes/Version 2/`. Please review and modify these constants as per your specific project setup and data requirements before running the scripts.

## Dataset Information
The project is designed to integrate various features for each miRNA-RRE-REV interaction pair. The data preparation scripts (e.g., `1. dataset_preparation.py`) expect raw input data that, when processed, can yield the following features for the deep learning model:

* `mirna_id`: Identifier for the miRNA.
* `sequence`: miRNA sequence (RNA).
* `gc_content`: GC content of the miRNA sequence.
* `dg`: Binding free energy (Delta G) of the miRNA.
* `conservation`: Conservation score of the miRNA.
* `affinity`: Binding affinity score between miRNA and RRE (used to derive the label).
* `structure_vector`: Numerical representation of the miRNA's secondary structure.
* `rre_id`: Identifier for the RRE.
* `rre_sequence`: RRE sequence (RNA).
* `region`: Categorical region of the RRE (e.g., "Primary_RRE_Binding_Region").
* `rev_id`: Identifier for the REV protein.
* `rev_sequence`: REV protein sequence.
* `label`: Binary classification label (0 for non-interaction, 1 for interaction) derived from the affinity score based on `AFFINITY_THRESHOLD`.

## Model Architecture (High-Level)
The deep learning model is a sophisticated multi-input architecture implemented using **TensorFlow/Keras**. It is designed to handle and learn from diverse input data types simultaneously:

* **Scalar & Structural Features:** These numerical inputs are fed into dedicated dense (fully connected) layers for initial processing.
* **Sequence Data (miRNA, RRE, REV):** Raw string sequences are expected to be pre-processed into a numerical representation (e.g., one-hot encoded) before being fed into specialized neural network layers such as Convolutional Neural Networks (CNNs) for capturing local patterns (as seen in `3. model_building.py`).
* **Categorical Region:** While not explicitly seen in the `3. model_building.py` snippet provided, the overall project description suggests categorical features might be handled, typically via embedding layers.
The outputs from these parallel processing branches are concatenated into a single feature vector, which is then fed into a series of final dense layers with activation functions leading to a binary classification output (predicting the likelihood of interaction) using a sigmoid activation.

## Addressing Data Bias (Current State & Ongoing Work)
As highlighted in the "Prepare the Dataset" section, the problem of class imbalance (specifically, a scarcity of label = 0 instances) is a significant challenge currently being addressed. An imbalanced dataset can lead to a biased model that performs poorly on the minority class and cannot reliably distinguish between interacting and non-interacting pairs.

**Current Approach:**
* The pipeline facilitates processing existing raw data and setting an affinity threshold for labeling.
* For immediate visualization needs, `6. present graph.py` offers a `DEMO_MODE` to generate synthetic balanced data for plots, allowing for visual analysis even with an imbalanced real dataset.

**Ongoing & Future Work for Bias Mitigation:**
* **Robust Negative Sample Generation:** This remains the most critical area. Future work will focus on refining and implementing robust methods to systematically generate realistic negative (non-interacting) samples directly within the data preparation phase.
* **Advanced Imbalance Handling:** Explore techniques such as weighted loss functions during training, data augmentation for the minority class (e.g., SMOTE), or strategic undersampling of the majority class to further improve model performance on imbalanced datasets.
* **Data Augmentation for Sequences:** Investigate methods to augment sequence data (e.g., synonymous mutations, slight shifts) to increase diversity, particularly for the minority class.

## Development Versions
This repository contains two primary code versions within the `codes/` directory:
* `Version 1/`: Represents an earlier stage of development, containing scripts prefixed with `P_`.
* `Version 2/`: Represents the most recent and active development branch of the project, containing updated scripts (now prefixed with numbers). All usage instructions in this `README` refer to the `Version 2` scripts.

## Future Work
* Refine negative sample generation strategy for more realistic non-interactions.
* Implement hyperparameter tuning for optimal model performance across different datasets.
* Investigate and integrate more diverse biological features (e.g., epigenetic marks, cellular localization data).
* Explore more complex or state-of-the-art deep learning architectures.
* Develop robust cross-validation strategies to ensure model generalization.
* Potentially, develop a user-friendly interface or deployment mechanism for predictions.

## Contributing
Contributions, issues, and feature requests are highly welcome! If you have suggestions for improvements, encounter bugs, or wish to add new features, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact
[Abdullah Al Marzan / Marzan1] - [marzansust16@gmail.com]