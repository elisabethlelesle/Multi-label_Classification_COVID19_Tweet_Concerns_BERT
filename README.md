# Multi-label_Classification_COVID19_Tweet_Concerns_BERT
Applying BERT-based models to predict various specific anti-vaccine concerns from COVID-19 anti-vaccine tweets in a multi-label setting

______

This is a well-structured Python script designed to perform multi-label classification on tweets about vaccine-related content using the `transformers` library. Below is a high-level explanation of its core components, along with insights into how it processes the `train.json`, `val.json`, and `test.json` files:

---

### **1. Overview of the Script**
- **Model:** Utilizes `roberta-large` from Hugging Face's `transformers` library for sequence classification.
- **Loss Function:** Implements a custom **Focal Loss**, which penalizes incorrect predictions more heavily to address class imbalance.
- **Hyperparameter Tuning:** Includes dropout regularization, learning rate scheduling, and weight decay.
- **Metrics:** Evaluates model performance using **macro F1-score** and plots loss and F1 trends over epochs.

---

### **2. Dataset Explanation**
The input JSON files (`train.json`, `val.json`, and `test.json`) are structured as:
- **`train.json`** and **`val.json`**: Contain labeled data where each tweet is assigned one or more labels (e.g., "ineffective," "pharma"). Labels are provided as dictionaries with details about specific terms in the text that correspond to the label.
- **`test.json`**: Unlabeled data used for prediction.

Each entry has:
- **`ID`**: A unique identifier for the tweet.
- **`tweet`**: The text of the tweet.
- **`labels`**: (only in `train.json` and `val.json`) Contains label categories and spans of text corresponding to each label.

---
### **3. Installation and Setup

To reproduce the results or test the model, follow these steps:

**Clone the Repository**
git clone <repository_url>
cd <repository_directory>

**Set Up Python Environment** 
It is recommended to use a virtual environment:

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

**Install Dependencies** 
Install the required libraries:

pip install -r requirements.txt

**Load the Dataset** 
Ensure the train.json, val.json, and test.json files are in the root directory. Modify the file paths in model5.py if needed.
Run the best-performing model:

python model5.py

This will:
* Load the training and validation datasets.
* Train the model on train.json.
* Evaluate the model on val.json and save the outputs (including the macro F1 score).
* Results and Performance Analysis

The best-performing model, model5.py, achieved a macro F1 score of 0.6652. 

---

### **4. Components of the Script**

#### **4.1 Data Preprocessing**
The `TweetDataset` class tokenizes the tweet text using the `AutoTokenizer` from `transformers`. For labeled datasets:
- Creates a **multi-hot encoding** tensor for the labels (`label_tensor`) corresponding to `label_to_index`.

---

#### **4.2 Model Training**
The script trains a `roberta-large` model using:
- **Loss Function:** The custom Focal Loss balances learning across classes by focusing on hard-to-classify examples.
- **Optimizer:** AdamW with weight decay to regularize weights.
- **Learning Rate Scheduler:** Linearly warms up and decays learning rates over epochs.

---

#### **4.3 Model Evaluation**
- **Validation Macro F1-Score:** Measures overall model performance across all labels.
- **Confusion Matrix Analysis:** Provides detailed True Positive (TP), False Positive (FP), False Negative (FN), and True Negative (TN) counts for each label to diagnose misclassification trends.

---

#### **4.4 Testing and Predictions**
- Predicts labels for tweets in `test.json` using a threshold (`label_thresholds`) for each category.
- Saves results as a **CSV file** (`submission.csv`).

---

### **5. Key Functions**
#### **Training and Validation**
- `train_epoch`: Executes a single epoch of training, returning the average loss.
- `val_f1_score`: Computes the macro F1-score on the validation set.

#### **Misclassification Analysis**
- `analyze_confusion_matrix`: Computes confusion matrices for each label to analyze prediction strengths and weaknesses.

#### **Predictions**
- `predict_with_label_thresholds`: Generates predictions on the test set using label-specific thresholds.

---

### **6. Execution Flow**
1. **Load Data:** Reads the JSON files for training, validation, and testing.
2. **Model Initialization:** Loads a pre-trained `roberta-large` model and tokenizer.
3. **Hyperparameter Tuning:** Adjusts dropout rates, learning rates, etc., for optimal performance.
4. **Training and Validation:** Iterates over multiple epochs, calculating loss and F1-score.
5. **Evaluation:** Analyzes misclassifications and saves predictions for the test set.

---

### **7. Improvements to Consider**
- **Threshold Tuning:** Instead of using a fixed threshold (e.g., 0.5), dynamically tune thresholds for each label based on validation performance.
- **Data Augmentation:** Enhance training data using techniques like synonym replacement or back-translation.
- **Early Stopping:** Terminate training if validation F1-score stops improving.
