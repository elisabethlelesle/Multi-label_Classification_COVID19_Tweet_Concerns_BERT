import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

label_to_index = {
    "ineffective": 0,
    "unnecessary": 1,
    "pharma": 2,
    "rushed": 3,
    "side-effect": 4,
    "mandatory": 5,
    "country": 6,
    "ingredients": 7,
    "political": 8,
    "none": 9,
    "conspiracy": 10,
    "religious": 11
}

# Label-specific thresholds (tune these based on validation performance)
label_thresholds = [0.5] * len(label_to_index)  # Start with 0.5 for all labels

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tweet = self.data[index]['tweet']
        inputs = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        if 'labels' in self.data[index]:
            labels = self.data[index]['labels']
            label_tensor = torch.zeros(len(label_to_index))
            for category, items in labels.items():
                if category in label_to_index:
                    label_index = label_to_index[category]
                    label_tensor[label_index] = 1
            return {
                'input_ids': inputs['input_ids'].flatten(),
                'attention_mask': inputs['attention_mask'].flatten(),
                'labels': label_tensor
            }
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten()
        }

def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    model = model.train()
    losses = []
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs.logits, labels)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return np.mean(losses)

def val_f1_score(model, data_loader, device):
    model = model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= label_thresholds).astype(int)
            
            all_labels.extend(labels)
            all_preds.extend(preds)

    return f1_score(np.array(all_labels), np.array(all_preds), average='macro')

def analyze_confusion_matrix(model, data_loader, device, label_names):
    model = model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)  # Use 0.5 as threshold or adjust as needed
            
            all_labels.extend(labels)
            all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Initialize a dictionary to store confusion details for each label
    confusion_details = {label: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for label in label_names}

    for i, label in enumerate(label_names):
        # Compute confusion matrix for the specific label
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        confusion_details[label]["TP"] = tp
        confusion_details[label]["FP"] = fp
        confusion_details[label]["FN"] = fn
        confusion_details[label]["TN"] = tn

        # Print detailed report for each label
        print(f"Confusion Matrix for '{label}':")
        print(f"  True Positives (TP): {tp}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Negatives (TN): {tn}\n")

    # Summarize the categories that are frequently misclassified
    misclassifications = {}
    for label in label_names:
        fp = confusion_details[label]["FP"]
        fn = confusion_details[label]["FN"]
        misclassifications[label] = {"False Positives": fp, "False Negatives": fn}

    return misclassifications


def predict_with_label_thresholds(model, data_loader, device, label_thresholds):
    model = model.eval()
    predictions = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            binary_preds = np.array([(probs[:, i] >= label_thresholds[i]).astype(int) for i in range(len(label_thresholds))]).T
            predictions.append(binary_preds)
    return np.vstack(predictions)

def plot_hyperparameter_tuning(x_values, train_losses, val_f1_scores, param_name):
    plt.figure(figsize=(12, 5))

    # Plot for training loss
    plt.subplot(1, 2, 1)
    plt.plot(x_values, train_losses, marker='o')
    plt.title(f'Training Loss vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Training Loss')

    # Plot for validation F1 score
    plt.subplot(1, 2, 2)
    plt.plot(x_values, val_f1_scores, marker='o', color='orange')
    plt.title(f'Validation F1 Score vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Validation F1 Score')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load dataset
    with open('train.json') as f:
        train_data = json.load(f)
    with open('val.json') as f:
        val_data = json.load(f)
    with open('test.json') as f:
        test_data = json.load(f)

    #learning_rates = [1e-5, 2e-5, 3e-5]
    #weight_decays = [0, 1e-3, 1e-2, 1e-1]
    #dropouts = [0, 0.1, 0.3, 0.5]
    #batch_sizes = [8, 16, 32]

    # Track results for each hyperparameter
    train_losses = []
    val_f1_scores = []
        
    #for lr in learning_rates:
      # Use roberta-large instead of roberta-base
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-large', num_labels=len(label_to_index))

    # Increase Dropout for Regularization
    model.config.hidden_dropout_prob = 0.3
    model.config.attention_probs_dropout_prob = 0.3

    train_dataset = TweetDataset(train_data, tokenizer, max_len=128)
    val_dataset = TweetDataset(val_data, tokenizer, max_len=128)
    test_dataset = TweetDataset(test_data, tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use weight decay and FocalLoss for regularization
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2) #lr=2e-5
    loss_fn = FocalLoss(alpha=1, gamma=2)

        # Implement learning rate scheduler
    epochs = 10
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
        val_f1 = val_f1_score(model, val_loader, device)
        train_losses.append(train_loss)
        val_f1_scores.append(val_f1)
        print(f"Epoch {epoch + 1}/{epochs}, Train loss: {train_loss}, , Validation F1: {val_f1}")
    
    plot_hyperparameter_tuning(epochs, train_losses, val_f1_scores, 'Learning Rate')

    # Call the function with the validation set loader and print the misclassification summary
    #label_names = list(label_to_index.keys())
    #misclassification_summary = analyze_confusion_matrix(model, val_loader, device, label_names)

    # Print a summary of misclassifications
    #print("\nMisclassification Summary by Category:")
    #for label, details in misclassification_summary.items():
    #    print(f"{label}: False Positives = {details['False Positives']}, False Negatives = {details['False Negatives']}")

    # Predict with label-specific thresholds and save submission
    predictions = predict_with_label_thresholds(model, test_loader, device, label_thresholds)
    submission_df = pd.DataFrame(predictions, columns=list(label_to_index.keys()))
    submission_df.insert(0, 'index', range(len(submission_df)))
    submission_df = submission_df.astype(int)
    submission_df.to_csv("submission.csv", index=False)